"""Parallel-delegation budget visibility + workspace off-loop I/O.

A delegation registers a live-usage slot in the run's shared state; every
usage event folds into it as the child spends, and each concurrently
running sibling's runtime counts the other slots against its own cap —
N parallel workers share one ceiling instead of each burning the full
max_cost. Sequential delegations keep their independent caps (slots are
popped at delegation end).
"""
from __future__ import annotations

import asyncio
import json

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats,
    ToolCategory, ToolRegistry, ToolResult, ToolSpec,
)
from lithe.bundles import JsonlRunStore, SubagentEngine, SubagentRoster, \
    SubagentSpec
from lithe.bundles.host import AgentHost
from lithe.bundles.subagents import register_delegate_tools


def _tc(name, args=None, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


class _ScriptedTransport:
    """Replays [tool-call, final]; every call reports a fixed usage cost."""

    def __init__(self, responses, cost):
        self._it = iter(list(responses))
        self.cost = cost
        self.calls = 0

    async def complete(self, client, **kw):
        # yield control so parallel children actually interleave
        await asyncio.sleep(0)
        self.calls += 1
        r = next(self._it)
        return {"content": r.get("content", ""),
                "tool_calls": r.get("tool_calls", []),
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15, "cost": self.cost}}


async def _note(ctx, args):
    return ToolResult(True, "ok", "ok")


async def test_over_budget_counts_inflight_siblings_not_self():
    rt = AgentRuntime(ToolRegistry(),
                      LLMConfig(model="m", base_url="x", api_key="k"),
                      max_cost=1.0)
    ctx = AgentContext(run_id="r", user_id="u", subagent="b",
                       extra={"_delegation_slot": "b:1"})
    ctx.shared["_inflight_sub_usage"] = {
        "a:1": {"cost": 0.4, "tokens": 60},
        "b:1": {"cost": 0.6, "tokens": 60},
    }
    stats = RunStats()
    stats.total_cost = 0.5
    stats.total_tokens = 50
    # own slot (0.6) excluded: 0.5 + sibling 0.4 = 0.9 stays under the cap
    assert rt._over_budget(ctx, stats) is False
    # a richer sibling pushes the combined total over
    ctx.shared["_inflight_sub_usage"]["a:1"]["cost"] = 0.6
    assert rt._over_budget(ctx, stats) is True
    # an orchestrator context (no slot) counts every inflight child
    orch = AgentContext(run_id="r", user_id="u")
    orch.shared["_inflight_sub_usage"] = {"a:1": {"cost": 1.2, "tokens": 10}}
    orch_stats = RunStats()
    orch_stats.total_cost = 0.0
    assert rt._over_budget(orch, orch_stats) is True
    # token caps see siblings too
    rt_tok = AgentRuntime(ToolRegistry(),
                          LLMConfig(model="m", base_url="x", api_key="k"),
                          max_total_tokens=100)
    tok_stats = RunStats()
    tok_stats.total_tokens = 50
    assert rt_tok._over_budget(ctx, tok_stats) is True  # 50 + 60 sibling


async def test_parallel_delegations_share_one_cost_ceiling(tmp_path):
    """Two children run via delegate_parallel under host max_cost=1.0; each
    alone would spend 1.2 (three calls × 0.4) and stop at its own cap only
    after the second call's post-check — but the shared ceiling stops the
    pair earlier: at most three calls are made in total."""
    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("note", "n", category=ToolCategory.READ), _note)
    host = AgentHost(reg,
                     LLMConfig(model="m", base_url="x", api_key="k"),
                     store, max_cost=1.0, max_steps=6)
    script = [
        {"tool_calls": [_tc("note", {})]},
        {"tool_calls": [_tc("note", {}, "c2")]},
        {"content": "child done"},
    ]
    roster = SubagentRoster([
        SubagentSpec("w1", "w1", "d", ["note"], "p",
                     transport=_ScriptedTransport(script, cost=0.4)),
        SubagentSpec("w2", "w2", "d", ["note"], "p",
                     transport=_ScriptedTransport(script, cost=0.4)),
    ])
    engine = SubagentEngine(host, roster)
    register_delegate_tools(reg, engine, parallel=True)

    ctx = AgentContext(run_id="r", user_id="u")
    res = await reg.dispatch(
        "delegate_parallel",
        {"tasks": [{"agent": "w1", "task": "t1"},
                   {"agent": "w2", "task": "t2"}]},
        ctx)

    assert "超出预算" in res.content, (
        "combined live spend (0.4 per call, cap 1.0) must stop a child "
        "while it still has tool calls to run")
    # w1 stopped mid-script (2 calls, not its alone-side 3) once w2's live
    # spend counted against it; the pair burned fewer than the 6 calls both
    # scripts hold
    total_calls = roster.get("w1").transport.calls + \
        roster.get("w2").transport.calls
    assert total_calls < 6, f"ceiling not enforced: {total_calls} calls made"


async def test_sequential_delegations_keep_independent_caps(tmp_path):
    """Back-to-back delegations do not see each other's spend: the slot is
    popped when each delegation ends (both children spend their full 1.2
    alone-side budget without the other's completed spend counting)."""
    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("note", "n", category=ToolCategory.READ), _note)
    host = AgentHost(reg,
                     LLMConfig(model="m", base_url="x", api_key="k"),
                     store, max_cost=1.0, max_steps=6)
    script = [
        {"tool_calls": [_tc("note", {})]},
        {"tool_calls": [_tc("note", {}, "c2")]},
        {"content": "child done"},
    ]
    tr = _ScriptedTransport(script * 2, cost=0.4)
    roster = SubagentRoster([
        SubagentSpec("w", "w", "d", ["note"], "p", transport=tr)])
    engine = SubagentEngine(host, roster)
    register_delegate_tools(reg, engine, parallel=True)

    ctx = AgentContext(run_id="r", user_id="u")
    for _ in range(2):
        res = await reg.dispatch(
            "delegate_parallel", {"tasks": [{"agent": "w", "task": "t"}]}, ctx)
        assert res.ok
        assert "超出预算" not in res.content, (
            "a completed delegation's spend must not count against the next")
    assert tr.calls == 6
    # the inflight map is fully cleaned up after the delegations
    assert ctx.shared.get("_inflight_sub_usage") == {}
