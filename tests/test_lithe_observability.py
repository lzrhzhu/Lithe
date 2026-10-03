"""Observability round: run_id/seq stamping, timing metrics.

Every event leaves the runtime stamped with its run's id and a monotonic
sequence number; runs report wall-clock duration, tool results carry
elapsed_ms, and streaming calls report time-to-first-token on the first
delta (echoed on the usage event).
"""
from __future__ import annotations

import asyncio

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolRegistry,
    ToolResult, ToolSpec,
)


def _resp(content="", tool_calls=None):
    return {"content": content, "tool_calls": tool_calls or []}


def _tc(name="echo", args=None, cid="c1"):
    import json
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


async def _slow_echo(ctx, args):
    await asyncio.sleep(0.01)
    return ToolResult(ok=True, summary="echoed", content="ok")


def _runtime(transport, *, registry=None, **kw):
    cfg = LLMConfig(model="m", base_url="x", api_key="k", transport=transport)
    reg = registry or ToolRegistry()
    reg.register(ToolSpec("echo", "e"), _slow_echo)
    return AgentRuntime(reg, cfg, max_steps=3, **kw)


def _ctx():
    return AgentContext(run_id="run-xyz", user_id="u")


async def test_every_event_stamped_with_run_id_and_rising_seq():
    rt = _runtime(_PlainTransport())
    events = [e async for e in rt.run(_ctx(),
                                     [{"role": "user", "content": "q"}], [])]
    assert events, "run produced no events"
    seqs = []
    for e in events:
        assert e["run_id"] == "run-xyz"
        seqs.append(e["seq"])
    assert seqs == sorted(seqs), "seq must be non-decreasing"
    assert len(set(seqs)) == len(seqs), "seq must not repeat within a run"
    # a fresh run restarts the counter
    events2 = [e async for e in rt.run(_ctx(),
                                      [{"role": "user", "content": "q"}], [])]
    assert events2[0]["seq"] == events[0]["seq"]


class _PlainTransport:
    """Tool call on step 1, final answer on step 2."""

    def __init__(self):
        self._it = iter([_resp("", [_tc("echo", {"a": 1})]), _resp("done")])

    async def complete(self, client, **kw):
        return next(self._it)


class _StreamTransport:
    """Streams deltas then the final result."""

    def __init__(self, answer="fin"):
        self._answer = answer

    async def complete(self, client, **kw):
        raise AssertionError("stream=True 时应走 complete_stream")

    async def complete_stream(self, client, **kw):
        await asyncio.sleep(0.01)
        yield {"delta": "hel"}
        yield {"delta": "lo"}
        yield {"result": _resp(self._answer)}


async def test_tool_result_carries_elapsed_ms():
    rt = _runtime(_PlainTransport())
    events = [e async for e in rt.run(_ctx(),
                                     [{"role": "user", "content": "q"}], [])]
    trs = [e for e in events if e["type"] == "tool_result"]
    assert len(trs) == 1
    assert isinstance(trs[0]["elapsed_ms"], int) and trs[0]["elapsed_ms"] >= 0


async def test_done_envelope_and_stats_carry_duration():
    rt = _runtime(_PlainTransport(), emit_envelope=True)
    stats = RunStats()
    events = [e async for e in rt.run(_ctx(),
                                      [{"role": "user", "content": "q"}], [],
                                      stats=stats)]
    done = events[-1]
    assert done["type"] == "done"
    assert isinstance(done["duration_s"], float) and done["duration_s"] >= 0
    assert stats.duration_s >= 0


async def test_streaming_ttft_on_first_delta_and_usage():
    cfg = LLMConfig(model="m", base_url="x", api_key="k",
                    transport=_StreamTransport(), stream=True)
    rt = AgentRuntime(ToolRegistry(), cfg, max_steps=2)
    events = [e async for e in rt.run(_ctx(),
                                      [{"role": "user", "content": "q"}], [])]
    deltas = [e for e in events if e["type"] == "assistant_delta"]
    assert len(deltas) == 2
    assert "ttft_ms" in deltas[0] and deltas[0]["ttft_ms"] >= 0
    assert "ttft_ms" not in deltas[1], "only the first delta carries ttft"
    usage = [e for e in events if e["type"] == "usage"]
    assert len(usage) == 1 and usage[0]["ttft_ms"] == deltas[0]["ttft_ms"]


async def test_non_streaming_usage_ttft_is_none():
    rt = _runtime(_PlainTransport())
    events = [e async for e in rt.run(_ctx(),
                                     [{"role": "user", "content": "q"}], [])]
    usage = [e for e in events if e["type"] == "usage"]
    assert usage[0]["ttft_ms"] is None
