"""Parallel-execution safety: same-file mutation races + delegation guards.

The stale-content guard is check-then-act; every ``await asyncio.to_thread``
between the check and the write is a suspension point where a parallel
subagent's write can land unseen. These tests pin the two fixes:

- the run-wide write lock (ctx.shared, shared with subagent contexts):
  write_file / edit_file / apply_patch hold it across check → write →
  revision-record, so of two concurrent same-file writers exactly one
  succeeds and the loser is refused with the stale-content error;
- read_file's stat-BEFORE-read snapshot ordering: a write landing between
  the stat and the content read leaves the snapshot OLDER than what the
  model saw, so the model's next write is refused (re-read), never a blind
  overwrite through a fresh-looking snapshot.

Plus the delegation guards: delegate_parallel per-task timeout isolation,
and the mode fence (an anchored/read-only orchestrator cannot delegate
write powers to a subagent).
"""
from __future__ import annotations

import asyncio
import json

from lithe import (
    AgentContext, EventType, LLMConfig, ToolCategory, ToolRegistry,
    ToolResult, ToolSpec,
)
from lithe.bundles import (
    AgentHost, JsonlRunStore, SubagentEngine, SubagentRoster, SubagentSpec,
    Workspace,
)
from lithe.bundles.patch import register_apply_patch_tool
from lithe.bundles.subagents import (
    make_parallel_delegate_tool, register_delegate_tool,
)
from lithe.bundles.workspace import register_file_tools


def _registry_with(ws):
    reg = ToolRegistry()
    reverters = register_file_tools(reg, lambda ctx: ws)
    return reg, reverters


async def _noop(ctx, args):
    return ToolResult(True, "ok", "ok")


# --------------------------------------------------------------------------- #
# run-wide write lock: concurrent same-file mutations serialize
# --------------------------------------------------------------------------- #

async def test_parallel_same_file_writes_serialize(tmp_path):
    """两个共享 shared 的上下文并发 write_file 同一文件：恰好一个成功，
    另一个被陈旧守卫拒绝（旧实现双双成功，后者静默覆盖前者）。"""
    for attempt in range(5):  # race must hold under any interleaving
        ws = Workspace(tmp_path / f"w{attempt}")
        reg, _ = _registry_with(ws)
        ws.write("f.txt", "base")
        shared = {}
        ca = AgentContext(run_id="r", user_id="u", subagent="A", shared=shared)
        cb = AgentContext(run_id="r", user_id="u", subagent="B", shared=shared)
        await reg.dispatch("read_file", {"path": "f.txt"}, ca)
        await reg.dispatch("read_file", {"path": "f.txt"}, cb)

        ra, rb = await asyncio.gather(
            reg.dispatch("write_file", {"path": "f.txt", "content": "from A"}, ca),
            reg.dispatch("write_file", {"path": "f.txt", "content": "from B"}, cb),
        )
        oks = [r for r in (ra, rb) if r.ok]
        refused = [r for r in (ra, rb) if not r.ok]
        assert len(oks) == 1 and len(refused) == 1, (
            f"attempt {attempt}: A.ok={ra.ok} B.ok={rb.ok} —— 并发同文件写"
            f"必须一胜一拒，不能双双成功（静默覆盖）")
        assert "已变更" in refused[0].summary
        assert ws.read("f.txt") == ("from A" if ra.ok else "from B")


async def test_parallel_edit_vs_write_serialize(tmp_path):
    ws = Workspace(tmp_path / "ew")
    reg, _ = _registry_with(ws)
    ws.write("f.txt", "line1\nline2\n")
    shared = {}
    ca = AgentContext(run_id="r", user_id="u", subagent="A", shared=shared)
    cb = AgentContext(run_id="r", user_id="u", subagent="B", shared=shared)
    await reg.dispatch("read_file", {"path": "f.txt"}, ca)
    await reg.dispatch("read_file", {"path": "f.txt"}, cb)

    ra, rb = await asyncio.gather(
        reg.dispatch("edit_file",
                     {"path": "f.txt", "old_text": "line1", "new_text": "edited"}, ca),
        reg.dispatch("write_file", {"path": "f.txt", "content": "rewritten"}, cb),
    )
    assert (ra.ok != rb.ok), f"edit/write 并发必须一胜一拒：{ra.ok}/{rb.ok}"
    refused = ra if not ra.ok else rb
    assert "已变更" in refused.summary
    final = ws.read("f.txt")
    assert final in ("edited\nline2\n", "rewritten")


async def test_apply_patch_vs_write_serialize(tmp_path):
    ws = Workspace(tmp_path / "pw")
    reg, _ = _registry_with(ws)
    register_apply_patch_tool(reg, lambda ctx: ws)
    ws.write("f.txt", "line1\n")
    shared = {}
    ca = AgentContext(run_id="r", user_id="u", subagent="A", shared=shared)
    cb = AgentContext(run_id="r", user_id="u", subagent="B", shared=shared)
    await reg.dispatch("read_file", {"path": "f.txt"}, ca)
    await reg.dispatch("read_file", {"path": "f.txt"}, cb)

    patch = ("*** Begin Patch\n"
             "*** Update File: f.txt\n"
             "@@\n"
             "-line1\n"
             "+patched\n"
             "*** End Patch\n")

    ra, rb = await asyncio.gather(
        reg.dispatch("apply_patch", {"patch_text": patch}, ca),
        reg.dispatch("write_file", {"path": "f.txt", "content": "rewritten"}, cb),
    )
    assert (ra.ok != rb.ok), f"apply_patch/write 并发必须一胜一拒：{ra.ok}/{rb.ok}"
    refused = ra if not ra.ok else rb
    assert "已变更" in refused.summary
    assert ws.read("f.txt") in ("patched\n", "rewritten")


# --------------------------------------------------------------------------- #
# read_file snapshot ordering: stat BEFORE read (safe direction)
# --------------------------------------------------------------------------- #

async def test_midread_write_makes_snapshot_stale(tmp_path):
    """读与 stat 之间发生写入：快照必须比模型看到的内容更旧（下次写入被
    拒、要求重读），而不是把写入后的 stat 盖在写入前的内容上放行覆盖。"""
    from lithe.bundles.workspace import Workspace as _W

    class _MidReadWorkspace(_W):
        """Simulates a concurrent writer between read_file's stat and read."""
        injected = False

        def read(self, rel):
            if rel == "f.txt" and not self.injected:
                self.injected = True
                _W.write(self, rel, "externally changed mid-read")
            return _W.read(self, rel)

    ws = _MidReadWorkspace(tmp_path / "mr")
    reg, _ = _registry_with(ws)
    ws.write("f.txt", "original")
    ctx = AgentContext(run_id="r", user_id="u")

    r = await reg.dispatch("read_file", {"path": "f.txt"}, ctx)
    assert r.ok and "externally changed mid-read" in r.content

    w = await reg.dispatch("write_file", {"path": "f.txt", "content": "mine"}, ctx)
    assert w.ok is False and "已变更" in w.summary, (
        "读到的是写入后内容、快照是写入前 stat —— 下一次写必须被拒")
    # re-read refreshes the snapshot, then the write lands
    await reg.dispatch("read_file", {"path": "f.txt"}, ctx)
    ok = await reg.dispatch("write_file", {"path": "f.txt", "content": "mine"}, ctx)
    assert ok.ok and ws.read("f.txt") == "mine"


# --------------------------------------------------------------------------- #
# delegate_parallel: per-task timeout isolation
# --------------------------------------------------------------------------- #

async def test_delegate_parallel_timeout_isolates_hung_agent(tmp_path):
    """超时只中止挂死的那个子代理（按任务计时），兄弟照常完成，批次整体
    如期返回且 inflight 预算槽被清理。"""

    class _Hung:
        async def complete(self, client, **kw):
            await asyncio.Event().wait()  # never resolves

    class _Fine:
        async def complete(self, client, **kw):
            return {"content": "fine", "tool_calls": [], "usage": {}}

    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("read", "r", category=ToolCategory.READ), _noop)
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=_Fine()), store)
    engine = SubagentEngine(host, SubagentRoster([
        SubagentSpec("hung", "H", "d", ["read"], "p", transport=_Hung()),
        SubagentSpec("good", "G", "d", ["read"], "p", transport=_Fine()),
    ]))
    ctx = AgentContext(run_id="r", user_id="u")
    store.create_run("r", "u", "t")

    _, handler = make_parallel_delegate_tool(engine, timeout=0.2)
    res = await asyncio.wait_for(handler(ctx, {"tasks": [
        {"agent": "hung", "task": "stuck"},
        {"agent": "good", "task": "ok"}]}), 5.0)
    assert res.ok is False and "1 成功" in res.summary
    assert "### hung（失败）" in res.content and "超时" in res.content
    assert "### good（成功）" in res.content and "fine" in res.content
    # the cancelled delegation cleaned up its live-usage slot
    assert ctx.shared.get("_inflight_sub_usage") == {}


# --------------------------------------------------------------------------- #
# mode fence: delegation must not widen a restricted mode's powers
# --------------------------------------------------------------------------- #

def _capture_transport(seen):
    class _T:
        async def complete(self, client, **kw):
            seen.append({t["function"]["name"] for t in (kw.get("tools") or [])})
            return {"content": "done", "tool_calls": [], "usage": {}}
    return _T()


async def test_anchored_mode_strips_write_tools_from_subagents(tmp_path):
    """anchored（只读）编排者委派出的子代理拿不到 WRITE 工具：shared 里
    的 _host_mode（host.run 写入）约束 trimmed_tools。"""
    seen: list[set[str]] = []
    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("read", "r", category=ToolCategory.READ), _noop)

    async def _write(ctx, args):
        return ToolResult(True, "wrote", "wrote")

    reg.register(ToolSpec("write", "w", category=ToolCategory.WRITE), _write)
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=_capture_transport(seen)), store)
    engine = SubagentEngine(host, SubagentRoster([
        SubagentSpec("w", "W", "d", ["read", "write"], "p")]))
    store.create_run("r", "u", "t")

    ctx = AgentContext(run_id="r", user_id="u")
    ctx.shared["_host_mode"] = "anchored"
    await engine.run("w", "t", ctx)
    assert seen, "子代理的模型调用应发生"
    assert "write" not in seen[-1] and "read" in seen[-1], (
        "anchored 委派必须剥掉 WRITE 工具")

    # no stashed mode (direct engine.run callers): declared list applies
    seen.clear()
    await engine.run("w", "t", AgentContext(run_id="r2", user_id="u"))
    assert "write" in seen[-1] and "read" in seen[-1]


async def test_host_run_stashes_mode_for_delegation(tmp_path):
    """host.run(mode=...) 把模式写进 shared：anchored 编排者经 delegate 工具
    委派，子代理的工具列表同样剥掉 WRITE（端到端路径）。"""
    seen: list[set[str]] = []
    sub_seen: list[set[str]] = []
    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("read", "r", category=ToolCategory.READ), _noop)

    async def _write(ctx, args):
        return ToolResult(True, "wrote", "wrote")

    reg.register(ToolSpec("write", "w", category=ToolCategory.WRITE), _write)

    def _tc(name, args, cid="c1"):
        return {"id": cid, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}

    class _Orch:
        def __init__(self):
            self.calls = 0

        async def complete(self, client, **kw):
            seen.append({t["function"]["name"] for t in (kw.get("tools") or [])})
            self.calls += 1
            if self.calls == 1:
                return {"content": "", "tool_calls": [_tc(
                    "delegate", {"agent": "w", "task": "do"})], "usage": {}}
            return {"content": "orchestrator done", "tool_calls": [], "usage": {}}

    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=_Orch()), store)
    engine = SubagentEngine(host, SubagentRoster([
        SubagentSpec("w", "W", "d", ["read", "write"], "p",
                     transport=_capture_transport(sub_seen))]))
    register_delegate_tool(reg, engine)

    events = [e async for e in host.run(
        AgentContext(run_id="r", user_id="u"), "delegate something",
        mode="anchored")]
    assert events[-1]["type"] == EventType.DONE and events[-1]["status"] == "done"
    # the anchored orchestrator itself saw delegate (META) but not write...
    assert "delegate" in seen[0] and "write" not in seen[0]
    # ...and its subagent got no WRITE tools through the delegation
    assert "write" not in sub_seen[-1] and "read" in sub_seen[-1]
