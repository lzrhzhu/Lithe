"""End-to-end: the kernel's components (runtime + registry + sink + memory +
undo) close the loop together with NO host application — proving lithe is
independently reusable. The LLM is mocked; tools / sink / state are tiny fakes.
"""
from __future__ import annotations

import json

from lithe import (
    Action, AgentContext, AgentRuntime, EventType, LLMConfig, RunStats,
    ToolCategory, ToolRegistry, ToolResult, ToolSpec, UndoEngine,
    replay_messages, to_sse,
)


class _MemSink:
    """A no-DB EventSink: just collects events + records in memory."""

    def __init__(self):
        self.events: list[dict] = []
        self.records: list[dict] = []

    async def on_event(self, ctx, event):
        self.events.append(event)

    async def on_record(self, ctx, record):
        self.records.append(record)


def _build_registry(state: dict) -> ToolRegistry:
    reg = ToolRegistry()

    async def write(ctx, args):
        state[args["path"]] = args["content"]
        return ToolResult(True, "wrote", f"wrote {args['path']}")

    async def read(ctx, args):
        return ToolResult(True, "read", state.get(args["path"], ""))

    reg.register(ToolSpec("write", "w", category=ToolCategory.WRITE), write,
                 reverter=lambda a, c: state.pop(a.target, None))
    reg.register(ToolSpec("read", "r", category=ToolCategory.READ), read)
    return reg


def _chat_transport(responses):
    """Fake transport yielding chat-shaped responses, parsed to transport shape."""
    it = iter(list(responses))

    class _T:
        async def complete(self, client, **kw):
            data = next(it)
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            return {"content": msg.get("content") or "",
                    "tool_calls": msg.get("tool_calls") or [], "usage": {}}
    return _T()


def _tc(name, args, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


async def test_kernel_full_loop_persist_replay_undo(monkeypatch):
    # 1. a tiny host: a 2-tool registry, an in-memory sink, shared file state
    state: dict = {}
    registry = _build_registry(state)
    sink = _MemSink()
    transport = _chat_transport([
        {"choices": [{"message": {"content": "",
                                  "tool_calls": [_tc("write", {"path": "a.txt", "content": "hi"})]}}]},
        {"choices": [{"message": {"content": "done"}}]},
    ])
    agent = AgentRuntime(registry,
                         LLMConfig(model="m", base_url="x", api_key="k", transport=transport),
                         sinks=[sink], max_steps=5)
    ctx = AgentContext(run_id="r1", user_id="u1")
    stats = RunStats()
    events = [e async for e in agent.run(ctx, [{"role": "user", "content": "go"}],
                                         registry.specs_for_mode(), stats=stats)]

    # 3. the write took effect, final answer captured, events are SSE-serializable
    assert state == {"a.txt": "hi"}
    assert stats.final_text == "done" and stats.status == "done"
    assert any(e["type"] == EventType.TOOL_RESULT and e["ok"] for e in events)
    assert to_sse(events[0]).startswith("data: ")

    # 4. the sink captured the full conversation → memory can replay it verbatim
    assert [r["role"] for r in sink.records] == ["assistant", "tool", "assistant"]
    replayed = replay_messages([
        {"role": "assistant", "content": "", "tool_calls": sink.records[0]["tool_calls"]},
        {"role": "tool", "tool_call_id": "c1", "content": sink.records[1]["content"]},
        {"role": "assistant", "content": "done"},
    ])
    assert replayed[-1] == {"role": "assistant", "content": "done"}
    assert replayed[0]["tool_calls"][0]["id"] == "c1"  # tool pairing survived replay

    # 5. the host recorded a mutation for undo → UndoEngine reverts it (pure, no DB)
    engine = UndoEngine(registry.reverters())
    report = await engine.undo([Action(kind="write", target="a.txt", status="applied")], ctx)
    assert report.reverted == 1 and report.ok is True
    assert "a.txt" not in state  # mutation undone


async def test_kernel_readonly_mode_blocks_write_tool(monkeypatch):
    # anchored/read-only mode must not expose write tools, even if registered
    state: dict = {}
    registry = _build_registry(state)
    auto = {s["function"]["name"] for s in registry.specs_for_mode("autonomous")}
    anchored = {s["function"]["name"] for s in registry.specs_for_mode("anchored")}
    assert auto == {"write", "read"}
    assert anchored == {"read"}  # write tool hidden in anchored mode
