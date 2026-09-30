"""Minimal lithe host — runnable offline.

A host supplies tools, a system prompt and (optionally) a store; lithe
supplies the engine, run envelope, persistence and undo. This example wires a
scripted fake transport so it runs without any API key; swap `_Scripted` for
`LLMConfig(model=..., base_url=..., api_key=...)` against a real endpoint.

Run:  python examples/minimal_host.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

# Run straight from a source checkout without installing; harmless (and
# unnecessary) once lithe is pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithe import AgentContext, LLMConfig, ToolCategory, ToolRegistry, ToolResult, ToolSpec  # noqa: E402
from lithe.bundles import AgentHost, JsonlRunStore, undo_run  # noqa: E402


def _scripted_transport(responses: list[dict]):
    """A tiny LLMTransport replaying canned model turns (tool calls, answers)."""
    it = iter(responses)

    class _Scripted:
        async def complete(self, client, **kw):
            r = next(it)
            return {"content": r.get("content", ""),
                    "tool_calls": r.get("tool_calls", []),
                    "usage": {"prompt_tokens": 120, "completion_tokens": 30,
                              "total_tokens": 150}}

    return _Scripted()


class ActionSink:
    """Turn a write tool's ``file_change`` UI events into undo-able actions.

    This is the host's job by design — the kernel emits, the host persists.
    The action *kind* is the host's domain vocabulary (here ``note_write``);
    undo maps kinds to reverters, independent of tool names. Because this
    sink records actions itself, the host is built with
    ``capture_actions=False`` so AgentHost's built-in StoreSink doesn't write
    a second (builtin-kind) row for the same event.
    """

    def __init__(self, store):
        self.store = store

    async def on_event(self, ctx, ev):
        if ev.get("type") == "file_change":
            self.store.log_action(ctx.run_id, ctx.user_id, "note_write",
                                  ev["path"], ev.get("old"), ev.get("new"))

    async def on_record(self, ctx, record):
        pass


def _tc(name: str, args: dict, cid: str) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


async def main() -> None:
    # 1. Tools: register spec + handler. Write tools can carry a reverter.
    state: dict[str, str] = {}
    reg = ToolRegistry()

    async def write_note(ctx, args):
        old = state.get(args["path"])
        state[args["path"]] = args["content"]
        return ToolResult(True, f"wrote {args['path']}", f"wrote {args['path']}",
                          ui=[{"type": "file_change", "path": args["path"],
                               "old": old, "new": args["content"]}])

    async def read_note(ctx, args):
        return ToolResult(True, "read", state.get(args["path"], "(empty)"))

    reg.register(ToolSpec(
        "write_note", "保存一条笔记（可撤销）",
        {"type": "object",
         "properties": {"path": {"type": "string"},
                        "content": {"type": "string"}},
         "required": ["path", "content"]}, ToolCategory.WRITE),
        write_note)
    reg.register(ToolSpec("read_note", "读取一条笔记",
                          {"type": "object",
                           "properties": {"path": {"type": "string"}},
                           "required": ["path"]}, ToolCategory.READ),
                 read_note)

    # 2. Zero-database store + host. Real apps: your own tools, prompt, store.
    #    capture_actions=False: this host's ActionSink below persists actions
    #    itself (with its own "note_write" kind) — leaving the default on
    #    would double-capture every file_change into a second action row.
    store = JsonlRunStore(tempfile.mkdtemp(prefix="lithe-demo-"))
    cfg = LLMConfig(model="demo", base_url="unused", api_key="unused",
                    transport=_scripted_transport([
                        {"tool_calls": [_tc("write_note",
                                            {"path": "a.txt", "content": "hi"}, "c1")]},
                        {"content": "已保存你的笔记。"},
                    ]))
    host = AgentHost(reg, cfg, store, max_steps=5, sinks=[ActionSink(store)],
                     capture_actions=False,
                     build_system_prompt=lambda ctx, mode, anchor: "你是笔记助手。")

    # 3. Run: iterate the event stream (forward over SSE in a web app).
    ctx = AgentContext(run_id="r1", user_id="u1")
    async for ev in host.run(ctx, "帮我保存 a.txt 内容为 hi"):
        print(f"[{ev['type']:>12}]",
              ev.get("text") or ev.get("summary") or ev.get("name") or "")
    print("state:", state)

    # 4. Undo reverts the run's mutations: kinds → reverters, newest-first.
    report = await undo_run(host, "r1", "u1",
                      reverters={"note_write": lambda a, c: state.pop(a.target, None)})
    print("undo:", report.reverted, "reverted ->", state)


if __name__ == "__main__":
    asyncio.run(main())
