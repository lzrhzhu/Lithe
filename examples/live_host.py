"""Live lithe run against a real OpenAI-compatible endpoint.

Offline examples use scripted transports; this one spends real tokens. All
secrets come from the environment (never committed, and no default endpoint —
it refuses to run rather than silently hitting some third-party URL):

    export LITHE_API_KEY=sk-...
    export LITHE_BASE_URL=https://your-endpoint/api/v1
    export LITHE_MODEL=your-model            # required too

    python examples/live_host.py             # add --stream for token streaming

Covers what the offline suite mocks: a real tool-calling round (write → read
→ final answer), live usage/context reporting, and undo of the run's
mutation.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithe import (  # noqa: E402
    AgentContext, EventType, LLMConfig, ToolCategory, ToolRegistry, ToolResult,
    ToolSpec,
)
from lithe.bundles import AgentHost, JsonlRunStore, undo_run  # noqa: E402

API_KEY = os.environ.get("LITHE_API_KEY")
BASE_URL = os.environ.get("LITHE_BASE_URL")
MODEL = os.environ.get("LITHE_MODEL")
STREAM = "--stream" in sys.argv

if not (API_KEY and BASE_URL and MODEL):
    sys.exit("需要 LITHE_API_KEY / LITHE_BASE_URL / LITHE_MODEL 环境变量"
             "（本示例不内置任何默认端点）。")


class Notebook:
    """Tiny stateful 'filesystem' the agent writes to (undo-able)."""

    def __init__(self):
        self.files: dict[str, str] = {}

    def tools(self) -> list[tuple[ToolSpec, object]]:
        async def write_file(ctx, args):
            old = self.files.get(args["path"])
            self.files[args["path"]] = args["content"]
            return ToolResult(True, f"wrote {args['path']}",
                              f"已写入 {args['path']}（{len(args['content'])} 字符）",
                              ui=[{"type": "file_change", "path": args["path"],
                                   "old": old, "new": args["content"]}])

        async def read_file(ctx, args):
            content = self.files.get(args["path"])
            if content is None:
                return ToolResult(False, "不存在", f"{args['path']} 不存在。")
            return ToolResult(True, f"read {args['path']}", content)

        return [
            (ToolSpec("write_file", "把文本写入一个文件（路径→内容）。",
                      {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}, ToolCategory.WRITE),
             write_file),
            (ToolSpec("read_file", "读取一个文件的内容。",
                      {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}, ToolCategory.READ),
             read_file),
        ]


async def main() -> int:
    if not API_KEY:
        print("set LITHE_API_KEY first")
        return 2

    nb = Notebook()
    reg = ToolRegistry()
    for spec, handler in nb.tools():
        reg.register(spec, handler)

    class ActionSink:
        def __init__(self, store):
            self.store = store

        async def on_event(self, ctx, ev):
            if ev.get("type") == "file_change":
                self.store.log_action(ctx.run_id, ctx.user_id, "note_write",
                                      ev["path"], ev.get("old"), ev.get("new"))

        async def on_record(self, ctx, record):
            pass

    workdir = tempfile.mkdtemp(prefix="lithe-live-")
    store = JsonlRunStore(workdir)
    host = AgentHost(
        reg,
        LLMConfig(model=MODEL, base_url=BASE_URL, api_key=API_KEY,
                  timeout=120.0, attempts=2, sleep_429=2.0, sleep_err=1.0,
                  stream=STREAM, context_window=128_000),
        store, max_steps=8, sinks=[ActionSink(store)],
        build_system_prompt=lambda ctx, mode, anchor:
            "你是文件助手。只用提供的工具完成工作，最后用中文简要汇报。")

    ctx = AgentContext(run_id="live1", user_id="demo")
    task = ("用 write_file 创建 todo.txt，内容为三行：喝水、写代码、散步。"
            "再用 read_file 读回来核对，最后告诉我核对结果。")

    print(f"== model: {MODEL}  stream={STREAM} ==")
    async for ev in host.run(ctx, task):
        t = ev["type"]
        if t == EventType.ASSISTANT_DELTA:
            print(ev["text"], end="", flush=True)
        elif t == EventType.ASSISTANT and not STREAM:
            print(f"[assistant] {ev['text']}")
        elif t == EventType.TOOL_CALL:
            print(f"[tool_call ] {ev['name']} {json.dumps(ev.get('args', {}), ensure_ascii=False)[:120]}")
        elif t == EventType.TOOL_RESULT:
            print(f"[tool_res  ] ok={ev['ok']} {ev['summary']}")
        elif t == EventType.USAGE:
            print(f"[usage     ] prompt={ev['prompt_tokens']} completion={ev['completion_tokens']}"
                  f" ctx={ev['context_tokens']}tk/{ev['context_chars']}ch"
                  f" window={ev['context_percent']}%")
        elif t == EventType.ERROR:
            print(f"[error     ] {ev.get('message')}")

    done = ev
    print(f"[done      ] status={done['status']} steps={done['steps']} "
          f"tokens={done['tokens']} (p{done['prompt_tokens']}/c{done['completion_tokens']}) "
          f"context={done['context_percent']}%")

    print("notebook:", nb.files)
    report = await undo_run(host, "live1", "demo",
                            reverters={"note_write": lambda a, c: nb.files.pop(a.target, None)})
    print("undo:", report.reverted, "reverted ->", nb.files)
    return 0 if done["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
