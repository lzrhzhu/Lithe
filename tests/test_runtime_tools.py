"""Tool execution policy inside a run: READ tools run in parallel, every
WRITE runs alone in model order (read/write/read is never reordered), and
tool_call events are announced before any execution so front ends can show
pending calls."""
from __future__ import annotations

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, ToolCategory, ToolRegistry,
    ToolResult, ToolSpec,
)

from _fakes import FakeTransport, resp, tc


def _runtime(responses, *, registry=None, sinks=None, max_steps=3):
    reg = registry or ToolRegistry()
    return AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                       transport=FakeTransport(responses)),
                        sinks=sinks, max_steps=max_steps)


# --- tool execution policy: reads parallel, writes serialized -----------------

async def test_write_tools_execute_sequentially_in_order():
    import asyncio

    state = {"active": 0, "max_active": 0, "order": []}

    async def write(ctx, args):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        state["order"].append(args["path"])
        await asyncio.sleep(0.02)
        state["active"] -= 1
        return ToolResult(True, "wrote", args["path"])

    reg = ToolRegistry()
    reg.register(ToolSpec("write_file", "w", category=ToolCategory.WRITE), write)
    rt = _runtime([resp(tool_calls=[tc("write_file", {"path": "a"}, "c1"),
                                     tc("write_file", {"path": "b"}, "c2")]),
                   resp("done")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode())]
    assert state["order"] == ["a", "b"], "写工具按模型给出的顺序执行"
    assert state["max_active"] == 1, "两个写工具绝不并发"
    # 事件仍按模型给出的顺序发出
    names = [e["name"] for e in events if e["type"] == "tool_call"]
    assert names == ["write_file", "write_file"]


async def test_consecutive_reads_run_in_parallel():
    import asyncio

    # 两个读工具互等对方启动：并行则都完成；串行则死锁（wait_for 超时失败）
    flags = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def read_a(ctx, args):
        flags["a"].set()
        await flags["b"].wait()
        return ToolResult(True, "a", "a")

    async def read_b(ctx, args):
        flags["b"].set()
        await flags["a"].wait()
        return ToolResult(True, "b", "b")

    reg = ToolRegistry()
    reg.register(ToolSpec("read_a", "a", category=ToolCategory.READ), read_a)
    reg.register(ToolSpec("read_b", "b", category=ToolCategory.READ), read_b)
    rt = _runtime([resp(tool_calls=[tc("read_a", {}, "c1"),
                                     tc("read_b", {}, "c2")]),
                   resp("done")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")

    async def collect():
        return [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                        reg.specs_for_mode())]

    events = await asyncio.wait_for(collect(), timeout=2.0)
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "done"


async def test_read_then_write_then_read_not_reordered():
    # 执行分组保持模型顺序：read_a → write → read_b（写单独成组，不与读并发）
    import asyncio

    order: list[str] = []

    def make(name, category):
        async def h(ctx, args):
            order.append(name)
            await asyncio.sleep(0)
            return ToolResult(True, name, name)
        return ToolSpec(name, name, category=category), h

    reg = ToolRegistry()
    for spec, h in (make("read_a", ToolCategory.READ),
                    make("write_m", ToolCategory.WRITE),
                    make("read_b", ToolCategory.READ)):
        reg.register(spec, h)
    rt = _runtime([resp(tool_calls=[tc("read_a", {}, "c1"),
                                      tc("write_m", {}, "c2"),
                                      tc("read_b", {}, "c3")]),
                   resp("done")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                 reg.specs_for_mode())]
    assert order == ["read_a", "write_m", "read_b"]


# --- tool_call events precede execution ----------------------------------------

async def test_tool_call_event_arrives_before_tool_executes():
    import asyncio

    started: list[str] = []

    async def slow(ctx, args):
        started.append(args["path"])
        await asyncio.sleep(0.02)
        return ToolResult(True, "done", args["path"])

    reg = ToolRegistry()
    reg.register(ToolSpec("slow_tool", "s", category=ToolCategory.WRITE), slow)
    rt = _runtime([resp(tool_calls=[tc("slow_tool", {"path": "a"}, "c1")]),
                   resp("final")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")
    announced = 0
    async for ev in rt.run(ctx, [{"role": "user", "content": "hi"}],
                           reg.specs_for_mode()):
        if ev["type"] == "tool_call":
            # 慢工具执行期间前端已能看到调用：事件先于任何执行
            assert started == [], "tool_call 事件必须在工具开始执行前发出"
            announced += 1
    assert announced == 1
    assert started == ["a"], "工具确实执行了"


async def test_all_tool_calls_announced_before_any_executes():
    order: list[str] = []

    async def write(ctx, args):
        order.append(f"run:{args['path']}")
        return ToolResult(True, "wrote", args["path"])

    reg = ToolRegistry()
    reg.register(ToolSpec("write_file", "w", category=ToolCategory.WRITE), write)
    rt = _runtime([resp(tool_calls=[tc("write_file", {"path": "a"}, "c1"),
                                     tc("write_file", {"path": "b"}, "c2")]),
                   resp("done")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")
    events = []
    async for ev in rt.run(ctx, [{"role": "user", "content": "hi"}],
                           reg.specs_for_mode()):
        events.append(ev)
        if ev["type"] == "tool_call":
            assert order == [], "所有 tool_call 事件先于任何执行"
    # 事件序：先全部播报，再按模型顺序给结果
    kinds = [e["type"] for e in events
             if e["type"] in ("tool_call", "tool_result")]
    assert kinds == ["tool_call", "tool_call", "tool_result", "tool_result"]
    assert order == ["run:a", "run:b"]
