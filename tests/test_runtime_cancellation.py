"""Cancellation: before the first model call, by callable after a tool
round, and mid-generation on a live stream -- the in-flight stream is
closed instead of being paid for, and status reports cancelled."""
from __future__ import annotations

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolCategory,
    ToolRegistry, ToolResult, ToolSpec,
)

from _fakes import resp, tc


# --- cancellation -------------------------------------------------------------

async def test_run_cancelled_before_first_model_call():
    import asyncio

    class _Never:
        async def complete(self, client, **kw):
            raise AssertionError("模型在取消后不应被调用")

    reg = ToolRegistry()
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_Never()))
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    stop = asyncio.Event()
    stop.set()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], stats=stats, stop=stop)]
    assert [e["type"] for e in events] == ["cancelled"]
    assert stats.status == "cancelled"


async def test_run_cancelled_by_callable_after_tool_round():
    calls = {"model": 0}

    class _T:
        async def complete(self, client, **kw):
            calls["model"] += 1
            return resp(tool_calls=[tc()]) if calls["model"] == 1 else resp("late")

    reg = ToolRegistry()

    async def echo_then_flip(ctx, args):
        # 工具执行完毕后请求取消：下一步模型调用必须被拦下
        calls["stop"] = True
        return ToolResult(True, "echoed", str(args))

    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo_then_flip)
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_T()))
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()

    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats,
                                      stop=lambda: calls.get("stop", False))]
    # step1 正常走完（含工具调用），第二轮模型调用前被取消
    assert calls["model"] == 1
    assert events[-1]["type"] == "cancelled"
    assert stats.status == "cancelled"
    assert "assistant" not in [e["type"] for e in events]


async def test_streaming_cancelled_mid_generation():
    """取消发生在流式生成中途：在途的模型流被掐断，而不是付费流完。"""
    import asyncio

    class _SlowStreamTransport:
        def __init__(self):
            self.finished = False   # 正常耗尽（结果已产出）
            self.closed_early = False

        async def complete(self, client, **kw):
            raise AssertionError("stream=True 时应走 complete_stream")

        async def complete_stream(self, client, **kw):
            try:
                for i in range(10):
                    await asyncio.sleep(0.01)
                    yield {"delta": f"chunk{i} "}
                self.finished = True
                yield {"result": resp("full text")}
            finally:
                self.closed_early = not self.finished

    transport = _SlowStreamTransport()
    rt = AgentRuntime(ToolRegistry(),
                      LLMConfig(model="m", base_url="x", api_key="k",
                                transport=transport, stream=True))
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    saw_delta = {"v": False}

    events = []
    async for ev in rt.run(ctx, [{"role": "user", "content": "hi"}], [],
                           stats=stats, stop=lambda: saw_delta["v"]):
        events.append(ev)
        if ev["type"] == "assistant_delta":
            saw_delta["v"] = True

    assert events[-1]["type"] == "cancelled"
    assert stats.status == "cancelled"
    # 流被提前关闭：既没有走完，也没有产出最终 assistant/usage
    assert transport.closed_early is True and transport.finished is False
    assert not any(e["type"] in ("assistant", "usage") for e in events)
    # 已流出的 delta 之后只剩 cancelled：生成确实被中止
    assert [e["type"] for e in events] == ["step", "assistant_delta", "cancelled"]
