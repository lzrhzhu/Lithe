"""Streaming runs: assistant_delta events followed by the full assistant
event (front ends that ignore deltas lose nothing), records stay
whole-message, graceful fallback when the transport lacks complete_stream,
and the end-to-end path through the real chat SSE parser."""
from __future__ import annotations

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolRegistry,
)

from _fakes import SSEBytes, FakeTransport, Sink, StreamTransport, resp


def _runtime(responses, *, registry=None, sinks=None, max_steps=3):
    reg = registry or ToolRegistry()
    return AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                       transport=FakeTransport(responses)),
                        sinks=sinks, max_steps=max_steps)


async def test_run_streaming_emits_deltas_then_full_assistant():
    reg = ToolRegistry()
    sink = Sink()
    rt = AgentRuntime(
        reg,
        LLMConfig(model="m", base_url="x", api_key="k",
                  transport=StreamTransport(["你", "好", "！"],
                                             resp("你好！")),
                  stream=True),
        sinks=[sink])
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], stats=stats)]
    deltas = [e for e in events if e["type"] == "assistant_delta"]
    assert [e["text"] for e in deltas] == ["你", "好", "！"]
    # 完整 assistant 事件仍然发出（忽略 delta 的前端不丢内容）
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "你好！"
    assert stats.final_text == "你好！"
    # 记录侧（持久化）只收到完整消息，与非流式一致
    assert sink.records == [{"role": "assistant", "content": "你好！",
                             "tool_calls": None}]


async def test_run_streaming_falls_back_when_transport_lacks_support():
    # stream=True 但 transport 没有 complete_stream → 静默回退到 complete
    rt = _runtime([resp("done")])
    rt.llm_config.stream = True
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [])]
    assert [e["type"] for e in events] == ["step", "assistant", "usage"]


async def test_run_streaming_end_to_end_with_chat_transport(monkeypatch):
    """runtime → ChatCompletionsTransport.complete_stream → 真实 SSE 解析全链路。"""
    import httpx

    import lithe.runtime as rt_mod

    frames = [
        'data: {"choices":[{"delta":{"content":"你"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"好"}}]}\n\n',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"total_tokens":5}}\n\n',
        'data: [DONE]\n\n',
    ]

    def handler(request):
        return httpx.Response(200, content=SSEBytes(frames))

    real_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    class _Factory:
        def __call__(self, timeout=None):
            return real_client

    monkeypatch.setattr(rt_mod.httpx, "AsyncClient", _Factory())
    rt = AgentRuntime(
        ToolRegistry(),
        LLMConfig(model="m", base_url="http://gw", api_key="k", stream=True))
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], stats=stats)]
    deltas = [e for e in events if e["type"] == "assistant_delta"]
    assert [e["text"] for e in deltas] == ["你", "好"]
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "你好"
    assert stats.status == "done" and stats.final_text == "你好"
    assert stats.total_tokens == 5
