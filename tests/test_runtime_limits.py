"""Run limits: max-steps forced toolless wrap-up (a stubborn model's calls
are never executed), cost/token budgets that cut the run short with an
honest status, finish_reason="length" marking truncated generations, and
the max-steps fallback text reaching the record channel."""
from __future__ import annotations

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolCategory,
    ToolRegistry, ToolResult, ToolSpec,
)

from _fakes import FakeTransport, KwTransport, Sink, echo, resp, tc


def _ctx():
    return AgentContext(run_id="r", user_id="u")


def _runtime_t(transport, *, registry=None, sinks=None, max_steps=3, **kw):
    reg = registry or ToolRegistry()
    return AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                       transport=transport),
                        sinks=sinks, max_steps=max_steps, **kw)


# --- max-step forced wrap-up ----------------------------------------------------



async def test_max_steps_forces_toolless_wrapup_summary():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = KwTransport([
        {"tool_calls": [tc()]},
        {"tool_calls": [tc(cid="c2")]},
        {"content": "已完成两轮检索，结论如下……", "usage": {"prompt_tokens": 10}},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport), max_steps=3)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    # 前两步正常带工具；最后一步被强制无工具
    assert len(transport.calls) == 3
    assert transport.calls[0]["tool_choice"] == "auto"
    assert transport.calls[0]["tools"]
    last = transport.calls[-1]
    assert last["tool_choice"] == "none" and last["tools"] is None
    # 收尾答案成为 final_text，状态如实标记截断
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "已完成两轮检索，结论如下……"
    assert stats.final_text == "已完成两轮检索，结论如下……"
    assert stats.status == "max_steps" and stats.last_step == 3


async def test_max_steps_stubborn_model_tools_not_executed():
    calls = {"tool": 0}

    async def echo(ctx, args):
        calls["tool"] += 1
        return ToolResult(True, "echoed", str(args))

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = KwTransport([
        {"tool_calls": [tc()]},
        # 收尾调用仍固执地要调工具（且无正文）
        {"tool_calls": [tc(cid="c9")], "content": ""},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport), max_steps=2)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert calls["tool"] == 1, "收尾步的工具调用绝不执行"
    assert stats.status == "max_steps"
    assert "最大步数" in stats.final_text
    assert any(e["type"] == "assistant" and "最大步数" in e.get("text", "")
               for e in events)


async def test_single_step_budget_still_offers_tools():
    """max_steps=1 不做强制收尾（否则 host 的工具永远不可用）。"""
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = KwTransport([{"tool_calls": [tc()]}])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport), max_steps=1)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert transport.calls[0]["tool_choice"] == "auto"
    assert transport.calls[0]["tools"], "单步预算仍提供工具"
    assert stats.status == "max_steps" and "最大步数" in stats.final_text
    assert any(e["type"] == "tool_result" for e in events)




# --- run budgets: max_cost / max_total_tokens -----------------------------------

async def test_max_cost_budget_cuts_run_short():
    calls = {"tool": 0}

    async def echo(ctx, args):
        calls["tool"] += 1
        return ToolResult(True, "echoed", str(args))

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = KwTransport([
        {"tool_calls": [tc()], "usage": {"total_tokens": 100, "cost": 1.0}},
        # 累计 2.5 > 2.0：此步的工具调用不得执行，也不得有第三次模型调用
        {"tool_calls": [tc(cid="c2")], "usage": {"total_tokens": 100, "cost": 1.5}},
        {"content": "never reached", "usage": {}},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport), max_cost=2.0)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert len(transport.calls) == 2, "超预算后不再发起模型调用"
    assert calls["tool"] == 1, "超预算步的工具调用不执行"
    assert stats.status == "budget_exceeded"
    assert "预算" in stats.final_text
    assert not any(e["type"] == "tool_result" and e.get("id") == "c2"
                   for e in events)
    # 未应答的 tool_calls 不产生 tool_result 事件；提示语作为收尾文本发出
    assert any(e["type"] == "assistant" and "预算" in e.get("text", "")
               for e in events)


async def test_max_total_tokens_budget_cuts_run_short():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = KwTransport([
        {"tool_calls": [tc()], "usage": {"total_tokens": 60, "cost": 0.01}},
        # 累计 120 > 100
        {"tool_calls": [tc(cid="c2")], "usage": {"total_tokens": 60, "cost": 0.01}},
        {"content": "never", "usage": {}},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport),
                      max_total_tokens=100)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                 reg.specs_for_mode(), stats=stats)]
    assert len(transport.calls) == 2
    assert stats.status == "budget_exceeded"


async def test_budget_not_marked_when_model_already_answered():
    """模型自然给出最终答案的那一步恰好超预算：不算截断，正常收尾。"""
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = KwTransport([
        {"tool_calls": [tc()], "usage": {"cost": 0.6}},
        {"content": "答案完成", "usage": {"cost": 0.6}},  # 累计 1.2 > 1.0
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport), max_cost=1.0)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert stats.status == "done" and stats.final_text == "答案完成"
    assert not any("预算" in e.get("text", "") for e in events
                   if e["type"] == "assistant")


async def test_seeded_stats_budget_blocks_first_model_call():
    """host 用带累计量的 stats 预算跨 run：第一步模型调用直接拦下。"""

    class _Never:
        async def complete(self, client, **kw):
            raise AssertionError("超预算后模型不应被调用")

    stats = RunStats(total_tokens=5000)
    rt = AgentRuntime(ToolRegistry(),
                      LLMConfig(model="m", base_url="x", api_key="k",
                                transport=_Never()),
                      max_total_tokens=1000)
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], stats=stats)]
    assert stats.status == "budget_exceeded"
    assert "预算" in events[-1]["text"]


# --- finish_reason="length": truncated generations are marked, not silent ------

async def test_length_truncated_final_answer_marked():
    reg = ToolRegistry()
    transport = KwTransport([
        {"content": "部分答案", "finish_reason": "length",
         "usage": {"total_tokens": 10}},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport))
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], stats=stats)]
    # 截断标记进入最终文本与事件，用户/模型都不会把半截答案当完整
    assert "截断" in stats.final_text and "部分答案" in stats.final_text
    final = [e for e in events if e["type"] == "assistant"][-1]
    assert "截断" in final["text"]
    assert events[-1]["finish_reason"] == "length", "usage 事件透传截断原因"


async def test_length_truncated_tool_args_error_explains_truncation():
    calls = {"tool": 0}

    async def echo(ctx, args):
        calls["tool"] += 1
        return ToolResult(True, "echoed", str(args))

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    bad = {"id": "c1", "type": "function",
           "function": {"name": "echo", "arguments": '{"path": "a', }}
    transport = KwTransport([
        {"tool_calls": [bad], "finish_reason": "length"},  # 参数 JSON 被截断
        {"content": "恢复完成"},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport))
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode())]
    assert calls["tool"] == 0, "截断的参数绝不能带着半截 JSON 执行"
    trs = [e for e in events if e["type"] == "tool_result"]
    assert len(trs) == 1 and trs[0]["ok"] is False
    # 错误必须说出真实原因（max_tokens 截断），而不是笼统的"非法 JSON"
    assert "max_tokens" in trs[0]["error"]
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "恢复完成"


async def test_max_steps_fallback_text_is_recorded():
    tr = FakeTransport([resp(tool_calls=[tc("echo")])] * 10)
    rt = _runtime_t(tr, max_steps=2)
    rt.registry.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    sink = Sink()
    rt.sinks = [sink]
    ctx = _ctx()
    messages = [{"role": "user", "content": "hi"}]
    _ = [e async for e in rt.run(ctx, messages, [])]
    assert messages[-1]["role"] == "assistant"
    assert "最大步数" in messages[-1]["content"]
    assert any("最大步数" in (r.get("content") or "")
               for r in sink.records if r["role"] == "assistant"), \
        "兜底文本必须进 record 通道，否则 sink 与下轮 replay 都看不到"
