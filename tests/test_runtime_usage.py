"""Per-call usage and cost: reasoning pass-back within the tool loop,
usage events with context fullness (tokens, percent, chars), degradation
when the gateway reports nothing, and pricing-based cost computation
(gateway-reported cost wins, cached prompts discounted, budgets enforced
from priced usage)."""
from __future__ import annotations

import pytest

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolCategory,
    ToolRegistry, ToolSpec,
)

from _fakes import FakeTransport, KwTransport, Sink, echo, resp, tc


def _runtime(responses, *, registry=None, sinks=None, max_steps=3):
    reg = registry or ToolRegistry()
    return AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                       transport=FakeTransport(responses)),
                        sinks=sinks, max_steps=max_steps)


# --- per-call usage / context fullness ----------------------------------------

class _UsageTransport:
    """Returns unified results carrying realistic usage dicts."""

    def __init__(self, results):
        self._it = iter(list(results))

    async def complete(self, client, **kw):
        r = next(self._it)
        return {"content": r.get("content", ""), "tool_calls": r.get("tool_calls", []),
                "usage": r.get("usage", {})}


async def test_reasoning_passback_within_tool_loop():
    """第一步的 reasoning 挂到 assistant 消息上、随循环回传给下一次调用，
    并以 REASONING 显示事件（摘要）与 record（原样）双通道输出。"""
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    item = {"type": "reasoning", "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "先查库"}],
            "encrypted_content": "ENC"}

    class _T:
        def __init__(self):
            self.seen = []

        async def complete(self, client, **kw):
            self.seen.append(kw["messages"])
            if len(self.seen) == 1:
                return {"content": "", "tool_calls": [tc()], "reasoning": [item],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                  "reasoning_tokens": 4}}
            return {"content": "done", "reasoning": [],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 2}}

    t = _T()
    sink = Sink()
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=t), sinks=[sink])
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    # 第二次调用看到的 messages 含第一步的 assistant 消息及其 reasoning
    asst = [m for m in t.seen[1] if m.get("role") == "assistant"][0]
    assert asst["reasoning"] == [item]
    # 显示事件：REASONING（摘要文本）在第二个 assistant 之前
    types = [e["type"] for e in events]
    assert "reasoning" in types
    r_ev = [e for e in events if e["type"] == "reasoning"][0]
    assert r_ev["text"] == "先查库"
    assert types.index("reasoning") < types.index("assistant")
    # record 原样携带（供 host 持久化 + 下轮回放）
    rec = [r for r in sink.records if r["role"] == "assistant"][0]
    assert rec["reasoning"] == [item]
    # usage / stats 透出 reasoning_tokens
    u = [e for e in events if e["type"] == "usage"][0]
    assert u["reasoning_tokens"] == 4 and stats.reasoning_tokens == 4


async def test_usage_event_and_stats_breakdown():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       context_window=1000,
                       transport=_UsageTransport([
                           {"content": "", "tool_calls": [tc()],
                            "usage": {"prompt_tokens": 400, "completion_tokens": 100,
                                      "cached_tokens": 300,
                                      "total_tokens": 500, "cost": 0.001}},
                           {"content": "done",
                            "usage": {"prompt_tokens": 800, "completion_tokens": 50,
                                      "cached_tokens": 600,
                                      "total_tokens": 850, "cost": 0.002}},
                       ])))
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    usages = [e for e in events if e["type"] == "usage"]
    assert len(usages) == 2
    first, last = usages
    assert first["prompt_tokens"] == 400 and first["completion_tokens"] == 100
    assert first["cached_tokens"] == 300
    assert first["context_tokens"] == 400 and first["context_percent"] == 40.0
    assert first["context_window"] == 1000 and first["cost"] == 0.001
    assert first["context_chars"] > 0
    assert last["context_percent"] == 80.0
    # 累计 vs 最近一次上下文
    assert stats.prompt_tokens == 1200 and stats.completion_tokens == 150
    assert stats.cached_tokens == 900
    assert stats.total_tokens == 1350 and stats.total_cost == 0.003
    assert stats.context_tokens == 800 and stats.context_percent == 80.0
    assert stats.context_window == 1000


async def test_usage_event_without_api_usage_still_reports_chars():
    rt = _runtime([resp("done")])  # 无 usage 字段
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], stats=stats)]
    usage = [e for e in events if e["type"] == "usage"][-1]
    assert usage["prompt_tokens"] == 0 and usage["context_percent"] is None
    assert usage["context_chars"] > 0, "无 usage 时仍给出字符量指示"
    assert stats.context_percent is None and stats.context_window is None


# --- P1-2: pricing ---------------------------------------------------------------

async def test_pricing_computes_cost_when_gateway_silent():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = KwTransport([
        {"tool_calls": [tc()],
         "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 2_000_000}},
        {"content": "done",
         "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}},
    ])
    cfg = LLMConfig(model="m", base_url="x", api_key="k", transport=transport,
                    pricing={"prompt": 3.0, "completion": 15.0})
    rt = AgentRuntime(reg, cfg)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert stats.status == "done"
    expected = 3 * 1.0 + 15 * 2.0 + (3 * 1000 + 15 * 1000) / 1e6
    assert stats.total_cost == pytest.approx(expected)
    usages = [e for e in events if e["type"] == "usage"]
    assert usages[0]["cost"] == pytest.approx(33.0)


async def test_pricing_enables_max_cost_budget():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = KwTransport([
        {"tool_calls": [tc()], "usage": {"prompt_tokens": 2_000_000}},
        {"content": "never", "usage": {}},
    ])
    cfg = LLMConfig(model="m", base_url="x", api_key="k", transport=transport,
                    pricing={"prompt": 3.0})
    rt = AgentRuntime(reg, cfg, max_cost=5.0)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                 reg.specs_for_mode(), stats=stats)]
    assert len(transport.calls) == 1, "超预算（按定价计）后不再调用模型"
    assert stats.status == "budget_exceeded"
    assert "预算" in stats.final_text


async def test_gateway_cost_wins_over_pricing():
    reg = ToolRegistry()
    transport = KwTransport([
        {"content": "done",
         "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 2_000_000,
                   "cost": 0.5}},
    ])
    cfg = LLMConfig(model="m", base_url="x", api_key="k", transport=transport,
                    pricing={"prompt": 3.0, "completion": 15.0})
    rt = AgentRuntime(reg, cfg)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                 [], stats=stats)]
    assert stats.total_cost == pytest.approx(0.5), "网关上报的 cost 优先"


def test_computed_cost_cached_prompt():
    from lithe.runtime import _computed_cost

    u = {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 800}
    p = {"prompt": 3.0, "completion": 15.0, "cached_prompt": 0.3}
    assert _computed_cost(p, u) == pytest.approx(
        (200 / 1e6) * 3.0 + (800 / 1e6) * 0.3 + (100 / 1e6) * 15.0)
    # 无 cached_prompt：缓存按 prompt 价计（宁可高估）
    assert _computed_cost({"prompt": 3.0, "completion": 15.0}, u) == \
        pytest.approx(3.0 * 1000 / 1e6 + 15.0 * 100 / 1e6)
    # 无表 / 空用量：0.0，不崩
    assert _computed_cost(None, u) == 0.0
    assert _computed_cost({"prompt": 3.0}, {}) == 0.0


def test_pricing_validation():
    with pytest.raises(ValueError):
        LLMConfig(model="m", base_url="x", api_key="k", pricing={"prompt": -1})
    with pytest.raises(ValueError):
        LLMConfig(model="m", base_url="x", api_key="k", pricing={"bogus": 1.0})
    with pytest.raises(ValueError):
        # 只有 cached_prompt、缺 prompt/completion 主体
        LLMConfig(model="m", base_url="x", api_key="k",
                  pricing={"cached_prompt": 0.3})
    LLMConfig(model="m", base_url="x", api_key="k", pricing={"prompt": 3.0})
