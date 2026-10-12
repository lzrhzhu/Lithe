"""The ReAct loop's core mechanics: step/assistant/usage event order, the
sink contract (records vs events), model-error reporting, malformed
tool-calls answered instead of crashed, tool_call id synthesis, empty
responses retried once, injected http clients, identical-repeat nudges,
LLMConfig field forwarding, and sink error isolation -- all against
scripted transports, no network."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolCategory,
    ToolRegistry, ToolResult, ToolSpec,
)

from _fakes import (
    BoomTransport, FakeTransport, KwTransport, Sink, echo, resp, tc,
)


def _ctx():
    return AgentContext(run_id="r", user_id="u")


def _runtime(responses, *, registry=None, sinks=None, max_steps=3):
    reg = registry or ToolRegistry()
    return AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                       transport=FakeTransport(responses)),
                        sinks=sinks, max_steps=max_steps)


def _runtime_t(transport, *, registry=None, sinks=None, max_steps=3, **kw):
    reg = registry or ToolRegistry()
    return AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                       transport=transport),
                        sinks=sinks, max_steps=max_steps, **kw)


class _MalformedTransport:
    """Returns a malformed tool_call (function as str) after a real one —
    the loop must skip it, not crash the whole run."""

    def __init__(self):
        self.calls = 0

    async def complete(self, client, **kw):
        self.calls += 1
        if self.calls == 1:
            return resp(tool_calls=[
                {"id": "c1", "type": "function",
                 "function": {"name": "echo", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": "echo"},
            ])
        return resp("恢复完成")


async def test_run_final_answer_no_tools():
    rt = _runtime([resp("done")])
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [],
                                      stats=stats)]
    assert [e["type"] for e in events] == ["step", "assistant", "usage"]
    assert events[1]["text"] == "done"
    assert stats.final_text == "done" and stats.status == "done"


async def test_run_tool_then_answer():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    rt = _runtime([resp(tool_calls=[tc()]), resp("final")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode())]
    assert [e["type"] for e in events] == ["step", "usage", "tool_call",
                                         "tool_result", "step", "assistant",
                                         "usage"]
    assert [e for e in events if e["type"] == "assistant"][-1]["text"] == "final"


async def test_sink_receives_events_and_records():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    sink = Sink()
    rt = _runtime([resp(tool_calls=[tc()]), resp("final")],
                  registry=reg, sinks=[sink])
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                 reg.specs_for_mode())]
    roles = [r["role"] for r in sink.records]
    assert roles == ["assistant", "tool", "assistant"]   # tool-call turn + final turn
    assert sink.records[0]["tool_calls"] is not None      # assistant carried tool_calls
    assert sink.records[1]["tool_name"] == "echo"
    assert any(e["type"] == "assistant" for e in sink.events)


async def test_run_model_error_marks_failed():
    rt = AgentRuntime(ToolRegistry(),
                      LLMConfig(model="m", base_url="x", api_key="k",
                                transport=BoomTransport()))
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [],
                                      stats=stats)]
    assert events[-1]["type"] == "error"
    assert stats.status == "failed" and stats.final_text == ""
    # the diagnostic (class + message) rides the event and stats, not just
    # the log — a generic 请稍后重试 leaves a failed run undiagnosable later
    assert "ConnectError" in events[-1]["message"] and "down" in events[-1]["message"]
    assert stats.error == "ConnectError: down"


async def test_run_model_http_error_keeps_status_and_detail():
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    exc = httpx.HTTPStatusError("HTTP 429", request=req, response=resp)

    class _T:
        async def complete(self, client, **kw):
            raise exc

    rt = AgentRuntime(ToolRegistry(),
                      LLMConfig(model="m", base_url="x", api_key="k",
                                transport=_T()))
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [],
                                      stats=stats)]
    err = events[-1]
    assert err["type"] == "error" and err.get("code") == 429
    assert "429" in err["message"]
    assert stats.status == "failed"
    assert stats.error is not None and "429" in stats.error


async def test_run_malformed_tool_call_skipped_not_fatal():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    transport = _MalformedTransport()
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport))
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode())]
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "恢复完成"
    tool_names = [e["name"] for e in events if e["type"] == "tool_call"]
    assert tool_names == ["echo"], "畸形 tool_call 被跳过，正常调用保留"


async def test_run_max_steps_fallback():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    # model always calls a tool, never finishes
    rt = _runtime([resp(tool_calls=[tc()])] * 5, registry=reg, max_steps=2)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert "最大步数" in events[-1]["text"]
    assert stats.last_step == 2
    assert stats.status == "max_steps", "截断的 run 不能伪装成 done"


# --- malformed tool-call arguments fed back to the model ----------------------

async def test_run_bad_json_arguments_become_error_tool_result():
    reg = ToolRegistry()

    calls = {"n": 0}

    async def echo(ctx, args):
        calls["n"] += 1
        return ToolResult(True, "echoed", str(args))

    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    bad = {"id": "c1", "type": "function",
           "function": {"name": "echo", "arguments": "{not json"}}
    rt = _runtime([resp(tool_calls=[bad]), resp("恢复完成")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    events = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    # 工具绝不能拿着空参数被真执行
    assert calls["n"] == 0
    trs = [e for e in events if e["type"] == "tool_result"]
    assert len(trs) == 1 and trs[0]["ok"] is False
    assert "JSON" in trs[0]["error"]
    # 错误以 tool 消息回给模型，run 继续，模型自我纠正后正常收尾
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and "JSON" in tool_msgs[0]["content"]
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "恢复完成"


async def test_run_non_object_json_arguments_rejected():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    bad = {"id": "c1", "type": "function",
           "function": {"name": "echo", "arguments": "[1, 2]"}}
    rt = _runtime([resp(tool_calls=[bad]), resp("ok")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode())]
    trs = [e for e in events if e["type"] == "tool_result"]
    assert trs[0]["ok"] is False and "JSON 对象" in trs[0]["error"]



async def test_llmconfig_temperature_max_tokens_forwarded():
    reg = ToolRegistry()
    transport = KwTransport([{"content": "ok"}])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     temperature=0.3, max_tokens=512,
                                     transport=transport))
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [])]
    assert transport.calls[0]["temperature"] == 0.3
    assert transport.calls[0]["max_tokens"] == 512


# --- identical-repeat guard ------------------------------------------------------

class _LoopToolTransport:
    """每次调用都以相同参数请求同一工具（id 递增），模型永不收尾。"""

    def __init__(self, name="echo", args=None):
        self.name = name
        self.args = json.dumps(args or {})
        self.calls = 0

    async def complete(self, client, **kw):
        self.calls += 1
        return resp(tool_calls=[{"id": f"c{self.calls}", "type": "function",
                                  "function": {"name": self.name,
                                               "arguments": self.args}}])


async def test_repeated_identical_calls_get_nudge():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = _LoopToolTransport()
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport),
                      max_steps=5, repeat_call_limit=3)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    stats = RunStats()
    _ = [e async for e in rt.run(ctx, messages, reg.specs_for_mode(),
                                 stats=stats)]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    # 第 5 步是强制收尾（工具被扣下）：实际执行 4 次
    assert len(tool_msgs) == 4
    # 前 3 次不打扰；第 4 次起在工具结果里追加提示，喂给模型自纠
    assert not any("完全相同的参数" in m["content"] for m in tool_msgs[:3])
    assert "完全相同的参数" in tool_msgs[3]["content"]
    assert "第 4 次" in tool_msgs[3]["content"]
    assert stats.status == "max_steps"


async def test_repeat_call_limit_none_disables_nudge():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_LoopToolTransport()),
                      max_steps=4, repeat_call_limit=None)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    _ = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    assert not any("完全相同的参数" in m["content"] for m in tool_msgs)


async def test_repeated_calls_with_different_args_not_flagged():
    class _CyclingTransport:
        def __init__(self):
            self.calls = 0

        async def complete(self, client, **kw):
            self.calls += 1
            return resp(tool_calls=[tc("echo", {"n": self.calls},
                                         f"c{self.calls}")])

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_CyclingTransport()),
                      max_steps=5, repeat_call_limit=2)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    _ = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 4
    assert not any("完全相同的参数" in m["content"] for m in tool_msgs)


async def test_repeat_nudge_resets_after_mutating_tool():
    """读→写→读（相同参数）：写工具使状态失效，重读是新的观察而非卡死重试，
    不得被重复调用提示误伤（stale-file guard 甚至要求写后重读）。"""
    reads = {"n": 0}

    async def do_read(ctx, args):
        reads["n"] += 1
        return ToolResult(True, "read", "content")

    async def do_write(ctx, args):
        return ToolResult(True, "wrote", args["path"])

    class _Alternating:
        def __init__(self):
            self.calls = 0

        async def complete(self, client, **kw):
            self.calls += 1
            if self.calls > 7:
                return resp("done")
            name = "read_file" if self.calls % 2 == 1 else "write_file"
            return resp(tool_calls=[tc(name, {"path": "a.txt"},
                                         f"c{self.calls}")])

    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "r", category=ToolCategory.READ), do_read)
    reg.register(ToolSpec("write_file", "w", category=ToolCategory.WRITE),
                 do_write)
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_Alternating()),
                      max_steps=8, repeat_call_limit=3)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    stats = RunStats()
    _ = [e async for e in rt.run(ctx, messages, reg.specs_for_mode(),
                                 stats=stats)]
    assert reads["n"] == 4, "四轮相同参数的读都真实执行"
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert not any("完全相同的参数" in m["content"] for m in tool_msgs), \
        "写工具之后的重读不被算作重复"


async def test_repeat_nudge_still_fires_for_identical_write_retries():
    """写工具自身的相同参数重试仍要计数：反复原样重写正是卡死循环，
    不得因 epoch 自增（写会 bump epoch）而被豁免。"""
    reg = ToolRegistry()

    async def do_write(ctx, args):
        return ToolResult(True, "wrote", args["path"])

    reg.register(ToolSpec("write_file", "w", category=ToolCategory.WRITE),
                 do_write)
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_LoopToolTransport(
                                         "write_file", {"path": "a.txt"})),
                      max_steps=5, repeat_call_limit=3)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    _ = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 4  # 第 5 步强制收尾
    assert not any("完全相同的参数" in m["content"] for m in tool_msgs[:3])
    assert "完全相同的参数" in tool_msgs[3]["content"]
    assert "第 4 次" in tool_msgs[3]["content"]



def test_normalize_assistant_keeps_reasoning():
    from lithe.runtime import _normalize_assistant
    item = {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC"}
    out = _normalize_assistant({"content": "", "tool_calls": None,
                                "reasoning": [item]})
    assert out["reasoning"] == [item]
    out2 = _normalize_assistant({"content": "x", "tool_calls": None,
                                 "reasoning": None})
    assert "reasoning" not in out2


def test_runtime_helper_extraction_keeps_compatibility_imports():
    import asyncio
    import queue

    from lithe.runtime import (
        _drain_inbox,
        _normalize_assistant,
        _parse_args,
        _stop_triggered,
    )

    args, error = _parse_args({"function": {"arguments": '{"x": 1}'}})
    assert args == {"x": 1} and error is None
    args, error = _parse_args({"function": {"arguments": "[]"}})
    assert args == {} and error is not None

    async_stop = asyncio.Event()
    assert not _stop_triggered(async_stop)
    async_stop.set()
    assert _stop_triggered(async_stop)
    assert _stop_triggered(lambda: True)

    inbox = queue.Queue()
    inbox.put("steer")
    assert asyncio.run(_drain_inbox(inbox)) == ["steer"]
    assert asyncio.run(_drain_inbox(inbox)) == []

    call = {"function": {"name": "read_file", "arguments": "{}"}}
    normalized = _normalize_assistant({"tool_calls": [call]})
    assert normalized["tool_calls"][0]["id"].startswith("call_")
    assert call["id"] == normalized["tool_calls"][0]["id"]


def test_llm_config_rejects_bad_reasoning_scope():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        LLMConfig(model="m", base_url="x", api_key="k",
                  reasoning_scope="bogus")
    assert LLMConfig(model="m", base_url="x", api_key="k").reasoning_scope == "loop"


# --- sink error isolation -------------------------------------------------------

class _BrokenEventSink:
    async def on_event(self, ctx, event):
        if event.get("type") == "step":
            raise RuntimeError("display sink broken")

    async def on_record(self, ctx, record):
        pass


class _BrokenRecordSink:
    async def on_event(self, ctx, event):
        pass

    async def on_record(self, ctx, record):
        raise RuntimeError("storage sink broken")


async def test_broken_display_sink_does_not_kill_run():
    good = Sink()
    rt = _runtime([resp("done")], sinks=[_BrokenEventSink(), good])
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [])]
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "done"
    # 好的 sink 照常收到全部事件
    assert [e["type"] for e in good.events] == ["step", "assistant", "usage"]


async def test_broken_record_sink_is_best_effort_by_default():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    rt = _runtime([resp(tool_calls=[tc()]), resp("final")],
                  registry=reg, sinks=[_BrokenRecordSink()])
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert any(e["type"] == "assistant" for e in events) and stats.status == "done"


async def test_strict_records_makes_record_failure_fatal():
    import pytest

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       transport=FakeTransport([resp(tool_calls=[tc()])])),
        sinks=[_BrokenRecordSink()], strict_records=True)
    ctx = AgentContext(run_id="r", user_id="u")

    async def consume():
        _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                     reg.specs_for_mode())]

    with pytest.raises(RuntimeError, match="storage sink broken"):
        await consume()


class _NamelessTransport:
    """Returns a tool_call whose function dict has NO name — the dispatch
    side used to KeyError out of the generator (no ERROR event, dead run)."""

    def __init__(self):
        self.calls = 0

    async def complete(self, client, **kw):
        self.calls += 1
        if self.calls == 1:
            return resp(tool_calls=[
                {"id": "c1", "type": "function", "function": {"arguments": "{}"}},
            ])
        return resp("final")


async def test_tool_call_missing_name_answered_not_crashed():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    cfg = LLMConfig(model="m", base_url="x", api_key="k",
                    transport=_NamelessTransport())
    rt = AgentRuntime(reg, cfg, max_steps=3)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    kinds = [e["type"] for e in events]
    assert "error" not in kinds
    # the nameless call is announced and answered as a failed result
    assert kinds.count("tool_call") == 1 and kinds.count("tool_result") == 1
    res = next(e for e in events if e["type"] == "tool_result")
    assert res["ok"] is False and "function.name" in res["error"]
    assert stats.status == "done" and stats.final_text == "final"



class _DirtyUsageTransport:
    """Custom transport returning non-integer usage values — parsing must
    degrade to zeros, not escape the run loop."""

    def __init__(self):
        self.calls = 0

    async def complete(self, client, **kw):
        self.calls += 1
        if self.calls == 1:
            return resp(tool_calls=[tc()])
        return {"content": "final", "tool_calls": [],
                "usage": {"prompt_tokens": "1,234",
                          "completion_tokens": [5], "total_tokens": None,
                          "cost": "x"}}


async def test_dirty_usage_degrades_to_zero_not_crash():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), echo)
    cfg = LLMConfig(model="m", base_url="x", api_key="k",
                    transport=_DirtyUsageTransport())
    rt = AgentRuntime(reg, cfg, max_steps=3)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert stats.status == "done" and stats.final_text == "final"
    usage_events = [e for e in events if e["type"] == "usage"]
    assert all(e["prompt_tokens"] == 0 and e["total_tokens"] == 0
               for e in usage_events)
    assert stats.total_cost == 0.0


async def test_missing_tool_call_id_synthesized_everywhere():
    """A gateway that omits ids gets one synthesized id shared by the
    assistant turn, the tool result record and the in-memory messages."""
    no_id = {"type": "function",
             "function": {"name": "echo", "arguments": "{}"}}
    rt = _runtime_t(FakeTransport([resp(tool_calls=[no_id]), resp("done")]))
    reg = rt.registry
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    sink = Sink()
    rt.sinks = [sink]
    ctx = _ctx()
    messages = [{"role": "user", "content": "hi"}]
    events = [e async for e in rt.run(ctx, messages, [])]
    synth = messages[1]["tool_calls"][0]["id"]
    assert isinstance(synth, str) and synth.startswith("call_") and len(synth) > 8
    # the tool result pairs with the synthesized id …
    tool_records = [r for r in sink.records if r["role"] == "tool"]
    assert tool_records[0]["tool_call_id"] == synth
    assert messages[2]["tool_call_id"] == synth
    # … and the tool_call display event announced it too
    tc_events = [e for e in events if e["type"] == "tool_call"]
    assert tc_events[0]["id"] == synth


async def test_two_steps_missing_ids_do_not_collide():
    t1 = {"type": "function", "function": {"name": "echo", "arguments": "{}"}}
    t2 = {"type": "function", "function": {"name": "echo", "arguments": "{}"}}
    rt = _runtime_t(FakeTransport([resp(tool_calls=[t1]), resp(tool_calls=[t2]),
                                  resp("done")]))
    rt.registry.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    ctx = _ctx()
    messages = [{"role": "user", "content": "hi"}]
    _ = [e async for e in rt.run(ctx, messages, [])]
    id1 = messages[1]["tool_calls"][0]["id"]
    id2 = messages[3]["tool_calls"][0]["id"]
    assert id1 != id2


async def test_empty_response_retried_once_then_real_answer():
    tr = FakeTransport([resp(""), resp("recovered")])
    rt = _runtime_t(tr)
    stats = RunStats()
    events = [e async for e in rt.run(_ctx(),
                                      [{"role": "user", "content": "hi"}], [],
                                      stats=stats)]
    assert tr.calls == 2
    assert stats.status == "done" and stats.final_text == "recovered"
    assert [e for e in events if e["type"] == "error"] == []


async def test_empty_response_twice_ends_with_distinguishable_status():
    tr = FakeTransport([resp(""), resp("")])
    rt = _runtime_t(tr)
    stats = RunStats()
    events = [e async for e in rt.run(_ctx(),
                                      [{"role": "user", "content": "hi"}], [],
                                      stats=stats)]
    assert tr.calls == 2, "恰好重试一次"
    assert stats.status == "empty_response"
    assert stats.final_text == ""
    assert any(e["type"] == "error" for e in events)


async def test_injected_http_client_forwarded_and_reused():
    sentinel = SimpleNamespace(name="shared-client")
    tr = FakeTransport([resp("a"), resp("b")])
    rt = _runtime_t(tr, http_client=sentinel)
    for _ in range(2):
        _ = [e async for e in rt.run(_ctx(),
                                     [{"role": "user", "content": "q"}], [])]
    assert tr.seen_clients == [sentinel, sentinel], "注入 client 跨 run 复用"


async def test_injected_http_client_default_timeout_upgraded():
    # A plain httpx.AsyncClient() carries 5s reads — fatal for reasoning
    # models. The runtime upgrades it to cfg.timeout instead of ignoring it.
    tr = FakeTransport([resp("a")])
    rt = _runtime_t(tr, http_client=httpx.AsyncClient())
    assert rt.http_client.timeout == httpx.Timeout(5.0)
    _ = [e async for e in rt.run(_ctx(),
                                 [{"role": "user", "content": "q"}], [])]
    assert rt.http_client.timeout == httpx.Timeout(180.0)  # cfg.timeout default
    await rt.http_client.aclose()


async def test_injected_http_client_custom_timeout_untouched():
    tr = FakeTransport([resp("a")])
    custom = httpx.Timeout(30.0)
    rt = _runtime_t(tr, http_client=httpx.AsyncClient(timeout=custom))
    _ = [e async for e in rt.run(_ctx(),
                                 [{"role": "user", "content": "q"}], [])]
    # a deliberately configured host timeout is respected, not overridden
    assert rt.http_client.timeout == custom
    await rt.http_client.aclose()


async def test_runtime_forwards_extra_body_and_headers():
    reg = ToolRegistry()
    transport = KwTransport([{"content": "ok"}])
    rt = AgentRuntime(reg, LLMConfig(
        model="m", base_url="x", api_key="k", transport=transport,
        extra_body={"top_p": 0.9}, default_headers={"X-Title": "t"}))
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [])]
    assert transport.calls[0]["extra_body"] == {"top_p": 0.9}
    assert transport.calls[0]["extra_headers"] == {"X-Title": "t"}


async def test_runtime_error_event_carries_extra_body_hint():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "Unknown argument: seed"}})

    realClient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    from lithe.runtime import AgentRuntime as _AR

    class _Factory:
        def __call__(self, timeout=None):
            return realClient

    import lithe.runtime as rt_mod
    original = rt_mod.httpx.AsyncClient
    rt_mod.httpx.AsyncClient = _Factory()
    try:
        rt = _AR(ToolRegistry(), LLMConfig(
            model="m", base_url="http://gw", api_key="k",
            extra_body={"seed": 7}))
        ctx = AgentContext(run_id="r", user_id="u")
        stats = RunStats()
        events = [e async for e in rt.run(ctx,
                                          [{"role": "user", "content": "hi"}],
                                          [], stats=stats)]
    finally:
        rt_mod.httpx.AsyncClient = original
    await realClient.aclose()
    err = [e for e in events if e["type"] == "error"][-1]
    assert err["code"] == 400 and stats.status == "failed"
    assert "模型请求失败（400）" in err["message"]
    assert "seed" in err["message"] and "extra_body" in err["message"]


async def test_runtime_forwards_reasoning_effort():
    reg = ToolRegistry()
    transport = KwTransport([{"content": "ok"}])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport,
                                     reasoning_effort="medium"))
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [])]
    assert transport.calls[0]["reasoning_effort"] == "medium"
