"""AgentRuntime: the ReAct loop drives the model ↔ tool cycle, emits display
events, and feeds sinks — without touching storage. The model call is mocked
(via a fake transport) so these pin the loop mechanics
(step/tool/final/error/max-steps) and the sink contract (on_event vs on_record)."""
from __future__ import annotations

import json

import httpx

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolCategory,
    ToolRegistry, ToolResult, ToolSpec,
)


def _resp(content="", tool_calls=None):
    # transport-level shape: unified {content, tool_calls}
    return {"content": content, "tool_calls": tool_calls or []}


def _tc(name="echo", args=None, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


class _Sink:
    def __init__(self):
        self.events = []
        self.records = []

    async def on_event(self, ctx, event):
        self.events.append(event)

    async def on_record(self, ctx, record):
        self.records.append(record)


class _FakeTransport:
    """Yields transport-level results in order."""
    def __init__(self, responses):
        self._it = iter(list(responses))

    async def complete(self, client, **kw):
        return next(self._it)


class _BoomTransport:
    async def complete(self, client, **kw):
        raise httpx.ConnectError("down")


class _MalformedTransport:
    """Returns a malformed tool_call (function as str) after a real one —
    the loop must skip it, not crash the whole run."""

    def __init__(self):
        self.calls = 0

    async def complete(self, client, **kw):
        self.calls += 1
        if self.calls == 1:
            return _resp(tool_calls=[
                {"id": "c1", "type": "function",
                 "function": {"name": "echo", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": "echo"},
            ])
        return _resp("恢复完成")


async def _echo(ctx, args):
    return ToolResult(ok=True, summary="echoed", content=str(args))


def _runtime(responses, *, registry=None, sinks=None, max_steps=3):
    cfg = LLMConfig(model="m", base_url="x", api_key="k",
                    transport=_FakeTransport(responses))
    reg = registry or ToolRegistry()
    return AgentRuntime(reg, cfg, sinks=sinks, max_steps=max_steps)


async def test_run_final_answer_no_tools():
    rt = _runtime([_resp("done")])
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [],
                                      stats=stats)]
    assert [e["type"] for e in events] == ["step", "assistant", "usage"]
    assert events[1]["text"] == "done"
    assert stats.final_text == "done" and stats.status == "done"


async def test_run_tool_then_answer():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), _echo)
    rt = _runtime([_resp(tool_calls=[_tc()]), _resp("final")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode())]
    assert [e["type"] for e in events] == ["step", "usage", "tool_call",
                                         "tool_result", "step", "assistant",
                                         "usage"]
    assert [e for e in events if e["type"] == "assistant"][-1]["text"] == "final"


async def test_sink_receives_events_and_records():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), _echo)
    sink = _Sink()
    rt = _runtime([_resp(tool_calls=[_tc()]), _resp("final")],
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
                                transport=_BoomTransport()))
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [],
                                      stats=stats)]
    assert events[-1]["type"] == "error"
    assert stats.status == "failed" and stats.final_text == ""


async def test_run_malformed_tool_call_skipped_not_fatal():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), _echo)
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
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), _echo)
    # model always calls a tool, never finishes
    rt = _runtime([_resp(tool_calls=[_tc()])] * 5, registry=reg, max_steps=2)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert "最大步数" in events[-1]["text"]
    assert stats.last_step == 2
    assert stats.status == "max_steps", "截断的 run 不能伪装成 done"


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
            return _resp(tool_calls=[_tc()]) if calls["model"] == 1 else _resp("late")

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
                yield {"result": _resp("full text")}
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
    rt = _runtime([_resp(tool_calls=[bad]), _resp("恢复完成")], registry=reg)
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
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), _echo)
    bad = {"id": "c1", "type": "function",
           "function": {"name": "echo", "arguments": "[1, 2]"}}
    rt = _runtime([_resp(tool_calls=[bad]), _resp("ok")], registry=reg)
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode())]
    trs = [e for e in events if e["type"] == "tool_result"]
    assert trs[0]["ok"] is False and "JSON 对象" in trs[0]["error"]


# --- streaming -----------------------------------------------------------------

class _StreamTransport:
    """complete_stream yields deltas then a final assembled result."""

    def __init__(self, deltas, result):
        self._deltas = list(deltas)
        self._result = result

    async def complete(self, client, **kw):
        raise AssertionError("stream=True 时应走 complete_stream")

    async def complete_stream(self, client, **kw):
        for d in self._deltas:
            yield {"delta": d}
        yield {"result": self._result}


async def test_run_streaming_emits_deltas_then_full_assistant():
    reg = ToolRegistry()
    sink = _Sink()
    rt = AgentRuntime(
        reg,
        LLMConfig(model="m", base_url="x", api_key="k",
                  transport=_StreamTransport(["你", "好", "！"],
                                             _resp("你好！")),
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
    rt = _runtime([_resp("done")])
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

    class _SSEBytes:
        def __init__(self, fs):
            self._it = iter([f.encode() for f in fs])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration from None

    def handler(request):
        return httpx.Response(200, content=_SSEBytes(frames))

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
    rt = _runtime([_resp(tool_calls=[_tc("write_file", {"path": "a"}, "c1"),
                                     _tc("write_file", {"path": "b"}, "c2")]),
                   _resp("done")], registry=reg)
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
    rt = _runtime([_resp(tool_calls=[_tc("read_a", {}, "c1"),
                                     _tc("read_b", {}, "c2")]),
                   _resp("done")], registry=reg)
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
    rt = _runtime([_resp(tool_calls=[_tc("read_a", {}, "c1"),
                                      _tc("write_m", {}, "c2"),
                                      _tc("read_b", {}, "c3")]),
                   _resp("done")], registry=reg)
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
    rt = _runtime([_resp(tool_calls=[_tc("slow_tool", {"path": "a"}, "c1")]),
                   _resp("final")], registry=reg)
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
    rt = _runtime([_resp(tool_calls=[_tc("write_file", {"path": "a"}, "c1"),
                                     _tc("write_file", {"path": "b"}, "c2")]),
                   _resp("done")], registry=reg)
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


# --- mid-run context budget ----------------------------------------------------

async def test_context_budget_trims_old_tool_results():
    async def big_echo(ctx, args):
        return ToolResult(True, "echoed", "x" * 3000)

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), big_echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       transport=_FakeTransport([
                           _resp(tool_calls=[_tc(cid="c1")]),
                           _resp(tool_calls=[_tc(cid="c2")]),
                           _resp(tool_calls=[_tc(cid="c3")]),
                           _resp("done")],
                       )),
        context_budget=6000)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    stats = RunStats()
    events = [e async for e in rt.run(ctx, messages, reg.specs_for_mode(),
                                      stats=stats)]
    assert any(e["type"] == "assistant" for e in events) and stats.status == "done"
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    # 最老的工具结果被压缩（保首尾），最新的原样保留
    assert len(tool_msgs[0]["content"]) < 600 and "已截断" in tool_msgs[0]["content"]
    assert len(tool_msgs[1]["content"]) == 3000
    assert len(tool_msgs[2]["content"]) == 3000
    # 结构有效性：每个 tool_call id 仍配对一个 tool 消息
    tc_ids = [tc["id"] for m in messages if m.get("role") == "assistant"
              for tc in (m.get("tool_calls") or [])]
    assert sorted(tc_ids) == sorted(m["tool_call_id"] for m in tool_msgs)


async def test_context_budget_none_disables_trimming():
    async def big_echo(ctx, args):
        return ToolResult(True, "echoed", "y" * 3000)

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), big_echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       transport=_FakeTransport([
                           _resp(tool_calls=[_tc(cid="c1")]),
                           _resp(tool_calls=[_tc(cid="c2")]),
                           _resp("done")])),
        context_budget=None)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    _ = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    assert all(len(m["content"]) == 3000
               for m in messages if m.get("role") == "tool")


async def test_running_context_size_stays_accurate():
    """增量计量的 chars 与全量重扫一致（含 trim 之后）。"""
    from lithe.runtime import _context_size

    async def big_echo(ctx, args):
        return ToolResult(True, "echoed", "z" * 2500)

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), big_echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       transport=_FakeTransport([
                           _resp(tool_calls=[_tc(cid="c1")]),
                           _resp(tool_calls=[_tc(cid="c2")]),
                           _resp(tool_calls=[_tc(cid="c3")]),
                           _resp("done")])),
        context_budget=6000)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    events = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    # 每一步上报的 context_chars 都等于该时刻的真实全量大小
    usages = [e for e in events if e["type"] == "usage"]
    assert len(usages) == 4
    # 语义：context_chars = 本次调用发送的载荷大小（不含当步模型回复自身，
    # 那是在调用之后才 append 的），与全量重扫 messages[:-1] 一致
    assert usages[-1]["context_chars"] == _context_size(messages[:-1])
    assert usages[0]["context_chars"] < usages[-1]["context_chars"]


# --- context trim escalation: drop old exchanges when shrinking isn't enough ----

def _mk_exchange(cid: str, asst_text: str, tool_text: str = "r") -> list[dict]:
    return [
        {"role": "assistant", "content": asst_text,
         "tool_calls": [{"id": cid, "type": "function",
                         "function": {"name": "echo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": cid, "content": tool_text},
    ]


def _pairing_valid(messages: list[dict]) -> bool:
    """每个 assistant tool_call id 恰有一个配对的 tool 消息，反之亦然。"""
    call_ids = [tc.get("id") for m in messages if m.get("role") == "assistant"
                for tc in (m.get("tool_calls") or [])]
    result_ids = [m.get("tool_call_id") for m in messages
                  if m.get("role") == "tool"]
    return sorted(call_ids) == sorted(result_ids)


def test_trim_context_drops_old_exchanges_when_shrinking_insufficient():
    """超限来自不可收缩的正文（assistant 长文本）时：整轮丢弃最旧交换、
    以一条省略注记替代，配对保持有效，system/user 永不丢弃。"""
    from lithe.runtime import _OMITTED_TURNS_NOTE, _context_size, _trim_context

    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "task"}]
    for cid, ch in (("c1", "a"), ("c2", "b"), ("c3", "c"), ("c4", "d")):
        messages += _mk_exchange(cid, ch * 1500)
    assert _context_size(messages) > 4500

    result = _trim_context(messages, budget=4500)

    assert result <= 4500
    notes = [m for m in messages if m.get("content") == _OMITTED_TURNS_NOTE]
    assert len(notes) == 1 and notes[0]["role"] == "assistant"
    # 最旧两轮被整轮丢弃；最新两轮原样保留
    remaining_calls = {tc["id"] for m in messages
                       if m.get("role") == "assistant"
                       for tc in (m.get("tool_calls") or [])}
    assert remaining_calls == {"c3", "c4"}
    assert _pairing_valid(messages)
    # system / user 消息永不丢弃；注记位于保留内容之前
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
    note_idx = messages.index(notes[0])
    kept_start = next(i for i, m in enumerate(messages)
                      if m.get("tool_call_id") == "c3")
    assert note_idx < kept_start


def test_trim_context_no_drop_when_budget_unreachable():
    """即使丢光所有可丢的交换也到不了预算（超限在受保护的近期内容里）：
    不做无谓丢弃，原样返回超限总量。"""
    from lithe.runtime import _OMITTED_TURNS_NOTE, _context_size, _trim_context

    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "task"}]
    for cid, ch in (("c1", "a"), ("c2", "b"), ("c3", "c"), ("c4", "d")):
        messages += _mk_exchange(cid, ch * 1500)
    before = _context_size(messages)

    result = _trim_context(messages, budget=2000)  # 丢光两轮也远不够

    assert result == before, "无谓的丢弃不得发生"
    assert not any(m.get("content") == _OMITTED_TURNS_NOTE for m in messages)
    assert len([m for m in messages if m.get("role") == "tool"]) == 4
    assert _pairing_valid(messages)


def test_trim_context_omission_note_unique_across_passes():
    """第二轮丢弃时旧注记被移除、新注记补位：上下文里永远至多一条注记。"""
    from lithe.runtime import _OMITTED_TURNS_NOTE, _trim_context

    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "task"}]
    for cid, ch in (("c1", "a"), ("c2", "b"), ("c3", "c"), ("c4", "d")):
        messages += _mk_exchange(cid, ch * 1500)
    _trim_context(messages, budget=4500)
    assert len([m for m in messages
                if m.get("content") == _OMITTED_TURNS_NOTE]) == 1

    messages += _mk_exchange("c5", "e" * 1500)   # 又超预算
    _trim_context(messages, budget=4500)

    notes = [m for m in messages if m.get("content") == _OMITTED_TURNS_NOTE]
    assert len(notes) == 1
    remaining_calls = {tc["id"] for m in messages
                       if m.get("role") == "assistant"
                       for tc in (m.get("tool_calls") or [])}
    assert remaining_calls == {"c4", "c5"}
    assert _pairing_valid(messages)


async def test_context_escalation_drop_reaches_model_context():
    """端到端：正文膨胀且压缩无效时，后续模型调用看到的上下文恰含一条
    省略注记、配对完整，run 正常走完（而不是超窗请求）。"""
    from lithe.runtime import _OMITTED_TURNS_NOTE

    big = "x" * 1500

    class _SnapTransport:
        def __init__(self):
            self.snapshots: list[list[dict]] = []

        async def complete(self, client, **kw):
            self.snapshots.append([dict(m) for m in kw["messages"]])
            n = len(self.snapshots)
            if n >= 6:
                return _resp("done")
            return _resp(content=big, tool_calls=[_tc(cid=f"c{n}")])

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    transport = _SnapTransport()
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport),
                      context_budget=4500, max_steps=6)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert stats.final_text == "done"
    assert [e["type"] for e in events][-1] in ("assistant", "usage")

    # 第一次调用：尚无注记；任何时刻至多一条；触发丢弃后恰好一条
    assert not any(m.get("content") == _OMITTED_TURNS_NOTE
                   for m in transport.snapshots[0])
    for snap in transport.snapshots:
        assert len([m for m in snap
                    if m.get("content") == _OMITTED_TURNS_NOTE]) <= 1
        assert _pairing_valid(snap)
    assert len([m for m in transport.snapshots[-1]
                if m.get("content") == _OMITTED_TURNS_NOTE]) == 1
    # 最早的一轮确实从后续上下文中消失了
    dropped = transport.snapshots[-1]
    assert not any(m.get("tool_call_id") == "c1" for m in dropped)


# --- max-step forced wrap-up ----------------------------------------------------

class _KwTransport:
    """Returns queued results and records the kwargs of every call."""

    def __init__(self, results):
        self._results = list(results)
        self.calls: list[dict] = []

    async def complete(self, client, **kw):
        self.calls.append(kw)
        r = self._results.pop(0)
        return {"content": r.get("content", ""),
                "tool_calls": r.get("tool_calls", []),
                "usage": r.get("usage", {}),
                "finish_reason": r.get("finish_reason")}


async def test_max_steps_forces_toolless_wrapup_summary():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    transport = _KwTransport([
        {"tool_calls": [_tc()]},
        {"tool_calls": [_tc(cid="c2")]},
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
    transport = _KwTransport([
        {"tool_calls": [_tc()]},
        # 收尾调用仍固执地要调工具（且无正文）
        {"tool_calls": [_tc(cid="c9")], "content": ""},
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
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    transport = _KwTransport([{"tool_calls": [_tc()]}])
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


async def test_llmconfig_temperature_max_tokens_forwarded():
    reg = ToolRegistry()
    transport = _KwTransport([{"content": "ok"}])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     temperature=0.3, max_tokens=512,
                                     transport=transport))
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [])]
    assert transport.calls[0]["temperature"] == 0.3
    assert transport.calls[0]["max_tokens"] == 512


# --- run budgets: max_cost / max_total_tokens -----------------------------------

async def test_max_cost_budget_cuts_run_short():
    calls = {"tool": 0}

    async def echo(ctx, args):
        calls["tool"] += 1
        return ToolResult(True, "echoed", str(args))

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = _KwTransport([
        {"tool_calls": [_tc()], "usage": {"total_tokens": 100, "cost": 1.0}},
        # 累计 2.5 > 2.0：此步的工具调用不得执行，也不得有第三次模型调用
        {"tool_calls": [_tc(cid="c2")], "usage": {"total_tokens": 100, "cost": 1.5}},
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
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    transport = _KwTransport([
        {"tool_calls": [_tc()], "usage": {"total_tokens": 60, "cost": 0.01}},
        # 累计 120 > 100
        {"tool_calls": [_tc(cid="c2")], "usage": {"total_tokens": 60, "cost": 0.01}},
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
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    transport = _KwTransport([
        {"tool_calls": [_tc()], "usage": {"cost": 0.6}},
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
    transport = _KwTransport([
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
    transport = _KwTransport([
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


# --- identical-repeat guard ------------------------------------------------------

class _LoopToolTransport:
    """每次调用都以相同参数请求同一工具（id 递增），模型永不收尾。"""

    def __init__(self, name="echo", args=None):
        self.name = name
        self.args = json.dumps(args or {})
        self.calls = 0

    async def complete(self, client, **kw):
        self.calls += 1
        return _resp(tool_calls=[{"id": f"c{self.calls}", "type": "function",
                                  "function": {"name": self.name,
                                               "arguments": self.args}}])


async def test_repeated_identical_calls_get_nudge():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
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
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
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
            return _resp(tool_calls=[_tc("echo", {"n": self.calls},
                                         f"c{self.calls}")])

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
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
                return _resp("done")
            name = "read_file" if self.calls % 2 == 1 else "write_file"
            return _resp(tool_calls=[_tc(name, {"path": "a.txt"},
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
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    item = {"type": "reasoning", "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "先查库"}],
            "encrypted_content": "ENC"}

    class _T:
        def __init__(self):
            self.seen = []

        async def complete(self, client, **kw):
            self.seen.append(kw["messages"])
            if len(self.seen) == 1:
                return {"content": "", "tool_calls": [_tc()], "reasoning": [item],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                  "reasoning_tokens": 4}}
            return {"content": "done", "reasoning": [],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 2}}

    t = _T()
    sink = _Sink()
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


def test_normalize_assistant_keeps_reasoning():
    from lithe.runtime import _normalize_assistant
    item = {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC"}
    out = _normalize_assistant({"content": "", "tool_calls": None,
                                "reasoning": [item]})
    assert out["reasoning"] == [item]
    out2 = _normalize_assistant({"content": "x", "tool_calls": None,
                                 "reasoning": None})
    assert "reasoning" not in out2


def test_llm_config_rejects_bad_reasoning_scope():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        LLMConfig(model="m", base_url="x", api_key="k",
                  reasoning_scope="bogus")
    assert LLMConfig(model="m", base_url="x", api_key="k").reasoning_scope == "loop"


async def test_token_calibrated_trim_uses_measured_ratio():
    """token 口径预算：按实测 chars/token 比率把 token 窗口换算成字符预算，
    第二步对旧工具结果触发压缩（旧 400k 字符预算下不会触发）。"""
    big = "x" * 20000

    async def _big_echo(**kw):
        return ToolResult(True, "done", big)

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _big_echo)

    class _T:
        def __init__(self):
            self.calls = []

        async def complete(self, client, **kw):
            self.calls.append(kw["messages"])
            # 第一次调用报大 prompt（与字符数失配）：把 cpt 校准得很小，
            # 使 window×threshold×cpt 预算低于当前对话体积
            if len(self.calls) == 1:
                return {"content": "", "tool_calls": [_tc()],
                        "usage": {"prompt_tokens": 100000}}
            return {"content": "done", "usage": {"prompt_tokens": 50000}}

    t = _T()
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       context_window=1000, transport=t),
        context_budget=None, context_token_threshold=0.5)
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode())]
    assert [e["type"] for e in events][-1] in ("assistant", "usage")
    # 第二次调用的 messages 中，旧工具结果已被压缩（远小于原 20000 字符）
    tool_msgs = [m for m in t.calls[1] if m.get("role") == "tool"]
    assert tool_msgs and all(len(m["content"]) < 20000 for m in tool_msgs)


async def test_usage_event_and_stats_breakdown():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       context_window=1000,
                       transport=_UsageTransport([
                           {"content": "", "tool_calls": [_tc()],
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
    rt = _runtime([_resp("done")])  # 无 usage 字段
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], stats=stats)]
    usage = [e for e in events if e["type"] == "usage"][-1]
    assert usage["prompt_tokens"] == 0 and usage["context_percent"] is None
    assert usage["context_chars"] > 0, "无 usage 时仍给出字符量指示"
    assert stats.context_percent is None and stats.context_window is None


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
    good = _Sink()
    rt = _runtime([_resp("done")], sinks=[_BrokenEventSink(), good])
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [])]
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "done"
    # 好的 sink 照常收到全部事件
    assert [e["type"] for e in good.events] == ["step", "assistant", "usage"]


async def test_broken_record_sink_is_best_effort_by_default():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), _echo)
    rt = _runtime([_resp(tool_calls=[_tc()]), _resp("final")],
                  registry=reg, sinks=[_BrokenRecordSink()])
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert any(e["type"] == "assistant" for e in events) and stats.status == "done"


async def test_strict_records_makes_record_failure_fatal():
    import pytest

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), _echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       transport=_FakeTransport([_resp(tool_calls=[_tc()])])),
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
            return _resp(tool_calls=[
                {"id": "c1", "type": "function", "function": {"arguments": "{}"}},
            ])
        return _resp("final")


async def test_tool_call_missing_name_answered_not_crashed():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), _echo)
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
            return _resp(tool_calls=[_tc()])
        return {"content": "final", "tool_calls": [],
                "usage": {"prompt_tokens": "1,234",
                          "completion_tokens": [5], "total_tokens": None,
                          "cost": "x"}}


async def test_dirty_usage_degrades_to_zero_not_crash():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.WRITE), _echo)
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
