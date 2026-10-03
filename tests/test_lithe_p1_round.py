"""P1 round: enum validation + tool-arg transforms, ``LLMConfig.extra_body``
/ ``default_headers`` passthrough, pricing-based cost accounting, Responses
multimodal mapping, and the steering inbox. Offline: scripted transports and
mocked httpx only."""
from __future__ import annotations

import asyncio
import json
import queue as thread_queue

import httpx
import pytest

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolCategory,
    ToolRegistry, ToolResult, ToolSpec,
)
from lithe.tools import validate_args
from lithe.transports import (
    ChatCompletionsTransport, ResponsesTransport, _messages_to_input,
)


def _resp(content="", tool_calls=None):
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


async def _echo(ctx, args):
    return ToolResult(ok=True, summary="echoed", content=str(args))


# --- P1-6a: enum validation ------------------------------------------------------

def test_validate_args_checks_enum():
    params = {"type": "object",
              "properties": {"mode": {"type": "string",
                                      "enum": ["fast", "slow"]}}}
    assert validate_args(params, {"mode": "fast"}) is None
    err = validate_args(params, {"mode": "turbo"})
    assert err and "fast" in err and "slow" in err and "turbo" in err


async def test_enum_violation_fed_back_and_model_self_corrects():
    calls = []

    async def pick(ctx, args):
        calls.append(args["mode"])
        return ToolResult(True, "picked", args["mode"])

    reg = ToolRegistry()
    reg.register(ToolSpec(
        "pick_mode", "p", category=ToolCategory.READ,
        parameters={"type": "object",
                    "properties": {"mode": {"type": "string",
                                            "enum": ["fast", "slow"]}},
                    "required": ["mode"]}), pick)

    class _T:
        def __init__(self):
            self.n = 0

        async def complete(self, client, **kw):
            self.n += 1
            if self.n == 1:
                return _resp(tool_calls=[_tc("pick_mode", {"mode": "turbo"},
                                             "c1")])
            if self.n == 2:
                return _resp(tool_calls=[_tc("pick_mode", {"mode": "fast"},
                                             "c2")])
            return _resp("选定 fast")

    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_T()), max_steps=4)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    events = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    # 非法值绝不能执行；错误列出全部合法值供模型自纠
    assert calls == ["fast"]
    trs = [e for e in events if e["type"] == "tool_result"]
    assert trs[0]["ok"] is False and "fast" in trs[0]["error"] \
        and "slow" in trs[0]["error"]
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "选定 fast"


# --- P1-6b: tool-arg transforms --------------------------------------------------

async def test_transform_normalizes_args():
    seen = []

    async def read(ctx, args):
        seen.append(args)
        return ToolResult(True, "ok", str(args))

    async def strip_dot(ctx, name, args):
        assert name == "read_file"
        p = args.get("path")
        if isinstance(p, str):
            return {**args, "path": p.lstrip("./")}
        return args

    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "r", category=ToolCategory.READ), read)
    reg.add_transform(strip_dot)
    res = await reg.dispatch("read_file", {"path": "./a.txt"},
                             AgentContext(run_id="r", user_id="u"))
    assert res.ok
    assert seen == [{"path": "a.txt"}]


async def test_transform_runs_before_validation():
    # 变换把模型给的数字路径规整成字符串，校验因此通过
    async def coerce(ctx, name, args):
        if isinstance(args.get("path"), int):
            args = {**args, "path": str(args["path"])}
        return args

    reg = ToolRegistry()
    reg.register(ToolSpec(
        "read_file", "r", category=ToolCategory.READ,
        parameters={"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]}), _echo)
    reg.add_transform(coerce)
    res = await reg.dispatch("read_file", {"path": 7},
                             AgentContext(run_id="r", user_id="u"))
    assert res.ok, "变换先于校验：规整后的参数应通过校验"


async def test_middleware_sees_transformed_args():
    vetoed = []

    async def canonicalize(ctx, name, args):
        return {**args, "path": "canonical:" + str(args.get("path", ""))}

    async def watch(ctx, name, args):
        vetoed.append(args["path"])
        return None

    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "r", category=ToolCategory.READ), _echo)
    reg.add_transform(canonicalize)
    reg.add_middleware(watch)
    res = await reg.dispatch("read_file", {"path": "x"},
                             AgentContext(run_id="r", user_id="u"))
    assert res.ok
    assert vetoed == ["canonical:x"], "middleware 看到的是变换后的参数"


async def test_transform_returning_junk_rejected():
    async def bad(ctx, name, args):
        return "oops"

    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "r", category=ToolCategory.READ), _echo)
    reg.add_transform(bad)
    res = await reg.dispatch("read_file", {"path": "x"},
                             AgentContext(run_id="r", user_id="u"))
    assert res.ok is False and "变换" in res.content


# --- P1-1: extra_body / default_headers -------------------------------------------

class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return self._body


class _CaptureClient:
    def __init__(self, body):
        self._body = body
        self.posts: list[dict] = []

    async def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return _Resp(200, self._body)


async def test_chat_extra_body_and_default_headers():
    client = _CaptureClient(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    t = ChatCompletionsTransport()
    res = await t.complete(client, base_url="http://gw", api_key="k",
                           model="m",
                           messages=[{"role": "user", "content": "hi"}],
                           temperature=0.2,
                           extra_body={"top_p": 0.9, "enable_thinking": False},
                           extra_headers={"HTTP-Referer": "https://app"})
    assert res["content"] == "ok"
    p = client.posts[0]["json"]
    assert p["top_p"] == 0.9 and p["enable_thinking"] is False
    assert p["temperature"] == 0.2  # 内部字段照常存在
    h = client.posts[0]["headers"]
    assert h["HTTP-Referer"] == "https://app"
    assert h["Authorization"] == "Bearer k" and h["Content-Type"]


async def test_extra_body_cannot_override_internal_keys():
    client = _CaptureClient(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    t = ChatCompletionsTransport()
    await t.complete(client, base_url="http://gw", api_key="k", model="m",
                     messages=[{"role": "user", "content": "hi"}],
                     tools=[{"type": "function",
                             "function": {"name": "e", "parameters": {}}}],
                     tool_choice="auto", temperature=0.5,
                     extra_body={"tool_choice": "none", "tools": [],
                                 "temperature": 0.0})
    p = client.posts[0]["json"]
    assert p["tool_choice"] == "auto"
    assert p["tools"] and p["tools"][0]["function"]["name"] == "e"
    assert p["temperature"] == 0.5


async def test_responses_extra_body_and_default_headers():
    client = _CaptureClient(
        {"output": [{"type": "message",
                     "content": [{"type": "output_text", "text": "ok"}]}]})
    t = ResponsesTransport()
    await t.complete(client, base_url="https://gw/responses", api_key="k",
                     model="m", messages=[{"role": "user", "content": "hi"}],
                     extra_body={"reasoning": {"effort": "low"},
                                 "top_p": 0.9},
                     extra_headers={"X-Title": "app"})
    p = client.posts[0]["json"]
    assert p["reasoning"] == {"effort": "low"} and p["top_p"] == 0.9
    assert p["model"] == "m", "model 不能被 extra_body 覆盖"
    assert client.posts[0]["headers"]["X-Title"] == "app"


class _SSEBytes:
    def __init__(self, frames):
        self._it = iter([f.encode() for f in frames])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


async def _drain(agen):
    deltas, result = [], None
    async for part in agen:
        if "delta" in part:
            deltas.append(part["delta"])
        elif "result" in part:
            result = part["result"]
    return deltas, result


async def test_chat_stream_extra_body_and_headers():
    frames = ['data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
              'data: [DONE]\n\n']
    seen = []

    def handler(request):
        seen.append((json.loads(request.content), request.headers))
        return httpx.Response(200, content=_SSEBytes(frames))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = ChatCompletionsTransport()
    deltas, result = await _drain(t.complete_stream(
        client, base_url="http://gw", api_key="k", model="m",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"top_p": 0.9}, extra_headers={"X-Title": "t"}))
    await client.aclose()
    assert deltas == ["ok"] and result["content"] == "ok"
    body, headers = seen[0]
    assert body["top_p"] == 0.9 and body["stream"] is True
    assert headers["x-title"] == "t" and headers["authorization"]


async def test_responses_stream_extra_body():
    frames = [
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"output":['
        '{"type":"message","content":[{"type":"output_text","text":"ok"}]}]'
        '}}\n\n',
    ]
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_SSEBytes(frames))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = ResponsesTransport()
    _, result = await _drain(t.complete_stream(
        client, base_url="http://gw/responses", api_key="k", model="m",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"top_p": 0.9}))
    await client.aclose()
    assert result["content"] == "ok"
    assert seen[0]["top_p"] == 0.9 and seen[0]["stream"] is True


async def test_runtime_forwards_extra_body_and_headers():
    reg = ToolRegistry()
    transport = _KwTransport([{"content": "ok"}])
    rt = AgentRuntime(reg, LLMConfig(
        model="m", base_url="x", api_key="k", transport=transport,
        extra_body={"top_p": 0.9}, default_headers={"X-Title": "t"}))
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [])]
    assert transport.calls[0]["extra_body"] == {"top_p": 0.9}
    assert transport.calls[0]["extra_headers"] == {"X-Title": "t"}


# --- 400 diagnostics: extra_body fields are pointed at, not guessed at ---------

async def test_chat_400_with_extra_body_annotated():
    def handler(request):
        return httpx.Response(
            400, json={"error": {"message":
                                 "Unrecognized request argument supplied: top_q"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = ChatCompletionsTransport()
    with pytest.raises(httpx.HTTPStatusError) as ei:
        await t.complete(client, base_url="http://gw", api_key="k", model="m",
                         messages=[{"role": "user", "content": "hi"}],
                         extra_body={"top_q": 0.9})
    await client.aclose()
    exc = ei.value
    # 消息带上提示与字段名；网关响应体里的原始报错也被保留
    assert "extra_body" in str(exc) and "top_q" in str(exc)
    assert "Unrecognized request argument" in str(exc)
    # runtime 读取的 lithe_hint 属性
    assert getattr(exc, "lithe_hint", None) and "top_q" in exc.lithe_hint


async def test_chat_400_without_extra_body_not_annotated():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad payload"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = ChatCompletionsTransport()
    with pytest.raises(httpx.HTTPStatusError) as ei:
        await t.complete(client, base_url="http://gw", api_key="k", model="m",
                         messages=[{"role": "user", "content": "hi"}])
    await client.aclose()
    assert "extra_body" not in str(ei.value)
    assert not getattr(ei.value, "lithe_hint", None)


async def test_chat_stream_400_with_extra_body_annotated():
    def handler(request):
        return httpx.Response(400, text="stream rejected")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = ChatCompletionsTransport()
    with pytest.raises(httpx.HTTPStatusError) as ei:
        async for _ in t.complete_stream(
                client, base_url="http://gw", api_key="k", model="m",
                messages=[{"role": "user", "content": "hi"}],
                extra_body={"enable_thinking": True}):
            pass
    await client.aclose()
    assert "extra_body" in str(ei.value)
    assert "enable_thinking" in getattr(ei.value, "lithe_hint", "")


async def test_responses_400_with_extra_body_annotated_after_cascade():
    """Responses 路径：include/reasoning 降级级联先跑，最终 raise 仍带
    extra_body 诊断（两级互不干扰）。"""
    posts = {"n": 0}

    def handler(request):
        posts["n"] += 1
        return httpx.Response(400, json={"error": {"message": "Unknown parameter: effort"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = ResponsesTransport()
    with pytest.raises(httpx.HTTPStatusError) as ei:
        await t.complete(client, base_url="https://gw/responses", api_key="k",
                         model="m", messages=[{"role": "user", "content": "hi"}],
                         extra_body={"reasoning": {"effort": "low"}})
    await client.aclose()
    # 级联先摘掉 include 重试一次，之后 400 上抛并附诊断
    assert posts["n"] == 2
    assert "extra_body" in str(ei.value) and "reasoning" in str(ei.value)
    # hint 列出的是 extra_body 的键；网关原始报错（含 effort）已并入消息
    assert "reasoning" in getattr(ei.value, "lithe_hint", "")
    assert "effort" in str(ei.value)


async def test_runtime_error_event_carries_extra_body_hint():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "Unknown argument: seed"}})

    real_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    from lithe.runtime import AgentRuntime as _AR

    class _Factory:
        def __call__(self, timeout=None):
            return real_client

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
    await real_client.aclose()
    err = [e for e in events if e["type"] == "error"][-1]
    assert err["code"] == 400 and stats.status == "failed"
    assert "模型请求失败（400）" in err["message"]
    assert "seed" in err["message"] and "extra_body" in err["message"]


# --- reasoning_effort: first-class knob, per-protocol emission ------------------

async def test_chat_transport_emits_reasoning_effort():
    client = _CaptureClient(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    t = ChatCompletionsTransport()
    await t.complete(client, base_url="http://gw", api_key="k", model="m",
                     messages=[{"role": "user", "content": "hi"}],
                     reasoning_effort="high")
    assert client.posts[0]["json"]["reasoning_effort"] == "high"
    # None → 字段不发
    client2 = _CaptureClient(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    await t.complete(client2, base_url="http://gw", api_key="k", model="m",
                     messages=[{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in client2.posts[0]["json"]


async def test_responses_transport_emits_reasoning_effort():
    client = _CaptureClient(
        {"output": [{"type": "message",
                     "content": [{"type": "output_text", "text": "ok"}]}]})
    t = ResponsesTransport()
    await t.complete(client, base_url="https://gw/responses", api_key="k",
                     model="m", messages=[{"role": "user", "content": "hi"}],
                     reasoning_effort="low")
    assert client.posts[0]["json"]["reasoning"] == {"effort": "low"}


async def test_reasoning_deep_merge_keeps_extra_body_siblings():
    """extra_body["reasoning"] 的兄弟键（max_tokens/exclude）不被内部
    {"effort": ...} 浅合并吞掉；同键内部优先。"""
    client = _CaptureClient(
        {"output": [{"type": "message",
                     "content": [{"type": "output_text", "text": "ok"}]}]})
    t = ResponsesTransport()
    await t.complete(client, base_url="https://gw/responses", api_key="k",
                     model="m", messages=[{"role": "user", "content": "hi"}],
                     reasoning_effort="high",
                     extra_body={"reasoning": {"max_tokens": 4000,
                                               "exclude": True,
                                               "effort": "low"}})
    r = client.posts[0]["json"]["reasoning"]
    assert r == {"effort": "high", "max_tokens": 4000, "exclude": True}, \
        "兄弟键保留，effort 内部优先"


async def test_runtime_forwards_reasoning_effort():
    reg = ToolRegistry()
    transport = _KwTransport([{"content": "ok"}])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport,
                                     reasoning_effort="medium"))
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}], [])]
    assert transport.calls[0]["reasoning_effort"] == "medium"


# --- P1-2: pricing ---------------------------------------------------------------

async def test_pricing_computes_cost_when_gateway_silent():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    transport = _KwTransport([
        {"tool_calls": [_tc()],
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
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    transport = _KwTransport([
        {"tool_calls": [_tc()], "usage": {"prompt_tokens": 2_000_000}},
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
    transport = _KwTransport([
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


# --- P1-5: multimodal on the responses path --------------------------------------

def test_messages_to_input_maps_text_and_image_blocks():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "看这张图"},
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,AAA"}},
    ]}]
    _, items = _messages_to_input(msgs)
    content = items[0]["content"]
    assert content[0] == {"type": "input_text", "text": "看这张图"}
    assert content[1] == {"type": "input_image", "image_url": "https://x/y.png"}
    assert content[2] == {"type": "input_image",
                          "image_url": "data:image/png;base64,AAA"}


def test_messages_to_input_unmappable_block_raises():
    msgs = [{"role": "user", "content": [{"type": "audio", "data": "..."}]}]
    with pytest.raises(ValueError, match="audio"):
        _messages_to_input(msgs)


def test_messages_to_input_text_only_fast_path_unchanged():
    _, items = _messages_to_input([{"role": "user", "content": "hi"}])
    assert items[0]["content"] == [{"type": "input_text", "text": "hi"}]
    _, items2 = _messages_to_input(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert items2[0]["content"] == [{"type": "input_text", "text": "hi"}]


async def test_chat_transport_passes_multimodal_content_verbatim():
    client = _CaptureClient(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    t = ChatCompletionsTransport()
    content = [{"type": "text", "text": "hi"},
               {"type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAA"}}]
    await t.complete(client, base_url="http://gw", api_key="k", model="m",
                     messages=[{"role": "user", "content": content}])
    assert client.posts[0]["json"]["messages"][0]["content"] == content


def test_message_size_counts_list_content():
    from lithe.runtime import _message_size

    url = "data:image/png;base64," + "A" * 100
    m = {"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}
    assert _message_size(m) == 5 + len(url)
    assert _message_size({"role": "user", "content": "hello"}) == 5


# --- P1-4: steering inbox --------------------------------------------------------

class _SnapTransport:
    """Snapshots the messages it is called with, then plays scripted results."""

    def __init__(self, results):
        self._results = list(results)
        self.snapshots: list[list[dict]] = []

    async def complete(self, client, **kw):
        self.snapshots.append([dict(m) for m in kw["messages"]])
        r = self._results.pop(0)
        return _resp(r.get("content", ""),
                     r.get("tool_calls"))


async def test_steering_inbox_injects_user_message():
    reg = ToolRegistry()
    inbox = asyncio.Queue()

    async def echo_then_steer(ctx, args):
        await inbox.put("别动 src/，只看 tests/")
        return ToolResult(True, "echoed", "ok")

    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ),
                 echo_then_steer)
    transport = _SnapTransport([
        {"tool_calls": [_tc(cid="c1")]},
        {"content": "收到，改看 tests/"},
    ])
    sink = _Sink()
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport),
                      sinks=[sink], max_steps=4)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats,
                                      inbox=inbox)]
    assert stats.status == "done" and stats.final_text == "收到，改看 tests/"
    # 第二次模型调用前，注入消息是上下文的最后一条
    second = transport.snapshots[1]
    assert second[-1] == {"role": "user", "content": "别动 src/，只看 tests/"}
    # 事件 + 记录双通道都看到它
    inj = [e for e in events if e["type"] == "user_injected"]
    assert len(inj) == 1 and inj[0]["text"] == "别动 src/，只看 tests/"
    types = [e["type"] for e in events]
    assert types.index("step") < types.index("user_injected") \
        < types.index("assistant")
    assert {"role": "user", "content": "别动 src/，只看 tests/"} in sink.records


async def test_steering_user_row_persists_through_storesink(tmp_path):
    """The injected line must survive persistence: StoreSink historically
    dropped user records (only assistant/tool were written), so a resumed
    conversation lost the very message that steered it."""
    from lithe.bundles import AgentHost, JsonlRunStore
    from lithe.bundles.host import StoreSink

    reg = ToolRegistry()
    inbox = asyncio.Queue()

    async def echo_then_steer(ctx, args):
        await inbox.put("改看 tests/")
        return ToolResult(True, "echoed", "ok")

    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ),
                 echo_then_steer)
    store = JsonlRunStore(tmp_path)
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=_SnapTransport([
                                        {"tool_calls": [_tc(cid="c1")]},
                                        {"content": "done"},
                                    ])), store, max_steps=4,
                     build_system_prompt=lambda ctx, mode, anchor: "sys")
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in host.run(ctx, "任务", inbox=inbox)]
    rows = store.messages_for_run("r", "u")
    injected = [r for r in rows
                if r.get("role") == "user" and r.get("content") == "改看 tests/"]
    assert len(injected) == 1
    # the host-written task row is not duplicated by the sink
    assert sum(1 for r in rows if r.get("role") == "user") == 2
    assert isinstance(StoreSink(store), StoreSink)  # import sanity


async def test_steering_multiple_messages_preserve_order():
    reg = ToolRegistry()
    inbox = asyncio.Queue()

    async def echo_then_steer(ctx, args):
        await inbox.put("第一条")
        await inbox.put("第二条")
        return ToolResult(True, "echoed", "ok")

    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ),
                 echo_then_steer)
    transport = _SnapTransport([
        {"tool_calls": [_tc(cid="c1")]},
        {"content": "done"},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport))
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                 reg.specs_for_mode(), inbox=inbox)]
    user_msgs = [m for m in transport.snapshots[1] if m.get("role") == "user"]
    assert user_msgs == [{"role": "user", "content": "hi"},
                         {"role": "user", "content": "第一条"},
                         {"role": "user", "content": "第二条"}]


async def test_steering_empty_inbox_changes_nothing():
    reg = ToolRegistry()
    rt_none = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                          transport=_SnapTransport(
                                              [{"content": "done"}])))
    rt_empty = AgentRuntime(reg, LLMConfig(
        model="m", base_url="x", api_key="k",
        transport=_SnapTransport([{"content": "done"}])))
    ctx = AgentContext(run_id="r", user_id="u")
    a = [e async for e in rt_none.run(ctx, [{"role": "user", "content": "hi"}],
                                      [])]
    b = [e async for e in rt_empty.run(ctx, [{"role": "user", "content": "hi"}],
                                       [], inbox=asyncio.Queue())]
    assert [e["type"] for e in a] == [e["type"] for e in b] == \
        ["step", "assistant", "usage"]


async def test_steering_stop_wins_over_inbox():
    reg = ToolRegistry()
    inbox = asyncio.Queue()
    await inbox.put("来不及了")

    class _Never:
        async def complete(self, client, **kw):
            raise AssertionError("取消后模型不应被调用")

    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_Never()))
    ctx = AgentContext(run_id="r", user_id="u")
    stop = asyncio.Event()
    stop.set()
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], stats=stats, stop=stop, inbox=inbox)]
    assert [e["type"] for e in events] == ["cancelled"]
    assert stats.status == "cancelled"
    assert not any(e["type"] == "user_injected" for e in events)


async def test_steering_wrapup_step_skips_drain():
    reg = ToolRegistry()
    inbox = asyncio.Queue()

    async def echo_then_steer(ctx, args):
        await inbox.put("最后一条")
        return ToolResult(True, "echoed", "ok")

    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ),
                 echo_then_steer)
    transport = _SnapTransport([
        {"tool_calls": [_tc(cid="c1")]},
        {"content": "收尾"},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport), max_steps=2)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats,
                                      inbox=inbox)]
    # 第 2 步是强制收尾：不注入（工具已被扣下，注入无法被行动）
    assert stats.final_text == "收尾"
    assert not any(e["type"] == "user_injected" for e in events)
    assert not any(m.get("role") == "user" and m.get("content") == "最后一条"
                   for m in transport.snapshots[1])
    assert not inbox.empty(), "未消费的消息留在宿主队列里"


async def test_steering_accepts_cross_thread_queue():
    reg = ToolRegistry()
    inbox = thread_queue.Queue()
    inbox.put("线程投递")
    transport = _SnapTransport([{"content": "done"}])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport))
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], inbox=inbox)]
    first = transport.snapshots[0]
    assert first == [{"role": "user", "content": "hi"},
                     {"role": "user", "content": "线程投递"}]
    assert any(e["type"] == "user_injected" for e in events)
