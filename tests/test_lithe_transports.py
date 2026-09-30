"""LLM transports: chat-completions default + Responses translation. The
responses transport must turn a ReAct-loop message history (system / user /
assistant-with-tool_calls / tool result) into Responses input items, and parse
output[] back — these pin that translation without any network."""
from __future__ import annotations

import json

import httpx
import pytest

from lithe.transports import (
    ChatCompletionsTransport, ResponsesTransport,
    _convert_tools, _messages_to_input, _parse_output, make_transport,
)


# -- pure translation helpers -------------------------------------------------

def test_messages_to_input_react_loop():
    instructions, items = _messages_to_input([
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "function": {"name": "echo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ])
    assert instructions == "SYS"
    # user → input_text
    assert {"role": "user", "content": [{"type": "input_text", "text": "go"}]} in items
    # assistant tool_call → function_call item
    assert {"type": "function_call", "call_id": "c1", "name": "echo",
            "arguments": "{}"} in items
    # tool result → function_call_output (critical for multi-turn responses)
    assert {"type": "function_call_output", "call_id": "c1",
            "output": "result"} in items


def test_convert_tools_flattens_chat_form():
    out = _convert_tools([{"type": "function", "function": {
        "name": "echo", "description": "d", "parameters": {"type": "object"}}}])
    assert out == [{"type": "function", "name": "echo", "description": "d",
                    "parameters": {"type": "object"}, "strict": False}]


def test_parse_output_message_and_function_call():
    res = _parse_output({"output": [
        {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
        {"type": "function_call", "call_id": "c1", "name": "echo",
         "arguments": "{\"x\":1}"},
    ], "usage": {"total_tokens": 5}})
    assert res["content"] == "hi"
    assert res["tool_calls"] == [{"id": "c1", "type": "function",
                                  "function": {"name": "echo", "arguments": "{\"x\":1}"}}]
    assert res["usage"]["total_tokens"] == 5


def test_parse_output_string_message_content_from_gateways():
    """Some gateways return message content as a plain string, not blocks —
    must flatten, not crash with "'str' object has no attribute 'get'\"."""
    res = _parse_output({"output": [
        {"type": "message", "content": "最终答案文本"},
        {"type": "message", "content": None},
        {"type": "message", "content": [
            {"type": "output_text", "text": "块文本"},
            "noise",
        ]},
    ]})
    assert res["content"] == "最终答案文本块文本noise"


# -- reasoning pass-back -------------------------------------------------------

_R_ITEM = {"type": "reasoning", "id": "rs_1",
           "summary": [{"type": "summary_text", "text": "先查库"}],
           "encrypted_content": "ENC"}


def test_parse_output_captures_reasoning_items():
    res = _parse_output({"output": [
        {"type": "reasoning", "id": "rs_1",
         "summary": [{"type": "summary_text", "text": "先查库"}],
         "encrypted_content": "ENC"},
        {"type": "function_call", "call_id": "c1", "name": "echo",
         "arguments": "{}"},
    ]})
    assert res["reasoning"] == [_R_ITEM]


def test_messages_to_input_replays_reasoning_in_active_loop_only():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧答案",
         "reasoning": [_R_ITEM]},          # 上一轮：新用户消息后应被丢弃
        {"role": "user", "content": "新问题"},
        {"role": "assistant", "content": "",
         "reasoning": [_R_ITEM],
         "tool_calls": [{"id": "c1", "function": {"name": "echo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    _, items = _messages_to_input(msgs)
    reasoning_positions = [i for i, it in enumerate(items)
                           if it.get("type") == "reasoning"]
    fc = next(i for i, it in enumerate(items) if it.get("type") == "function_call")
    assert len(reasoning_positions) == 1     # 旧轮的被丢，本轮的保留
    assert reasoning_positions[0] < fc       # 必须在对应 function_call 之前
    assert items[reasoning_positions[0]] == _R_ITEM  # 原样回传（含 encrypted_content）


def test_messages_to_input_reasoning_replay_disabled():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning": [_R_ITEM],
         "tool_calls": [{"id": "c1", "function": {"name": "echo", "arguments": "{}"}}]},
    ]
    _, items = _messages_to_input(msgs, include_reasoning=False)
    assert not [it for it in items if it.get("type") == "reasoning"]


def test_messages_to_input_conversation_scope_keeps_old_turn_reasoning():
    msgs = [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧答案",
         "reasoning": [_R_ITEM]},          # loop 模式下会被丢弃
        {"role": "user", "content": "新问题"},
        {"role": "assistant", "content": "", "reasoning": [_R_ITEM],
         "tool_calls": [{"id": "c1", "function": {"name": "echo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    _, items = _messages_to_input(msgs, reasoning_scope="conversation")
    rs = [i for i, it in enumerate(items) if it.get("type") == "reasoning"]
    assert len(rs) == 2                    # 新旧两轮的 reasoning 都回传
    assert rs[0] < rs[1]                   # 保持时序
    # loop 模式（默认）下旧轮仍被丢弃
    _, items2 = _messages_to_input(msgs)
    assert len([it for it in items2 if it.get("type") == "reasoning"]) == 1
    # 非法 scope 直接拒绝
    import pytest as _pytest
    with _pytest.raises(ValueError):
        _messages_to_input(msgs, reasoning_scope="bogus")


async def test_responses_transport_400_drops_reasoning_input_last():
    """两级降级：400 → 去 include → 仍 400 且输入含 reasoning → 去 reasoning → 成功。"""
    payloads = []

    class _C:
        def __init__(self):
            self.n = 0

        async def post(self, url, json=None, headers=None):
            payloads.append(dict(json) if isinstance(json, dict) else json)
            self.n += 1
            if self.n <= 2:
                return _Resp(400, {})
            return _Resp(200, {"output": [{"type": "message",
                                           "content": [{"type": "output_text", "text": "ok"}]}]})

    ResponsesTransport._INCLUDE_REASONING = True
    t = ResponsesTransport()
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning": [_R_ITEM],
         "tool_calls": [{"id": "c1", "function": {"name": "echo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    result = await t.complete(_C(), base_url="https://gw/responses", api_key="k",
                              model="m", messages=msgs,
                              reasoning_scope="conversation")
    assert result["content"] == "ok"
    assert "include" in payloads[0] and "include" not in payloads[1]
    assert any(i.get("type") == "reasoning" for i in payloads[1]["input"])
    assert not any(i.get("type") == "reasoning" for i in payloads[2]["input"])
    ResponsesTransport._INCLUDE_REASONING = True  # 恢复类标记


# -- transport end-to-end (mocked httpx) --------------------------------------

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


class _Client:
    def __init__(self, body):
        self._body = body
        self.url = None
        self.payload = None

    async def post(self, url, json=None, headers=None):
        self.url = url
        self.payload = json
        return _Resp(200, self._body)


async def test_responses_transport_complete_translates_and_parses():
    body = {"output": [
        {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
        {"type": "function_call", "call_id": "c1", "name": "echo",
         "arguments": "{\"x\":1}"},
    ], "usage": {"total_tokens": 9}}
    client = _Client(body)
    t = ResponsesTransport()
    result = await t.complete(
        client, base_url="https://gw.example/responses", api_key="k", model="m",
        messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "echo", "parameters": {}}}],
        tool_choice="auto", max_tokens=100, temperature=0.5)
    assert result["content"] == "hello"
    assert result["tool_calls"][0]["function"]["name"] == "echo"
    # payload translated to responses shape
    p = client.payload
    assert client.url == "https://gw.example/responses"
    assert p["instructions"] == "SYS"
    assert p["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    assert p["tools"][0]["name"] == "echo" and p["tools"][0]["strict"] is False
    assert p["tool_choice"] == "auto"
    assert p["max_output_tokens"] == 100 and p["temperature"] == 0.5


async def test_responses_transport_retries_on_429():
    bodies = [_Resp(429, {}),
              _Resp(200, {"output": [{"type": "message",
                                      "content": [{"type": "output_text", "text": "ok"}]}]})]

    class _C:
        async def post(self, url, json=None, headers=None):
            return bodies.pop(0)

    t = ResponsesTransport()
    result = await t.complete(_C(), base_url="https://gw/responses", api_key="k",
                              model="m", messages=[{"role": "user", "content": "hi"}],
                              attempts=2, sleep_429=0.01)
    assert result["content"] == "ok"


async def test_responses_transport_requests_encrypted_reasoning():
    body = {"output": [{"type": "message",
                        "content": [{"type": "output_text", "text": "ok"}]}]}
    client = _Client(body)
    t = ResponsesTransport()
    ResponsesTransport._INCLUDE_REASONING = True  # 类标记可能被其它测试翻转
    await t.complete(client, base_url="https://gw/responses", api_key="k",
                     model="m", messages=[{"role": "user", "content": "hi"}])
    assert client.payload["include"] == ["reasoning.encrypted_content"]
    # 关闭回传时不带 include
    client2 = _Client(body)
    await t.complete(client2, base_url="https://gw/responses", api_key="k",
                     model="m", messages=[{"role": "user", "content": "hi"}],
                     include_reasoning=False)
    assert "include" not in client2.payload


async def test_responses_transport_400_on_include_retries_without():
    """网关 400 拒绝 include 字段：翻转类标记并去掉 include 重试一次。"""
    payloads = []

    class _C:
        def __init__(self):
            self.n = 0

        async def post(self, url, json=None, headers=None):
            payloads.append(dict(json) if isinstance(json, dict) else json)
            self.n += 1
            if self.n == 1:
                return _Resp(400, {})
            return _Resp(200, {"output": [{"type": "message",
                                           "content": [{"type": "output_text", "text": "ok"}]}]})

    ResponsesTransport._INCLUDE_REASONING = True
    t = ResponsesTransport()
    result = await t.complete(_C(), base_url="https://gw/responses", api_key="k",
                              model="m", messages=[{"role": "user", "content": "hi"}])
    assert result["content"] == "ok"
    assert "include" in payloads[0] and "include" not in payloads[1]
    assert ResponsesTransport._INCLUDE_REASONING is False
    ResponsesTransport._INCLUDE_REASONING = True  # 恢复，避免影响其它测试


async def test_responses_transport_429_retries_via_retry_after_without_sleep():
    """与 chat 路径对齐：即使 sleep_429=0，只要还有重试名额，429 也按
    Retry-After 重试（旧实现无 sleep_429 时直接放弃）。"""
    bodies = [_Resp(429, {}, headers={"Retry-After": "0"}),
              _Resp(200, {"output": [{"type": "message",
                                      "content": [{"type": "output_text", "text": "ok"}]}]})]

    class _C:
        async def post(self, url, json=None, headers=None):
            return bodies.pop(0)

    t = ResponsesTransport()
    result = await t.complete(_C(), base_url="https://gw/responses", api_key="k",
                              model="m", messages=[{"role": "user", "content": "hi"}],
                              attempts=2, sleep_429=0.0)
    assert result["content"] == "ok"


# -- finish_reason passthrough ---------------------------------------------------

def test_to_result_carries_finish_reason():
    res = ChatCompletionsTransport._to_result({
        "choices": [{"message": {"content": "hi"}, "finish_reason": "length"}],
        "usage": {}})
    assert res["finish_reason"] == "length"
    assert ChatCompletionsTransport._to_result(
        {"choices": [{"message": {"content": "hi"}}]})["finish_reason"] is None


def test_parse_output_incomplete_maps_to_length():
    # Responses 用 status+incomplete_details 表达截断：映射到 chat 词汇 "length"
    res = _parse_output({"output": [], "status": "incomplete",
                         "incomplete_details": {"reason": "max_output_tokens"}})
    assert res["finish_reason"] == "length"
    res2 = _parse_output({"output": [], "status": "completed"})
    assert res2["finish_reason"] is None
    # 其它原因的 incomplete 不冒充 token 截断
    res3 = _parse_output({"output": [], "status": "incomplete",
                          "incomplete_details": {"reason": "content_filter"}})
    assert res3["finish_reason"] is None


# -- usage normalization --------------------------------------------------------

def test_norm_usage_maps_responses_shape():
    from lithe.transports import norm_usage
    u = norm_usage({"input_tokens": 7, "output_tokens": 3,
                    "cost_breakdown": {"total_cost": 0.01}})
    assert u["prompt_tokens"] == 7 and u["completion_tokens"] == 3
    assert u["total_tokens"] == 10          # computed when the vendor omits it
    assert u["cost_breakdown"] == {"total_cost": 0.01}   # extras pass through


def test_norm_usage_completes_chat_shape():
    from lithe.transports import norm_usage
    u = norm_usage({"prompt_tokens": 5, "completion_tokens": 2})
    assert u["total_tokens"] == 7
    assert norm_usage(None) == {}
    assert norm_usage({}) == {}


def test_norm_usage_lifts_cached_tokens():
    from lithe.transports import norm_usage
    # chat-completions shape
    u = norm_usage({"prompt_tokens": 100, "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 80}})
    assert u["cached_tokens"] == 80
    # responses shape
    u = norm_usage({"input_tokens": 50, "output_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 30}})
    assert u["prompt_tokens"] == 50 and u["completion_tokens"] == 2
    assert u["total_tokens"] == 52
    assert u["cached_tokens"] == 30
    # absent details → no cached_tokens key (not a zero that lies)
    assert "cached_tokens" not in norm_usage({"input_tokens": 5,
                                              "output_tokens": 1})


def test_parse_output_usage_normalized():
    res = _parse_output({"output": [], "usage": {"input_tokens": 4,
                                                 "output_tokens": 1}})
    assert res["usage"]["prompt_tokens"] == 4
    assert res["usage"]["total_tokens"] == 5


# -- factory ------------------------------------------------------------------

def test_make_transport_builtins_and_custom():
    assert isinstance(make_transport("chat"), ChatCompletionsTransport)
    assert isinstance(make_transport("responses"), ResponsesTransport)
    custom = ChatCompletionsTransport()
    assert make_transport(custom) is custom
    with pytest.raises(ValueError):
        make_transport("bogus")


# -- streaming (mocked httpx SSE) ----------------------------------------------

class _SSEBytes:
    def __init__(self, frames):
        self._it = iter([f.encode("utf-8") for f in frames])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _sse(frames, status=200):
    return httpx.Response(status, content=_SSEBytes(frames))


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


_CHAT_FRAMES = [
    'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
    '"function":{"name":"echo","arguments":"{\\"x\\":"}}]}}]}\n\n',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"1}"}}]}}]}\n\n',
    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],'
    '"usage":{"total_tokens":9}}\n\n',
    'data: [DONE]\n\n',
]


async def _drain(agen):
    deltas, result = [], None
    async for part in agen:
        if "delta" in part:
            deltas.append(part["delta"])
        elif "result" in part:
            result = part["result"]
    return deltas, result


async def test_chat_complete_stream_assembles_deltas_toolcalls_usage():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _sse(_CHAT_FRAMES)

    async with _client(handler) as client:
        t = ChatCompletionsTransport()
        deltas, result = await _drain(t.complete_stream(
            client, base_url="http://gw", api_key="k", model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "echo",
                                                     "parameters": {}}}]))

    assert deltas == ["Hel", "lo"]
    assert result["content"] == "Hello"
    assert result["tool_calls"] == [{"id": "c1", "type": "function",
                                     "function": {"name": "echo",
                                                  "arguments": '{"x":1}'}}]
    assert result["usage"]["total_tokens"] == 9
    # payload asked to stream and requested in-stream usage
    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"] == {"include_usage": True}


async def test_chat_stream_options_adaptive_downgrade():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if "stream_options" in body:
            return httpx.Response(400, text="stream_options unsupported")
        return _sse(_CHAT_FRAMES)

    t = ChatCompletionsTransport()
    async with _client(handler) as client:
        deltas, result = await _drain(t.complete_stream(
            client, base_url="http://gw", api_key="k", model="m",
            messages=[{"role": "user", "content": "hi"}]))
    assert result["content"] == "Hello" and deltas == ["Hel", "lo"]
    assert t._include_usage is False
    # 第一次带 stream_options 被 400，第二次省略后成功
    assert len(requests) == 2
    assert "stream_options" in requests[0] and "stream_options" not in requests[1]
    # the downgrade is remembered: the next streamed call omits it up front
    async with _client(handler) as client:
        await _drain(t.complete_stream(
            client, base_url="http://gw", api_key="k", model="m",
            messages=[{"role": "user", "content": "hi"}]))
    assert len(requests) == 3


async def test_chat_stream_retries_429_before_first_delta():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return _sse(_CHAT_FRAMES)

    async with _client(handler) as client:
        t = ChatCompletionsTransport()
        deltas, result = await _drain(t.complete_stream(
            client, base_url="http://gw", api_key="k", model="m",
            messages=[{"role": "user", "content": "hi"}],
            attempts=2, sleep_429=0.01))
    assert calls["n"] == 2
    assert deltas == ["Hel", "lo"] and result["content"] == "Hello"


_RESP_FRAMES = [
    'event: response.output_text.delta\n'
    'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
    'event: response.output_text.delta\n'
    'data: {"type":"response.output_text.delta","delta":"!"}\n\n',
    'event: response.completed\n'
    'data: {"type":"response.completed","response":{"output":['
    '{"type":"message","content":[{"type":"output_text","text":"Hi!"}]},'
    '{"type":"function_call","call_id":"c1","name":"echo","arguments":"{}"}'
    '],"usage":{"total_tokens":7}}}\n\n',
]


async def test_responses_complete_stream_deltas_and_final():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _sse(_RESP_FRAMES)

    async with _client(handler) as client:
        t = ResponsesTransport()
        deltas, result = await _drain(t.complete_stream(
            client, base_url="http://gw/responses", api_key="k", model="m",
            messages=[{"role": "user", "content": "hi"}]))
    assert deltas == ["Hi", "!"]
    assert result["content"] == "Hi!"
    assert result["tool_calls"][0]["function"]["name"] == "echo"
    assert result["usage"]["total_tokens"] == 7
    assert seen[0]["stream"] is True and seen[0]["input"][0]["role"] == "user"


# -- tool_choice never travels without tools ------------------------------------

def test_chat_payload_omits_tool_choice_without_tools():
    """OpenAI 端点拒绝「有 tool_choice 无 tools」（400）：零工具 host 与
    max-steps 收尾调用（tool_choice="none"）都不能带上该字段。"""
    t = ChatCompletionsTransport()
    extra = t._payload_extra(None, "none", None, None)
    assert "tools" not in extra and "tool_choice" not in extra
    extra = t._payload_extra([], "auto", None, None)
    assert "tool_choice" not in extra
    extra = t._payload_extra([{"type": "function", "function": {"name": "e"}}],
                             "auto", None, None)
    assert extra["tool_choice"] == "auto"


async def test_responses_payload_omits_tool_choice_without_tools():
    body = {"output": [{"type": "message",
                        "content": [{"type": "output_text", "text": "ok"}]}]}
    client = _Client(body)
    t = ResponsesTransport()
    await t.complete(client, base_url="https://gw/responses", api_key="k",
                     model="m", messages=[{"role": "user", "content": "hi"}],
                     tools=None, tool_choice="none")
    assert "tool_choice" not in client.payload and "tools" not in client.payload
    # 有 tools 时照常发送
    client2 = _Client(body)
    await t.complete(client2, base_url="https://gw/responses", api_key="k",
                     model="m", messages=[{"role": "user", "content": "hi"}],
                     tools=[{"type": "function",
                             "function": {"name": "echo", "parameters": {}}}],
                     tool_choice="auto")
    assert client2.payload["tool_choice"] == "auto"


# -- fatal 4xx fail-fast on the responses paths ----------------------------------

async def test_responses_fatal_4xx_not_retried():
    """401 是致命错误：Responses 路径必须像 chat 路径一样一次即败，
    不按 attempts 空转。"""
    posts = {"n": 0}

    class _C:
        async def post(self, url, json=None, headers=None):
            posts["n"] += 1
            return _Resp(401, {})

    t = ResponsesTransport()
    try:
        await t.complete(_C(), base_url="https://gw/responses", api_key="bad",
                         model="m", messages=[{"role": "user", "content": "hi"}],
                         attempts=3, sleep_err=0.01)
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("401 should raise")
    assert posts["n"] == 1, "致命 4xx 不得重试"


async def test_chat_stream_options_downgrade_not_triggered_by_401():
    """只有 400 才可能是 stream_options 字段被拒；401 直接上抛，
    不翻转 _include_usage、不重发请求。"""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        body = json.loads(request.content)
        assert "stream_options" in body  # 首帧仍带
        return httpx.Response(401, text="bad key")

    t = ChatCompletionsTransport()
    async with _client(handler) as client:
        try:
            await _drain(t.complete_stream(
                client, base_url="http://gw", api_key="k", model="m",
                messages=[{"role": "user", "content": "hi"}]))
        except httpx.HTTPStatusError:
            pass
        else:
            raise AssertionError("401 should raise")
    assert calls["n"] == 1
    assert t._include_usage is True, "401 不得触发 stream_options 降级"
