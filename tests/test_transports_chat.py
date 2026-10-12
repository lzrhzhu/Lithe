"""Chat-completions wire: the one-shot client and the transport together --
payload shape (tool_choice never travels without tools), SSE stream assembly
of deltas/tool_calls/usage, stream_options adaptive downgrade, retry
classification (fatal 4xx once, 429 via Retry-After), extra_body /
default_headers passthrough, reasoning_effort emission, multimodal
passthrough and 400 diagnostics. All offline: scripted clients and mocked
httpx only."""
from __future__ import annotations

import json
import time

import httpx
import pytest

from lithe.transports import ChatCompletionsTransport
from lithe.transports.openai_chat import chat_completion, first_content

from _fakes import SSEBytes, CaptureClient, drain, mock_client, sse


def _ok_body(text="ok"):
    return {"choices": [{"message": {"content": text}}]}


# -- finish_reason passthrough ---------------------------------------------------


def test_to_result_carries_finish_reason():
    res = ChatCompletionsTransport._to_result(
        {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "length"}],
            "usage": {},
        }
    )
    assert res["finish_reason"] == "length"
    assert (
        ChatCompletionsTransport._to_result(
            {"choices": [{"message": {"content": "hi"}}]}
        )["finish_reason"]
        is None
    )


async def test_fatal_401_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    async with mock_client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await chat_completion(client, base_url="http://x", api_key="k",
                                  model="m", messages=[], attempts=3,
                                  sleep_err=0.01)
    assert calls["n"] == 1, "401 重试不可能成功，不应烧掉 attempts"


async def test_fatal_401_still_yields_fallback_after_one_request():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404, text="no such model")

    async with mock_client(handler) as client:
        data = await chat_completion(client, base_url="http://x", api_key="k",
                                     model="m", messages=[], attempts=3,
                                     fallback="抱歉，模型暂不可用。")
    assert first_content(data) == "抱歉，模型暂不可用。"
    assert calls["n"] == 1


async def test_429_honors_retry_after_header():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.05"},
                                  text="slow down")
        return httpx.Response(200, json=_ok_body())

    started = time.monotonic()
    async with mock_client(handler) as client:
        data = await chat_completion(client, base_url="http://x", api_key="k",
                                     model="m", messages=[], attempts=2)
    elapsed = time.monotonic() - started
    assert calls["n"] == 2 and first_content(data) == "ok"
    # sleep_429=0：若不读 Retry-After 会立刻重试；此处必须等满服务器要求
    assert elapsed >= 0.045


async def test_500_then_success_is_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=_ok_body())

    async with mock_client(handler) as client:
        data = await chat_completion(client, base_url="http://x", api_key="k",
                                     model="m", messages=[], attempts=2,
                                     sleep_err=0.01)
    assert calls["n"] == 2 and first_content(data) == "ok"


async def test_malformed_200_is_retryable():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"choices": []})  # 网关坏响应
        return httpx.Response(200, json=_ok_body())

    async with mock_client(handler) as client:
        data = await chat_completion(client, base_url="http://x", api_key="k",
                                     model="m", messages=[], attempts=2)
    assert calls["n"] == 2 and first_content(data) == "ok"


async def test_streamed_tool_calls_leave_id_none_for_single_synthesis():
    """The streaming assembler must not mint per-response call_0 ids — the
    runtime's normalize step owns id synthesis."""
    from lithe.transports.openai_chat import iter_chat_completion

    def handler(request):
        sse = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"name":"f","arguments":"{\\"a\\":"}}]}}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"1}"}}]}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=sse,
                              headers={"content-type": "text/event-stream"})

    async with mock_client(handler) as c:
        result = None
        async for part in iter_chat_completion(c, base_url="http://x",
                                                api_key="k", model="m",
                                                messages=[]):
            if "result" in part:
                result = part["result"]
    tcs = result["choices"][0]["message"]["tool_calls"]
    assert tcs[0]["id"] is None, "streaming 侧不得自行合成 id"
    assert tcs[0]["function"]["arguments"] == '{"a":1}'


_CHAT_FRAMES = [
    'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
    '"function":{"name":"echo","arguments":"{\\"x\\":"}}]}}]}\n\n',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"1}"}}]}}]}\n\n',
    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],'
    '"usage":{"total_tokens":9}}\n\n',
    "data: [DONE]\n\n",
]


async def test_chat_complete_stream_assembles_deltas_toolcalls_usage():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return sse(_CHAT_FRAMES)

    async with mock_client(handler) as client:
        t = ChatCompletionsTransport()
        deltas, result = await drain(
            t.complete_stream(
                client,
                base_url="http://gw",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                tools=[
                    {"type": "function", "function": {"name": "echo", "parameters": {}}}
                ],
            )
        )

    assert deltas == ["Hel", "lo"]
    assert result["content"] == "Hello"
    assert result["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"x":1}'},
        }
    ]
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
        return sse(_CHAT_FRAMES)

    t = ChatCompletionsTransport()
    async with mock_client(handler) as client:
        deltas, result = await drain(
            t.complete_stream(
                client,
                base_url="http://gw",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
    assert result["content"] == "Hello" and deltas == ["Hel", "lo"]
    assert t._include_usage is False
    # 第一次带 stream_options 被 400，第二次省略后成功
    assert len(requests) == 2
    assert "stream_options" in requests[0] and "stream_options" not in requests[1]
    # the downgrade is remembered: the next streamed call omits it up front
    async with mock_client(handler) as client:
        await drain(
            t.complete_stream(
                client,
                base_url="http://gw",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
    assert len(requests) == 3


async def test_chat_stream_retries_429_before_first_delta():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return sse(_CHAT_FRAMES)

    async with mock_client(handler) as client:
        t = ChatCompletionsTransport()
        deltas, result = await drain(
            t.complete_stream(
                client,
                base_url="http://gw",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                attempts=2,
                sleep_429=0.01,
            )
        )
    assert calls["n"] == 2
    assert deltas == ["Hel", "lo"] and result["content"] == "Hello"


def test_chat_payload_omits_tool_choice_without_tools():
    """OpenAI 端点拒绝「有 tool_choice 无 tools」（400）：零工具 host 与
    max-steps 收尾调用（tool_choice="none"）都不能带上该字段。"""
    t = ChatCompletionsTransport()
    extra = t._payload_extra(None, "none", None, None)
    assert "tools" not in extra and "tool_choice" not in extra
    extra = t._payload_extra([], "auto", None, None)
    assert "tool_choice" not in extra
    extra = t._payload_extra(
        [{"type": "function", "function": {"name": "e"}}], "auto", None, None
    )
    assert extra["tool_choice"] == "auto"


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
    async with mock_client(handler) as client:
        try:
            await drain(
                t.complete_stream(
                    client,
                    base_url="http://gw",
                    api_key="k",
                    model="m",
                    messages=[{"role": "user", "content": "hi"}],
                )
            )
        except httpx.HTTPStatusError:
            pass
        else:
            raise AssertionError("401 should raise")
    assert calls["n"] == 1
    assert t._include_usage is True, "401 不得触发 stream_options 降级"


async def test_chat_extra_body_and_default_headers():
    client = CaptureClient(
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
    client = CaptureClient(
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


async def test_chat_stream_extra_body_and_headers():
    frames = ['data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
              'data: [DONE]\n\n']
    seen = []

    def handler(request):
        seen.append((json.loads(request.content), request.headers))
        return httpx.Response(200, content=SSEBytes(frames))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = ChatCompletionsTransport()
    deltas, result = await drain(t.complete_stream(
        client, base_url="http://gw", api_key="k", model="m",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"top_p": 0.9}, extra_headers={"X-Title": "t"}))
    await client.aclose()
    assert deltas == ["ok"] and result["content"] == "ok"
    body, headers = seen[0]
    assert body["top_p"] == 0.9 and body["stream"] is True
    assert headers["x-title"] == "t" and headers["authorization"]


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


async def test_chat_transport_emits_reasoning_effort():
    client = CaptureClient(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    t = ChatCompletionsTransport()
    await t.complete(client, base_url="http://gw", api_key="k", model="m",
                     messages=[{"role": "user", "content": "hi"}],
                     reasoning_effort="high")
    assert client.posts[0]["json"]["reasoning_effort"] == "high"
    # None → 字段不发
    client2 = CaptureClient(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    await t.complete(client2, base_url="http://gw", api_key="k", model="m",
                     messages=[{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in client2.posts[0]["json"]


async def test_chat_transport_passes_multimodal_content_verbatim():
    client = CaptureClient(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    t = ChatCompletionsTransport()
    content = [{"type": "text", "text": "hi"},
               {"type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAA"}}]
    await t.complete(client, base_url="http://gw", api_key="k", model="m",
                     messages=[{"role": "user", "content": content}])
    assert client.posts[0]["json"]["messages"][0]["content"] == content
