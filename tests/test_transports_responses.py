"""OpenAI Responses wire: ReAct history translated to input items, output[]
parsed back to the unified result, encrypted-reasoning replay with include
degradation, Retry-After handling, SSE streaming, multimodal mapping and
reasoning effort -- all offline via scripted clients and mocked httpx."""
from __future__ import annotations

import json

import httpx
import pytest

from lithe.transports import (
    ResponsesTransport,
    _convert_tools,
    _messages_to_input,
    _parse_output,
)

from _fakes import (
    CaptureClient, Client, Resp, SSEBytes, drain, mock_client, sse,
)


# -- pure translation helpers -------------------------------------------------


def test_messages_to_input_react_loop():
    instructions, items = _messages_to_input(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "echo", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
    )
    assert instructions == "SYS"
    # user → input_text
    assert {"role": "user", "content": [{"type": "input_text", "text": "go"}]} in items
    # assistant tool_call → function_call item
    assert {
        "type": "function_call",
        "call_id": "c1",
        "name": "echo",
        "arguments": "{}",
    } in items
    # tool result → function_call_output (critical for multi-turn responses)
    assert {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "result",
    } in items


def test_convert_tools_flattens_chat_form():
    out = _convert_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "d",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )
    assert out == [
        {
            "type": "function",
            "name": "echo",
            "description": "d",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]


def test_parse_output_message_and_function_call():
    res = _parse_output(
        {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "echo",
                    "arguments": '{"x":1}',
                },
            ],
            "usage": {"total_tokens": 5},
        }
    )
    assert res["content"] == "hi"
    assert res["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"x":1}'},
        }
    ]
    assert res["usage"]["total_tokens"] == 5


def test_parse_output_string_message_content_from_gateways():
    """Some gateways return message content as a plain string, not blocks —
    must flatten, not crash with "'str' object has no attribute 'get'\"."""
    res = _parse_output(
        {
            "output": [
                {"type": "message", "content": "最终答案文本"},
                {"type": "message", "content": None},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "块文本"},
                        "noise",
                    ],
                },
            ]
        }
    )
    assert res["content"] == "最终答案文本块文本noise"


# -- reasoning pass-back -------------------------------------------------------

_R_ITEM = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "先查库"}],
    "encrypted_content": "ENC",
}


def test_parse_output_captures_reasoning_items():
    res = _parse_output(
        {
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "先查库"}],
                    "encrypted_content": "ENC",
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "echo",
                    "arguments": "{}",
                },
            ]
        }
    )
    assert res["reasoning"] == [_R_ITEM]


def test_messages_to_input_replays_reasoning_in_active_loop_only():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "旧问题"},
        {
            "role": "assistant",
            "content": "旧答案",
            "reasoning": [_R_ITEM],
        },  # 上一轮：新用户消息后应被丢弃
        {"role": "user", "content": "新问题"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": [_R_ITEM],
            "tool_calls": [
                {"id": "c1", "function": {"name": "echo", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    _, items = _messages_to_input(msgs)
    reasoning_positions = [
        i for i, it in enumerate(items) if it.get("type") == "reasoning"
    ]
    fc = next(i for i, it in enumerate(items) if it.get("type") == "function_call")
    assert len(reasoning_positions) == 1  # 旧轮的被丢，本轮的保留
    assert reasoning_positions[0] < fc  # 必须在对应 function_call 之前
    assert items[reasoning_positions[0]] == _R_ITEM  # 原样回传（含 encrypted_content）


def test_messages_to_input_reasoning_replay_disabled():
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": [_R_ITEM],
            "tool_calls": [
                {"id": "c1", "function": {"name": "echo", "arguments": "{}"}}
            ],
        },
    ]
    _, items = _messages_to_input(msgs, include_reasoning=False)
    assert not [it for it in items if it.get("type") == "reasoning"]


def test_messages_to_input_conversation_scope_keeps_old_turn_reasoning():
    msgs = [
        {"role": "user", "content": "旧问题"},
        {
            "role": "assistant",
            "content": "旧答案",
            "reasoning": [_R_ITEM],
        },  # loop 模式下会被丢弃
        {"role": "user", "content": "新问题"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": [_R_ITEM],
            "tool_calls": [
                {"id": "c1", "function": {"name": "echo", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    _, items = _messages_to_input(msgs, reasoning_scope="conversation")
    rs = [i for i, it in enumerate(items) if it.get("type") == "reasoning"]
    assert len(rs) == 2  # 新旧两轮的 reasoning 都回传
    assert rs[0] < rs[1]  # 保持时序
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
                return Resp(400, {})
            return Resp(
                200,
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ]
                },
            )

    ResponsesTransport()
    t = ResponsesTransport()
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": [_R_ITEM],
            "tool_calls": [
                {"id": "c1", "function": {"name": "echo", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    result = await t.complete(
        _C(),
        base_url="https://gw/responses",
        api_key="k",
        model="m",
        messages=msgs,
        reasoning_scope="conversation",
    )
    assert result["content"] == "ok"
    assert "include" in payloads[0] and "include" not in payloads[1]
    assert any(i.get("type") == "reasoning" for i in payloads[1]["input"])
    assert not any(i.get("type") == "reasoning" for i in payloads[2]["input"])


async def test_responses_transport_complete_translates_and_parses():
    body = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "echo",
                "arguments": '{"x":1}',
            },
        ],
        "usage": {"total_tokens": 9},
    }
    client = Client(body)
    t = ResponsesTransport()
    result = await t.complete(
        client,
        base_url="https://gw.example/responses",
        api_key="k",
        model="m",
        messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ],
        tools=[{"type": "function", "function": {"name": "echo", "parameters": {}}}],
        tool_choice="auto",
        max_tokens=100,
        temperature=0.5,
    )
    assert result["content"] == "hello"
    assert result["tool_calls"][0]["function"]["name"] == "echo"
    # payload translated to responses shape
    p = client.payload
    assert client.url == "https://gw.example/responses"
    assert p["instructions"] == "SYS"
    assert p["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]
    assert p["tools"][0]["name"] == "echo" and p["tools"][0]["strict"] is False
    assert p["tool_choice"] == "auto"
    assert p["max_output_tokens"] == 100 and p["temperature"] == 0.5


async def test_responses_transport_retries_on_429():
    bodies = [
        Resp(429, {}),
        Resp(
            200,
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ]
            },
        ),
    ]

    class _C:
        async def post(self, url, json=None, headers=None):
            return bodies.pop(0)

    t = ResponsesTransport()
    result = await t.complete(
        _C(),
        base_url="https://gw/responses",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        attempts=2,
        sleep_429=0.01,
    )
    assert result["content"] == "ok"


async def test_responses_transport_requests_encrypted_reasoning():
    body = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
        ]
    }
    client = Client(body)
    t = ResponsesTransport()
    await t.complete(
        client,
        base_url="https://gw/responses",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert client.payload["include"] == ["reasoning.encrypted_content"]
    # 关闭回传时不带 include
    client2 = Client(body)
    await t.complete(
        client2,
        base_url="https://gw/responses",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        include_reasoning=False,
    )
    assert "include" not in client2.payload


async def test_responses_transport_400_on_include_retries_without():
    """网关 400 拒绝 include 字段：翻转实例标记并去掉 include 重试一次。"""
    payloads = []

    class _C:
        def __init__(self):
            self.n = 0

        async def post(self, url, json=None, headers=None):
            payloads.append(dict(json) if isinstance(json, dict) else json)
            self.n += 1
            if self.n == 1:
                return Resp(400, {})
            return Resp(
                200,
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ]
                },
            )

    t = ResponsesTransport()
    result = await t.complete(
        _C(),
        base_url="https://gw/responses",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result["content"] == "ok"
    assert "include" in payloads[0] and "include" not in payloads[1]
    assert t._include_reasoning is False
    # 降级记忆是实例级：另一个实例（另一端点/host）不受影响
    t2 = ResponsesTransport()
    assert t2._include_reasoning is True


async def test_include_reasoning_degradation_is_per_instance():
    """include 降级记忆是实例级：一个网关的 400 不得关闭同进程其它端点的
    reasoning 请求（此前为类属性，进程内全局串扰）。"""
    bodies = []

    class _C:
        async def post(self, url, json=None, headers=None):
            bodies.append(json)
            if len(bodies) == 1:
                return Resp(400, {})  # 网关 A：拒绝 include
            return Resp(
                200,
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ]
                },
            )

    t_a = ResponsesTransport()
    await t_a.complete(
        _C(),
        base_url="https://a/responses",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert t_a._include_reasoning is False
    # 网关 B（新实例）：第一次请求仍带 include
    client_b = Client(
        {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
            ]
        }
    )
    t_b = ResponsesTransport()
    await t_b.complete(
        client_b,
        base_url="https://b/responses",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert client_b.payload["include"] == ["reasoning.encrypted_content"]
    assert t_b._include_reasoning is True


async def test_responses_transport_429_retries_via_retry_after_without_sleep():
    """与 chat 路径对齐：即使 sleep_429=0，只要还有重试名额，429 也按
    Retry-After 重试（旧实现无 sleep_429 时直接放弃）。"""
    bodies = [
        Resp(429, {}, headers={"Retry-After": "0"}),
        Resp(
            200,
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ]
            },
        ),
    ]

    class _C:
        async def post(self, url, json=None, headers=None):
            return bodies.pop(0)

    t = ResponsesTransport()
    result = await t.complete(
        _C(),
        base_url="https://gw/responses",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        attempts=2,
        sleep_429=0.0,
    )
    assert result["content"] == "ok"


def test_parse_output_incomplete_maps_to_length():
    # Responses 用 status+incomplete_details 表达截断：映射到 chat 词汇 "length"
    res = _parse_output(
        {
            "output": [],
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }
    )
    assert res["finish_reason"] == "length"
    res2 = _parse_output({"output": [], "status": "completed"})
    assert res2["finish_reason"] is None
    # 其它原因的 incomplete 不冒充 token 截断
    res3 = _parse_output(
        {
            "output": [],
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
        }
    )
    assert res3["finish_reason"] is None


def test_parse_output_usage_normalized():
    res = _parse_output(
        {"output": [], "usage": {"input_tokens": 4, "output_tokens": 1}}
    )
    assert res["usage"]["prompt_tokens"] == 4
    assert res["usage"]["total_tokens"] == 5


_RESP_FRAMES = [
    "event: response.output_text.delta\n"
    'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
    "event: response.output_text.delta\n"
    'data: {"type":"response.output_text.delta","delta":"!"}\n\n',
    "event: response.completed\n"
    'data: {"type":"response.completed","response":{"output":['
    '{"type":"message","content":[{"type":"output_text","text":"Hi!"}]},'
    '{"type":"function_call","call_id":"c1","name":"echo","arguments":"{}"}'
    '],"usage":{"total_tokens":7}}}\n\n',
]


async def test_responses_complete_stream_deltas_and_final():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return sse(_RESP_FRAMES)

    async with mock_client(handler) as client:
        t = ResponsesTransport()
        deltas, result = await drain(
            t.complete_stream(
                client,
                base_url="http://gw/responses",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
    assert deltas == ["Hi", "!"]
    assert result["content"] == "Hi!"
    assert result["tool_calls"][0]["function"]["name"] == "echo"
    assert result["usage"]["total_tokens"] == 7
    assert seen[0]["stream"] is True and seen[0]["input"][0]["role"] == "user"


async def test_responses_payload_omits_tool_choice_without_tools():
    body = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
        ]
    }
    client = Client(body)
    t = ResponsesTransport()
    await t.complete(
        client,
        base_url="https://gw/responses",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        tool_choice="none",
    )
    assert "tool_choice" not in client.payload and "tools" not in client.payload
    # 有 tools 时照常发送
    client2 = Client(body)
    await t.complete(
        client2,
        base_url="https://gw/responses",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "echo", "parameters": {}}}],
        tool_choice="auto",
    )
    assert client2.payload["tool_choice"] == "auto"


async def test_responses_fatal_4xx_not_retried():
    """401 是致命错误：Responses 路径必须像 chat 路径一样一次即败，
    不按 attempts 空转。"""
    posts = {"n": 0}

    class _C:
        async def post(self, url, json=None, headers=None):
            posts["n"] += 1
            return Resp(401, {})

    t = ResponsesTransport()
    try:
        await t.complete(
            _C(),
            base_url="https://gw/responses",
            api_key="bad",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            attempts=3,
            sleep_err=0.01,
        )
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("401 should raise")
    assert posts["n"] == 1, "致命 4xx 不得重试"


async def test_responses_extra_body_and_default_headers():
    client = CaptureClient(
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
        return httpx.Response(200, content=SSEBytes(frames))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = ResponsesTransport()
    _, result = await drain(t.complete_stream(
        client, base_url="http://gw/responses", api_key="k", model="m",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"top_p": 0.9}))
    await client.aclose()
    assert result["content"] == "ok"
    assert seen[0]["top_p"] == 0.9 and seen[0]["stream"] is True


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


async def test_responses_transport_emits_reasoning_effort():
    client = CaptureClient(
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
    client = CaptureClient(
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
