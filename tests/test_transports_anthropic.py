"""Anthropic Messages wire: system/turns translation with tool_result
batching and role merging, thinking-block replay, image blocks,
prompt-cache breakpoints (and their 400 degradation), thinking budgets and
SSE stream accumulation -- all offline via scripted clients and mocked
httpx."""
from __future__ import annotations

import json

import pytest

from lithe.transports import (
    MessagesTransport,
    _messages_to_anthropic,
    _tool_choice_to_anthropic,
    _tools_to_anthropic,
    make_transport,
)

from _fakes import Client, Resp, drain, mock_client, sse


# -- messages transport (Anthropic Messages API) --------------------------------

_ANTH_THINK = {"type": "thinking", "thinking": "先查配置", "signature": "sig1"}


def test_messages_to_anthropic_react_loop():
    system, turns = _messages_to_anthropic(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "echo", "arguments": '{"x":1}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
    )
    assert system == "SYS"
    # user → text block；assistant tool_call → tool_use block（input 为 dict）
    # tool 结果 → 紧随其后的 user turn 里的 tool_result block
    assert turns == [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "c1", "name": "echo", "input": {"x": 1}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "result"}
            ],
        },
    ]


def test_messages_merges_same_role_and_batches_tool_results():
    _, turns = _messages_to_anthropic(
        [
            {"role": "user", "content": "part1"},
            {"role": "user", "content": [{"type": "text", "text": "part2"}]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "c2", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "user", "content": "next"},
        ]
    )
    # 相邻同角色合并（协议要求 user/assistant 交替）
    assert turns[0]["content"] == [
        {"type": "text", "text": "part1"},
        {"type": "text", "text": "part2"},
    ]
    # 并行工具结果汇入同一条 user 消息，并与其后的用户文本合并
    assert turns[2]["role"] == "user"
    kinds = [(b["type"], b.get("tool_use_id")) for b in turns[2]["content"]]
    assert kinds == [("tool_result", "c1"), ("tool_result", "c2"), ("text", None)]


def test_messages_replays_thinking_in_active_loop_only():
    msgs = [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧答案", "reasoning": [_ANTH_THINK]},
        {"role": "user", "content": "新问题"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": [_ANTH_THINK],
            "tool_calls": [
                {"id": "c1", "function": {"name": "echo", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    _, turns = _messages_to_anthropic(msgs)

    def _think_turns(ts):
        return [
            t
            for t in ts
            if t["role"] == "assistant"
            and any(b.get("type") == "thinking" for b in t["content"])
        ]

    assert len(_think_turns(turns)) == 1  # 旧轮丢弃，本轮保留
    loop_turn = next(t for t in turns if any(b.get("id") == "c1" for b in t["content"]))
    assert loop_turn["content"][0] == _ANTH_THINK  # thinking 在 tool_use 前，原样回传
    # conversation scope 保留新旧两轮
    _, turns2 = _messages_to_anthropic(msgs, reasoning_scope="conversation")
    assert len(_think_turns(turns2)) == 2
    # 关闭回放则全部不带
    _, turns3 = _messages_to_anthropic(msgs, include_reasoning=False)
    assert not _think_turns(turns3)
    # 非法 scope 直接拒绝
    with pytest.raises(ValueError):
        _messages_to_anthropic(msgs, reasoning_scope="bogus")


def test_messages_image_blocks():
    _, turns = _messages_to_anthropic(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看图"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                    {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
                ],
            },
        ]
    )
    blocks = turns[0]["content"]
    assert blocks[0] == {"type": "text", "text": "看图"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }
    assert blocks[2] == {
        "type": "image",
        "source": {"type": "url", "url": "https://x/img.png"},
    }
    # 无法映射的块 loud fail，不静默丢弃
    with pytest.raises(ValueError):
        _messages_to_anthropic(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "ftp://x"}}],
                }
            ]
        )


def test_messages_tools_and_choice_mapping():
    tools = _tools_to_anthropic(
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
    assert tools == [
        {"name": "echo", "description": "d", "input_schema": {"type": "object"}}
    ]
    assert _tool_choice_to_anthropic("auto") == {"type": "auto"}
    assert _tool_choice_to_anthropic("required") == {"type": "any"}
    assert _tool_choice_to_anthropic("none") is None
    assert _tool_choice_to_anthropic(
        {"type": "function", "function": {"name": "echo"}}
    ) == {"type": "tool", "name": "echo"}


def test_parse_anthropic_blocks():
    from lithe.transports import _parse_anthropic

    res = _parse_anthropic(
        {
            "content": [
                {"type": "thinking", "thinking": "想一下", "signature": "s"},
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "t1", "name": "echo", "input": {"x": 1}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
    )
    assert res["content"] == "hi"
    assert res["tool_calls"] == [
        {
            "id": "t1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"x": 1}'},
        }
    ]
    assert res["reasoning"][0]["type"] == "thinking"
    assert res["finish_reason"] == "tool_calls"
    assert res["usage"]["prompt_tokens"] == 7
    assert res["usage"]["total_tokens"] == 10
    # stop_reason 映射：max_tokens → length；end_turn → None
    res2 = _parse_anthropic(
        {"content": [{"type": "text", "text": "cut"}], "stop_reason": "max_tokens"}
    )
    assert res2["finish_reason"] == "length"
    res3 = _parse_anthropic(
        {
            "content": [{"type": "tool_use", "id": "t", "name": "e", "input": None}],
            "stop_reason": "end_turn",
        }
    )
    assert res3["finish_reason"] is None
    assert res3["tool_calls"][0]["function"]["arguments"] == "{}"


async def test_messages_transport_complete_translates_and_parses():
    body = {
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "t1", "name": "echo", "input": {"x": 1}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 4, "output_tokens": 5},
    }
    client = Client(body)
    t = MessagesTransport()
    result = await t.complete(
        client,
        base_url="https://gw.example/v1",
        api_key="k",
        model="claude-x",
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
    assert result["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    assert result["usage"]["total_tokens"] == 9
    p = client.payload
    assert client.url == "https://gw.example/v1/messages"
    # system 尾带缓存断点（覆盖 tools+system 前缀），文本等价于原字符串
    assert p["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
    assert p["max_tokens"] == 100 and p["temperature"] == 0.5
    # 空 parameters 兜底为最小合法 schema（协议要求 input_schema 非空）；
    # 断点收尾 system 后 tools 本身无需再标记
    assert p["tools"] == [
        {"name": "echo", "description": "", "input_schema": {"type": "object"}}
    ]
    assert p["tool_choice"] == {"type": "auto"}


async def test_messages_transport_headers_and_default_max_tokens():
    seen = {}

    class _HC:
        async def post(self, url, json=None, headers=None):
            seen["url"] = url
            seen["headers"] = headers
            seen["payload"] = json
            return Resp(200, {"content": [{"type": "text", "text": "ok"}]})

    t = MessagesTransport()
    res = await t.complete(
        _HC(),
        base_url="https://gw/v1",
        api_key="sk",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert res["content"] == "ok"
    assert seen["url"] == "https://gw/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in seen["headers"]
    # 协议必填 max_tokens：未设置时内核默认 4096；无工具时 tools/tool_choice 均不带
    assert seen["payload"]["max_tokens"] == 4096
    assert "tools" not in seen["payload"]
    assert "tool_choice" not in seen["payload"]


async def test_messages_reasoning_effort_maps_to_thinking_budget():
    seen = {}

    class _HC:
        async def post(self, url, json=None, headers=None):
            seen[dict(json)["max_tokens"]] = json
            return Resp(200, {"content": [{"type": "text", "text": "ok"}]})

    t = MessagesTransport()
    await t.complete(
        _HC(),
        base_url="https://gw/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
        max_tokens=2048,
        temperature=0.3,
    )
    p = next(iter(seen.values()))
    assert p["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "temperature" not in p  # 思考开启时协议钉死 temperature
    assert p["max_tokens"] == 16384 + 4096  # max_tokens 必须大于预算 → 抬升
    # minimal / 未知档位：不开启思考，temperature 保留
    seen.clear()
    await t.complete(
        _HC(),
        base_url="https://gw/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="minimal",
        temperature=0.3,
    )
    p2 = next(iter(seen.values()))
    assert "thinking" not in p2 and p2["temperature"] == 0.3
    assert p2["max_tokens"] == 4096


async def test_messages_payload_omits_tools_when_choice_none():
    client = Client({"content": [{"type": "text", "text": "ok"}]})
    t = MessagesTransport()
    await t.complete(
        client,
        base_url="https://gw/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "e", "parameters": {}}}],
        tool_choice="none",
    )
    # 协议无法表达"带 tools 但不用"：none 连 tools 一起省略
    assert "tools" not in client.payload
    assert "tool_choice" not in client.payload


async def test_messages_400_drops_thinking_input():
    payloads = []

    class _C:
        def __init__(self):
            self.n = 0

        async def post(self, url, json=None, headers=None):
            payloads.append(json)
            self.n += 1
            if self.n == 1:
                return Resp(400, {})
            return Resp(200, {"content": [{"type": "text", "text": "ok"}]})

    t = MessagesTransport()
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": [_ANTH_THINK],
            "tool_calls": [
                {"id": "c1", "function": {"name": "echo", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    result = await t.complete(
        _C(),
        base_url="https://gw/v1",
        api_key="k",
        model="m",
        messages=msgs,
        reasoning_scope="conversation",
    )
    assert result["content"] == "ok"
    assert any(
        b.get("type") == "thinking"
        for t_ in payloads[0]["messages"]
        for b in t_["content"]
    )
    assert not any(
        b.get("type") == "thinking"
        for t_ in payloads[1]["messages"]
        for b in t_["content"]
    )


def _cache_marked(block: dict) -> bool:
    return block.get("cache_control") == {"type": "ephemeral"}


async def test_messages_prompt_cache_breakpoints_roll_on_turn_tails():
    client = Client({"content": [{"type": "text", "text": "ok"}]})
    t = MessagesTransport()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "echo", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    tools = [{"type": "function", "function": {"name": "echo", "parameters": {}}}]
    await t.complete(
        client,
        base_url="https://gw/v1",
        api_key="k",
        model="m",
        messages=msgs,
        tools=tools,
    )
    p = client.payload
    # 前缀顺序 tools→system→messages：断点收尾 system 即覆盖 tools+system，
    # tools 本身不重复标记
    assert p["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
    assert not any(_cache_marked(b) for b in p["tools"])
    turns = p["messages"]
    assert [t_["role"] for t_ in turns] == ["user", "assistant", "user"]
    # 只有最后两个 turn 的尾块带断点（上一轮写下的条目＝下一轮的读命中）
    assert not any(_cache_marked(b) for b in turns[0]["content"])
    assert not any(_cache_marked(b) for b in turns[1]["content"][:-1])
    assert _cache_marked(turns[1]["content"][-1])  # tool_use 尾
    assert not any(_cache_marked(b) for b in turns[2]["content"][:-1])
    assert _cache_marked(turns[2]["content"][-1])  # tool_result 尾
    # 协议上限 4 个断点：system + 两个 turn 尾 = 3
    n_marks = (
        sum(_cache_marked(b) for t_ in turns for b in t_["content"]) + 1
    )  # + system
    assert n_marks <= 4
    # 主机侧 spec 对象不被就地污染（前缀字节稳定性）
    assert "cache_control" not in tools[0]


async def test_messages_prompt_cache_no_system_marks_last_tool():
    client = Client({"content": [{"type": "text", "text": "ok"}]})
    t = MessagesTransport()
    tools = [{"type": "function", "function": {"name": "e", "parameters": {}}}]
    await t.complete(
        client,
        base_url="https://gw/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    )
    p = client.payload
    assert "system" not in p
    assert _cache_marked(p["tools"][-1])
    assert "cache_control" not in tools[0]
    # 只有一个 turn：仅该 turn 尾块带断点
    assert _cache_marked(p["messages"][0]["content"][-1])


async def test_messages_400_drops_cache_control_and_remembers():
    payloads = []

    class _C:
        def __init__(self):
            self.n = 0

        async def post(self, url, json=None, headers=None):
            payloads.append(json)
            self.n += 1
            if self.n == 1:
                return Resp(400, {})
            return Resp(200, {"content": [{"type": "text", "text": "ok"}]})

    t = MessagesTransport()
    client = _C()
    await t.complete(
        client,
        base_url="https://gw/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )

    def _any_cache(pl):
        return (_any_cache_in(pl.get("system")) if pl.get("system") else False) or any(
            _cache_marked(b) for t_ in pl["messages"] for b in t_["content"]
        )

    def _any_cache_in(system):
        return isinstance(system, list) and any(
            isinstance(b, dict) and _cache_marked(b) for b in system
        )

    assert _any_cache(payloads[0])
    assert not _any_cache(payloads[1])  # 400 后摘除断点重试
    # 实例级记忆：同一 transport 的后续调用不再带断点
    client2 = Client({"content": [{"type": "text", "text": "ok"}]})
    await t.complete(
        client2,
        base_url="https://gw/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "again"}],
    )
    assert not _any_cache(client2.payload)


_ANTH_FRAMES = [
    "event: message_start\n"
    'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}\n\n',
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"He"}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"y"}}\n\n',
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":1,'
    '"content_block":{"type":"tool_use","id":"t1","name":"echo"}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":1,'
    '"delta":{"type":"input_json_delta","partial_json":"{\\"x\\":"}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":1,'
    '"delta":{"type":"input_json_delta","partial_json":"1}"}}\n\n',
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
    '"usage":{"output_tokens":6}}\n\n',
    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


async def test_messages_complete_stream_deltas_and_final():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return sse(_ANTH_FRAMES)

    async with mock_client(handler) as client:
        t = MessagesTransport()
        deltas, result = await drain(
            t.complete_stream(
                client,
                base_url="http://gw/v1",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
    assert deltas == ["He", "y"]
    assert result["content"] == "Hey"
    # input_json_delta 片段拼接为 chat 形状的 arguments 字符串
    assert result["tool_calls"] == [
        {
            "id": "t1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"x":1}'},
        }
    ]
    # usage：message_start 的 input + message_delta 的 output
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 6
    assert result["finish_reason"] == "tool_calls"
    assert seen[0]["stream"] is True and seen[0]["max_tokens"] == 4096


_ANTH_THINK_FRAMES = [
    "event: message_start\n"
    'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}\n\n',
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"thinking","thinking":""}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"thinking_delta","thinking":"推理"}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"signature_delta","signature":"sig"}}\n\n',
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":1,'
    '"content_block":{"type":"text","text":""}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":1,'
    '"delta":{"type":"text_delta","text":"答案"}}\n\n',
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":2}}\n\n',
    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


async def test_messages_stream_accumulates_thinking_into_reasoning():
    def handler(request):
        return sse(_ANTH_THINK_FRAMES)

    async with mock_client(handler) as client:
        t = MessagesTransport()
        deltas, result = await drain(
            t.complete_stream(
                client,
                base_url="http://gw/v1",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
    assert deltas == ["答案"]  # thinking_delta 不进 delta 通道
    assert result["content"] == "答案"
    assert result["reasoning"] == [
        {"type": "thinking", "thinking": "推理", "signature": "sig"}
    ]
    assert result["finish_reason"] is None


async def test_messages_transport_retries_on_429():
    bodies = [Resp(429, {}), Resp(200, {"content": [{"type": "text", "text": "ok"}]})]

    class _C:
        async def post(self, url, json=None, headers=None):
            return bodies.pop(0)

    t = MessagesTransport()
    result = await t.complete(
        _C(),
        base_url="https://gw/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        attempts=2,
        sleep_429=0.01,
    )
    assert result["content"] == "ok"


def test_make_transport_messages_builtin():
    assert isinstance(make_transport("messages"), MessagesTransport)
