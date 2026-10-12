"""Chat-completions reasoning capture: DeepSeek/GLM-style ``reasoning_content``
and inline ``<think>`` blocks must reach the display layer (REASONING events)
without ever being replayed on the wire — the per-turn-state contract the
deepseek preset's notes promise. All HTTP via httpx.MockTransport."""
from __future__ import annotations

import json

import httpx

from lithe import AgentContext, AgentRuntime, EventType, LLMConfig
from lithe.memory import reasoning_summary_text
from lithe.transports import ChatCompletionsTransport
from lithe.transports.openai_chat import iter_chat_completion


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- _to_result: capture + think-tag extraction --------------------------------

def test_to_result_captures_reasoning_content():
    data = {"choices": [{"message": {
        "content": "答案",
        "reasoning_content": "先分析再回答",
    }, "finish_reason": "stop"}]}
    r = ChatCompletionsTransport._to_result(data)
    assert r["content"] == "答案"
    assert reasoning_summary_text(r["reasoning"]) == "先分析再回答"


def test_to_result_extracts_think_blocks_from_content():
    data = {"choices": [{"message": {
        "content": "<think>内部推理</think>可见的回答",
    }, "finish_reason": "stop"}]}
    r = ChatCompletionsTransport._to_result(data)
    # thinking moved to the reasoning items; the answer stays clean
    assert r["content"] == "可见的回答"
    assert reasoning_summary_text(r["reasoning"]) == "内部推理"


def test_to_result_without_reasoning_stays_reasoning_free():
    data = {"choices": [{"message": {"content": "普通回答"},
                         "finish_reason": "stop"}]}
    r = ChatCompletionsTransport._to_result(data)
    assert r["reasoning"] == [] and r["content"] == "普通回答"


def test_to_result_marks_encrypted_only_reasoning():
    """GPT-6 风格中转只回加密 reasoning_details：无正文可显，但必须有诚实的
    “思考过”标记，否则用户以为没在思考。"""
    data = {"choices": [{"message": {
        "content": "答案",
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "gAAA", "id": "rs_1"},
            {"type": "reasoning.encrypted", "data": "gAAB", "id": "rs_2"},
        ],
    }, "finish_reason": "stop"}]}
    r = ChatCompletionsTransport._to_result(data)
    digest = reasoning_summary_text(r["reasoning"])
    assert "加密" in digest and "2 段" in digest
    # 有可读 reasoning_content 时，加密标记不再叠加
    data["choices"][0]["message"]["reasoning_content"] = "可读思考"
    r = ChatCompletionsTransport._to_result(data)
    assert reasoning_summary_text(r["reasoning"]) == "可读思考"


# --- streaming: reasoning_content deltas reassemble onto the message -----------


def _sse(*chunks) -> bytes:
    out = []
    for c in chunks:
        out.append("data: " + json.dumps(c, ensure_ascii=False))
    out.append("data: [DONE]")
    return ("\n\n".join(out) + "\n\n").encode("utf-8")


async def test_stream_accumulates_reasoning_content_deltas():
    body = _sse(
        {"choices": [{"delta": {"reasoning_content": "先想"}}]},
        {"choices": [{"delta": {"reasoning_content": "清楚，再答"}}]},
        {"choices": [{"delta": {"content": "回答"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"usage": {"prompt_tokens": 3, "completion_tokens": 4}},
    )

    async def handler(request):
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})

    parts = []
    async with _client(handler) as client:
        async for part in iter_chat_completion(
                client, base_url="http://x", api_key="k", model="m",
                messages=[]):
            parts.append(part)
    result = next(p["result"] for p in parts if "result" in p)
    msg = result["choices"][0]["message"]
    assert msg["content"] == "回答"
    assert msg["reasoning_content"] == "先想清楚，再答"
    r = ChatCompletionsTransport._to_result(result)
    assert reasoning_summary_text(r["reasoning"]) == "先想清楚，再答"


async def test_stream_marks_encrypted_reasoning_details():
    body = _sse(
        {"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.encrypted", "data": "gAAA"}]}}]},
        {"choices": [{"delta": {"content": "回答"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )

    async def handler(request):
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})

    parts = []
    async with _client(handler) as client:
        async for part in iter_chat_completion(
                client, base_url="http://x", api_key="k", model="m",
                messages=[]):
            parts.append(part)
    result = next(p["result"] for p in parts if "result" in p)
    msg = result["choices"][0]["message"]
    assert "加密" in msg["reasoning_content"]


# --- no pass-back: outgoing chat messages never carry reasoning ----------------

async def test_complete_strips_reasoning_from_outbound_messages():
    seen: list[list[dict]] = []

    async def handler(request):
        seen.append(request.read())
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "ok"}, "finish_reason": "stop"}]})

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "prev",
         "reasoning": [{"type": "reasoning", "thinking": "旧思考"}]},
        {"role": "assistant", "content": "prev2", "reasoning_content": "旧链"},
        {"role": "user", "content": "again"},
    ]
    async with _client(handler) as client:
        t = ChatCompletionsTransport()
        r = await t.complete(client, base_url="http://x", api_key="k",
                             model="m", messages=messages)
    assert r["content"] == "ok"
    payload = json.loads(seen[0])
    for m in payload["messages"]:
        assert "reasoning" not in m and "reasoning_content" not in m
    # the caller's list itself is untouched (strip works on a copy)
    assert "reasoning" in messages[1]


# --- runtime end-to-end: chat reasoning becomes REASONING events ---------------

async def test_chat_transport_reasoning_reaches_events():
    async def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "完成",
            "reasoning_content": "用户要结果，直接给",
        }, "finish_reason": "stop"}]})

    events = []
    async with _client(handler) as client:
        rt = AgentRuntime(None, LLMConfig(model="m", base_url="http://x",
                                          api_key="k", transport="chat"),
                          http_client=client)
        ctx = AgentContext(run_id="r", user_id="u")
        events = [e async for e in rt.run(
            ctx, [{"role": "user", "content": "hi"}], [])]
    types = [e["type"] for e in events]
    assert EventType.REASONING in types
    r_ev = [e for e in events if e["type"] == EventType.REASONING][0]
    assert r_ev["text"] == "用户要结果，直接给"
    assert types.index(EventType.REASONING) < types.index(EventType.ASSISTANT)
