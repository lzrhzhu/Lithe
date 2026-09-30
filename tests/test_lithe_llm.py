"""lithe.llm chat client: retry classification and Retry-After handling.
Fatal 4xx (401/404/400/...) must fail after ONE request instead of burning the
attempt budget; 429 must honor the server's Retry-After; 5xx / network errors /
malformed 200s stay retryable. All requests are mocked via httpx.MockTransport."""
from __future__ import annotations

import time

import httpx
import pytest

from lithe.llm import chat_completion, first_content


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok_body(text="ok"):
    return {"choices": [{"message": {"content": text}}]}


async def test_fatal_401_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    async with _client(handler) as client:
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

    async with _client(handler) as client:
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
    async with _client(handler) as client:
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

    async with _client(handler) as client:
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

    async with _client(handler) as client:
        data = await chat_completion(client, base_url="http://x", api_key="k",
                                     model="m", messages=[], attempts=2)
    assert calls["n"] == 2 and first_content(data) == "ok"
