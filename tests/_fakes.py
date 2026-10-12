"""Shared offline test doubles for the kernel suite.

One toolbox instead of per-file copies: the scripted transports the runtime
tests drive (they never touch httpx), the sink that collects events and
records, small payload builders, and the mocked-httpx helpers the transport
tests use (fake responses, SSE frames, capturing clients)."""
from __future__ import annotations

import asyncio
import json

import httpx

from lithe import ToolResult


# -- payload builders -----------------------------------------------------------


def resp(content="", tool_calls=None):
    """Transport-level unified result shape {content, tool_calls}."""
    return {"content": content, "tool_calls": tool_calls or []}


def tc(name="echo", args=None, cid="c1"):
    """A chat-shaped tool_call with JSON-serialized arguments."""
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


async def echo(ctx, args):
    """The minimal tool every loop test registers."""
    return ToolResult(ok=True, summary="echoed", content=str(args))


class Sink:
    """Collects both channels: display events and persistence records."""

    def __init__(self):
        self.events = []
        self.records = []

    async def on_event(self, ctx, event):
        self.events.append(event)

    async def on_record(self, ctx, record):
        self.records.append(record)


# -- scripted transports ----------------------------------------------------------


class FakeTransport:
    """Yields transport-level results in order; records call count and clients."""

    def __init__(self, responses):
        self._it = iter(list(responses))
        self.calls = 0
        self.seen_clients = []

    async def complete(self, client, **kw):
        self.calls += 1
        self.seen_clients.append(client)
        return next(self._it)


class BoomTransport:
    """complete() always raises — a network error by default."""

    def __init__(self, exc=None):
        self._exc = exc if exc is not None else httpx.ConnectError("down")

    async def complete(self, client, **kw):
        raise self._exc


class StreamTransport:
    """complete_stream yields deltas then a final assembled result."""

    def __init__(self, deltas, result, delay=0.0):
        self._deltas = list(deltas)
        self._result = result
        self._delay = delay

    async def complete(self, client, **kw):
        raise AssertionError("stream=True 时应走 complete_stream")

    async def complete_stream(self, client, **kw):
        if self._delay:
            await asyncio.sleep(self._delay)
        for d in self._deltas:
            yield {"delta": d}
        yield {"result": self._result}


class KwTransport:
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


# -- mocked httpx helpers (transport tests) ---------------------------------------


class Resp:
    """The slice of httpx.Response the transports touch."""

    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return self._body


class Client:
    """Posts to any URL, captures url/payload, returns one fixed body."""

    def __init__(self, body):
        self._body = body
        self.url = None
        self.payload = None

    async def post(self, url, json=None, headers=None):
        self.url = url
        self.payload = json
        return Resp(200, self._body)


class CaptureClient:
    """Like Client, but records every post's url/json/headers (multi-post)."""

    def __init__(self, body):
        self._body = body
        self.posts: list[dict] = []

    async def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return Resp(200, self._body)


class SSEBytes:
    """An async byte iterator over SSE frames — httpx stream body stand-in."""

    def __init__(self, frames):
        self._it = iter([f.encode("utf-8") for f in frames])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def sse(frames, status=200):
    """An httpx.Response carrying SSE frames (use with MockTransport)."""
    return httpx.Response(status, content=SSEBytes(frames))


def mock_client(handler):
    """An AsyncClient over httpx.MockTransport(handler)."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def drain(agen):
    """Collect (deltas, final result) from a complete_stream generator."""
    deltas, result = [], None
    async for part in agen:
        if "delta" in part:
            deltas.append(part["delta"])
        elif "result" in part:
            result = part["result"]
    return deltas, result
