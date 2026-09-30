"""OpenAI-compatible chat-completion client with configurable retry/backoff.

``chat_completion`` centralizes the request boilerplate (headers, POST,
``raise_for_status``, JSON parse + shape validation, retry loop with jittered
backoff) and takes the policy as plain arguments so each caller keeps its own
semantics:

  - ``attempts`` / ``sleep_429`` / ``sleep_err`` : retry count + backoff bases
    (a base of ``0`` means retry immediately).
  - ``fallback`` : a stand-in string returned after exhaustion; ``None`` re-raises.

A 200 response with no ``choices`` / ``message`` is treated as a retryable
failure rather than silently returned as an empty "success". The full parsed
dict is returned so callers that need ``tool_calls`` can read it; text-only
callers extract ``content`` via :func:`first_content`.

This module is part of the storage-free agent kernel — it depends only on the
standard library and httpx, never on a host application.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from email.utils import parsedate_to_datetime
from collections.abc import AsyncIterator

import httpx

log = logging.getLogger("llm")

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

# Statuses worth retrying: the same request might succeed later. Every other
# 4xx (401 bad key, 404 wrong model, 400 bad payload, ...) is fatal — retrying
# cannot fix it, so callers fail fast instead of burning the attempt budget.
_RETRYABLE_STATUS = frozenset({408, 425, 429}).union(range(500, 600))


def _retryable_status(code: int) -> bool:
    return code in _RETRYABLE_STATUS


def _retry_after(resp: httpx.Response) -> float | None:
    """Parse a ``Retry-After`` header (delay-seconds or HTTP-date) into a
    delay in seconds; ``None`` when absent or unparseable."""
    val = resp.headers.get("Retry-After")
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(val)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def bearer_headers(api_key: str) -> dict:
    """Standard ``Authorization: Bearer`` + JSON content-type header."""
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def strip_think(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning leakage from agent-style models."""
    return _THINK_RE.sub("", text).strip()


def _content(message: dict | None) -> str:
    return (message or {}).get("content") or ""


def _jitter(base: float) -> float:
    """Equal-jitter backoff: a sleep of ``base`` scaled by a random factor in
    ``[0.5, 1.5)``, so concurrent callers (e.g. many translation chunks hitting
    a 429 together) don't all retry in lockstep and worsen the rate limit."""
    return base * random.uniform(0.5, 1.5)


async def chat_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    payload_extra: dict | None = None,
    attempts: int = 1,
    sleep_429: float = 0.0,
    sleep_err: float = 0.0,
    fallback: str | None = None,
    log_name: str = "llm",
) -> dict:
    """POST ``{base_url}/chat/completions`` with a configurable retry policy.

    Returns the parsed response dict. On exhaustion:

    - ``fallback is None``  -> re-raise the last error.
    - ``fallback is not None`` -> return a synthetic ``{choices:[{message:
      {content: fallback}}]}`` so callers' content extraction yields the
      fallback uniformly.

    ``sleep_429`` / ``sleep_err`` are base seconds, scaled by the attempt
    number and jittered. ``attempts=1`` means no retry. Retrying is reserved
    for statuses where it can help (408/425/429/5xx, network errors, malformed
    200s); other 4xx are fatal and fail immediately — after any remaining
    ``fallback`` is applied. A 429 honors the server's ``Retry-After`` header
    (seconds or HTTP-date) over the ``sleep_429`` base.
    """
    payload: dict = {"model": model, "messages": messages}
    if payload_extra:
        payload.update(payload_extra)
    headers = bearer_headers(api_key)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = await client.post(
                f"{base_url}/chat/completions", json=payload, headers=headers
            )
            if resp.status_code == 429 and attempt + 1 < attempts:
                delay = _retry_after(resp)
                if delay is None and sleep_429:
                    delay = _jitter(sleep_429 * (attempt + 1))
                if delay:
                    await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            # A 200 with no choices / no message is a malformed gateway response,
            # not a usable answer — treat it as a retryable failure instead of
            # silently returning an empty result that callers mistake for success.
            if not data.get("choices") or not data["choices"][0].get("message"):
                raise ValueError("model response has no choices/message")
            return data
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            fatal = (isinstance(exc, httpx.HTTPStatusError)
                     and not _retryable_status(exc.response.status_code))
            log.warning("[%s] chat attempt %d/%d failed%s: %s",
                        log_name, attempt + 1, attempts,
                        " (fatal, not retrying)" if fatal else "", exc)
            if fatal or attempt + 1 >= attempts:
                break
            if sleep_err:
                await asyncio.sleep(_jitter(sleep_err * (attempt + 1)))
    if fallback is not None:
        return {"choices": [{"message": {"content": fallback}}]}
    assert last_exc is not None
    raise last_exc


def first_content(data: dict) -> str:
    """Extract the assistant text from a :func:`chat_completion` response."""
    choices = data.get("choices") or [{}]
    return _content(choices[0].get("message"))


async def iter_sse_lines(resp) -> AsyncIterator[tuple[str | None, str]]:
    """Incremental SSE parser over a streaming httpx response.

    Yields ``(event, data)`` per SSE frame: ``event`` is the server's
    ``event:`` name when present (``None`` otherwise), ``data`` the joined
    ``data:`` payload. Frames are dispatched on blank lines per the SSE
    framing, so multi-``data:`` events and streamed bodies both work.
    """
    event: str | None = None
    data: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].strip())
    if data:
        yield event, "\n".join(data)


async def iter_chat_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    payload_extra: dict | None = None,
    attempts: int = 1,
    sleep_429: float = 0.0,
    sleep_err: float = 0.0,
    log_name: str = "llm",
) -> AsyncIterator[dict]:
    """POST ``{base_url}/chat/completions`` with ``stream: true``.

    Yields ``{"delta": text}`` per content chunk as it arrives, then one final
    ``{"result": data}`` where ``data`` mirrors :func:`chat_completion`'s
    return: an assembled API-shape dict whose ``choices[0].message`` carries
    the full content and tool_calls (fragments joined in order) and whose
    ``usage`` is whatever the gateway sent in-stream (empty when it sends
    none — request ``stream_options.include_usage`` via ``payload_extra``).

    Retry policy matches :func:`chat_completion` (fatal 4xx never retried, 429
    honoring ``Retry-After``), with one streaming-specific rule: once a delta
    has been yielded it cannot be unsent, so a failure after that point raises
    immediately instead of retrying.
    """
    payload: dict = {"model": model, "messages": messages, "stream": True}
    if payload_extra:
        payload.update(payload_extra)
    headers = bearer_headers(api_key)
    url = f"{base_url}/chat/completions"
    last_exc: Exception | None = None
    for attempt in range(attempts):
        delivered = False
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code == 429 and attempt + 1 < attempts:
                    await resp.aread()  # drain before retrying
                    delay = _retry_after(resp)
                    if delay is None and sleep_429:
                        delay = _jitter(sleep_429 * (attempt + 1))
                    if delay:
                        await asyncio.sleep(delay)
                    continue
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {body[:200]}",
                        request=resp.request, response=resp)
                content_parts: list[str] = []
                tool_frags: dict[int, dict] = {}
                usage: dict = {}
                finish_reason = None
                async for _event, data_txt in iter_sse_lines(resp):
                    if data_txt == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_txt)
                    except ValueError:
                        continue
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        content_parts.append(text)
                        delivered = True
                        yield {"delta": text}
                    for frag in delta.get("tool_calls") or []:
                        if not isinstance(frag, dict):
                            continue
                        idx = frag.get("index") or 0
                        slot = tool_frags.setdefault(
                            idx, {"id": None, "name": None, "args": []})
                        if frag.get("id"):
                            slot["id"] = frag["id"]
                        fn = frag.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"].append(fn["arguments"])
            tool_calls = [
                # A missing id stays None here: the runtime's
                # _normalize_assistant is the single synthesis point, so a
                # gateway that never sends ids gets collision-free ids
                # everywhere (messages, records, events) instead of
                # per-response "call_0"-style ids that collide across steps.
                {"id": slot["id"], "type": "function",
                 "function": {"name": slot["name"] or "",
                              "arguments": "".join(slot["args"])}}
                for _, slot in sorted(tool_frags.items())
            ]
            message: dict = {"content": "".join(content_parts)}
            if tool_calls:
                message["tool_calls"] = tool_calls
            yield {"result": {"choices": [{"message": message,
                                           "finish_reason": finish_reason}],
                              "usage": usage}}
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            fatal = (isinstance(exc, httpx.HTTPStatusError)
                     and not _retryable_status(exc.response.status_code))
            log.warning("[%s] stream attempt %d/%d failed%s: %s",
                        log_name, attempt + 1, attempts,
                        " (fatal, not retrying)" if fatal else "", exc)
            if delivered or fatal:
                raise
            if attempt + 1 < attempts and sleep_err:
                await asyncio.sleep(_jitter(sleep_err * (attempt + 1)))
    assert last_exc is not None
    raise last_exc
