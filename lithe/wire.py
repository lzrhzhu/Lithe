"""Protocol-neutral HTTP wire plumbing for model endpoints.

Everything here is shared by *every* wire protocol the kernel speaks —
retry policy (which statuses are retryable, ``Retry-After`` parsing,
equal-jitter backoff), header assembly, SSE frame parsing, and the 400
diagnosis annotation for host-supplied ``extra_body`` fields. The chat
client (:mod:`lithe.transports.openai_chat`) and every transport
(:mod:`lithe.transports`) build on this module, so the transport layer
never depends on a legacy client — the chat client and the transports are
siblings over ``wire``.

This module is part of the storage-free agent kernel — it depends only
on the standard library and httpx, never on a host application.
"""
from __future__ import annotations

import random
import time
from email.utils import parsedate_to_datetime
from collections.abc import AsyncIterator

import httpx

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


def bearer_headers(api_key: str, extra: dict | None = None) -> dict:
    """Standard ``Authorization: Bearer`` + JSON content-type header.

    ``extra`` (e.g. ``LLMConfig.default_headers``) is merged on top, so a
    host can add gateway-specific headers (OpenRouter's ``HTTP-Referer`` /
    ``X-Title``, ``OpenAI-Organization``, ...) or override the defaults.
    """
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    if extra:
        headers.update(extra)
    return headers


def extra_body_hint(extra_body: dict | None) -> str | None:
    """Diagnosis hint for a 400 that may stem from host-supplied body fields.

    Strict gateways (OpenAI's API among them) reject unknown request
    arguments with a 400; lenient ones silently ignore them. When a request
    carried ``LLMConfig.extra_body`` fields and came back 400, point at them
    — the alternative is a generic "bad request" that hides the one knob the
    host actually controls. ``None`` when there is nothing to point at.
    """
    if not extra_body:
        return None
    keys = ", ".join(sorted(map(str, extra_body)))
    return (f"请求携带了 LLMConfig.extra_body 字段（{keys}）。"
            f"严格的网关会拒绝不认识的参数（OpenAI 官方端点即如此）；"
            f"请确认该端点支持这些字段，或移除后再试")


def _annotate_extra_body(exc: Exception, extra_body: dict | None) -> Exception:
    """Attach :func:`extra_body_hint` to a 400 before it propagates.

    Returns the exception unchanged for any other status / empty extra_body.
    The hint rides both the message (logs, ``log.warning``) and a
    ``lithe_hint`` attribute (the runtime appends it to the run's error
    event), and the response body's first 200 chars are folded into the
    message when cheaply available — OpenAI-style 400s name the offending
    argument there.
    """
    hint = extra_body_hint(extra_body)
    if hint is None or not isinstance(exc, httpx.HTTPStatusError) \
            or exc.response.status_code != 400:
        return exc
    try:
        body = (exc.response.text or "").strip()[:200]
    except Exception:  # noqa: BLE001 — body may be unreadable; hint suffices
        body = ""
    message = f"{exc}"
    if body and body not in message:
        message = f"{message}：{body}"
    annotated = httpx.HTTPStatusError(
        f"{message}；{hint}", request=exc.request, response=exc.response)
    # dynamic attribute: the runtime reads it back via getattr (loop.py)
    annotated.lithe_hint = hint  # type: ignore[attr-defined]
    return annotated


def _jitter(base: float) -> float:
    """Equal-jitter backoff: a sleep of ``base`` scaled by a random factor in
    ``[0.5, 1.5)``, so concurrent callers (e.g. many translation chunks hitting
    a 429 together) don't all retry in lockstep and worsen the rate limit."""
    return base * random.uniform(0.5, 1.5)


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
