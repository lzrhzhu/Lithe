"""The OpenAI chat-completions wire protocol: client + transport.

Two layers, one module, because they are one protocol:

- :func:`chat_completion` / :func:`iter_chat_completion` — the request
  boilerplate (headers, POST, ``raise_for_status``, JSON parse + shape
  validation, retry loop with jittered backoff; the streamed variant
  reassembles SSE chunks into a response-shaped dict). A 200 with no
  ``choices`` / ``message`` is treated as a retryable failure rather than
  silently returned as an empty "success". Retry/backoff, ``Retry-After``,
  headers and SSE framing come from :mod:`lithe.wire`.
- :class:`ChatCompletionsTransport` — maps those API-shaped responses onto
  the unified :class:`~lithe.transports.base.TransportResult`. What is
  chat-specific stays here: reasoning leakage (``reasoning_content`` /
  inline ``<think>`` blocks) is lifted into display-only reasoning items,
  and the outbound pass strips reasoning state the chat protocol must never
  replay (:func:`strip_think` is exported for hosts that render chat-model
  output themselves).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from collections.abc import AsyncIterator

import httpx

from lithe.transports.base import TransportResult, _merge_extra, norm_usage
from lithe.wire import (
    _annotate_extra_body, _jitter, _retry_after, _retryable_status,
    bearer_headers, iter_sse_lines,
)

log = logging.getLogger("lithe.transports")
# The chat *client* functions kept their historical "llm" logger so log
# filters / caplog assertions naming it stay stable across the move.
chat_log = logging.getLogger("llm")

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

# Inline reasoning leakage in agent-style chat content (extracted into a
# display-only reasoning item; the tags never reach the user's answer).
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def strip_think(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning leakage from agent-style models."""
    return _THINK_RE.sub("", text).strip()


def _content(message: dict | None) -> str:
    return (message or {}).get("content") or ""


async def chat_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    payload_extra: dict | None = None,
    extra_headers: dict | None = None,
    extra_body: dict | None = None,
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
    (seconds or HTTP-date) over the ``sleep_429`` base. ``extra_headers`` is
    merged over the bearer/JSON defaults on every attempt. ``extra_body`` is
    the host-supplied vendor-field dict (already merged into
    ``payload_extra`` by the transport) — it is only used to diagnose a 400
    with :func:`lithe.wire.extra_body_hint`.
    """
    payload: dict = {"model": model, "messages": messages}
    if payload_extra:
        payload.update(payload_extra)
    headers = bearer_headers(api_key, extra_headers)
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
            data: dict = resp.json()
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
            chat_log.warning("[%s] chat attempt %d/%d failed%s: %s",
                             log_name, attempt + 1, attempts,
                             " (fatal, not retrying)" if fatal else "", exc)
            if fatal or attempt + 1 >= attempts:
                break
            if sleep_err:
                await asyncio.sleep(_jitter(sleep_err * (attempt + 1)))
    if fallback is not None:
        return {"choices": [{"message": {"content": fallback}}]}
    assert last_exc is not None
    raise _annotate_extra_body(last_exc, extra_body)


def first_content(data: dict) -> str:
    """Extract the assistant text from a :func:`chat_completion` response."""
    choices = data.get("choices") or [{}]
    return _content(choices[0].get("message"))


async def iter_chat_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    payload_extra: dict | None = None,
    extra_headers: dict | None = None,
    extra_body: dict | None = None,
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
    headers = bearer_headers(api_key, extra_headers)
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
                reasoning_parts: list[str] = []
                saw_encrypted_reasoning = False
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
                    # Reasoning chunks (DeepSeek/GLM-style ``reasoning_content``
                    # deltas; OpenRouter streams the same as ``reasoning``):
                    # per-turn display state, accumulated onto the assembled
                    # message — the transport lifts it into a display-only
                    # reasoning item and never sends it back.
                    rtoken = delta.get("reasoning_content")
                    if not isinstance(rtoken, str) or not rtoken:
                        rtoken = delta.get("reasoning")
                    if isinstance(rtoken, str) and rtoken:
                        reasoning_parts.append(rtoken)
                        delivered = True
                    elif delta.get("reasoning_details"):
                        # GPT-6-style relays stream encrypted reasoning items
                        # — no readable text exists; flag their presence so
                        # the assembled message can carry an honest marker.
                        saw_encrypted_reasoning = True
                        delivered = True
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
            reasoning_text = "".join(reasoning_parts).strip()
            if not reasoning_text and saw_encrypted_reasoning:
                reasoning_text = "（模型思考中；本端点仅返回加密推理，无可显示内容）"
            if reasoning_text:
                message["reasoning_content"] = reasoning_text
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
            chat_log.warning("[%s] stream attempt %d/%d failed%s: %s",
                             log_name, attempt + 1, attempts,
                             " (fatal, not retrying)" if fatal else "", exc)
            if delivered or fatal:
                # Re-raise annotated when a 400 may stem from extra_body
                # fields (bare `raise` would skip the final annotation).
                raise _annotate_extra_body(exc, extra_body) from exc
            if attempt + 1 < attempts and sleep_err:
                await asyncio.sleep(_jitter(sleep_err * (attempt + 1)))
    assert last_exc is not None
    raise _annotate_extra_body(last_exc, extra_body)


def _chat_reasoning_items(msg: dict, content: str) -> list[dict]:
    """Display-only reasoning items from a chat-completions message.

    DeepSeek/GLM-style endpoints send the chain as a ``reasoning_content``
    string (streamed deltas are reassembled onto the message); agent-style
    models leak ``<think>…</think>`` into content itself. Both become
    ``{"type": "reasoning", "thinking": text}`` items — the shape
    :func:`lithe.memory.reasoning_summary_text` renders and the runtime
    turns into REASONING events. GPT-6-style relays send only encrypted
    ``reasoning_details``; nothing readable exists there, so an honest
    marker item reports that thinking happened (the chain itself is
    opaque by the vendor's design). Per chat protocol these items are
    display/record state only and never go back on the wire
    (see :func:`_chat_outbound`).
    """
    items: list[dict] = []
    raw = msg.get("reasoning_content")
    if isinstance(raw, str) and raw.strip():
        items.append({"type": "reasoning", "thinking": raw.strip()})
    leaked = [t.strip() for t in _THINK_BLOCK_RE.findall(content) if t.strip()]
    if leaked:
        items.append({"type": "reasoning", "thinking": "\n".join(leaked)})
    if not items:
        details = msg.get("reasoning_details")
        if isinstance(details, list) and details:
            items.append(
                {
                    "type": "reasoning",
                    "thinking": f"（模型思考 {len(details)} 段；"
                    "本端点仅返回加密推理，无可显示内容）",
                }
            )
    return items


def _chat_outbound(messages: list[dict]) -> list[dict]:
    """Strip reasoning state from assistant messages before the wire.

    Chat protocol never replays reasoning — DeepSeek's ``reasoning_content``
    is documented per-turn state, and a ``reasoning`` field is unknown to
    strict gateways (a 400). The runtime attaches display/record items to
    assistant messages (so Responses/Messages transports can replay theirs);
    the chat transport drops them per message here.
    """
    out = []
    for m in messages:
        if m.get("role") == "assistant" and (
            "reasoning" in m or "reasoning_content" in m
        ):
            m = {
                k: v
                for k, v in m.items()
                if k not in ("reasoning", "reasoning_content")
            }
        out.append(m)
    return out


class ChatCompletionsTransport:
    """``/chat/completions`` transport — the pre-0.2 wire behavior.

    Reasoning the response carries (DeepSeek/GLM-style ``reasoning_content``,
    inline ``<think>`` blocks) is lifted into display-only reasoning items
    for the runtime's REASONING events; chat protocol never sends them back
    (``_chat_outbound`` strips them from outgoing messages).

    Also streams (:meth:`complete_stream`) when the runtime asks for it:
    server-sent-event chunks are parsed incrementally so the runtime can emit
    ``assistant_delta`` events while the model is still generating.
    """

    def __init__(self) -> None:
        # Remembered per process: cleared after a gateway 4xx that looks like a
        # rejection of `stream_options`, so later streamed calls omit it.
        self._include_usage = True

    def _payload_extra(
        self, tools, tool_choice, max_tokens, temperature, reasoning_effort=None
    ) -> dict:
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
            # tool_choice without tools is rejected by OpenAI-compatible
            # endpoints (400) — only send it alongside a tools list.
            if tool_choice is not None:
                extra["tool_choice"] = tool_choice
        if max_tokens is not None:
            extra["max_tokens"] = max_tokens
        if temperature is not None:
            extra["temperature"] = temperature
        if reasoning_effort is not None:
            # OpenAI chat-completions spelling; OpenRouter-style gateways
            # that only speak `reasoning.effort` take it via extra_body
            # (and a rejection surfaces through the 400 diagnostics).
            extra["reasoning_effort"] = reasoning_effort
        return extra

    @staticmethod
    def _to_result(data: dict) -> TransportResult:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = _chat_reasoning_items(msg, content)
        if reasoning and _THINK_BLOCK_RE.search(content):
            # the leaked <think> blocks moved into the reasoning items —
            # the answer text itself stays clean
            content = strip_think(content)
        return {
            "content": content,
            "tool_calls": msg.get("tool_calls") or [],
            "usage": norm_usage(data.get("usage")),
            "reasoning": reasoning,
            # Surfaced so the runtime can tell a truncated generation
            # (finish_reason == "length") from a complete one instead of
            # diagnosing half-written tool_call JSON as "invalid JSON".
            "finish_reason": choice.get("finish_reason"),
        }

    async def complete(
        self,
        client,
        *,
        base_url,
        api_key,
        model,
        messages,
        tools=None,
        tool_choice=None,
        max_tokens=None,
        temperature=None,
        attempts=1,
        sleep_429=0.0,
        sleep_err=0.0,
        include_reasoning=True,
        reasoning_scope="loop",
        reasoning_effort=None,
        extra_body=None,
        extra_headers=None,
    ) -> TransportResult:
        # include_reasoning / reasoning_scope are no-ops here: chat-completions
        # reasoning (DeepSeek-style reasoning_content / <think>) is per-turn
        # display state that must NOT be replayed — the responses-protocol
        # reasoning items are the only pass-back flavor. The reasoning the
        # response *carries* is captured for display (see _to_result).
        payload_extra = _merge_extra(
            extra_body,
            self._payload_extra(
                tools, tool_choice, max_tokens, temperature, reasoning_effort
            ),
        )
        data = await chat_completion(
            client,
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=_chat_outbound(messages),
            payload_extra=payload_extra or None,
            extra_headers=extra_headers,
            extra_body=extra_body,
            attempts=attempts,
            sleep_429=sleep_429,
            sleep_err=sleep_err,
            log_name="assistant",
        )
        return self._to_result(data)

    async def complete_stream(
        self,
        client,
        *,
        base_url,
        api_key,
        model,
        messages,
        tools=None,
        tool_choice=None,
        max_tokens=None,
        temperature=None,
        attempts=1,
        sleep_429=0.0,
        sleep_err=0.0,
        include_reasoning=True,
        reasoning_scope="loop",
        reasoning_effort=None,
        extra_body=None,
        extra_headers=None,
    ) -> AsyncIterator[dict]:
        """Streamed variant of :meth:`complete`.

        Yields ``{"delta": text}`` parts as content chunks arrive, then one
        ``{"result": TransportResult}``. ``stream_options.include_usage`` is
        requested so the final chunk carries token usage; gateways that reject
        that field get one retry without it (remembered for the process).
        """
        extra = _merge_extra(
            extra_body,
            self._payload_extra(
                tools, tool_choice, max_tokens, temperature, reasoning_effort
            ),
        )
        if self._include_usage:
            extra = {**extra, "stream_options": {"include_usage": True}}
        try:
            async for part in iter_chat_completion(
                client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=_chat_outbound(messages),
                payload_extra=extra or None,
                extra_headers=extra_headers,
                extra_body=extra_body,
                attempts=attempts,
                sleep_429=sleep_429,
                sleep_err=sleep_err,
                log_name="assistant-stream",
            ):
                if "result" in part:
                    yield {"result": self._to_result(part["result"])}
                else:
                    yield part
            return
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            # Only a 400 can plausibly be a `stream_options` field rejection;
            # 401/403/404/422 have other causes and must not silently flip
            # process-wide state nor be replayed under a fresh attempt budget.
            if code != 400 or not self._include_usage:
                raise
            self._include_usage = False  # probably a stream_options rejection
            log.warning(
                "[assistant-stream] retrying without stream_options after HTTP %d", code
            )
        extra.pop("stream_options", None)
        async for part in iter_chat_completion(
            client,
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=_chat_outbound(messages),
            payload_extra=extra or None,
            extra_headers=extra_headers,
            extra_body=extra_body,
            attempts=1,  # one replay, not a fresh retry budget
            sleep_429=sleep_429,
            sleep_err=sleep_err,
            log_name="assistant-stream",
        ):
            if "result" in part:
                yield {"result": self._to_result(part["result"])}
            else:
                yield part
