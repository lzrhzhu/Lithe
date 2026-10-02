"""LLM transports: protocol-agnostic model calls.

The runtime drives the model via an :class:`LLMTransport`, which sends one
request and returns a *unified* shape ``{content, tool_calls, usage}`` — so the
runtime never branches on chat-completions vs responses wire formats.

Two built-in transports:

- :class:`ChatCompletionsTransport` — OpenAI ``/chat/completions`` (default,
  backward compatible; wraps :func:`lithe.llm.chat_completion`).
- :class:`ResponsesTransport` — OpenAI ``Responses API`` (``/responses``), the
  2025 standard that supersedes chat completions. It translates the runtime's
  chat-shaped ``messages`` (system / user / assistant-with-tool_calls / tool
  result) into Responses ``input`` items (``instructions`` + role items +
  ``function_call`` + ``function_call_output``), and parses ``output[]`` back.

A host picks one via ``LLMConfig(transport="chat" | "responses")`` or passes a
custom :class:`LLMTransport`. ``lithe`` stays storage-/host-agnostic — these
are OpenAI standard protocols, not host specifics.

ReAct-loop note: unlike a single-shot caller, the runtime accumulates tool
results across steps, so :meth:`ResponsesTransport.complete` maps each chat
``{role:"tool", tool_call_id, content}`` to a Responses
``function_call_output`` item. Omitting that breaks multi-turn responses.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol
from collections.abc import AsyncIterator

import httpx

from .llm import (
    _annotate_extra_body, _jitter, _retry_after, _retryable_status,
    bearer_headers, chat_completion, iter_chat_completion, iter_sse_lines,
)

log = logging.getLogger("lithe.transports")

# Unified result every transport returns.
TransportResult = dict[str, Any]


class LLMTransport(Protocol):
    """Send one model call; return ``{content, tool_calls, usage}``.

    ``tool_calls`` mirrors the chat-completions shape
    (``{id, type:"function", function:{name, arguments(str)}}``) regardless of
    the underlying protocol, so the runtime's dispatch is protocol-agnostic.

    ``reasoning`` (optional, Responses transport) is the raw list of reasoning
    output items — the runtime attaches them to the assistant message so the
    *next* call of the same tool loop can pass them back (reasoning continuity
    for reasoning models). ``include_reasoning`` tells the transport whether
    the runtime wants reasoning replayed at all; ``reasoning_scope`` selects
    which turns replay it (``"loop"``: active tool loop only, ``"conversation"``:
    every turn — see :func:`_messages_to_input`).
    """

    async def complete(
        self, client: Any, *, base_url: str, api_key: str, model: str,
        messages: list[dict], tools: list[dict] | None = None,
        tool_choice: Any = None, max_tokens: int | None = None,
        temperature: float | None = None, attempts: int = 1,
        sleep_429: float = 0.0, sleep_err: float = 0.0,
        include_reasoning: bool = True, reasoning_scope: str = "loop",
        extra_body: dict | None = None, extra_headers: dict | None = None,
    ) -> TransportResult: ...


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def _flatten_text(content: Any) -> str:
    """Coerce a message content (str / list of {type:text/text} blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"):
                out.append(p.get("text", "") or "")
            elif isinstance(p, str):
                out.append(p)
        return "".join(out)
    return str(content)


def _content_blocks(content: Any, *, role: str) -> list[dict] | None:
    """Map chat content blocks to Responses content blocks, or return None.

    ``None`` means "plain text is enough" (a string, an empty list, or a
    list of only-string parts): the caller then uses its text-only fast
    path, which keeps the 99%-text case byte-identical to before. Otherwise:

    - text blocks (``text`` / ``input_text`` / ``output_text``) map to the
      role-appropriate Responses text type;
    - ``image_url`` blocks map to ``input_image`` (the URL — remote
      ``https://`` or inline ``data:`` base64 — passes through verbatim);
    - anything else **raises**: silently dropping a block would have the
      model answer about content it never saw, which is the worst failure
      mode (plausible but wrong). A loud error lets the host switch to the
      chat transport or drop the block deliberately.
    """
    if not isinstance(content, list) or not content:
        return None
    text_type = "output_text" if role == "assistant" else "input_text"
    out: list[dict] = []
    for part in content:
        if isinstance(part, str):
            if part:
                out.append({"type": text_type, "text": part})
            continue
        if not isinstance(part, dict):
            raise ValueError(
                f"cannot map content block {part!r} onto the Responses "
                f"input format (supported: text, image_url)")
        ptype = part.get("type")
        if ptype in ("text", "input_text", "output_text"):
            out.append({"type": text_type, "text": part.get("text") or ""})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url") \
                if isinstance(part.get("image_url"), dict) else None
            if not isinstance(url, str) or not url:
                raise ValueError("image_url block without a usable url")
            out.append({"type": "input_image", "image_url": url})
        else:
            raise ValueError(
                f"cannot map content block type {ptype!r} onto the Responses "
                f"input format (supported: text, image_url)")
    return out or None


def _merge_extra(extra_body: dict | None, internal: dict) -> dict:
    """Merge host-supplied body fields under the internally computed ones.

    ``extra_body`` (e.g. ``LLMConfig.extra_body``) carries vendor extensions
    the kernel does not model (``top_p``, ``seed``, ``enable_thinking``,
    ``reasoning`` effort, ...). Internal keys (``tools``, ``tool_choice``,
    ``max_tokens`` / ``max_output_tokens``, ...) win on collision: the loop
    mechanics must not be silently broken by a payload field, and those knobs
    have first-class ``LLMConfig`` members already.
    """
    if not extra_body:
        return internal
    merged = dict(extra_body)
    merged.update(internal)
    return merged


def norm_usage(usage: dict | None) -> dict:
    """Normalize vendor usage shapes into one contract.

    Chat-completions reports ``prompt_tokens``/``completion_tokens``/
    ``total_tokens``; the Responses API reports ``input_tokens``/
    ``output_tokens``. The unified shape always carries ``prompt_tokens``,
    ``completion_tokens`` and ``total_tokens`` (computed when the vendor omits
    it). Cached input tokens are lifted to a top-level ``cached_tokens`` from
    either vendor detail shape (``prompt_tokens_details.cached_tokens`` /
    ``input_tokens_details.cached_tokens``) — they are a subset of
    ``prompt_tokens``, not additive. Any extra vendor fields
    (``cost_breakdown``, ...) pass through.
    """
    u = dict(usage or {})
    p = u.get("prompt_tokens", u.get("input_tokens"))
    c = u.get("completion_tokens", u.get("output_tokens"))
    if p is not None:
        u["prompt_tokens"] = int(p)
    if c is not None:
        u["completion_tokens"] = int(c)
    t = u.get("total_tokens")
    if t is None and p is not None and c is not None:
        t = int(p) + int(c)
    if t is not None:
        u["total_tokens"] = int(t)
    for detail_key in ("prompt_tokens_details", "input_tokens_details"):
        details = u.get(detail_key)
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            u["cached_tokens"] = int(details["cached_tokens"])
            break
    out_details = u.get("output_tokens_details")
    if isinstance(out_details, dict) and out_details.get("reasoning_tokens") is not None:
        u["reasoning_tokens"] = int(out_details["reasoning_tokens"])
    return u


# --------------------------------------------------------------------------- #
# ChatCompletionsTransport (default, wraps chat_completion)
# --------------------------------------------------------------------------- #

class ChatCompletionsTransport:
    """``/chat/completions`` transport — the pre-0.2 behavior, unchanged.

    Also streams (:meth:`complete_stream`) when the runtime asks for it:
    server-sent-event chunks are parsed incrementally so the runtime can emit
    ``assistant_delta`` events while the model is still generating.
    """

    def __init__(self) -> None:
        # Remembered per process: cleared after a gateway 4xx that looks like a
        # rejection of `stream_options`, so later streamed calls omit it.
        self._include_usage = True

    def _payload_extra(self, tools, tool_choice, max_tokens, temperature) -> dict:
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
        return extra

    @staticmethod
    def _to_result(data: dict) -> TransportResult:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return {
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "usage": norm_usage(data.get("usage")),
            # Surfaced so the runtime can tell a truncated generation
            # (finish_reason == "length") from a complete one instead of
            # diagnosing half-written tool_call JSON as "invalid JSON".
            "finish_reason": choice.get("finish_reason"),
        }

    async def complete(
        self, client, *, base_url, api_key, model, messages,
        tools=None, tool_choice=None, max_tokens=None, temperature=None,
        attempts=1, sleep_429=0.0, sleep_err=0.0, include_reasoning=True,
        reasoning_scope="loop", extra_body=None, extra_headers=None,
    ) -> TransportResult:
        # include_reasoning / reasoning_scope are no-ops here: chat-completions
        # reasoning (DeepSeek-style reasoning_content / <think>) is per-turn
        # state that must NOT be replayed — the responses-protocol reasoning
        # items are the only pass-back flavor.
        payload_extra = _merge_extra(
            extra_body,
            self._payload_extra(tools, tool_choice, max_tokens, temperature))
        data = await chat_completion(
            client, base_url=base_url, api_key=api_key, model=model,
            messages=messages, payload_extra=payload_extra or None,
            extra_headers=extra_headers, extra_body=extra_body,
            attempts=attempts, sleep_429=sleep_429, sleep_err=sleep_err,
            log_name="assistant",
        )
        return self._to_result(data)

    async def complete_stream(
        self, client, *, base_url, api_key, model, messages,
        tools=None, tool_choice=None, max_tokens=None, temperature=None,
        attempts=1, sleep_429=0.0, sleep_err=0.0, include_reasoning=True,
        reasoning_scope="loop", extra_body=None, extra_headers=None,
    ) -> AsyncIterator[dict]:
        """Streamed variant of :meth:`complete`.

        Yields ``{"delta": text}`` parts as content chunks arrive, then one
        ``{"result": TransportResult}``. ``stream_options.include_usage`` is
        requested so the final chunk carries token usage; gateways that reject
        that field get one retry without it (remembered for the process).
        """
        extra = _merge_extra(
            extra_body,
            self._payload_extra(tools, tool_choice, max_tokens, temperature))
        if self._include_usage:
            extra = {**extra, "stream_options": {"include_usage": True}}
        try:
            async for part in iter_chat_completion(
                    client, base_url=base_url, api_key=api_key, model=model,
                    messages=messages, payload_extra=extra or None,
                    extra_headers=extra_headers, extra_body=extra_body,
                    attempts=attempts, sleep_429=sleep_429, sleep_err=sleep_err,
                    log_name="assistant-stream"):
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
            log.warning("[assistant-stream] retrying without stream_options "
                        "after HTTP %d", code)
        extra.pop("stream_options", None)
        async for part in iter_chat_completion(
                client, base_url=base_url, api_key=api_key, model=model,
                messages=messages, payload_extra=extra or None,
                extra_headers=extra_headers, extra_body=extra_body,
                attempts=1,  # one replay, not a fresh retry budget
                sleep_429=sleep_429, sleep_err=sleep_err,
                log_name="assistant-stream"):
            if "result" in part:
                yield {"result": self._to_result(part["result"])}
            else:
                yield part


# --------------------------------------------------------------------------- #
# ResponsesTransport (OpenAI Responses API)
# --------------------------------------------------------------------------- #

def _messages_to_input(messages: list[dict], *,
                       include_reasoning: bool = True,
                       reasoning_scope: str = "loop") -> tuple[str | None, list[dict]]:
    """Translate chat ``messages`` to Responses ``(instructions, input[])``.

    - ``system`` → top-level ``instructions`` (aggregated)
    - ``user`` / ``assistant`` → role items with ``input_text`` / ``output_text``
    - ``assistant`` ``tool_calls`` → ``function_call`` items
    - ``{role:"tool"}`` results → ``function_call_output`` items
    - ``assistant`` ``reasoning`` → the stored reasoning items, re-emitted
      verbatim ahead of that turn's role/function_call items.

    ``reasoning_scope`` governs which assistant turns replay their reasoning:

    - ``"loop"`` (default): only turns after the last user message — the
      active tool loop. Reasoning from previous turns (and thus from a
      previous model, after a host-side model switch) is dropped once a new
      user message arrives, per the vendor's cost guidance.
    - ``"conversation"``: every assistant turn that carries reasoning items
      replays them — cross-turn reasoning continuity for hosts that prefer
      quality over token cost. The kernel still never compresses reasoning;
      a host decides when (or whether) to compact. A gateway 400 that clears
      after dropping reasoning input degrades gracefully (see
      :class:`ResponsesTransport`).

    The re-emitted items are shallow-copied but otherwise untouched:
    ``encrypted_content`` is opaque to this kernel and must round-trip
    exactly (id included) for the gateway to reattach the model's reasoning.
    """
    if reasoning_scope not in ("loop", "conversation"):
        raise ValueError(f"unknown reasoning_scope: {reasoning_scope!r} "
                         "(use 'loop' or 'conversation')")
    instruction_parts: list[str] = []
    input_items: list[dict] = []
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_idx = i
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            text = _flatten_text(content)
            if text:
                instruction_parts.append(text)
            continue

        if role == "tool":
            # tool result → function_call_output (call_id links to the call)
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id"),
                "output": content if isinstance(content, str) else _flatten_text(content),
            })
            continue

        # reasoning first: in a raw Responses output the reasoning item
        # precedes the message/function_call it produced — replay keeps that
        # order so the gateway pairs chain and call correctly.
        if (include_reasoning and role == "assistant"
                and (reasoning_scope == "conversation" or idx > last_user_idx)):
            for r in msg.get("reasoning") or []:
                if isinstance(r, dict) and r.get("type") == "reasoning":
                    input_items.append(dict(r))

        # user / assistant (and any other role)
        blocks = _content_blocks(content, role=role or "user")
        if blocks is not None:
            # Multimodal content (text + image blocks) maps block-by-block;
            # unmappable blocks raise rather than degrade silently.
            input_items.append({"role": role or "user", "content": blocks})
        else:
            text = _flatten_text(content)
            ctype = "output_text" if role == "assistant" else "input_text"
            input_items.append({
                "role": role or "user",
                "content": [{"type": ctype, "text": text or ""}],
            })
        # assistant tool_calls → function_call items
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            input_items.append({
                "type": "function_call",
                "call_id": tc.get("id") or tc.get("call_id"),
                "name": fn.get("name"),
                "arguments": fn.get("arguments", ""),
            })

    instructions = "\n\n".join(p for p in instruction_parts if p).strip() or None
    return instructions, input_items


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Flatten chat tools ``{type:function, function:{...}}`` → responses tools."""
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            out.append({
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
                "strict": False,
            })
        else:
            out.append(t)  # pass built-in/unknown tools through
    return out


def _convert_tool_choice(choice: Any) -> Any:
    """Map chat ``tool_choice`` to responses form. ``None`` → omit."""
    if choice is None:
        return None
    if isinstance(choice, str):
        return choice  # "auto" / "required" / "none"
    if isinstance(choice, dict):
        if choice.get("type") == "function" and isinstance(choice.get("function"), dict):
            return {"type": "function", "name": choice["function"].get("name")}
        return choice
    return choice


def _parse_output(data: dict) -> TransportResult:
    """Parse a Responses body into ``{content, tool_calls, reasoning, usage}``.

    ``reasoning`` collects the raw reasoning output items (id + encrypted
    content + summary). They are captured regardless of any replay setting:
    the runtime decides whether to attach them; keeping capture unconditional
    means a host that only wants the human-readable summary for display still
    gets it.
    """
    content_parts: list[str] = []
    tool_calls: list[dict] = []
    reasoning: list[dict] = []
    output = data.get("output")
    if not isinstance(output, list):
        output = []
    for item in output:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "reasoning":
            reasoning.append(item)
        elif itype == "message":
            # Some OpenAI-compatible gateways return message content as a
            # plain string instead of the spec's block list — flatten either.
            content_parts.append(_flatten_text(item.get("content")))
        elif itype == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id"),
                "type": "function",
                "function": {
                    "name": item.get("name"),
                    "arguments": item.get("arguments", ""),
                },
            })
    # Truncation signal: the Responses API reports an incomplete response via
    # status + incomplete_details rather than a finish_reason field; map the
    # token-cap case onto the chat-completions vocabulary so the runtime's
    # handling is protocol-agnostic.
    incomplete = data.get("incomplete_details") or {}
    finish_reason = ("length"
                     if data.get("status") == "incomplete"
                      and incomplete.get("reason") == "max_output_tokens"
                      else None)
    return {
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "reasoning": reasoning,
        "usage": norm_usage(data.get("usage")),
        "finish_reason": finish_reason,
    }


async def _post_responses(
    client, base_url: str, api_key: str, payload: dict, *,
    attempts: int, sleep_429: float, sleep_err: float,
    extra_headers: dict | None = None, extra_body: dict | None = None,
) -> dict:
    """POST to the Responses endpoint (``base_url`` is the full ``/responses`` URL)
    with the same retry policy as :func:`chat_completion`."""
    headers = bearer_headers(api_key, extra_headers)
    url = base_url
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 429 and attempt + 1 < attempts:
                # Same policy as the chat path: retry whenever attempts
                # remain, honoring the server's Retry-After (seconds or
                # HTTP-date) over the sleep_429 base, and immediately when
                # neither is available.
                delay = _retry_after(resp)
                if delay is None and sleep_429:
                    delay = _jitter(sleep_429 * (attempt + 1))
                if delay:
                    await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data.get("output"), list):
                raise ValueError("responses body has no 'output' list")
            return data
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            fatal = (isinstance(exc, httpx.HTTPStatusError)
                     and not _retryable_status(exc.response.status_code))
            log.warning("[responses] attempt %d/%d failed%s: %s",
                        attempt + 1, attempts,
                        " (fatal, not retrying)" if fatal else "", exc)
            if fatal or attempt + 1 >= attempts:
                break
            if sleep_err:
                await asyncio.sleep(_jitter(sleep_err * (attempt + 1)))
    assert last_exc is not None
    raise _annotate_extra_body(last_exc, extra_body)


class ResponsesTransport:
    """OpenAI Responses API transport.

    ``base_url`` is the **full** responses endpoint (e.g.
    ``…/api/v1/responses``), posted directly — unlike chat-completions, which
    appends ``/chat/completions`` to a root URL.

    Reasoning models return their chain-of-thought as ``reasoning`` output
    items (encrypted; only a human-readable ``summary`` is inspectable). For
    multi-step tool loops the items must be passed back on the next call in
    the same loop, so this transport (a) requests them via
    ``include: ["reasoning.encrypted_content"]`` and (b) re-emits stored
    reasoning items from ``messages`` into ``input`` ahead of that turn's
    function_call items (see :func:`_messages_to_input` and its
    ``reasoning_scope``).

    Degradation cascade on a 400 the gateway rejects the payload for: first
    the ``include`` parameter is dropped (remembered per transport instance —
    one gateway's rejection must not flip behavior for other hosts/endpoints
    in the same process), then — if the input still carried reasoning items —
    reasoning replay is dropped for that one call. Both fall back on
    availability over quality; a genuinely malformed payload fails
    identically on the final attempt.
    """

    def __init__(self) -> None:
        # Remembered per instance (a runtime owns one transport for its
        # lifetime): cleared after a gateway 400 that looks like a rejection
        # of the `include` parameter, so later calls from THIS transport omit
        # it. Instance-level mirrors ChatCompletionsTransport._include_usage;
        # the previous class-level flag leaked one gateway's quirk to every
        # other endpoint in the process.
        self._include_reasoning = True

    def _payload(self, messages, tools, tool_choice, max_tokens, temperature,
                 *, stream: bool, include_reasoning: bool,
                 reasoning_scope: str) -> dict:
        instructions, input_items = _messages_to_input(
            messages, include_reasoning=include_reasoning,
            reasoning_scope=reasoning_scope)
        payload: dict[str, Any] = {"model": "", "input": input_items}
        if stream:
            payload["stream"] = True
        if temperature is not None:
            payload["temperature"] = temperature
        if instructions:
            payload["instructions"] = instructions
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if tools:
            payload["tools"] = _convert_tools(tools)
            tc = _convert_tool_choice(tool_choice)
            if tc is not None:
                payload["tool_choice"] = tc
        return payload

    @staticmethod
    def _has_reasoning(payload: dict) -> bool:
        return any(isinstance(i, dict) and i.get("type") == "reasoning"
                   for i in payload.get("input") or [])

    async def _complete_with_fallback(
        self, client, *, base_url, api_key, model, messages, tools,
        tool_choice, max_tokens, temperature, attempts, sleep_429,
        sleep_err, include_reasoning, reasoning_scope,
        extra_body=None, extra_headers=None,
    ):
        """Non-stream call with the 400-degradation cascade (see class doc)."""
        want_reasoning = include_reasoning
        for _pass in range(3):
            payload = self._payload(messages, tools, tool_choice, max_tokens,
                                    temperature, stream=False,
                                    include_reasoning=want_reasoning,
                                    reasoning_scope=reasoning_scope)
            payload = _merge_extra(extra_body, payload)
            payload["model"] = model
            if include_reasoning and self._include_reasoning:
                payload["include"] = ["reasoning.encrypted_content"]
            try:
                return await _post_responses(
                    client, base_url, api_key, payload,
                    attempts=attempts, sleep_429=sleep_429, sleep_err=sleep_err,
                    extra_headers=extra_headers, extra_body=extra_body)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 400:
                    raise
                if "include" in payload and self._include_reasoning:
                    # plausibly an `include` field rejection: flip this
                    # instance's flag and retry without it.
                    self._include_reasoning = False
                    log.warning("[responses] retrying without reasoning include "
                                "after HTTP 400")
                    continue
                if want_reasoning and self._has_reasoning(payload):
                    # gateway rejects reasoning *input* items (e.g. items
                    # minted by another model after a host-side switch):
                    # drop reasoning replay for this call — availability over
                    # continuity.
                    want_reasoning = False
                    log.warning("[responses] retrying without reasoning input "
                                "items after HTTP 400")
                    continue
                raise
        raise RuntimeError("unreachable: 400-degradation cascade exhausted")

    async def complete(
        self, client, *, base_url, api_key, model, messages,
        tools=None, tool_choice=None, max_tokens=None, temperature=None,
        attempts=1, sleep_429=0.0, sleep_err=0.0, include_reasoning=True,
        reasoning_scope="loop", extra_body=None, extra_headers=None,
    ) -> TransportResult:
        data = await self._complete_with_fallback(
            client, base_url=base_url, api_key=api_key, model=model,
            messages=messages, tools=tools, tool_choice=tool_choice,
            max_tokens=max_tokens, temperature=temperature,
            attempts=attempts, sleep_429=sleep_429, sleep_err=sleep_err,
            include_reasoning=include_reasoning,
            reasoning_scope=reasoning_scope,
            extra_body=extra_body, extra_headers=extra_headers)
        return _parse_output(data)

    async def _stream_attempts(
        self, client, base_url: str, api_key: str, payload: dict, *,
        attempts: int, sleep_429: float, sleep_err: float,
        extra_headers: dict | None = None, extra_body: dict | None = None,
    ) -> AsyncIterator[dict]:
        """One streamed attempt cycle over an established payload (see
        :meth:`complete_stream`)."""
        headers = bearer_headers(api_key, extra_headers)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            delivered = False
            try:
                async with client.stream("POST", base_url, json=payload,
                                          headers=headers) as resp:
                    if resp.status_code == 429 and attempt + 1 < attempts:
                        await resp.aread()
                        # Same Retry-After-aware policy as the chat stream
                        # path (_post_responses): honor the header over the
                        # sleep_429 base, retry immediately without either.
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
                    items: list[dict] = []
                    final_response: dict | None = None
                    async for event, data_txt in iter_sse_lines(resp):
                        try:
                            data = json.loads(data_txt)
                        except ValueError:
                            continue
                        if not isinstance(data, dict):
                            continue
                        etype = event or data.get("type")
                        if etype == "response.output_text.delta":
                            text = data.get("delta")
                            if text:
                                delivered = True
                                yield {"delta": text}
                        elif etype == "response.output_item.done":
                            if isinstance(data.get("item"), dict):
                                items.append(data["item"])
                        elif etype == "response.completed":
                            final_response = data.get("response") or {}
                        elif etype in ("response.failed", "response.error",
                                       "error"):
                            err = data.get("error") or {}
                            why = err.get("message") or data_txt[:200]
                            raise RuntimeError(f"responses 流错误：{why}")
                response = final_response if final_response is not None \
                    else {"output": items, "usage": {}}
                yield {"result": _parse_output(response)}
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                fatal = (isinstance(exc, httpx.HTTPStatusError)
                         and not _retryable_status(exc.response.status_code))
                log.warning("[responses-stream] attempt %d/%d failed%s: %s",
                            attempt + 1, attempts,
                            " (fatal, not retrying)" if fatal else "", exc)
                if delivered or fatal:
                    # Annotated when a 400 may stem from extra_body fields
                    # (bare `raise` would skip the final annotation).
                    raise _annotate_extra_body(exc, extra_body) from exc
                if attempt + 1 >= attempts:
                    break
                if sleep_err:
                    await asyncio.sleep(_jitter(sleep_err * (attempt + 1)))
        assert last_exc is not None
        raise _annotate_extra_body(last_exc, extra_body)

    async def complete_stream(
        self, client, *, base_url, api_key, model, messages,
        tools=None, tool_choice=None, max_tokens=None, temperature=None,
        attempts=1, sleep_429=0.0, sleep_err=0.0, include_reasoning=True,
        reasoning_scope="loop", extra_body=None, extra_headers=None,
    ) -> AsyncIterator[dict]:
        """Streamed variant of :meth:`complete` over the Responses API.

        Yields ``{"delta": text}`` for ``response.output_text.delta`` events,
        then one ``{"result": TransportResult}`` assembled from the terminal
        ``response.completed`` event (falling back to accumulated
        ``output_item.done`` items if the gateway ends the stream without one).
        Deltas stream straight through; the 400-degradation cascade (drop
        ``include``, then drop reasoning input) only runs while nothing has
        been delivered — a failure after the first delta raises instead
        (deltas cannot be unsent).
        """
        want_reasoning = include_reasoning
        for _pass in range(3):
            payload = self._payload(messages, tools, tool_choice, max_tokens,
                                    temperature, stream=True,
                                    include_reasoning=want_reasoning,
                                    reasoning_scope=reasoning_scope)
            payload = _merge_extra(extra_body, payload)
            payload["model"] = model
            if include_reasoning and self._include_reasoning:
                payload["include"] = ["reasoning.encrypted_content"]
            delivered = False
            try:
                async for part in self._stream_attempts(
                        client, base_url, api_key, payload,
                        attempts=attempts, sleep_429=sleep_429,
                        sleep_err=sleep_err, extra_headers=extra_headers,
                        extra_body=extra_body):
                    if "delta" in part:
                        delivered = True
                    yield part
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 400 or delivered:
                    raise
                if "include" in payload and self._include_reasoning:
                    self._include_reasoning = False
                    log.warning("[responses-stream] retrying without reasoning "
                                "include after HTTP 400")
                    continue
                if want_reasoning and self._has_reasoning(payload):
                    want_reasoning = False
                    log.warning("[responses-stream] retrying without reasoning "
                                "input items after HTTP 400")
                    continue
                raise
        raise RuntimeError("unreachable: 400-degradation cascade exhausted")


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #

_BUILTIN = {"chat": ChatCompletionsTransport, "responses": ResponsesTransport}


def make_transport(spec: str | LLMTransport) -> LLMTransport:
    """Resolve a transport spec: ``"chat"`` / ``"responses"`` → built-in instance;
    an :class:`LLMTransport` instance → returned as-is."""
    if isinstance(spec, str):
        cls = _BUILTIN.get(spec)
        if cls is None:
            raise ValueError(f"unknown transport: {spec!r} (use 'chat' or 'responses')")
        return cls()  # type: ignore[return-value]
    return spec


__all__ = [
    "ChatCompletionsTransport", "LLMTransport", "ResponsesTransport",
    "TransportResult", "make_transport",
]
