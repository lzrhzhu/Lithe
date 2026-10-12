"""The Anthropic Messages API transport (``/messages``).

Chat ``messages`` become ``(system, turns)`` with ``tool_use`` /
``tool_result`` content blocks, and ``content[]`` blocks parse back into the
unified :class:`~lithe.transports.base.TransportResult`. Thinking blocks
replay within the tool loop like Responses reasoning items (signature
included — it must round-trip exactly for the model to reattach its chain).

Protocol-specific mechanics kept here and nowhere else: the ``x-api-key``
auth header, the required ``max_tokens`` (with a conservative default), the
effort→thinking-budget bands, prompt-cache ``cache_control`` breakpoints, and
the 400-degradation passes (drop thinking replay, then cache breakpoints) —
availability over continuity/economy, mirroring the Responses transport.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from collections.abc import AsyncIterator

import httpx

from lithe.transports.base import (
    TransportResult, _flatten_text, _merge_extra, _to_int, norm_usage,
)
from lithe.wire import (
    _annotate_extra_body, _jitter, _retry_after, _retryable_status,
    iter_sse_lines,
)

log = logging.getLogger("lithe.transports")

# The messages protocol requires max_tokens on every request; when the config
# leaves it unset this conservative default never 400s on any Claude model
# (older ones cap at 4096 output), and truncation surfaces as
# finish_reason="length" instead of a silent cap raise.
_MESSAGES_DEFAULT_MAX_TOKENS = 4096

# The messages protocol has no effort vocabulary — reasoning intensity is a
# token budget. These bands let LLMConfig.reasoning_effort (and a host's
# in-session switch on top of it) work uniformly across protocols; a host
# wanting exact budgets sets extra_body["thinking"] instead of the effort
# knob (the merge rules make the two exclusive — internal keys win per key).
_EFFORT_THINKING_BUDGETS = {"low": 2048, "medium": 8192, "high": 16384}

_ANTHROPIC_VERSION = "2023-06-01"

_STOP_REASON_MAP = {"max_tokens": "length", "tool_use": "tool_calls"}

# Prompt-cache marker (5-minute TTL, refreshed on every hit) — the right
# horizon for a tool loop whose next call is seconds away. Written prefixes
# bill at 1.25×, cache reads at 0.1×, so re-sending the whole conversation
# every step becomes an incremental cost instead of a full re-price.
_EPHEMERAL = {"type": "ephemeral"}


def anthropic_headers(api_key: str, extra: dict | None = None) -> dict:
    """Messages-protocol headers: ``x-api-key`` + version, ``extra`` on top.

    The protocol's own auth is the API-key header (no ``Authorization:
    Bearer``); ``extra`` (``LLMConfig.default_headers``) merges over the
    defaults so a host can add gateway headers.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _image_source(url: str) -> dict:
    """Map a chat ``image_url`` URL onto a Messages image ``source``.

    Inline ``data:<mediatype>;base64,<data>`` URLs (the OpenAI vision shape)
    become base64 sources; remote ``http(s)://`` URLs become url sources.
    Anything else raises — same discipline as the Responses mapping: a
    silently dropped block would have the model answer about an image it
    never saw.
    """
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        media_type = head[len("data:") :].split(";", 1)[0] or "image/png"
        return {"type": "base64", "media_type": media_type, "data": data}
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "url", "url": url}
    raise ValueError(
        f"cannot map image_url {url[:60]!r} onto the messages input "
        f"format (supported: data:/http(s):// URLs)"
    )


def _user_blocks(content: Any) -> list[dict]:
    """Chat user content → Messages content blocks (text / image).

    Empty text parts are dropped (the protocol rejects empty text blocks);
    a wholly empty message yields no blocks, skipping the turn.
    """
    if not isinstance(content, list) or not content:
        text = _flatten_text(content)
        return [{"type": "text", "text": text}] if text else []
    out: list[dict] = []
    for part in content:
        if isinstance(part, str):
            if part:
                out.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            raise ValueError(
                f"cannot map content block {part!r} onto the messages "
                f"input format (supported: text, image_url)"
            )
        ptype = part.get("type")
        if ptype in ("text", "input_text", "output_text"):
            text = part.get("text") or ""
            if text:
                out.append({"type": "text", "text": text})
        elif ptype == "image_url":
            iu = part.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else None
            if not isinstance(url, str) or not url:
                raise ValueError("image_url block without a usable url")
            out.append({"type": "image", "source": _image_source(url)})
        else:
            raise ValueError(
                f"cannot map content block type {ptype!r} onto the messages "
                f"input format (supported: text, image_url)"
            )
    return out


def _tool_input(args: Any) -> dict:
    """Chat tool-call ``arguments`` (a JSON string) → the protocol's dict.

    A malformed string degrades to ``{}`` — the runtime re-validates
    arguments at dispatch anyway, and a hard failure here would turn a
    replayable history into an API-rejected payload.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _messages_to_anthropic(
    messages: list[dict],
    *,
    include_reasoning: bool = True,
    reasoning_scope: str = "loop",
) -> tuple[str | None, list[dict]]:
    """Translate chat ``messages`` to Messages ``(system, turns)``.

    - ``system`` messages → one top-level ``system`` string (aggregated)
    - ``user`` / ``assistant`` → role turns with content blocks
    - ``assistant`` ``tool_calls`` → ``tool_use`` blocks (canonical order:
      thinking replay first, then text, then tool_use)
    - ``{role:"tool"}`` results → ``tool_result`` blocks batched into the
      user turn that immediately follows the assistant tool_use turn
    - ``assistant`` ``reasoning`` → thinking / redacted_thinking blocks,
      re-emitted verbatim (signature included — it must round-trip exactly
      for the model to reattach its chain), scoped like Responses reasoning
      items (``reasoning_scope``: ``"loop"`` = active tool loop only,
      ``"conversation"`` = every turn)

    Consecutive same-role chat turns merge into one turn (the protocol
    requires alternating user/assistant roles); empty text is dropped.
    """
    if reasoning_scope not in ("loop", "conversation"):
        raise ValueError(
            f"unknown reasoning_scope: {reasoning_scope!r} "
            "(use 'loop' or 'conversation')"
        )
    system_parts: list[str] = []
    turns: list[dict] = []
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_idx = i

    def _append(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": list(blocks)})

    pending_results: list[dict] = []

    def _flush_results() -> None:
        if pending_results:
            _append("user", pending_results)
            pending_results.clear()

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            text = _flatten_text(content)
            if text:
                system_parts.append(text)
            continue

        if role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": content
                    if isinstance(content, str)
                    else _flatten_text(content),
                }
            )
            continue

        # A non-tool message after tool results: flush them into their own
        # user turn first so they sit right after the assistant tool_use
        # turn, whatever comes next.
        _flush_results()

        if role == "assistant":
            blocks: list[dict] = []
            if include_reasoning and (
                reasoning_scope == "conversation" or idx > last_user_idx
            ):
                for r in msg.get("reasoning") or []:
                    if isinstance(r, dict) and r.get("type") in (
                        "thinking",
                        "redacted_thinking",
                    ):
                        blocks.append(dict(r))
            text = _flatten_text(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    fn = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or tc.get("call_id"),
                        "name": fn.get("name") or "",
                        "input": _tool_input(fn.get("arguments", "")),
                    }
                )
            _append("assistant", blocks)
        else:  # user (any other role maps onto user, like _messages_to_input)
            _append("user", _user_blocks(content))
    _flush_results()

    system = "\n\n".join(p for p in system_parts if p).strip() or None
    return system, turns


def _tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Flatten chat tools ``{type:function, function:{...}}`` → Messages tools."""
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            out.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object"},
                }
            )
        else:
            out.append(t)  # pass server/built-in tools through
    return out


def _tool_choice_to_anthropic(choice: Any) -> dict | None:
    """Map chat ``tool_choice`` onto the protocol's object form.

    ``"none"`` maps to ``None`` — the protocol cannot express "tools present
    but do not use them", so the caller drops the tools list entirely.
    """
    if choice is None or choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return None
    if isinstance(choice, dict):
        if choice.get("type") == "function" and isinstance(
            choice.get("function"), dict
        ):
            return {"type": "tool", "name": choice["function"].get("name")}
        return choice
    return {"type": "auto"}


def _blocks_to_result(
    blocks: list, usage: dict | None, stop_reason: str | None
) -> TransportResult:
    """Parse Messages ``content[]`` blocks into the unified shape.

    ``tool_use.input`` (a dict) is serialized back to the chat shape's
    JSON-string ``arguments``; streaming callers pre-join fragments into
    ``input_json`` instead. Thinking blocks are captured as ``reasoning``
    items regardless of any replay setting — the runtime decides.
    """
    content_parts: list[str] = []
    tool_calls: list[dict] = []
    reasoning: list[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            if block.get("text"):
                content_parts.append(block["text"])
        elif btype == "tool_use":
            raw = block.get("input_json")
            if raw is None:
                inp = block.get("input")
                raw = json.dumps(inp) if isinstance(inp, dict) else (inp or "{}")
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": raw or "{}",
                    },
                }
            )
        elif btype in ("thinking", "redacted_thinking"):
            reasoning.append(block)
    return {
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "reasoning": reasoning,
        "usage": norm_usage(usage),
        "finish_reason": _STOP_REASON_MAP.get(stop_reason or ""),
    }


def _parse_anthropic(data: dict) -> TransportResult:
    """Parse a Messages response body via :func:`_blocks_to_result`."""
    content = data.get("content")
    if not isinstance(content, list):
        content = []
    return _blocks_to_result(content, data.get("usage"), data.get("stop_reason"))


async def _post_messages(
    client,
    base_url: str,
    api_key: str,
    payload: dict,
    *,
    attempts: int,
    sleep_429: float,
    sleep_err: float,
    extra_headers: dict | None = None,
    extra_body: dict | None = None,
) -> dict:
    """POST to ``{base_url}/messages`` with the shared retry policy."""
    headers = anthropic_headers(api_key, extra_headers)
    url = f"{base_url}/messages"
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 429 and attempt + 1 < attempts:
                # Same policy as the chat / responses paths: honor the
                # server's Retry-After (seconds or HTTP-date) over the
                # sleep_429 base, immediately when neither is available.
                delay = _retry_after(resp)
                if delay is None and sleep_429:
                    delay = _jitter(sleep_429 * (attempt + 1))
                if delay:
                    await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            data: dict = resp.json()
            if not isinstance(data.get("content"), list):
                raise ValueError("messages body has no 'content' list")
            return data
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            fatal = isinstance(exc, httpx.HTTPStatusError) and not _retryable_status(
                exc.response.status_code
            )
            log.warning(
                "[messages] attempt %d/%d failed%s: %s",
                attempt + 1,
                attempts,
                " (fatal, not retrying)" if fatal else "",
                exc,
            )
            if fatal or attempt + 1 >= attempts:
                break
            if sleep_err:
                await asyncio.sleep(_jitter(sleep_err * (attempt + 1)))
    assert last_exc is not None
    raise _annotate_extra_body(last_exc, extra_body)


class MessagesTransport:
    """Anthropic Messages API transport.

    ``base_url`` is the API root (e.g. ``…/v1``); ``/messages`` is appended,
    matching the chat transport's root-URL convention. Auth is the
    protocol's own ``x-api-key`` header (see :func:`anthropic_headers`) —
    no bearer token is sent.

    ``max_tokens`` is required by the protocol and defaulted
    (:data:`_MESSAGES_DEFAULT_MAX_TOKENS`) when the config leaves it unset;
    a host's ``LLMConfig.max_tokens`` wins. ``reasoning_effort`` maps onto
    the protocol's thinking budget via conservative bands
    (:data:`_EFFORT_THINKING_BUDGETS`); ``"minimal"`` and unknown levels
    enable nothing, and a host wanting an exact budget sets
    ``extra_body["thinking"]`` instead of the effort knob (the merge makes
    the two exclusive — internal keys win per key). Protocol shims after
    the merge (:meth:`_finalize`): with thinking enabled, ``temperature``
    is dropped (the protocol pins it) and ``max_tokens`` is raised to
    budget + 4096 headroom when it would not exceed the budget.

    Degradation on a 400: if the payload carried thinking input blocks
    (e.g. minted by another model after a host-side switch, or the endpoint
    rejects thinking replay), they are dropped for that one call —
    availability over continuity, mirroring the Responses transport.

    Prompt caching: every call carries ``cache_control`` breakpoints
    (:meth:`_apply_prompt_cache`) — the end of ``system`` (or the last
    tool when there is no system), plus the tails of the last two turns —
    so the next step of a tool loop reads the previous step's prefix at
    0.1× instead of re-paying full input price (writes cost 1.25×, and
    this kernel's step loop re-sends the whole conversation each call —
    exactly the repeated-prefix shape caching exists for). A gateway 400
    that clears after dropping the breakpoints disables them for the
    transport instance — availability over economy, mirroring the
    thinking-replay pass.
    """

    def __init__(self) -> None:
        # Remembered per instance (a runtime owns one transport for its
        # lifetime): cleared after a gateway 400 that looks like a
        # cache_control rejection, so later calls from THIS transport omit
        # the breakpoints. Mirrors ResponsesTransport._include_reasoning —
        # one gateway's rejection must not flip behavior for other
        # endpoints in the same process.
        self._cache_control = True

    def _apply_prompt_cache(self, payload: dict) -> bool:
        """Mark prompt-cache breakpoints on the outgoing payload.

        The protocol caches a request's *prefix* at up to four
        ``cache_control`` breakpoints, and the prefix grows in the order
        tools → system → messages. Breakpoints placed (3 of the 4 allowed):

        - end of ``system`` — covers tools + system (tools precede system
          in the prefix); when there is no system, the last tool instead;
        - last content block of the second-to-last turn;
        - last content block of the last turn.

        The two message breakpoints roll forward one turn per call: the
        entry a step wrote at its tail is the next step's read hit, so each
        step pays full price only for the assistant turn + tool results it
        added. Marked blocks are rebuilt as new dicts, never mutated in
        place — pass-through tool dicts and replayed thinking blocks can be
        shared with host-owned spec lists / stored turns, which must stay
        byte-stable for future prefix matches anyway. Returns whether
        anything was marked.
        """
        placed = False
        system = payload.get("system")
        if isinstance(system, str) and system:
            payload["system"] = [
                {"type": "text", "text": system, "cache_control": dict(_EPHEMERAL)}
            ]
            placed = True
        elif isinstance(system, list) and system:
            last = system[-1]
            if isinstance(last, dict):
                system[-1] = {**last, "cache_control": dict(_EPHEMERAL)}
                placed = True
        elif not system and payload.get("tools"):
            tools = list(payload["tools"])
            last = tools[-1]
            if isinstance(last, dict):
                tools[-1] = {**last, "cache_control": dict(_EPHEMERAL)}
                payload["tools"] = tools
                placed = True
        turns = payload.get("messages") or []
        for turn in turns[-2:]:
            blocks = turn.get("content") if isinstance(turn, dict) else None
            if isinstance(blocks, list) and blocks and isinstance(blocks[-1], dict):
                blocks[-1] = {**blocks[-1], "cache_control": dict(_EPHEMERAL)}
                placed = True
        return placed

    def _payload(
        self,
        messages,
        tools,
        tool_choice,
        max_tokens,
        temperature,
        *,
        stream: bool,
        include_reasoning: bool,
        reasoning_scope: str,
        reasoning_effort: str | None = None,
    ) -> dict:
        system, turns = _messages_to_anthropic(
            messages,
            include_reasoning=include_reasoning,
            reasoning_scope=reasoning_scope,
        )
        payload: dict[str, Any] = {
            "model": "",
            "max_tokens": (
                max_tokens if max_tokens is not None else _MESSAGES_DEFAULT_MAX_TOKENS
            ),
            "messages": turns,
        }
        if stream:
            payload["stream"] = True
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            ts = _tools_to_anthropic(tools)
            if ts and tool_choice != "none":
                payload["tools"] = ts
                payload["tool_choice"] = _tool_choice_to_anthropic(tool_choice)
        if reasoning_effort is not None:
            budget = _EFFORT_THINKING_BUDGETS.get(reasoning_effort)
            if budget:
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        return payload

    @staticmethod
    def _finalize(payload: dict) -> dict:
        """Post-merge protocol shims (thinking × temperature × max_tokens)."""
        thinking = payload.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            payload.pop("temperature", None)
            budget = _to_int(thinking.get("budget_tokens"))
            if budget and payload.get("max_tokens", 0) <= budget:
                payload["max_tokens"] = budget + _MESSAGES_DEFAULT_MAX_TOKENS
        return payload

    @staticmethod
    def _has_thinking(payload: dict) -> bool:
        for turn in payload.get("messages") or []:
            if not (isinstance(turn, dict) and turn.get("role") == "assistant"):
                continue
            for block in turn.get("content") or []:
                if isinstance(block, dict) and block.get("type") in (
                    "thinking",
                    "redacted_thinking",
                ):
                    return True
        return False

    async def _complete_with_fallback(
        self,
        client,
        *,
        base_url,
        api_key,
        model,
        messages,
        tools,
        tool_choice,
        max_tokens,
        temperature,
        attempts,
        sleep_429,
        sleep_err,
        include_reasoning,
        reasoning_scope,
        reasoning_effort=None,
        extra_body=None,
        extra_headers=None,
    ) -> TransportResult:
        """Non-stream call with the 400-degradation pass (see class doc)."""
        want_reasoning = include_reasoning
        for _pass in range(3):
            payload = self._payload(
                messages,
                tools,
                tool_choice,
                max_tokens,
                temperature,
                stream=False,
                include_reasoning=want_reasoning,
                reasoning_scope=reasoning_scope,
                reasoning_effort=reasoning_effort,
            )
            payload = _merge_extra(extra_body, payload)
            payload["model"] = model
            self._finalize(payload)
            cache_marked = (
                self._apply_prompt_cache(payload) if self._cache_control else False
            )
            try:
                data = await _post_messages(
                    client,
                    base_url,
                    api_key,
                    payload,
                    attempts=attempts,
                    sleep_429=sleep_429,
                    sleep_err=sleep_err,
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                )
                return _parse_anthropic(data)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 400:
                    raise
                if want_reasoning and self._has_thinking(payload):
                    want_reasoning = False
                    log.warning(
                        "[messages] retrying without thinking input "
                        "blocks after HTTP 400"
                    )
                    continue
                if cache_marked:
                    self._cache_control = False
                    log.warning(
                        "[messages] retrying without prompt-cache "
                        "breakpoints after HTTP 400"
                    )
                    continue
                raise
        raise RuntimeError("unreachable: 400-degradation pass exhausted")

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
        return await self._complete_with_fallback(
            client,
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            attempts=attempts,
            sleep_429=sleep_429,
            sleep_err=sleep_err,
            include_reasoning=include_reasoning,
            reasoning_scope=reasoning_scope,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            extra_headers=extra_headers,
        )

    async def _stream_attempts(
        self,
        client,
        base_url: str,
        api_key: str,
        payload: dict,
        *,
        attempts: int,
        sleep_429: float,
        sleep_err: float,
        extra_headers: dict | None = None,
        extra_body: dict | None = None,
    ) -> AsyncIterator[dict]:
        """One streamed attempt cycle over an established payload (see
        :meth:`complete_stream`)."""
        headers = anthropic_headers(api_key, extra_headers)
        url = f"{base_url}/messages"
        last_exc: Exception | None = None
        for attempt in range(attempts):
            delivered = False
            try:
                async with client.stream(
                    "POST", url, json=payload, headers=headers
                ) as resp:
                    if resp.status_code == 429 and attempt + 1 < attempts:
                        await resp.aread()
                        # Same Retry-After-aware policy as the chat /
                        # responses stream paths.
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
                            request=resp.request,
                            response=resp,
                        )
                    blocks: dict[int, dict] = {}
                    usage: dict = {}
                    stop_reason = None
                    async for event, data_txt in iter_sse_lines(resp):
                        try:
                            data = json.loads(data_txt)
                        except ValueError:
                            continue
                        if not isinstance(data, dict):
                            continue
                        etype = event or data.get("type")
                        if etype == "content_block_start":
                            cb = data.get("content_block")
                            if isinstance(cb, dict):
                                blocks[data.get("index") or 0] = dict(cb)
                        elif etype == "content_block_delta":
                            delta = data.get("delta") or {}
                            idx = data.get("index") or 0
                            slot = blocks.setdefault(idx, {"type": "text", "text": ""})
                            dtype = delta.get("type")
                            if dtype == "text_delta":
                                text = delta.get("text") or ""
                                if text:
                                    slot["text"] = (slot.get("text") or "") + text
                                    delivered = True
                                    yield {"delta": text}
                            elif dtype == "input_json_delta":
                                slot["input_json"] = (slot.get("input_json") or "") + (
                                    delta.get("partial_json") or ""
                                )
                            elif dtype == "thinking_delta":
                                slot["thinking"] = (slot.get("thinking") or "") + (
                                    delta.get("thinking") or ""
                                )
                            elif dtype == "signature_delta":
                                slot["signature"] = (slot.get("signature") or "") + (
                                    delta.get("signature") or ""
                                )
                        elif etype == "message_start":
                            # input tokens arrive on message_start only
                            mu = (data.get("message") or {}).get("usage")
                            if isinstance(mu, dict):
                                usage.update(mu)
                        elif etype == "message_delta":
                            d = data.get("delta") or {}
                            stop_reason = d.get("stop_reason") or stop_reason
                            if isinstance(data.get("usage"), dict):
                                usage.update(data["usage"])
                        elif etype == "message_stop":
                            break
                        elif etype == "error":
                            err = data.get("error") or {}
                            why = err.get("message") or data_txt[:200]
                            raise RuntimeError(f"messages 流错误：{why}")
                yield {
                    "result": _blocks_to_result(
                        [blocks[i] for i in sorted(blocks)], usage, stop_reason
                    )
                }
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                fatal = isinstance(
                    exc, httpx.HTTPStatusError
                ) and not _retryable_status(exc.response.status_code)
                log.warning(
                    "[messages-stream] attempt %d/%d failed%s: %s",
                    attempt + 1,
                    attempts,
                    " (fatal, not retrying)" if fatal else "",
                    exc,
                )
                if delivered or fatal:
                    raise _annotate_extra_body(exc, extra_body) from exc
                if attempt + 1 >= attempts:
                    break
                if sleep_err:
                    await asyncio.sleep(_jitter(sleep_err * (attempt + 1)))
        assert last_exc is not None
        raise _annotate_extra_body(last_exc, extra_body)

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
        """Streamed variant of :meth:`complete` over the Messages API.

        Yields ``{"delta": text}`` for ``content_block_delta`` text deltas,
        then one ``{"result": TransportResult}`` assembled from the
        accumulated blocks (tool inputs joined from ``input_json_delta``
        fragments, thinking blocks captured as ``reasoning``) and the
        terminal usage (``message_start`` input + ``message_delta`` output).
        The 400-degradation passes (drop thinking input, then prompt-cache
        breakpoints) only run while nothing has been delivered — a failure
        after the first delta raises instead (deltas cannot be unsent).
        """
        want_reasoning = include_reasoning
        for _pass in range(3):
            payload = self._payload(
                messages,
                tools,
                tool_choice,
                max_tokens,
                temperature,
                stream=True,
                include_reasoning=want_reasoning,
                reasoning_scope=reasoning_scope,
                reasoning_effort=reasoning_effort,
            )
            payload = _merge_extra(extra_body, payload)
            payload["model"] = model
            self._finalize(payload)
            cache_marked = (
                self._apply_prompt_cache(payload) if self._cache_control else False
            )
            delivered = False
            try:
                async for part in self._stream_attempts(
                    client,
                    base_url,
                    api_key,
                    payload,
                    attempts=attempts,
                    sleep_429=sleep_429,
                    sleep_err=sleep_err,
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                ):
                    if "delta" in part:
                        delivered = True
                    yield part
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 400 or delivered:
                    raise
                if want_reasoning and self._has_thinking(payload):
                    want_reasoning = False
                    log.warning(
                        "[messages-stream] retrying without thinking "
                        "input blocks after HTTP 400"
                    )
                    continue
                if cache_marked:
                    self._cache_control = False
                    log.warning(
                        "[messages-stream] retrying without "
                        "prompt-cache breakpoints after HTTP 400"
                    )
                    continue
                raise
        raise RuntimeError("unreachable: 400-degradation pass exhausted")
