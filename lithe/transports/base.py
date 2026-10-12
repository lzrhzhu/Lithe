"""The unified transport contract and the shaping helpers every wire shares.

An :class:`LLMTransport` sends one model request and returns the *unified*
shape ``{content, tool_calls, usage, ...}`` (see :data:`TransportResult`) so
the runtime never branches on chat-completions vs Responses vs Messages wire
formats. The helpers here — usage normalization, content flattening, the
``extra_body`` merge discipline — are protocol-agnostic by construction and
are imported by each transport module, never copied.

Wire-level plumbing (retry, ``Retry-After``, headers, SSE framing) lives one
level lower, in :mod:`lithe.wire`.
"""
from __future__ import annotations

from typing import Any, Protocol

# Unified result every transport returns.
TransportResult = dict[str, Any]


class LLMTransport(Protocol):
    """Send one model call; return ``{content, tool_calls, usage}``.

    ``tool_calls`` mirrors the chat-completions shape
    (``{id, type:"function", function:{name, arguments(str)}}``) regardless of
    the underlying protocol, so the runtime's dispatch is protocol-agnostic.

    ``reasoning`` (optional, Responses/Messages transports) is the raw list of
    reasoning output items — the runtime attaches them to the assistant
    message so the *next* call of the same tool loop can pass them back
    (reasoning continuity for reasoning models). ``include_reasoning`` tells
    the transport whether the runtime wants reasoning replayed at all;
    ``reasoning_scope`` selects which turns replay it (``"loop"``: active
    tool loop only, ``"conversation"``: every turn — see
    ``lithe.transports.openai_responses._messages_to_input``).
    """

    async def complete(
        self,
        client: Any,
        *,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        attempts: int = 1,
        sleep_429: float = 0.0,
        sleep_err: float = 0.0,
        include_reasoning: bool = True,
        reasoning_scope: str = "loop",
        reasoning_effort: str | None = None,
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
    ) -> TransportResult: ...


def _flatten_text(content: Any) -> str:
    """Coerce a message content (str / list of {type:text/text} blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in (
                "text",
                "input_text",
                "output_text",
            ):
                out.append(p.get("text", "") or "")
            elif isinstance(p, str):
                out.append(p)
        return "".join(out)
    return str(content)


def _merge_extra(extra_body: dict | None, internal: dict) -> dict:
    """Merge host-supplied body fields under the internally computed ones.

    ``extra_body`` (e.g. ``LLMConfig.extra_body``) carries vendor extensions
    the kernel does not model (``top_p``, ``seed``, ``enable_thinking``,
    ``reasoning`` budget fields, ...). Internal keys (``tools``,
    ``tool_choice``, ``max_tokens`` / ``max_output_tokens``, ...) win on
    collision: the loop mechanics must not be silently broken by a payload
    field, and those knobs have first-class ``LLMConfig`` members already.

    Dict-vs-dict collisions merge **per key** (internal still winning each
    key): the Responses transport emits ``reasoning: {"effort": ...}`` while
    a host's ``extra_body["reasoning"]`` may carry sibling keys
    (``max_tokens``, ``exclude``) — a shallow update would drop them.
    """
    if not extra_body:
        return internal
    merged = dict(extra_body)
    for key, val in internal.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            combined = dict(merged[key])
            combined.update(val)
            merged[key] = combined
        else:
            merged[key] = val
    return merged


def _to_int(val) -> int:
    """Best-effort integer coercion for gateway usage fields.

    Real gateways send ``"1,234"``, float-strings or worse in usage blocks;
    an unparseable value degrades to 0 rather than crashing an otherwise
    successful run — the same contract the runtime applies to custom
    transports (``_as_int``).
    """
    try:
        return int(val)
    except (TypeError, ValueError):
        try:
            return int(str(val).replace(",", "").strip())
        except (TypeError, ValueError):
            return 0


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
    (``cost_breakdown``, ...) pass through. All integer coercion is
    fault-tolerant (see :func:`_to_int`): a dirty value becomes 0 instead of
    a ValueError that would kill the run.
    """
    u = dict(usage or {})
    p_raw = u.get("prompt_tokens", u.get("input_tokens"))
    c_raw = u.get("completion_tokens", u.get("output_tokens"))
    p, c = _to_int(p_raw), _to_int(c_raw)
    if p_raw is not None:
        u["prompt_tokens"] = p
    if c_raw is not None:
        u["completion_tokens"] = c
    t_raw = u.get("total_tokens")
    t = _to_int(t_raw)
    if t_raw is None and p_raw is not None and c_raw is not None:
        t = p + c
    if t_raw is not None or (p_raw is not None and c_raw is not None):
        u["total_tokens"] = t
    for detail_key in ("prompt_tokens_details", "input_tokens_details"):
        details = u.get(detail_key)
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            u["cached_tokens"] = _to_int(details["cached_tokens"])
            break
    out_details = u.get("output_tokens_details")
    if (
        isinstance(out_details, dict)
        and out_details.get("reasoning_tokens") is not None
    ):
        u["reasoning_tokens"] = _to_int(out_details["reasoning_tokens"])
    return u
