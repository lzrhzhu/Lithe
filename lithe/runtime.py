"""The ReAct runtime: drives the model ↔ tool loop and emits events.

``AgentRuntime.run`` is the kernel's only active component. It owns the step
loop, calls the model (via :mod:`lithe.llm`), dispatches returned
``tool_calls`` through the :class:`~lithe.tools.ToolRegistry`, and yields
display events. It never persists anything: every assistant turn and tool
result is also handed to attached :class:`~lithe.events.EventSink`\\ s via
``on_record``, so a host decides whether and how to store the conversation.

The runtime is storage-free and host-agnostic: a host builds the initial
``messages`` (system prompt + history + user task) and the tool spec list, then
iterates ``run`` to stream events to its frontend. The display-event vocabulary
is minimal; domain UI (file diffs, guidance proposals, ...) rides inside each
tool result's ``ui`` list and is forwarded verbatim.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue as _queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any
from collections.abc import AsyncIterator, Callable

import httpx

from lithe.context import AgentContext
from lithe.events import Event, EventSink, EventType
from lithe.memory import reasoning_summary_text, truncate_tool_result
from lithe.modes import ToolCategory
from lithe.tools import ToolRegistry, ToolResult
from lithe.transports import LLMTransport, make_transport

log = logging.getLogger("lithe.runtime")

_MAX_STEPS_FALLBACK = "（已达到最大步数。如尚未完成，请补充细节后继续提问。）"

# Run cut short by its cost/token budget (instead of the step cap): the notice
# becomes the run's final text so the frontend can show why it stopped.
_BUDGET_FALLBACK = ("（已达到本次运行的成本或 token 预算，运行提前结束。"
                    "已完成的步骤仍然有效；如需继续，请提高预算后重新提问。）")

# Nudge appended to a tool result when the *identical* (tool, args) call is
# issued more than this many times with no mutating (WRITE/META) tool in
# between — the classic stuck-model loop. A mutating tool bumps the "epoch"
# and resets every signature's count: a read after a write is a fresh
# observation (the workspace stale-file guard even demands re-reads), not a
# stuck repeat. None disables the repeat guard.
DEFAULT_REPEAT_CALL_LIMIT = 3

# Default mid-run context budget (~chars of messages incl. tool results).
# Generous enough that normal runs never notice; only genuinely huge
# conversations trigger trimming. None disables trimming entirely.
DEFAULT_CONTEXT_BUDGET = 400_000

# When over budget, old tool-result contents are shrunk head+tail through
# these caps (each pass smaller) until the conversation fits. Only `content`
# strings of tool messages shrink — roles/ids stay intact, so the payload
# remains structurally valid for the chat-completions API.
_TRIM_STAGES = (1200, 400, 100)
_TRIM_FLOOR = 60
_KEEP_RECENT_TOOL_MSGS = 2

# Escalation when shrinking cannot reach the budget (the overage is prose /
# reasoning / protected-recent content, none of which may shrink): whole old
# exchanges — one assistant turn plus its tool results, always as a complete
# unit — are dropped and replaced by a single omission note, so the next
# request still goes out under budget instead of a guaranteed window
# overflow. Dropped turns are already recorded with sinks; only the live
# model context loses them. The newest exchanges are never dropped.
_KEEP_RECENT_EXCHANGES = 2
_OMITTED_TURNS_NOTE = ("（系统注：为控制上下文长度，此前若干轮工具调用过程已从"
                       "上下文中移除；这些步骤的结论已体现在其后的对话内容中，"
                       "无需重复执行。）")


def _message_size(m: dict) -> int:
    n = 0
    content = m.get("content")
    if isinstance(content, str):
        n += len(content)
    elif isinstance(content, list):
        # Multimodal content: text blocks count as their text length, image
        # blocks by their payload length (an inline data: URL IS the image,
        # in chars). Previously list content counted as 0, so a multimodal
        # conversation under-reported its size and skewed the trim budget
        # and the chars-per-token calibration.
        for part in content:
            if isinstance(part, str):
                n += len(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    n += len(text)
                iu = part.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else None
                if isinstance(url, str):
                    n += len(url)
    tcs = m.get("tool_calls")
    if tcs:
        try:
            n += len(json.dumps(tcs, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            n += 1024
    # Reasoning items count toward the conversation's size (encrypted_content
    # blobs can be kilobytes) — but are never trimmed: dropping one orphans
    # the function_call it precedes and breaks the pass-back pairing.
    rs = m.get("reasoning")
    if rs:
        try:
            n += len(json.dumps(rs, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            n += 1024
    return n


def _context_size(messages: list[dict]) -> int:
    """Rough size of the outgoing conversation (chars, incl. tool_calls JSON).
    Used for the context-fullness indication and by :func:`_trim_context`."""
    return sum(_message_size(m) for m in messages)


def _exchange_units(messages: list[dict]) -> list[tuple[int, int]]:
    """``(start, end)`` index ranges of complete exchanges — one assistant
    message carrying ``tool_calls`` plus the contiguous tool results that
    answer it. Dropping such a range never orphans a call or a result."""
    units: list[tuple[int, int]] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                j += 1
            units.append((i, j))
            i = j
        else:
            i += 1
    return units


def _trim_context(messages: list[dict], *, budget: int,
                  keep_recent: int = _KEEP_RECENT_TOOL_MSGS) -> int:
    """Best-effort mid-run context compaction, in two tiers.

    Tier 1 shrinks old tool-result contents (head+tail, like memory replay)
    until the conversation fits ``budget`` chars. The newest ``keep_recent``
    tool results are never touched — they are what the model is actively
    working from. Structural validity is preserved by construction: only
    ``content`` strings change, so every assistant ``tool_call`` keeps its
    matching tool message.

    Tier 2 (escalation) fires when tier 1 cannot reach the budget — the
    overage then lives in content tier 1 must not touch (user/assistant
    prose, reasoning items, protected-recent tool results). Complete old
    exchanges are dropped oldest-first and replaced by one omission note
    (:data:`_OMITTED_TURNS_NOTE`), keeping the payload under budget instead
    of sending a request that overflows the model's window. System and user
    messages are never dropped, the newest ``_KEEP_RECENT_EXCHANGES``
    exchanges are protected, and — to avoid losing history for nothing —
    drops are committed only when they actually reach the budget; if even
    dropping every droppable exchange would stay over, nothing is dropped.

    Returns the conversation's size in chars afterwards (also when no trim
    was needed), so callers can keep a running total without rescanning.
    """
    total = _context_size(messages)
    if total <= budget:
        return total
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    shrinkable = tool_idx[:-keep_recent] if keep_recent else list(tool_idx)
    for cap in _TRIM_STAGES:
        for i in shrinkable:
            if total <= budget:
                return total
            m = messages[i]
            c = m.get("content")
            if isinstance(c, str) and len(c) > cap:
                new = truncate_tool_result(c, cap)
                total += len(new) - len(c)
                m["content"] = new
    for i in shrinkable:  # still over: hard floor, oldest first
        if total <= budget:
            return total
        m = messages[i]
        c = m.get("content")
        if isinstance(c, str) and len(c) > _TRIM_FLOOR:
            cut = c[:_TRIM_FLOOR] + "…[已压缩]"
            total += len(cut) - len(c)
            m["content"] = cut
    if total <= budget:
        return total

    # Tier 2: simulate dropping whole old exchanges; commit only if the
    # simulation actually reaches the budget.
    protected = set(tool_idx[-keep_recent:]) if keep_recent else set()
    units = _exchange_units(messages)
    droppable = units[:-_KEEP_RECENT_EXCHANGES] if _KEEP_RECENT_EXCHANGES \
        else units
    sim = total
    drop_ranges: list[tuple[int, int]] = []
    for start, end in droppable:
        if sim <= budget:
            break
        if any(i in protected for i in range(start, end)):
            break  # protection covers a suffix of exchanges: nothing older left
        sim -= sum(_message_size(messages[i]) for i in range(start, end))
        drop_ranges.append((start, end))
    if not drop_ranges or sim > budget:
        return total  # hopeless: overage lives in undroppable content
    first_drop = drop_ranges[0][0]
    drop_idxs = {i for start, end in drop_ranges for i in range(start, end)}
    # stale omission notes from an earlier pass carry no state; remove them
    # so the fresh note below stays unique
    drop_idxs.update(i for i, m in enumerate(messages)
                     if m.get("role") == "assistant" and not m.get("tool_calls")
                     and m.get("content") == _OMITTED_TURNS_NOTE)
    kept_before = sum(1 for i in range(first_drop) if i not in drop_idxs)
    new_messages = [m for i, m in enumerate(messages) if i not in drop_idxs]
    new_messages.insert(kept_before,
                        {"role": "assistant", "content": _OMITTED_TURNS_NOTE})
    messages[:] = new_messages
    return _context_size(messages)


@dataclass
class LLMConfig:
    """Endpoint + retry policy for the runtime's model calls.

    ``transport`` selects the wire protocol: ``"chat"`` (OpenAI
    chat-completions, default) or ``"responses"`` (OpenAI Responses API).
    A host may also pass a custom :class:`~lithe.transports.LLMTransport`
    instance. ``stream=True`` asks a streaming-capable transport for token
    deltas (the runtime then emits ``assistant_delta`` events ahead of the
    final ``assistant`` one; unsupported transports fall back silently).
    ``context_window`` is the model's token window (host knowledge — the
    kernel cannot know it); when set, usage events and :class:`RunStats`
    report ``context_percent`` so a frontend can show how full the
    conversation is. ``temperature`` / ``max_tokens`` are plain sampling
    parameters forwarded on every model call (``None`` leaves them unset).
    ``reasoning_replay`` (default True) asks a reasoning-capable transport to
    pass reasoning items back (Responses protocol); set False for gateways
    that reject reasoning input items. ``reasoning_scope`` selects which turns
    replay their reasoning: ``"loop"`` (default) scopes pass-back to the
    active tool loop (vendor's cost guidance — dropped once a new user
    message arrives); ``"conversation"`` replays every turn's reasoning,
    giving cross-turn chain continuity for hosts that prefer quality over
    token cost (the kernel never compresses reasoning either way; a host
    decides when to compact).

    ``extra_body`` carries vendor request fields the kernel does not model
    (``top_p``, ``seed``, ``stop``, ``response_format``,
    ``enable_thinking``, ``reasoning`` effort, ...) into the JSON payload of
    every call, both transports; internally computed keys (``tools``,
    ``tool_choice``, ``max_tokens`` ...) win on collision — those knobs have
    first-class members here. ``default_headers`` merges over the bearer /
    JSON content-type headers (OpenRouter's ``HTTP-Referer`` /
    ``X-Title``, ``OpenAI-Organization``, ...). Values in both must be
    JSON-serializable / header-safe.

    ``pricing`` is a host-declared per-1M-token price table,
    ``{"prompt": float, "completion": float, "cached_prompt": float?}``,
    used to compute call cost when the gateway reports none (OpenAI's API
    and many gateways never send ``usage.cost`` — without a table the
    ``max_cost`` budget and cost accounting silently read 0 there). A
    gateway-reported cost always wins over the computed one; cached input
    is billed at ``cached_prompt`` when given, else at the ``prompt`` price
    (over-counting is the safe direction for a budget).
    """
    model: str
    base_url: str
    api_key: str
    timeout: float = 180.0
    attempts: int = 1
    sleep_429: float = 0.0
    sleep_err: float = 0.0
    transport: str | LLMTransport = "chat"
    stream: bool = False
    context_window: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_replay: bool = True
    reasoning_scope: str = "loop"
    extra_body: dict | None = None
    default_headers: dict | None = None
    pricing: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.reasoning_scope not in ("loop", "conversation"):
            raise ValueError(
                f"LLMConfig.reasoning_scope must be 'loop' or 'conversation', "
                f"got {self.reasoning_scope!r}")
        if self.pricing is not None:
            bad = sorted(k for k, v in self.pricing.items()
                         if k not in ("prompt", "completion", "cached_prompt")
                         or isinstance(v, bool)
                         or not isinstance(v, (int, float)) or v < 0)
            if bad or not ({"prompt", "completion"}
                           & set(self.pricing)):
                raise ValueError(
                    "LLMConfig.pricing must map 'prompt' and/or 'completion' "
                    "(optionally 'cached_prompt') to non-negative "
                    f"per-1M-token prices; bad keys/values: {bad}")


@dataclass
class RunStats:
    """Mutable accounting for one run; the host reads it to build its ``done`` event.

    Token fields are cumulative across every model call; ``cached_tokens``
    counts input tokens served from the provider's cache (a subset of
    ``prompt_tokens``, not additive); ``context_tokens`` is the *input*
    context length of the last call (what the model actually saw —
    the number to compare against ``context_window`` for a fullness gauge).
    """
    final_text: str = ""
    status: str = "done"
    last_step: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    context_tokens: int = 0
    context_window: int | None = None

    @property
    def context_percent(self) -> float | None:
        """How full the context window was at the last model call (0-100),
        or None when either side of the ratio is unknown."""
        if not self.context_window or not self.context_tokens:
            return None
        return round(100.0 * self.context_tokens / self.context_window, 1)


def _parse_args(tc: dict) -> tuple[dict, str | None]:
    """Parse one tool_call's ``arguments`` into ``(args, error)``.

    A malformed arguments string (not valid JSON, or valid JSON that is not an
    object) yields ``( {}, message )`` so the runtime can hand the error back
    to the model as a failed tool result instead of silently executing the
    tool with empty args.
    """
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return {}, None
    raw = fn.get("arguments", "")
    if isinstance(raw, dict):
        return raw, None
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return {}, f"{exc}（原文片段：{raw[:120]!r}）"
    if not isinstance(parsed, dict):
        return {}, f"arguments 应为 JSON 对象，得到 {type(parsed).__name__}"
    return parsed, None


def _stop_triggered(stop) -> bool:
    """A stop handle is an ``asyncio.Event`` / ``threading.Event``-like object
    (anything with ``is_set()``) or a zero-arg callable returning truthiness."""
    if stop is None:
        return False
    is_set = getattr(stop, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    if callable(stop):
        return bool(stop())
    return bool(stop)


async def _drain_inbox(inbox) -> list[str]:
    """Non-blocking drain of the steering inbox.

    Accepts anything with a ``get_nowait()`` — an ``asyncio.Queue`` (same
    loop) or a ``queue.Queue`` (a cross-thread host pushing from another
    thread; thread-safe by construction). An empty queue ends the drain; a
    broken inbox logs and returns what it had rather than killing the run —
    steering is an auxiliary channel, never a load-bearing one.
    """
    if inbox is None:
        return []
    out: list[str] = []
    while True:
        try:
            out.append(inbox.get_nowait())
        except (asyncio.QueueEmpty, _queue.Empty):
            return out
        except Exception:  # noqa: BLE001 — a broken inbox must not kill the run
            log.warning("steering inbox drain failed (ignored)",
                        exc_info=True)
            return out


def _normalize_assistant(msg: dict) -> dict:
    """Keep only the OpenAI-relevant keys for the next request.

    This is the single point where a missing ``tool_call`` id is synthesized
    (``call_`` + uuid fragment): some gateways omit ids on non-streamed
    calls, and the streaming assembler deliberately leaves them ``None``
    rather than minting per-response ``call_0``-style ids that collide across
    steps. Without an id the paired tool result carries
    ``tool_call_id: null`` and the next request is rejected with a 400. The
    synthesis mutates the caller's tool_call dicts in place, so the recorded
    turn, the display events and the in-memory ``messages`` all agree on one
    id.

    Reasoning items (Responses protocol) ride the message under the
    ``reasoning`` key: kept verbatim so the next call in the same tool loop
    can pass them back, and so sinks can persist them. They are opaque to
    this kernel — never parsed or rewritten.
    """
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
    reasoning = msg.get("reasoning")
    if reasoning:
        out["reasoning"] = reasoning
    tcs = msg.get("tool_calls")
    if tcs:
        norm: list[dict] = []
        for tc in tcs:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict):
                continue
            tc_id = tc.get("id")
            if not isinstance(tc_id, str) or not tc_id:
                tc_id = "call_" + uuid.uuid4().hex[:20]
                tc["id"] = tc_id
            norm.append({
                "id": tc_id,
                "type": tc.get("type", "function"),
                "function": {
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments", ""),
                },
            })
        if norm:
            out["tool_calls"] = norm
    return out


def _as_int(val) -> int:
    """Coerce a (possibly dirty, custom-transport-supplied) usage value.

    Built-in transports sanitize via ``norm_usage``; a custom ``LLMTransport``
    may return raw shapes ("1,234", [1], None-ish objects). A parse failure
    must degrade to 0, never escape the run loop.
    """
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _gateway_cost(data: dict) -> float | None:
    """Cost as reported by the gateway, when it reports one at all.

    Returns ``None`` when no cost field is present (OpenAI's API and many
    gateways never send one) so the caller can fall back to a host-declared
    price table — a reported 0.0 is a real answer ("this call was free"),
    not the same as silence.
    """
    usage = data.get("usage") or {}
    breakdown = usage.get("cost_breakdown") or {}
    for src in (breakdown.get("total_cost"), usage.get("cost"), data.get("cost")):
        if src is not None:
            try:
                return float(src)
            except (TypeError, ValueError):
                continue
    return None


def _computed_cost(pricing: dict | None, usage: dict) -> float:
    """Cost from a host-declared per-1M-token price table (``LLMConfig.pricing``).

    Cached input tokens are billed at ``cached_prompt`` when given, else at
    the ``prompt`` price — over-counting a cache discount is the safe
    direction for a budget. Returns 0.0 without a table, matching the
    gateway-silent "unknown" accounting.
    """
    if not pricing:
        return 0.0
    p = pricing.get("prompt")
    c = pricing.get("completion")
    if p is None and c is None:
        return 0.0
    prompt_tok = _as_int(usage.get("prompt_tokens"))
    comp_tok = _as_int(usage.get("completion_tokens"))
    cached = min(_as_int(usage.get("cached_tokens")), prompt_tok)
    cost = 0.0
    if p is not None:
        cached_price = pricing.get("cached_prompt", p)
        cost += ((prompt_tok - cached) / 1e6) * p \
            + (cached / 1e6) * cached_price
    if c is not None:
        cost += (comp_tok / 1e6) * c
    return cost


def _call_cost(cfg: LLMConfig, usage: dict) -> float:
    """Per-call cost: the gateway's own number when it sends one, else the
    host's price table, else 0.0 (unknown)."""
    reported = _gateway_cost({"usage": usage})
    if reported is not None:
        return reported
    return _computed_cost(cfg.pricing, usage)


class AgentRuntime:
    """Drives one ReAct run, yielding display events and feeding sinks.

    Construct once (with a registry, an :class:`LLMConfig`, optional sinks, a
    step cap, and optional run budgets ``max_cost`` / ``max_total_tokens``
    that end a runaway run with ``status="budget_exceeded"``); call :meth:`run`
    per task. ``repeat_call_limit`` (default 3) nudges the model when it
    issues the *identical* (tool, args) call yet again with no WRITE/META
    tool in between (a mutating tool resets the counters — reads after
    writes are fresh observations, not stuck repeats) — the hint rides inside
    the tool result, invisible to display events. Pass ``run`` a ``stats`` to
    read the final status / cost / step count after the generator is
    exhausted.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm_config: LLMConfig,
        *,
        sinks: list[EventSink] | None = None,
        max_steps: int = 35,
        context_budget: int | None = DEFAULT_CONTEXT_BUDGET,
        context_token_threshold: float | None = 0.8,
        strict_records: bool = False,
        max_cost: float | None = None,
        max_total_tokens: int | None = None,
        repeat_call_limit: int | None = DEFAULT_REPEAT_CALL_LIMIT,
        http_client: httpx.AsyncClient | None = None,
        emit_envelope: bool = False,
    ):
        self.registry = registry
        self.llm_config = llm_config
        self.sinks = sinks or []
        self.max_steps = max_steps
        self.context_budget = context_budget
        # Token-calibrated trim trigger: when set (>0) and the host declared
        # LLMConfig.context_window, the mid-run char budget derives from the
        # real token window (window × threshold × measured chars-per-token)
        # instead of the plain char budget — the token window is the
        # authoritative limit, and chars-per-token auto-calibrates per traffic
        # mix (CJK vs code). None disables token calibration.
        self.context_token_threshold = context_token_threshold
        self.strict_records = strict_records
        self.max_cost = max_cost
        self.max_total_tokens = max_total_tokens
        self.repeat_call_limit = repeat_call_limit
        # Host-injected HTTP client: when set it is REUSED across runs and
        # never closed here (the host owns its lifetime) — a long-lived
        # service keeps connection pooling, limits, proxy / verify / redirect
        # settings that a fresh per-run AsyncClient cannot carry. Default
        # (None): one client per run, closed when the run ends.
        self.http_client = http_client
        # Emit the run envelope (RUN_START before / DONE after the step loop)
        # from the runtime itself, for hosts that use AgentRuntime directly
        # instead of through AgentHost (which builds a richer envelope).
        # Off by default so AgentHost-based hosts never see duplicates.
        self.emit_envelope = emit_envelope
        self.transport = make_transport(llm_config.transport)

    def _over_budget(self, stats: RunStats) -> bool:
        """True when cumulative cost / token accounting has crossed the
        configured budget caps. Checked after each model call (before any
        further tool execution) and before the next call — the latter matters
        when a host seeds ``stats`` with prior-conversation totals to budget
        across runs."""
        return (
            (self.max_cost is not None and stats.total_cost > self.max_cost)
            or (self.max_total_tokens is not None
                and stats.total_tokens > self.max_total_tokens)
        )

    def _effective_char_budget(self, cfg: LLMConfig, size_state: dict) -> int | None:
        """Char budget for mid-run trimming, token-calibrated when possible.

        Token mode (active when ``context_token_threshold`` is set and the
        host declared ``cfg.context_window``) supersedes the plain char
        ``context_budget``: budget_chars = window × threshold × cpt, where
        cpt (chars per token) is calibrated from the last measured API usage
        (``size_state["cpt"]``; a conservative 2.0 default stands in before
        the first measurement — CJK-heavy traffic). Without a declared window
        the classic char budget applies unchanged.
        """
        if (self.context_token_threshold is not None
                and self.context_token_threshold > 0 and cfg.context_window):
            cpt = size_state.get("cpt") or 2.0
            return int(cfg.context_window * self.context_token_threshold * cpt)
        return self.context_budget

    async def _publish(self, ctx: AgentContext, event: Event) -> None:
        """Fan an event out to sinks. Display sinks are best-effort by design:
        a broken frontend/metrics sink logs and gets out of the way — it must
        never kill the agent run it is only observing."""
        for sink in self.sinks:
            try:
                await sink.on_event(ctx, event)
            except Exception as exc:  # noqa: BLE001
                log.warning("sink %s on_event failed (ignored): %s",
                            type(sink).__name__, exc)

    async def _record(self, ctx: AgentContext, record: dict) -> None:
        """Persist one message record. Also best-effort by default (a flaky
        sink delays nothing); hosts that treat silent persistence gaps as
        worse than a dead run construct the runtime with
        ``strict_records=True`` and the first record failure fails the run."""
        for sink in self.sinks:
            try:
                await sink.on_record(ctx, record)
            except Exception as exc:  # noqa: BLE001
                if self.strict_records:
                    raise
                log.warning("sink %s on_record failed (ignored): %s",
                            type(sink).__name__, exc)

    async def _synth_end(self, ctx: AgentContext, messages: list[dict],
                         size_state: dict, text: str) -> Event:
        """Close a run with synthesized fallback text (max-step / budget
        notices): append to ``messages`` AND record it, then build the
        ASSISTANT event. Previously the notice was only emitted as an event,
        so sinks and the next run's replay never learned why the run ended."""
        messages.append({"role": "assistant", "content": text})
        size_state["chars"] += _message_size(messages[-1])
        await self._record(ctx, {"role": "assistant", "content": text,
                                 "tool_calls": None})
        return await self._emit(ctx, {"type": EventType.ASSISTANT, "text": text})

    async def _emit(self, ctx: AgentContext, event: Event) -> Event:
        """Publish to sinks and return the event for the caller to yield."""
        await self._publish(ctx, event)
        return event

    async def run(
        self,
        ctx: AgentContext,
        messages: list[dict],
        tools_spec: list[dict],
        *,
        stats: RunStats | None = None,
        stop: asyncio.Event | threading.Event | Callable[[], bool] | None = None,
        inbox: Any | None = None,
    ) -> AsyncIterator[Event]:
        """Run the model ↔ tool loop until a final answer, an error, the step
        cap, or a cancellation.

        ``messages`` is the host-built initial list (system + history + user
        task); the runtime appends each assistant/tool turn in place, so a host
        or sink can read the full conversation afterwards. ``tools_spec`` is the
        OpenAI function spec list (e.g. ``registry.specs_for_mode(...)``).

        ``stop`` is a cancellation handle — an ``asyncio.Event``, a
        ``threading.Event``, or any zero-arg callable returning truthiness. It
        is checked before each model call, again before dispatching tool
        calls, and between streaming deltas (a cancellation mid-generation
        closes the in-flight model stream instead of paying for the rest of a
        response nobody wants); when triggered the run ends with a
        ``cancelled`` event and ``stats.status == "cancelled"`` (in-flight
        tool calls still complete, so no mutation is left half-applied).

        ``inbox`` is the steering channel: a queue of user texts (anything
        with ``get_nowait()`` — ``asyncio.Queue`` in-loop, ``queue.Queue``
        cross-thread) drained at each step boundary. Queued texts are
        appended as user messages (and recorded / announced via
        ``user_injected`` events) so the model incorporates them on its next
        call — "the user typed while the agent worked". Precedence: stop
        and budgets are checked first (steering cannot buy budget); the
        forced wrap-up step skips the drain, since its tools are already
        withheld and a new request could not be acted on — unconsumed items
        stay in the host's queue.
        """
        stats = stats if stats is not None else RunStats()
        cfg = self.llm_config
        stats.context_window = cfg.context_window
        # Publish the stop handle into the run's shared state so derived
        # runtimes (subagents) can propagate cancellation. Always set (even
        # to None) so a reused context never inherits a previous run's handle.
        ctx.shared["_runtime_stop"] = stop
        # The steering handle is stashed for symmetry (a host-side tool or
        # middleware may find it via ctx.shared), but subagent runtimes
        # deliberately do NOT forward it: steering addresses the orchestrator.
        ctx.shared["_runtime_inbox"] = inbox
        if self.emit_envelope:
            yield await self._emit(ctx, {"type": EventType.RUN_START,
                                         "run_id": ctx.run_id,
                                         "model": cfg.model,
                                         "max_steps": self.max_steps})
        # Running conversation size (chars), maintained incrementally so the
        # per-step context-fullness report doesn't rescan/serialize everything.
        size_state = {"chars": _context_size(messages)}
        # Per-run identical-repeat state, feeding the stuck-model nudge:
        # ``counts`` maps (tool name, canonical args) → (invocations, epoch);
        # ``epoch`` bumps whenever a WRITE/META tool dispatches, which resets
        # every signature's count (see _dispatch_at). Fresh for every `run`.
        repeat_state: dict = {"epoch": 0, "counts": {}}
        if self.http_client is not None:
            async for event in self._drive(ctx, messages, tools_spec,
                                           self.http_client, cfg, stats,
                                           size_state, repeat_state, stop,
                                           inbox):
                yield event
        else:
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                async for event in self._drive(ctx, messages, tools_spec,
                                               client, cfg, stats,
                                               size_state, repeat_state, stop,
                                               inbox):
                    yield event
        if self.emit_envelope:
            yield await self._emit(ctx, {
                "type": EventType.DONE, "run_id": ctx.run_id,
                "steps": stats.last_step, "status": stats.status,
                "cost": round(stats.total_cost, 6),
                "tokens": stats.total_tokens})

    async def _drive(
        self, ctx: AgentContext, messages: list[dict], tools_spec: list[dict],
        client: httpx.AsyncClient, cfg: LLMConfig, stats: RunStats,
        size_state: dict, repeat_state: dict, stop=None, inbox=None,
    ) -> AsyncIterator[Event]:
        """The step loop over an established HTTP client (see :meth:`run`)."""
        for step in range(1, self.max_steps + 1):
            stats.last_step = step
            # The last step of a multi-step budget is a forced wrap-up:
            # tools are withheld (tool_choice="none") so the model must
            # summarize what it has instead of starting another tool round
            # it cannot finish. Single-step budgets (max_steps=1) behave
            # like a normal call — withholding tools there would mean the
            # host's tools never work at all.
            wrap_up = step == self.max_steps and step > 1
            if wrap_up:
                stats.status = "max_steps"  # honest marker; errors overwrite
            state: dict = {"finished": False}
            # aclosing: when the consumer abandons this run mid-step, the
            # step generator is closed deterministically instead of waiting
            # for the GC's asyncgen finalizer.
            async with contextlib.aclosing(self._loop_step(
                    ctx, messages, tools_spec, client, cfg, stats, step,
                    state, size_state, repeat_state, stop, wrap_up=wrap_up,
                    inbox=inbox)) as step_stream:
                async for event in step_stream:
                    yield event
            if state["finished"]:
                return
        stats.status = "max_steps"
        stats.final_text = _MAX_STEPS_FALLBACK
        yield await self._synth_end(ctx, messages, size_state, stats.final_text)

    async def _loop_step(
        self, ctx: AgentContext, messages: list[dict], tools_spec: list[dict],
        client: httpx.AsyncClient, cfg: LLMConfig, stats: RunStats, step: int,
        state: dict, size_state: dict, repeat_state: dict, stop=None,
        wrap_up: bool = False, inbox=None,
    ) -> AsyncIterator[Event]:
        """Execute one model step, yielding its events as they happen.

        Sets ``state["finished"] = True`` when the loop should stop (final
        answer, error, cancellation, or budget cap); yields a
        streaming-friendly event sequence (``step`` → ``assistant_delta``* →
        ``assistant`` → ``usage`` → tool events, with every ``tool_call``
        announced before any of them executes). ``wrap_up`` runs the model
        with tools withheld (``tool_choice="none"``) for the forced final
        summary of a run that exhausted its step budget. Mutates ``messages``
        and ``stats``.
        """
        if _stop_triggered(stop):
            stats.status = "cancelled"
            state["finished"] = True
            yield await self._emit(ctx, {"type": EventType.CANCELLED, "step": step})
            return
        if self._over_budget(stats):
            # Budget guard before spending another model call. Unreachable in
            # the normal loop (the post-call guard below cuts first), but real
            # for a host seeding `stats` with prior-conversation totals to
            # budget across runs.
            stats.status = "budget_exceeded"
            stats.final_text = _BUDGET_FALLBACK
            state["finished"] = True
            yield await self._synth_end(ctx, messages, size_state,
                                        stats.final_text)
            return
        yield await self._emit(ctx, {"type": EventType.STEP, "step": step})

        if inbox is not None and not wrap_up:
            # Steering channel: host-queued user texts enter the loop at the
            # step boundary. Order of guarantees: stop and budgets were
            # checked above (steering cannot buy budget, cancel always
            # wins); wrap-up steps skip the drain — their tools are already
            # withheld, so an injected request could not be acted on.
            for text in await _drain_inbox(inbox):
                if not (isinstance(text, str) and text.strip()):
                    log.warning("ignoring non-text steering message: %r", text)
                    continue
                messages.append({"role": "user", "content": text})
                size_state["chars"] += _message_size(messages[-1])
                await self._record(ctx, {"role": "user", "content": text})
                yield await self._emit(
                    ctx, {"type": EventType.USER_INJECTED, "step": step,
                          "text": text})

        if self.context_budget is not None or (
                self.context_token_threshold and cfg.context_window):
            # Safety valve for long runs: without it a 35-step run of fat tool
            # results grows past the model's context window mid-run and the
            # API hard-fails the run. Old tool outputs shrink head+tail; the
            # structure (tool_call ↔ tool_result pairing) stays valid. The
            # budget is token-calibrated when the host declared a window.
            # When shrinking cannot reach the budget, whole old exchanges are
            # dropped (see _trim_context) — and if even that is not enough,
            # the run logs once and still sends, rather than silently
            # overflowing the window.
            budget = self._effective_char_budget(cfg, size_state)
            if budget is not None:
                size_state["chars"] = _trim_context(messages, budget=budget)
                if (size_state["chars"] > budget
                        and not size_state.get("over_budget_logged")):
                    size_state["over_budget_logged"] = True
                    log.warning(
                        "context still over budget after trimming and "
                        "dropping old exchanges (%d > %d chars); the next "
                        "model call may be rejected if the payload exceeds "
                        "the model's window",
                        size_state["chars"], budget)
        ctx_chars = size_state["chars"]

        eff_tools = None if wrap_up else (tools_spec or None)
        tool_choice = "none" if wrap_up else "auto"
        # One retry when the model returns a completely empty response (no
        # text, no tool calls) — a transient gateway hiccup that previously
        # ended the run "successfully" with an empty final answer.
        for empty_attempt in range(2):
            result = None
            try:
                stream_fn = getattr(self.transport, "complete_stream", None)
                if cfg.stream and stream_fn is not None:
                    stream = stream_fn(
                        client, base_url=cfg.base_url, api_key=cfg.api_key,
                        model=cfg.model, messages=messages,
                        tools=eff_tools, tool_choice=tool_choice,
                        max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                        attempts=cfg.attempts, sleep_429=cfg.sleep_429,
                        sleep_err=cfg.sleep_err,
                        include_reasoning=cfg.reasoning_replay,
                        reasoning_scope=cfg.reasoning_scope,
                        extra_body=cfg.extra_body,
                        extra_headers=cfg.default_headers)
                    # aclosing: a cancellation (or a mid-stream failure) must
                    # close the transport stream — and drop the underlying
                    # HTTP response — instead of leaving it to the GC.
                    async with contextlib.aclosing(stream):
                        async for part in stream:
                            if _stop_triggered(stop):
                                # Cancelled mid-generation: the partial
                                # deltas were display-only; no assistant turn
                                # is recorded, so memory replay has nothing
                                # dangling to reconcile.
                                stats.status = "cancelled"
                                state["finished"] = True
                                yield await self._emit(
                                    ctx, {"type": EventType.CANCELLED,
                                          "step": step})
                                return
                            if part.get("delta"):
                                yield await self._emit(
                                    ctx, {"type": EventType.ASSISTANT_DELTA,
                                          "text": part["delta"]})
                            elif "result" in part:
                                result = part["result"]
                else:
                    result = await self.transport.complete(
                        client, base_url=cfg.base_url, api_key=cfg.api_key,
                        model=cfg.model, messages=messages,
                        tools=eff_tools, tool_choice=tool_choice,
                        max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                        attempts=cfg.attempts, sleep_429=cfg.sleep_429,
                        sleep_err=cfg.sleep_err,
                        include_reasoning=cfg.reasoning_replay,
                        reasoning_scope=cfg.reasoning_scope,
                        extra_body=cfg.extra_body,
                        extra_headers=cfg.default_headers,
                    )
            except httpx.HTTPStatusError as exc:
                log.warning("agent model error: %s", exc)
                message = f"模型请求失败（{exc.response.status_code}）"
                hint = getattr(exc, "lithe_hint", None)
                if hint:
                    # Annotated by the transport: a 400 that may stem from
                    # host-supplied extra_body fields the endpoint rejects.
                    message = f"{message}。{hint}"
                yield await self._emit(
                    ctx, {"type": EventType.ERROR, "code": exc.response.status_code,
                          "message": message})
                stats.status = "failed"
                state["finished"] = True
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("agent model error: %s", exc)
                yield await self._emit(
                    ctx, {"type": EventType.ERROR,
                          "message": "模型请求出错，请稍后重试。"})
                stats.status = "failed"
                state["finished"] = True
                return
            if result is None:
                yield await self._emit(
                    ctx, {"type": EventType.ERROR,
                          "message": "模型流式响应未返回结果，请稍后重试。"})
                stats.status = "failed"
                state["finished"] = True
                return

            content = result.get("content") or ""
            tool_calls = result.get("tool_calls") or []
            finish_reason = result.get("finish_reason")
            reasoning = result.get("reasoning") or []
            usage = result.get("usage") or {}
            prompt = _as_int(usage.get("prompt_tokens"))
            completion = _as_int(usage.get("completion_tokens"))
            cached = _as_int(usage.get("cached_tokens"))
            reason_tok = _as_int(usage.get("reasoning_tokens"))
            total = _as_int(usage.get("total_tokens")) or (prompt + completion)
            call_cost = _call_cost(cfg, usage)
            stats.prompt_tokens += prompt
            stats.completion_tokens += completion
            stats.cached_tokens += cached
            stats.reasoning_tokens += reason_tok
            stats.total_tokens += total
            stats.total_cost += call_cost
            stats.context_tokens = prompt  # measured input length of this call
            if prompt > 0:
                # Calibrate chars-per-token from the measured request: this
                # ratio converts the token-window-based trim budget into the
                # char unit _trim_context works in (updated every call, so it
                # tracks the conversation's actual CJK/code mix).
                size_state["cpt"] = max(1.0, size_state["chars"]) / prompt

            if (empty_attempt == 0 and not wrap_up and not tool_calls
                    and not content.strip() and finish_reason != "length"):
                # Empty, untruncated, toolless: almost certainly a gateway
                # hiccup — pay one extra call before trusting it.
                log.warning("model returned an empty response (step %d); "
                            "retrying once", step)
                state["empty_retried"] = True
                continue
            break

        truncated = finish_reason == "length"
        if truncated:
            # max_tokens cut the generation mid-flight: mark that for the
            # model and the user instead of silently treating a partial
            # answer (or a half-written tool_call) as complete.
            notice = "（输出已达到 max_tokens 上限，内容被截断。）"
            content = f"{content.rstrip()}\n\n{notice}" if content.strip() else notice

        messages.append(_normalize_assistant(
            {"content": content, "tool_calls": tool_calls or None,
             "reasoning": reasoning or None}))
        size_state["chars"] += _message_size(messages[-1])
        # reasoning rides the record only when present: the persisted-turn
        # contract stays byte-identical for reasoning-less models.
        rec: dict[str, Any] = {"role": "assistant", "content": content,
                               "tool_calls": tool_calls or None}
        if reasoning:
            rec["reasoning"] = reasoning
        await self._record(ctx, rec)
        reasoning_digest = reasoning_summary_text(reasoning)
        if reasoning_digest:
            # Display-only digest of the model's reasoning (gateway-sent
            # summaries). The encrypted items themselves ride the message
            # records for pass-back; losing this event loses no state.
            yield await self._emit(ctx, {"type": EventType.REASONING,
                                         "step": step, "text": reasoning_digest})
        if content:
            yield await self._emit(ctx, {"type": EventType.ASSISTANT, "text": content})

        window = cfg.context_window
        percent = round(100.0 * prompt / window, 1) if window and prompt else None
        yield await self._emit(ctx, {
            "type": EventType.USAGE, "step": step,
            "prompt_tokens": prompt, "completion_tokens": completion,
            "cached_tokens": cached, "reasoning_tokens": reason_tok,
            "total_tokens": total, "cost": call_cost,
            "context_tokens": prompt, "context_chars": ctx_chars,
            "context_window": window, "context_percent": percent,
            "finish_reason": finish_reason,
        })

        if not tool_calls:
            if not content.strip() and not truncated:
                # Still empty after the one retry: end with a distinguishable
                # status instead of "successfully" reporting empty text.
                stats.status = "empty_response"
                stats.final_text = ""
                state["finished"] = True
                yield await self._emit(
                    ctx, {"type": EventType.ERROR,
                          "message": "模型返回了空响应（无文本、无工具调用），"
                                     "请稍后重试或换一种问法。"})
                return
            stats.final_text = content
            state["finished"] = True
            return

        if self._over_budget(stats):
            # The call that just accounted crossed the budget: execute no
            # further tool calls (their results would only buy another model
            # call we refuse to make). The run ends with the model's own text
            # or the budget notice; its unanswered tool_calls reconcile in
            # memory replay exactly like a cancellation's.
            stats.status = "budget_exceeded"
            stats.final_text = content or _BUDGET_FALLBACK
            if not content:
                yield await self._synth_end(ctx, messages, size_state,
                                            stats.final_text)
            state["finished"] = True
            return

        if wrap_up:
            # Budget exhausted and the model STILL tried tool_calls on a
            # no-tools call: never execute them — the run ends with whatever
            # text the model produced, or the fallback notice.
            stats.final_text = content or _MAX_STEPS_FALLBACK
            if not content:
                yield await self._synth_end(ctx, messages, size_state,
                                            stats.final_text)
            state["finished"] = True
            return

        if _stop_triggered(stop):
            # The assistant turn is recorded; its tool calls stay unanswered,
            # which memory replay reconciles (orphan tool_calls are dropped).
            stats.status = "cancelled"
            state["finished"] = True
            yield await self._emit(ctx, {"type": EventType.CANCELLED, "step": step})
            return

        parsed: list[tuple[dict, dict, str | None]] = []
        for tc in tool_calls:
            if not (isinstance(tc, dict) and isinstance(tc.get("function"), dict)):
                continue
            args, err = _parse_args(tc)
            name = tc["function"].get("name")
            if not (isinstance(name, str) and name.strip()):
                # A function dict without a usable name cannot be dispatched;
                # answer it as a failed call (the id still needs a tool
                # result) instead of crashing the run with a KeyError.
                err = err or "缺少可用的 function.name"
            parsed.append((tc, args, err))

        # Announce every tool call before anything executes: a slow tool must
        # not hide what the model just asked for — the frontend shows the
        # pending calls while they run. Result events still follow in model
        # order after execution, so turn reconstruction is unchanged.
        for tc, args, _err in parsed:
            yield await self._emit(
                ctx, {"type": EventType.TOOL_CALL, "step": step,
                      "id": tc.get("id"),
                      "name": tc["function"].get("name"),
                      "args": args})

        # Execution policy: calls run in the model's issued order, grouped so
        # that consecutive READ tools execute in parallel (the common
        # read-everything-first pattern), while every WRITE/META tool runs
        # alone — two writes to the same target can never race, and a read
        # never observes a half-applied write from its own step.
        def _batchable(name: str) -> bool:
            ts = self.registry.spec(name)
            return ts is None or ts.category == ToolCategory.READ

        groups: list[tuple[bool, list[int]]] = []
        for idx, (tc, _args, err) in enumerate(parsed):
            if err is not None:
                continue
            batch = _batchable(tc["function"].get("name"))
            if groups and groups[-1][0] == batch:
                groups[-1][1].append(idx)
            else:
                groups.append((batch, [idx]))

        results: dict[int, ToolResult] = {}

        async def _dispatch_at(i: int) -> ToolResult:
            tc, args, _err = parsed[i]
            name = tc["function"].get("name")
            counts = repeat_state["counts"]
            bumped = False
            if not _batchable(name):
                # A WRITE/META tool may change state (even a failed one may
                # have mutated something before erroring): it invalidates
                # "same call → same result" reasoning for every OTHER
                # signature, so the epoch bump restarts their counters.
                repeat_state["epoch"] += 1
                bumped = True
            res = await self.registry.dispatch(name, args, ctx)
            if self.repeat_call_limit is not None:
                # Stuck-model guard: the identical (tool, args) call issued
                # yet again — with no mutating tool in between — gets an
                # inline nudge in its tool result, feeding self-correction
                # instead of burning steps on retries that cannot return
                # anything different.
                sig = name + "\n" + json.dumps(args, sort_keys=True,
                                               ensure_ascii=False, default=str)
                prev, epoch = counts.get(sig, (0, repeat_state["epoch"]))
                if bumped or epoch == repeat_state["epoch"]:
                    # `bumped`: a mutating tool's own retries still count —
                    # re-sending an identical write is the stuck loop this
                    # guard exists for, not a fresh observation.
                    n = prev + 1
                else:
                    n = 1  # state changed since the last identical call
                counts[sig] = (n, repeat_state["epoch"])
                if n > self.repeat_call_limit:
                    hint = (f"提示：这是连续第 {n} 次以完全相同的参数调用 "
                            f"{name}。重复同样的调用大概率得到相同结果；请修改"
                            f"参数、更换工具，或基于已有结果换一种做法。")
                    res.content = f"{res.content}\n\n{hint}" if res.content else hint
            return res

        for parallel, idxs in groups:
            if parallel and len(idxs) > 1:
                for i, res in zip(idxs, await asyncio.gather(
                        *[_dispatch_at(i) for i in idxs]), strict=True):
                    results[i] = res
            else:
                for i in idxs:
                    results[i] = await _dispatch_at(i)

        for idx, (tc, _args, err) in enumerate(parsed):
            if err is None:
                continue
            name = (tc.get("function") or {}).get("name") or ""
            if truncated:
                # finish_reason=="length": the arguments JSON was cut off by
                # the token cap, not mis-written by the model. Tell it so —
                # "invalid JSON" alone invites a byte-identical retry.
                msg = (f"工具 {name} 的 arguments 因输出达到 max_tokens 上限被截断，"
                       f"不是完整 JSON。请重新发起该调用，必要时精简参数或"
                       f"减少同一步的工具调用数量。")
            else:
                msg = (f"工具 {name} 的 arguments 不是合法 JSON 对象：{err}。"
                       f"请修正后重新调用。")
            results[idx] = ToolResult(False, "参数错误", msg)
        for idx, (tc, _args, _err) in enumerate(parsed):
            name = (tc.get("function") or {}).get("name") or ""
            tcid = tc.get("id")
            res = results[idx]
            tr: Event = {"type": EventType.TOOL_RESULT, "step": step, "id": tcid,
                         "name": name, "ok": res.ok, "summary": res.summary}
            if not res.ok:
                tr["error"] = res.content
            yield await self._emit(ctx, tr)
            for ui in res.ui:
                yield await self._emit(ctx, ui)
            await self._record(ctx, {
                "role": "tool", "content": res.content, "tool_name": name,
                "tool_call_id": tcid,
                "meta": {"ok": res.ok, "summary": res.summary, "ui": res.ui},
            })
            messages.append({"role": "tool", "tool_call_id": tcid, "content": res.content})
            size_state["chars"] += _message_size(messages[-1])
        return
