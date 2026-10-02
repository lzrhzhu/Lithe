"""Event protocol for the agent runtime.

The kernel emits a small set of *generic* events describing the ReAct loop's
progress (run lifecycle, assistant steps, tool calls/results). Domain-specific
UI events (file changes, guidance proposals, library additions, ...) are NOT
defined here: they travel inside a tool result's ``ui`` list and the kernel
forwards them verbatim, so the frontend renders them with host-supplied
components. This keeps the kernel's event vocabulary minimal and stable.

Persistence is just another event consumer: a host wires an :class:`EventSink`
to write runs to a DB, a JSONL log, or nowhere at all.
"""
from __future__ import annotations

import json
from typing import Any, Protocol


class EventType:
    """Generic event type tags emitted by the runtime.

    Hosts may emit additional tags via tool-result ``ui`` payloads; those are
    passed through untouched and need not (should not) be listed here.
    """
    RUN_START = "run_start"
    STEP = "step"
    # Per-model-call usage snapshot for live display: prompt/completion/total
    # tokens, that call's cost, and context fullness (measured context_tokens
    # when the API reports them, context_chars always, context_window and
    # context_percent when the host configured LLMConfig.context_window).
    USAGE = "usage"
    # Streaming text fragment (only when LLMConfig.stream=True and the
    # transport supports it). A final ASSISTANT event with the full text still
    # follows, so frontends that ignore deltas lose nothing.
    ASSISTANT_DELTA = "assistant_delta"
    # Human-readable digest of the model's reasoning for one step (the
    # reasoning items' summary text, when the gateway sends any). Emitted
    # before the step's ASSISTANT event; frontends that ignore it lose nothing
    # — the encrypted reasoning itself rides the message records, not events.
    REASONING = "reasoning"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DONE = "done"
    ERROR = "error"
    # The run was cancelled through its ``stop`` handle before finishing.
    CANCELLED = "cancelled"
    # A host-queued user message injected at a step boundary while the run
    # was in flight (the steering channel: the user typed while the agent
    # worked). The text also enters `messages` and the record channel like
    # any user turn, so replay and persistence see it in order.
    USER_INJECTED = "user_injected"


# An event is an ordinary dict carrying a ``type`` tag plus arbitrary fields.
Event = dict[str, Any]


class EventSink(Protocol):
    """Observer of the runtime's event stream.

    Two channels keep *display* and *persistence* concerns apart:

    - ``on_event`` receives the lightweight display events the runtime yields
      (step / assistant text / tool_call / tool_result summary / done / error).
      A host forwards these to the frontend (SSE) or uses them for metrics.
    - ``on_record`` receives the full, raw message records (an assistant turn
      with its ``tool_calls``, or a tool result with its ``content`` / ``meta``)
      so a host can persist the complete conversation — without the kernel ever
      touching a database.

    Both are entirely optional: a host that only wants the live stream attaches
    no sinks, and the kernel stores nothing. Multiple sinks may be attached.
    """

    async def on_event(self, ctx: Any, event: Event) -> None: ...

    async def on_record(self, ctx: Any, record: dict) -> None: ...


def to_sse(event: Event) -> str:
    """Serialize one event as a Server-Sent-Events ``data:`` line.

    Framework-agnostic: the host's web layer calls this when forwarding events
    over an SSE/streaming response. Non-ASCII is preserved (``ensure_ascii`` is
    False) so e.g. Chinese content isn't escaped, and non-JSON values (a
    ``datetime`` inside a tool's ``ui`` payload, a Path, ...) fall back to
    ``str()`` instead of raising TypeError out of the host's SSE layer.
    """
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
