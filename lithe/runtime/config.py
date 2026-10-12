"""Runtime configuration and accounting types: :class:`LLMConfig` and
:class:`RunStats`.

``LLMConfig`` is the host-declared endpoint + policy for every model call
(the same object drives the main loop and the perception calls the
images/documents bundles make); ``RunStats`` is the mutable per-run
accounting the host reads to build its ``done`` event. Pure data — no I/O,
no loop machinery (that is :mod:`lithe.runtime.loop`).
"""
from __future__ import annotations

from dataclasses import dataclass

from lithe.transports import LLMTransport


@dataclass
class LLMConfig:
    """Endpoint + retry policy for the runtime's model calls.

    ``transport`` selects the wire protocol: ``"chat"`` (OpenAI
    chat-completions, default), ``"responses"`` (OpenAI Responses API) or
    ``"messages"`` (Anthropic Messages API). A host may also pass a custom
    :class:`~lithe.transports.LLMTransport`
    instance. ``stream=True`` asks a streaming-capable transport for token
    deltas (the runtime then emits ``assistant_delta`` events ahead of the
    final ``assistant`` one; unsupported transports fall back silently).
    ``context_window`` is the model's token window (host knowledge — the
    kernel cannot know it); when set, usage events and :class:`RunStats`
    report ``context_percent`` so a frontend can show how full the
    conversation is. ``temperature`` / ``max_tokens`` are plain sampling
    parameters forwarded on every model call (``None`` leaves them unset).
    ``reasoning_replay`` (default True) asks a reasoning-capable transport to
    pass reasoning items back (Responses / Messages protocols); set False
    for gateways that reject reasoning input items. ``reasoning_scope`` selects which turns
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

    ``reasoning_effort`` is the vendor reasoning-intensity knob
    (``"minimal"`` / ``"low"`` / ``"medium"`` / ``"high"`` — model
    dependent). The value passes through *verbatim*: the chat transport
    sends it as OpenAI-style top-level ``reasoning_effort``, the Responses
    transport as ``reasoning: {"effort": ...}``, deep-merged with any
    ``extra_body["reasoning"]`` keys (internal wins per key, so a host can
    still add ``max_tokens``/``exclude`` alongside); the Messages transport
    maps levels onto thinking budgets (low/medium/high → 2k/8k/16k, with
    the ``temperature``/``max_tokens`` shims the protocol requires). ``None``
    sends nothing. Which values a model actually accepts is the endpoint's
    call — an unsupported value surfaces through the 400 diagnostics instead
    of silent mangling.

    ``document_format`` is the document-content-block dialect this endpoint
    family accepts (``"inline-file"`` / ``"files-api"`` / ``"none"`` —
    see ``lithe.bundles.documents``). The tool loop itself ignores it:
    documents never enter the main conversation. It is endpoint knowledge
    the documents bundle reads off its perception ``LLMConfig``, so a
    provider preset can contribute a default (provider type constrains the
    format family) while a hand-written profile field or an explicit
    registration argument overrides it — what matters is what the gateway
    accepts, not what the URL looks like.
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
    reasoning_effort: str | None = None
    extra_body: dict | None = None
    default_headers: dict | None = None
    pricing: dict[str, float] | None = None
    document_format: str | None = None

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
    ``duration_s`` is the wall-clock length of the run, stamped when it
    ends (the DONE event carries it too). ``error`` carries the diagnostic
    of a failed call (exception class + message, capped) so hosts, stores
    and delegation summaries can show *why* a run failed — not only that
    it did; ``None`` on every non-failed path.
    """
    final_text: str = ""
    status: str = "done"
    error: str | None = None
    last_step: int = 0
    duration_s: float = 0.0
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
