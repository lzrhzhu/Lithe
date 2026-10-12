"""LLM transports: protocol-agnostic model calls.

The runtime drives the model via an :class:`LLMTransport`
(:mod:`lithe.transports.base`), which sends one request and returns a
*unified* shape ``{content, tool_calls, usage}`` — so the runtime never
branches on chat-completions vs responses wire formats.

Three built-in transports, one module each:

- :class:`ChatCompletionsTransport` (:mod:`lithe.transports.openai_chat`) —
  OpenAI ``/chat/completions`` (default, backward compatible; the module
  also carries the one-shot :func:`~lithe.transports.openai_chat.chat_completion`
  client used by the perception bundles).
- :class:`ResponsesTransport` (:mod:`lithe.transports.openai_responses`) —
  OpenAI ``Responses API`` (``/responses``), the 2025 standard that
  supersedes chat completions.
- :class:`MessagesTransport` (:mod:`lithe.transports.anthropic`) — Anthropic
  ``Messages API`` (``/messages``), with thinking-block replay and
  prompt-cache breakpoints.

A host picks one via ``LLMConfig(transport="chat" | "responses" |
"messages")`` or passes a custom :class:`LLMTransport`. ``lithe`` stays
storage-/host-agnostic — these are standard wire protocols (whoever runs the
endpoint), not host specifics: provider endpoints map onto protocols via the
``providers`` bundle's preset ``transport`` field, which is the only place
vendor knowledge enters.

This package was a single ``lithe/transports.py`` module until 0.1.9; every
name importable from it back then is still importable from ``lithe.transports``
(the per-protocol modules carry the implementation, this ``__init__`` the
compatibility surface and the factory).
"""
from __future__ import annotations

from lithe.transports.anthropic import (  # noqa: F401
    _ANTHROPIC_VERSION, _EFFORT_THINKING_BUDGETS, _EPHEMERAL,
    _MESSAGES_DEFAULT_MAX_TOKENS, _STOP_REASON_MAP, _blocks_to_result,
    _image_source, _messages_to_anthropic, _parse_anthropic, _post_messages,
    _tool_choice_to_anthropic, _tool_input, _tools_to_anthropic, _user_blocks,
    MessagesTransport, anthropic_headers,
)
from lithe.transports.base import (  # noqa: F401
    LLMTransport, TransportResult, _flatten_text, _merge_extra, _to_int,
    norm_usage,
)
from lithe.transports.openai_chat import (  # noqa: F401
    ChatCompletionsTransport, _THINK_BLOCK_RE, _chat_outbound,
    _chat_reasoning_items,
)
from lithe.transports.openai_responses import (  # noqa: F401
    ResponsesTransport, _content_blocks, _convert_tool_choice, _convert_tools,
    _messages_to_input, _parse_output, _post_responses,
)

_BUILTIN = {
    "chat": ChatCompletionsTransport,
    "responses": ResponsesTransport,
    "messages": MessagesTransport,
}


def make_transport(spec: str | LLMTransport) -> LLMTransport:
    """Resolve a transport spec: ``"chat"`` / ``"responses"`` / ``"messages"``
    → built-in instance; an :class:`LLMTransport` instance → returned as-is."""
    if isinstance(spec, str):
        cls = _BUILTIN.get(spec)
        if cls is None:
            raise ValueError(
                f"unknown transport: {spec!r} "
                f"(use {' or '.join(repr(k) for k in _BUILTIN)})"
            )
        return cls()  # type: ignore[return-value]
    return spec


__all__ = [
    "ChatCompletionsTransport",
    "LLMTransport",
    "MessagesTransport",
    "ResponsesTransport",
    "TransportResult",
    "make_transport",
]
