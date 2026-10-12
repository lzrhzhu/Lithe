"""The ReAct runtime: drives the model ↔ tool loop and emits events.

One package, four modules:

- :mod:`lithe.runtime.config` — :class:`LLMConfig` (endpoint + policy) and
  :class:`RunStats` (per-run accounting), pure data.
- :mod:`lithe.runtime.loop` — :class:`AgentRuntime`, the kernel's only
  active component.
- :mod:`lithe.runtime.protocol` — tool-call argument parsing, the
  cancellation-handle adapter, steering-inbox drain, assistant-message
  normalization.
- :mod:`lithe.runtime.accounting` — usage coercion and cost computation.

This package was ``runtime.py`` + ``runtime_helpers.py`` +
``runtime_accounting.py`` until the 0.1.10 reorganization; everything those
modules ever exported (including the underscore aliases tests pin) is still
importable from ``lithe.runtime``.
"""
from __future__ import annotations

import httpx  # noqa: F401 — part of the pre-package module's namespace

from lithe.context import AgentContext  # noqa: F401
from lithe.events import Event, EventSink, EventType  # noqa: F401
from lithe.memory import reasoning_summary_text  # noqa: F401
from lithe.modes import ToolCategory  # noqa: F401
from lithe.tools import ToolRegistry, ToolResult  # noqa: F401
from lithe.transports import LLMTransport, make_transport  # noqa: F401
from lithe.windowing import (  # noqa: F401
    DEFAULT_CONTEXT_BUDGET, _OMITTED_TURNS_NOTE, _context_size, _message_size,
    _trim_context,
)
from lithe.runtime.accounting import (  # noqa: F401
    as_int, call_cost, computed_cost, gateway_cost,
)
from lithe.runtime.config import LLMConfig, RunStats  # noqa: F401
from lithe.runtime.loop import (  # noqa: F401
    DEFAULT_REPEAT_CALL_LIMIT, AgentRuntime, _BUDGET_FALLBACK,
    _MAX_STEPS_FALLBACK, _exc_detail,
)
from lithe.runtime.protocol import (  # noqa: F401
    drain_inbox, normalize_assistant, parse_args, stop_triggered,
)

# Underscore aliases the old flat module carried (imported under these
# private names; tests pin them).
_as_int = as_int                                  # noqa: F401
_call_cost = call_cost                            # noqa: F401
_computed_cost = computed_cost                    # noqa: F401
_gateway_cost = gateway_cost                      # noqa: F401
_drain_inbox = drain_inbox                        # noqa: F401
_normalize_assistant = normalize_assistant        # noqa: F401
_parse_args = parse_args                          # noqa: F401
_stop_triggered = stop_triggered                  # noqa: F401
