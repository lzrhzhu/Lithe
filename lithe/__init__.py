"""lithe — a reusable, storage-free ReAct agent kernel.

Hosts (e.g. a thesis-writing application) plug in their own tools, prompt,
domain model and persistence. The kernel knows nothing about how runs are
stored, or even whether they are: it emits events (which a host may sink to a
DB, a file, or drop) and reads prior turns through an optional memory provider
(added in a later step).

Layout:
  - llm        : OpenAI-compatible chat client (retry/backoff/jitter)
  - events     : Event vocabulary, EventSink, SSE serialization
  - context    : AgentContext (typed, mapping-compatible)
  - modes      : AgentMode / ToolCategory + category filtering
  - actions    : Action model + UndoEngine (pure logic, no storage)
  - tools      : ToolSpec / ToolResult / ToolRegistry (dispatch + undo wiring)
"""
# Single source of truth for the package version; pyproject reads it via
# [tool.setuptools.dynamic] (statically, without importing this module).
__version__ = "0.1.5"

from lithe.actions import (  # noqa: F401
    Action, Reverter, UndoEngine, UndoReport,
)
from lithe.context import AgentContext  # noqa: F401
from lithe.events import Event, EventSink, EventType, to_sse  # noqa: F401
from lithe.llm import (  # noqa: F401
    bearer_headers, chat_completion, first_content, strip_think,
)
from lithe.memory import (  # noqa: F401
    DEFAULT_REPLAY_LIMIT, DEFAULT_TOOL_RESULT_CAP, MemoryProvider,
    recap_text, replay_messages, reasoning_summary_text, run_timeline,
    truncate_tool_result, window_with_recap,
)
from lithe.modes import (  # noqa: F401
    AgentMode, ToolCategory, categories_for, register_mode,
)
from lithe.runtime import (  # noqa: F401
    DEFAULT_CONTEXT_BUDGET, DEFAULT_REPEAT_CALL_LIMIT, AgentRuntime,
    LLMConfig, RunStats,
)
from lithe.tools import (  # noqa: F401
    ToolHandler, ToolMiddleware, ToolRegistry, ToolResult, ToolSpec,
    ToolTransform,
)
from lithe.transports import (  # noqa: F401
    ChatCompletionsTransport, LLMTransport, ResponsesTransport, make_transport,
)

__all__ = [
    "Action", "AgentContext", "AgentMode", "AgentRuntime",
    "ChatCompletionsTransport",
    "DEFAULT_CONTEXT_BUDGET", "DEFAULT_REPEAT_CALL_LIMIT",
    "DEFAULT_REPLAY_LIMIT", "DEFAULT_TOOL_RESULT_CAP",
    "Event", "EventSink", "EventType", "LLMConfig", "LLMTransport",
    "MemoryProvider",
    "ResponsesTransport",
    "Reverter", "RunStats",
    "ToolCategory", "ToolHandler", "ToolMiddleware", "ToolTransform",
    "ToolRegistry", "ToolResult", "ToolSpec", "UndoEngine", "UndoReport",
    "bearer_headers", "categories_for", "chat_completion", "first_content",
    "make_transport", "register_mode",
    "recap_text", "replay_messages", "reasoning_summary_text",
    "run_timeline", "strip_think",
    "to_sse", "truncate_tool_result", "window_with_recap",
]
