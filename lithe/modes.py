"""Agent modes and tool categories.

A *mode* selects which tool categories the runtime exposes to the model. Tools
are registered with a category (read / write / meta); the runtime filters by
mode so e.g. a read-only "anchored" discussion mode can never call a write
tool, regardless of what the host registered.

Hosts may register their own modes (e.g. a supervised mode exposing read +
meta only) via :func:`register_mode`. Unknown modes raise :class:`ValueError`
— a typo'd mode name silently degrading to read-only was a footgun.
"""
from __future__ import annotations

from enum import Enum
from collections.abc import Iterable


class ToolCategory(str, Enum):
    READ = "read"
    WRITE = "write"
    # Tools that drive other modifications indirectly (delegate / propose_edit).
    META = "meta"


class AgentMode(str, Enum):
    # Full autonomy: read + write + meta tools.
    AUTONOMOUS = "autonomous"
    # Collaborative: read-only + a propose/meta tool (never writes directly).
    ANCHORED = "anchored"


# Which categories each mode admits. Keyed by the mode's plain string value so
# both ``AgentMode.AUTONOMOUS`` and ``"autonomous"`` (and host-registered
# custom names) resolve through one table.
_MODE_CATEGORIES: dict[str, frozenset[ToolCategory]] = {
    AgentMode.AUTONOMOUS.value: frozenset(
        {ToolCategory.READ, ToolCategory.WRITE, ToolCategory.META}),
    AgentMode.ANCHORED.value: frozenset({ToolCategory.READ, ToolCategory.META}),
}


def register_mode(name: str, categories: Iterable[ToolCategory | str]) -> None:
    """Register (or replace) a host-defined agent mode.

    ``categories`` accepts :class:`ToolCategory` members or their string
    values (``"read"`` / ``"write"`` / ``"meta"``). Registering an existing
    name (including the built-ins) overwrites it — a host tuning ``anchored``
    to also admit a specific meta tool does so without kernel changes.
    """
    if not name or not isinstance(name, str):
        raise ValueError("mode name must be a non-empty string")
    cats = frozenset(ToolCategory(c) if isinstance(c, str) else c
                     for c in categories)
    if not cats:
        raise ValueError(f"mode {name!r} must admit at least one category")
    _MODE_CATEGORIES[name] = cats


def categories_for(mode: AgentMode | str) -> frozenset[ToolCategory]:
    """Resolve a mode to its admitted tool categories.

    Raises :class:`ValueError` for unknown modes — fail fast at
    ``specs_for_mode`` time instead of silently running read-only.
    """
    key = mode.value if isinstance(mode, AgentMode) else mode
    cats = _MODE_CATEGORIES.get(key)
    if cats is None:
        raise ValueError(
            f"unknown agent mode: {mode!r} (known: {sorted(_MODE_CATEGORIES)})")
    return cats
