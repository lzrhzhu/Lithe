"""Reversible mutations and the undo engine.

A write tool records an :class:`Action` (what it changed: kind/target/old/new)
and registers a :class:`Reverter` that knows how to reverse that kind.
:class:`UndoEngine` replays the reversals newest-first. None of this touches
storage: the engine takes a list of actions (from this run's in-memory state,
or read back by a host from wherever it persisted them), so undo works with or
without a database.

Reverters may be sync or async (``async def`` / coroutine-returning) — an
async reverter can roll back through remote APIs; :meth:`UndoEngine.undo`
awaits whatever the reverter returns.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from lithe.context import AgentContext

# Action statuses that mean "the mutation is currently in effect" and so should
# be reverted. Pending/proposed changes that were never applied are skipped.
_REVERTIBLE = frozenset({"applied", "approved"})


@dataclass
class Action:
    """One recorded mutation produced by a write tool.

    ``id`` is opaque and host-assigned (e.g. a DB row id); the kernel only uses
    it for ordering/reporting. ``old_value``/``new_value`` are domain-shaped and
    interpreted by the matching reverter.
    """
    kind: str
    target: str
    old_value: Any = None
    new_value: Any = None
    status: str = "applied"
    id: Any = None
    subagent: str | None = None


# A reverter reverses one action kind: ``fn(action, ctx)`` returning None (sync)
# or an awaitable (async, e.g. rolling back through a remote API). It is
# best-effort: a partial failure (e.g. a file already gone) must not abort the
# rest of an undo run, so each reverter owns its own error handling.
Reverter = Callable[[Action, AgentContext], Any]


@dataclass
class UndoReport:
    ok: bool = True
    reverted: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


class UndoEngine:
    """Revert a list of actions newest-first via registered reverters.

    Pure logic: no storage. Unknown kinds and non-revertible statuses are
    skipped; a reverter that raises is recorded as an error (not fatal) so the
    rest of the run still reverts.
    """

    def __init__(self, reverters: dict[str, Reverter] | None = None):
        self._reverters: dict[str, Reverter] = dict(reverters or {})

    def register(self, kind: str, reverter: Reverter) -> None:
        self._reverters[kind] = reverter

    @property
    def reverters(self) -> dict[str, Reverter]:
        return dict(self._reverters)

    async def undo(self, actions: list[Action], ctx: AgentContext) -> UndoReport:
        """Revert *actions* newest-first, awaiting async reverters.

        Pure logic: no storage. Unknown kinds and non-revertible statuses are
        skipped; a reverter that raises is recorded as an error (not fatal) so
        the rest of the run still reverts. A reverter's awaitable result is
        awaited — an error *inside* the awaitable counts the same as a sync
        raise.
        """
        report = UndoReport()
        for action in reversed(actions):
            if action.status not in _REVERTIBLE:
                report.skipped += 1
                continue
            fn = self._reverters.get(action.kind)
            if fn is None:
                report.skipped += 1
                continue
            try:
                result = fn(action, ctx)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{action.kind} {action.target}: {exc}")
                report.ok = False
                continue
            report.reverted += 1
        return report
