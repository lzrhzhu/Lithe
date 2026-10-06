"""Storage protocol + record types for the agent kernel.

The core engine is storage-free by design; persistence is a *host concern*.
This module defines the contract a host's conversation store implements so the
host-adapter bundle (:mod:`lithe.bundles.host`) can record and replay runs
without knowing *where* or *how* they are stored.

The default backend lives in :mod:`lithe.bundles.store.jsonl` — a
zero-database JSONL store. Hosts wanting indexed queries / compaction implement
this Protocol against their own DB (Postgres, Redis, an event log, ...).

Read methods return plain ``dict`` rows in the shape the kernel's
:mod:`lithe.memory` transforms already consume
(``role`` / ``content`` / ``tool_calls`` / ``tool_call_id`` / ``meta`` /
``run_id`` / ``kind`` / ``target`` / ``status``), so replay stays a pure
transform over rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from collections.abc import Collection


@dataclass
class StoredMessage:
    """One recorded message of a run (user / assistant / tool)."""

    role: str
    content: str | None
    run_id: str
    user_id: str
    tool_calls: Any = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    meta: Any = None
    subagent: str | None = None
    id: int | None = None
    ts: float | None = None


@dataclass
class StoredAction:
    """One recorded mutation a write tool performed (the unit of undo)."""

    run_id: str
    user_id: str
    kind: str
    target: str
    old_value: Any = None
    new_value: Any = None
    status: str = "applied"
    subagent: str | None = None
    id: int | None = None
    ts: float | None = None


@dataclass
class StoredRun:
    """One agent run (one turn of a conversation).

    ``run_id`` / ``user_id`` / ``task`` / ``status`` / ``final`` / ``error`` /
    ``steps`` / ``cost`` plus token fields. ``error`` is the diagnostic of a
    failed run (exception class + message from the runtime, capped) —
    ``None`` on every other path, so a post-mortem over stored runs can say
    *why* something failed without the process's stderr log.

    ``created_at`` / ``finished_at`` are unix epoch floats (the store stamps
    them automatically; legacy rows read back as ``None``). The token fields
    are the run's cumulative usage as reported by the final ``done`` state —
    ``None`` when the writer predates token persistence or the run never
    finished, so a summary can distinguish "zero tokens" from "unknown".
    """

    run_id: str
    user_id: str
    task: str
    status: str = "running"
    final: str | None = None
    error: str | None = None
    conversation_id: int | None = None
    model: str | None = None
    steps: int = 0
    cost: float = 0.0
    created_at: float | None = None
    finished_at: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None


class RunStore(Protocol):
    """Persistence contract for runs / messages / actions / conversations.

    Implementations decide the medium. The default :class:`JsonlRunStore` writes
    one JSONL file per run plus a conversation index — no database. All reads
    are scoped by ``user_id`` so a multi-tenant host never crosses users.
    """

    # -- runs -----------------------------------------------------------------
    def create_run(
        self,
        run_id: str,
        user_id: str,
        task: str,
        *,
        conversation_id: int | None = None,
        model: str | None = None,
        created_at: float | None = None,
    ) -> None: ...

    def finish_run(
        self,
        run_id: str,
        status: str,
        steps: int,
        cost: float,
        final: str | None,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cached_tokens: int | None = None,
        total_tokens: int | None = None,
        finished_at: float | None = None,
        error: str | None = None,
    ) -> None: ...

    def get_run(self, run_id: str, user_id: str) -> StoredRun | None: ...

    def list_runs(self, user_id: str, limit: int = 30) -> list[StoredRun]: ...

    # -- messages -------------------------------------------------------------
    def add_message(self, msg: StoredMessage) -> None: ...

    def messages_for_run(self, run_id: str, user_id: str) -> list[dict]: ...

    def messages_for_runs(
        self, run_ids: Collection[str], user_id: str, *, exclude_subagent: bool = True
    ) -> list[dict]: ...

    # -- actions --------------------------------------------------------------
    # StoreSink calls this automatically for the undo-bearing UI events tools
    # emit (file_change → kind "file_write"/"file_edit"/"file_delete",
    # todo_change → "todo_replace"); undo_run reads them back via list_actions.
    # list_actions' ``subagent`` filter narrows to one subagent's rows (or,
    # with the sentinel "", to the orchestrator's own) — implementations
    # should honor it without loading/rehydrating the rest when they can.
    def log_action(
        self,
        run_id: str,
        user_id: str,
        kind: str,
        target: str,
        old_value: Any,
        new_value: Any,
        *,
        status: str = "applied",
        subagent: str | None = None,
    ) -> int: ...

    def list_actions(
        self,
        run_id: str,
        user_id: str,
        *,
        status_in: Collection[str] = (),
        subagent: str | None = None,
    ) -> list[StoredAction]: ...

    def actions_for_runs(
        self, run_ids: Collection[str], user_id: str
    ) -> list[StoredAction]: ...

    def get_action(self, action_id: int, user_id: str) -> StoredAction | None: ...

    def set_action_status(
        self, action_id: int, user_id: str, status: str
    ) -> StoredAction | None: ...


class ConversationStore(Protocol):
    """Optional conversation-grouping operations.

    Hosts that only need single-shot runs may implement :class:`RunStore` alone.
    Hosts with multi-turn conversations group runs under a conversation id.

    A conversation row is ``{"id", "user_id", "title", "meta"}`` where ``meta``
    is a free-form host-owned dict (e.g. workspace, pinned profile/model).
    """

    def create_conversation(
        self, user_id: str, title: str, *, meta: dict | None = None
    ) -> dict: ...

    def get_conversation(self, conversation_id: int, user_id: str) -> dict | None: ...

    def list_conversations(self, user_id: str, limit: int = 40) -> list[dict]: ...

    def rename_conversation(
        self, conversation_id: int, user_id: str, title: str
    ) -> None: ...

    def update_conversation_meta(
        self, conversation_id: int, user_id: str, meta: dict
    ) -> int: ...

    def delete_conversation(self, conversation_id: int, user_id: str) -> int: ...

    def runs_for_conversation(
        self, conversation_id: int, user_id: str
    ) -> list[StoredRun]: ...

    def conversation_summaries(
        self, user_id: str, limit: int = 40
    ) -> list[dict]: ...

    def messages_for_conversation(
        self, conversation_id: int, user_id: str, *, exclude_subagent: bool = True
    ) -> list[dict]: ...


class BlobStore(Protocol):
    """Externalize large payloads (tool output, snapshots) out of message rows.

    Following the pattern used by Claude Code / OpenCode / Kilo: a tool result
    bigger than a threshold is spilled to a content-addressed blob and the
    message stores only a small ``ref``. This keeps the message store compact
    and inspectable regardless of backend.
    """

    def spill(self, blob: bytes) -> str: ...

    def load(self, ref: str) -> bytes: ...


def action_to_row(a: StoredAction) -> dict:
    """Convenience: StoredAction → the dict row shape replay transforms expect."""
    return {
        "id": a.id,
        "run_id": a.run_id,
        "kind": a.kind,
        "target": a.target,
        "old_value": a.old_value,
        "new_value": a.new_value,
        "status": a.status,
        "subagent": a.subagent,
    }


def message_to_row(m: StoredMessage) -> dict:
    """Convenience: StoredMessage → the dict row shape replay transforms expect."""
    return {
        "role": m.role,
        "content": m.content,
        "tool_calls": m.tool_calls,
        "tool_name": m.tool_name,
        "tool_call_id": m.tool_call_id,
        "meta": m.meta,
        "subagent": m.subagent,
        "id": m.id,
    }
