"""JsonlRunStore — the default, zero-database conversation store.

Following the Claude Code pattern: conversations / runs / messages / actions are
appended as JSONL lines (one logical stream each), and large payloads are
spilled to content-addressed blob files. No SQLite, no server — just files you
can ``grep``, ``tail``, or commit. Append-only writes make it crash-safe; reads
filter by ``user_id`` (and ``run_id`` / ``conversation_id``) so a multi-tenant
host never crosses users.

Line discriminators: each JSONL line carries a ``kind`` tag naming its stream
(``run`` / ``run_final`` / ``message`` / ``action`` / ``action_status`` /
conversation base / ``rename`` / ``meta`` / ``delete``). A mutation's *domain* kind
(``file_write`` / ``guidance`` / ...) is stored under ``mkind`` to avoid
colliding with the line discriminator.

This is a *default* backend, sized for moderate volume: each read scans the
relevant file. Hosts with high volume / many users should back the Protocol
(:mod:`lithe.bundles.store.protocol`) with a real DB instead.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

from lithe.bundles.store.protocol import (
    StoredAction, StoredMessage, StoredRun,
)

log = logging.getLogger("lithe.store")

_REVERTIBLE = ("applied", "approved")
_NO_LIMIT = 10**9


class JsonlRunStore:
    """Append-only JSONL + blob-file store implementing Run/Conversation/Blob.

    ``spill_threshold`` (bytes of one serialized value, default 8KB): action
    ``old_value``/``new_value`` strings larger than it are externalized to a
    content-addressed blob and the row stores a small ``blob:sha256:…`` ref;
    reads rehydrate transparently, so reverters still see the full content
    while the actions file stays small. ``0`` disables spilling.

    ``fsync=True`` flushes each append to disk before returning (matching the
    crash-safe claim in the module docstring) at the cost of one fsync per
    write; the default buffers like any normal file and relies on the OS.

    Torn/corrupt lines (a crash mid-append) are skipped on read but never
    silently: each drop counts in ``dropped_lines`` (per JSONL file) and logs
    a warning, because a torn line can be an undo-able action going missing.
    """

    def __init__(self, root: Path | str, *, spill_threshold: int = 8192,
                 fsync: bool = False):
        self.root = Path(root)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        self._runs = self.root / "runs.jsonl"
        self._messages = self.root / "messages.jsonl"
        self._actions = self.root / "actions.jsonl"
        self._conversations = self.root / "conversations.jsonl"
        self.spill_threshold = spill_threshold
        self.fsync = fsync
        # path -> count of unparseable lines dropped by reads so far.
        self.dropped_lines: dict[Path, int] = {}
        # Lazily-initialized id counters: seeded once from the files on first
        # write, then incremented in memory — log_action/create_conversation
        # no longer rescan (and rewrite) the whole stream per append.
        self._next_action_id: int | None = None
        self._next_conversation_id: int | None = None
        self._known_run_ids: set[str] | None = None

    # -- low-level jsonl helpers ---------------------------------------------
    def _read(self, path: Path) -> list[dict]:
        if not path.is_file():
            return []
        out = []
        dropped = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except (ValueError, TypeError):
                    dropped += 1
        if dropped:
            self.dropped_lines[path] = self.dropped_lines.get(path, 0) + dropped
            log.warning("JsonlRunStore: dropped %d unparseable line(s) from %s "
                        "(torn write or foreign content — a dropped action "
                        "line means lost undo history)", dropped, path)
        return out

    def _append(self, path: Path, obj: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if self.fsync:
                fh.flush()
                os.fsync(fh.fileno())

    # -- runs -----------------------------------------------------------------
    def _all_run_ids(self) -> set[str]:
        """Every ``run_id`` ever created (all users — ids are globally unique).

        Lazily seeded once, then maintained in memory like the id counters, so
        ``create_run`` doesn't rescan the file per call.
        """
        if self._known_run_ids is None:
            ids = set()
            for ln in self._read(self._runs):
                if ln.get("kind") == "run" and isinstance(ln.get("run_id"), str):
                    ids.add(ln["run_id"])
            self._known_run_ids = ids
        return self._known_run_ids

    def create_run(self, run_id, user_id, task, *, conversation_id=None,
                    model=None, created_at=None) -> None:
        known = self._all_run_ids()
        if run_id in known:
            # Duplicate ids corrupt the fold (one StoredRun chimera, doubled
            # listings) and — since run_final lines carry no user_id — let
            # one user's finish overwrite another's run.
            raise ValueError(f"run_id already exists: {run_id!r} "
                             "(run ids must be globally unique)")
        self._append(self._runs, {"kind": "run", "run_id": run_id,
                                  "user_id": user_id,
                                  "conversation_id": conversation_id,
                                  "task": task, "status": "running",
                                  "model": model,
                                  "created_at": created_at or time.time()})
        known.add(run_id)

    def finish_run(self, run_id, status, steps, cost, final, *,
                   prompt_tokens=None, completion_tokens=None,
                   cached_tokens=None, total_tokens=None,
                   finished_at=None, error=None) -> None:
        """Close a run; token fields are optional and additive (legacy rows
        written without them read back as ``None`` — "unknown", not zero).
        ``error`` is the failed-run diagnostic (see protocol.StoredRun);
        written only when set so legacy lines stay shape-identical."""
        row = {"kind": "run_final", "run_id": run_id,
               "status": status, "steps": steps,
               "cost": cost, "final": final,
               "prompt_tokens": prompt_tokens,
               "completion_tokens": completion_tokens,
               "cached_tokens": cached_tokens,
               "total_tokens": total_tokens,
               "finished_at": finished_at or time.time()}
        if error is not None:
            row["error"] = error
        self._append(self._runs, row)

    def _fold_runs(self, user_id):
        """Return ``(by_id, order)`` of StoredRun for *user_id*, newest state wins.

        ``run_final`` lines carry no ``user_id``; they are only applied to runs
        already admitted by a user-scoped ``run`` header (run_ids are globally
        unique), so a final-status line can never leak across users.
        """
        by_id: dict[str, StoredRun] = {}
        order: list[str] = []
        for ln in self._read(self._runs):
            kind = ln.get("kind")
            rid = ln.get("run_id")
            if kind == "run":
                if ln.get("user_id") != user_id:
                    continue
                by_id[rid] = StoredRun(run_id=rid, user_id=user_id,
                                       task=ln.get("task", ""), status="running",
                                       conversation_id=ln.get("conversation_id"),
                                       model=ln.get("model"),
                                       created_at=ln.get("created_at"))
                if rid not in order:  # legacy duplicate headers list once
                    order.append(rid)
            elif kind == "run_final" and rid in by_id:
                r = by_id[rid]
                r.status = ln.get("status", r.status)
                r.steps = ln.get("steps", 0)
                r.cost = ln.get("cost", 0.0)
                r.final = ln.get("final")
                # Optional fields: only overwrite when the final line carries
                # them, so a legacy finish never clobbers stamped values with
                # None and a tokenless finish stays honestly None.
                if ln.get("finished_at") is not None:
                    r.finished_at = ln["finished_at"]
                if ln.get("error") is not None:
                    r.error = ln["error"]
                for f in ("prompt_tokens", "completion_tokens",
                          "cached_tokens", "total_tokens"):
                    if ln.get(f) is not None:
                        setattr(r, f, ln[f])
        return by_id, order

    def get_run(self, run_id, user_id) -> StoredRun | None:
        by_id, _ = self._fold_runs(user_id)
        return by_id.get(run_id)

    def list_runs(self, user_id, limit=30) -> list[StoredRun]:
        by_id, order = self._fold_runs(user_id)
        return [by_id[rid] for rid in order if rid in by_id][-limit:]

    # -- messages -------------------------------------------------------------
    def add_message(self, msg: StoredMessage) -> None:
        self._append(self._messages, {"kind": "message", "role": msg.role,
                                      "content": msg.content, "run_id": msg.run_id,
                                      "user_id": msg.user_id,
                                      "tool_calls": msg.tool_calls,
                                      "tool_name": msg.tool_name,
                                      "tool_call_id": msg.tool_call_id,
                                      "meta": msg.meta, "subagent": msg.subagent})

    def _msg_rows(self, run_ids, user_id, *, exclude_subagent):
        rids = set(run_ids)
        out = []
        for ln in self._read(self._messages):
            if ln.get("kind") != "message" or ln.get("user_id") != user_id:
                continue
            if ln.get("run_id") not in rids:
                continue
            if exclude_subagent and ln.get("subagent"):
                continue
            out.append({"role": ln.get("role"), "content": ln.get("content"),
                        "tool_calls": ln.get("tool_calls"),
                        "tool_name": ln.get("tool_name"),
                        "tool_call_id": ln.get("tool_call_id"),
                        "meta": ln.get("meta"), "subagent": ln.get("subagent")})
        return out

    def messages_for_run(self, run_id, user_id) -> list[dict]:
        return self._msg_rows([run_id], user_id, exclude_subagent=False)

    def messages_for_runs(self, run_ids, user_id, *, exclude_subagent=True) -> list[dict]:
        return self._msg_rows(run_ids, user_id, exclude_subagent=exclude_subagent)

    # -- blob spill for large action values ------------------------------------
    def _spill_value(self, v):
        if (self.spill_threshold and isinstance(v, str)
                and len(v) > self.spill_threshold):
            return "blob:" + self.spill(v.encode("utf-8"))
        return v

    def _unspill_value(self, v):
        if isinstance(v, str) and v.startswith("blob:"):
            try:
                return self.load(v[len("blob:"):]).decode("utf-8")
            except (OSError, UnicodeDecodeError, ValueError):
                # blob gone, or a tool-supplied literal that merely looks like
                # a ref: surface the original string rather than crash undo
                return v
        return v

    # -- actions --------------------------------------------------------------
    def _fold_actions(self, *, subagent: str | None = None):
        """Return ``(by_id, order)``; latest ``action_status`` wins per id.

        With ``subagent`` set, only that subagent's action rows are folded
        (others are skipped before their spilled values rehydrate)."""
        by_id: dict[int, StoredAction] = {}
        order: list[int] = []
        for ln in self._read(self._actions):
            if ln.get("kind") == "action":
                if subagent is not None and ln.get("subagent") != subagent:
                    continue
                aid = ln.get("id")
                by_id[aid] = StoredAction(
                    run_id=ln.get("run_id"), user_id=ln.get("user_id"),
                    kind=ln.get("mkind", ""), target=ln.get("target", ""),
                    old_value=self._unspill_value(ln.get("old_value")),
                    new_value=self._unspill_value(ln.get("new_value")),
                    status=ln.get("status", "applied"),
                    subagent=ln.get("subagent"), id=aid)
                order.append(aid)
            elif ln.get("kind") == "action_status" and ln.get("id") in by_id:
                by_id[ln["id"]].status = ln.get("status",
                                                by_id[ln["id"]].status)
        return by_id, order

    def log_action(self, run_id, user_id, kind, target, old_value, new_value, *,
                   status="applied", subagent=None) -> int:
        if self._next_action_id is None:
            by_id, _ = self._fold_actions()
            self._next_action_id = max(by_id, default=0) + 1
        next_id = self._next_action_id
        self._next_action_id += 1
        self._append(self._actions, {"kind": "action", "id": next_id,
                                     "run_id": run_id, "user_id": user_id,
                                     "mkind": kind, "target": target,
                                     "old_value": self._spill_value(old_value),
                                     "new_value": self._spill_value(new_value),
                                     "status": status, "subagent": subagent})
        return next_id

    def list_actions(self, run_id, user_id, *, status_in=(),
                     subagent: str | None = None) -> list[StoredAction]:
        """Actions of one run, optionally narrowed by status and subagent.

        ``subagent=None`` (default) applies no filter; a string narrows to
        that subagent's rows. The filter applies during the fold, so
        unselected rows skip blob rehydration — the subagent engine's
        per-delegation snapshots no longer rehydrate every earlier value.
        """
        sin = set(status_in)
        by_id, order = self._fold_actions(subagent=subagent)
        out = []
        for aid in order:
            a = by_id[aid]
            if a.run_id != run_id or a.user_id != user_id:
                continue
            if sin and a.status not in sin:
                continue
            out.append(a)
        return out

    def actions_for_runs(self, run_ids, user_id) -> list[StoredAction]:
        rids = set(run_ids)
        by_id, order = self._fold_actions()
        return [by_id[aid] for aid in order
                if by_id[aid].run_id in rids and by_id[aid].user_id == user_id]

    def get_action(self, action_id, user_id) -> StoredAction | None:
        by_id, _ = self._fold_actions()
        a = by_id.get(action_id)
        return a if a and a.user_id == user_id else None

    def set_action_status(self, action_id, user_id, status) -> StoredAction | None:
        a = self.get_action(action_id, user_id)
        if a is None:
            return None
        self._append(self._actions, {"kind": "action_status", "id": action_id,
                                     "status": status})
        a.status = status
        return a

    # -- conversations --------------------------------------------------------
    def _fold_conversations(self, user_id):
        by_id: dict[int, dict] = {}
        order: list[int] = []
        deleted: set[int] = set()
        for ln in self._read(self._conversations):
            k = ln.get("kind")
            cid = ln.get("id")
            if k == "rename":
                if cid in by_id:
                    by_id[cid]["title"] = ln.get("title", by_id[cid].get("title"))
            elif k == "delete":
                deleted.add(cid)
            elif k == "meta":
                if cid in by_id:
                    by_id[cid].setdefault("meta", {})
                    by_id[cid]["meta"].update(ln.get("meta") or {})
            elif k is None and cid is not None:
                if cid not in by_id and ln.get("user_id") == user_id:
                    conv = dict(ln)
                    if conv.get("meta"):
                        conv["meta"] = dict(conv["meta"])
                    else:
                        conv["meta"] = {}
                    by_id[cid] = conv
                    order.append(cid)
        return [by_id[c] for c in order if c not in deleted]

    def create_conversation(self, user_id, title, *, meta=None) -> dict:
        if self._next_conversation_id is None:
            all_rows = self._read(self._conversations)
            self._next_conversation_id = max(
                (r.get("id", 0) for r in all_rows if r.get("kind") is None),
                default=0) + 1
        next_id = self._next_conversation_id
        self._next_conversation_id += 1
        conv = {"id": next_id, "user_id": user_id, "title": title,
                "meta": dict(meta or {})}
        self._append(self._conversations, conv)
        return dict(conv)

    def get_conversation(self, conversation_id, user_id) -> dict | None:
        for c in self._fold_conversations(user_id):
            if c.get("id") == conversation_id:
                return c
        return None

    def list_conversations(self, user_id, limit=40) -> list[dict]:
        return self._fold_conversations(user_id)[-limit:]

    def rename_conversation(self, conversation_id, user_id, title) -> None:
        if self.get_conversation(conversation_id, user_id) is None:
            return
        self._append(self._conversations,
                     {"kind": "rename", "id": conversation_id, "title": title})

    def delete_conversation(self, conversation_id, user_id) -> int:
        if self.get_conversation(conversation_id, user_id) is None:
            return 0
        self._append(self._conversations,
                     {"kind": "delete", "id": conversation_id})
        return 1

    def update_conversation_meta(self, conversation_id, user_id, meta) -> int:
        """Merge *meta* into a conversation's host-owned metadata dict."""
        if self.get_conversation(conversation_id, user_id) is None:
            return 0
        self._append(self._conversations,
                     {"kind": "meta", "id": conversation_id,
                      "meta": dict(meta or {})})
        return 1

    def runs_for_conversation(self, conversation_id, user_id) -> list[StoredRun]:
        return [r for r in self.list_runs(user_id, limit=_NO_LIMIT)
                if r.conversation_id == conversation_id]

    def conversation_summaries(self, user_id, limit=40) -> list[dict]:
        """One aggregated row per conversation, newest activity first.

        Activity ordering uses the last run's ``finished_at``/``created_at``
        when present and falls back to file order for legacy rows, so a
        summary list needs no timestamps of its own. Token totals sum the
        runs that reported them (``None`` on legacy/unfinished runs means
        those runs contribute nothing, not zero-cost knowledge).
        """
        runs = self.list_runs(user_id, limit=_NO_LIMIT)
        by_conv: dict[int, list[StoredRun]] = {}
        for r in runs:
            if r.conversation_id is not None:
                by_conv.setdefault(r.conversation_id, []).append(r)
        summaries = []
        for conv in self._fold_conversations(user_id):
            rs = by_conv.get(conv["id"], [])
            last = rs[-1] if rs else None

            def _sum(field, rs=rs):
                vals = [getattr(r, field) for r in rs
                        if getattr(r, field) is not None]
                return sum(vals) if vals else None

            summaries.append({
                "id": conv["id"],
                "title": conv.get("title", ""),
                "meta": dict(conv.get("meta") or {}),
                "n_runs": len(rs),
                "last_status": last.status if last else None,
                "last_task": last.task if last else None,
                "last_model": last.model if last else None,
                "updated_at": (last.finished_at or last.created_at) if last
                              else None,
                "total_cost": sum(r.cost for r in rs) if rs else 0.0,
                "prompt_tokens": _sum("prompt_tokens"),
                "completion_tokens": _sum("completion_tokens"),
                "cached_tokens": _sum("cached_tokens"),
                "total_tokens": _sum("total_tokens"),
            })
        # Newest activity first; stamped rows by time, legacy (unstamped)
        # rows keep file order after them — insertion scale is host-small.
        stamped = [s for s in summaries if s["updated_at"] is not None]
        unstamped = [s for s in summaries if s["updated_at"] is None]
        stamped.sort(key=lambda s: s["updated_at"], reverse=True)
        return (stamped + unstamped)[:limit]

    def messages_for_conversation(self, conversation_id, user_id, *,
                                  exclude_subagent=True) -> list[dict]:
        rids = [r.run_id for r in
                self.runs_for_conversation(conversation_id, user_id)]
        return self._msg_rows(rids, user_id,
                              exclude_subagent=exclude_subagent)

    # -- blobs ----------------------------------------------------------------
    # Honest refs are always "sha256:" + exactly 64 lowercase hex chars; the
    # anchor makes forged refs (e.g. "sha256:../../.ssh/id_rsa" planted as a
    # tool-visible old_value) fail validation instead of reading arbitrary
    # host files on rehydrate.
    _BLOB_DIGEST = re.compile(r"[0-9a-f]{64}")

    def spill(self, blob) -> str:
        data = blob if isinstance(blob, (bytes, bytearray)) else str(blob).encode()
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / "blobs" / digest
        if not path.exists():
            path.write_bytes(data)
        return f"sha256:{digest}"

    def load(self, ref: str) -> bytes:
        digest = ref.split(":", 1)[-1]
        if not self._BLOB_DIGEST.fullmatch(digest):
            raise ValueError(f"invalid blob ref: {ref!r}")
        return (self.root / "blobs" / digest).read_bytes()


def revertible_actions(store: JsonlRunStore, run_id: str,
                       user_id: str) -> list[StoredAction]:
    """Helper: actions of a run that are currently in effect (undo candidates)."""
    return store.list_actions(run_id, user_id, status_in=_REVERTIBLE)
