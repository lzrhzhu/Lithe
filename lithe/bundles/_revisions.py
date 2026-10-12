"""Per-run workspace revision bookkeeping, shared by every file-writing bundle.

Three guards live here, all keyed off the run context (never the tool that
happens to trigger them), so the workspace / patch / download bundles enforce
the exact same discipline:

- **stale-content guard** — ``read_file`` records ``(mtime_ns, size)`` per
  absolute path; the write paths refuse to touch a path whose current stat no
  longer matches the last read snapshot (optimistic concurrency, Kilo's
  writeIfUnchanged semantics), and a per-path write *generation* stamps what
  each context has personally observed, so parallel subagents cannot clobber
  a file a sibling wrote after they read it;
- **runaway-loop brake** — how many NEW files one run may create through the
  bundled write tools (never a limit on the workspace's pre-existing
  content);
- **the run-wide write lock** — one ``asyncio.Lock`` per run, shared by
  reference with every subagent context, serializing every bundled file
  mutation against parallel subagent writers.

The state lives in ``ctx.shared`` (run-wide, shared by reference with derived
subagent contexts) and ``ctx.extra`` (per-context observation stamps), never
on any tool object. Hosts and reverters writing through ``Workspace`` I/O
directly bypass all of it by design.

Internal module — the sibling of ``_textmatch``; bundles import from here,
hosts import nothing.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from lithe.context import AgentContext
from lithe.tools import ToolResult

if TYPE_CHECKING:
    from lithe.bundles.workspace import Workspace


def _revisions(ctx: AgentContext) -> dict:
    return cast("dict", ctx.shared.setdefault("_file_revisions", {}))


def _generations(ctx: AgentContext) -> dict:
    """Per-path write generation (shared run-wide): bumped by every in-run
    write through the bundled tools. A context whose last read/write of a
    path happened at an older generation is working from stale content —
    typically a parallel subagent whose sibling (or the orchestrator) wrote
    the file after it read."""
    return cast("dict", ctx.shared.setdefault("_ws_generations", {}))


def _seen_gens(ctx: AgentContext) -> dict:
    """Per-context \"I last observed path P at generation G\" stamps.

    Lives in ``ctx.extra`` (copied per context — the orchestrator and every
    subagent keep their OWN observation history; a subagent inherits the
    orchestrator's stamps at delegation start, correctly so: it has not
    seen anything newer)."""
    return cast("dict", ctx.extra.setdefault("_ws_seen_gens", {}))


def _created_paths(ctx: AgentContext) -> set[str]:
    """Paths this run created through the bundled write tools (the set the
    runaway-loop brake counts — overwriting existing files never counts, and
    direct ``Workspace`` I/O from hosts / reverters is never refused)."""
    return cast("set[str]", ctx.shared.setdefault("_ws_created_files", set()))


def _check_run_growth(ctx: AgentContext, ws: Workspace, rels) -> ToolResult | None:
    """``None`` when creating the not-yet-existing paths in *rels* stays within
    ``ws.max_files`` for this run; a refusal result otherwise."""
    fresh: set[str] = set()
    for rel in rels:
        try:
            norm = ws.safe_path(rel).relative_to(ws.root).as_posix()
        except (PermissionError, ValueError):
            continue
        if norm not in _created_paths(ctx) and not ws.exists(rel):
            fresh.add(norm)
    if not fresh:
        return None
    if len(_created_paths(ctx) | fresh) > ws.max_files:
        return ToolResult(
            False, "文件数超限",
            f"本次运行通过工具新建文件将超过 {ws.max_files} 个（防失控上限）。"
            f"请停止继续新建文件；若任务确实需要更多，请向用户说明并建议"
            f"分批执行或调大 max_files。")
    return None


def _note_created(ctx: AgentContext, ws: Workspace, rel: str) -> None:
    try:
        _created_paths(ctx).add(ws.safe_path(rel).relative_to(ws.root).as_posix())
    except (PermissionError, ValueError):
        pass


def _stat(ws: Workspace, rel: str):
    try:
        st = ws.safe_path(rel).stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _apply_revision(ctx: AgentContext, ws: Workspace, rel: str, snap,
                    *, written: bool = False) -> None:
    """Record one observation of *rel*: refresh the run-wide snapshot, and
    stamp the observing context's generation (writes bump the run's
    generation first, so the writer itself stays fresh for chained
    write→edit flows while every OTHER context's older stamp goes stale)."""
    key = str(ws.safe_path(rel))
    if snap is None:
        _revisions(ctx).pop(key, None)
    else:
        _revisions(ctx)[key] = snap
    gens = _generations(ctx)
    if written:
        gens[key] = gens.get(key, 0) + 1
    _seen_gens(ctx)[key] = gens.get(key, 0)


def _record_revision(ctx: AgentContext, ws: Workspace, rel: str) -> None:
    _apply_revision(ctx, ws, rel, _stat(ws, rel), written=True)


def _ws_write_lock(ctx: AgentContext) -> asyncio.Lock:
    """The run's exclusive file-mutation lock.

    Lives in ``ctx.shared`` so every subagent context of the run holds the
    SAME lock by reference: within one runtime WRITE tools already run alone,
    but parallel subagents each drive their own runtime — without this lock
    their write tools interleave at every ``await asyncio.to_thread(...)``
    hop, and the freshness check-then-act loses to a concurrent write (both
    writers pass before either lands). The lock serializes the section so
    the generation check in :func:`_check_fresh` — not lock ordering alone —
    deterministically refuses the second writer.

    Readers do NOT take the lock (READ tools stay parallel); their safety
    comes from read_file's stat-BEFORE-read snapshot ordering: a write that
    slips between stat and read leaves the recorded snapshot OLDER than the
    content the model saw, so the model's next write is (conservatively)
    refused and it re-reads. Undo reverters write directly and bypass the
    lock; undo runs standalone, never concurrent with agent runs.
    """
    lock = ctx.shared.get("_ws_write_lock")
    if lock is None:
        lock = asyncio.Lock()
        ctx.shared["_ws_write_lock"] = lock
    return lock


def _check_fresh(ctx: AgentContext, ws: Workspace, rel: str) -> ToolResult | None:
    """``None`` when fresh (or never read this run); a refusal otherwise.

    Two detection layers, both required:

    - the run-wide snapshot comparison catches EXTERNAL changes (out-of-band
      edits after any context of this run read the path);
    - the per-context generation comparison catches IN-RUN writers: the
      first bundled write refreshes the shared snapshot to its own stat, so
      a stat comparison alone would wave a sibling's stale write through —
      the generation stamp ("what I personally last observed") does not
      refresh with it, so a parallel subagent (or the orchestrator) that
      has not re-read since the other's write is refused.
    """
    key = str(ws.safe_path(rel))
    recorded = _revisions(ctx).get(key)
    if recorded is None:
        return None
    current = _stat(ws, rel)
    seen = _seen_gens(ctx).get(key)
    stale = (current is not None and current != recorded) or (
        seen is not None and seen != _generations(ctx).get(key, 0))
    if stale:
        return ToolResult(
            False, "文件已变更",
            f"文件 {rel} 在上次读取之后被修改（可能是外部改动，或本运行的"
            f"其它工具/并行子代理写入）。请先重新 read_file 获取最新内容，"
            f"再执行写入或编辑。")
    return None
