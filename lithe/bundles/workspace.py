"""Workspace bundle: a sandboxed file root + the read/write/edit/list tools
every agent needs, with file undo reverters. Directory layout and persistence
stay with the host — this bundle only knows "a root path + safe file I/O".

This is an OPTIONAL capability bundle, not part of the zero-I/O core engine.
A host opts in by giving a ``workspace_for(ctx) -> Workspace`` callable (it
decides where each user's root is) and calling :func:`register_file_tools`.

The tools emit ``file_change`` UI events (old/new/created) so a host
:class:`~lithe.events.EventSink` can record undo actions; the bundle also
returns the matching reverters (keyed by mutation *kind*, not tool name) for the
host's :class:`~lithe.actions.UndoEngine`. The bundle itself stores nothing.
"""
from __future__ import annotations

import asyncio
import difflib
import fnmatch
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Iterator

from lithe.context import AgentContext
from lithe.bundles._textmatch import line_span_hits
from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec

_DEFAULT_IGNORED = frozenset({
    ".DS_Store", "__pycache__", ".ipynb_checkpoints", ".pytest_cache",
    ".venv", "venv", "node_modules", ".git", ".hg", ".svn", ".tox",
    ".mypy_cache", ".ruff_cache",
})

_SEARCH_MAX_MATCHES = 50
_SEARCH_MAX_FILE = 1_000_000     # skip regex-scanning files larger than 1MB
# A line longer than this is skipped rather than regex-matched: catastrophic
# backtracking is exponential in the subject, so one huge minified line with
# an unlucky pattern could otherwise hang the whole event loop (re runs
# synchronously and cannot be cancelled).
_SEARCH_MAX_LINE = 10_000
# Whole-scan wall budget; exceeded → honest partial result instead of a hang.
_SEARCH_TIME_BUDGET = 10.0
_SEARCH_YIELD_EVERY = 2000       # lines between event-loop yields
_SEARCH_READ_BATCH = 16          # files per asyncio.to_thread read batch
_MATCH_LINE_CAP = 200
_GLOB_MAX_RESULTS = 200
_DIFF_CAP = 2000


@dataclass
class Workspace:
    """A sandboxed file root: every path is resolved strictly inside it.

    Hosts create one per user (the root is wherever they decide); this class
    only enforces traversal safety and provides file I/O. ``protected_dirs`` are
    folder skeletons ``delete`` refuses to remove (a host passes its reserved
    layout names). ``max_files`` is the runaway-loop brake on the bundled write
    tools — how many NEW files one run may create through them — never a limit
    on the workspace's pre-existing content (a project workspace may hold any
    number of the user's own files; see ``_check_run_growth``).
    """
    root: Path
    max_files: int = 500
    protected_dirs: frozenset[str] = field(default_factory=frozenset)
    ignored: frozenset[str] = field(default_factory=lambda: _DEFAULT_IGNORED)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    def safe_path(self, rel: str) -> Path:
        rel = (rel or "").strip()
        if rel in ("", ".", "./"):
            return self.root
        rel = rel.lstrip("/\\")
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError(f"path escapes workspace: {rel}")
        return target

    def exists(self, rel: str) -> bool:
        return self.safe_path(rel).exists()

    def read(self, rel: str) -> str:
        target = self.safe_path(rel)
        if not target.is_file():
            raise FileNotFoundError(rel)
        return target.read_text(encoding="utf-8", errors="replace")

    def write(self, rel: str, content: str) -> Path:
        self._check_rel(rel)
        target = self.safe_path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def write_bytes(self, rel: str, data: bytes) -> Path:
        self._check_rel(rel)
        target = self.safe_path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def delete(self, rel: str) -> None:
        rel_norm = (rel or "").lstrip("/\\").rstrip("/")
        if rel_norm in self.protected_dirs:
            raise PermissionError(f"目录 {rel} 是保留目录，不能删除")
        target = self.safe_path(rel)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_file():
            target.unlink(missing_ok=True)

    def new_folder(self, rel: str) -> None:
        self.safe_path(rel).mkdir(parents=True, exist_ok=True)

    def rename(self, old: str, new: str) -> None:
        src = self.safe_path(old)
        dst = self.safe_path(new)
        if not src.exists():
            raise FileNotFoundError(old)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)

    def walk(self) -> Iterator[tuple[str, Path]]:
        for p in sorted(self.root.rglob("*")):
            if any(part in self.ignored for part in p.parts):
                continue
            # Never follow symlinks out of the sandbox: model-executed code
            # (run_code binds the workspace read-write) can plant a link to a
            # host file; walking it would read host content that safe_path
            # would refuse to serve directly.
            if p.is_symlink():
                continue
            if p.is_file():
                yield p.relative_to(self.root).as_posix(), p

    def list(self, subdirs: tuple[str, ...] = ()) -> list[dict]:
        """List entries; if *subdirs* is given, only those top-level folders are
        walked (each shown as an entry too, so empty folders stay visible);
        otherwise the whole tree."""
        roots: list[Path] = []
        for d in subdirs:
            try:
                base = self.safe_path(d)
            except PermissionError as exc:
                raise PermissionError(f"非法目录：{d}（{exc}）") from None
            if base == self.root:
                roots = [self.root]  # "." degenerates to the full walk
                break
            roots.append(base)
        if not subdirs:
            roots = [self.root]
        out: list[dict] = []
        for base in roots:
            if not base.is_dir():
                continue
            if subdirs and base != self.root:
                rel = base.relative_to(self.root).as_posix()
                out.append({"path": rel, "type": "dir", "size": 0,
                            "modified": _mtime(base)})
            for p in sorted(base.rglob("*")):
                if any(part in self.ignored for part in p.parts):
                    continue
                if p.is_symlink():
                    # Same containment rule as walk(): a symlink (file or
                    # dir) is listed as opaque, never followed. lstat — a
                    # dangling link must list, not raise.
                    rel = p.relative_to(self.root).as_posix()
                    out.append({"path": rel, "type": "symlink", "size": 0,
                                "modified": _mtime(p, follow=False)})
                    continue
                rel = p.relative_to(self.root).as_posix()
                out.append({"path": rel, "type": "dir" if p.is_dir() else "file",
                            "size": 0 if p.is_dir() else p.stat().st_size,
                            "modified": _mtime(p)})
        return out

    def _check_rel(self, rel: str) -> None:
        if rel.strip() == "" or rel.endswith("/"):
            raise ValueError("invalid file path")

    def _count(self) -> int:
        # Counts what the workspace itself sees (ignored dirs excluded), so
        # the number is consistent with walk()/list() instead of counting a
        # venv/node_modules the tools deliberately never look at.
        return sum(1 for p in self.root.rglob("*")
                   if not any(part in self.ignored for part in p.parts))


def _mtime(p: Path, *, follow: bool = True) -> str:
    stat = p.stat() if follow else p.lstat()
    return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()


def _line_delta(old, new) -> tuple[int, int]:
    """(added, removed) line counts between two file contents (None = empty)."""
    old_lines = old.splitlines() if old else []
    new_lines = new.splitlines() if new else []
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    added = removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            removed += i2 - i1
            added += j2 - j1
    return added, removed


def _delta_note(added: int, removed: int, created: bool = False) -> str:
    """Coding-agent style line-count suffix, e.g. ``（+3 -1 行）``."""
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    if not parts:
        return ""
    return "（" + " ".join(parts) + " 行" + ("，新建" if created else "") + "）"


def _file_change(action: str, path: str, old, new) -> dict:
    added, removed = _line_delta(old, new)
    return {"type": "file_change", "action": action, "path": path,
            "created": old is None, "old": old, "new": new,
            "added": added, "removed": removed}


def _compact_diff(rel: str, old: str, new: str) -> str:
    """Small unified diff of an edit for the model to verify its own change
    (capped; the tool result is model-facing context, not a storage record)."""
    diff = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=rel, tofile=rel))
    diff = diff.rstrip("\n")
    if len(diff) > _DIFF_CAP:
        diff = diff[:_DIFF_CAP] + "\n…[diff 过长已截断]"
    return diff


# --- per-run file revisions (stale-content guard) ---------------------------
#
# read_file records (mtime_ns, size) per absolute path; the write tools refuse
# to touch a path whose current stat no longer matches the last read snapshot
# — an external edit between the model's read and its write surfaces as
# "re-read the file" instead of a silently clobbered change (the
# optimistic-concurrency semantics of Kilo's writeIfUnchanged). Successful
# writes refresh the snapshot, so chained write→edit flows never false-alarm.
#
# The map lives in ctx.shared (not extra): derived subagent contexts share the
# SAME dict by reference, so the guard covers the whole run — two parallel
# subagents (or a subagent and its orchestrator) cannot clobber a file the
# other read this run, because the second writer's freshness check sees the
# first's snapshot. Per-run and in-memory only; paths never read this run are
# unchecked.


def _revisions(ctx: AgentContext) -> dict:
    return ctx.shared.setdefault("_file_revisions", {})


# --- per-run file-creation guard (runaway-loop brake) ------------------------
#
# The OLD guard refused new files when the whole tree held more than
# ``max_files`` entries — measuring the user's pre-existing project (venv,
# node_modules, .git, or simply a large repo) instead of what the agent is
# doing. A real project workspace therefore rejected every new-file write
# (and undo's restore path died the same way). The brake now measures the
# right thing: how many NEW files this run creates through the bundled write
# tools, tracked in ctx.shared — shared by reference with subagent contexts,
# so parallel workers count against the same budget. Overwriting existing
# files never counts (no growth); direct Workspace I/O (hosts, reverters)
# is never refused.


def _created_paths(ctx: AgentContext) -> set[str]:
    return ctx.shared.setdefault("_ws_created_files", set())


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


def _record_revision(ctx: AgentContext, ws: Workspace, rel: str) -> None:
    key = str(ws.safe_path(rel))
    snap = _stat(ws, rel)
    if snap is None:
        _revisions(ctx).pop(key, None)
    else:
        _revisions(ctx)[key] = snap


def _check_fresh(ctx: AgentContext, ws: Workspace, rel: str) -> ToolResult | None:
    """``None`` when fresh (or never read this run); a refusal otherwise."""
    key = str(ws.safe_path(rel))
    recorded = _revisions(ctx).get(key)
    if recorded is None:
        return None
    current = _stat(ws, rel)
    if current is not None and current != recorded:
        return ToolResult(
            False, "文件已变更",
            f"文件 {rel} 在上次读取之后被修改（可能是外部改动或其它工具写入）。"
            f"请先重新 read_file 获取最新内容，再执行写入或编辑。")
    return None


# a line copied from read_file's numbered output, e.g. "12: foo"
_LINE_NO_PREFIX = re.compile(r"^\s*\d+:\s")


def _path_params() -> dict:
    return {"type": "object",
            "properties": {"path": {"type": "string", "description": "工作区内相对路径"}},
            "required": ["path"]}


def _read_params() -> dict:
    p = _path_params()
    p["properties"]["offset"] = {
        "type": "integer", "description": "从第几行开始读（1 起）；省略则从文件头"}
    p["properties"]["limit"] = {
        "type": "integer", "description": "最多读取的行数；省略则读到文件尾"}
    return p


def _write_params() -> dict:
    p = _path_params()
    p["properties"]["content"] = {"type": "string", "description": "文件新内容"}
    p["required"].append("content")
    return p


def _edit_params() -> dict:
    p = _path_params()
    p["properties"]["old_text"] = {"type": "string", "description": "要替换的精确文本"}
    p["properties"]["new_text"] = {"type": "string", "description": "替换后的文本"}
    p["properties"]["replace_all"] = {
        "type": "boolean",
        "description": "old_text 出现多处时是否全部替换；默认 false（多处匹配会报错，"
                       "需提供更长且唯一的 old_text）"}
    p["required"] += ["old_text", "new_text"]
    return p


def _list_params() -> dict:
    return {"type": "object",
            "properties": {"dirs": {"type": "array", "items": {"type": "string"},
                                    "description": "只列出这些顶层目录；省略则列全部"}}}


def _search_params() -> dict:
    return {"type": "object",
            "properties": {
                "pattern": {"type": "string",
                            "description": "正则表达式（Python re 语法）"},
                "dir": {"type": "string",
                        "description": "只搜索该子目录；省略则搜全部"},
                "glob": {"type": "string",
                         "description": "文件名过滤，如 *.py；省略不过滤"},
            },
            "required": ["pattern"]}


def _glob_params() -> dict:
    return {"type": "object",
            "properties": {"pattern": {"type": "string",
                                       "description": "通配符模式（fnmatch 语法，"
                                                      "如 *.py、data/*.csv）"}},
            "required": ["pattern"]}


def register_file_tools(
    registry: ToolRegistry,
    workspace_for: Callable[[AgentContext], Workspace],
    *,
    max_read: int = 20000,
) -> dict[str, Callable]:
    """Register read_file/write_file/edit_file/list_files on *registry*.

    ``workspace_for(ctx) -> Workspace`` is host-supplied (it maps a context to
    that user's workspace root). The write/edit tools emit ``file_change`` UI
    events carrying old/new content so a host sink can persist undo actions.

    Returns a ``{kind: reverter}`` mapping (``file_write`` / ``file_edit``) for
    the host's UndoEngine — keyed by mutation *kind* (not tool name), since undo
    reverses domain actions, not tool calls. The bundle stores nothing itself.
    """

    async def read_file(ctx, args):
        rel = (args.get("path") or "").strip()
        if not rel:
            return ToolResult(False, "缺少 path", "缺少 path 参数。")
        ws = workspace_for(ctx)
        try:
            # Disk I/O runs off the event loop (asyncio.to_thread, the same
            # policy as the images/documents bundles): one slow read on a
            # cold/NFS workspace must not stall every concurrent agent run
            # in the process.
            content = await asyncio.to_thread(ws.read, rel)
        except FileNotFoundError:
            return ToolResult(False, "文件不存在", f"文件不存在：{rel}")
        except PermissionError as exc:
            return ToolResult(False, "非法路径", str(exc))
        await asyncio.to_thread(_record_revision, ctx, ws, rel)
        offset = args.get("offset")
        limit = args.get("limit")
        if offset is not None or limit is not None:
            # line-window read: the model paginates long files instead of
            # paying for a head-only truncation of the interesting part.
            lines = content.splitlines()
            total = len(lines)
            start = max(int(offset) - 1, 0) if offset is not None else 0
            if start >= total:
                return ToolResult(True, "超出范围",
                                  f"{rel} 共 {total} 行，第 {start + 1} 行起已超出文件尾。")
            n = int(limit) if limit is not None else total - start
            window = lines[start:start + n]
            # every line carries its 1-based number — the model builds unique
            # old_text spans and patch @@ contexts off these numbers
            shown = "\n".join(f"{start + j + 1}: {line}"
                              for j, line in enumerate(window))[:max_read]
            end = start + len(window)
            return ToolResult(
                True, f"读取 {rel} 第 {start + 1}–{end} 行",
                shown + f"\n\n（{rel} 第 {start + 1}–{end} 行，共 {total} 行）")
        numbered = "\n".join(f"{i}: {line}"
                             for i, line in enumerate(content.splitlines(), 1))
        shown = numbered[:max_read]
        if len(numbered) > max_read:
            shown += f"\n\n……（已截断，文件共 {len(content)} 字符）"
        return ToolResult(True, f"读取 {rel}", shown)

    async def write_file(ctx, args):
        rel = (args.get("path") or "").strip()
        content = args.get("content", "")
        if not rel:
            return ToolResult(False, "缺少 path", "缺少 path 参数。")
        ws = workspace_for(ctx)
        stale = await asyncio.to_thread(_check_fresh, ctx, ws, rel)
        if stale is not None:
            return stale
        guard = await asyncio.to_thread(_check_run_growth, ctx, ws, [rel])
        if guard is not None:
            return guard

        def _read_old():
            return ws.read(rel) if ws.exists(rel) else None

        old = await asyncio.to_thread(_read_old)
        try:
            await asyncio.to_thread(ws.write, rel, content)
        except (PermissionError, ValueError, RuntimeError) as exc:
            return ToolResult(False, "写入失败", str(exc))
        await asyncio.to_thread(_record_revision, ctx, ws, rel)
        if old is None:
            await asyncio.to_thread(_note_created, ctx, ws, rel)
        added, removed = await asyncio.to_thread(_line_delta, old, content)
        ui = await asyncio.to_thread(_file_change, "write", rel, old, content)
        return ToolResult(True, f"写入 {rel}{_delta_note(added, removed, created=old is None)}",
                          f"已写入 {rel}（{len(content)} 字符）。旧版本可撤销。",
                          ui=[ui])

    async def edit_file(ctx, args):
        rel = (args.get("path") or "").strip()
        old_text = args.get("old_text") or ""
        new_text = args.get("new_text") or ""
        replace_all = bool(args.get("replace_all"))
        if not rel:
            return ToolResult(False, "缺少 path", "缺少 path 参数。")
        if not old_text:
            return ToolResult(False, "缺少 old_text", "edit 需要精确的 old_text。")
        if old_text == new_text:
            return ToolResult(False, "无变化", "old_text 与 new_text 相同，无需编辑。")
        ws = workspace_for(ctx)
        try:
            content = await asyncio.to_thread(ws.read, rel)
        except FileNotFoundError:
            return ToolResult(False, "文件不存在", f"文件不存在：{rel}")
        except PermissionError as exc:
            return ToolResult(False, "非法路径", str(exc))
        stale = await asyncio.to_thread(_check_fresh, ctx, ws, rel)
        if stale is not None:
            return stale
        count = content.count(old_text)
        fuzzy_note = ""
        if count == 0:
            # Exact substring failed — fall back to a whole-line ladder match
            # (shared with apply_patch): tolerates stray trailing whitespace,
            # missing indentation and typographic Unicode punctuation, and
            # strips "N: " prefixes copied from read_file's numbered output.
            # The replaced span becomes whole lines, apply_patch-style.
            lines = content.split("\n")
            pattern = old_text.split("\n")
            # Mirror apply_patch's trailing-blank alignment: when the model's
            # old_text carries a phantom trailing newline, drop it AND the
            # replacement's counterpart — otherwise every fuzzy hit inserts
            # one extra blank line per edit.
            replacement = new_text.split("\n")
            if pattern and pattern[-1] == "":
                pattern = pattern[:-1]
                if replacement and replacement[-1] == "":
                    replacement = replacement[:-1]
            hits = line_span_hits(lines, pattern) if pattern else []
            if not hits:
                stripped = [_LINE_NO_PREFIX.sub("", p, count=1) for p in pattern]
                if any(s != p for s, p in zip(stripped, pattern, strict=True)):
                    hits = line_span_hits(lines, stripped)
                    pattern = stripped
            if not hits:
                return ToolResult(False, "未匹配", f"在 {rel} 中未找到 old_text。")
            if len(hits) > 1 and not replace_all:
                shown = ",".join(str(h + 1) for h in hits[:5])
                more = "…" if len(hits) > 5 else ""
                return ToolResult(
                    False, "匹配多处",
                    f"old_text（模糊整行匹配）在 {rel} 中出现 {len(hits)} 次"
                    f"（第 {shown}{more} 行）。请提供更长且唯一的 old_text，"
                    f"或设置 replace_all=true 全部替换。")
            new_lines = lines[:]
            for start in reversed(hits):
                new_lines[start:start + len(pattern)] = replacement
            new_content = "\n".join(new_lines)
            replaced = len(hits)
            fuzzy_note = "（模糊整行匹配：忽略空白/标点差异后定位）"
        elif count > 1 and not replace_all:
            # Old behavior (silent replace-all) let one vague old_text clobber
            # every occurrence. Refuse and show where the matches are so the
            # model can either widen old_text or pass replace_all explicitly.
            lines: list[int] = []
            pos = content.find(old_text)
            while pos != -1 and len(lines) < 5:
                lines.append(content[:pos].count("\n") + 1)
                pos = content.find(old_text, pos + 1)
            more = "…" if count > len(lines) else ""
            return ToolResult(
                False, "匹配多处",
                f"old_text 在 {rel} 中出现 {count} 次（第 "
                f"{','.join(map(str, lines))}{more} 行）。请提供更长且唯一的 "
                f"old_text，或设置 replace_all=true 全部替换。")
        else:
            new_content = (content.replace(old_text, new_text) if replace_all
                           else content.replace(old_text, new_text, 1))
            replaced = count if replace_all else 1
        try:
            await asyncio.to_thread(ws.write, rel, new_content)
        except (PermissionError, ValueError, RuntimeError) as exc:
            return ToolResult(False, "写入失败", str(exc))
        await asyncio.to_thread(_record_revision, ctx, ws, rel)
        diff = await asyncio.to_thread(_compact_diff, rel, content, new_content)
        body = f"已替换 {rel} 中的指定文本（{replaced} 处）{fuzzy_note}。"
        if diff:
            body += "\n\n" + diff
        added, removed = await asyncio.to_thread(_line_delta, content, new_content)
        ui = await asyncio.to_thread(_file_change, "edit", rel, content,
                                     new_content)
        return ToolResult(True, f"编辑 {rel}{_delta_note(added, removed)}", body,
                          ui=[ui])

    async def search_files(ctx, args):
        pattern = (args.get("pattern") or "").strip()
        if not pattern:
            return ToolResult(False, "缺少 pattern", "search_files 需要 pattern。")
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return ToolResult(False, "正则无效", f"pattern 不是合法正则：{exc}")
        rel_dir = (args.get("dir") or "").strip()
        name_pat = (args.get("glob") or "").strip()
        ws = workspace_for(ctx)
        try:
            base = (await asyncio.to_thread(ws.safe_path, rel_dir)
                    if rel_dir else ws.root)
        except PermissionError as exc:
            return ToolResult(False, "非法路径", str(exc))
        if rel_dir and not await asyncio.to_thread(base.is_dir):
            return ToolResult(False, "目录不存在", f"目录不存在：{rel_dir}")
        prefix = rel_dir.rstrip("/") + "/" if rel_dir else ""
        matches: list[str] = []
        truncated = False
        deadline = time.monotonic() + _SEARCH_TIME_BUDGET
        scanned = skipped_binary = skipped_big = skipped_long = 0
        out_of_time = False

        def _collect():
            # The walk itself is disk-bound (a full sorted rglob of the
            # tree); filtering is pure Python and rides along for free.
            return [(rel, p) for rel, p in ws.walk()
                    if (not prefix or rel.startswith(prefix))
                    and (not name_pat
                         or fnmatch.fnmatch(rel, name_pat)
                         or fnmatch.fnmatch(p.name, name_pat))]

        def _read_entry(p: Path):
            """(tag, text): 'big' / 'os' skip reasons, else file text."""
            try:
                if p.stat().st_size > _SEARCH_MAX_FILE:
                    return "big", None
                return "", p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return "os", None

        # The tree walk materializes off the loop (walk() sorts the whole
        # rglob before yielding anything, so nothing is lost vs. lazy scan),
        # then files are read in thread batches: a slow disk never stalls
        # concurrent runs, and the awaits between batches keep the tool
        # timeout responsive.
        entries = await asyncio.to_thread(_collect)
        scanned = len(entries)
        for start in range(0, len(entries), _SEARCH_READ_BATCH):
            batch = entries[start:start + _SEARCH_READ_BATCH]
            texts = await asyncio.gather(
                *[asyncio.to_thread(_read_entry, p) for _, p in batch])
            for (rel, _p), (tag, text) in zip(batch, texts, strict=True):
                if tag == "big":
                    skipped_big += 1
                    continue
                if tag == "os":
                    continue
                if "\x00" in text[:1024]:
                    skipped_binary += 1
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    if (lineno % _SEARCH_YIELD_EVERY == 0
                            and time.monotonic() > deadline):
                        out_of_time = True
                        break
                    if len(line) > _SEARCH_MAX_LINE:
                        skipped_long += 1
                        continue
                    if rx.search(line):
                        if len(matches) >= _SEARCH_MAX_MATCHES:
                            truncated = True
                            break
                        matches.append(
                            f"{rel}:{lineno}: {line.strip()[:_MATCH_LINE_CAP]}")
                if out_of_time or truncated:
                    break
            if out_of_time or truncated:
                break
        if not matches:
            notes = [f"扫描 {scanned} 个文件"]
            if skipped_binary:
                notes.append(f"跳过二进制 {skipped_binary}")
            if skipped_big:
                notes.append(f"跳过超大 {skipped_big}")
            if skipped_long:
                notes.append(f"跳过长行 {skipped_long}")
            if out_of_time:
                notes.append("已达时间上限，扫描未完成")
            return ToolResult(True, "无匹配" if not out_of_time else "未扫完",
                              f"未找到匹配（{'，'.join(notes)}）。")
        body = "\n".join(matches)
        notes = []
        if truncated:
            notes.append(f"匹配过多，仅显示前 {_SEARCH_MAX_MATCHES} 条；"
                         f"请收窄 pattern 或用 glob 过滤")
        if out_of_time:
            notes.append("已达扫描时间上限，结果可能不完整；请用 dir/glob 收窄范围")
        if skipped_long:
            notes.append(f"跳过 {skipped_long} 条超长行（>10k 字符）")
        if notes:
            body += "\n…（" + "；".join(notes) + "）"
        return ToolResult(True, f"{len(matches)} 处匹配", f"匹配结果：\n{body}")

    async def glob_files(ctx, args):
        pattern = (args.get("pattern") or "").strip()
        if not pattern:
            return ToolResult(False, "缺少 pattern", "glob_files 需要 pattern。")
        ws = workspace_for(ctx)

        def _scan():
            hits: list[str] = []
            truncated = False
            for rel, p in ws.walk():
                if not (fnmatch.fnmatch(rel, pattern)
                        or fnmatch.fnmatch(p.name, pattern)):
                    continue
                if len(hits) >= _GLOB_MAX_RESULTS:
                    truncated = True
                    break
                hits.append(f"- {rel} ({p.stat().st_size}B)")
            return hits, truncated

        hits, truncated = await asyncio.to_thread(_scan)
        if not hits:
            return ToolResult(True, "无匹配", f"没有匹配 {pattern} 的文件。")
        body = "\n".join(hits)
        if truncated:
            body += f"\n…（结果过多，仅显示前 {_GLOB_MAX_RESULTS} 条）"
        return ToolResult(True, f"{len(hits)} 个文件", body)

    async def list_files(ctx, args):
        subdirs = tuple(args.get("dirs") or ())
        ws = workspace_for(ctx)
        entries = await asyncio.to_thread(ws.list, subdirs)
        if not entries:
            return ToolResult(True, "空", "工作区无文件。")
        lines = [f"- [{e['type'][:1]}] {e['path']} ({e['size']}B)" for e in entries]
        return ToolResult(True, f"{len(entries)} 项", "\n".join(lines))

    def _revert_write(action, ctx):
        ws = workspace_for(ctx)
        if action.old_value is None:
            ws.delete(action.target)
        else:
            ws.write(action.target, action.old_value)

    def _revert_edit(action, ctx):
        ws = workspace_for(ctx)
        if action.old_value is not None:
            ws.write(action.target, action.old_value)

    registry.register(
        ToolSpec("read_file", "读取工作区内一个文件的内容，每行带 行号: 前缀"
                              "（大文件可用 offset/limit 分行阅读）。",
                 _read_params(), ToolCategory.READ),
        read_file)
    registry.register(
        ToolSpec("write_file", "写入或覆盖工作区内一个文件（可撤销）。"
                              "若该文件本轮读取后又被外部修改，会拒绝写入并要求先重新读取。",
                 _write_params(), ToolCategory.WRITE),
        write_file, reverter=_revert_write, revert_kind="file_write")
    registry.register(
        ToolSpec("edit_file", "局部替换文件中的文本 old_text→new_text（可撤销）。"
                              "old_text 必须唯一；多处匹配时提供更长上下文或设置 "
                              "replace_all。精确匹配失败时按整行模糊匹配兜底"
                              "（容忍行尾空白/缩进/中文标点差异及误带的行号前缀）。",
                 _edit_params(), ToolCategory.WRITE),
        edit_file, reverter=_revert_edit, revert_kind="file_edit")
    registry.register(
        ToolSpec("list_files", "列出工作区文件。", _list_params(), ToolCategory.READ),
        list_files)
    registry.register(
        ToolSpec("search_files", "在工作区文件内容中按正则搜索，返回 path:行号: 行。"
                                 "可用 dir 限定子目录、glob 过滤文件名。"
                                 "超长行（>10k 字符）会被跳过。",
                 _search_params(), ToolCategory.READ,
                 timeout=30.0),
        search_files)
    registry.register(
        ToolSpec("glob_files", "按通配符模式查找工作区文件路径（如 *.py、data/*.csv）。",
                 _glob_params(), ToolCategory.READ),
        glob_files)

    return {"file_write": _revert_write, "file_edit": _revert_edit}
