"""Codex-style ``apply_patch``: a line-oriented multi-file patch tool.

The patch language is the OpenAI Codex ``apply_patch`` envelope — a
stripped-down, file-oriented diff format models emit reliably::

    *** Begin Patch
    *** Add File: hello.txt
    +Hello world
    *** Update File: src/app.py
    @@ def greet():
    -print("Hi")
    +print("Hello, world!")
    *** Delete File: obsolete.txt
    *** End Patch

Design references (semantics only; this is an independent implementation):
``opencode``'s ``packages/opencode/src/patch/index.ts`` (MIT) and OpenAI's
``codex-rs/apply-patch`` (Apache-2.0). Deliberate divergences, both directions:

- *Stricter parsing* than the opencode repo module: a non-blank line that is
  not a file header / chunk line is an error, never silently skipped (a model
  must learn its patch was malformed, not believe it applied). Blank separator
  lines between sections are the one thing tolerated.
- *All-or-nothing application*: every hunk is resolved and derived BEFORE the
  first write, against an in-memory overlay chained hunk-over-hunk (a later
  Update of the same file sees an earlier hunk's effect, and an Update of a
  file an earlier Add created is legal), so a bad hunk leaves the workspace
  untouched — lithe has an UndoEngine, partial application buys nothing.
  The disk is then touched once per path, in one commit pass.

The tool emits ``file_change`` UI events (old/new carried) exactly like
``write_file``/``edit_file``, so host sinks and the undo path work unchanged;
``action="delete"`` pairs with the new ``file_delete`` reverter kind returned
here (a ``move`` is applied as write-destination + delete-source and emits one
event for each side, which the full-content reverters undo cleanly).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lithe.context import AgentContext
from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec

_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_ADD = "*** Add File:"
_DEL = "*** Delete File:"
_UPD = "*** Update File:"
_MOVE = "*** Move to:"
_EOF_MARKER = "*** End of File"

_HEREDOC_RE = re.compile(r"^(?:cat\s+)?<<['\"]?(\w+)['\"]?\s*\n([\s\S]*?)\n\1\s*$")


class PatchError(ValueError):
    """A patch that cannot be parsed (structure) or derived (content)."""


class PatchFormatError(PatchError):
    """The patch text violates the envelope/section/chunk grammar."""


class PatchApplyError(PatchError):
    """A chunk could not be located in (or reconciled with) the target file."""


# --- patch model ---------------------------------------------------------


@dataclass
class AddFile:
    path: str
    contents: str


@dataclass
class DeleteFile:
    path: str


@dataclass
class UpdateChunk:
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    # A locator line after ``@@``: seek it first, then match old_lines after it.
    change_context: str | None = None
    is_end_of_file: bool = False


@dataclass
class UpdateFile:
    path: str
    chunks: list[UpdateChunk] = field(default_factory=list)
    move_path: str | None = None


Hunk = AddFile | DeleteFile | UpdateFile


@dataclass
class DerivedUpdate:
    new_text: str
    bom: bool


# --- parsing ---------------------------------------------------------------


def strip_heredoc(text: str) -> str:
    """Unwrap a ``cat <<'EOF' ... EOF`` wrapper models sometimes add."""
    m = _HEREDOC_RE.match(text)
    return m.group(2) if m else text


def parse_patch(patch_text: str) -> list[Hunk]:
    """Parse the envelope into ordered hunks. Raises :class:`PatchFormatError`.

    Grammar (strict): between Begin/End markers only blank lines, file headers,
    and — inside an Update section — ``@@`` chunks of `` ``/``-``/``+`` lines
    plus one ``*** End of File`` anchor per chunk are accepted.
    """
    lines = strip_heredoc(patch_text.strip()).split("\n")
    begin = end = None
    for i, line in enumerate(lines):
        if line.strip() == _BEGIN and begin is None:
            begin = i
        elif line.strip() == _END:
            end = i
            break
    if begin is None or end is None or begin >= end:
        raise PatchFormatError("缺少 *** Begin Patch / *** End Patch 标记")

    hunks: list[Hunk] = []
    i = begin + 1
    while i < end:
        line = lines[i]
        if not line.strip():  # tolerate blank separators between sections
            i += 1
            continue

        if line.startswith(_ADD):
            path = line[len(_ADD) :].strip()
            if not path:
                raise PatchFormatError("Add File 缺少路径")
            i += 1
            body: list[str] = []
            while i < end and not lines[i].startswith("***"):
                if not lines[i].startswith("+"):
                    raise PatchFormatError(
                        f"Add File 的内容行必须以 + 开头：{lines[i]!r}"
                    )
                body.append(lines[i][1:])
                i += 1
            contents = "\n".join(body)
            if contents and not contents.endswith("\n"):
                contents += "\n"
            hunks.append(AddFile(path, contents))

        elif line.startswith(_DEL):
            path = line[len(_DEL) :].strip()
            if not path:
                raise PatchFormatError("Delete File 缺少路径")
            hunks.append(DeleteFile(path))
            i += 1

        elif line.startswith(_UPD):
            path = line[len(_UPD) :].strip()
            if not path:
                raise PatchFormatError("Update File 缺少路径")
            i += 1
            move_path: str | None = None
            if i < end and lines[i].startswith(_MOVE):
                move_path = lines[i][len(_MOVE) :].strip()
                if not move_path:
                    raise PatchFormatError("Move to 缺少路径")
                i += 1
            chunks: list[UpdateChunk] = []
            while i < end and not lines[i].startswith("***"):
                if not lines[i].strip():  # blank: chunk/section separator
                    i += 1
                    continue
                if not lines[i].startswith("@@"):
                    raise PatchFormatError(
                        f"Update File 内期望 @@ chunk，得到：{lines[i]!r}"
                    )
                change_context = lines[i][2:].strip() or None
                i += 1
                old_lines: list[str] = []
                new_lines: list[str] = []
                is_eof = False
                while (
                    i < end
                    and not lines[i].startswith("@@")
                    and not lines[i].startswith("***")
                ):
                    change = lines[i]
                    if change == "":
                        break  # truly empty line: separator, chunk ends
                    # A lone " " is the canonical encoding of a BLANK CONTEXT
                    # line (common in real multi-function edits) — treating it
                    # as a terminator silently truncated chunks and let the
                    # fuzzy ladder mis-anchor the shortened pattern.
                    if change.startswith(" "):
                        old_lines.append(change[1:])
                        new_lines.append(change[1:])
                    elif change.startswith("-"):
                        old_lines.append(change[1:])
                    elif change.startswith("+"):
                        new_lines.append(change[1:])
                    else:
                        raise PatchFormatError(
                            f"chunk 内的行须以空格/-/+ 开头：{change!r}"
                        )
                    i += 1
                if i < end and lines[i].strip() == _EOF_MARKER:
                    is_eof = True
                    i += 1
                chunks.append(UpdateChunk(old_lines, new_lines, change_context, is_eof))
            if not chunks:
                raise PatchFormatError(f"Update File {path} 至少需要一个 @@ chunk")
            hunks.append(UpdateFile(path, chunks, move_path))

        else:
            raise PatchFormatError(f"无法识别的 patch 行：{line!r}")

    return hunks


# --- line seeking (the fuzzy ladder) ---------------------------------------
# The ladder lives in ``_textmatch`` (shared with ``edit_file``'s whole-line
# fallback); re-exported here so patch callers keep one import point.

from lithe.bundles._textmatch import (  # noqa: E402
    seek_sequence,
    seek_tail,
)


# --- deriving new contents --------------------------------------------------

_MAX_FAIL_CHUNKS = 4   # failed chunks listed verbatim before the "more" note
_MAX_FAIL_LINES = 5    # missing lines echoed per failed chunk


def _chunk_error(path: str, seq: int, reason: str,
                 old_lines: list[str] | None = None) -> str:
    """One failed chunk, rendered for the combined patch error."""
    note = f"[{path} 第 {seq + 1} 个 chunk] {reason}"
    if old_lines:
        shown = old_lines[:_MAX_FAIL_LINES]
        note += "：\n" + "\n".join(shown)
        if len(old_lines) > len(shown):
            note += f"\n……共 {len(old_lines)} 行"
    return note


def derive_new_contents(
    path: str, chunks: list[UpdateChunk], original_text: str
) -> DerivedUpdate:
    """Apply update chunks to *original_text*, returning the full new text.

    Chunks are located sequentially (each search starts after the previous
    match), collected as ``[start, delete_count, insert_lines]`` splices, then
    applied back-to-front so earlier indices never shift. Same-position
    splices (two pure insertions at end of file) tie-break on chunk order so
    they land in document order, not reversed. A chunk marked
    ``*** End of File`` runs the tail-anchored comparator ladder at full
    strength BEFORE any forward match — a whitespace-mismatched tail wins
    over an exact look-alike earlier in the file, instead of silently
    editing the wrong site. Updates normalize the file to a trailing newline
    (reference behavior); a UTF-8 BOM survives.

    A chunk that cannot be located does not stop the scan: every failure is
    recorded and the scan continues from the last good position, so the one
    final raise reports ALL bad chunks at once instead of surfacing them one
    retry round-trip at a time.
    """
    bom = original_text.startswith("\ufeff")
    text = original_text[1:] if bom else original_text
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    # [start, delete_count, insert_lines, chunk_seq] — the seq breaks ties
    # for same-position splices (back-to-front application of equal starts
    # would otherwise reverse two insertions issued in one section).
    splices: list[tuple[int, int, list[str], int]] = []
    errors: list[str] = []
    search_from = 0
    for seq, chunk in enumerate(chunks):
        if chunk.change_context is not None:
            ctx = seek_sequence(lines, [chunk.change_context], search_from)
            if ctx == -1:
                errors.append(_chunk_error(
                    path, seq, f"未找到上下文行 {chunk.change_context!r}"))
                continue
            search_from = ctx + 1

        if not chunk.old_lines:  # pure insertion at end of file
            insert_at = len(lines) - 1 if lines and lines[-1] == "" else len(lines)
            splices.append((insert_at, 0, list(chunk.new_lines), seq))
            continue

        pattern = list(chunk.old_lines)
        new_slice = list(chunk.new_lines)
        # models routinely mirror a phantom trailing blank; drop it and its
        # replacement counterpart before seeking
        if pattern[-1] == "":
            pattern = pattern[:-1]
            if new_slice and new_slice[-1] == "":
                new_slice = new_slice[:-1]
        if chunk.is_end_of_file:
            # Tail-dedicated ladder FIRST, at full strength: previously the
            # eof preference was interleaved comparator-by-comparator, so an
            # exact forward hit at a look-alike earlier site beat a
            # whitespace-tolerant tail match and silently edited the wrong
            # place. Only when the tail fails every comparator does the
            # forward ladder run (the codex fallback for markers the model
            # attaches to non-tail patterns).
            found = seek_tail(lines, pattern)
            if found == -1:
                found = seek_sequence(lines, pattern, search_from)
            if found == -1:
                errors.append(_chunk_error(
                    path, seq,
                    "*** End of File 锚定的 chunk 在文件中未找到",
                    chunk.old_lines))
                continue
        else:
            found = seek_sequence(lines, pattern, search_from)
            if found == -1:
                errors.append(_chunk_error(
                    path, seq, "未找到要替换的行", chunk.old_lines))
                continue
        splices.append((found, len(pattern), new_slice, seq))
        search_from = found + len(pattern)

    if errors:
        shown = errors[:_MAX_FAIL_CHUNKS]
        if len(errors) > len(shown):
            shown.append(f"……另有 {len(errors) - len(shown)} 处 chunk 定位失败")
        raise PatchApplyError(
            f"{len(errors)} 处 chunk 定位失败：\n" + "\n".join(shown))

    result = list(lines)
    for start, delete_count, insert_lines, _seq in sorted(
        splices, key=lambda s: (s[0], s[3]), reverse=True
    ):
        result[start : start + delete_count] = insert_lines
    if not result or result[-1] != "":
        result.append("")

    new_text = "\n".join(result)
    return DerivedUpdate(("\ufeff" if bom else "") + new_text, bom)


# --- tool registration ------------------------------------------------------


def _patch_params() -> dict:
    return {
        "type": "object",
        "properties": {
            "patch_text": {"type": "string", "description": "完整 patch 文本"}
        },
        "required": ["patch_text"],
    }


_DESCRIPTION = (
    "以行级 patch 一次修改多个文件（新增/更新/删除，可撤销、可移动）。patch 格式：\n"
    "*** Begin Patch 与 *** End Patch 之间是若干文件节；每节以三个头之一开始：\n"
    "*** Add File: <路径>（新文件，随后每行以 + 前缀给出全文）；\n"
    "*** Delete File: <路径>（删除文件，无内容行）；\n"
    "*** Update File: <路径>（可再跟 *** Move to: <新路径> 实现改名）。\n"
    "Update 节由若干 @@ chunk 组成：@@ 后可跟一行定位上下文；chunk 内空格前缀是"
    "上下文行（上下文中的空行写成单个空格），- 是删除行，+ 是新增行；"
    "*** End of File 表示从文件尾锚定。\n"
    "定位按 精确→忽略行尾空白→忽略首尾空白→Unicode标点归一→反斜杠折叠 的顺序匹配；"
    "写盘前先整体校验全部 hunk，任何 hunk 定位失败时一次列出全部失败 chunk，"
    "且不会改动任何文件。"
)


def register_apply_patch_tool(
    registry: ToolRegistry,
    workspace_for,
) -> dict:
    """Register ``apply_patch`` on *registry* against a host workspace.

    Returns reverters for the host's UndoEngine keyed by mutation kind. Only
    ``file_delete`` is new here: add/update reuse the ``file_write`` /
    ``file_edit`` kinds whose reverters ``register_file_tools`` already
    supplies (hosts merge the two maps). A move emits one ``write`` event for
    the destination plus one ``delete`` event for the source, so undo restores
    both sides from the carried old/new contents.
    """

    async def apply_patch(ctx: AgentContext, args: dict) -> ToolResult:
        from lithe.bundles.workspace import (
            _check_fresh, _check_run_growth, _compact_diff, _delta_note,
            _file_change, _note_created, _record_revision, _ws_write_lock,
        )

        patch_text = args.get("patch_text") or ""
        if not patch_text.strip():
            return ToolResult(False, "缺少 patch_text", "apply_patch 需要 patch_text。")
        try:
            hunks = parse_patch(patch_text)
        except PatchFormatError as exc:
            return ToolResult(False, "patch 无效", f"patch 解析失败：{exc}")
        if not hunks:
            return ToolResult(False, "空 patch", "patch 不包含任何文件操作。")
        ws = workspace_for(ctx)

        # Phase 1 — resolve + derive against an in-memory overlay, hunk after
        # hunk (a later section of the same file sees an earlier section's
        # effect). No writes: any failure here leaves the workspace untouched.
        overlay: dict[str, str | None] = {}  # path -> final content (None = gone)
        ui: list[dict] = []
        applied: list[str] = []
        diff_parts: list[str] = []

        def current(rel: str) -> str | None:
            """Content before the next hunk: overlay if touched, else disk."""
            if rel in overlay:
                return overlay[rel]
            return ws.read(rel) if ws.exists(rel) else None

        # The freshness checks in phase 1 and the commit in phase 2 must be
        # one unit against parallel subagent writers: without the run-wide
        # mutation lock (shared with every subagent context), a concurrent
        # write_file can land between the checks and the commit and be
        # silently overwritten. All-or-nothing derivation is unaffected —
        # it is in-memory — but its inputs (current()/freshness) are read
        # under the lock so the commit lands on exactly what was checked.
        async with _ws_write_lock(ctx):
            # Per-hunk location failures are collected across ALL files so
            # one rejection lists every bad chunk (see derive_new_contents);
            # a failure skips just its own section — nothing is written
            # unless the list is empty at the end (all-or-nothing).
            apply_errors: list[str] = []
            try:
                for hunk in hunks:
                    ws.safe_path(hunk.path)  # traversal check up front
                    stale = _check_fresh(ctx, ws, hunk.path)
                    if stale is not None:
                        return stale
                    if isinstance(hunk, UpdateFile) and hunk.move_path:
                        stale = _check_fresh(ctx, ws, hunk.move_path)
                        if stale is not None:
                            return stale
                    if isinstance(hunk, AddFile):
                        old = current(hunk.path)
                        overlay[hunk.path] = hunk.contents
                        ui.append(_file_change("write", hunk.path, old, hunk.contents))
                        applied.append(f"A {hunk.path}")
                    elif isinstance(hunk, DeleteFile):
                        old = current(hunk.path)
                        if old is None:
                            apply_errors.append(f"[{hunk.path}] 文件不存在")
                            continue
                        overlay[hunk.path] = None
                        ui.append(_file_change("delete", hunk.path, old, None))
                        applied.append(f"D {hunk.path}")
                    else:
                        if hunk.move_path:
                            ws.safe_path(hunk.move_path)
                        old = current(hunk.path)
                        if old is None:
                            apply_errors.append(f"[{hunk.path}] 文件不存在")
                            continue
                        try:
                            new_text = derive_new_contents(
                                hunk.path, hunk.chunks, old).new_text
                        except PatchApplyError as exc:
                            apply_errors.append(str(exc))
                            continue
                        if hunk.move_path:
                            dest_old = current(hunk.move_path)
                            overlay[hunk.move_path] = new_text
                            overlay[hunk.path] = None
                            ui.append(
                                _file_change("write", hunk.move_path, dest_old, new_text)
                            )
                            ui.append(_file_change("delete", hunk.path, old, None))
                            applied.append(f"M {hunk.path} -> {hunk.move_path}")
                        else:
                            overlay[hunk.path] = new_text
                            ui.append(_file_change("edit", hunk.path, old, new_text))
                            applied.append(f"M {hunk.path}")
                        diff = _compact_diff(hunk.path, old, new_text)
                        if diff:
                            diff_parts.append(diff)
            except PatchApplyError as exc:
                return ToolResult(False, "未应用", f"patch 未做任何修改。{exc}")
            except PermissionError as exc:
                return ToolResult(False, "非法路径", f"patch 未做任何修改。{exc}")
            except (OSError, RuntimeError, ValueError) as exc:
                return ToolResult(False, "预检失败", f"patch 未做任何修改：{exc}")
            if apply_errors:
                return ToolResult(
                    False, "未应用",
                    "patch 未做任何修改。\n" + "\n".join(apply_errors))

            # Phase 2 — commit: one write/delete per touched path. A mid-commit
            # failure reports what landed (the events carry full undo data).
            new_rels = [rel for rel, value in overlay.items()
                        if value is not None and not ws.exists(rel)]
            guard = _check_run_growth(ctx, ws, new_rels)
            if guard is not None:
                return guard
            landed: list[str] = []
            try:
                for rel, value in overlay.items():
                    if value is None:
                        if ws.exists(rel):
                            ws.delete(rel)
                    else:
                        ws.write(rel, value)
                    _record_revision(ctx, ws, rel)  # refresh (or drop) the snapshot
                    landed.append(rel)
            except (PermissionError, ValueError, RuntimeError, OSError) as exc:
                done = "、".join(landed) if landed else "无"
                return ToolResult(
                    False,
                    "部分应用后失败",
                    f"patch 写入失败：{exc}。已落盘：{done}。",
                    ui=ui,
                )
            for rel in new_rels:
                _note_created(ctx, ws, rel)

        body = "已应用 patch：\n" + "\n".join(applied)
        if diff_parts:
            body += "\n\n" + "\n\n".join(diff_parts)
        added = sum(ev.get("added") or 0 for ev in ui)
        removed = sum(ev.get("removed") or 0 for ev in ui)
        return ToolResult(
            True, f"应用 {len(applied)} 项{_delta_note(added, removed)}", body, ui=ui)

    def _revert_delete(action, ctx):
        if action.old_value is not None:
            workspace_for(ctx).write(action.target, action.old_value)

    registry.register(
        ToolSpec("apply_patch", _DESCRIPTION, _patch_params(), ToolCategory.WRITE),
        apply_patch,
        reverter=_revert_delete, revert_kind="file_delete",
    )
    return {"file_delete": _revert_delete}
