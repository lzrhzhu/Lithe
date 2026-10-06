"""lithe.bundles.patch: the apply_patch tool — parser, the fuzzy-seek
ladder, all-or-nothing application, sandbox/UI contracts and undo reverters.

Scenario set ported from codex-rs/apply-patch's own test suite (multi-chunk
updates, interleaved edits, pure-insertion ordering, unicode-dash fuzzy
matching, moves, EOF appends) plus lithe-specific guards (traversal,
all-or-nothing preflight, file_delete reverter)."""

from __future__ import annotations

import pytest

from lithe import Action, AgentContext, ToolRegistry, UndoEngine
from lithe.bundles.patch import (
    AddFile,
    DeleteFile,
    PatchApplyError,
    PatchFormatError,
    UpdateChunk,
    UpdateFile,
    derive_new_contents,
    parse_patch,
    register_apply_patch_tool,
    seek_sequence,
    strip_heredoc,
)
from lithe.bundles.workspace import Workspace, register_file_tools


def _wrap(body: str) -> str:
    return f"*** Begin Patch\n{body}\n*** End Patch"


def _registry_with(ws):
    reg = ToolRegistry()
    reverters = register_file_tools(reg, lambda ctx: ws)
    reverters.update(register_apply_patch_tool(reg, lambda ctx: ws))
    return reg, reverters


def _ctx():
    return AgentContext(run_id="r", user_id="u")


# --- parser ---------------------------------------------------------------


def test_parse_rejects_missing_markers():
    with pytest.raises(PatchFormatError):
        parse_patch("not a patch")
    with pytest.raises(PatchFormatError):
        parse_patch("*** Begin Patch\n+orphan\n*** End Patch")


def test_parse_rejects_unknown_line_strictly():
    with pytest.raises(PatchFormatError):
        parse_patch(_wrap("some random line"))
    # a chunk line without a @@ opener is not silently skipped
    with pytest.raises(PatchFormatError):
        parse_patch(_wrap("*** Update File: a.py\n-broken"))


def test_parse_add_delete_update_shapes():
    hunks = parse_patch(
        _wrap(
            "*** Add File: hello.txt\n"
            "+Hello world\n"
            "*** Update File: src/app.py\n"
            "*** Move to: src/main.py\n"
            "@@ def greet():\n"
            '-print("Hi")\n'
            '+print("Hello, world!")\n'
            "*** Delete File: obsolete.txt"
        )
    )
    assert hunks == [
        AddFile("hello.txt", "Hello world\n"),
        UpdateFile(
            "src/app.py",
            [
                UpdateChunk(
                    ['print("Hi")'], ['print("Hello, world!")'], "def greet():", False
                )
            ],
            "src/main.py",
        ),
        DeleteFile("obsolete.txt"),
    ]


def test_parse_eof_marker_and_blank_separators():
    # *** End of File is a chunk-TAIL anchor (marks the chunk as EOF-bound);
    # blank lines are tolerated between sections.
    hunks = parse_patch(
        _wrap(
            "*** Update File: a.txt\n"
            "@@\n"
            " last\n"
            "+appended\n"
            "*** End of File\n"
            "\n"
            "*** Delete File: b.txt"
        )
    )
    assert hunks[0].chunks[0].old_lines == ["last"]
    assert hunks[0].chunks[0].new_lines == ["last", "appended"]
    assert hunks[0].chunks[0].is_end_of_file is True
    assert hunks[1] == DeleteFile("b.txt")


def test_strip_heredoc_wrapper():
    wrapped = "cat <<'EOF'\n*** Begin Patch\n*** End Patch\nEOF"
    assert strip_heredoc(wrapped) == "*** Begin Patch\n*** End Patch"


# --- seek ladder / derive ----------------------------------------------------


def test_seek_ladder_exact_to_unicode_fold():
    lines = ["x = 1", 'msg = "hi"', "dash – here", "tail"]
    assert seek_sequence(lines, ["x = 1"]) == 0
    # trailing whitespace tolerated (rstrip pass)
    assert seek_sequence(lines, ["x = 1   "]) == 0
    # leading whitespace tolerated (strip pass)
    assert seek_sequence(lines, ['   msg = "hi"']) == 1
    # typographic dash folded to ASCII (unicode pass)
    assert seek_sequence(lines, ["dash - here"]) == 2
    assert seek_sequence(["nope"], ["absent"]) == -1
    assert seek_sequence(lines, []) == -1


def test_seek_eof_prefers_tail_anchor():
    lines = ["a", "a", "a"]
    assert seek_sequence(lines, ["a"], eof=True) == 2
    assert seek_sequence(lines, ["a"], eof=False) == 0


def test_derive_multiple_chunks_and_interleaved_changes():
    # codex-rs test_update_file_hunk_interleaved_changes: replace, context-
    # anchored replace, EOF append — all in one Update section
    original = "a\nb\nc\nd\ne\nf\n"
    chunks = [
        _chunk(["b"], ["B"]),
        _chunk(["c", "d", "e"], ["c", "d", "E"]),
        _chunk(["f"], ["f", "g"], eof=True),
    ]
    out = derive_new_contents("f.txt", chunks, original)
    assert out.new_text == "a\nB\nc\nd\nE\nf\ng\n"


def _chunk(old, new, ctx=None, eof=False):
    return UpdateChunk(list(old), list(new), ctx, eof)


def test_derive_pure_addition_chunk_then_removal_ordering():
    # codex-rs test_pure_addition_chunk_followed_by_removal: a pure-insert
    # chunk (no old lines) lands at EOF, then the removal chunk still finds
    # its lines — order must not corrupt indices.
    original = "line1\nline2\nline3\n"
    chunks = [
        _chunk([], ["after-context", "second-line"]),
        _chunk(["line1", "line2", "line3"], ["line1", "line2-replacement"]),
    ]
    out = derive_new_contents("p.txt", chunks, original)
    assert out.new_text == "line1\nline2-replacement\nafter-context\nsecond-line\n"


def test_derive_unicode_dash_fuzzy_match():
    original = "import asyncio  # local import \u2013 avoids top\u2011level dep\n"
    chunks = [
        _chunk(
            ["import asyncio  # local import - avoids top-level dep"],
            ["import asyncio  # HELLO"],
        )
    ]
    out = derive_new_contents("u.py", chunks, original)
    assert out.new_text == "import asyncio  # HELLO\n"


def test_derive_change_context_seeks_first():
    original = "def a():\n    x = 1\n\ndef b():\n    x = 1\n"
    # two identical `x = 1` bodies; @@ context pins the chunk to def b()
    chunks = [_chunk(["    x = 1"], ["    x = 2"], ctx="def b():")]
    out = derive_new_contents("m.py", chunks, original)
    lines = out.new_text.splitlines()
    assert lines[1] == "    x = 1" and lines[4] == "    x = 2"


def test_derive_phantom_trailing_blank_retry():
    original = "one\ntwo\n"
    chunks = [_chunk(["one", "two", ""], ["ONE", "TWO", ""])]
    out = derive_new_contents("t.txt", chunks, original)
    assert out.new_text == "ONE\nTWO\n"


def test_derive_missing_lines_raises_with_echo():
    with pytest.raises(PatchApplyError, match="absent"):
        derive_new_contents("t.txt", [_chunk(["absent"], [])], "one\n")


def test_derive_reports_all_failed_chunks():
    """定位失败不再首个即抛：一次 raise 列出全部坏 chunk（带序号与原文）。"""
    original = "one\ntwo\nthree\n"
    chunks = [
        _chunk(["absent"], []),
        _chunk(["two"] + [f"pad {i}" for i in range(6)], []),
    ]
    with pytest.raises(PatchApplyError) as ei:
        derive_new_contents("t.txt", chunks, original)
    msg = str(ei.value)
    assert "2 处 chunk 定位失败" in msg
    assert "第 1 个 chunk" in msg and "absent" in msg
    assert "第 2 个 chunk" in msg and "two" in msg
    assert "……共 7 行" in msg  # 超出逐行回显上限时给出总行数


def test_derive_preserves_bom_and_normalizes_trailing_newline():
    original = "\ufeffa\nb"  # no trailing newline
    out = derive_new_contents("b.txt", [_chunk(["b"], ["c"])], original)
    assert out.bom is True
    assert out.new_text == "\ufeffa\nc\n"


# --- tool: scenarios ported from codex-rs -----------------------------------


async def test_add_update_delete_roundtrip(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("update.txt", "relative old\n")
    ws.write("delete.txt", "gone\n")
    reg, _ = _registry_with(ws)

    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": _wrap(
                "*** Add File: relative-add.txt\n"
                "+relative add\n"
                "*** Update File: update.txt\n"
                "@@\n"
                "-relative old\n"
                "+relative new\n"
                "*** Delete File: delete.txt"
            )
        },
        _ctx(),
    )
    assert r.ok, r.content
    assert ws.read("relative-add.txt") == "relative add\n"
    assert ws.read("update.txt") == "relative new\n"
    assert not ws.exists("delete.txt")
    assert "A relative-add.txt" in r.content
    assert "M update.txt" in r.content
    assert "D delete.txt" in r.content


async def test_move_renames_via_write_plus_delete(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("src.txt", "line\n")
    reg, _ = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": _wrap(
                "*** Update File: src.txt\n*** Move to: dst.txt\n@@\n-line\n+line2"
            )
        },
        _ctx(),
    )
    assert r.ok, r.content
    assert not ws.exists("src.txt")
    assert ws.read("dst.txt") == "line2\n"
    assert "M src.txt -> dst.txt" in r.content


async def test_two_sections_same_file_chain_via_overlay(tmp_path):
    """Regression: a second Update section of the same file must derive
    against the FIRST section's result (in-memory overlay), not the on-disk
    content — otherwise the commit overwrites the earlier change."""
    ws = Workspace(tmp_path)
    ws.write("app.py", 'def greet():\n    print("Hi")\n')
    reg, _ = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": _wrap(
                "*** Update File: app.py\n"
                "@@ def greet():\n"
                '-    print("Hi")\n'
                '+    print("你好")\n'
                "*** Update File: app.py\n"
                "@@\n"
                " def greet():\n"
                '+    print("tail")\n'
                "*** End of File"
            )
        },
        _ctx(),
    )
    assert r.ok, r.content
    assert ws.read("app.py") == 'def greet():\n    print("tail")\n    print("你好")\n'
    # the chained undo events revert cleanly newest-first
    events = [e for e in r.ui if e["action"] == "edit"]
    assert events[0]["new"] == events[1]["old"]


async def test_update_of_file_created_earlier_in_same_patch(tmp_path):
    """The overlay makes a patch that creates then edits a file legal."""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": _wrap(
                "*** Add File: gen.py\n+x = 1\n*** Update File: gen.py\n@@\n-x = 1\n+x = 2"
            )
        },
        _ctx(),
    )
    assert r.ok, r.content
    assert ws.read("gen.py") == "x = 2\n"


async def test_multiple_chunks_and_interleaved_changes(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("multi.txt", "foo\nbar\nbaz\nqux\n")
    ws.write("interleaved.txt", "a\nb\nc\nd\ne\nf\n")
    reg, _ = _registry_with(ws)

    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": _wrap(
                "*** Update File: multi.txt\n"
                "@@\n"
                " foo\n"
                "-bar\n"
                "+BAR\n"
                "@@\n"
                " baz\n"
                "-qux\n"
                "+QUX\n"
                "*** Update File: interleaved.txt\n"
                "@@\n"
                " a\n"
                "-b\n"
                "+B\n"
                "@@\n"
                " c\n"
                " d\n"
                "-e\n"
                "+E\n"
                "@@\n"
                " f\n"
                "+g\n"
                "*** End of File"
            )
        },
        _ctx(),
    )
    assert r.ok, r.content
    assert ws.read("multi.txt") == "foo\nBAR\nbaz\nQUX\n"
    assert ws.read("interleaved.txt") == "a\nB\nc\nd\nE\nf\ng\n"


async def test_unmatched_hunk_leaves_workspace_untouched(tmp_path):
    """All-or-nothing: preflight failure must not apply the good hunks."""
    ws = Workspace(tmp_path)
    ws.write("good.txt", "keep\n")
    reg, _ = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": _wrap(
                "*** Update File: good.txt\n"
                "@@\n"
                "-keep\n"
                "+KEPT\n"
                "*** Update File: missing.txt\n"
                "@@\n"
                "-nothing\n"
                "+anything"
            )
        },
        _ctx(),
    )
    assert not r.ok
    assert "未做任何修改" in r.content
    assert ws.read("good.txt") == "keep\n"  # first hunk was NOT applied


async def test_apply_patch_lists_failures_across_files(tmp_path):
    """两个文件各有一个坏 chunk：一次拒绝同时点名两处，且零落盘——
    不再逐轮只暴露第一个失败。"""
    ws = Workspace(tmp_path)
    ws.write("a.txt", "aaa\n")
    ws.write("b.txt", "bbb\n")
    reg, _ = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {"patch_text": _wrap(
            "*** Update File: a.txt\n@@\n-nope-a\n+A\n"
            "*** Update File: b.txt\n@@\n-nope-b\n+B")},
        _ctx(),
    )
    assert not r.ok
    assert "未做任何修改" in r.content
    assert "a.txt" in r.content and "nope-a" in r.content
    assert "b.txt" in r.content and "nope-b" in r.content
    assert ws.read("a.txt") == "aaa\n" and ws.read("b.txt") == "bbb\n"


async def test_apply_patch_summary_carries_line_delta(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("u.txt", "one\ntwo\n")
    reg, _ = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {"patch_text": _wrap(
            "*** Update File: u.txt\n"
            "@@\n"
            "-two\n"
            "+TWO\n"
            "+THREE\n")},
        _ctx(),
    )
    assert r.ok, r.content
    assert r.summary == "应用 1 项（+2 -1 行）"


async def test_apply_patch_refuses_stale_snapshot(tmp_path):
    """A file read this run then changed out-of-band: refuse, ask to re-read."""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = _ctx()
    ws.write("a.txt", "one\n")
    await reg.dispatch("read_file", {"path": "a.txt"}, ctx)
    ws.write("a.txt", "changed externally\n")
    r = await reg.dispatch(
        "apply_patch",
        {"patch_text": _wrap("*** Update File: a.txt\n@@\n-one\n+ONE")},
        ctx,
    )
    assert not r.ok and "重新 read_file" in r.content
    assert ws.read("a.txt") == "changed externally\n"


async def test_sandbox_rejects_traversal(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {"patch_text": _wrap("*** Add File: ../escape.txt\n+nope")},
        _ctx(),
    )
    assert not r.ok
    assert "非法路径" in r.content or "未做任何修改" in r.content
    assert not (tmp_path.parent / "escape.txt").exists()


async def test_empty_patch_and_bad_format_rejected(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    r = await reg.dispatch("apply_patch", {"patch_text": _wrap("")}, _ctx())
    assert not r.ok and "不包含任何文件操作" in r.content
    r = await reg.dispatch("apply_patch", {"patch_text": "garbage"}, _ctx())
    assert not r.ok and "解析失败" in r.content
    r = await reg.dispatch("apply_patch", {}, _ctx())
    assert not r.ok and "patch_text" in r.content


async def test_heredoc_wrapped_patch_applies(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": (
                "cat <<'EOF'\n"
                "*** Begin Patch\n"
                "*** Add File: h.txt\n"
                "+heredoc body\n"
                "*** End Patch\n"
                "EOF"
            )
        },
        _ctx(),
    )
    assert r.ok, r.content
    assert ws.read("h.txt") == "heredoc body\n"


# --- UI contract + undo ------------------------------------------------------


async def test_ui_events_carry_old_new(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("u.txt", "old\n")
    ws.write("d.txt", "del\n")
    reg, _ = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": _wrap(
                "*** Update File: u.txt\n@@\n-old\n+new\n"
                "*** Delete File: d.txt\n"
                "*** Add File: n.txt\n+brand new"
            )
        },
        _ctx(),
    )
    assert r.ok
    by_action = {e["action"]: e for e in r.ui}
    assert by_action["edit"]["old"] == "old\n" and by_action["edit"]["new"] == "new\n"
    assert by_action["delete"]["old"] == "del\n" and by_action["delete"]["new"] is None
    assert (
        by_action["write"]["old"] is None and by_action["write"]["new"] == "brand new\n"
    )


async def test_undo_reverts_whole_patch_including_delete(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("u.txt", "old\n")
    ws.write("d.txt", "del\n")
    reg, reverters = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": _wrap(
                "*** Update File: u.txt\n@@\n-old\n+new\n"
                "*** Delete File: d.txt\n"
                "*** Add File: n.txt\n+brand new"
            )
        },
        _ctx(),
    )
    assert r.ok
    assert "file_delete" in reverters  # the one new kind this bundle adds

    engine = UndoEngine(reverters)
    actions = [
        Action(
            kind=f"file_{e['action']}",
            target=e["path"],
            old_value=e["old"],
            new_value=e["new"],
        )
        for e in r.ui
    ]
    report = await engine.undo(actions, _ctx())
    assert report.ok and report.reverted == 3
    assert ws.read("u.txt") == "old\n"
    assert ws.read("d.txt") == "del\n"
    assert not ws.exists("n.txt")


async def test_undo_move_restores_both_sides(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("src.txt", "line\n")
    reg, reverters = _registry_with(ws)
    r = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": _wrap(
                "*** Update File: src.txt\n*** Move to: dst.txt\n@@\n-line\n+line2"
            )
        },
        _ctx(),
    )
    assert r.ok
    engine = UndoEngine(reverters)
    actions = [
        Action(
            kind=f"file_{e['action']}",
            target=e["path"],
            old_value=e["old"],
            new_value=e["new"],
        )
        for e in r.ui
    ]
    report = await engine.undo(actions, _ctx())
    assert report.ok
    assert ws.read("src.txt") == "line\n" and not ws.exists("dst.txt")


def test_parse_chunk_with_blank_context_line():
    """单个空格是「空白上下文行」的规范编码（真实多函数补丁几乎必含空行）：
    旧实现把任何空白行当 chunk 结束符——要么解析失败，要么 chunk 被静默
    截断后模糊错位。只有真正的空行才是分隔符。"""
    hunks = parse_patch(
        _wrap(
            "*** Update File: a.txt\n"
            "@@\n"
            " def f():\n"
            "     return 1\n"
            " \n"
            " def g():\n"
            "-    return 2\n"
            "+    return 3\n"
        )
    )
    chunk = hunks[0].chunks[0]
    assert chunk.old_lines == ["def f():", "    return 1", "",
                               "def g():", "    return 2"]
    assert chunk.new_lines == ["def f():", "    return 1", "",
                               "def g():", "    return 3"]


async def test_patch_applies_across_blank_line(tmp_path):
    """端到端：跨空行的 update 补丁应用成功且内容正确。"""
    ws = Workspace(tmp_path)
    ws.write("a.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    reg = ToolRegistry()
    register_apply_patch_tool(reg, lambda ctx: ws)

    patch_text = ("*** Begin Patch\n"
                  "*** Update File: a.py\n"
                  "@@\n"
                  " def g():\n"
                  "-    return 2\n"
                  "+    return 3\n"
                  "*** End Patch\n")
    res = await reg.dispatch("apply_patch", {"patch_text": patch_text},
                             AgentContext(run_id="r", user_id="u"))
    assert res.ok, res.content
    assert ws.read("a.py") == "def f():\n    return 1\n\n\ndef g():\n    return 3\n"
