"""lithe.bundles.workspace: sandboxed file I/O + the read/write/edit/list
tools every agent needs, with file undo reverters. Pinning the sandbox safety
(traversal), the tool round-trips, the file_change UI contract and the undo
reverters — all with no host application and no DB."""
from __future__ import annotations

import pytest

from lithe import Action, AgentContext, ToolRegistry, UndoEngine
from lithe.bundles.workspace import Workspace, register_file_tools


# --- Workspace: sandboxed file I/O ---

def test_safe_path_rejects_traversal(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("a/b.txt", "x")
    assert ws.safe_path("a/b.txt") == (tmp_path / "a" / "b.txt").resolve()
    for evil in ("../escape", "../../etc/passwd", "a/../../escape"):
        with pytest.raises(PermissionError):
            ws.safe_path(evil)


def test_read_write_delete_roundtrip(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("code/x.py", "print(1)")
    assert ws.read("code/x.py") == "print(1)"
    assert ws.exists("code/x.py")
    ws.delete("code/x.py")
    assert not ws.exists("code/x.py")


def test_protected_dirs_not_deletable(tmp_path):
    ws = Workspace(tmp_path, protected_dirs=frozenset({"code"}))
    ws.new_folder("code")
    with pytest.raises(PermissionError):
        ws.delete("code")
    # but a file inside the protected dir is deletable
    ws.write("code/a.py", "1")
    ws.delete("code/a.py")
    assert not ws.exists("code/a.py")


def test_ignored_dirs_cover_venv_git_node_modules(tmp_path):
    """依赖/VCS 目录对工具不可见，也不计入 _count（回归：带 venv 的项目
    工作区曾被 list_files 吐出 3824 项、写入被总量守卫拒绝）。"""
    ws = Workspace(tmp_path, max_files=2)
    for d in ("venv", ".venv", "node_modules", ".git"):
        d_ = tmp_path / d
        d_.mkdir()
        (d_ / "f.py").write_text("x", encoding="utf-8")
    ws.write("real.txt", "x")
    assert {p for p, _ in ws.walk()} == {"real.txt"}
    assert ws._count() == 1


def test_workspace_write_ignores_pre_existing_file_count(tmp_path):
    """Workspace.write 不再看目录总量：大项目里新建/覆盖/undo 恢复都不受
    max_files 影响（限流职责移交工具层，见下）。"""
    ws = Workspace(tmp_path, max_files=1)
    for i in range(5):
        ws.write(f"project{i}.py", "x")     # direct I/O: never refused
    ws.write("project0.py", "overwritten")  # overwrite fine
    assert ws.read("project0.py") == "overwritten"


async def test_write_file_works_in_project_workspace_with_venv(tmp_path):
    ws = Workspace(tmp_path, max_files=3)
    venv = tmp_path / "venv" / "lib"
    venv.mkdir(parents=True)
    for i in range(20):
        (venv / f"m{i}.py").write_text("x", encoding="utf-8")
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    r = await reg.dispatch("write_file",
                           {"path": "projectile.py", "content": "print(1)"}, ctx)
    assert r.ok, r.content


async def test_run_creation_cap_brakes_runaway_loops(tmp_path):
    """max_files 现在限的是“本次运行通过工具新建的文件数”。"""
    ws = Workspace(tmp_path, max_files=2)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    assert (await reg.dispatch("write_file", {"path": "a", "content": "1"}, ctx)).ok
    assert (await reg.dispatch("write_file", {"path": "b", "content": "1"}, ctx)).ok
    cap = await reg.dispatch("write_file", {"path": "c", "content": "1"}, ctx)
    assert cap.ok is False and "上限" in cap.content
    # overwrite of an existing file never counts (no growth)
    assert (await reg.dispatch("write_file", {"path": "a", "content": "2"}, ctx)).ok
    # subagent contexts share the budget (ctx.shared by reference)
    sub = AgentContext(run_id="r", user_id="u", subagent="w", shared=ctx.shared)
    assert (await reg.dispatch("write_file", {"path": "d", "content": "1"}, sub)).ok is False


async def test_undo_restore_not_blocked_by_file_count(tmp_path):
    """undo 的恢复写入不再被总量守卫拒绝（旧守卫的第二个受害者）。"""
    ws = Workspace(tmp_path, max_files=1)
    reg, reverters = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("many1", "x")
    ws.write("many2", "x")                     # total already > max_files
    ws.write("g.txt", "old")
    await reg.dispatch("edit_file",
                       {"path": "g.txt", "old_text": "old", "new_text": "new"}, ctx)
    ws.delete("g.txt")
    report = await UndoEngine(reverters).undo(
        [Action(kind="file_edit", target="g.txt", old_value="old",
                status="applied")], ctx)
    assert report.ok and ws.read("g.txt") == "old"


def test_list_scoped_to_subdirs_and_walk(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("code/a.py", "1")
    ws.write("thesis/intro.md", "2")
    walked = {p for p, _ in ws.walk()}
    assert walked == {"code/a.py", "thesis/intro.md"}
    only_code = {e["path"] for e in ws.list(("code",))}
    assert only_code == {"code", "code/a.py"}   # includes the folder itself


def test_rename(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("old.txt", "x")
    ws.rename("old.txt", "new.txt")
    assert ws.exists("new.txt") and not ws.exists("old.txt")


# --- file tools + undo reverters ---

def _registry_with(ws):
    reg = ToolRegistry()
    reverters = register_file_tools(reg, lambda ctx: ws)
    return reg, reverters


async def test_file_tools_roundtrip_and_ui_contract(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")

    w = await reg.dispatch("write_file", {"path": "code/a.py", "content": "hi"}, ctx)
    assert w.ok and w.ui[0]["action"] == "write" and w.ui[0]["created"] is True
    assert w.ui[0]["old"] is None and w.ui[0]["new"] == "hi"

    r = await reg.dispatch("read_file", {"path": "code/a.py"}, ctx)
    assert r.ok and "hi" in r.content

    e = await reg.dispatch("edit_file",
                           {"path": "code/a.py", "old_text": "hi", "new_text": "hello"}, ctx)
    assert e.ok and ws.read("code/a.py") == "hello"
    assert e.ui[0]["old"] == "hi" and e.ui[0]["new"] == "hello"

    lst = await reg.dispatch("list_files", {"dirs": ["code"]}, ctx)
    assert lst.ok and "code/a.py" in lst.content


async def test_read_file_line_window_pagination(tmp_path):
    """大文件按行分页：offset/limit 读取任意窗口，不再只能读头部。"""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("long.txt", "\n".join(f"line{i}" for i in range(1, 11)))

    page = await reg.dispatch("read_file",
                              {"path": "long.txt", "offset": 3, "limit": 4}, ctx)
    assert page.ok
    assert "line3" in page.content and "line6" in page.content
    assert "line2" not in page.content and "line7" not in page.content
    assert "3–6 行，共 10 行" in page.content

    tail = await reg.dispatch("read_file", {"path": "long.txt", "offset": 8}, ctx)
    assert "line8" in tail.content and "line10" in tail.content \
        and "line7" not in tail.content

    limit_only = await reg.dispatch("read_file", {"path": "long.txt", "limit": 2}, ctx)
    assert "line1" in limit_only.content and "line3" not in limit_only.content

    beyond = await reg.dispatch("read_file", {"path": "long.txt", "offset": 99}, ctx)
    assert beyond.ok and "超出" in beyond.content

    # 无分页参数：行为与之前完全一致
    whole = await reg.dispatch("read_file", {"path": "long.txt"}, ctx)
    assert "line1" in whole.content and "共 10 行" not in whole.content


async def test_edit_rejects_missing_old_text(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    await reg.dispatch("write_file", {"path": "f", "content": "abc"}, ctx)
    r = await reg.dispatch("edit_file", {"path": "f", "old_text": "zzz", "new_text": "y"}, ctx)
    assert r.ok is False


async def test_edit_ambiguous_old_text_rejected_with_line_hints(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    await reg.dispatch("write_file",
                       {"path": "f.md", "content": "头\n重复段\n中\n重复段\n尾\n重复段"}, ctx)
    r = await reg.dispatch("edit_file",
                           {"path": "f.md", "old_text": "重复段", "new_text": "X"}, ctx)
    assert r.ok is False and "3 次" in r.content and "replace_all" in r.content
    assert "2" in r.content and "4" in r.content   # 行号线索
    assert ws.read("f.md").startswith("头\n重复段"), "多处匹配时不得改动文件"


async def test_edit_replace_all_and_diff(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    await reg.dispatch("write_file",
                       {"path": "f.md", "content": "a\nb\na\na"}, ctx)
    r = await reg.dispatch("edit_file",
                           {"path": "f.md", "old_text": "a", "new_text": "z",
                            "replace_all": True}, ctx)
    assert r.ok and ws.read("f.md") == "z\nb\nz\nz"
    assert "3 处" in r.content and "-a" in r.content and "+z" in r.content
    # 单处替换的结果也带 diff，且 ui 契约不变（旧/新全文供 undo）
    r1 = await reg.dispatch("edit_file",
                            {"path": "f.md", "old_text": "z\nb", "new_text": "y"}, ctx)
    assert r1.ok and r1.ui[0]["old"] == "z\nb\nz\nz" and r1.ui[0]["new"] == "y\nz\nz"


async def test_search_files_regex_dir_and_glob(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("code/a.py", "alpha = 1\nbeta = 2\n")
    ws.write("code/b.py", "gamma\n")
    ws.write("notes/a.txt", "alpha note\n")

    hit = await reg.dispatch("search_files", {"pattern": "alpha"}, ctx)
    assert hit.ok
    assert "code/a.py:1: alpha = 1" in hit.content
    assert "notes/a.txt:1: alpha note" in hit.content
    assert "code/b.py" not in hit.content

    scoped = await reg.dispatch("search_files",
                                {"pattern": "alpha", "dir": "code"}, ctx)
    assert scoped.ok and "notes/" not in scoped.content

    only_py = await reg.dispatch("search_files",
                                 {"pattern": "gamma", "glob": "*.py"}, ctx)
    assert only_py.ok and "code/b.py:1" in only_py.content

    none = await reg.dispatch("search_files", {"pattern": "nothing_here"}, ctx)
    assert none.ok and "未找到匹配" in none.content

    bad = await reg.dispatch("search_files", {"pattern": "a("}, ctx)
    assert bad.ok is False and "正则" in bad.content


async def test_glob_files_matches_nested_names(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("code/a.py", "1")
    ws.write("deep/nested/b.py", "2")
    ws.write("notes/c.md", "3")

    py = await reg.dispatch("glob_files", {"pattern": "*.py"}, ctx)
    assert py.ok and "code/a.py" in py.content and "deep/nested/b.py" in py.content
    assert "notes/c.md" not in py.content

    scoped = await reg.dispatch("glob_files", {"pattern": "notes/*.md"}, ctx)
    assert scoped.ok and "notes/c.md" in scoped.content and "a.py" not in scoped.content

    none = await reg.dispatch("glob_files", {"pattern": "*.rs"}, ctx)
    assert none.ok and "没有匹配" in none.content


async def test_undo_reverters_reverse_in_order(tmp_path):
    ws = Workspace(tmp_path)
    reg, reverters = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    # forward: create v1, then edit v1->v2
    await reg.dispatch("write_file", {"path": "f.txt", "content": "v1"}, ctx)
    await reg.dispatch("edit_file", {"path": "f.txt", "old_text": "v1", "new_text": "v2"}, ctx)
    assert ws.read("f.txt") == "v2"
    # undo newest-first: actions listed oldest->newest; UndoEngine reverses them
    engine = UndoEngine(reverters)
    report = await engine.undo([
        Action(kind="file_write", target="f.txt", old_value=None, status="applied"),
        Action(kind="file_edit", target="f.txt", old_value="v1", status="applied"),
    ], ctx)
    assert report.reverted == 2 and report.ok is True
    assert not ws.exists("f.txt")   # edit restored v1, then write deleted the file


async def test_file_tools_filtered_by_mode(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    names = {s["function"]["name"] for s in reg.specs_for_mode("autonomous")}
    assert names == {"read_file", "write_file", "edit_file", "list_files",
                     "search_files", "glob_files"}
    # anchored (read-only) mode hides the write tools but keeps the search ones
    anchored = {s["function"]["name"] for s in reg.specs_for_mode("anchored")}
    assert anchored == {"read_file", "list_files", "search_files", "glob_files"}


# --- read_file line-number prefixes ------------------------------------------


async def test_read_file_numbers_every_line(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("n.txt", "alpha\nbeta\n")

    whole = await reg.dispatch("read_file", {"path": "n.txt"}, ctx)
    assert whole.ok
    assert whole.content.splitlines() == ["1: alpha", "2: beta"]

    page = await reg.dispatch("read_file",
                              {"path": "n.txt", "offset": 2, "limit": 1}, ctx)
    assert page.ok and page.content.splitlines()[0] == "2: beta"


# --- edit_file: identical guard + fuzzy whole-line fallback -------------------


async def test_file_tool_summaries_carry_line_delta(tmp_path):
    """write_file/edit_file summaries carry coding-agent style +N -M counts."""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")

    w = await reg.dispatch("write_file", {"path": "f.py", "content": "a\nb\n"}, ctx)
    assert w.ok and w.summary == "写入 f.py（+2 行，新建）"
    assert w.ui[0]["added"] == 2 and w.ui[0]["removed"] == 0

    e = await reg.dispatch("edit_file",
                           {"path": "f.py", "old_text": "b", "new_text": "c\nd"}, ctx)
    assert e.ok and e.summary == "编辑 f.py（+2 -1 行）"
    assert e.ui[0]["added"] == 2 and e.ui[0]["removed"] == 1

    same = await reg.dispatch("write_file", {"path": "f.py", "content": "a\nc\nd\n"}, ctx)
    assert same.ok and same.summary == "写入 f.py"  # identical rewrite: no delta note


async def test_edit_identical_old_new_rejected(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f", "abc")
    r = await reg.dispatch("edit_file",
                           {"path": "f", "old_text": "abc", "new_text": "abc"}, ctx)
    assert r.ok is False and "相同" in r.content


async def test_edit_fuzzy_trailing_whitespace(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.py", "x = 1\ny = 2\n")
    r = await reg.dispatch("edit_file",
                           {"path": "f.py", "old_text": "x = 1   ",
                            "new_text": "x = 10"}, ctx)
    assert r.ok, r.content
    assert ws.read("f.py") == "x = 10\ny = 2\n"
    assert "模糊" in r.content


async def test_edit_fuzzy_over_indented_old_text(tmp_path):
    """Exact substring is primary; the ladder catches over-indented old_text
    (a substring miss) and replaces whole lines."""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.py", "def a():\n    return 1\n")
    r = await reg.dispatch("edit_file",
                           {"path": "f.py", "old_text": "        return 1",
                            "new_text": "    return 2"}, ctx)
    assert r.ok, r.content
    assert ws.read("f.py") == "def a():\n    return 2\n"


async def test_edit_fuzzy_unicode_punctuation(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.md", "结果 – 完成\n")
    r = await reg.dispatch("edit_file",
                           {"path": "f.md", "old_text": "结果 - 完成",
                            "new_text": "done"}, ctx)
    assert r.ok, r.content
    assert ws.read("f.md") == "done\n"


async def test_edit_fuzzy_strips_copied_line_number_prefix(tmp_path):
    """old_text copied verbatim from read_file's numbered output still lands."""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.txt", "keep\nchange me\n")
    r = await reg.dispatch("edit_file",
                           {"path": "f.txt", "old_text": "2: change me",
                            "new_text": "changed"}, ctx)
    assert r.ok, r.content
    assert ws.read("f.txt") == "keep\nchanged\n"


async def test_edit_fuzzy_multi_match_rejected_with_line_hints(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.md", "头\n重复段\n中\n重复段\n")
    # exact match fails only when whitespace diverges → ladder sees both hits
    r = await reg.dispatch("edit_file",
                           {"path": "f.md", "old_text": "重复段 ",
                            "new_text": "X"}, ctx)
    assert r.ok is False and "2 次" in r.content and "replace_all" in r.content
    assert "2" in r.content and "4" in r.content
    # replace_all resolves the ambiguity explicitly
    r2 = await reg.dispatch("edit_file",
                            {"path": "f.md", "old_text": "重复段 ",
                             "new_text": "X", "replace_all": True}, ctx)
    assert r2.ok and ws.read("f.md") == "头\nX\n中\nX\n"


# --- stale-content guard (optimistic concurrency) -----------------------------


async def test_write_file_refuses_stale_snapshot_then_re_read(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    await reg.dispatch("write_file", {"path": "f", "content": "v1"}, ctx)
    await reg.dispatch("read_file", {"path": "f"}, ctx)
    ws.write("f", "externally changed")          # out-of-band mutation

    stale = await reg.dispatch("write_file", {"path": "f", "content": "v2"}, ctx)
    assert stale.ok is False and "重新 read_file" in stale.content
    assert ws.read("f") == "externally changed"  # not clobbered

    await reg.dispatch("read_file", {"path": "f"}, ctx)
    ok = await reg.dispatch("write_file", {"path": "f", "content": "v2"}, ctx)
    assert ok.ok and ws.read("f") == "v2"


async def test_edit_file_refuses_stale_snapshot(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f", "old line\n")
    await reg.dispatch("read_file", {"path": "f"}, ctx)
    ws.write("f", "rewritten elsewhere\n")

    r = await reg.dispatch("edit_file",
                           {"path": "f", "old_text": "old line",
                            "new_text": "new line"}, ctx)
    assert r.ok is False and "重新 read_file" in r.content


async def test_successful_writes_refresh_snapshot_no_false_alarm(tmp_path):
    """Chained write→edit→edit on one context must never trip the guard."""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    await reg.dispatch("write_file", {"path": "f", "content": "a\n"}, ctx)
    await reg.dispatch("read_file", {"path": "f"}, ctx)
    e1 = await reg.dispatch("edit_file",
                            {"path": "f", "old_text": "a", "new_text": "b"}, ctx)
    e2 = await reg.dispatch("edit_file",
                            {"path": "f", "old_text": "b", "new_text": "c"}, ctx)
    w = await reg.dispatch("write_file", {"path": "f", "content": "d"}, ctx)
    assert e1.ok and e2.ok and w.ok
    assert ws.read("f") == "d"


async def test_stale_guard_spans_subagent_contexts(tmp_path):
    """陈旧检查按 ctx.shared 共享：编排者读过、外部改动的文件，一个共享
    shared 的子代理 ctx 写它同样被拒（否则并行子代理可互相覆盖）。"""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    parent = AgentContext(run_id="r", user_id="u")
    sub = AgentContext(run_id="r", user_id="u", subagent="w",
                       shared=parent.shared)   # 引用同一 map

    await reg.dispatch("write_file", {"path": "f", "content": "v1"}, parent)
    await reg.dispatch("read_file", {"path": "f"}, parent)
    ws.write("f", "externally changed")          # out-of-band mutation

    r = await reg.dispatch("write_file", {"path": "f", "content": "v2"}, sub)
    assert r.ok is False and "重新 read_file" in r.content
    assert ws.read("f") == "externally changed"  # not clobbered


# --- symlink containment: the walk family must never leave the root ------

async def test_walk_and_search_skip_symlinks(tmp_path):
    """模型代码可在沙箱内种符号链接指向宿主文件（run_code 把工作区以读写
    bind 进去）；walk/search/glob/list 不得跟随，否则绕过 safe_path 读宿主
    内容。"""
    import os

    root = tmp_path / "ws"
    root.mkdir()
    (root / "note.txt").write_text("inside secret", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("HOST SECRET should-not-leak", encoding="utf-8")
    os.symlink(outside, root / "leak.txt")
    os.symlink(tmp_path / "nope-missing", root / "dangling")

    ws = Workspace(root)
    walked = [rel for rel, _p in ws.walk()]
    assert "leak.txt" not in walked and "dangling" not in walked
    assert "note.txt" in walked

    # list_files marks symlinks opaque instead of following
    entries = {e["path"]: e["type"] for e in ws.list()}
    assert entries["leak.txt"] == "symlink"

    reg = ToolRegistry()
    register_file_tools(reg, lambda ctx: ws)
    ctx = AgentContext(run_id="r", user_id="u")

    res = await reg.dispatch("search_files", {"pattern": "HOST SECRET"}, ctx)
    assert res.ok and "should-not-leak" not in res.content

    res = await reg.dispatch("glob_files", {"pattern": "leak*"}, ctx)
    assert res.ok and "leak.txt" not in res.content

    # direct read of a symlinked path is still refused by safe_path
    res = await reg.dispatch("read_file", {"path": "leak.txt"}, ctx)
    assert res.ok is False
