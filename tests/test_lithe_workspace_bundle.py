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

    # dir= 指向一个已存在的文件：等价于“只 grep 这个文件”，而不是报
    # “目录不存在”（路径明明存在——那条假错误会把模型逼进改目录的
    # 重试循环）。glob 在单文件模式下被忽略。
    as_file = await reg.dispatch("search_files",
                                 {"pattern": "beta", "dir": "code/a.py",
                                  "glob": "*.md"}, ctx)
    assert as_file.ok and "code/a.py:2: beta = 2" in as_file.content
    assert "alpha" not in as_file.content  # 只搜该文件，不含同文件其它行以外的输出
    assert "notes/" not in as_file.content and "b.py" not in as_file.content

    miss = await reg.dispatch("search_files",
                              {"pattern": "beta", "dir": "code/nope.py"}, ctx)
    assert miss.ok is False and "目录不存在" in miss.content


async def test_search_files_ignore_case_literal_context_limit(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")

    # ignore_case：命中大小写混合文本；缺省仍区分大小写
    ws.write("case.txt", "Hello world\nhello again\nhallo\n")
    ci = await reg.dispatch("search_files",
                            {"pattern": "HELLO", "ignore_case": True}, ctx)
    assert ci.ok and "case.txt:1: Hello world" in ci.content
    assert "case.txt:2: hello again" in ci.content and "2 处匹配" in ci.summary
    cs = await reg.dispatch("search_files", {"pattern": "HELLO"}, ctx)
    assert cs.ok and "未找到匹配" in cs.content

    # literal：元字符按字面匹配，"f(" 这类非法正则也能搜
    ws.write("lit.txt", "xa.b*cy\nplain\n")
    lit = await reg.dispatch("search_files",
                             {"pattern": "a.b*c", "literal": True}, ctx)
    assert lit.ok and "lit.txt:1: xa.b*cy" in lit.content
    raw_paren = await reg.dispatch("search_files",
                                   {"pattern": "f(", "literal": True}, ctx)
    assert raw_paren.ok, "literal 模式不应走正则编译"
    re_bad = await reg.dispatch("search_files", {"pattern": "f("}, ctx)
    assert re_bad.ok is False and "正则" in re_bad.content

    # context=1：邻近命中合并为一个块，非邻接块以 -- 分隔，块外行不出现
    ws.write("blk.txt", "l1\nhit l2\nl3\nl4\nl5\nhit l6\nl7\n")
    withctx = await reg.dispatch("search_files",
                                 {"pattern": "hit", "context": 1}, ctx)
    assert withctx.ok
    for frag in ("blk.txt:1: l1", "blk.txt:2: hit l2", "blk.txt:3: l3",
                 "blk.txt:5: l5", "blk.txt:6: hit l6", "blk.txt:7: l7"):
        assert frag in withctx.content
    assert "blk.txt:4" not in withctx.content
    assert "\n--\n" in withctx.content
    # 相邻命中（2、3 行）合并：无 -- 分隔
    ws.write("near.txt", "x\nm1\nm2\ny\n")
    merged = await reg.dispatch("search_files",
                                {"pattern": "m", "context": 1}, ctx)
    assert merged.ok and "\n--\n" not in merged.content

    # limit：截断并如实提示；负数 limit 拒绝
    ws.write("many.txt", "\n".join(f"term {i}" for i in range(5)) + "\n")
    lim = await reg.dispatch("search_files",
                             {"pattern": "term", "limit": 2}, ctx)
    assert lim.ok and "2 处匹配" in lim.summary
    assert "仅显示前 2 条" in lim.content
    assert "term 2" not in lim.content and "term 3" not in lim.content
    neg = await reg.dispatch("search_files",
                             {"pattern": "term", "limit": -1}, ctx)
    assert neg.ok is False and "limit" in neg.content


async def test_read_file_directory_listing(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("a.txt", "1")
    ws.write("code/b.py", "2")
    ws.new_folder("empty")

    root = await reg.dispatch("read_file", {"path": "."}, ctx)
    assert root.ok
    assert "a.txt" in root.content and "code/" in root.content
    assert "empty/" in root.content and "共 3 项" in root.content

    page = await reg.dispatch("read_file", {"path": ".", "offset": 2, "limit": 1}, ctx)
    assert page.ok and "2: code/" in page.content
    assert "1: a.txt" not in page.content and "第 2–2 项" in page.content

    sub = await reg.dispatch("read_file", {"path": "code"}, ctx)
    assert sub.ok and "b.py" in sub.content

    empty = await reg.dispatch("read_file", {"path": "empty"}, ctx)
    assert empty.ok and "空目录" in empty.content

    beyond = await reg.dispatch("read_file", {"path": ".", "offset": 9}, ctx)
    assert beyond.ok and "超出" in beyond.content


async def test_read_file_refuses_images_docs_binary(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")

    # PNG 魔数：拒绝文本读入（不产生替换字符垃圾），指向图像侧信道
    ws.write_bytes("img.png", b"\x89PNG\r\n\x1a\n" + b"\x00\x10\x20" * 64)
    img = await reg.dispatch("read_file", {"path": "img.png"}, ctx)
    assert img.ok is False and img.summary == "图片文件"
    assert "analyze_image" in img.content
    assert "\ufffd" not in img.content and len(img.content) < 400  # 无 token 浪费

    # PDF 魔数 / 仅扩展名的 OOXML：指向文档侧信道
    ws.write_bytes("doc.pdf", b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n" + b"x" * 64)
    pdf = await reg.dispatch("read_file", {"path": "doc.pdf"}, ctx)
    assert pdf.ok is False and pdf.summary == "文档文件"
    assert "analyze_document" in pdf.content
    ws.write_bytes("sheet.docx", b"PK\x03\x04not-really")
    ext = await reg.dispatch("read_file", {"path": "sheet.docx"}, ctx)
    assert ext.ok is False and ext.summary == "文档文件"

    # 其它二进制：直接拒绝
    ws.write_bytes("f.zip", b"PK\x05\x06" + bytes(range(64)))
    binf = await reg.dispatch("read_file", {"path": "f.zip"}, ctx)
    assert binf.ok is False and binf.summary == "二进制文件"

    # UTF-16 BOM：报编码问题而不是 mojibake
    ws.write_bytes("u16.txt", "ÿþH\x00i\x00".encode("latin-1"))
    u16 = await reg.dispatch("read_file", {"path": "u16.txt"}, ctx)
    assert u16.ok is False and u16.summary == "编码不支持"
    assert "UTF-16" in u16.content

    # 魔数优先于扩展名：文本内容顶着 .png 名字照常读
    ws.write("notes.png", "其实我是文本")
    plain = await reg.dispatch("read_file", {"path": "notes.png"}, ctx)
    assert plain.ok and "其实我是文本" in plain.content


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


# --- edit_file: identical guard + strict exact matching -----------------------
#
# Kilo-style semantics: edit_file matches old_text EXACTLY — character for
# character, whitespace and indentation included. A miss is a hard error
# that teaches the exactness discipline and shows the nearest real block
# (read-only diagnostics); it never fuzzy-guesses a replacement location.
# The tolerant whole-line ladder still lives in apply_patch, which anchors
# edits with @@ context lines instead of a bare old/new pair.


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


async def test_edit_strict_trailing_whitespace_rejected(tmp_path):
    """Whitespace divergence is a hard miss: the fuzzy ladder that silently
    tolerated it could anchor the wrong span; the model re-edits with the
    exact bytes after seeing the real content."""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.py", "x = 1\ny = 2\n")
    r = await reg.dispatch("edit_file",
                           {"path": "f.py", "old_text": "x = 1   ",
                            "new_text": "x = 10"}, ctx)
    assert r.ok is False and "未找到 old_text" in r.content
    assert "行尾空格" in r.content
    assert ws.read("f.py") == "x = 1\ny = 2\n"          # untouched
    ok = await reg.dispatch("edit_file",
                            {"path": "f.py", "old_text": "x = 1",
                             "new_text": "x = 10"}, ctx)
    assert ok.ok, ok.content
    assert ws.read("f.py") == "x = 10\ny = 2\n"


async def test_edit_strict_over_indented_old_text_rejected(tmp_path):
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.py", "def a():\n    return 1\n")
    r = await reg.dispatch("edit_file",
                           {"path": "f.py", "old_text": "        return 1",
                            "new_text": "    return 2"}, ctx)
    assert r.ok is False and "缩进" in r.content
    assert ws.read("f.py") == "def a():\n    return 1\n"
    ok = await reg.dispatch("edit_file",
                            {"path": "f.py", "old_text": "    return 1",
                             "new_text": "    return 2"}, ctx)
    assert ok.ok and ws.read("f.py") == "def a():\n    return 2\n"


async def test_edit_strict_unicode_punctuation_rejected(tmp_path):
    """半角/全角标点差异不再被静默容忍——精确匹配要求逐字符一致。"""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.md", "结果 – 完成\n")
    r = await reg.dispatch("edit_file",
                           {"path": "f.md", "old_text": "结果 - 完成",
                            "new_text": "done"}, ctx)
    assert r.ok is False and "全角" in r.content
    assert ws.read("f.md") == "结果 – 完成\n"
    ok = await reg.dispatch("edit_file",
                            {"path": "f.md", "old_text": "结果 – 完成",
                             "new_text": "done"}, ctx)
    assert ok.ok and ws.read("f.md") == "done\n"


async def test_edit_strict_copied_line_number_prefix_rejected(tmp_path):
    """old_text 照抄 read_file 的行号前缀时精确匹配失败，报错指引剥前缀。"""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.txt", "keep\nchange me\n")
    r = await reg.dispatch("edit_file",
                           {"path": "f.txt", "old_text": "2: change me",
                            "new_text": "changed"}, ctx)
    assert r.ok is False and "行号前缀" in r.content
    assert ws.read("f.txt") == "keep\nchange me\n"
    ok = await reg.dispatch("edit_file",
                            {"path": "f.txt", "old_text": "change me",
                            "new_text": "changed"}, ctx)
    assert ok.ok and ws.read("f.txt") == "keep\nchanged\n"


async def test_edit_strict_whitespace_divergence_is_zero_hit_not_multi(tmp_path):
    """带尾随空格的重复段：精确计数为 0（不再是模糊多处命中）；剥掉差异后
    用 replace_all 显式解决歧义。"""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.md", "头\n重复段\n中\n重复段\n")
    r = await reg.dispatch("edit_file",
                           {"path": "f.md", "old_text": "重复段 ",
                            "new_text": "X"}, ctx)
    assert r.ok is False and "未找到 old_text" in r.content
    assert ws.read("f.md") == "头\n重复段\n中\n重复段\n"
    r2 = await reg.dispatch("edit_file",
                            {"path": "f.md", "old_text": "重复段",
                             "new_text": "X", "replace_all": True}, ctx)
    assert r2.ok and ws.read("f.md") == "头\nX\n中\nX\n"


async def test_edit_strict_backslash_double_escaping_rejected(tmp_path):
    """JSON 双重转义的 LaTeX 定界符（\\\\( vs \\(）不再折叠命中——必须按
    文件里的实际转义层数构造 old_text。"""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.md", "常用 \\(p:q\\) 表示频率比\n")
    r = await reg.dispatch(
        "edit_file",
        {"path": "f.md", "old_text": "常用 \\\\(p:q\\\\) 表示频率比",
         "new_text": "共振"}, ctx)
    assert r.ok is False and "转义" in r.content
    assert ws.read("f.md") == "常用 \\(p:q\\) 表示频率比\n"
    ok = await reg.dispatch(
        "edit_file",
        {"path": "f.md", "old_text": "常用 \\(p:q\\) 表示频率比",
         "new_text": "共振"}, ctx)
    assert ok.ok, ok.content
    assert ws.read("f.md") == "共振\n"


async def test_edit_zero_hit_shows_nearest_block_hint(tmp_path):
    """零命中时报错展示最相似的实际块（带 read_file 风格行号），模型可照抄
    实际内容重试——真实事故里 docstring 尾部记错曾导致连续 5 次失败。
    诊断是只读的：绝不用猜测的跨度做替换。"""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    actual = ('async def undo(cfg):\n'
              '    """Revert a run\'s mutations; returns how many actions'
              ' were reverted."""\n'
              '    reg = build_registry(cfg)\n')
    ws.write("agent.py", actual)
    # 模型凭记忆重构：首行一致、docstring 尾部不同
    r = await reg.dispatch(
        "edit_file",
        {"path": "agent.py",
         "old_text": ('async def undo(cfg):\n'
                      '    """Revert a run\'s mutations; returns the'
                      ' reverted count."""\n'
                      '    reg = build_registry(cfg)\n'),
         "new_text": "X"}, ctx)
    assert r.ok is False
    assert "未找到 old_text" in r.content
    assert "最接近的候选" in r.content
    assert "how many actions were reverted" in r.content  # 实际内容被展示
    assert "2: " in r.content  # read_file 风格行号，便于照抄去前缀重试
    assert "read_file" in r.content
    assert ws.read("agent.py") == actual  # 未改动


async def test_edit_zero_hit_without_similar_line_advises_re_read(tmp_path):
    """完全无相似内容时不展示候选，只引导重新 read_file。"""
    ws = Workspace(tmp_path)
    reg, _ = _registry_with(ws)
    ctx = AgentContext(run_id="r", user_id="u")
    ws.write("f.txt", "alpha\nbeta\n")
    r = await reg.dispatch(
        "edit_file", {"path": "f.txt", "old_text": "zzz qqq",
                      "new_text": "x"}, ctx)
    assert r.ok is False
    assert "未找到 old_text" in r.content and "read_file" in r.content
    assert "最接近的候选" not in r.content


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
    try:
        os.symlink(outside, root / "leak.txt")
        os.symlink(tmp_path / "nope-missing", root / "dangling")
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            import pytest

            pytest.skip("creating symlinks requires Windows Developer Mode or privileges")
        raise

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
