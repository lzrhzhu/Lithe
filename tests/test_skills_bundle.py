"""lithe.bundles.skills: scan a directory of *.md skill files into an index
+ a load_skill tool. Pinning the scan (package.md excluded), description
extraction (frontmatter else first line), uniqueness resolution, disabled
filtering, and the tool's list/load behaviour — with no host application."""
from __future__ import annotations

from lithe import AgentContext, ToolRegistry
from lithe.bundles.skills import SkillLibrary, register_skill_tool


def _make_skills(tmp_path):
    (tmp_path / "a.md").write_text("---\ndescription: 技能A\n---\n# A\n内容A", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n首行描述B", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "package.md").write_text("---\nname: pkg\n---", encoding="utf-8")  # metadata
    (sub / "c.md").write_text("---\ndescription: 技能C\n---\nC内容", encoding="utf-8")
    return tmp_path


def test_files_exclude_package_metadata(tmp_path):
    lib = SkillLibrary(_make_skills(tmp_path))
    assert {p.stem for p in lib.files()} == {"a", "b", "c"}


def test_description_frontmatter_or_first_line(tmp_path):
    lib = SkillLibrary(_make_skills(tmp_path))
    desc = {p.stem: lib.description(p) for p in lib.files()}
    assert desc["a"] == "技能A"
    assert desc["b"] == "首行描述B"
    assert desc["c"] == "技能C"


def test_resolve_unique_missing_invalid(tmp_path):
    lib = SkillLibrary(_make_skills(tmp_path))
    p, err = lib.resolve("a")
    assert p is not None and err is None
    p, err = lib.resolve("nope")
    assert p is None and err is None      # not found (no error)
    p, err = lib.resolve("bad name!")
    assert p is None and err              # invalid name


def test_resolve_collision_reported(tmp_path):
    (tmp_path / "x.md").write_text("x", encoding="utf-8")
    sub = tmp_path / "p"
    sub.mkdir()
    (sub / "x.md").write_text("x2", encoding="utf-8")
    lib = SkillLibrary(tmp_path)
    p, err = lib.resolve("x")
    assert p is None and "重复" in err


def test_index_and_load_and_disabled(tmp_path):
    lib = SkillLibrary(_make_skills(tmp_path))
    idx = lib.index_text()
    assert "a：技能A" in idx and "b：首行描述B" in idx
    # disabled skills are hidden from the index
    assert "技能A" not in lib.index_text(frozenset({"a"}))
    assert "内容A" in lib.load("a")
    assert lib.load("nope") is None


async def test_tool_list_then_load(tmp_path):
    lib = SkillLibrary(_make_skills(tmp_path))
    reg = ToolRegistry()
    register_skill_tool(reg, lib)
    ctx = AgentContext(run_id="r", user_id="u")
    lst = await reg.dispatch("load_skill", {}, ctx)
    assert lst.ok and "技能A" in lst.content
    loaded = await reg.dispatch("load_skill", {"name": "a"}, ctx)
    assert loaded.ok and "内容A" in loaded.content


async def test_tool_disabled_and_missing(tmp_path):
    lib = SkillLibrary(_make_skills(tmp_path))
    reg = ToolRegistry()
    register_skill_tool(reg, lib, disabled_for=lambda ctx: frozenset({"a"}))
    ctx = AgentContext(run_id="r", user_id="u")
    assert (await reg.dispatch("load_skill", {"name": "a"}, ctx)).ok is False   # disabled
    assert (await reg.dispatch("load_skill", {"name": "zzz"}, ctx)).ok is False  # missing


def test_empty_library_index(tmp_path):
    lib = SkillLibrary(tmp_path)  # no .md files
    assert lib.index_text() == "（暂无可用技能）"
    assert lib.files() == []
