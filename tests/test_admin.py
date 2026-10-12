"""Admin bundle: tool management helpers over a ToolRegistry (category grouping,
package grouping, integrity check). Pure over the registry + a packages list."""
from __future__ import annotations

from lithe import ToolCategory, ToolRegistry, ToolResult, ToolSpec
from lithe.bundles import (
    ToolPackage, check_packages, list_tool_packages_admin, list_tools_admin,
    package_meta, package_of, tool_categories,
)


def _reg() -> ToolRegistry:
    reg = ToolRegistry()

    async def h(ctx, args):
        return ToolResult(True, "ok", "ok")

    reg.register(ToolSpec("read_a", "ra", category=ToolCategory.READ), h)
    reg.register(ToolSpec("write_b", "wb", category=ToolCategory.WRITE), h)
    reg.register(ToolSpec("delegate", "del", category=ToolCategory.META), h)
    return reg


def test_tool_categories_groups_by_category():
    c = tool_categories(_reg())
    assert [e["name"] for e in c["read"]] == ["read_a"]
    assert [e["name"] for e in c["write"]] == ["write_b"]
    assert [e["name"] for e in c["meta"]] == ["delegate"]
    assert set(c["read"][0]) == {"name", "description", "parameters"}


def test_list_tools_admin_order_enabled_package():
    pkgs = [ToolPackage("grp1", "组1", "d", ["read_a"]),
            ToolPackage("grp2", "组2", "d", ["write_b", "delegate"])]
    out = list_tools_admin(_reg(), disabled={"write_b"}, packages=pkgs)
    assert [e["category"] for e in out] == ["read", "write", "meta"]  # ordered
    d = {e["name"]: e for e in out}
    assert d["write_b"]["enabled"] is False and d["read_a"]["enabled"] is True
    assert d["read_a"]["package"] == "grp1" and d["delegate"]["package"] == "grp2"


def test_check_packages_missing_extra_ok():
    reg = _reg()
    assert check_packages(reg, [ToolPackage("g", "g", "d", ["read_a", "write_b"])]) == {
        "ok": False, "missing": ["delegate"], "extra": []}
    assert check_packages(
        reg, [ToolPackage("g", "g", "d", ["read_a", "write_b", "delegate", "ghost"])]) == {
        "ok": False, "missing": [], "extra": ["ghost"]}
    assert check_packages(
        reg, [ToolPackage("g", "g", "d", ["read_a", "write_b", "delegate"])])["ok"] is True


def test_check_packages_extra_names():
    # a tool known to the host but not registered (e.g. a meta tool) counts as declared
    reg = _reg()
    rep = check_packages(reg, [ToolPackage("g", "g", "d",
                                            ["read_a", "write_b", "delegate"])],
                         extra_names=("phantom",))
    assert rep["missing"] == ["phantom"]


def test_list_tool_packages_admin():
    pkgs = [ToolPackage("g", "组", "d", ["read_a", "write_b"])]
    out = list_tool_packages_admin(_reg(), pkgs, disabled={"write_b"})
    assert len(out) == 1 and out[0]["id"] == "g" and out[0]["display"] == "组"
    td = {t["name"]: t for t in out[0]["tools"]}
    assert td["read_a"]["category"] == "read" and td["read_a"]["enabled"] is True
    assert td["write_b"]["category"] == "write" and td["write_b"]["enabled"] is False


def test_package_of_and_meta():
    pkgs = [ToolPackage("g", "组", "d", ["read_a"])]
    assert package_of(pkgs, "read_a") == "g"
    assert package_of(pkgs, "nope") is None
    assert package_meta(pkgs, "g").display == "组"
    assert package_meta(pkgs, "x") is None
