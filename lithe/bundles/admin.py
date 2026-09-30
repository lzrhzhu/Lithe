"""Admin bundle: tool-management helpers over a :class:`~lithe.tools.ToolRegistry`.

Source for an admin panel: group the registered tools by category and by display
package, flag enabled/disabled, and check package integrity at startup. Pure over
the registry + a packages list — **packages carry no enable/disable semantics**
(tools stay togglable per-name); they are display grouping only, exactly as a
host's ``TOOL_PACKAGES`` already are.

The package *content* (ids, displays, which tools belong where) is host data
passed in as :class:`ToolPackage`; only the *mechanism* lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Collection, Iterable

from lithe.tools import ToolRegistry

_CAT_ORDER = ("read", "write", "meta")


@dataclass
class ToolPackage:
    """Display-only grouping of tools (no enable/disable semantics)."""
    id: str
    display: str
    description: str = ""
    tools: list[str] = field(default_factory=list)


def _entry(ts) -> dict:
    return {"name": ts.name, "description": ts.description or "",
            "parameters": ts.parameters or {"type": "object", "properties": {}}}


def tool_categories(registry: ToolRegistry) -> dict[str, list[dict]]:
    """Group every registered tool by read/write/meta category.

    Each entry is a compact ``{name, description, parameters}`` (no OpenAI
    envelope) suitable for display. Order within a category is registration order.
    """
    cats: dict[str, list[dict]] = {c: [] for c in _CAT_ORDER}
    for name in registry.names():
        ts = registry.spec(name)
        if ts is None:
            continue
        cats.setdefault(ts.category.value, []).append(_entry(ts))
    return cats


def package_of(packages: Iterable[ToolPackage], name: str) -> str | None:
    """Return the package id a tool belongs to, or None if unregistered."""
    for p in packages:
        if name in p.tools:
            return p.id
    return None


def package_meta(packages: Iterable[ToolPackage], pkg_id: str) -> ToolPackage | None:
    """Return the raw package (id/display/description/tools) or None."""
    for p in packages:
        if p.id == pkg_id:
            return p
    return None


def list_tools_admin(registry: ToolRegistry, *,
                     disabled: Collection[str] = (),
                     packages: Iterable[ToolPackage] = ()) -> list[dict]:
    """Flat tool list for the admin panel, ordered read → write → meta.

    Each entry: ``{name, description, parameters, category, package, enabled}``.
    ``disabled`` is the per-name set the host turns tools off with; there is no
    package-level disable.
    """
    dis = frozenset(disabled or ())
    pkgs = tuple(packages or ())
    buckets: dict[str, list[dict]] = {c: [] for c in _CAT_ORDER}
    for name in registry.names():
        ts = registry.spec(name)
        if ts is None:
            continue
        cat = ts.category.value
        buckets.setdefault(cat, []).append(
            {**_entry(ts), "category": cat,
             "package": package_of(pkgs, name), "enabled": name not in dis})
    return [e for c in _CAT_ORDER for e in buckets.get(c, [])]


def list_tool_packages_admin(registry: ToolRegistry,
                             packages: Iterable[ToolPackage], *,
                             disabled: Collection[str] = ()) -> list[dict]:
    """Group every tool by its package for the admin panel (display only).

    One entry per package in the given order; each package carries
    ``id`` / ``display`` / ``description`` and a ``tools[]`` list where every tool
    has ``name`` / ``description`` / ``parameters`` / ``category`` / ``enabled``.
    """
    dis = frozenset(disabled or ())
    cat_of = {e["name"]: cat for cat, entries in tool_categories(registry).items()
              for e in entries}
    out: list[dict] = []
    for p in packages:
        tools = []
        for name in p.tools:
            ts = registry.spec(name)
            entry = _entry(ts) if ts is not None else {
                "name": name, "description": "", "parameters": {}}
            tools.append({**entry, "category": cat_of.get(name, ""),
                          "enabled": name not in dis})
        out.append({"id": p.id, "display": p.display,
                    "description": p.description, "tools": tools})
    return out


def check_packages(registry: ToolRegistry,
                   packages: Iterable[ToolPackage], *,
                   extra_names: Collection[str] = ()) -> dict:
    """Integrity check: packages must cover every tool exactly once.

    Returns ``{'ok': bool, 'missing': [...], 'extra': [...]}`` for startup
    logging. ``missing`` = registry tools (plus ``extra_names``) with no package;
    ``extra`` = registered names that no longer exist. Non-blocking.
    """
    declared = set(registry.names()) | set(extra_names or ())
    registered = {t for p in packages for t in p.tools}
    return {"ok": declared == registered,
            "missing": sorted(declared - registered),
            "extra": sorted(registered - declared)}


__all__ = ["ToolPackage", "check_packages", "list_tool_packages_admin",
           "list_tools_admin", "package_meta", "package_of", "tool_categories"]
