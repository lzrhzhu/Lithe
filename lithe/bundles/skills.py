"""Skills bundle: skill libraries + the ``load_skill`` tool.

Three layers, all host-optional:

- :class:`SkillLibrary` — a plain directory of ``*.md`` skill files. Skill id =
  file stem (globally unique; a collision is reported so the host can fix it).
- :class:`SkillPackages` — the package-aware superset: skills grouped into
  *packages* (subdirectories, each optionally carrying a ``package.md`` with
  frontmatter metadata), an enabled-package filter, and remote registry
  mirrors via :class:`RemoteSkillSource`. Duck-types :class:`SkillLibrary`
  (``index_text`` / ``resolve`` / ``load``) so it plugs straight into
  :func:`register_skill_tool`.
- :func:`register_skill_tool` — the ``load_skill`` tool: no args lists skills,
  with a name loads that skill's full markdown into context.

The *mechanism* is generic; skill *content* (thesis-writing,
numerical-computing, ...) is whatever markdown the host drops in the
directory. Skills do file (and, with remote sources, network) I/O — that is
why they live in this optional bundle and not in the zero-I/O core.

Optional bundle — the core engine never imports this.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path

import httpx

from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec

log = logging.getLogger("lithe.skills")

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
PACKAGE_FILE = "package.md"
ROOT_PACKAGE = "_root"          # skills placed directly in root (no subdir)
SKILL_MAIN = "SKILL.md"         # canonical single-file skill body served remotely


# --------------------------------------------------------------------------- #
# shared markdown helpers
# --------------------------------------------------------------------------- #

def _frontmatter(md: Path) -> dict:
    """Best-effort flat YAML frontmatter (``key: value`` lines only)."""
    out: dict = {}
    try:
        text = md.read_text(encoding="utf-8")
    except OSError:
        return out
    if not text.startswith("---"):
        return out
    for line in text.splitlines()[1:]:
        if line.strip() == "---":
            break
        if ":" in line and not line.startswith((" ", "\t", "-")):
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _first_text_line(md: Path) -> str:
    """First non-heading, non-frontmatter line, capped for index display."""
    try:
        text = md.read_text(encoding="utf-8")
    except OSError:
        return ""
    in_fm = text.startswith("---")
    for line in text.splitlines():
        s = line.strip()
        if in_fm:
            if s == "---":
                in_fm = False
            continue
        if s and not s.startswith("#") and not s.startswith("---"):
            return s[:80]
    return ""


def _skill_description(md: Path) -> str:
    fm = _frontmatter(md)
    return fm.get("description") or _first_text_line(md)


# --------------------------------------------------------------------------- #
# SkillLibrary (flat directory)
# --------------------------------------------------------------------------- #

class SkillLibrary:
    """A directory of ``*.md`` skill files.

    ``package.md`` files are metadata, not skills. Subdirectories are just
    grouping — the bundle flattens them (a host wanting package-level
    enable/disable uses :class:`SkillPackages` instead).
    """

    def __init__(self, root, *, package_file: str = PACKAGE_FILE,
                 max_read: int = 20000):
        self.root = Path(root)
        self.package_file = package_file
        self.max_read = max_read

    def files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.rglob("*.md")
                      if p.name != self.package_file)

    def description(self, md: Path) -> str:
        return _skill_description(md)

    def resolve(self, name: str) -> tuple[Path | None, str | None]:
        """Map a bare stem to its path. Returns ``(path, error)``; ``(None, None)``
        means not-found (no error), ``(None, msg)`` means invalid/colliding."""
        if not name or not _NAME_RE.match(name):
            return None, f"无效的技能名：{name}（仅允许字母、数字、下划线、短横线）"
        matches = [p for p in self.files() if p.stem == name]
        if not matches:
            return None, None
        if len(matches) > 1:
            return None, f"技能名 {name} 在多处重复，请避免重名"
        return matches[0], None

    def names(self, disabled: frozenset[str] = frozenset()) -> list[str]:
        return [md.stem for md in self.files() if md.stem not in disabled]

    def index_text(self, disabled: frozenset[str] = frozenset()) -> str:
        lines = [f"- {md.stem}：{self.description(md)}"
                 for md in self.files() if md.stem not in disabled]
        return "\n".join(lines) or "（暂无可用技能）"

    def load(self, name: str) -> str | None:
        path, _err = self.resolve(name)
        if path is None:
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        return content[:self.max_read]


# --------------------------------------------------------------------------- #
# SkillPackages (package-aware, multi-root, remote mirrors)
# --------------------------------------------------------------------------- #

class SkillPackages:
    """Read-only, package-aware skill library (local root + remote caches).

    A host enables a subset of packages and may mirror a remote registry (the
    Kilo ``index.json`` protocol) via :class:`RemoteSkillSource`, so skills
    load from a central skill server without a local checkout. ``ttl=0`` + a
    fresh :class:`SkillPackages` per run means server-side edits/toggles take
    effect on the next call; the index-signature short-circuit keeps it cheap
    (only ``index.json`` is fetched unless the manifest changed).

    All methods are synchronous and may do network I/O (via the remote
    sources). On an event loop, go through the ``load_skill`` tool (it
    offloads), or wrap direct calls — e.g. ``index_text`` in a prompt
    builder — in ``asyncio.to_thread`` yourself.
    """

    def __init__(self, root, *, enabled_packages=None,
                 package_file: str = PACKAGE_FILE, max_read: int = 20000,
                 remote_sources=None):
        self.root = Path(root)
        self.enabled_packages = (
            None if enabled_packages is None else frozenset(enabled_packages))
        self.package_file = package_file
        self.max_read = max_read
        self._remote = list(remote_sources or [])
        self._remote_refreshed = False

    # --------------------------------------------------------------- multi-root
    def _all_roots(self) -> list[Path]:
        self._ensure_remote()
        roots = [self.root]
        for rs in self._remote:
            if rs.cache_root.is_dir():
                roots.append(rs.cache_root)
        return roots

    def _root_of(self, path: Path) -> Path:
        path = Path(path)
        for root in self._all_roots():
            try:
                path.relative_to(root)
                return root
            except ValueError:
                continue
        return self.root

    def _ensure_remote(self) -> None:
        if self._remote_refreshed or not self._remote:
            return
        for rs in self._remote:
            try:
                rs.refresh()
            except Exception:  # noqa: BLE001 — offline-tolerant
                log.debug("remote skill source %s refresh failed; "
                          "using previous cache", rs.base_url, exc_info=True)
        self._remote_refreshed = True

    # ------------------------------------------------------------------ scan
    def package_of(self, path: Path) -> str:
        root = self._root_of(path)
        try:
            rel = path.relative_to(root)
        except ValueError:
            return ROOT_PACKAGE
        parts = rel.parts
        return parts[0] if len(parts) >= 2 else ROOT_PACKAGE

    def files(self) -> list[Path]:
        out: list[Path] = []
        seen: set[str] = set()
        for root in self._all_roots():
            if not root.is_dir():
                continue
            root_stems: set[str] = set()
            for p in sorted(root.rglob("*.md")):
                if p.name == self.package_file:
                    continue
                if self.enabled_packages is not None and \
                        self.package_of(p) not in self.enabled_packages:
                    continue
                if p.stem in seen:
                    continue
                out.append(p)
                root_stems.add(p.stem)
            seen |= root_stems
        return out

    # ---------------------------------------------------- SkillLibrary-compatible
    def frontmatter(self, md: Path) -> dict:
        return _frontmatter(md)

    def description(self, md: Path) -> str:
        return _skill_description(md)

    def resolve(self, name: str) -> tuple[Path | None, str | None]:
        if not name or not _NAME_RE.match(name):
            return None, f"无效的技能名：{name}（仅允许字母、数字、下划线、短横线）"
        matches = [p for p in self.files() if p.stem == name]
        if not matches:
            return None, None
        if len(matches) > 1:
            pkgs = ", ".join(sorted(self.package_of(p) for p in matches))
            return None, f"技能名 {name} 在多个包中重复（{pkgs}），请避免重名"
        return matches[0], None

    def index_text(self, disabled: frozenset[str] = frozenset()) -> str:
        lines = [f"- {md.stem}：{self.description(md)}"
                 for md in self.files() if md.stem not in disabled]
        return "\n".join(lines) or "（暂无可用技能）"

    def load(self, name: str) -> str | None:
        path, _err = self.resolve(name)
        if path is None:
            return None
        try:
            return path.read_text(encoding="utf-8")[:self.max_read]
        except OSError:
            return None

    # -------------------------------------------------------- package-aware API
    def is_enabled(self, stem: str, package: str,
                   disabled_skills=None, disabled_packages=None) -> bool:
        return (stem not in (disabled_skills or ())) and \
               (package not in (disabled_packages or ()))

    def list_skills_text(self, disabled_skills=None,
                         disabled_packages=None) -> str:
        ds = frozenset(disabled_skills or ())
        dp = frozenset(disabled_packages or ())
        lines = [f"- {md.stem}：{self.description(md)}"
                 for md in self.files()
                 if self.is_enabled(md.stem, self.package_of(md), ds, dp)]
        return "\n".join(lines) or "（暂无可用技能）"

    def package_meta(self, pkg: str) -> dict:
        if pkg == ROOT_PACKAGE:
            return {"id": pkg, "name": "builtin",
                    "description": "直接放在技能根目录下的技能",
                    "source": "", "version": ""}
        fm: dict = {}
        for root in self._all_roots():
            meta_file = root / pkg / self.package_file
            if meta_file.is_file():
                fm = _frontmatter(meta_file)
                break
        return {
            "id": pkg,
            "name": fm.get("name") or pkg,
            "description": fm.get("description", ""),
            "source": fm.get("source", ""),
            "version": fm.get("version", ""),
        }


# --------------------------------------------------------------------------- #
# RemoteSkillSource (Kilo index.json registry mirror)
# --------------------------------------------------------------------------- #

class RemoteSkillSource:
    """A remote skill registry (Kilo ``index.json`` protocol) mirrored to a
    local cache dir so :class:`SkillPackages` can treat it as an extra
    read-only root.

    Each remote skill ``<name>/SKILL.md`` is cached as ``<package>/<name>.md``
    — the local flat convention — so the skill stem and package mapping survive
    the round trip. :class:`SkillPackages` calls :meth:`refresh` once on first
    use (offline-tolerant: a network failure leaves the previous cache).

    ``client`` may be injected (anything with ``get(url, timeout=...)``
    returning an object with ``status_code``/``json()``/``raise_for_status()``,
    e.g. an ``httpx.Client`` over an ``ASGITransport`` for in-process tests).
    """

    def __init__(self, base_url: str, cache_dir, *, timeout: float = 10.0,
                 ttl: float = 300.0, client=None):
        self.base_url = base_url.rstrip("/") + "/"
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.ttl = ttl
        self._client = client
        self._last_refresh = 0.0
        self._index_sig: str | None = None
        self._lock = threading.Lock()

    @property
    def cache_root(self) -> Path:
        return self.cache_dir

    def _get(self, client, path: str):
        return client.get(self.base_url + path, timeout=self.timeout)

    def refresh(self) -> None:
        """Synchronize the cache with the remote registry (blocking, sync).

        Callers on an event loop must offload this (e.g. ``asyncio.to_thread``)
        — it performs real network and filesystem I/O;
        :func:`register_skill_tool`'s handler already does.
        """
        with self._lock:
            if self._last_refresh and time.monotonic() - self._last_refresh < self.ttl:
                return
            own = self._client is None
            client = self._client or httpx.Client()
            staging = self.cache_dir.parent / (self.cache_dir.name + ".staging")
            try:
                resp = self._get(client, "index.json")
                resp.raise_for_status()
                skills = resp.json().get("skills", [])
                sig = json.dumps(
                    [(s.get("name"), s.get("package"), s.get("version"))
                     for s in skills], sort_keys=True)
                if sig == self._index_sig and self.cache_dir.is_dir():
                    self._last_refresh = time.monotonic()
                    return
                if staging.exists():
                    shutil.rmtree(staging)
                staging.mkdir(parents=True)
                for entry in skills:
                    name = entry.get("name")
                    if not name or not _NAME_RE.match(name):
                        continue
                    files = entry.get("files") or [SKILL_MAIN]
                    if SKILL_MAIN not in files:
                        continue
                    r = self._get(client, f"{name}/{SKILL_MAIN}")
                    if r.status_code != 200:
                        continue
                    pkg = entry.get("package") or "remote"
                    if not _NAME_RE.match(pkg):
                        pkg = "remote"
                    pkg_dir = staging / pkg
                    pkg_dir.mkdir(parents=True, exist_ok=True)
                    (pkg_dir / f"{name}.md").write_bytes(r.content)
            except Exception:
                # Half-downloaded staging must not be left behind (it would
                # be rmtree'd on the NEXT refresh, but until then it's debris
                # a host might mistake for content). The previous cache stays
                # authoritative; count this as a refresh so a dead server
                # isn't hammered on every call.
                shutil.rmtree(staging, ignore_errors=True)
                self._last_refresh = time.monotonic()
                log.debug("skill refresh from %s failed; keeping previous "
                          "cache", self.base_url, exc_info=True)
                return
            finally:
                if own:
                    client.close()
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            staging.rename(self.cache_dir)
            # Commit the signature only after the swap: a failed refresh
            # must not leave the new sig recorded over the old cache (the
            # next refresh would then short-circuit on a stale-match).
            self._index_sig = sig
            self._last_refresh = time.monotonic()

    @classmethod
    def from_urls(cls, urls, cache_dir, *, cache=None, **kwargs) -> list:
        """Build a memoized list of RemoteSkillSource, one per URL. ``cache`` is
        a process-level dict shared across calls so sources (and their cache +
        TTL) persist across per-run SkillPackages rebuilds."""
        cache = {} if cache is None else cache
        root = Path(cache_dir)
        out = []
        for url in urls:
            src = cache.get(url)
            if src is None:
                sub = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
                src = cls(url, root / sub, **kwargs)
                cache[url] = src
            out.append(src)
        return out


# --------------------------------------------------------------------------- #
# load_skill tool
# --------------------------------------------------------------------------- #

def register_skill_tool(
    registry: ToolRegistry,
    library: SkillLibrary,
    *,
    disabled_for=None,
    name: str = "load_skill",
) -> None:
    """Register a ``load_skill`` tool: with no args it lists enabled skills; with a
    name it loads that skill's full markdown into context.

    ``library`` may be a :class:`SkillLibrary` or anything duck-typing it
    (``index_text`` / ``resolve`` / ``load`` — :class:`SkillPackages` does).
    ``disabled_for(ctx)`` lets a host hide skills per-context (e.g. an admin
    disabled some); omit it to expose all skills everywhere.

    Library calls are offloaded to a worker thread via ``asyncio.to_thread``:
    a :class:`SkillPackages` over :class:`RemoteSkillSource`\\ s does sync
    network + filesystem work (a registry refresh), which must never block the
    event loop — one slow skill server would otherwise freeze every concurrent
    agent run in the process. Hosts calling the library directly (e.g. baking
    ``index_text`` into a system prompt) are responsible for the same offload.
    """
    async def load_skill(ctx, args):
        disabled = disabled_for(ctx) if disabled_for else frozenset()
        req = (args.get("name") or "").strip()
        if not req:  # list mode
            listing = await asyncio.to_thread(library.index_text, disabled)
            return ToolResult(True, "可用技能列表",
                              "可用技能（用 load_skill(name) 加载详细规范）：\n"
                              + listing)
        if req in disabled:
            return ToolResult(False, "已停用", f"技能 {req} 已被停用。")
        _path, err = await asyncio.to_thread(library.resolve, req)
        if err:
            return ToolResult(False, "重名/无效", err)
        content = await asyncio.to_thread(library.load, req)
        if content is None:
            avail = await asyncio.to_thread(library.index_text, disabled)
            return ToolResult(False, "未找到",
                              f"未找到技能 {req}。可用：\n" + avail)
        return ToolResult(True, f"加载技能 {req}", content)

    registry.register(
        ToolSpec(name,
                 "加载一项工作技能的详细规范到上下文。不带参数时列出可用技能。",
                 {"type": "object",
                  "properties": {"name": {"type": "string",
                                          "description": "技能名（文件名去 .md）"}}},
                 ToolCategory.READ),
        load_skill)


__all__ = ["PACKAGE_FILE", "ROOT_PACKAGE", "SKILL_MAIN", "RemoteSkillSource",
           "SkillLibrary", "SkillPackages", "register_skill_tool"]
