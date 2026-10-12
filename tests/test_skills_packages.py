"""SkillPackages (package-aware local + remote mirror) + RemoteSkillSource.
Pins local scanning / package filter / frontmatter + a remote refresh against
an injected fake HTTP client — no network."""
from __future__ import annotations

import asyncio
import time

import httpx

from lithe import AgentContext, ToolRegistry
from lithe.bundles import SKILL_MAIN, RemoteSkillSource, SkillPackages
from lithe.bundles.skills import register_skill_tool


def _make_local(tmp_path):
    pkg = tmp_path / "cpss-core"
    pkg.mkdir()
    (pkg / "package.md").write_text(
        "---\nname: cpss-core\ndescription: d\n---\n", encoding="utf-8")
    (pkg / "question-authoring.md").write_text(
        "---\ndescription: write questions\n---\n# QA\nbody", encoding="utf-8")
    other = tmp_path / "other-pkg"
    other.mkdir()
    (other / "misc.md").write_text(
        "---\ndescription: misc\n---\n# misc", encoding="utf-8")
    return tmp_path


def test_scan_and_filter_by_package(tmp_path):
    root = _make_local(tmp_path)
    lib = SkillPackages(root, enabled_packages={"cpss-core"})
    names = sorted(md.stem for md in lib.files())
    assert names == ["question-authoring"]  # other-pkg filtered out
    assert "question-authoring：write questions" in lib.index_text()
    path, err = lib.resolve("question-authoring")
    assert path is not None and err is None
    assert "# QA" in lib.load("question-authoring")
    assert lib.package_of(path) == "cpss-core"


def test_package_meta(tmp_path):
    root = _make_local(tmp_path)
    lib = SkillPackages(root)
    meta = lib.package_meta("cpss-core")
    assert meta["name"] == "cpss-core" and meta["description"] == "d"


class _FakeResp:
    def __init__(self, status, data=None, content=None):
        self.status_code = status
        self._data = data
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data

    @property
    def content(self):
        return self._content


class _FakeClient:
    def __init__(self, index_body, skill_bodies):
        self._index = index_body
        self._skills = skill_bodies  # name -> content bytes
        self.closed = False

    def get(self, url, timeout=None):
        if url.endswith("index.json"):
            return _FakeResp(200, data=self._index)
        # url = .../{name}/SKILL.md
        name = url.rstrip("/").split("/")[-2]
        return _FakeResp(200, content=self._skills.get(name, b""))

    def close(self):
        self.closed = True


def test_remote_source_mirrors_to_cache(tmp_path):
    cache = tmp_path / "cache"
    index = {"skills": [
        {"name": "approval-discipline", "package": "cpss-core",
         "files": [SKILL_MAIN], "version": "1.0"},
    ]}
    bodies = {"approval-discipline": b"---\ndescription: approval\n---\n# AD"}
    src = RemoteSkillSource("https://skills.example/cpss/", cache,
                            ttl=0, client=_FakeClient(index, bodies))
    src.refresh()
    cached = cache / "cpss-core" / "approval-discipline.md"
    assert cached.is_file()
    assert b"# AD" in cached.read_bytes()


def test_skillpackages_reads_remote_cache(tmp_path):
    root = tmp_path / "empty-local"
    root.mkdir()
    cache = tmp_path / "cache"
    index = {"skills": [
        {"name": "bank-safety", "package": "cpss-core", "files": [SKILL_MAIN]},
    ]}
    bodies = {"bank-safety": b"---\ndescription: safety\n---\n# BS"}
    src = RemoteSkillSource("https://skills.example/cpss/", cache,
                            ttl=0, client=_FakeClient(index, bodies))
    lib = SkillPackages(root, enabled_packages={"cpss-core"},
                        remote_sources=[src])
    names = sorted(md.stem for md in lib.files())
    assert names == ["bank-safety"]
    assert "bank-safety：safety" in lib.index_text()


async def test_load_skill_tool_offloads_blocking_refresh(tmp_path):
    """load_skill 的库访问必须跑在工作线程：远程刷新是同步网络 I/O，
    若留在事件循环上，一个慢注册服务器会冻结进程内所有并发 agent。"""
    root = tmp_path / "local"
    root.mkdir()
    cache = tmp_path / "cache"
    index = {"skills": [{"name": "slow-skill", "package": "p",
                         "files": [SKILL_MAIN]}]}

    class _SlowClient:
        def get(self, url, timeout=None):
            time.sleep(0.15)  # 同步阻塞，模拟一次慢刷新
            if url.endswith("index.json"):
                return _FakeResp(200, data=index)
            return _FakeResp(200, content=b"---\ndescription: s\n---\n# S")

    src = RemoteSkillSource("https://skills.example/", cache,
                            ttl=0, client=_SlowClient())
    lib = SkillPackages(root, remote_sources=[src])
    reg = ToolRegistry()
    register_skill_tool(reg, lib)
    ctx = AgentContext(run_id="r", user_id="u")

    beats = 0

    async def heartbeat():
        nonlocal beats
        while True:
            await asyncio.sleep(0.01)
            beats += 1

    hb = asyncio.create_task(heartbeat())
    try:
        res = await reg.dispatch("load_skill", {"name": "slow-skill"}, ctx)
        assert res.ok and "# S" in res.content
        # 刷新期间事件循环仍在调度（阻塞实现下 beats 停在 0）
        assert beats > 5
    finally:
        hb.cancel()


# --------------------------------------------------------------------------- #
# skills: remote refresh cleanup (the deprecated lithe.skills alias was
# removed in the 0.1.10 reorganization — import from lithe.bundles)
# --------------------------------------------------------------------------- #


def test_remote_refresh_failure_cleans_staging(tmp_path, caplog):
    from lithe.bundles.skills import RemoteSkillSource

    class _BoomClient:
        def get(self, url, timeout=None):
            raise httpx.ConnectError("down")

        def close(self):
            pass

    src = RemoteSkillSource("http://down.example/", tmp_path / "cache",
                            ttl=0, client=_BoomClient())
    with caplog.at_level("DEBUG"):
        src.refresh()
    assert not (tmp_path / "cache.staging").exists(), "失败不得遗留 staging"
    assert not (tmp_path / "cache").exists()


def _resp_ok(json_body=None, text=None, url="http://x/"):
    kw = {"json": json_body} if json_body is not None else {"text": text}
    return httpx.Response(200, request=httpx.Request("GET", url), **kw)


def test_remote_refresh_signature_not_committed_on_failure(tmp_path):
    from lithe.bundles.skills import RemoteSkillSource

    good = {"skills": [{"name": "a", "package": "p", "version": "1"}]}
    body = "# skill a\n"

    class _HalfClient:
        def get(self, url, timeout=None):
            if "index.json" in url:
                return _resp_ok(json_body=good, url=url)
            raise httpx.ConnectError("mid-download failure")

        def close(self):
            pass

    class _OkClient:
        def get(self, url, timeout=None):
            if "index.json" in url:
                return _resp_ok(json_body=good, url=url)
            return _resp_ok(text=body, url=url)

        def close(self):
            pass

    src = RemoteSkillSource("http://x/", tmp_path / "c2", ttl=0,
                            client=_HalfClient())
    src.refresh()  # 半途失败：缓存不换，签名不提交
    assert not (tmp_path / "c2").exists()
    assert src._index_sig is None
    # 之后服务器恢复：同一 index 必须仍然触发完整刷新（不得因错误的
    # 签名提交而短路）
    src2 = RemoteSkillSource("http://x/", tmp_path / "c3", ttl=0,
                             client=_OkClient())
    src2.refresh()
    assert (tmp_path / "c3" / "p" / "a.md").read_text(encoding="utf-8") == body
