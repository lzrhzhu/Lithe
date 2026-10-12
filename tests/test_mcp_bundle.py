"""lithe.bundles.mcp: bridge external MCP stdio servers into a registry.

Covers the handshake, tools/list -> ToolSpec translation (prefixing, schema
passthrough, readOnlyHint -> READ / default category), tools/call dispatch
(result text, isError -> ok=False, timeout, dead session), config parsing,
session reuse across attach() calls (same pid), respawn after close, graceful
degradation when a server fails to start, allow/blocklist filtering and
collision skipping — against the fake server subprocess, no host application.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lithe import AgentContext, AgentMode, ToolRegistry
from lithe.bundles.mcp import MCPManager, MCPServerConfig, parse_servers

FAKE = Path(__file__).parent / "fake_mcp_server.py"


def _cfg(**over) -> MCPServerConfig:
    kw = {"name": "fake", "command": [sys.executable, str(FAKE)],
          "env": {}, "startup_timeout": 15.0, "timeout": 5.0}
    kw.update(over)
    return MCPServerConfig(**kw)


def _ctx() -> AgentContext:
    return AgentContext(run_id="r", user_id="u")


async def test_attach_registers_prefixed_tools_with_categories():
    mgr = MCPManager([_cfg()])
    reg = ToolRegistry()
    try:
        report = await mgr.attach(reg)
        assert report["fake"] == ["fake__echo", "fake__boom",
                                  "fake__echo_env", "fake__slow"]
        auto = {t["function"]["name"] for t in reg.specs_for_mode(AgentMode.AUTONOMOUS)}
        anchored = {t["function"]["name"] for t in reg.specs_for_mode(AgentMode.ANCHORED)}
        assert "fake__echo" in auto and "fake__boom" in auto
        assert "fake__echo" in anchored and "fake__boom" not in anchored
        spec = reg.spec("fake__echo")
        assert spec.parameters["properties"]["text"]["type"] == "string"
        assert spec.category.value == "read"
        assert reg.spec("fake__boom").category.value == "write"
    finally:
        await mgr.close()


async def test_dispatch_ok_iserror_and_env_passthrough(tmp_path):
    log = tmp_path / "log"
    mgr = MCPManager([_cfg(env={"FAKE_SECRET": "s3cret", "FAKE_MCP_LOG": str(log)})])
    reg = ToolRegistry()
    try:
        await mgr.attach(reg)
        ctx = _ctx()
        ok = await reg.dispatch("fake__echo", {"text": "你好"}, ctx)
        assert ok.ok and ok.content == "echo:你好"
        assert ok.ui == [], "ui 必须是列表（字符串会被 runtime 逐字符当事件喷出）"
        boom = await reg.dispatch("fake__boom", {}, ctx)
        assert boom.ok is False and "炸了" in boom.content
        env = await reg.dispatch("fake__echo_env", {}, ctx)
        assert env.ok and env.content == "s3cret"
    finally:
        await mgr.close()
    assert log.exists() and len(log.read_text().splitlines()) >= 5


async def test_timeout_returns_error_result(tmp_path):
    mgr = MCPManager([_cfg(env={"FAKE_SLEEP": "2"}, timeout=0.3)])
    reg = ToolRegistry()
    try:
        await mgr.attach(reg)
        res = await reg.dispatch("fake__slow", {}, _ctx())
        assert res.ok is False and "超时" in res.summary
    finally:
        await mgr.close()


async def test_sessions_reused_across_attaches_and_respawn_after_close(tmp_path):
    log = tmp_path / "log"
    mgr = MCPManager([_cfg(env={"FAKE_MCP_LOG": str(log)})])
    try:
        await mgr.attach(ToolRegistry())
        await mgr.attach(ToolRegistry())
        pids = {line.split()[0] for line in log.read_text().splitlines()}
        assert len(pids) == 1, "第二个 registry 复用同一子进程"
        old_pid = int(pids.pop())
        assert mgr.status()["fake"]["alive"] and mgr.status()["fake"]["pid"] == old_pid
        await mgr._sessions["fake"].close()
        assert not mgr.status()["fake"]["alive"]
        report = await mgr.attach(ToolRegistry())
        assert isinstance(report["fake"], list) and len(report["fake"]) == 4
        pids = {line.split()[0] for line in log.read_text().splitlines()}
        assert old_pid not in pids, "断线后重新拉起子进程"
    finally:
        await mgr.close()


async def test_dead_session_dispatch_lazily_revives():
    """缓存型 registry + 死会话：工具调用应触发一次惰性重启并成功，而不是
    报“已断开”直到外部再调 ensure/attach。"""
    mgr = MCPManager([_cfg()])
    reg = ToolRegistry()
    try:
        await mgr.attach(reg)
        old_pid = mgr.status()["fake"]["pid"]
        await mgr._sessions["fake"].close()
        assert not mgr.status()["fake"]["alive"]
        res = await reg.dispatch("fake__echo", {"text": "x"}, _ctx())
        assert res.ok and res.content == "echo:x", "惰性重连后调用应自愈成功"
        assert mgr.status()["fake"]["alive"]
        assert mgr.status()["fake"]["pid"] != old_pid, "应是新拉起的子进程"
    finally:
        await mgr.close()


async def test_revive_failure_reports_disconnected(tmp_path):
    """重启也失败（命令失效）时才报“已断开”。"""
    mgr = MCPManager([_cfg(env={"FAKE_MCP_LOG": str(tmp_path / "log")})])
    reg = ToolRegistry()
    try:
        await mgr.attach(reg)
        await mgr._sessions["fake"].close()
        mgr.configs["fake"].command = ["/nonexistent/binary"]  # 重启必败
        res = await reg.dispatch("fake__echo", {"text": "x"}, _ctx())
        assert res.ok is False and "断开" in res.content
    finally:
        await mgr.close()


async def test_failed_server_degrades_gracefully():
    mgr = MCPManager([_cfg(command=["/nonexistent/binary", "--x"])])
    reg = ToolRegistry()
    try:
        report = await mgr.attach(reg)
        assert isinstance(report["fake"], str) and "启动失败" in report["fake"]
        assert reg.names() == []
    finally:
        await mgr.close()


async def test_allow_blocklist_and_collision_skip():
    mgr = MCPManager([_cfg(tool_allowlist=["echo", "slow"],
                           tool_blocklist=["slow"])])
    reg = ToolRegistry()
    reg_collide = ToolRegistry()
    from lithe import ToolSpec
    reg_collide.register(ToolSpec("fake__echo", "占位"),
                         _placeholder_handler)
    try:
        report = await mgr.attach(reg)
        assert report["fake"] == ["fake__echo"]
        await mgr.attach(reg_collide)  # 撞名跳过，不抛错
        assert "fake__echo" in reg_collide.names()
    finally:
        await mgr.close()


async def _placeholder_handler(ctx, args):
    from lithe import ToolResult
    return ToolResult(True, "占位")


def test_parse_servers_json_shapes():
    text = json.dumps([
        {"name": "zai", "command": "npx -y @z_ai/mcp-server",
         "env": {"Z_AI_API_KEY": "k"}, "default_category": "read"},
    ])
    cfgs = parse_servers(text)
    assert cfgs[0].command == ["npx", "-y", "@z_ai/mcp-server"]
    assert cfgs[0].default_category.value == "read"
    assert cfgs[0].tool_prefix() == "zai__"
    cfgs = parse_servers('{"local": {"command": ["python", "srv.py"]}}')
    assert cfgs[0].name == "local" and cfgs[0].timeout == 60.0
    assert parse_servers("") == []
    assert parse_servers("[]") == []
    with pytest.raises(ValueError):
        parse_servers('{"name": "x"}')
    with pytest.raises(ValueError):
        MCPManager([_cfg(), _cfg()])


async def test_concurrent_dispatch_isolated():
    mgr = MCPManager([_cfg()])
    reg = ToolRegistry()
    try:
        await mgr.attach(reg)
        results = await asyncio.gather(*[
            reg.dispatch("fake__echo", {"text": str(i)}, _ctx())
            for i in range(8)])
        assert all(r.ok for r in results)
        assert [r.content for r in results] == [f"echo:{i}" for i in range(8)]
    finally:
        await mgr.close()


async def test_close_graceful_and_stubborn_process_group():
    import os
    import time

    async def lifecycle(env: dict, grace: float) -> tuple[float, int]:
        mgr = MCPManager([_cfg(env=env)])
        try:
            await mgr.attach(ToolRegistry())
            proc = mgr._sessions["fake"].proc
            pid = proc.pid
            started = time.monotonic()
            await mgr._sessions["fake"].close(grace=grace)
            elapsed = time.monotonic() - started
            if os.name == "nt":
                assert proc.returncode is not None
            else:
                try:
                    os.kill(pid, 0)
                    raise AssertionError("进程仍存活")
                except ProcessLookupError:
                    pass
            return elapsed, pid
        finally:
            await mgr.close()

    graceful, _ = await lifecycle({}, grace=5.0)
    assert graceful < 5.0, "正常服务器应在宽限期内自行退出"
    stubborn, pid = await lifecycle({"FAKE_STUBBORN": "1"}, grace=0.5)
    assert stubborn < 10.0, "顽固服务器也应被进程组击杀并回收"
    if os.name == "nt":
        import re

        result = await asyncio.create_subprocess_exec(
            "tasklist", "/FI", f"PID eq {pid}", "/NH",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        assert result.returncode == 0
        assert re.search(rf"\b{pid}\b", stdout.decode()) is None


async def test_cached_registry_self_heals_after_restart():
    """缓存型 registry：session 重启后旧 handler 闭包按名解析到新 session，
    惰性重连让 dispatch 直接自愈，且 ensure 保持幂等。"""
    mgr = MCPManager([_cfg()])
    reg = ToolRegistry()
    try:
        await mgr.attach(reg)
        await mgr._sessions["fake"].close()
        assert not mgr.status()["fake"]["alive"]
        healed = await reg.dispatch("fake__echo", {"text": "y"}, _ctx())
        assert healed.ok and healed.content == "echo:y"
        names_before = set(reg.names())
        report = await mgr.ensure(reg)
        assert set(reg.names()) == names_before, "ensure 幂等，不重复注册"
        assert report["fake"] == []
    finally:
        await mgr.close()


async def test_server_initiated_request_does_not_corrupt_pending(tmp_path):
    """JSON-RPC 的客户端/服务器 id 空间是独立的：服务器在 tools/list 待决时
    推送一个同 id(=2) 的服务器请求，客户端绝不能拿它 resolve 自己的 future，
    且必须回复 method-not-found 让服务器不悬等。"""
    log = tmp_path / "log"
    mgr = MCPManager([_cfg(env={"FAKE_PUSH_REQUEST": "1",
                                "FAKE_MCP_LOG": str(log)})])
    reg = ToolRegistry()

    async def _wait_log(needle: str, timeout: float = 3.0) -> str:
        # attach() 在 tools/list 响应到达即返回，服务器消费我们的错误回复
        # 可能略晚 —— 短暂轮询而不是立刻断言，避免竞态假阴性。
        deadline = time.monotonic() + timeout
        text = ""
        while time.monotonic() < deadline:
            text = log.read_text() if log.exists() else ""
            if needle in text:
                return text
            await asyncio.sleep(0.02)
        return text

    try:
        report = await mgr.attach(reg)
        # 旧实现会把 tools/list 的 future 错误 resolve 成服务器请求 → 注册 0 个工具
        assert isinstance(report["fake"], list) and len(report["fake"]) == 4
        # 服务器收到了我们的 -32601 错误回复
        lines = await _wait_log("reply 2 error")
        assert "reply 2 error" in lines, "客户端应回 method-not-found"
        # 会话此后依旧可用
        res = await reg.dispatch("fake__echo", {"text": "hi"}, _ctx())
        assert res.ok and res.content == "echo:hi"
    finally:
        await mgr.close()


async def test_concurrent_revive_spawns_exactly_one_process(tmp_path):
    """READ 类 MCP 工具会被并行 gather：死会话上两个并发 dispatch 不得
    各拉起一个子进程（输家进程泄漏、close 收不到）。"""
    import asyncio

    log = tmp_path / "log"
    mgr = MCPManager([_cfg(env={"FAKE_MCP_LOG": str(log)})])
    reg = ToolRegistry()
    try:
        await mgr.attach(reg)
        old_pid = mgr.status()["fake"]["pid"]
        await mgr._sessions["fake"].close()
        assert not mgr.status()["fake"]["alive"]

        results = await asyncio.gather(
            reg.dispatch("fake__echo", {"text": "a"}, _ctx()),
            reg.dispatch("fake__echo", {"text": "b"}, _ctx()),
        )
        assert all(r.ok for r in results)

        pids = {line.split()[0] for line in log.read_text().splitlines()}
        new_pids = pids - {str(old_pid)}
        assert len(new_pids) == 1, f"并发复活应只拉起一个子进程，实际 {new_pids}"
        assert mgr.status()["fake"]["pid"] == int(new_pids.pop())
    finally:
        await mgr.close()


# --------------------------------------------------------------------------- #
# MCP
# --------------------------------------------------------------------------- #

class _PageSession:
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    async def call(self, method, params=None, *, timeout=None):
        self.calls.append((method, params))
        return self._pages.pop(0)


async def test_tools_list_pagination_followed():
    from lithe.bundles.mcp import _list_all_tools
    session = _PageSession([
        {"tools": [{"name": "a"}], "nextCursor": "p2"},
        {"tools": [{"name": "b"}], "nextCursor": "p3"},
        {"tools": [{"name": "c"}]},
    ])
    tools = await _list_all_tools(session, page_timeout=1.0)
    assert [t["name"] for t in tools] == ["a", "b", "c"]
    assert session.calls[1] == ("tools/list", {"cursor": "p2"})
    assert session.calls[2] == ("tools/list", {"cursor": "p3"})


def test_stdio_spawn_env_allowlist(monkeypatch):
    from lithe.bundles.mcp import MCPServerConfig, _StdioSession
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("AUTH_SECRET", "hunter2")
    monkeypatch.setenv("MY_API_KEY", "sk-leak")
    cfg = MCPServerConfig(name="s", command=["x"],
                          env={"TOOL_KEY": "tv"})
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    for key, value in {
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "TEMP": r"C:\Temp",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
    }.items():
        monkeypatch.setenv(key, value)
    env = _StdioSession(cfg)._spawn_env(windows=True)
    assert env["PATH"] == "/usr/bin" and env["TOOL_KEY"] == "tv"
    assert env["SYSTEMROOT"] == r"C:\Windows"
    assert env["WINDIR"] == r"C:\Windows"
    assert "AUTH_SECRET" not in env and "MY_API_KEY" not in env

    legacy = _StdioSession(
        MCPServerConfig(name="s", command=["x"], inherit_env=True)
    )._spawn_env()
    assert legacy.get("AUTH_SECRET") == "hunter2", "显式 True 保留旧行为"

    closed = _StdioSession(
        MCPServerConfig(name="s", command=["x"], inherit_env=False)
    )._spawn_env(windows=True)
    assert closed.get("TOOL_KEY") is None
    assert set(closed).issubset({
        "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "USERPROFILE",
        "HOMEDRIVE", "HOMEPATH", "TEMP", "TMP", "OS", "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
    })


def test_log_host_strips_credentials():
    from lithe.bundles.mcp import _log_host
    assert _log_host("https://user:pass@evil.example/api?k=secret") == \
        "https://evil.example"
    assert _log_host("https://plain.example/api") == "https://plain.example"
    assert _log_host(None) == "?"


def test_parse_servers_error_does_not_embed_value():
    from lithe.bundles.mcp import parse_servers
    with pytest.raises(ValueError) as ei:
        parse_servers(json.dumps({"bad": "https://k@host/?key=SECRET"}))
    assert "SECRET" not in str(ei.value)


async def test_stdio_read_loop_ignores_non_object_json():
    """stdout 出现 JSON 数组/标量行不得杀死会话（reader 不得 AttributeError）。"""
    from lithe.bundles.mcp import MCPServerConfig, _StdioSession
    session = _StdioSession(MCPServerConfig(name="s", command=["x"]))
    reader = asyncio.StreamReader()
    reader.feed_data(b'[1,2]\n"hello"\n42\n')
    reader.feed_data(b'{"jsonrpc":"2.0","id":7,"result":{"ok":true}}\n')
    reader.feed_eof()
    fut = asyncio.get_running_loop().create_future()
    session._pending[7] = fut
    session.proc = SimpleNamespace(stdout=reader, stdin=None)
    await session._read_loop()
    assert fut.done() and fut.result()["result"] == {"ok": True}
