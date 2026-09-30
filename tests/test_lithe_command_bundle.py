from __future__ import annotations

import sys

from lithe import AgentContext, ToolCategory, ToolRegistry
from lithe.bundles.command import (
    CommandRunner,
    _command_argv,
    _command_env,
    register_command_tools,
)


def _runner(**kwargs) -> CommandRunner:
    return CommandRunner(timeout=kwargs.pop("timeout", 5), **kwargs)


async def test_run_command_auto_shell_in_workspace(tmp_path):
    result = await _runner().run(tmp_path, "printf 'hello'; pwd")
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
    assert str(tmp_path) in result["stdout"]


async def test_run_command_returns_failure_and_stderr(tmp_path):
    result = await _runner().run(tmp_path, "printf 'failure' >&2; exit 7")
    assert result["exit_code"] == 7
    assert "failure" in result["stderr"]


async def test_run_command_timeout_kills_process_group(tmp_path):
    result = await _runner(timeout=0.3).run(tmp_path, "sleep 10")
    assert result["timed_out"]
    assert result["exit_code"] == -1
    assert "超时" in result["stderr"]


async def test_run_command_truncates_output(tmp_path):
    result = await _runner(max_output=100).run(
        tmp_path, f"{sys.executable} -c \"print('x' * 1000)\""
    )
    assert result["exit_code"] == 0
    assert len(result["stdout"]) <= 200
    assert "已省略中间部分" in result["stdout"]


async def test_run_command_accepts_stdin(tmp_path):
    result = await _runner().run(tmp_path, "read value; printf '%s' \"$value\"", stdin="input")
    assert result["exit_code"] == 0
    assert result["stdout"] == "input"


async def test_register_command_tool_is_write_classified(tmp_path):
    registry = ToolRegistry()
    register_command_tools(registry, lambda ctx: tmp_path, _runner())
    spec = registry.spec("run_command")
    assert spec is not None and spec.category is ToolCategory.WRITE
    assert registry.specs_for_mode("anchored") == []
    result = await registry.dispatch(
        "run_command", {"command": "printf registered"},
        AgentContext(run_id="r", user_id="u"),
    )
    assert result.ok and "registered" in result.content


async def test_run_command_rejects_empty_command(tmp_path):
    registry = ToolRegistry()
    register_command_tools(registry, lambda ctx: tmp_path, _runner())
    result = await registry.dispatch(
        "run_command", {"command": " "}, AgentContext(run_id="r", user_id="u")
    )
    assert not result.ok


def test_command_env_excludes_lithe_secrets(monkeypatch):
    monkeypatch.setenv("LITHE_API_KEY", "secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = _command_env()
    assert env["PATH"] == "/usr/bin"
    assert "LITHE_API_KEY" not in env


def test_command_argv_selects_windows_native_shell(monkeypatch):
    monkeypatch.setattr("lithe.bundles.command._find_shell", lambda names: "pwsh.exe")
    assert _command_argv("Write-Output ok", windows=True)[0] == "pwsh.exe"
    assert _command_argv("Write-Output ok", "powershell", windows=True)[-1] == "Write-Output ok"


def test_command_argv_supports_cmd_fallback(monkeypatch):
    monkeypatch.setattr(
        "lithe.bundles.command._find_shell",
        lambda names: "cmd.exe" if "cmd.exe" in names else None,
    )
    argv = _command_argv("echo ok", windows=True)
    assert argv == ["cmd.exe", "/d", "/s", "/c", "echo ok"]
