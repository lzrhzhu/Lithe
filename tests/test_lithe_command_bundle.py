from __future__ import annotations

import base64
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
    command = (
        "Write-Output 'hello'; Get-Location"
        if sys.platform == "win32"
        else "printf 'hello'; pwd"
    )
    result = await _runner().run(tmp_path, command)
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
    assert str(tmp_path) in result["stdout"]


async def test_run_command_returns_failure_and_stderr(tmp_path):
    command = (
        "[Console]::Error.WriteLine('failure'); exit 7"
        if sys.platform == "win32"
        else "printf 'failure' >&2; exit 7"
    )
    result = await _runner().run(tmp_path, command)
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
    command = (
        "$value = [Console]::In.ReadToEnd(); [Console]::Out.Write($value)"
        if sys.platform == "win32"
        else "read value; printf '%s' \"$value\""
    )
    result = await _runner().run(tmp_path, command, stdin="input")
    assert result["exit_code"] == 0
    assert result["stdout"] == "input"


async def test_register_command_tool_is_write_classified(tmp_path):
    registry = ToolRegistry()
    register_command_tools(registry, lambda ctx: tmp_path, _runner())
    spec = registry.spec("run_command")
    assert spec is not None and spec.category is ToolCategory.WRITE
    assert registry.specs_for_mode("anchored") == []
    command = "Write-Output registered" if sys.platform == "win32" else "printf registered"
    result = await registry.dispatch(
        "run_command", {"command": command},
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
    argv = _command_argv("Write-Output ok", "powershell", windows=True)
    assert argv[-2] == "-EncodedCommand"
    assert base64.b64decode(argv[-1]).decode("utf-16le").endswith("Write-Output ok")


def test_command_argv_supports_cmd_fallback(monkeypatch):
    monkeypatch.setattr("lithe.bundles.command.os", type("OS", (), {"name": "nt", "environ": {}}))
    monkeypatch.setattr(
        "lithe.bundles.command._find_shell",
        lambda names: "cmd.exe" if "cmd.exe" in names else None,
    )
    argv = _command_argv("echo ok", windows=True)
    assert argv == ["cmd.exe", "/d", "/s", "/c", "chcp 65001 >NUL & echo ok"]


def test_command_argv_configures_powershell_utf8(monkeypatch):
    monkeypatch.setattr("lithe.bundles.command._find_shell", lambda names: "pwsh.exe")
    argv = _command_argv("Write-Output '中文输出'", "powershell", windows=True)
    assert argv[:4] == ["pwsh.exe", "-NoLogo", "-NoProfile", "-NonInteractive"]
    assert argv[4:8] == ["-InputFormat", "Text", "-OutputFormat", "Text"]
    assert argv[8] == "-EncodedCommand"
    script = base64.b64decode(argv[-1]).decode("utf-16le")
    assert "[Console]::InputEncoding" in script
    assert "[Console]::OutputEncoding" in script
    assert "Write-Output '中文输出'" in script


def test_command_argv_keeps_powershell_script_single_argument(monkeypatch):
    monkeypatch.setattr("lithe.bundles.command._find_shell", lambda names: "powershell.exe")
    command = "Write-Output 'line one'\nWrite-Output 'line two'"
    argv = _command_argv(command, "powershell", windows=True)
    script = base64.b64decode(argv[-1]).decode("utf-16le")
    assert script.endswith(command)
