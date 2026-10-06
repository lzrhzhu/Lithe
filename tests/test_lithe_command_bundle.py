from __future__ import annotations

import base64
import os
import sys

import pytest

from lithe import AgentContext, ToolCategory, ToolRegistry
from lithe.bundles.command import (
    CommandRunner,
    _command_argv,
    _command_env,
    _decode_output,
    _powershell_script,
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


def test_powershell_script_suppresses_progress_records():
    script = _powershell_script("Write-Output ok")
    assert "$ProgressPreference = 'SilentlyContinue';" in script
    assert script.endswith("Write-Output ok")


def test_command_env_injects_utf8_runtime_defaults(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    env = _command_env()
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    # explicit extra wins over the defaults
    env = _command_env({"PYTHONUTF8": "0"})
    assert env["PYTHONUTF8"] == "0"


def test_command_env_forwards_windows_account_vars(monkeypatch):
    monkeypatch.setattr("lithe.bundles.command.os", type(
        "OS", (), {"name": "nt",
                   "environ": {"USERNAME": "u", "USERDOMAIN": "D",
                               "COMPUTERNAME": "C", "PSMODULEPATH": "m",
                               "SYSTEMDRIVE": "C:", "OS": "Windows_NT",
                               "MY_TOKEN": "secret"}}))
    env = _command_env()
    for key in ("USERNAME", "USERDOMAIN", "COMPUTERNAME", "PSMODULEPATH",
                "SYSTEMDRIVE", "OS"):
        assert key in env, f"{key} stripped — breaks ACL/account lookups"
    assert "MY_TOKEN" not in env


def test_bash_on_windows_rejects_wsl_stub_loudly(monkeypatch, tmp_path):
    """shell="bash" on Windows must never silently run inside WSL — the
    dominant real-world failure (different OS, interpreters, PATH). With
    only the WSL stub present the call fails with rewrite guidance."""
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        monkeypatch.delenv(var, raising=False)

    def fake(names):
        if "bash" in names:
            return r"C:\Windows\System32\bash.exe"  # the WSL stub
        if "wsl" in names:
            return r"C:\Windows\System32\wsl.exe"
        if names and names[0] in ("pwsh", "powershell"):
            return "powershell.exe"
        return None

    monkeypatch.setattr("lithe.bundles.command._find_shell", fake)
    with pytest.raises(FileNotFoundError) as exc:
        _command_argv("ls -la", "bash", windows=True)
    assert "Git for Windows" in str(exc.value)
    assert "wsl" in str(exc.value)
    assert "PowerShell" in str(exc.value)


def test_bash_on_windows_prefers_git_bash_from_programfiles(monkeypatch, tmp_path):
    git_bash = tmp_path / "Git" / "bin" / "bash.exe"
    git_bash.parent.mkdir(parents=True)
    git_bash.write_bytes(b"")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setattr("lithe.bundles.command._find_shell",
                        lambda names: r"C:\Windows\System32\bash.exe")
    argv = _command_argv("ls -la", "bash", windows=True)
    assert argv == [str(git_bash), "-lc", "ls -la"]


def test_wsl_is_an_explicit_windows_only_shell(monkeypatch):
    monkeypatch.setattr("lithe.bundles.command._find_shell",
                        lambda names: "wsl.exe" if "wsl" in names else None)
    argv = _command_argv("grep -r foo .", "wsl", windows=True)
    assert argv == ["wsl.exe", "-e", "bash", "-lc", "grep -r foo ."]
    with pytest.raises(ValueError):
        _command_argv("ls", "wsl", windows=False)


def test_run_command_description_carries_host_shell_note(tmp_path):
    registry = ToolRegistry()
    register_command_tools(registry, lambda ctx: tmp_path, _runner())
    spec = registry.spec("run_command")
    assert "当前宿主" in spec.description
    assert ("Windows" in spec.description) == (os.name == "nt")
    shell_enum = spec.parameters["properties"]["shell"]["enum"]
    assert "wsl" in shell_enum
    assert "timeout" in spec.parameters["properties"]
    # registry timeout sits above the per-call maximum so the runner's
    # output-preserving timeout always fires first
    assert spec.timeout > 1800


def test_call_timeout_resolution():
    runner = CommandRunner(timeout=10)
    assert runner._call_timeout(None) == 10.0
    assert runner._call_timeout(0.2) == 1.0          # clamped up
    assert runner._call_timeout(99999) == 1800.0     # clamped down
    assert runner._call_timeout("junk") == 10.0      # junk → runner default


async def test_run_command_per_call_timeout(tmp_path):
    runner = CommandRunner(timeout=30)
    result = await runner.run(tmp_path, "sleep 30", timeout=1)
    assert result["timed_out"]
    assert result["exit_code"] == -1
    assert "超时" in result["stderr"]


def test_decode_output_prefers_utf8_then_locale_fallback():
    assert _decode_output("中文".encode(), "gbk") == "中文"
    assert _decode_output("未安装 numba".encode("gbk"), "gbk") == "未安装 numba"
    # no fallback: undecodable bytes degrade to replacements, never raise
    out = _decode_output(b"\xd5\xe2\xca\xc7", None)
    assert "\ufffd" in out
