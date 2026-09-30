from __future__ import annotations

import asyncio
import codecs
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from collections.abc import Callable, Mapping

from lithe.context import AgentContext
from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec


_SHELLS = {"auto", "bash", "sh", "powershell", "cmd"}
_PROXY_ENV = {"http_proxy", "https_proxy", "no_proxy"}
_UNIX_ENV = {
    "path", "home", "user", "logname", "shell", "term", "lang", "language",
    "lc_all", "lc_ctype", "tmpdir",
}
_WINDOWS_ENV = {
    "path", "systemroot", "windir", "comspec", "pathext", "userprofile",
    "homedrive", "homepath", "temp", "tmp", "appdata", "localappdata",
    "programdata", "programfiles", "programfiles(x86)", "processor_architecture",
    "number_of_processors", "public", "commonprogramfiles",
}


def _command_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    allowed = _WINDOWS_ENV if os.name == "nt" else _UNIX_ENV
    env = {
        key: value for key, value in os.environ.items()
        if key.lower() in allowed or key.lower() in _PROXY_ENV
        or (os.name != "nt" and key.lower().startswith("lc_"))
    }
    env.update({str(key): str(value) for key, value in (extra or {}).items()})
    return env


def _find_shell(candidates: tuple[str, ...]) -> str | None:
    return next((path for name in candidates if (path := shutil.which(name))), None)


def _command_argv(command: str, shell: str = "auto", *, windows: bool | None = None) -> list[str]:
    is_windows = os.name == "nt" if windows is None else windows
    selected = (shell or "auto").lower()
    if selected not in _SHELLS:
        raise ValueError(f"不支持的 shell：{shell}")
    if selected == "auto":
        if is_windows:
            shell_path = _find_shell(("pwsh", "powershell"))
            if shell_path:
                selected = "powershell"
            else:
                selected = "cmd"
        else:
            selected = "bash" if _find_shell(("bash",)) else "sh"
    if selected == "bash":
        executable = _find_shell(("bash",))
        if executable:
            return [executable, "-lc", command]
    elif selected == "sh":
        executable = _find_shell(("sh",))
        if executable:
            return [executable, "-c", command]
    elif selected == "powershell":
        executable = _find_shell(("pwsh", "powershell", "powershell.exe"))
        if executable:
            return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
    elif selected == "cmd":
        executable = os.environ.get("COMSPEC") if is_windows else None
        executable = executable or _find_shell(("cmd.exe", "cmd"))
        if executable:
            return [executable, "/d", "/s", "/c", command]
    raise FileNotFoundError(f"未找到可用的 {selected} 执行器")


async def _read_capped(stream, limit: int) -> str:
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    cap = max(1, limit)
    first_limit = max(1, cap // 2)
    tail_limit = max(1, cap - first_limit)
    complete = ""
    first = ""
    tail = ""
    total = 0
    while chunk := await stream.read(4096):
        text = decoder.decode(chunk)
        total += len(text)
        if first:
            tail = (tail + text)[-tail_limit:]
        else:
            complete += text
            if len(complete) > cap:
                first = complete[:first_limit]
                tail = complete[-tail_limit:]
                complete = ""
    final = decoder.decode(b"", final=True)
    total += len(final)
    if final:
        if first:
            tail = (tail + final)[-tail_limit:]
        else:
            complete += final
            if len(complete) > cap:
                first = complete[:first_limit]
                tail = complete[-tail_limit:]
    if not first:
        return complete
    marker = f"\n…[输出过长，共 {total} 字符，已省略中间部分]…\n"
    return first + marker + tail


async def _write_stdin(stream, value: str | None) -> None:
    try:
        if value is not None:
            stream.write(value.encode("utf-8"))
            await stream.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        stream.close()


async def _stop_process(proc: asyncio.subprocess.Process) -> None:
    if os.name == "nt":
        taskkill = shutil.which("taskkill")
        if taskkill:
            try:
                killer = await asyncio.create_subprocess_exec(
                    taskkill, "/PID", str(proc.pid), "/T", "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(killer.wait(), timeout=3)
            except (OSError, asyncio.TimeoutError):
                pass
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


class CommandRunner:
    def __init__(
        self,
        *,
        timeout: float = 120.0,
        max_output: int = 16000,
        extra_env: Mapping[str, str] | None = None,
    ):
        self.timeout = float(timeout)
        self.max_output = int(max_output)
        self.extra_env = dict(extra_env or {})

    async def run(self, root: str | Path, command: str, shell: str = "auto", stdin: str | None = None) -> dict:
        if not command.strip():
            return {"stdout": "", "stderr": "命令不能为空。", "exit_code": -1,
                    "duration": 0.0, "timed_out": False}
        try:
            argv = _command_argv(command, shell)
        except (FileNotFoundError, ValueError) as exc:
            return {"stdout": "", "stderr": str(exc), "exit_code": -1,
                    "duration": 0.0, "timed_out": False}
        cwd = os.fspath(root)
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=_command_env(self.extra_env),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **kwargs,
            )
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
            return {"stdout": "", "stderr": f"命令启动失败：{exc}", "exit_code": -1,
                    "duration": round(time.monotonic() - started, 3), "timed_out": False}

        stdout_task = asyncio.create_task(_read_capped(proc.stdout, self.max_output))
        stderr_task = asyncio.create_task(_read_capped(proc.stderr, self.max_output))
        input_task = asyncio.create_task(_write_stdin(proc.stdin, stdin))
        wait_task = asyncio.create_task(proc.wait())
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), self.timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await _stop_process(proc)
            try:
                await asyncio.wait_for(asyncio.shield(wait_task), timeout=3)
            except asyncio.TimeoutError:
                pass
        except asyncio.CancelledError:
            await _stop_process(proc)
            input_task.cancel()
            stdout_task.cancel()
            stderr_task.cancel()
            raise
        if not input_task.done():
            input_task.cancel()
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task), timeout=2
            )
        except asyncio.TimeoutError:
            await _stop_process(proc)
            stdout_task.cancel()
            stderr_task.cancel()
            stdout, stderr = "", ""
        if timed_out:
            stderr = (stderr + "\n" if stderr else "") + f"[执行超时（{self.timeout:g}s 已终止）]"
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": -1 if timed_out else proc.returncode,
            "duration": round(time.monotonic() - started, 3),
            "timed_out": timed_out,
        }


def register_command_tools(
    registry: ToolRegistry,
    workspace_for: Callable[[AgentContext], str | Path],
    runner: CommandRunner | None = None,
) -> None:
    command_runner = runner or CommandRunner()

    async def run_command(ctx: AgentContext, args: dict) -> ToolResult:
        command = args.get("command") or ""
        if not command.strip():
            return ToolResult(False, "缺少 command", "run_command 需要非空 command。")
        shell = args.get("shell") or "auto"
        result = await command_runner.run(workspace_for(ctx), command, shell, args.get("stdin"))
        parts = []
        if result["stdout"].strip():
            parts.append(result["stdout"].rstrip())
        if result["stderr"].strip():
            parts.append(f"[stderr]\n{result['stderr'].rstrip()}")
        if result["timed_out"]:
            parts.append("（命令超时，已终止）")
        elif result["exit_code"] != 0:
            parts.append(f"（退出码：{result['exit_code']}）")
        content = "\n".join(parts) or "（无输出）"
        return ToolResult(
            result["exit_code"] == 0,
            f"命令完成，退出码 {result['exit_code']}，耗时 {result['duration']:.1f}s",
            content,
        )

    registry.register(
        ToolSpec(
            "run_command",
            "在主机上执行工作区目录下的系统 shell 命令。auto 在 Linux/macOS 使用 bash（缺少时用 sh），在 Windows 使用 PowerShell（缺少时用 cmd）；也可指定 bash、sh、powershell 或 cmd。命令拥有当前用户的主机权限，可访问工作区以外的文件和网络，副作用不可撤销；仅在用户明确启用此能力后调用。",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的系统命令"},
                    "shell": {
                        "type": "string",
                        "enum": ["auto", "bash", "sh", "powershell", "cmd"],
                        "description": "shell 类型，省略时按当前操作系统自动选择",
                    },
                    "stdin": {"type": "string", "description": "可选标准输入"},
                },
                "required": ["command"],
            },
            ToolCategory.WRITE,
            timeout=command_runner.timeout + 10,
        ),
        run_command,
    )
