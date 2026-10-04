from __future__ import annotations

import asyncio
import base64
import codecs
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from collections.abc import Awaitable, Callable, Mapping

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


def _powershell_script(command: str) -> str:
    return (
        "[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "$OutputEncoding = [Console]::OutputEncoding; "
        + command
    )


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
            script = _powershell_script(command)
            encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
            return [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-InputFormat",
                "Text",
                "-OutputFormat",
                "Text",
                "-EncodedCommand",
                encoded,
            ]
    elif selected == "cmd":
        executable = os.environ.get("COMSPEC") if is_windows else None
        executable = executable or _find_shell(("cmd.exe", "cmd"))
        if executable:
            if is_windows:
                command = "chcp 65001 >NUL & " + command
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


# --- dangerous-command policy ------------------------------------------------
#
# run_command runs with the host user's permissions, unsandboxed and
# irreversible. The guard classifies a command BEFORE execution:
#
# - ``"deny"``    — catastrophic (disk nukes, raw-device writes, shutdown,
#                   fork bombs, recursive deletion aimed at a filesystem
#                   anchor). Refused outright, no approver can override.
# - ``"approve"`` — plausibly legitimate but destructive (``rm -r`` on a
#                   scoped path, ``git push --force``, ``sudo``, broad
#                   kills...). Runs only when the host-supplied approver
#                   (a human) says yes.
# - ``None``      — ordinary command, runs as before.
#
# Shell is Turing-complete, so classification is a heuristic — it errs on
# the side of *asking*: anything that looks like a recursive deleter or a
# privilege escalator lands in "approve" even when the target analysis
# can't prove it dangerous. Only unambiguous catastrophes are denied.


def _anchor(target: str) -> bool:
    """True for filesystem anchors whose recursive deletion is catastrophic."""
    t = target.strip().strip("'\"").rstrip("/\\").lower()
    t = t[:-2] if t.endswith("/*") or t.endswith("\\*") else t
    return t in {
        "",  # bare "/" or "c:\" after stripping separators
        ".", "..", "*", "$home", "~", "~/",
        "/etc", "/usr", "/var", "/boot", "/bin", "/sbin", "/lib", "/lib64",
        "/opt", "/home", "/root", "/users", "/srv", "/windows", "/system32",
        "/program files", "/programdata", "/program files (x86)",
        "c:", "d:", "e:", "c:/users", "c:/windows",
    }


# (command word, recursion flag test) for the recursive-deleter family.
# Windows PowerShell aliases `rm`/`ri` to Remove-Item; cmd's `rd`/`del` take
# /s instead of -r — one table covers every dialect.
_DELETERS: dict[str, Callable[[str], bool]] = {
    "rm": lambda f: "recursive" in f or ("-" in f and not f.startswith("--")
                                         and "r" in f),
    "remove-item": lambda f: "recurse" in f or f in ("-r", "-r:"),
    "ri": lambda f: "recurse" in f or f in ("-r", "-r:"),
    "rd": lambda f: f in ("/s", "/s:"),
    "rmdir": lambda f: f in ("/s", "/s:"),
    "del": lambda f: f in ("/s", "/s:"),
    "erase": lambda f: f in ("/s", "/s:"),
    "chmod": lambda f: "recursive" in f or ("-" in f and not f.startswith("--")
                                            and "r" in f),
    "chown": lambda f: "recursive" in f or ("-" in f and not f.startswith("--")
                                            and "r" in f),
}

# command-position prefix: start of string or after a separator
_CMDPOS = r"(?:^|[;&|(]\s*)"

_DENY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(p) for p in (
    rf"{_CMDPOS}(?:sudo\s+|doas\s+)?mkfs(?:\.\w+)?\b",
    rf"{_CMDPOS}(?:sudo\s+|doas\s+)?mke2fs\b",
    rf"{_CMDPOS}(?:sudo\s+|doas\s+)?dd\b[^|]*\bof=/dev/",
    rf"{_CMDPOS}(?:sudo\s+|doas\s+)?(?:shutdown|reboot|halt|poweroff)\b",
    rf"{_CMDPOS}(?:sudo\s+|doas\s+)?(?:halt|poweroff)\b",
    r":\(\)\s*\{.*\}.*;",          # bash fork bomb, whitespace-insensitive
    r"%0\s*\|\s*%0",               # cmd fork bomb
    r">\s*/dev/(?:sd|hd|nvme|disk)",
    rf"{_CMDPOS}format\s+[a-z]:",
    rf"{_CMDPOS}diskpart\b",
    rf"{_CMDPOS}cipher\s+/w",
    rf"{_CMDPOS}reg\s+delete\s+(?:hklm|hkey_local_machine)\b[^|]*\s/f\b",
))

_APPROVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(p) for p in (
    rf"{_CMDPOS}(?:sudo|doas)\b",
    rf"{_CMDPOS}git\s+push\b[^|]*?(\s-f\b|\s--force(?!-with-lease)\b)",
    rf"{_CMDPOS}git\s+reset\s+--hard\b",
    rf"{_CMDPOS}git\s+clean\s+-\w*f\w*\b",
    rf"{_CMDPOS}git\s+(?:checkout|restore)\s+(?:--\s*)?\.",
    rf"{_CMDPOS}kill\s+-9\b",
    rf"{_CMDPOS}(?:killall|pkill)\b",
    rf"{_CMDPOS}taskkill\b[^|]*\s/f\b",
    rf"{_CMDPOS}shred\b",
    rf"{_CMDPOS}crontab\s+-r\b",
    r"\b(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b",
    rf"{_CMDPOS}truncate\s+-s\s*0\b[^|]*\s/",
))


def _deleter_verdict(norm: str) -> str | None:
    """Verdict for the recursive-deleter family (rm / rd / Remove-Item /
    chmod -R ...): deny at a filesystem anchor, approve scoped, None when
    not recursive."""
    for m in re.finditer(rf"{_CMDPOS}(?:sudo\s+|doas\s+)?(\w[\w.-]*)\b", norm):
        word = m.group(1)
        flag_test = _DELETERS.get(word)
        if flag_test is None:
            continue
        tail = norm[m.end():].split()
        # cmd dialects spell flags /s; POSIX/PowerShell spell them -r/--recursive.
        # Only the matching syntax counts, so `rm -rf /usr` keeps "/usr" a
        # TARGET (bash), while `rd /s /q C:\` keeps its slash flags (cmd).
        slash_flags = word in ("rd", "rmdir", "del", "erase")
        flags: list[str] = []
        targets: list[str] = []
        for tok in tail:
            if tok == "--":
                targets.extend(tail[tail.index("--") + 1:])
                break
            if tok.startswith("-") or (slash_flags and tok.startswith("/")):
                flags.append(tok)
            else:
                targets.append(tok)
        if not any(flag_test(f) for f in flags):
            continue
        if any(_anchor(t) for t in targets):
            return "deny"
        return "approve"
    return None


def classify_command(command: str) -> str | None:
    """Classify *command* as ``"deny"`` / ``"approve"`` / ``None`` (safe).

    Pure and dialect-agnostic: POSIX shells, PowerShell and cmd patterns are
    all matched against the normalized text. Heuristic by necessity — the
    guard errs toward asking.
    """
    norm = " ".join((command or "").split()).lower()
    if not norm:
        return None
    verdict = _deleter_verdict(norm)
    if verdict is not None:
        return verdict
    for rx in _DENY_PATTERNS:
        if rx.search(norm):
            return "deny"
    for rx in _APPROVE_PATTERNS:
        if rx.search(norm):
            return "approve"
    return None


def make_command_guard(
    approve: Callable[[str], Awaitable[bool]] | None = None,
    *,
    tool_name: str = "run_command",
) -> Callable[[AgentContext, str, dict], Awaitable[ToolResult | None]]:
    """Build a pre-dispatch middleware guarding ``run_command``.

    ``approve`` is the host's human-confirmation channel: ``async
    (command) -> bool``. Without one, "approve"-class commands are denied
    with guidance to hand them to the user — an unattended agent must not
    run destructive operations silently.
    """

    async def guard(ctx: AgentContext, name: str, args: dict) -> ToolResult | None:
        if name != tool_name:
            return None
        command = (args.get("command") or "").strip()
        if not command:
            return None
        verdict = classify_command(command)
        if verdict is None:
            return None
        preview = command if len(command) <= 120 else command[:117] + "..."
        if verdict == "deny":
            return ToolResult(
                False, "危险命令已拦截",
                f"命令属于不可逆的毁灭性操作，已被安全策略直接拒绝，"
                f"不会执行：\n{preview}\n"
                f"请改用安全方式完成目标（如限定目录的删除、版本控制回退），"
                f"或把该命令原样告诉用户由其手动执行。")
        if approve is None:
            return ToolResult(
                False, "需用户审批",
                f"命令具有破坏性，需用户确认后才允许执行：\n{preview}\n"
                f"当前没有交互审批通道：请把命令原样告诉用户、由用户自行"
                f"执行；或在交互式界面（TUI）中运行以获得审批确认。")
        try:
            allowed = await approve(command)
        except Exception:  # noqa: BLE001 — a broken approver denies, never runs
            return ToolResult(False, "审批失败",
                              "审批通道异常，为安全起见命令未执行。")
        if allowed:
            return None
        return ToolResult(
            False, "用户已拒绝",
            f"用户拒绝执行该命令：\n{preview}\n"
            f"请询问用户希望如何调整，不要原样重试。")

    return guard


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
            "在主机上执行工作区目录下的系统 shell 命令。auto 在 Linux/macOS 使用 bash（缺少时用 sh），在 Windows 使用 PowerShell（缺少时用 cmd）；也可指定 bash、sh、powershell 或 cmd。命令拥有当前用户的主机权限，可访问工作区以外的文件和网络，副作用不可撤销；仅在用户明确启用此能力后调用。"
            "危险命令有防线：毁灭性操作（如 rm -rf /、mkfs、format 盘符）会被直接拒绝；"
            "破坏性但可控的操作（如 rm -r 指定目录、git push --force、sudo）需要用户审批，"
            "被拒后不要原样重试。",
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
