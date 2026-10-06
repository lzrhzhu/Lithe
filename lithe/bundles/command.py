from __future__ import annotations

import asyncio
import base64
import locale
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


_SHELLS = {"auto", "bash", "sh", "powershell", "cmd", "wsl"}
# Longest per-call timeout the model may request (seconds). The registry-level
# ToolSpec timeout sits above it so the runner's internal, output-preserving
# timeout always fires before asyncio's hard task cancellation.
_MAX_CALL_TIMEOUT = 1800.0
_PROXY_ENV = {"http_proxy", "https_proxy", "no_proxy"}
_UNIX_ENV = {
    "path", "home", "user", "logname", "shell", "term", "lang", "language",
    "lc_all", "lc_ctype", "tmpdir",
}
# Account identity vars (USERNAME / USERDOMAIN / COMPUTERNAME) are load-bearing
# for real tooling — icacls/ACL checks, git and pip user-path resolution all
# read them; stripping them broke pytest runs of this very project in the wild
# ("unable to determine current Windows account"). PSModulePath keeps
# PowerShell from re-scanning modules (the #< CLIXML "Preparing modules"
# stderr spam plus 1-3s startup per call). SYSTEMDRIVE/OS are read by many
# installers. None of these carry secrets.
_WINDOWS_ENV = {
    "path", "systemroot", "systemdrive", "windir", "os", "comspec", "pathext",
    "userprofile", "homedrive", "homepath", "temp", "tmp", "appdata",
    "localappdata", "programdata", "programfiles", "programfiles(x86)",
    "programw6432", "processor_architecture", "processor_identifier",
    "processor_level", "processor_revision", "number_of_processors", "public",
    "commonprogramfiles", "commonprogramfiles(x86)", "commonprogramw6432",
    "username", "userdomain", "computername", "psmodulepath",
}
# Opinionated runtime defaults for every child: children speaking UTF-8 is
# what makes the UTF-8 output decode reliable (on CP936 hosts a Python child
# pipes GBK otherwise). Explicit ``extra`` entries still win.
_RUNTIME_DEFAULTS = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def _command_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    allowed = _WINDOWS_ENV if os.name == "nt" else _UNIX_ENV
    env = {
        key: value for key, value in os.environ.items()
        if key.lower() in allowed or key.lower() in _PROXY_ENV
        or (os.name != "nt" and key.lower().startswith("lc_"))
    }
    env.update(_RUNTIME_DEFAULTS)
    env.update({str(key): str(value) for key, value in (extra or {}).items()})
    return env


def _find_shell(candidates: tuple[str, ...]) -> str | None:
    return next((path for name in candidates if (path := shutil.which(name))), None)


def _is_wsl_bash(path: str) -> bool:
    """True for WSL's bash stub under the Windows directory (System32/SysWOW64).

    That exe runs commands inside the Linux subsystem — a different OS with
    its own interpreters, PATH and filesystem view; a "bash" that silently
    crosses that boundary is the single largest failure source observed in
    real usage (python not found, venvs unusable on /mnt drvfs, ...).
    """
    low = str(path).replace("/", "\\").lower()
    windir = (os.environ.get("WINDIR") or os.environ.get("SystemRoot")
              or r"c:\windows").replace("/", "\\").lower().rstrip("\\")
    return low.startswith(windir + "\\") and (
        "\\system32\\" in low or low.endswith("\\system32")
        or "\\syswow64\\" in low or low.endswith("\\syswow64"))


def _windows_bash() -> str | None:
    """A Windows-side bash (Git for Windows), never WSL's stub.

    Git-Bash is the only "bash on Windows" that shares this OS's filesystem,
    PATH and interpreters, so it is the sole correct target when the model
    insists on POSIX syntax. Missing Git-Bash is *not* an error to paper
    over: the caller fails loudly with rewrite guidance instead (see
    ``_bash_unavailable_message``).
    """
    found = _find_shell(("bash",))
    if found and not _is_wsl_bash(found):
        return found
    for env_name in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        base = os.environ.get(env_name)
        if not base:
            continue
        cand = Path(base) / "Git" / "bin" / "bash.exe"
        if cand.is_file():
            return str(cand)
    return None


def _workspace_root(root: str | Path) -> Path:
    """Resolve the workspace anchor (sync helper: pathlib stays out of the
    async bodies, which ASYNC240 polices)."""
    return Path(root).resolve()


def _resolve_cwd(root: str | Path, raw: str | Path | None,
                 *, allow_external: bool = False) -> Path:
    """Resolve a run_command *cwd* against the workspace *root*.

    Relative paths anchor at *root* (never at the process cwd); absolute
    paths are taken as-is. Containment is checked post-``resolve()`` so
    ``..`` segments and in-root symlinks pointing outside cannot slip the
    anchor. Results outside the root raise :class:`PermissionError` — the
    tool layer routes that through the host's approval channel (direct
    ``CommandRunner`` callers just get the refusal unless they pass
    ``allow_external``); a missing/non-directory raises
    :class:`NotADirectoryError` so the call fails fast with a clear
    message instead of a subprocess spawn error.
    """
    text = "" if raw is None else str(raw).strip()
    if not text or text in (".", "./"):
        return _workspace_root(root)
    candidate = Path(text)
    anchor = _workspace_root(root)
    target = candidate if candidate.is_absolute() else anchor / candidate
    target = target.resolve()
    if target != anchor and anchor not in target.parents and not allow_external:
        raise PermissionError(f"cwd escapes workspace: {raw}")
    if not target.is_dir():
        raise NotADirectoryError(f"cwd 不存在或不是目录：{raw}")
    return target


def _bash_unavailable_message() -> str:
    has_wsl = _find_shell(("wsl",)) is not None
    msg = ("未找到 Windows 侧的 bash（Git for Windows）。"
           "已排除 WSL 的 bash.exe——它运行在 Linux 子系统中，"
           "与本机的文件、PATH 和解释器不互通。")
    if has_wsl:
        msg += ('如确需 POSIX 环境请显式 shell="wsl"；'
                "否则请把命令改写为 PowerShell 语法后重试。")
    else:
        msg += "请把命令改写为 PowerShell 语法后重试。"
    return msg


def _host_shell_note() -> str:
    """Host-platform fact sheet appended to the run_command description.

    Resolved once at registration from what is actually installed, so the
    model never guesses the platform or dialect: auto's real target, the
    PowerShell 5.1 caveats (no ``&&`` / ``VAR=x cmd`` / POSIX utils), the
    shells this machine really has. Windows Terminal is a console host, not
    a shell — irrelevant here; pwsh/powershell/cmd are the guaranteed floor.
    """
    if os.name != "nt":
        auto = "bash" if _find_shell(("bash",)) else "sh"
        return (f"当前宿主是 Linux/macOS（POSIX），auto 使用 {auto}。"
                "惯用法：依赖前一条成功的命令用 && 连接；"
                "读取文件内容优先用文件工具，避免 shell 编码差异。")
    has_pwsh = _find_shell(("pwsh",)) is not None
    has_ps = has_pwsh or _find_shell(("powershell",)) is not None
    git_bash = _windows_bash()
    has_wsl = _find_shell(("wsl",)) is not None
    avail = (["powershell", "cmd"] if has_ps else ["cmd"])
    if git_bash:
        avail.append("bash（Git-Bash）")
    if has_wsl:
        avail.append("wsl（Linux 子系统，路径与解释器和 Windows 侧不同）")
    note = "当前宿主是 Windows"
    if has_ps:
        note += "，auto 使用 powershell"
        if not has_pwsh:
            note += ("（Windows PowerShell 5.1：不支持 && 连接、VAR=x 前缀和 "
                     "tail/head/grep 等 POSIX 写法）")
    else:
        note += "，auto 使用 cmd"
    note += f"；本机可用 shell：{'、'.join(avail)}"
    if not git_bash and has_wsl:
        note += ('；本机没有 Windows 侧 bash，shell="bash" 会直接报错——'
                 "POSIX 命令请改写为 PowerShell，或显式 shell=\"wsl\" 在 Linux "
                 "子系统中执行")
    if has_ps:
        note += ("。PowerShell 惯用法：依赖前一条成功的命令写 "
                 "cmd1; if ($?) { cmd2 }（5.1 没有 &&）；"
                 "调用当前目录下的程序必须带 .\\ 前缀（如 .\\tool.exe），"
                 "路径带空格时用调用操作符 & \"...\"；"
                 "读取文件内容优先用文件工具（编码可靠），"
                 "确需 Get-Content 时加 -Encoding UTF8，避免中文乱码；"
                 "特殊字符用反引号转义，子表达式用 $(...)")
    return note + "。"


def _powershell_script(command: str) -> str:
    return (
        "[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "$OutputEncoding = [Console]::OutputEncoding; "
        # A non-interactive host with redirected stderr serializes progress
        # records as #< CLIXML noise into stderr ("Preparing modules for
        # first use", ...) — silence them instead of making the model parse
        # XML garbage.
        "$ProgressPreference = 'SilentlyContinue'; "
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
        # Windows: only a same-OS bash counts (Git-Bash). WSL's bash.exe is
        # excluded by _windows_bash; falling back to it silently sent commands
        # into another OS — the dominant failure class in real usage. No
        # Windows-side bash at all: fail loudly with actionable guidance
        # rather than silently switching dialects (a POSIX command that runs
        # "successfully" under PowerShell-ish semantics is worse than an
        # error the model can self-correct from).
        executable = _windows_bash() if is_windows else _find_shell(("bash",))
        if executable:
            return [executable, "-lc", command]
        if is_windows:
            raise FileNotFoundError(_bash_unavailable_message())
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
    elif selected == "wsl":
        # Explicit cross-boundary choice: the command runs inside the Linux
        # subsystem. wsl.exe translates the Windows cwd to /mnt/<drive>/...
        # for the child; killing the Windows relay does not guarantee the
        # Linux-side process dies, so long-running WSL work should mind the
        # timeout.
        if not is_windows:
            raise ValueError('shell="wsl" 仅在 Windows 宿主上可用')
        executable = _find_shell(("wsl",))
        if executable:
            return [executable, "-e", "bash", "-lc", command]
    raise FileNotFoundError(f"未找到可用的 {selected} 执行器")


def _fallback_encoding() -> str | None:
    """Locale encoding used when a child's bytes are not valid UTF-8.

    On CP936 hosts native tools still emit locale-encoded output despite the
    UTF-8 setup around them; decoding that as UTF-8 produced mojibake
    (“δ��װ Numba...”). POSIX hosts are UTF-8 by convention — None there.
    """
    if os.name != "nt":
        return None
    try:
        return locale.getpreferredencoding(False) or None
    except Exception:  # noqa: BLE001 — diagnostics must not break dispatch
        return None


def _decode_output(data: bytes, fallback: str | None) -> str:
    """Decode a child's output: UTF-8 first, locale fallback, then replace."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if fallback:
        try:
            return data.decode(fallback)
        except (UnicodeDecodeError, LookupError):
            pass
    return data.decode("utf-8", "replace")


async def _read_capped(stream, limit: int, fallback: str | None = None) -> str:
    """Read one output pipe to EOF, keeping head+tail when it overflows.

    Buffers bytes (bounded: first + tail windows) and decodes at the end so a
    non-UTF-8 child can fall back to the locale encoding — an incremental
    UTF-8 decoder cannot retry. For every encoding a byte window of N bytes
    decodes to ≤ N chars, so the char cap holds unchanged.
    """
    cap = max(1, limit)
    first_limit = max(1, cap // 2)
    tail_limit = max(1, cap - first_limit)
    complete = b""
    first = b""
    tail = b""
    total = 0
    while chunk := await stream.read(65536):
        total += len(chunk)
        if first:
            tail = (tail + chunk)[-tail_limit:]
        else:
            complete += chunk
            if len(complete) > cap:
                first = complete[:first_limit]
                tail = complete[-tail_limit:]
                complete = b""
    if not first:
        return _decode_output(complete, fallback)
    marker = f"\n…[输出过长，共 {total} 字节，已省略中间部分]…\n"
    return (_decode_output(first, fallback) + marker
            + _decode_output(tail, fallback))


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
        timeout: float = 300.0,
        max_output: int = 16000,
        extra_env: Mapping[str, str] | None = None,
    ):
        self.timeout = float(timeout)
        self.max_output = int(max_output)
        self.extra_env = dict(extra_env or {})

    def _call_timeout(self, timeout: float | int | None) -> float:
        """Resolve one call's timeout: per-call value clamped to a sane range,
        the runner default when the caller (tests, hosts) passes none."""
        if timeout is None:
            return self.timeout
        try:
            value = float(timeout)
        except (TypeError, ValueError):
            return self.timeout
        return min(max(value, 1.0), _MAX_CALL_TIMEOUT)

    async def run(self, root: str | Path, command: str, shell: str = "auto",
                  stdin: str | None = None, timeout: float | None = None,
                  cwd: str | Path | None = None, *,
                  allow_external_cwd: bool = False) -> dict:
        call_timeout = self._call_timeout(timeout)
        if not command.strip():
            return {"stdout": "", "stderr": "命令不能为空。", "exit_code": -1,
                    "duration": 0.0, "timed_out": False}
        try:
            argv = _command_argv(command, shell)
        except (FileNotFoundError, ValueError) as exc:
            return {"stdout": "", "stderr": str(exc), "exit_code": -1,
                    "duration": 0.0, "timed_out": False}
        root_path = _workspace_root(root)
        try:
            cwd_path = _resolve_cwd(root_path, cwd,
                                    allow_external=allow_external_cwd)
        except NotADirectoryError as exc:
            return {"stdout": "", "stderr": str(exc), "exit_code": -1,
                    "duration": 0.0, "timed_out": False}
        except PermissionError:
            return {"stdout": "",
                    "stderr": (f"cwd 指向工作区之外：{cwd}。工作区外的执行"
                               "位置需要用户审批；交互式前端的审批通道"
                               "通过后才会到达这里。"),
                    "exit_code": -1, "duration": 0.0, "timed_out": False}
        cwd = os.fspath(cwd_path)
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

        fallback = _fallback_encoding()
        stdout_task = asyncio.create_task(_read_capped(proc.stdout, self.max_output, fallback))
        stderr_task = asyncio.create_task(_read_capped(proc.stderr, self.max_output, fallback))
        input_task = asyncio.create_task(_write_stdin(proc.stdin, stdin))
        wait_task = asyncio.create_task(proc.wait())
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), call_timeout)
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
            stderr = (stderr + "\n" if stderr else "") + f"[执行超时（{call_timeout:g}s 已终止）]"
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
    approver: Callable[[str], Awaitable[bool]] | None = None,
) -> None:
    """Register ``run_command``.

    ``approver`` is the host's human-confirmation channel (the same one
    ``make_command_guard`` uses for destructive commands); here it gates
    only the one escalation ``cwd`` represents: a working directory
    outside the workspace. Without an approver, external cwd is refused
    with guidance — an unattended agent must not relocate execution
    silently.
    """
    command_runner = runner or CommandRunner()

    async def run_command(ctx: AgentContext, args: dict) -> ToolResult:
        command = args.get("command") or ""
        if not command.strip():
            return ToolResult(False, "缺少 command", "run_command 需要非空 command。")
        shell = args.get("shell") or "auto"
        root_path = _workspace_root(workspace_for(ctx))
        external_approved = False
        try:
            cwd_path = _resolve_cwd(root_path, args.get("cwd"))
        except NotADirectoryError as exc:
            return ToolResult(
                False, "cwd 无效",
                f"run_command 的 {exc}。cwd 必须是已存在的目录："
                f"相对工作区根的路径，或工作区内的绝对路径。")
        except PermissionError:
            preview = command if len(command) <= 100 else command[:97] + "..."
            if approver is None:
                return ToolResult(
                    False, "需用户审批",
                    f"cwd 指向工作区之外，需要用户确认后才允许在那里执行：\n"
                    f"cwd={args.get('cwd')}\n命令：{preview}\n"
                    f"请改用工作区内的目录（相对工作区根），"
                    f"或在交互式界面（TUI）中运行以获得审批确认。")
            try:
                allowed = await approver(f"{command}（cwd={args.get('cwd')}）")
            except Exception:  # noqa: BLE001 — a broken approver denies
                return ToolResult(False, "审批失败",
                                  "审批通道异常，为安全起见命令未执行。")
            if not allowed:
                return ToolResult(
                    False, "用户已拒绝",
                    f"用户拒绝在工作区之外执行（cwd={args.get('cwd')}）：\n{preview}\n"
                    f"请改用工作区内目录，不要原样重试。")
            cwd_path = _resolve_cwd(root_path, args.get("cwd"),
                                    allow_external=True)
            external_approved = True
        # cwd kwargs only when they carry information: custom runners with
        # the pre-cwd signature keep working for the default (root) case.
        run_kwargs: dict = {}
        if cwd_path != root_path:
            run_kwargs["cwd"] = cwd_path
            if external_approved:
                run_kwargs["allow_external_cwd"] = True
        result = await command_runner.run(
            workspace_for(ctx), command, shell, args.get("stdin"),
            timeout=args.get("timeout"), **run_kwargs)
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

    description = (
        "在主机上执行系统 shell 命令，默认在工作区根目录执行；"
        "需要在内层目录运行时（子项目、嵌套仓库、CI/构建脚本的相对路径）"
        "用 cwd 参数指定工作目录，不要在命令里拼接 cd。"
        "auto 在 Linux/macOS 使用 bash（缺少时用 sh），"
        "在 Windows 使用 PowerShell（缺少时用 cmd）；也可指定 bash（仅 Windows 侧的 Git-Bash / POSIX 主机）、"
        "sh、powershell、cmd，或 wsl（仅在 Windows 宿主可用：命令在 Linux 子系统中执行，"
        "路径与解释器和 Windows 侧不同）。命令拥有当前用户的主机权限，可访问工作区以外的文件和网络，"
        "副作用不可撤销；仅在用户明确启用此能力后调用。"
        "危险命令有防线：毁灭性操作（如 rm -rf /、mkfs、format 盘符）会被直接拒绝；"
        "破坏性但可控的操作（如 rm -r 指定目录、git push --force、sudo）需要用户审批，"
        "被拒后不要原样重试；工作区之外的 cwd 同样需要用户审批。\n"
        + _host_shell_note()
    )

    registry.register(
        ToolSpec(
            "run_command",
            description,
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的系统命令"},
                    "cwd": {
                        "type": "string",
                        "description": (
                            "可选工作目录：相对工作区根的路径，或工作区内的"
                            "绝对路径，必须是已存在的目录。项目里的命令"
                            "（pytest、构建脚本、CI 步骤）假设自己在项目根"
                            "执行，对子项目/嵌套仓库运行时用它指定目录。"
                            "省略时在工作区根执行；工作区之外的 cwd 需要"
                            "用户审批"),
                    },
                    "shell": {
                        "type": "string",
                        "enum": ["auto", "bash", "sh", "powershell", "cmd", "wsl"],
                        "description": "shell 类型，省略时按当前操作系统自动选择",
                    },
                    "stdin": {"type": "string", "description": "可选标准输入"},
                    "timeout": {
                        "type": "number",
                        "description": (
                            "可选执行超时（秒），范围 1-1800，默认 300。"
                            "pip install、构建、测试等长命令请显式给较大值，"
                            "超时后命令被终止并返回已产生的输出"),
                    },
                },
                "required": ["command"],
            },
            ToolCategory.WRITE,
            # Above the per-call maximum so the runner's internal timeout
            # (which preserves partial output) always fires first.
            timeout=_MAX_CALL_TIMEOUT + 15,
        ),
        run_command,
    )
