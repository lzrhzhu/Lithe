"""Sandbox bundle: run Python in an isolated sandbox with a secret-free
environment, timeout and output cap. Provides run_code / run_file tools.

Optional bundle. The backend is pluggable:

- ``bwrap`` (default, Linux): a bubblewrap mount namespace — filesystem and
  network isolated, sibling users' files / /etc / /root / .env secrets invisible,
  and ``--clearenv`` so AUTH_SECRET / *_API_KEY never leak via os.environ.
- ``none``: runs the interpreter directly with only the secret-free env (NO
  filesystem/network isolation) — for local dev / tests; never expose untrusted
  code this way.

All config (interpreter, venv, extra binds, timeout) is host-supplied; the
bundle knows nothing of app.config.
"""
from __future__ import annotations

import asyncio
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from collections.abc import Callable

from lithe.context import AgentContext
from lithe.memory import truncate_tool_result
from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec

# Parent-process vars safe to forward: pure runtime config, never secrets.
_SAFE_ENV_NAMES = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM")

# Windows runtime vars the child cannot start or work without: SYSTEMROOT is
# required to initialize hash randomization (CryptoAPI lives under it) — a
# Python started without it dies with "_Py_HashRandomization_Init" before
# running any code. TEMP/TMP back tempfile, COMSPEC/PATHEXT back subprocess
# spawning. None of these carry secrets.
_WIN_SAFE_ENV_NAMES = (
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "USERPROFILE",
    "HOMEDRIVE", "HOMEPATH", "TEMP", "TMP", "OS", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)


def _sandbox_env(*, windows: bool | None = None) -> dict[str, str]:
    """Secret-free env for the child: a fixed allow-list forwarded from the
    parent plus a few hard-coded runtime vars. Never copies os.environ wholesale,
    so secrets (AUTH_SECRET / *_API_KEY / ...) can never reach executed code."""
    is_windows = os.name == "nt" if windows is None else windows
    env = {"HOME": "/tmp", "MPLBACKEND": "Agg", "XDG_CACHE_HOME": "/tmp/.cache",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8"}

    names = _SAFE_ENV_NAMES + (_WIN_SAFE_ENV_NAMES if is_windows else ())
    for name in names:
        val = os.environ.get(name)
        if val:
            env[name] = val
    return env


def _err_result(msg: str) -> dict:
    return {"stdout": "", "stderr": msg, "exit_code": -1,
            "duration": 0.0, "timed_out": False}


async def _drain(proc) -> tuple[bytes, bytes]:
    out = err = b""
    try:
        if proc.stdout:
            out = await proc.stdout.read()
        if proc.stderr:
            err = await proc.stderr.read()
    except Exception:
        pass
    return out, err


async def _kill_tree(proc) -> None:
    """Kill the child and its subprocess tree. POSIX: the child started a new
    session, so killpg takes spawned grandchildren too. Windows: no process
    groups (and no os.killpg) — taskkill /T /F walks the tree instead."""
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
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


class CodeRunner:
    """Runs Python in a sandbox (see module docstring for the two backends)."""

    def __init__(self, interpreter, *, venv=None, extra_ro_binds=(),
                 timeout: float = 15.0, max_output: int = 8000,
                 backend: str = "bwrap", mount_proc: bool = True):
        self.interpreter = str(interpreter)
        self.venv = str(venv) if venv else None
        self.extra_ro_binds = [str(b) for b in extra_ro_binds]
        self.timeout = float(timeout)
        self.max_output = int(max_output)
        self.backend = backend
        # Docker containers commonly forbid mounting a fresh procfs even with
        # seccomp/apparmor unconfined (mount("proc") → EPERM), while pid/mount
        # namespaces work fine. mount_proc=False skips --proc: ordinary
        # compute code doesn't need /proc (no multiprocessing/psutil).
        self.mount_proc = bool(mount_proc)

    def ready(self) -> bool:
        if not Path(self.interpreter).is_file():
            return False
        if self.backend == "bwrap":
            return bool(shutil.which("bwrap"))
        return True

    async def run_code(self, root, code: str, stdin: str | None = None) -> dict:
        return await self._execute([self.interpreter, "-u", "-c", code], root, stdin)

    async def run_file(self, root, rel: str, stdin: str | None = None) -> dict:
        root = Path(root)
        target = (root / rel).resolve()
        root_resolved = root.resolve()
        if target != root_resolved and root_resolved not in target.parents:
            return _err_result("路径越界，拒绝执行。")
        if not target.is_file():
            return _err_result(f"文件不存在: {rel}")
        return await self._execute([self.interpreter, "-u", rel], root, stdin)

    def _full_argv(self, tail, root) -> tuple[list[str], str | None]:
        """Return ``(argv, cwd)``. bwrap wraps *tail* (cwd=None); none returns
        *tail* unchanged with cwd=root."""
        if self.backend != "bwrap":
            return tail, str(root)
        env = _sandbox_env()
        argv = [
            shutil.which("bwrap") or "bwrap", "--unshare-all", "--die-with-parent",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/sbin", "/sbin",
            "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--ro-bind", "/etc/fonts", "/etc/fonts",
        ]
        if self.venv:
            argv += ["--ro-bind", self.venv, self.venv]
        for src in self.extra_ro_binds:
            argv += ["--ro-bind", src, src]
        argv += ["--bind", str(Path(root).resolve()), "/workspace",
                 "--dev", "/dev", "--tmpfs", "/tmp"]
        if self.mount_proc:
            argv += ["--proc", "/proc"]
        argv += ["--chdir", "/workspace", "--clearenv"]
        for k, v in env.items():
            argv += ["--setenv", k, v]
        return argv + tail, None

    async def _execute(self, tail, root, stdin: str | None) -> dict:
        if not self.ready():
            return _err_result("代码沙箱不可用（缺少解释器或 bwrap），已禁止执行。")
        env = _sandbox_env()
        full, cwd = self._full_argv(tail, root)
        started = time.time()
        process_options = (
            {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *full,
                cwd=cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                **process_options,
            )

        except FileNotFoundError:
            return _err_result(f"执行器未找到: {full[0]}")

        stdin_bytes = (stdin or "").encode() if stdin else None
        timed_out = False
        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes), timeout=self.timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await _kill_tree(proc)
            await proc.wait()
            # A grandchild that escaped the group kill can still hold the
            # pipe write end; bound the drain so the timeout actually ends
            # the tool instead of blocking on that grandchild's lifetime.
            try:
                out_b, err_b = await asyncio.wait_for(_drain(proc), timeout=5.0)
            except asyncio.TimeoutError:
                out_b, err_b = b"", b""

        # Head+tail truncation, not head-only: Python prints tracebacks at the
        # END of stderr and results at the END of stdout — cutting the tail
        # would hide exactly the part the model needs to self-correct.
        out_text = out_b.decode("utf-8", "replace").replace("\r\n", "\n")
        err_text = err_b.decode("utf-8", "replace").replace("\r\n", "\n")
        out = truncate_tool_result(out_text, self.max_output)
        err = truncate_tool_result(err_text, self.max_output)
        if timed_out:
            err += f"\n[执行超时（{self.timeout:.0f}s 已终止）]"
        return {"stdout": out, "stderr": err,
                "exit_code": proc.returncode if not timed_out else -1,
                "duration": round(time.time() - started, 3), "timed_out": timed_out}


def register_code_tools(
    registry: ToolRegistry,
    workspace_for: Callable[[AgentContext], str],
    runner: CodeRunner,
) -> None:
    """Register ``run_code`` (inline) and ``run_file`` (saved file) tools bound
    to a host-supplied ``workspace_for(ctx) -> root path``."""

    def _format(res: dict) -> ToolResult:
        ok = res["exit_code"] == 0
        if ok and not res["stderr"].strip():
            return ToolResult(True, "运行成功",
                              f"运行结果：\n{res['stdout']}".rstrip() or "（无输出）")
        parts = []
        if res["stdout"].strip():
            parts.append(res["stdout"])
        if res["stderr"].strip():
            parts.append(f"[stderr]\n{res['stderr']}")
        if res["timed_out"]:
            parts.append("（执行超时已终止）")
        return ToolResult(ok, "运行完成" if ok else "运行出错",
                          "\n".join(parts) or "（无输出）")

    async def run_code(ctx, args):
        code = args.get("code") or ""
        if not code.strip():
            return ToolResult(False, "缺少 code", "run_code 需要 code。")
        return _format(await runner.run_code(workspace_for(ctx), code, args.get("stdin")))

    async def run_file(ctx, args):
        rel = (args.get("path") or "").strip()
        if not rel:
            return ToolResult(False, "缺少 path", "run_file 需要 path。")
        return _format(await runner.run_file(workspace_for(ctx), rel, args.get("stdin")))

    # Executing model-written code can mutate the workspace (run_code may
    # create/overwrite files), so both tools are WRITE-classified: they are
    # excluded from read-only agent modes and never run in parallel with
    # other tool calls — two run_code invocations racing on the same output
    # file is exactly the write race the runtime's grouping prevents.
    registry.register(
        ToolSpec("run_code", "运行内联 Python 代码（沙箱内，禁网，超时终止；"
                             "可能在沙箱内写入文件，故按写工具对待）。"
                             "注意：代码造成的文件改动不会产生可撤销记录，"
                             "undo 无法回滚 run_code 的副作用；重要内容请先用 "
                             "write_file/edit_file 保存，或让代码把产物写到新文件。",
                 {"type": "object",
                  "properties": {"code": {"type": "string", "description": "Python 代码"},
                                 "stdin": {"type": "string", "description": "标准输入（可选）"}},
                  "required": ["code"]}, ToolCategory.WRITE,
                 # Kernel-level backstop past the runner's own timeout, so a
                 # drain edge case can never wedge the whole step.
                 timeout=runner.timeout + 10),
        run_code)
    registry.register(
        ToolSpec("run_file", "运行工作区内已保存的 Python 文件"
                             "（同样按写工具对待；其文件改动同样不可撤销）。",
                 {"type": "object",
                  "properties": {"path": {"type": "string", "description": "工作区内 .py 相对路径"},
                                 "stdin": {"type": "string", "description": "标准输入（可选）"}},
                  "required": ["path"]}, ToolCategory.WRITE,
                 timeout=runner.timeout + 10),
        run_file)
