"""lithe.bundles.sandbox: run Python in a sandbox (bwrap or a direct
passthrough), with a secret-free env, timeout and output cap. These run against
the ``none`` backend (direct interpreter exec) so they need no bwrap/Linux — the
bwrap *argv shape* is asserted separately without executing it."""
from __future__ import annotations

import shutil
import sys

from lithe import AgentContext, ToolCategory, ToolRegistry
from lithe.bundles.sandbox import (
    CodeRunner, register_code_tools, _sandbox_env,
)


def _runner(**kw) -> CodeRunner:
    return CodeRunner(sys.executable, backend="none", **kw)


async def test_run_code_stdout(tmp_path):
    res = await _runner().run_code(tmp_path, "print('hello')")
    assert res["exit_code"] == 0 and "hello" in res["stdout"]


async def test_run_code_exit_code_captured(tmp_path):
    res = await _runner().run_code(tmp_path, "import sys; sys.exit(3)")
    assert res["exit_code"] == 3


async def test_run_code_timeout(tmp_path):
    res = await _runner(timeout=0.4).run_code(tmp_path, "import time; time.sleep(5)")
    assert res["timed_out"] is True and "超时" in res["stderr"]


async def test_run_code_max_output_truncates(tmp_path):
    res = await _runner(max_output=40).run_code(tmp_path, "print('A' * 1000)")
    # 保首尾：头 20 + 截断标记 + 尾 20（尾部换行来自 print）
    assert res["stdout"].startswith("A" * 20)
    assert res["stdout"].rstrip("\n").endswith("A" * 19)
    assert "已截断" in res["stdout"] and len(res["stdout"]) <= 120


async def test_run_code_stderr_tail_kept_for_traceback(tmp_path):
    """Python 的 traceback 在 stderr 尾部——超限时尾部必须保留，否则模型
    看不到错误无法自我纠正。"""
    code = ("print('x' * 3000)\n"
            "raise ValueError('boom-at-the-end')\n")
    res = await _runner(max_output=500).run_code(tmp_path, code)
    assert res["exit_code"] != 0
    assert "boom-at-the-end" in res["stderr"]
    assert "ValueError" in res["stderr"]
    assert len(res["stderr"]) <= 600


async def test_run_code_stdin(tmp_path):
    res = await _runner().run_code(tmp_path, "print(input().upper())", stdin="hi")
    assert "HI" in res["stdout"]


async def test_run_file(tmp_path):
    (tmp_path / "s.py").write_text("print(1 + 1)")
    res = await _runner().run_file(tmp_path, "s.py")
    assert res["exit_code"] == 0 and "2" in res["stdout"]


async def test_run_file_traversal_rejected(tmp_path):
    res = await _runner().run_file(tmp_path, "../escape.py")
    assert res["exit_code"] == -1 and "越界" in res["stderr"]


def test_bwrap_argv_shape_and_venv_bind(tmp_path):
    r = CodeRunner(sys.executable, venv="/opt/venv", backend="bwrap")
    argv, cwd = r._full_argv([sys.executable, "-c", "x"], tmp_path)
    assert argv[1] == "--unshare-all"
    assert "--clearenv" in argv          # env wiped then re-injected (no secret leak)
    assert "/workspace" in argv          # workspace bound
    assert "/opt/venv" in argv           # venv bound read-only
    assert cwd is None                   # bwrap sets cwd via --chdir
    assert r.ready() == bool(shutil.which("bwrap"))


def test_bwrap_argv_mount_proc_toggle(tmp_path):
    default = CodeRunner(sys.executable, backend="bwrap")
    argv, _ = default._full_argv([sys.executable, "-c", "x"], tmp_path)
    assert "--proc" in argv              # procfs mounted by default

    noproc = CodeRunner(sys.executable, backend="bwrap", mount_proc=False)
    argv2, _ = noproc._full_argv([sys.executable, "-c", "x"], tmp_path)
    assert "--proc" not in argv2         # containers that forbid mount("proc")

    def strip_proc(av):
        out, skip = [], False
        for a in av:
            if a == "--proc":
                skip = True
                continue
            if skip:
                skip = False
                continue
            out.append(a)
        return out

    assert strip_proc(argv) == argv2     # everything else identical


async def test_register_code_tools_dispatch(tmp_path):
    reg = ToolRegistry()
    register_code_tools(reg, lambda ctx: str(tmp_path), _runner())
    ctx = AgentContext(run_id="r", user_id="u")
    res = await reg.dispatch("run_code", {"code": "print(7 * 6)"}, ctx)
    assert res.ok and "42" in res.content
    # missing code is rejected, not crash
    bad = await reg.dispatch("run_code", {}, ctx)
    assert bad.ok is False


def test_code_tools_are_write_classified():
    """执行模型写的代码可在沙箱内写文件：必须按写工具归类——只读模式下
    不可见，且绝不与其它调用并行（避免对同一输出文件的写竞争）。"""
    reg = ToolRegistry()
    register_code_tools(reg, lambda ctx: "/tmp", _runner())
    for name in ("run_code", "run_file"):
        spec = reg.spec(name)
        assert spec is not None
        assert spec.category is ToolCategory.WRITE
    assert reg.specs_for_mode("anchored") == [], "只读模式不暴露代码执行"


# --- Secret isolation: executed code must never observe backend secrets. The
# env-construction / argv unit tests pin this without needing bwrap; the e2e
# case runs only when bubblewrap is present. ---
_SECRETS = {
    "AUTH_SECRET": "jwt-signing-secret-xxx",
    "AGENT_API_KEY": "sk-agent-xxx",
    "AGENT_BASE_URL": "https://gateway.example/api",
    "LLM_API_KEY": "sk-llm-xxx",
    "RELAY_KEY": "relay-secret",
    "DB_PATH": "/secret/app.db",
}


def test_sandbox_env_drops_all_secrets(monkeypatch):
    for k, v in _SECRETS.items():
        monkeypatch.setenv(k, v)
    env = _sandbox_env()
    for name in _SECRETS:
        assert name not in env, f"{name} leaked into sandbox env"
    blob = "\n".join(f"{k}={v}" for k, v in env.items())
    for val in _SECRETS.values():
        assert val not in blob


def test_sandbox_env_keeps_runtime_vars():
    env = _sandbox_env()
    assert env["MPLBACKEND"] == "Agg"
    assert env["HOME"] == "/tmp"
    assert env["XDG_CACHE_HOME"] == "/tmp/.cache"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_sandbox_env_forwards_only_safe_names(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("MY_CUSTOM_TOKEN", "should-not-leak")
    env = _sandbox_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["LANG"] == "C.UTF-8"
    assert "MY_CUSTOM_TOKEN" not in env


def test_bwrap_argv_clearenv_reinjects_safe_subset(tmp_path):
    env = _sandbox_env()
    r = CodeRunner(sys.executable, backend="bwrap")
    argv, cwd = r._full_argv([sys.executable, "-c", "x"], tmp_path)
    assert "--clearenv" in argv
    set_idx = [i for i, a in enumerate(argv) if a == "--setenv"]
    assert len(set_idx) == len(env)
    reinjected = {argv[i + 1]: argv[i + 2] for i in set_idx}
    assert reinjected == env
    assert cwd is None


async def test_execution_hides_secrets_from_child(monkeypatch, tmp_path):
    import json
    # backend="none" still hands the child the secret-free env (_sandbox_env),
    # so this runs anywhere without bubblewrap; the bwrap --clearenv path is
    # pinned by the argv unit test above.
    r = CodeRunner(sys.executable, backend="none")
    for k, v in _SECRETS.items():
        monkeypatch.setenv(k, v)
    code = "import os, json; print(json.dumps(dict(os.environ)))"
    res = await r.run_code(tmp_path, code)
    child_env = json.loads(res["stdout"])
    for name in _SECRETS:
        assert name not in child_env, f"{name} visible inside sandbox"


async def test_run_code_timeout_kills_grandchildren(tmp_path):
    """孙子进程持有 stdout 管道：超时必须杀整个进程组并限时回收，
    不能被一个 Popen(['sleep', ...]) 拖到永远。"""
    import time

    runner = _runner(timeout=0.5)
    code = ("import subprocess\n"
            "subprocess.Popen(['sleep', '30'])\n"
            "import time\n"
            "time.sleep(30)\n")
    t0 = time.monotonic()
    res = await runner.run_code(str(tmp_path), code)
    elapsed = time.monotonic() - t0
    assert res["timed_out"] is True
    assert elapsed < 5.0, f"超时后 {elapsed:.1f}s 才返回，孙子进程拖住了回收"
