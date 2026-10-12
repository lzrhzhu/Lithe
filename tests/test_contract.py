"""lithe contract layer: the kernel's tool registry, undo engine, context,
events and modes are pure logic with no host/storage dependency. These pin
their behaviour so later steps (runtime, memory, host migration) build on a
stable base."""
from __future__ import annotations

import asyncio

import pytest

from lithe import (
    Action, AgentContext, AgentMode, EventType, ToolCategory, ToolRegistry,
    ToolResult, ToolSpec, UndoEngine, to_sse,
)


# --- AgentContext: typed + mapping-compatible ---

def test_context_kernel_fields_and_extra():
    ctx = AgentContext(run_id="r1", user_id="u1", extra={"thesis_id": "T1"})
    assert ctx["run_id"] == "r1"
    assert ctx["user_id"] == "u1"
    assert ctx["thesis_id"] == "T1"          # host field via extra
    assert ctx.get("missing", "d") == "d"
    assert "user_id" in ctx and "thesis_id" in ctx and "nope" not in ctx


def test_context_disabled_tools_default():
    ctx = AgentContext(run_id="r", user_id="u")
    assert ctx.disabled_tools == frozenset()


# --- ToolSpec / ToolResult ---

def test_tool_spec_openai_shape():
    spec = ToolSpec(name="read_x", description="d", category=ToolCategory.READ)
    oa = spec.to_openai()
    assert oa["type"] == "function"
    assert oa["function"]["name"] == "read_x"
    assert oa["function"]["parameters"] == {"type": "object", "properties": {}}


def test_tool_result_roundtrip():
    r = ToolResult(ok=True, summary="s", content="c", ui=[{"type": "x"}])
    assert ToolResult.from_dict(r.as_dict()) == r


# --- ToolRegistry: mode filtering + dispatch ---

async def _ok_handler(ctx, args):
    return ToolResult(ok=True, summary="ok", content=str(args))


async def _dict_handler(ctx, args):
    return {"ok": True, "summary": "dict", "content": "c", "ui": []}  # legacy dict


async def _boom_handler(ctx, args):
    raise RuntimeError("boom")


def _build_registry():
    reg = ToolRegistry()
    reg.register(ToolSpec("read_a", "r", category=ToolCategory.READ), _ok_handler)
    reg.register(ToolSpec("write_b", "w", category=ToolCategory.WRITE), _ok_handler,
                 reverter=lambda a, c: None)
    reg.register(ToolSpec("meta_c", "m", category=ToolCategory.META), _dict_handler)
    reg.register(ToolSpec("boom_d", "b", category=ToolCategory.WRITE), _boom_handler)
    return reg


def test_specs_for_mode_filters_by_category():
    reg = _build_registry()
    auto = {s["function"]["name"] for s in reg.specs_for_mode(AgentMode.AUTONOMOUS)}
    anchored = {s["function"]["name"] for s in reg.specs_for_mode(AgentMode.ANCHORED)}
    assert auto == {"read_a", "write_b", "meta_c", "boom_d"}
    assert anchored == {"read_a", "meta_c"}     # no write tools in anchored


def test_specs_for_mode_respects_disabled():
    reg = _build_registry()
    names = {s["function"]["name"] for s in
             reg.specs_for_mode(AgentMode.AUTONOMOUS, disabled=frozenset({"write_b"}))}
    assert "write_b" not in names and "read_a" in names


def test_registry_reverters_collected():
    reg = _build_registry()
    assert set(reg.reverters()) == {"write_b"}


def test_register_rejects_duplicate():
    reg = ToolRegistry()
    reg.register(ToolSpec("x", "d"), _ok_handler)
    with pytest.raises(ValueError):
        reg.register(ToolSpec("x", "d"), _ok_handler)


def test_unregister_removes_tool_and_its_reverter():
    reg = ToolRegistry()
    reg.register(ToolSpec("gone", "g", category=ToolCategory.WRITE), _ok_handler,
                 reverter=lambda a, c: None)
    reg.register(ToolSpec("stay", "s"), _ok_handler)
    assert reg.unregister("gone") is True
    assert reg.names() == ["stay"]
    assert "gone" not in reg.reverters(), "reverter 随工具一起摘除"
    # 再删返回 False；同名工具可重新注册（MCP 服务器重新 attach）
    assert reg.unregister("gone") is False
    reg.register(ToolSpec("gone", "g2"), _ok_handler)
    assert "gone" in reg.names()


# --- pre-dispatch middleware (audit / quota / confirmation) --------------------

async def test_middleware_can_deny_and_short_circuit():
    from lithe import ToolMiddleware  # noqa: F401

    reg = _build_registry()
    calls = {"handler": 0}
    denied: list[str] = []

    async def confirm_writes(ctx, name, args):
        spec = reg.spec(name)
        if spec is not None and spec.category is ToolCategory.WRITE:
            denied.append(name)
            return ToolResult(False, "待确认", f"工具 {name} 需要用户确认。")
        return None

    reg.add_middleware(confirm_writes)
    ctx = AgentContext(run_id="r", user_id="u")

    r = await reg.dispatch("write_b", {"k": 1}, ctx)
    assert r.ok is False and "需要用户确认" in r.content
    assert denied == ["write_b"]
    # 读工具不受影响，正常执行
    ok = await reg.dispatch("read_a", {}, ctx)
    assert ok.ok and calls == {"handler": 0}


async def test_middleware_chains_in_installation_order():
    reg = ToolRegistry()
    seen: list[str] = []

    async def first(ctx, name, args):
        seen.append("first")
        return None

    async def second(ctx, name, args):
        seen.append("second")
        return ToolResult(True, "substituted", "由中间件代答")

    reg.add_middleware(first)
    reg.add_middleware(second)
    reg.register(ToolSpec("t", "t"), _ok_handler)
    r = await reg.dispatch("t", {}, AgentContext(run_id="r", user_id="u"))
    assert seen == ["first", "second"]
    assert r.summary == "substituted", "第二个中间件代答，handler 未执行"


async def test_middleware_runs_after_validation():
    reg = ToolRegistry()
    order: list[str] = []

    async def mw(ctx, name, args):
        order.append("mw")
        return None

    reg.add_middleware(mw)
    reg.register(ToolSpec(
        "t", "t",
        parameters={"type": "object", "properties": {"p": {"type": "string"}},
                    "required": ["p"]}), _ok_handler)
    r = await reg.dispatch("t", {}, AgentContext(run_id="r", user_id="u"))
    assert r.ok is False, "校验失败"
    assert order == [], "校验先于中间件：坏参数不该打扰确认/审计中间件"


# --- modes: host-defined + fail-fast on unknown -----------------------------

def test_register_mode_custom_and_override():
    from lithe import ToolCategory as TC, categories_for, register_mode

    reg = _build_registry()
    register_mode("supervised", {"read", "meta"})
    names = {s["function"]["name"] for s in reg.specs_for_mode("supervised")}
    assert names == {"read_a", "meta_c"}, "自定义模式按类别过滤"

    # 覆盖内置模式（host 调优），再恢复，避免影响其他测试
    register_mode("anchored", ["read"])
    assert {s["function"]["name"] for s in reg.specs_for_mode("anchored")} == \
        {"read_a"}
    register_mode("anchored", [TC.READ, TC.META])
    assert categories_for("anchored") == frozenset({TC.READ, TC.META})


def test_unknown_mode_raises_instead_of_silent_readonly():
    import pytest as _pytest

    from lithe import categories_for

    with _pytest.raises(ValueError, match="unknown agent mode"):
        categories_for("writemode")


async def test_dispatch_normal_and_legacy_dict():
    reg = _build_registry()
    ctx = AgentContext(run_id="r", user_id="u")
    r = await reg.dispatch("read_a", {"k": 1}, ctx)
    assert r.ok and "k" in r.content
    # a handler returning a plain dict is normalized to ToolResult
    r2 = await reg.dispatch("meta_c", {}, ctx)
    assert isinstance(r2, ToolResult) and r2.summary == "dict"


async def test_dispatch_unknown_and_disabled():
    reg = _build_registry()
    ctx = AgentContext(run_id="r", user_id="u", disabled_tools=frozenset({"read_a"}))
    assert (await reg.dispatch("nope", {}, ctx)).ok is False
    assert (await reg.dispatch("read_a", {}, ctx)).ok is False  # disabled


async def test_dispatch_swallows_handler_exception():
    reg = _build_registry()
    ctx = AgentContext(run_id="r", user_id="u")
    r = await reg.dispatch("boom_d", {}, ctx)
    assert r.ok is False and "boom" in r.content


# --- dispatch-time argument validation (fed back for self-correction) ---------

def _validated_registry():
    reg = ToolRegistry()
    reg.register(ToolSpec(
        "demo", "d",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "n": {"type": "number"},
                                   "flag": {"type": "boolean"}},
                    "required": ["path"]},
        category=ToolCategory.READ), _ok_handler)
    return reg


async def test_dispatch_rejects_missing_required_and_bad_type():
    reg = _validated_registry()
    ctx = AgentContext(run_id="r", user_id="u")

    missing = await reg.dispatch("demo", {"n": 1}, ctx)
    assert missing.ok is False and "缺少必填参数 path" in missing.content

    bad_type = await reg.dispatch("demo", {"path": "a", "n": True}, ctx)
    assert bad_type.ok is False and "n 应为 number" in bad_type.content

    ok = await reg.dispatch("demo", {"path": "a", "n": 2.5, "extra": 1}, ctx)
    assert ok.ok, "未声明字段与合法类型不应被拦截"


async def test_dispatch_validation_accepts_type_unions_and_opt_out():
    reg = ToolRegistry()
    reg.register(ToolSpec(
        "union", "u",
        parameters={"type": "object",
                    "properties": {"v": {"type": ["string", "null"]}},
                    "required": ["v"]},
        category=ToolCategory.READ), _ok_handler)
    reg.register(ToolSpec("loose", "l", category=ToolCategory.READ, validate=False),
                 _ok_handler)
    ctx = AgentContext(run_id="r", user_id="u")
    assert (await reg.dispatch("union", {"v": None}, ctx)).ok
    assert (await reg.dispatch("union", {"v": "x"}, ctx)).ok
    assert (await reg.dispatch("union", {"v": 3}, ctx)).ok is False
    # validate=False：schema 再苛刻也不拦
    assert (await reg.dispatch("loose", {"whatever": object()}, ctx)).ok


async def test_dispatch_per_tool_timeout_cancels_handler():
    import asyncio
    import time

    finished = {"flag": False}

    async def slow(ctx, args):
        await asyncio.sleep(5)
        finished["flag"] = True
        return ToolResult(True, "late", "late")

    reg = ToolRegistry()
    reg.register(ToolSpec("slow", "s", category=ToolCategory.READ, timeout=0.05),
                 slow)
    ctx = AgentContext(run_id="r", user_id="u")
    started = time.monotonic()
    r = await reg.dispatch("slow", {}, ctx)
    assert r.ok is False and r.summary == "执行超时" and "0.05s" in r.content
    assert time.monotonic() - started < 1.0
    assert finished["flag"] is False, "超时后 handler 任务应被取消"
    # 未设置 timeout 的工具不受影响
    reg.register(ToolSpec("fast", "f", category=ToolCategory.READ), _ok_handler)
    assert (await reg.dispatch("fast", {}, ctx)).ok


# --- UndoEngine: pure, newest-first, fault-tolerant ---

async def test_undo_reverts_newest_first_skips_non_revertible():
    order: list[tuple[str, str]] = []

    def rev(kind):
        def _r(action, ctx):
            order.append((kind, action.target))
        return _r

    engine = UndoEngine({"file_write": rev("file_write")})
    ctx = AgentContext(run_id="r", user_id="u")
    actions = [
        Action(kind="file_write", target="a", status="applied"),
        Action(kind="file_write", target="b", status="reverted"),   # skipped
        Action(kind="file_write", target="c", status="applied"),
        Action(kind="unknown_kind", target="d", status="applied"),  # no reverter
    ]
    report = await engine.undo(actions, ctx)
    assert report.reverted == 2
    assert report.skipped == 2
    assert order == [("file_write", "c"), ("file_write", "a")]   # newest first


async def test_undo_reverter_failure_is_non_fatal():
    def bad(action, ctx):
        raise RuntimeError("nope")

    engine = UndoEngine({"file_write": bad})
    ctx = AgentContext(run_id="r", user_id="u")
    report = await engine.undo(
        [Action(kind="file_write", target="a", status="applied"),
         Action(kind="file_write", target="b", status="applied")], ctx)
    assert report.reverted == 0 and report.ok is False
    assert len(report.errors) == 2


async def test_undo_supports_async_and_mixed_reverters():
    """异步 reverter（远程 API 回滚场景）与同步混用，顺序与计数不受影响。"""
    order: list[str] = []

    async def rev_remote(action, ctx):
        await asyncio.sleep(0)
        order.append(f"async:{action.target}")

    def rev_local(action, ctx):
        order.append(f"sync:{action.target}")

    engine = UndoEngine({"remote_write": rev_remote, "file_write": rev_local})
    ctx = AgentContext(run_id="r", user_id="u")
    report = await engine.undo([
        Action(kind="remote_write", target="r1", status="applied"),
        Action(kind="file_write", target="f1", status="applied"),
        Action(kind="remote_write", target="r2", status="applied"),
    ], ctx)
    assert report.reverted == 3 and report.ok is True
    assert order == ["async:r2", "sync:f1", "async:r1"], "最新优先，同步/异步交错"

    # 异步 reverter 内部抛错同样按非致命错误计
    async def bad_async(action, ctx):
        raise RuntimeError("remote down")

    engine2 = UndoEngine({"remote_write": bad_async})
    report2 = await engine2.undo(
        [Action(kind="remote_write", target="x", status="applied")], ctx)
    assert report2.reverted == 0 and report2.ok is False and report2.errors


# --- events ---

def test_to_sse_format():
    line = to_sse({"type": EventType.ASSISTANT, "text": "你好"})
    assert line.startswith("data: ") and line.endswith("\n\n")
    assert "你好" in line            # non-ASCII preserved
    assert '"type": "assistant"' in line
