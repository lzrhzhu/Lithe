"""The tool registry's argument pipeline: schema validation (enums listed
for self-correction), pre-validation transforms, middleware seeing
transformed args, junk transforms rejected, and non-object dispatch args
answered as failed results instead of raising."""
from __future__ import annotations

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, ToolCategory, ToolRegistry,
    ToolResult, ToolSpec,
)
from lithe.tools import validate_args

from _fakes import echo, resp, tc


def _ctx():
    return AgentContext(run_id="r", user_id="u")


# --- P1-6a: enum validation ------------------------------------------------------

def test_validate_args_checks_enum():
    params = {"type": "object",
              "properties": {"mode": {"type": "string",
                                      "enum": ["fast", "slow"]}}}
    assert validate_args(params, {"mode": "fast"}) is None
    err = validate_args(params, {"mode": "turbo"})
    assert err and "fast" in err and "slow" in err and "turbo" in err


async def test_enum_violation_fed_back_and_model_self_corrects():
    calls = []

    async def pick(ctx, args):
        calls.append(args["mode"])
        return ToolResult(True, "picked", args["mode"])

    reg = ToolRegistry()
    reg.register(ToolSpec(
        "pick_mode", "p", category=ToolCategory.READ,
        parameters={"type": "object",
                    "properties": {"mode": {"type": "string",
                                            "enum": ["fast", "slow"]}},
                    "required": ["mode"]}), pick)

    class _T:
        def __init__(self):
            self.n = 0

        async def complete(self, client, **kw):
            self.n += 1
            if self.n == 1:
                return resp(tool_calls=[tc("pick_mode", {"mode": "turbo"},
                                             "c1")])
            if self.n == 2:
                return resp(tool_calls=[tc("pick_mode", {"mode": "fast"},
                                             "c2")])
            return resp("选定 fast")

    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_T()), max_steps=4)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    events = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    # 非法值绝不能执行；错误列出全部合法值供模型自纠
    assert calls == ["fast"]
    trs = [e for e in events if e["type"] == "tool_result"]
    assert trs[0]["ok"] is False and "fast" in trs[0]["error"] \
        and "slow" in trs[0]["error"]
    final = [e for e in events if e["type"] == "assistant"]
    assert final and final[-1]["text"] == "选定 fast"


# --- P1-6b: tool-arg transforms --------------------------------------------------

async def test_transform_normalizes_args():
    seen = []

    async def read(ctx, args):
        seen.append(args)
        return ToolResult(True, "ok", str(args))

    async def strip_dot(ctx, name, args):
        assert name == "read_file"
        p = args.get("path")
        if isinstance(p, str):
            return {**args, "path": p.lstrip("./")}
        return args

    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "r", category=ToolCategory.READ), read)
    reg.add_transform(strip_dot)
    res = await reg.dispatch("read_file", {"path": "./a.txt"},
                             AgentContext(run_id="r", user_id="u"))
    assert res.ok
    assert seen == [{"path": "a.txt"}]


async def test_transform_runs_before_validation():
    # 变换把模型给的数字路径规整成字符串，校验因此通过
    async def coerce(ctx, name, args):
        if isinstance(args.get("path"), int):
            args = {**args, "path": str(args["path"])}
        return args

    reg = ToolRegistry()
    reg.register(ToolSpec(
        "read_file", "r", category=ToolCategory.READ,
        parameters={"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]}), echo)
    reg.add_transform(coerce)
    res = await reg.dispatch("read_file", {"path": 7},
                             AgentContext(run_id="r", user_id="u"))
    assert res.ok, "变换先于校验：规整后的参数应通过校验"


async def test_middleware_sees_transformed_args():
    vetoed = []

    async def canonicalize(ctx, name, args):
        return {**args, "path": "canonical:" + str(args.get("path", ""))}

    async def watch(ctx, name, args):
        vetoed.append(args["path"])
        return None

    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "r", category=ToolCategory.READ), echo)
    reg.add_transform(canonicalize)
    reg.add_middleware(watch)
    res = await reg.dispatch("read_file", {"path": "x"},
                             AgentContext(run_id="r", user_id="u"))
    assert res.ok
    assert vetoed == ["canonical:x"], "middleware 看到的是变换后的参数"


async def test_transform_returning_junk_rejected():
    async def bad(ctx, name, args):
        return "oops"

    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "r", category=ToolCategory.READ), echo)
    reg.add_transform(bad)
    res = await reg.dispatch("read_file", {"path": "x"},
                             AgentContext(run_id="r", user_id="u"))
    assert res.ok is False and "变换" in res.content



# --------------------------------------------------------------------------- #
# tools: non-object args on the public dispatch path
# --------------------------------------------------------------------------- #

async def test_dispatch_non_object_args_is_failed_result_not_typeerror():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e"), echo)
    ctx = _ctx()
    for bad in ('[1,2]', '"str"', 5, b"x"):
        res = await reg.dispatch("echo", bad, ctx)
        assert isinstance(res, ToolResult) and not res.ok
        assert "JSON 对象" in res.content
    # 合法 JSON 字符串仍照旧解析
    ok = await reg.dispatch("echo", '{"a": 1}', ctx)
    assert ok.ok and ok.content == "{'a': 1}"


def test_validate_args_non_dict_args_returns_message():
    assert validate_args({"type": "object", "properties": {}}, [1]) == \
        "参数应为 JSON 对象，得到 list"
