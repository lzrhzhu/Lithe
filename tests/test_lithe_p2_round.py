"""P2 hardening round: regression tests for the fixes landed after 0.7.0.

Groups mirror the changelog: kernel robustness (id synthesis, empty
responses, recorded synthetic endings, envelope, http_client injection),
subagent small fixes, store integrity (subagent filter, torn lines, undo
mark failures, todos), tool bundles (MCP pagination/env/logs, edit_file
trailing blanks, patch EOF anchoring / insertion order, images, search),
and the facade (DictToolAdapter, deprecated alias).
"""
from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolCategory,
    ToolRegistry, ToolResult, ToolSpec,
)
from lithe.events import to_sse
from lithe.memory import replay_messages, run_timeline
from lithe.tools import validate_args


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def _resp(content="", tool_calls=None):
    return {"content": content, "tool_calls": tool_calls or []}


def _tc(name="echo", args=None, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


class _FakeTransport:
    def __init__(self, responses):
        self._it = iter(list(responses))
        self.calls = 0
        self.seen_clients = []

    async def complete(self, client, **kw):
        self.calls += 1
        self.seen_clients.append(client)
        return next(self._it)


class _Sink:
    def __init__(self):
        self.events = []
        self.records = []

    async def on_event(self, ctx, event):
        self.events.append(event)

    async def on_record(self, ctx, record):
        self.records.append(record)


async def _echo(ctx, args):
    return ToolResult(ok=True, summary="echoed", content=str(args))


def _runtime(transport, *, registry=None, sinks=None, max_steps=3, **kw):
    cfg = LLMConfig(model="m", base_url="x", api_key="k", transport=transport)
    reg = registry or ToolRegistry()
    return AgentRuntime(reg, cfg, sinks=sinks, max_steps=max_steps, **kw)


def _ctx():
    return AgentContext(run_id="r", user_id="u")


# --------------------------------------------------------------------------- #
# kernel: tool_call id synthesis
# --------------------------------------------------------------------------- #

async def test_missing_tool_call_id_synthesized_everywhere():
    """A gateway that omits ids gets one synthesized id shared by the
    assistant turn, the tool result record and the in-memory messages."""
    no_id = {"type": "function",
             "function": {"name": "echo", "arguments": "{}"}}
    rt = _runtime(_FakeTransport([_resp(tool_calls=[no_id]), _resp("done")]))
    reg = rt.registry
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    sink = _Sink()
    rt.sinks = [sink]
    ctx = _ctx()
    messages = [{"role": "user", "content": "hi"}]
    events = [e async for e in rt.run(ctx, messages, [])]
    synth = messages[1]["tool_calls"][0]["id"]
    assert isinstance(synth, str) and synth.startswith("call_") and len(synth) > 8
    # the tool result pairs with the synthesized id …
    tool_records = [r for r in sink.records if r["role"] == "tool"]
    assert tool_records[0]["tool_call_id"] == synth
    assert messages[2]["tool_call_id"] == synth
    # … and the tool_call display event announced it too
    tc_events = [e for e in events if e["type"] == "tool_call"]
    assert tc_events[0]["id"] == synth


async def test_two_steps_missing_ids_do_not_collide():
    t1 = {"type": "function", "function": {"name": "echo", "arguments": "{}"}}
    t2 = {"type": "function", "function": {"name": "echo", "arguments": "{}"}}
    rt = _runtime(_FakeTransport([_resp(tool_calls=[t1]), _resp(tool_calls=[t2]),
                                  _resp("done")]))
    rt.registry.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    ctx = _ctx()
    messages = [{"role": "user", "content": "hi"}]
    _ = [e async for e in rt.run(ctx, messages, [])]
    id1 = messages[1]["tool_calls"][0]["id"]
    id2 = messages[3]["tool_calls"][0]["id"]
    assert id1 != id2


async def test_streamed_tool_calls_leave_id_none_for_single_synthesis():
    """The streaming assembler must not mint per-response call_0 ids — the
    runtime's normalize step owns id synthesis."""
    from lithe.llm import iter_chat_completion

    def handler(request):
        sse = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"name":"f","arguments":"{\\"a\\":"}}]}}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"1}"}}]}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=sse,
                              headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        result = None
        async for part in iter_chat_completion(c, base_url="http://x",
                                                api_key="k", model="m",
                                                messages=[]):
            if "result" in part:
                result = part["result"]
    tcs = result["choices"][0]["message"]["tool_calls"]
    assert tcs[0]["id"] is None, "streaming 侧不得自行合成 id"
    assert tcs[0]["function"]["arguments"] == '{"a":1}'


# --------------------------------------------------------------------------- #
# kernel: empty responses / synthetic endings / envelope / http_client
# --------------------------------------------------------------------------- #

async def test_empty_response_retried_once_then_real_answer():
    tr = _FakeTransport([_resp(""), _resp("recovered")])
    rt = _runtime(tr)
    stats = RunStats()
    events = [e async for e in rt.run(_ctx(),
                                      [{"role": "user", "content": "hi"}], [],
                                      stats=stats)]
    assert tr.calls == 2
    assert stats.status == "done" and stats.final_text == "recovered"
    assert [e for e in events if e["type"] == "error"] == []


async def test_empty_response_twice_ends_with_distinguishable_status():
    tr = _FakeTransport([_resp(""), _resp("")])
    rt = _runtime(tr)
    stats = RunStats()
    events = [e async for e in rt.run(_ctx(),
                                      [{"role": "user", "content": "hi"}], [],
                                      stats=stats)]
    assert tr.calls == 2, "恰好重试一次"
    assert stats.status == "empty_response"
    assert stats.final_text == ""
    assert any(e["type"] == "error" for e in events)


async def test_max_steps_fallback_text_is_recorded():
    tr = _FakeTransport([_resp(tool_calls=[_tc("echo")])] * 10)
    rt = _runtime(tr, max_steps=2)
    rt.registry.register(ToolSpec("echo", "e", category=ToolCategory.READ), _echo)
    sink = _Sink()
    rt.sinks = [sink]
    ctx = _ctx()
    messages = [{"role": "user", "content": "hi"}]
    _ = [e async for e in rt.run(ctx, messages, [])]
    assert messages[-1]["role"] == "assistant"
    assert "最大步数" in messages[-1]["content"]
    assert any("最大步数" in (r.get("content") or "")
               for r in sink.records if r["role"] == "assistant"), \
        "兜底文本必须进 record 通道，否则 sink 与下轮 replay 都看不到"


async def test_runtime_envelope_events_when_enabled():
    rt = _runtime(_FakeTransport([_resp("done")]), emit_envelope=True)
    events = [e async for e in rt.run(_ctx(),
                                      [{"role": "user", "content": "hi"}], [])]
    assert events[0]["type"] == "run_start"
    assert events[-1]["type"] == "done"
    assert events[-1]["status"] == "done"
    assert events[-1]["steps"] == 1


async def test_runtime_no_envelope_by_default():
    rt = _runtime(_FakeTransport([_resp("done")]))
    events = [e async for e in rt.run(_ctx(),
                                      [{"role": "user", "content": "hi"}], [])]
    assert all(e["type"] not in ("run_start", "done") for e in events)


async def test_injected_http_client_forwarded_and_reused():
    sentinel = SimpleNamespace(name="shared-client")
    tr = _FakeTransport([_resp("a"), _resp("b")])
    rt = _runtime(tr, http_client=sentinel)
    for _ in range(2):
        _ = [e async for e in rt.run(_ctx(),
                                     [{"role": "user", "content": "q"}], [])]
    assert tr.seen_clients == [sentinel, sentinel], "注入 client 跨 run 复用"


async def test_injected_http_client_default_timeout_upgraded():
    # A plain httpx.AsyncClient() carries 5s reads — fatal for reasoning
    # models. The runtime upgrades it to cfg.timeout instead of ignoring it.
    tr = _FakeTransport([_resp("a")])
    rt = _runtime(tr, http_client=httpx.AsyncClient())
    assert rt.http_client.timeout == httpx.Timeout(5.0)
    _ = [e async for e in rt.run(_ctx(),
                                 [{"role": "user", "content": "q"}], [])]
    assert rt.http_client.timeout == httpx.Timeout(180.0)  # cfg.timeout default
    await rt.http_client.aclose()


async def test_injected_http_client_custom_timeout_untouched():
    tr = _FakeTransport([_resp("a")])
    custom = httpx.Timeout(30.0)
    rt = _runtime(tr, http_client=httpx.AsyncClient(timeout=custom))
    _ = [e async for e in rt.run(_ctx(),
                                 [{"role": "user", "content": "q"}], [])]
    # a deliberately configured host timeout is respected, not overridden
    assert rt.http_client.timeout == custom
    await rt.http_client.aclose()


# --------------------------------------------------------------------------- #
# tools: non-object args on the public dispatch path
# --------------------------------------------------------------------------- #

async def test_dispatch_non_object_args_is_failed_result_not_typeerror():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e"), _echo)
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


# --------------------------------------------------------------------------- #
# memory: window-start orphans / timeline ok=unknown
# --------------------------------------------------------------------------- #

def test_replay_window_starting_at_tool_row_drops_orphan():
    rows = [
        {"role": "tool", "tool_call_id": "c-old", "content": "stale"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "fresh"},
    ]
    out = replay_messages(rows)
    assert out[0]["role"] == "assistant", "首消息不得是孤儿 tool 行"
    assert all(m.get("tool_call_id") != "c-old" for m in out)


def test_replay_legacy_positional_pairing_still_works():
    rows = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": "r1"},  # 无 id，按位置配对
    ]
    out = replay_messages(rows)
    assert out[1]["tool_call_id"] == "c1"


def test_run_timeline_missing_meta_reports_ok_unknown():
    msgs = [{"role": "tool", "content": "工具调用失败：x", "tool_name": "t",
             "tool_call_id": "c1"}]
    events = run_timeline({}, msgs, [],
                          tool_category={"t": "file"},
                          action_category=lambda a: None,
                          action_events_fn=lambda a: [])
    tr = events[0]
    assert tr["ok"] is None, "缺 meta 时 ok 未知，而不是嗅探错误前缀"


# --------------------------------------------------------------------------- #
# events: to_sse default=str
# --------------------------------------------------------------------------- #

def test_to_sse_serializes_datetime_via_str():
    from datetime import datetime
    line = to_sse({"type": "user", "at": datetime(2026, 1, 1, 12, 0)})
    assert line.startswith("data: ")
    payload = json.loads(line[len("data: "):])
    assert payload["at"] == "2026-01-01 12:00:00"


# --------------------------------------------------------------------------- #
# subagents
# --------------------------------------------------------------------------- #

def test_thin_event_caps_args_and_summary():
    from lithe.bundles.subagents import _THIN_ARGS_CAP, _thin_event
    fat = "x" * 5000
    thin = _thin_event({"type": "tool_call", "id": "c", "name": "write_file",
                        "args": {"content": fat}})
    assert len(thin["args"]) <= _THIN_ARGS_CAP + 1
    thin = _thin_event({"type": "tool_result", "id": "c", "name": "n",
                        "ok": True, "summary": fat})
    assert len(thin["summary"]) <= 301
    thin = _thin_event({"type": "tool_result", "id": "c", "name": "n",
                        "ok": False, "error": fat})
    assert len(thin["error"]) <= 301


async def test_delegate_parallel_accepts_duplicate_agents(tmp_path):
    """同一子代理的多项任务不再整批拒绝：按列出顺序串行执行，批次成功。
    （旧契约是返回"重复子代理"错误，模型被迫合并任务或拆成多次调用。）"""
    from lithe.bundles import JsonlRunStore, SubagentEngine, SubagentRoster
    from lithe.bundles.subagents import (SubagentSpec,
                                            make_parallel_delegate_tool)
    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    host_like = SimpleNamespace(
        registry=reg,
        llm_config=LLMConfig(model="m", base_url="x", api_key="k",
                             transport=_FakeTransport([_resp("done a"),
                                                       _resp("done b")])),
        store=store, context_budget=None, strict_records=False,
        max_cost=None, max_total_tokens=None, repeat_call_limit=None,
        max_steps=2, http_client=None)
    engine = SubagentEngine(host_like, SubagentRoster(
        [SubagentSpec("w", "写者", "写", ["echo"], "p")]))
    spec, handler = make_parallel_delegate_tool(engine)
    res = await handler(_ctx(), {"tasks": [
        {"agent": "w", "task": "a"}, {"agent": "w", "task": "b"}]})
    assert res.ok
    assert "重复" not in res.content
    assert res.content.count("### w（成功）") == 2


def test_subagent_engine_roster_is_read_only():
    from lithe.bundles.subagents import SubagentEngine, SubagentRoster, \
        SubagentSpec
    host = SimpleNamespace()
    engine = SubagentEngine(host, SubagentRoster(
        [SubagentSpec("a", "A", "d", [], "p")]))
    assert engine.roster.ids() == ["a"]
    with pytest.raises(AttributeError):
        engine.roster = SubagentRoster([SubagentSpec("b", "B", "d", [], "p")])


def test_delegate_tool_timeout_wired():
    from lithe.bundles.subagents import SubagentEngine, SubagentRoster, \
        SubagentSpec, make_delegate_tool
    engine = SubagentEngine(SimpleNamespace(),
                            SubagentRoster([SubagentSpec("a", "A", "d", [],
                                                         "p")]))
    spec, _ = make_delegate_tool(engine, timeout=123.0)
    assert spec.timeout == 123.0


# --------------------------------------------------------------------------- #
# store: subagent filter / torn lines / undo mark failures
# --------------------------------------------------------------------------- #

def _store_with_actions(tmp_path):
    from lithe.bundles import JsonlRunStore
    store = JsonlRunStore(tmp_path)
    store.log_action("r", "u", "file_write", "a", None, "1", subagent="w1")
    store.log_action("r", "u", "file_write", "b", None, "2", subagent="w2")
    store.log_action("r", "u", "file_write", "c", None, "3")  # 编排者自身
    return store


def test_list_actions_subagent_filter(tmp_path):
    store = _store_with_actions(tmp_path)
    w1 = store.list_actions("r", "u", subagent="w1")
    assert [a.target for a in w1] == ["a"]
    allr = store.list_actions("r", "u")
    assert [a.target for a in allr] == ["a", "b", "c"]


def test_list_actions_subagent_filter_skips_blob_rehydrate(tmp_path,
                                                            monkeypatch):
    from lithe.bundles import JsonlRunStore
    store = JsonlRunStore(tmp_path)
    big = "v" * (store.spill_threshold + 100)
    store.log_action("r", "u", "k", "a", big, big, subagent="w1")
    store.log_action("r", "u", "k", "b", big, big, subagent="w2")
    unspills = {"n": 0}
    orig = JsonlRunStore._unspill_value

    def counting(self, v):
        if isinstance(v, str) and v.startswith("blob:"):
            unspills["n"] += 1
        return orig(self, v)

    monkeypatch.setattr(JsonlRunStore, "_unspill_value", counting)
    w2 = store.list_actions("r", "u", subagent="w2")
    assert [a.target for a in w2] == ["b"]
    assert unspills["n"] == 2, "只 rehydrate 被选中行的 old/new"


def test_torn_lines_counted_and_logged(tmp_path, caplog):
    store = _store_with_actions(tmp_path)
    with (tmp_path / "actions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "action", "id": 99, "broken')  # 撕裂行
    with caplog.at_level("WARNING"):
        acts = store.list_actions("r", "u")
    assert [a.target for a in acts] == ["a", "b", "c"]
    assert store.dropped_lines.get(tmp_path / "actions.jsonl") == 1
    assert any("dropped 1" in r.message for r in caplog.records)


async def test_undo_run_reports_status_mark_failure(tmp_path):
    from lithe.bundles import AgentHost, JsonlRunStore
    from lithe.bundles.host import undo_run

    class _FailingMark(JsonlRunStore):
        def set_action_status(self, action_id, user_id, status):
            raise OSError("disk full")

    store = _FailingMark(tmp_path)
    store.create_run("r", "u", "t")
    store.log_action("r", "u", "k", "target", "old", "new")
    reg = ToolRegistry()
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k"),
                     store)
    calls = {"n": 0}

    def revert(action, ctx):
        calls["n"] += 1

    report = await undo_run(host, "r", "u", reverters={"k": revert})
    assert calls["n"] == 1, "reverter 本身执行成功"
    assert not report.ok
    assert report.errors and "标记撤销状态失败" in report.errors[0]


# --------------------------------------------------------------------------- #
# todos
# --------------------------------------------------------------------------- #

def test_json_todo_store_drops_malformed_items(tmp_path):
    from lithe.bundles import JsonTodoStore
    p = tmp_path / "todos.json"
    p.write_text(json.dumps({"items": [
        {"id": "i1", "content": "ok", "status": "pending",
         "priority": "medium"},
        {"content": "bad status", "status": "bogus"},
        "not-a-dict",
        {"content": "", "status": "pending"},
    ]}), encoding="utf-8")
    store = JsonTodoStore(p)
    items = store.list()
    assert len(items) == 1 and items[0]["content"] == "ok"
    block = store.to_block()  # 不得抛异常（它渲染进每次 run 的系统提示词）
    assert "ok" in block


def test_json_todo_store_save_is_atomic_no_tmp_left(tmp_path):
    from lithe.bundles import JsonTodoStore
    p = tmp_path / "todos.json"
    store = JsonTodoStore(p)
    store.replace([{"content": "a", "status": "pending"}])
    store.replace([{"content": "b", "status": "in_progress"}])
    assert not list(tmp_path.glob("*.tmp"))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["items"][0]["content"] == "b"


# --------------------------------------------------------------------------- #
# MCP
# --------------------------------------------------------------------------- #

class _PageSession:
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    async def call(self, method, params=None, *, timeout=None):
        self.calls.append((method, params))
        return self._pages.pop(0)


async def test_tools_list_pagination_followed():
    from lithe.bundles.mcp import _list_all_tools
    session = _PageSession([
        {"tools": [{"name": "a"}], "nextCursor": "p2"},
        {"tools": [{"name": "b"}], "nextCursor": "p3"},
        {"tools": [{"name": "c"}]},
    ])
    tools = await _list_all_tools(session, page_timeout=1.0)
    assert [t["name"] for t in tools] == ["a", "b", "c"]
    assert session.calls[1] == ("tools/list", {"cursor": "p2"})
    assert session.calls[2] == ("tools/list", {"cursor": "p3"})


def test_stdio_spawn_env_allowlist(monkeypatch):
    from lithe.bundles.mcp import MCPServerConfig, _StdioSession
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("AUTH_SECRET", "hunter2")
    monkeypatch.setenv("MY_API_KEY", "sk-leak")
    cfg = MCPServerConfig(name="s", command=["x"],
                          env={"TOOL_KEY": "tv"})
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    for key, value in {
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "TEMP": r"C:\Temp",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
    }.items():
        monkeypatch.setenv(key, value)
    env = _StdioSession(cfg)._spawn_env(windows=True)
    assert env["PATH"] == "/usr/bin" and env["TOOL_KEY"] == "tv"
    assert env["SYSTEMROOT"] == r"C:\Windows"
    assert env["WINDIR"] == r"C:\Windows"
    assert "AUTH_SECRET" not in env and "MY_API_KEY" not in env

    legacy = _StdioSession(
        MCPServerConfig(name="s", command=["x"], inherit_env=True)
    )._spawn_env()
    assert legacy.get("AUTH_SECRET") == "hunter2", "显式 True 保留旧行为"

    closed = _StdioSession(
        MCPServerConfig(name="s", command=["x"], inherit_env=False)
    )._spawn_env(windows=True)
    assert closed.get("TOOL_KEY") is None
    assert set(closed).issubset({
        "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "USERPROFILE",
        "HOMEDRIVE", "HOMEPATH", "TEMP", "TMP", "OS", "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
    })


def test_log_host_strips_credentials():
    from lithe.bundles.mcp import _log_host
    assert _log_host("https://user:pass@evil.example/api?k=secret") == \
        "https://evil.example"
    assert _log_host("https://plain.example/api") == "https://plain.example"
    assert _log_host(None) == "?"


def test_parse_servers_error_does_not_embed_value():
    from lithe.bundles.mcp import parse_servers
    with pytest.raises(ValueError) as ei:
        parse_servers(json.dumps({"bad": "https://k@host/?key=SECRET"}))
    assert "SECRET" not in str(ei.value)


async def test_stdio_read_loop_ignores_non_object_json():
    """stdout 出现 JSON 数组/标量行不得杀死会话（reader 不得 AttributeError）。"""
    from lithe.bundles.mcp import MCPServerConfig, _StdioSession
    session = _StdioSession(MCPServerConfig(name="s", command=["x"]))
    reader = asyncio.StreamReader()
    reader.feed_data(b'[1,2]\n"hello"\n42\n')
    reader.feed_data(b'{"jsonrpc":"2.0","id":7,"result":{"ok":true}}\n')
    reader.feed_eof()
    fut = asyncio.get_running_loop().create_future()
    session._pending[7] = fut
    session.proc = SimpleNamespace(stdout=reader, stdin=None)
    await session._read_loop()
    assert fut.done() and fut.result()["result"] == {"ok": True}


# --------------------------------------------------------------------------- #
# workspace: edit_file trailing blanks / search hardening / list subdirs
# --------------------------------------------------------------------------- #

def _ws_reg(tmp_path):
    from lithe.bundles.workspace import Workspace, register_file_tools
    ws = Workspace(tmp_path)
    reg = ToolRegistry()
    register_file_tools(reg, lambda ctx: ws)
    return ws, reg


async def test_edit_file_fuzzy_trailing_blank_alignment(tmp_path):
    ws, reg = _ws_reg(tmp_path)
    ws.write("f.txt", "a  \nb\n")  # 行尾空白使精确子串 "a\n" 无法命中
    res = await reg.dispatch("edit_file", {
        "path": "f.txt", "old_text": "a\n", "new_text": "X\n"}, _ctx())
    assert res.ok, res.content
    assert ws.read("f.txt") == "X\nb\n", "模糊命中不得每次多插一个空行"


async def test_search_files_skips_overlong_lines(tmp_path):
    ws, reg = _ws_reg(tmp_path)
    ws.write("big.txt", "needle" + "x" * 20000 + "\n")  # needle 在超长行上
    ws.write("small.txt", "needle here\n")
    res = await reg.dispatch("search_files", {"pattern": "needle"}, _ctx())
    assert res.ok
    assert "small.txt:1" in res.content
    assert "big.txt:1" not in res.content, "超长行必须跳过"
    assert "超长行" in res.content


def test_search_files_has_timeout_spec(tmp_path):
    _, reg = _ws_reg(tmp_path)
    assert reg.spec("search_files").timeout == 30.0


def test_workspace_list_subdirs_go_through_guard(tmp_path):
    from lithe.bundles.workspace import Workspace
    ws = Workspace(tmp_path)
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    with pytest.raises(PermissionError):
        ws.list(("../outside-target",))


# --------------------------------------------------------------------------- #
# patch: EOF anchoring / insertion order
# --------------------------------------------------------------------------- #

def _patch_reg(tmp_path):
    from lithe.bundles.patch import register_apply_patch_tool
    from lithe.bundles.workspace import Workspace
    ws = Workspace(tmp_path)
    reg = ToolRegistry()
    register_apply_patch_tool(reg, lambda ctx: ws)
    return ws, reg


async def test_eof_chunk_tail_ladder_beats_forward_exact(tmp_path):
    """尾部仅差空白时，EOF chunk 必须锚定到文件尾，而不是前向精确命中
    更早的相似行。"""
    ws, reg = _patch_reg(tmp_path)
    ws.write("f.txt", "dupe\nmiddle\ndupe \n")  # 尾行多一个行尾空格
    res = await reg.dispatch("apply_patch", {"patch_text":
        "*** Begin Patch\n"
        "*** Update File: f.txt\n"
        "@@\n"
        "-dupe\n"
        "+REPLACED\n"
        "*** End of File\n"
        "*** End Patch\n"}, _ctx())
    assert res.ok, res.content
    out = ws.read("f.txt")
    assert out.startswith("dupe\nmiddle\n"), "前部的 dupe 不得被改"
    assert "REPLACED" in out


async def test_two_pure_insertions_keep_document_order(tmp_path):
    ws, reg = _patch_reg(tmp_path)
    ws.write("f.txt", "base\n")
    res = await reg.dispatch("apply_patch", {"patch_text":
        "*** Begin Patch\n"
        "*** Update File: f.txt\n"
        "@@\n"
        "+first-append\n"
        "@@\n"
        "+second-append\n"
        "*** End Patch\n"}, _ctx())
    assert res.ok, res.content
    out = ws.read("f.txt")
    assert out.index("first-append") < out.index("second-append"), \
        "同节两个纯插入 chunk 不得颠倒顺序"


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #

_PNG_MIN = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" +
            (4).to_bytes(4, "big") + (3).to_bytes(4, "big") +
            b"\x08\x06\x00\x00\x00")


async def test_analyze_image_detail_auto_shares_cache_key(tmp_path,
                                                           monkeypatch):
    from lithe.bundles.images import register_image_tools
    from lithe.bundles.workspace import Workspace
    ws = Workspace(tmp_path)
    ws.write_bytes("img.png", _PNG_MIN)
    reg = ToolRegistry()
    register_image_tools(
        reg, lambda ctx: ws,
        llm_config=LLMConfig(model="m", base_url="x", api_key="k"))
    calls = {"n": 0}

    async def fake_completion(client, **kw):
        calls["n"] += 1
        return {"choices": [{"message": {"content": "看到一张图"}}]}

    monkeypatch.setattr("lithe.bundles.images.chat_completion",
                        fake_completion)
    r1 = await reg.dispatch("analyze_image",
                            {"path": "img.png", "question": "是什么",
                             "detail": "auto"}, _ctx())
    r2 = await reg.dispatch("analyze_image",
                            {"path": "img.png", "question": "是什么"}, _ctx())
    assert r1.ok and r2.ok
    assert calls["n"] == 1, "detail=auto 与缺省是同一缓存键，不得重复付费"


async def test_image_info_probe_window_bounded(tmp_path):
    from lithe.bundles.images import _PROBE_WINDOW, register_image_tools
    from lithe.bundles.workspace import Workspace
    ws = Workspace(tmp_path)
    ws.write_bytes("big.png", _PNG_MIN + b"\x00" * (_PROBE_WINDOW + 65536))
    reg = ToolRegistry()
    register_image_tools(reg, lambda ctx: ws)
    res = await reg.dispatch("image_info", {"path": "big.png"}, _ctx())
    assert res.ok
    assert "PNG" in res.content and "4×3" in res.content


async def test_analyze_image_oversize_refused_by_stat(tmp_path):
    from lithe.bundles.images import register_image_tools
    from lithe.bundles.workspace import Workspace
    ws = Workspace(tmp_path)
    ws.write_bytes("huge.png", _PNG_MIN + b"\x00" * 1024)  # 先写小文件
    reg = ToolRegistry()
    read_bytes_calls = {"n": 0}
    orig_read = Path.read_bytes

    def counting_read(self):
        read_bytes_calls["n"] += 1
        return orig_read(self)

    register_image_tools(
        reg, lambda ctx: ws, max_image_bytes=8,
        llm_config=LLMConfig(model="m", base_url="x", api_key="k"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "read_bytes", counting_read)
        res = await reg.dispatch("analyze_image",
                                 {"path": "huge.png", "question": "?"}, _ctx())
    assert not res.ok and "超过上限" in res.content
    assert read_bytes_calls["n"] == 0, "超限应先 stat 拒绝，不整读文件"


# --------------------------------------------------------------------------- #
# host: DictToolAdapter / capture_actions
# --------------------------------------------------------------------------- #

def test_dict_adapter_warns_on_missing_handler(caplog):
    from lithe.bundles.host import DictToolAdapter
    async def h(ctx, args):
        return {"ok": True, "summary": "s"}

    with caplog.at_level("WARNING"):
        adapter = DictToolAdapter(
            [{"type": "function", "function": {
                "name": "typo-tool", "parameters": {}}}],
            {"real_tool": h})
    assert "typo-tool" in caplog.text
    assert adapter.names() == []


def test_dict_adapter_reverters_map():
    from lithe.bundles.host import DictToolAdapter

    async def h(ctx, args):
        return {"ok": True, "summary": "s"}

    def rev(action, ctx):
        pass

    adapter = DictToolAdapter(
        [{"type": "function", "function": {"name": "t", "parameters": {}}}],
        {"t": h}, reverters={"custom_kind": rev})
    assert adapter.reverters().get("custom_kind") is rev


async def test_host_capture_actions_false(tmp_path):
    from lithe.bundles import AgentHost, JsonlRunStore
    ws, reg = _ws_reg(tmp_path)
    store = JsonlRunStore(tmp_path / "store")
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=_FakeTransport([
                                        _resp(tool_calls=[_tc(
                                            "write_file",
                                            {"path": "f.txt",
                                             "content": "x"}, "c1")]),
                                        _resp("done")])),
                     store, capture_actions=False)
    ctx = _ctx()
    _ = [e async for e in host.run(ctx, "write it")]
    assert store.list_actions(ctx.run_id, ctx.user_id) == [], \
        "capture_actions=False 时 StoreSink 不得再记 file_change 动作行"


# --------------------------------------------------------------------------- #
# skills: deprecated alias + refresh cleanup
# --------------------------------------------------------------------------- #

def test_deprecated_skills_alias_warns():
    with pytest.warns(DeprecationWarning):
        importlib.import_module("lithe.skills")


def test_remote_refresh_failure_cleans_staging(tmp_path, caplog):
    from lithe.bundles.skills import RemoteSkillSource

    class _BoomClient:
        def get(self, url, timeout=None):
            raise httpx.ConnectError("down")

        def close(self):
            pass

    src = RemoteSkillSource("http://down.example/", tmp_path / "cache",
                            ttl=0, client=_BoomClient())
    with caplog.at_level("DEBUG"):
        src.refresh()
    assert not (tmp_path / "cache.staging").exists(), "失败不得遗留 staging"
    assert not (tmp_path / "cache").exists()


def _resp_ok(json_body=None, text=None, url="http://x/"):
    kw = {"json": json_body} if json_body is not None else {"text": text}
    return httpx.Response(200, request=httpx.Request("GET", url), **kw)


def test_remote_refresh_signature_not_committed_on_failure(tmp_path):
    from lithe.bundles.skills import RemoteSkillSource

    good = {"skills": [{"name": "a", "package": "p", "version": "1"}]}
    body = "# skill a\n"

    class _HalfClient:
        def get(self, url, timeout=None):
            if "index.json" in url:
                return _resp_ok(json_body=good, url=url)
            raise httpx.ConnectError("mid-download failure")

        def close(self):
            pass

    class _OkClient:
        def get(self, url, timeout=None):
            if "index.json" in url:
                return _resp_ok(json_body=good, url=url)
            return _resp_ok(text=body, url=url)

        def close(self):
            pass

    src = RemoteSkillSource("http://x/", tmp_path / "c2", ttl=0,
                            client=_HalfClient())
    src.refresh()  # 半途失败：缓存不换，签名不提交
    assert not (tmp_path / "c2").exists()
    assert src._index_sig is None
    # 之后服务器恢复：同一 index 必须仍然触发完整刷新（不得因错误的
    # 签名提交而短路）
    src2 = RemoteSkillSource("http://x/", tmp_path / "c3", ttl=0,
                             client=_OkClient())
    src2.refresh()
    assert (tmp_path / "c3" / "p" / "a.md").read_text(encoding="utf-8") == body


# --------------------------------------------------------------------------- #
# store: fsync option exists (smoke)
# --------------------------------------------------------------------------- #

def test_store_fsync_option_roundtrip(tmp_path):
    from lithe.bundles import JsonlRunStore
    store = JsonlRunStore(tmp_path, fsync=True)
    store.create_run("r", "u", "t")
    assert store.get_run("r", "u").task == "t"
