"""Subagent bundle: isolated worker agents + the delegate tool. The LLM is
mocked; the write tool logs mutations to a JsonlRunStore tagged with the current
subagent so the engine's high-water snapshot + summary can be exercised."""
from __future__ import annotations

import json

from lithe import (
    AgentContext, EventType, LLMConfig, ToolCategory, ToolRegistry, ToolResult,
    ToolSpec,
)
from lithe.bundles import (
    AgentHost, JsonlRunStore, SubagentEngine, SubagentRoster, SubagentSpec,
    make_delegate_tool, register_delegate_tool,
)


def _tc(name, args, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _chat_transport(responses):
    """Fake transport yielding chat-shaped responses, parsed to transport shape."""
    it = iter(list(responses))

    class _T:
        async def complete(self, client, **kw):
            data = next(it)
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            return {"content": msg.get("content") or "",
                    "tool_calls": msg.get("tool_calls") or [], "usage": {}}
    return _T()


def _make_writer(store):
    """A write tool that logs a mutation tagged with the active subagent."""
    async def write(ctx, args):
        path = args["path"]
        content = args.get("content", "")
        store.log_action(ctx.run_id, ctx.user_id, "file_write", path, None,
                         content, status="applied", subagent=ctx.subagent)
        return ToolResult(True, f"wrote {path}", f"wrote {path}",
                          ui=[{"type": "file_change", "path": path}])
    return write


async def _noop(ctx, args):
    return ToolResult(True, "ok", "ok")


async def test_llm_for_preserves_host_policy_fields(tmp_path):
    """子代理的 LLM 配置必须继承 host 的全部策略字段（stream/context_window/
    重试策略…）——手抄字段的历史 bug 是每加一个字段就漏一个。"""
    store = JsonlRunStore(tmp_path)
    host = AgentHost(ToolRegistry(),
                     LLMConfig(model="host-model", base_url="x", api_key="k",
                               stream=True, context_window=128_000,
                               attempts=4, sleep_429=2.0, sleep_err=1.0),
                     store)
    engine = SubagentEngine(host, SubagentRoster([
        SubagentSpec(id="w", display="writer", description="d", tools=[],
                     prompt="p"),
        SubagentSpec(id="m", display="model-override", description="d",
                     tools=[], prompt="p", model="stronger-model"),
    ]))
    base = engine._llm_for(engine.roster.get("w"))
    assert base.model == "host-model" and base.stream is True
    assert base.context_window == 128_000
    assert base.attempts == 4 and base.sleep_429 == 2.0 and base.sleep_err == 1.0
    over = engine._llm_for(engine.roster.get("m"))
    assert over.model == "stronger-model" and over.stream is True
    assert over.context_window == 128_000, "覆盖 model 也不能丢策略字段"


def _build(tmp_path, *, transport=None, label_fn=None, **kw):
    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("write", "w", category=ToolCategory.WRITE),
                 _make_writer(store))
    reg.register(ToolSpec("read", "r", category=ToolCategory.READ), _noop)
    host = AgentHost(reg,
                     LLMConfig(model="m", base_url="x", api_key="k", transport=transport),
                     store, max_steps=5)
    roster = SubagentRoster(
        [SubagentSpec("writer", "写者", "写字", ["write"], "你是写者。")])
    engine = SubagentEngine(host, roster, label_fn=label_fn, **kw)
    return store, reg, host, roster, engine


# --------------------------------------------------------------------------- #
# engine.run: tagging + high-water snapshot + summary
# --------------------------------------------------------------------------- #
async def test_capture_actions_false_leaves_domain_logging_to_the_host(tmp_path):
    """Hosts whose tools log their own domain actions (custom undo kinds a
    generic file_change capture cannot know — the thesis app pattern) build
    the engine with capture_actions=False. The subagent run's StoreSink then
    records messages but does not ALSO capture the tool's file_change ui into
    a second action row; the default (True) keeps the legacy double for hosts
    that rely on auto-capture."""
    def make_writer(store):
        async def write(ctx, args):
            store.log_action(ctx.run_id, ctx.user_id, "file_write",
                             args["path"], None, args.get("content", ""),
                             status="applied", subagent=ctx.subagent)
            return ToolResult(True, "wrote", "wrote",
                              ui=[{"type": "file_change", "action": "write",
                                   "path": args["path"], "old": None,
                                   "new": args.get("content", "")}])
        return write

    def build(root):
        store = JsonlRunStore(root)
        reg = ToolRegistry()
        reg.register(ToolSpec("write", "w", category=ToolCategory.WRITE),
                     make_writer(store))
        transport = _chat_transport([
            {"choices": [{"message": {"content": "",
                                      "tool_calls": [_tc("write", {"path": "d.txt", "content": "x"}, cid="w1")]}}]},
            {"choices": [{"message": {"content": "done"}}]},
        ])
        host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                        transport=transport),
                         store, max_steps=5)
        roster = SubagentRoster(
            [SubagentSpec("writer", "写者", "写字", ["write"], "你是写者。")])
        return store, host, roster

    # default: self-log + auto-capture = two rows (legacy behavior kept)
    store, host, roster = build(tmp_path / "on")
    store.create_run("r1", "u1", "t")
    await SubagentEngine(host, roster).run(
        "writer", "write", AgentContext(run_id="r1", user_id="u1"))
    assert len(store.list_actions("r1", "u1")) == 2

    # capture_actions=False: only the host's own domain row
    store, host, roster = build(tmp_path / "off")
    store.create_run("r2", "u1", "t")
    await SubagentEngine(host, roster, capture_actions=False).run(
        "writer", "write", AgentContext(run_id="r2", user_id="u1"))
    rows = store.list_actions("r2", "u1")
    assert len(rows) == 1 and rows[0].subagent.startswith("writer:")
    # messages are still recorded (tagged) — capture off affects actions only
    assert store.messages_for_run("r2", "u1")


async def test_engine_run_tags_and_summarizes(tmp_path):
    transport = _chat_transport([
        {"choices": [{"message": {"content": "",
                                  "tool_calls": [_tc("write", {"path": "b.txt", "content": "hi"}, cid="w1")]}}]},
        {"choices": [{"message": {"content": "wrote b.txt"}}]},
    ])
    store, _, host, roster, engine = _build(tmp_path, transport=transport,
                                            label_fn=lambda a: f"写入 {a['target']}")
    store.create_run("r1", "u1", "t")
    ctx = AgentContext(run_id="r1", user_id="u1")

    stats, sub_actions = await engine.run("writer", "write b.txt", ctx)
    assert stats.status == "done" and stats.final_text == "wrote b.txt"
    assert len(sub_actions) == 1 and sub_actions[0].subagent.startswith("writer:")

    s = engine.summarize(roster.get("writer"), stats, sub_actions)
    assert "写者" in s and "写入 b.txt" in s

    # subagent messages recorded under the parent run, tagged with the
    # delegation instance ("<agent>:<hex8>")
    all_msgs = store.messages_for_runs(["r1"], "u1", exclude_subagent=False)
    orch_msgs = store.messages_for_runs(["r1"], "u1", exclude_subagent=True)
    assert len(all_msgs) > len(orch_msgs)
    assert all(m["subagent"].startswith("writer:")
               for m in all_msgs if m["subagent"])


async def test_engine_high_water_not_recounted(tmp_path):
    transport = _chat_transport([
        {"choices": [{"message": {
            "content": "", "tool_calls": [_tc("write", {"path": "a", "content": "x"})]}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {"content": "nothing more"}}]},
    ])
    store, _, host, roster, engine = _build(tmp_path, transport=transport)
    store.create_run("r1", "u1", "t")
    ctx = AgentContext(run_id="r1", user_id="u1")

    _, first = await engine.run("writer", "first", ctx)
    assert len(first) == 1

    # second delegation does no new work → its own actions are empty (not recounted)
    _, second = await engine.run("writer", "again", ctx)
    assert second == []


async def test_engine_high_water_with_opaque_string_ids(tmp_path):
    """Action.id 契约是 opaque/host-assigned：字符串 id 的 store 也必须正确
    区分“本次委派的新动作”（旧的数值比较会做字典序比较而漏计/误计）。"""
    from dataclasses import replace

    from lithe.bundles import JsonlRunStore as _JRS

    class _StrIdStore(_JRS):
        def list_actions(self, run_id, user_id, *, status_in=(),
                         subagent=None):
            # 稳定的字符串 id（底层持久 int id 派生），模拟 opaque-id store
            return [replace(a, id=f"a-{a.id}" if a.id is not None else None)
                    for a in super().list_actions(run_id, user_id,
                                                  status_in=status_in,
                                                  subagent=subagent)]

    transport = _chat_transport([
        {"choices": [{"message": {
            "content": "", "tool_calls": [_tc("write", {"path": "a", "content": "x"})]}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {
            "content": "", "tool_calls": [_tc("write", {"path": "b", "content": "y"})]}}]},
        {"choices": [{"message": {"content": "done again"}}]},
    ])
    store = _StrIdStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("write", "w", category=ToolCategory.WRITE),
                 _make_writer(store))
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=transport), store)
    engine = SubagentEngine(host, SubagentRoster(
        [SubagentSpec("writer", "写者", "写字", ["write"], "你是写者。")]))
    store.create_run("r1", "u1", "t")
    ctx = AgentContext(run_id="r1", user_id="u1")

    _, first = await engine.run("writer", "first", ctx)
    assert len(first) == 1 and isinstance(first[0].id, str)
    _, second = await engine.run("writer", "again", ctx)
    assert len(second) == 1 and second[0].target == "b", \
        "第二次委派的新动作按 id 集合成员关系识别，而非数值比较"


async def test_failed_delegation_summary_carries_reason(tmp_path):
    """失败的委派必须把失败原因带回编排者的工具结果：只写“运行失败”时，
    模型无法区分瞬时网关错误（值得重试一次）与任务本身无法执行——这正是
    会话存储里“1 成功 1 失败”之后无从排查的原因。"""
    import httpx

    class _NetDown:
        async def complete(self, client, **kw):
            raise httpx.ConnectError("upstream refused")

    store, reg, host, roster, engine = _build(tmp_path, transport=_NetDown())
    store.create_run("r1", "u1", "t")
    ctx = AgentContext(run_id="r1", user_id="u1")

    stats, sub_actions = await engine.run("writer", "write something", ctx)
    assert stats.status == "failed" and sub_actions == []
    assert stats.error == "ConnectError: upstream refused"
    s = engine.summarize(roster.get("writer"), stats, sub_actions)
    assert "运行失败" in s
    assert "失败原因：ConnectError: upstream refused" in s
    r = await engine.delegate({"agent": "writer", "task": "t"}, ctx)
    assert r.ok is False
    assert "ConnectError: upstream refused" in r.content

    # empty_response keeps its own honest head (not 已完成) + the reason
    from lithe import RunStats as _RS
    s2 = engine.summarize(roster.get("writer"), _RS(status="empty_response",
                                                    error="model returned nothing"), [])
    assert "未返回任何结果" in s2 and "失败原因" in s2


async def test_subagent_cancellation_propagates_from_parent_run(tmp_path):
    """编排者的 stop 句柄（run 期间写入 ctx.shared）必须同样终止子代理：
    取消后子代理不再调用模型，状态如实标记。"""
    import asyncio

    class _Never:
        async def complete(self, client, **kw):
            raise AssertionError("取消后子代理的模型不应被调用")

    store, reg, host, roster, engine = _build(tmp_path, transport=_Never())
    ctx = AgentContext(run_id="r1", user_id="u1")
    store.create_run("r1", "u1", "t")
    stop = asyncio.Event()
    stop.set()
    ctx.shared["_runtime_stop"] = stop

    stats, sub_actions = await engine.run("writer", "write something", ctx)
    assert stats.status == "cancelled"
    assert sub_actions == []
    s = engine.summarize(roster.get("writer"), stats, sub_actions)
    assert "取消" in s
    r = await engine.delegate({"agent": "writer", "task": "t"}, ctx)
    assert r.ok is False, "被取消的委派不能对模型呈现为成功"


# --------------------------------------------------------------------------- #
# delegate tool end-to-end (orchestrator → delegate → subagent → final)
# --------------------------------------------------------------------------- #
async def test_delegate_tool_end_to_end(tmp_path):
    transport = _chat_transport([
        {"choices": [{"message": {"content": "",
                                  "tool_calls": [_tc("delegate", {"agent": "writer", "task": "write b.txt=hi"}, cid="d1")]}}]},
        {"choices": [{"message": {"content": "",
                                  "tool_calls": [_tc("write", {"path": "b.txt", "content": "hi"}, cid="w1")]}}]},
        {"choices": [{"message": {"content": "wrote b.txt"}}]},
        {"choices": [{"message": {"content": "delegation complete"}}]},
    ])
    store, reg, host, roster, engine = _build(tmp_path, transport=transport,
                                              label_fn=lambda a: f"写入 {a['target']}")
    register_delegate_tool(reg, engine)

    events = [e async for e in host.run(AgentContext(run_id="r1", user_id="u1"),
                                        "please delegate writing b.txt")]
    assert events[-1]["type"] == EventType.DONE and events[-1]["status"] == "done"

    # the subagent's mutation is recorded under the parent run, tagged with
    # its delegation instance ("<agent>:<hex8>" — never the bare roster id)
    acts = store.list_actions("r1", "u1")
    assert any(a.kind == "file_write" and a.subagent.startswith("writer:")
               for a in acts)

    # the delegate tool result carries subagent_start/subagent_end + summary
    all_msgs = store.messages_for_runs(["r1"], "u1", exclude_subagent=False)
    delegate_msg = [m for m in all_msgs if m["role"] == "tool"
                    and m["tool_name"] == "delegate"]
    assert len(delegate_msg) == 1
    dm = delegate_msg[0]
    assert "写者" in dm["content"] and "写入 b.txt" in dm["content"]
    ui = dm["meta"]["ui"]
    assert any(e["type"] == "subagent_start" for e in ui)
    assert any(e["type"] == "subagent_end" and e["ok"] for e in ui)


# --------------------------------------------------------------------------- #
# delegate validation + enabled gate
# --------------------------------------------------------------------------- #
async def test_delegate_validation_and_gate(tmp_path):
    store, _, host, roster, engine = _build(tmp_path)
    ctx = AgentContext(run_id="r1", user_id="u1")

    r = await engine.delegate({"agent": "ghost", "task": "x"}, ctx)
    assert not r.ok and "未知子代理" in r.content

    r = await engine.delegate({"agent": "writer", "task": ""}, ctx)
    assert not r.ok and "task" in r.content

    engine.enabled = False
    r = await engine.delegate({"agent": "writer", "task": "x"}, ctx)
    assert not r.ok and "未启用" in r.content


# --------------------------------------------------------------------------- #
# trimmed_tools + skill injection (pure)
# --------------------------------------------------------------------------- #
def test_trimmed_tools_excludes_delegate_and_disabled(tmp_path):
    store, _, host, roster, engine = _build(tmp_path)
    spec = SubagentSpec("w", "写者", "d", ["write", "read", "delegate"], "p")
    tools = engine.trimmed_tools(spec, frozenset({"read"}))
    names = {t["function"]["name"] for t in tools}
    assert names == {"write"}  # delegate dropped (no recursion), disabled read dropped


def test_skill_injector_inlines_into_prompt(tmp_path):
    store, _, host, _, _ = _build(tmp_path)
    roster = SubagentRoster([SubagentSpec("w", "写者", "d", ["write"], "BODY",
                                          auto_skills=["s1", "s2"])])
    engine = SubagentEngine(host, roster,
                            skill_injector=lambda names, ctx: "SKILL:" + ",".join(names))
    prompt = engine.system_prompt(roster.get("w"), AgentContext(run_id="r", user_id="u"))
    assert "BODY" in prompt and "SKILL:s1,s2" in prompt


def test_roster_text_and_lookup(tmp_path):
    store, _, host, _, _ = _build(tmp_path)
    roster = SubagentRoster([
        SubagentSpec("a", "A", "does a", ["write"], "p"),
        SubagentSpec("b", "B", "does b", ["read"], "p"),
    ])
    assert roster.ids() == ["a", "b"]
    assert "does a" in roster.text() and "does b" in roster.text()
    assert "a" in roster and roster.get("z") is None


# --------------------------------------------------------------------------- #
# delegate_parallel: fan-out, failure isolation, progress callback
# --------------------------------------------------------------------------- #
async def test_delegate_parallel_runs_subagents_concurrently(tmp_path):
    """两个子代理的工具互等对方启动：串行会死锁，并行必须在限时内完成。"""
    import asyncio

    flags = {"a": asyncio.Event(), "b": asyncio.Event()}

    def _flag_tool(name, flag, other):
        async def h(ctx, args):
            flag.set()
            await other.wait()
            return ToolResult(True, name, f"{name} done")
        return ToolSpec(name, name, category=ToolCategory.READ), h

    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    for spec, h in (_flag_tool("flag_a", flags["a"], flags["b"]),
                    _flag_tool("flag_b", flags["b"], flags["a"])):
        reg.register(spec, h)

    def _sub_transport(final: str):
        return _chat_transport([
            {"choices": [{"message": {"content": "",
                                      "tool_calls": [_tc(
                                          "flag" + final, {}, cid="x1")]}}]},
            {"choices": [{"message": {"content": final}}]},
        ])

    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=_sub_transport("a")), store)
    roster = SubagentRoster([
        SubagentSpec("sa", "A", "does a", ["flag_a"], "p",
                     transport=_sub_transport("a")),
        SubagentSpec("sb", "B", "does b", ["flag_b"], "p",
                     transport=_sub_transport("b")),
    ])
    engine = SubagentEngine(host, roster)
    ctx = AgentContext(run_id="r1", user_id="u1")
    store.create_run("r1", "u1", "t")

    async def call():
        return await engine.delegate(
            {"tasks": [{"agent": "sa", "task": "a"},
                       {"agent": "sb", "task": "b"}]}, ctx)

    from lithe.bundles import make_parallel_delegate_tool
    spec, handler = make_parallel_delegate_tool(engine)
    res = await asyncio.wait_for(handler(ctx, {"tasks": [
        {"agent": "sa", "task": "a"}, {"agent": "sb", "task": "b"}]}), 3.0)
    assert res.ok, "两个互等的子代理都完成了——真正并行"
    assert "2 成功" in res.summary
    assert "### sa（成功）" in res.content and "### sb（成功）" in res.content
    starts = [e for e in res.ui if e["type"] == "subagent_start"]
    ends = [e for e in res.ui if e["type"] == "subagent_end"]
    assert {e["agent"] for e in starts} == {"sa", "sb"} and len(ends) == 2


async def test_delegate_parallel_failure_isolated(tmp_path):
    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("read", "r", category=ToolCategory.READ), _noop)

    class _Boom:
        async def complete(self, client, **kw):
            raise RuntimeError("subagent model down")

    ok_t = _chat_transport([{"choices": [{"message": {"content": "fine"}}]}])
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=ok_t), store)
    roster = SubagentRoster([
        SubagentSpec("good", "G", "works", ["read"], "p", transport=ok_t),
        SubagentSpec("bad", "B", "fails", ["read"], "p", transport=_Boom()),
    ])
    engine = SubagentEngine(host, roster)
    ctx = AgentContext(run_id="r1", user_id="u1")
    store.create_run("r1", "u1", "t")

    from lithe.bundles import make_parallel_delegate_tool
    _, handler = make_parallel_delegate_tool(engine)
    res = await handler(ctx, {"tasks": [{"agent": "bad", "task": "x"},
                                        {"agent": "good", "task": "y"}]})
    assert res.ok is False and "1 成功" in res.summary
    assert "### bad（失败）" in res.content and "### good（成功）" in res.content


async def test_delegate_parallel_crash_isolated(tmp_path):
    """delegate 内部真异常（如 store 故障，而非失败的 ToolResult）也要隔离：
    报告为该 agent 的失败块，不炸掉整步、不把兄弟任务丢成无人 await 的孤儿。"""
    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("read", "r", category=ToolCategory.READ), _noop)

    ok_t = _chat_transport([{"choices": [{"message": {"content": "fine"}}]}])
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=ok_t), store)
    roster = SubagentRoster([
        SubagentSpec("bad", "B", "crashes", ["read"], "p"),
        SubagentSpec("good", "G", "works", ["read"], "p", transport=ok_t),
    ])
    engine = SubagentEngine(host, roster)
    ctx = AgentContext(run_id="r1", user_id="u1")
    store.create_run("r1", "u1", "t")

    real_run = engine.run

    async def booming_run(sub_id, task, parent_ctx, grounding="", **kw):
        if sub_id == "bad":
            raise RuntimeError("store exploded")
        return await real_run(sub_id, task, parent_ctx, grounding=grounding,
                              **kw)

    engine.run = booming_run

    from lithe.bundles import make_parallel_delegate_tool
    _, handler = make_parallel_delegate_tool(engine)
    res = await handler(ctx, {"tasks": [{"agent": "bad", "task": "x"},
                                        {"agent": "good", "task": "y"}]})
    assert res.ok is False and "1 成功" in res.summary
    assert "### bad（失败）" in res.content and "store exploded" in res.content
    assert "### good（成功）" in res.content and "fine" in res.content


async def test_delegate_parallel_validates_batch_and_agents(tmp_path):
    store, reg, host, roster, engine = _build(tmp_path)
    from lithe.bundles import make_parallel_delegate_tool
    _, handler = make_parallel_delegate_tool(engine)
    ctx = AgentContext(run_id="r1", user_id="u1")

    bad_agent = await handler(ctx, {"tasks": [{"agent": "ghost", "task": "x"}]})
    assert bad_agent.ok is False and "未知子代理" in bad_agent.content

    too_many = await handler(ctx, {"tasks": [{"agent": "writer", "task": f"t{i}"}
                                              for i in range(9)]})
    assert too_many.ok is False and "最多并行委派" in too_many.content

    empty = await handler(ctx, {"tasks": []})
    assert empty.ok is False


async def test_delegate_parallel_same_agent_tasks_run_concurrently(tmp_path):
    """同一子代理出现在 tasks 多项：批次整体成功，且这些任务真正并发执行
    （模型把多项只读审查都派给同一个 agent 是自然行为——串行会让
    “并行委派”名存实亡）。并发安全性来自按委派实例记账：每个任务拿到
    唯一实例标签（"<agent>:<hex8>"），消息、动作归属与预算槽都键在实例上，
    互不认领。"""
    import asyncio

    state = {"inflight": 0, "max_inflight": 0, "probe_log": []}

    async def probe(ctx, args):
        # ctx.subagent 是本次委派的实例标签：并发同名任务各自独立
        state["inflight"] += 1
        state["max_inflight"] = max(state["max_inflight"],
                                    state["inflight"])
        await asyncio.sleep(0.08)  # 足以让并发同类任务的模型调用重叠
        state["inflight"] -= 1
        state["probe_log"].append(ctx.subagent)
        return ToolResult(True, "probe", "probe done")

    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    reg.register(ToolSpec("probe", "p", category=ToolCategory.READ), probe)

    class _ProbeOnce:
        """每次委派恰好一次模型调用（max_steps=1）：返回一个 probe 调用。
        共享一个实例也无需按调用方区分——响应形状对每个委派相同。"""

        async def complete(self, client, **kw):
            state["inflight"] += 1
            state["max_inflight"] = max(state["max_inflight"],
                                        state["inflight"])
            await asyncio.sleep(0.08)
            state["inflight"] -= 1
            return {"content": "", "tool_calls": [_tc("probe", {})],
                    "usage": {}}

    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=_ProbeOnce()), store)
    roster = SubagentRoster([
        SubagentSpec("sa", "A", "does a", ["probe"], "p", max_steps=1),
    ])
    engine = SubagentEngine(host, roster)
    ctx = AgentContext(run_id="r1", user_id="u1")
    store.create_run("r1", "u1", "t")

    from lithe.bundles import make_parallel_delegate_tool
    _, handler = make_parallel_delegate_tool(engine)
    res = await asyncio.wait_for(handler(ctx, {"tasks": [
        {"agent": "sa", "task": "one"}, {"agent": "sa", "task": "two"}]}), 5.0)
    assert res.ok, "同名两任务并发执行，批次整体成功"
    assert "2 成功" in res.summary
    assert res.content.count("### sa（成功）") == 2
    # 真并发：两个委派的模型调用在时间上重叠（串行实现只会得到 1）
    assert state["max_inflight"] == 2, (
        f"同名任务的模型调用应重叠运行，max_inflight={state['max_inflight']}")
    # 实例标签两两不同，且都是 sa 的实例
    tags = state["probe_log"]
    assert len(tags) == 2 and len(set(tags)) == 2
    assert all(t.startswith("sa:") for t in tags)
    # ui 事件携带 instance，前端能区分同名并行实例
    instances = [e["instance"] for e in
                 (res.ui) if e.get("type") == "subagent_start"]
    assert len(set(instances)) == 2 and all(i.startswith("sa:") for i in instances)


async def test_parallel_same_agent_delegations_never_cross_claim(tmp_path):
    """并发的同名委派各记各的动作：每个实例的 sub_actions 只含自己那次
    写入（身份键实现里这是强制串行的理由——实例化后约束自然成立）。"""
    import asyncio

    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()

    def make_writer():
        async def write(ctx, args):
            return ToolResult(
                True, f"wrote {args['path']}", f"wrote {args['path']}",
                ui=[{"type": "file_change", "action": "write",
                     "path": args["path"], "old": None,
                     "new": f"content of {args['path']}"}])
        return write

    reg.register(ToolSpec("write_left", "wl", category=ToolCategory.WRITE),
                 make_writer())
    reg.register(ToolSpec("write_right", "wr", category=ToolCategory.WRITE),
                 make_writer())

    class _T:
        """奇数号调用写 left、偶数号写 right；哪个委派先到无所谓——
        断言只看归属，不看顺序。"""

        def __init__(self):
            self.lock = asyncio.Lock()
            self.calls = 0

        async def complete(self, client, **kw):
            async with self.lock:
                self.calls += 1
                n = self.calls
            await asyncio.sleep(0.05)  # 让两个委派的调用交错
            tool = "write_left" if n % 2 == 1 else "write_right"
            return {"content": "", "tool_calls": [_tc(tool, {"path": tool[6:]}),
                                             ], "usage": {}}

    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=_T()), store)
    engine = SubagentEngine(host, SubagentRoster([
        SubagentSpec("w", "W", "d", ["write_left", "write_right"], "p",
                     max_steps=1)]))
    ctx = AgentContext(run_id="r1", user_id="u1")
    store.create_run("r1", "u1", "t")

    (s1, a1), (s2, a2) = await asyncio.gather(
        engine.run("w", "left job", ctx),
        engine.run("w", "right job", ctx))
    # 两个独立实例，各恰好认领自己那一次写入
    assert len(a1) == 1 and len(a2) == 1
    assert a1[0].subagent != a2[0].subagent
    assert a1[0].subagent.startswith("w:") and a2[0].subagent.startswith("w:")
    assert {a1[0].target, a2[0].target} == {"left", "right"}


async def test_subagent_progress_callback_gets_live_events(tmp_path):
    transport = _chat_transport([
        {"choices": [{"message": {"content": "",
                                  "tool_calls": [_tc("write", {"path": "p", "content": "c"}, cid="w1")]}}]},
        {"choices": [{"message": {"content": "finished writing"}}]},
    ])
    progress: list[dict] = []

    async def on_event(sub_ctx, ev):
        progress.append(ev)

    store, _, host, roster, engine = _build(tmp_path, transport=transport,
                                            on_subagent_event=on_event)
    ctx = AgentContext(run_id="r1", user_id="u1")
    store.create_run("r1", "u1", "t")
    await engine.run("writer", "write p", ctx)

    kinds = [(p["agent"], p["event"]["type"]) for p in progress]
    assert all(a == "writer" for a, _ in kinds)
    assert "step" in [k for _, k in kinds]
    assert "tool_call" in [k for _, k in kinds]
    assert "tool_result" in [k for _, k in kinds]
    # 文本类事件有但被截断保护
    texts = [p["event"]["text"] for p in progress
             if p["event"]["type"] == "assistant"]
    assert texts == ["finished writing"]


async def test_delegation_spend_folds_into_done_event(tmp_path):
    """子代理花费必须并入父 run 的 done/统计/store 行——旧实现 done 只报
    编排器自身，20 次委派的 run 成本被低估一个量级。"""
    class _SubTransport:
        """子代理专用 transport：直接给最终答案，带 usage/cost。"""

        async def complete(self, client, **kw):
            return {"content": "subagent done", "tool_calls": [],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                              "total_tokens": 150, "cost": 0.01}}

    class _OrchTransport:
        def __init__(self):
            self.calls = 0

        async def complete(self, client, **kw):
            self.calls += 1
            if self.calls == 1:
                return {"content": "", "tool_calls": [_tc(
                    "delegate", {"agent": "writer", "task": "写点东西"})],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15}}
            return {"content": "orchestrator done", "tool_calls": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15}}

    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    orch = _OrchTransport()
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=orch), store, max_steps=4)
    engine = SubagentEngine(host, SubagentRoster([
        SubagentSpec("writer", "写者", "写", [], "p", transport=_SubTransport()),
    ]))
    spec, handler = make_delegate_tool(engine)
    reg.register(spec, handler)

    events = [e async for e in host.run(AgentContext(run_id="r1", user_id="u1"),
                                        "delegate something")]
    done = events[-1]
    assert done["type"] == EventType.DONE and done["status"] == "done"
    assert done["subagent_delegations"] == 1
    assert done["subagent_cost"] == 0.01 and done["subagent_tokens"] == 150
    # 总账包含子代理：编排器自身 30 tokens + 子代理 150
    assert done["tokens"] == 30 + 150
    # store 行同样落全量
    assert store.get_run("r1", "u1").cost == 0.01
