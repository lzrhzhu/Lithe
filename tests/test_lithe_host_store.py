"""Round-2 bundles: storage (JsonlRunStore), host adapter (AgentHost/undo_run/
DictToolAdapter/assemble_messages), and the new memory transforms
(window_with_recap / run_timeline). The LLM is mocked; stores use tmp_path —
nothing touches real data."""
from __future__ import annotations

import json

import pytest

from lithe import (
    AgentContext, EventType, LLMConfig, ToolCategory, ToolRegistry, ToolResult,
    ToolSpec, run_timeline, window_with_recap,
)
from lithe.bundles import (
    AgentHost, DictToolAdapter, JsonlRunStore, StoredMessage,
    assemble_messages, undo_run,
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


class _BoomTransport:
    async def complete(self, client, **kw):
        raise RuntimeError("boom")


# --------------------------------------------------------------------------- #
# JsonlRunStore
# --------------------------------------------------------------------------- #
def test_jsonl_store_run_message_action_roundtrip(tmp_path):
    s = JsonlRunStore(tmp_path)
    s.create_run("r1", "u1", "do task", conversation_id=5, model="m")
    s.add_message(StoredMessage(role="user", content="hi", run_id="r1", user_id="u1"))
    s.add_message(StoredMessage(role="assistant", content="hello", run_id="r1",
                                user_id="u1", tool_calls=[{"id": "c1"}]))
    aid = s.log_action("r1", "u1", "file_write", "a.txt", None, "data",
                       status="applied")
    s.finish_run("r1", "done", 2, 0.001, "hello")

    run = s.get_run("r1", "u1")
    assert run.status == "done" and run.final == "hello"
    assert run.steps == 2 and run.cost == 0.001 and run.conversation_id == 5

    msgs = s.messages_for_run("r1", "u1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["tool_calls"] == [{"id": "c1"}]

    acts = s.list_actions("r1", "u1")
    assert len(acts) == 1 and acts[0].kind == "file_write"

    s.set_action_status(aid, "u1", "reverted")
    assert s.get_action(aid, "u1").status == "reverted"
    assert s.list_actions("r1", "u1", status_in=("applied",)) == []


def test_jsonl_store_multi_user_isolation(tmp_path):
    s = JsonlRunStore(tmp_path)
    s.create_run("r1", "u1", "t")
    s.log_action("r1", "u1", "file_write", "a", None, "x")
    assert s.get_run("r1", "u2") is None
    assert s.list_actions("r1", "u2") == []
    assert s.messages_for_run("r1", "u2") == []


def test_jsonl_store_conversations(tmp_path):
    s = JsonlRunStore(tmp_path)
    c = s.create_conversation("u1", "t1")
    assert s.get_conversation(c["id"], "u1")["title"] == "t1"
    s.rename_conversation(c["id"], "u1", "t2")
    assert s.get_conversation(c["id"], "u1")["title"] == "t2"
    other = s.create_conversation("u2", "other")
    assert s.get_conversation(other["id"], "u1") is None  # cross-user hidden
    assert len(s.list_conversations("u1")) == 1
    assert s.delete_conversation(c["id"], "u1") == 1
    assert s.get_conversation(c["id"], "u1") is None
    assert s.list_conversations("u1") == []


def test_jsonl_store_blob_spillover(tmp_path):
    s = JsonlRunStore(tmp_path)
    payload = b"x" * 100
    ref = s.spill(payload)
    assert ref.startswith("sha256:") and s.load(ref) == payload
    assert s.spill(payload) == ref  # content-addressed: idempotent


# --------------------------------------------------------------------------- #
# AgentHost end-to-end (mocked LLM)
# --------------------------------------------------------------------------- #
def _registry(state):
    reg = ToolRegistry()

    async def write(ctx, args):
        state[args["path"]] = args["content"]
        return ToolResult(True, "wrote", f"wrote {args['path']}",
                          ui=[{"type": "file_change", "path": args["path"]}])

    async def read(ctx, args):
        return ToolResult(True, "read", state.get(args["path"], ""))

    reg.register(ToolSpec("write", "w", category=ToolCategory.WRITE), write,
                 reverter=lambda a, c: state.pop(a.target, None))
    reg.register(ToolSpec("read", "r", category=ToolCategory.READ), read)
    return reg


async def test_host_run_envelope_persistence_replay(tmp_path, monkeypatch):
    state: dict = {}
    store = JsonlRunStore(tmp_path)
    transport = _chat_transport([
        {"choices": [{"message": {"content": "",
                                  "tool_calls": [_tc("write", {"path": "a.txt", "content": "hi"})]}}]},
        {"choices": [{"message": {"content": "done"}}]},
    ])
    host = AgentHost(_registry(state),
                     LLMConfig(model="m", base_url="x", api_key="k", transport=transport),
                     store,
                     build_system_prompt=lambda ctx, mode, anchor: "SYS",
                     max_steps=5)

    ctx = AgentContext(run_id="r1", user_id="u1")
    stats: dict = {}
    events = [e async for e in host.run(ctx, "write a.txt=hi",
                                        history=[{"role": "user", "content": "prev"},
                                                 {"role": "assistant", "content": "ok"}],
                                        stats=stats)]

    assert events[0]["type"] == EventType.RUN_START
    assert events[-1]["type"] == EventType.DONE
    assert events[-1]["status"] == "done"
    assert state == {"a.txt": "hi"}
    assert stats["status"] == "done" and stats["final_text"] == "done"

    msgs = store.messages_for_run("r1", "u1")
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    run = store.get_run("r1", "u1")
    assert run.status == "done" and run.final == "done"

    # the recorded assistant tool_call round-trips through kernel replay
    from lithe import replay_messages
    replayed = replay_messages(msgs[1:])
    assert replayed[-1] == {"role": "assistant", "content": "done"}
    assert replayed[0]["tool_calls"][0]["id"] == "c1"


async def test_host_run_error_funneled(tmp_path, monkeypatch):
    store = JsonlRunStore(tmp_path)
    host = AgentHost(_registry({}),
                     LLMConfig(model="m", base_url="x", api_key="k",
                               transport=_BoomTransport()), store,
                     max_steps=3)

    events = [e async for e in host.run(AgentContext(run_id="r9", user_id="u1"), "go")]
    assert any(e["type"] == EventType.ERROR for e in events)
    assert events[-1]["type"] == EventType.DONE and events[-1]["status"] == "failed"
    assert store.get_run("r9", "u1").status == "failed"


async def test_host_run_abandoned_midstream_closes_run(tmp_path):
    """消费方中途断开（GeneratorExit/CancelledError 是 BaseException，
    旧实现绕过 except Exception）：run 不得永远停在 running——以
    abandoned 收尾、stats 同步更新，取消照常传播。"""
    store = JsonlRunStore(tmp_path)
    host = AgentHost(_registry({}),
                     LLMConfig(model="m", base_url="x", api_key="k",
                               transport=_chat_transport([])),
                     store)
    ctx = AgentContext(run_id="r20", user_id="u1")
    stats: dict = {}
    gen = host.run(ctx, "go", stats=stats)
    async for _ev in gen:
        break  # consumer disconnects right after run_start
    await gen.aclose()
    run = store.get_run("r20", "u1")
    assert run.status == "abandoned"
    assert stats["status"] == "abandoned"
    # user turn 在断开前已记录
    assert [m["role"] for m in store.messages_for_run("r20", "u1")] == ["user"]


async def test_done_event_carries_usage_breakdown(tmp_path):
    """DONE 事件带 token/上下文明细，host 前端可直接渲染用量。"""
    store = JsonlRunStore(tmp_path)

    class _UsageTransport:
        async def complete(self, client, **kw):
            return {"content": "done", "tool_calls": [],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 30,
                              "total_tokens": 150, "cost": 0.001}}

    host = AgentHost(_registry({}),
                     LLMConfig(model="m", base_url="x", api_key="k",
                               context_window=1000,
                               transport=_UsageTransport()), store)
    events = [e async for e in host.run(AgentContext(run_id="r10", user_id="u1"), "go")]
    done = events[-1]
    assert done["type"] == EventType.DONE
    assert done["prompt_tokens"] == 120 and done["completion_tokens"] == 30
    assert done["tokens"] == 150 and done["cost"] == 0.001
    assert done["context_tokens"] == 120 and done["context_window"] == 1000
    assert done["context_percent"] == 12.0
    # 运行中的 usage 事件也被转发，前端可实时显示
    assert any(e["type"] == "usage" for e in events)


async def test_host_extra_sinks_attached_to_run(tmp_path):
    """AgentHost(sinks=[...])：额外 sink（指标/动作捕获）随每个 run 挂载。"""
    store = JsonlRunStore(tmp_path)
    captured: list[dict] = []

    class _Capture:
        async def on_event(self, ctx, ev):
            captured.append(ev)

        async def on_record(self, ctx, record):
            pass

    class _Ok:
        async def complete(self, client, **kw):
            return {"content": "done", "tool_calls": [], "usage": {}}

    host = AgentHost(_registry({}),
                     LLMConfig(model="m", base_url="x", api_key="k",
                               transport=_Ok()), store, sinks=[_Capture()])
    events = [e async for e in host.run(AgentContext(run_id="r11", user_id="u1"), "go")]
    assert events[-1]["type"] == EventType.DONE
    # host 级 sink 看到完整生命周期：run_start … done
    assert [e["type"] for e in captured] == ["run_start", "step", "assistant",
                                             "usage", "done"]


# --------------------------------------------------------------------------- #
# undo_run
# --------------------------------------------------------------------------- #
async def test_undo_run_reverts_and_marks(tmp_path):
    store = JsonlRunStore(tmp_path)
    state = {"a.txt": "hi"}
    reg = ToolRegistry()

    async def noop(ctx, args):
        return ToolResult(True, "ok", "ok")

    reg.register(ToolSpec("noop", "n"), noop)
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k"), store)

    def rev_file(action, ctx):
        state.pop(action.target, None)

    store.create_run("r1", "u1", "t")
    store.log_action("r1", "u1", "file_write", "a.txt", None, "hi", status="applied")

    report = await undo_run(host, "r1", "u1", reverters={"file_write": rev_file})
    assert report.reverted == 1 and report.ok is True
    assert "a.txt" not in state
    assert store.list_actions("r1", "u1", status_in=("applied",)) == []  # marked reverted


async def test_undo_run_passes_extra_to_reverter_context(tmp_path):
    """A reverter reading a host field (e.g. workspace_for(ctx) needing
    thesis_id) must not KeyError — undo_run has to seed ctx.extra."""
    store = JsonlRunStore(tmp_path)
    seen: dict = {}
    reg = ToolRegistry()

    async def noop(ctx, args):
        return ToolResult(True, "ok", "ok")

    reg.register(ToolSpec("noop", "n"), noop)
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k"), store)

    def rev_file(action, ctx):
        seen["root"] = f"/data/{ctx['thesis_id']}"

    store.create_run("r2", "u1", "t")
    store.log_action("r2", "u1", "file_write", "a.txt", None, "hi", status="applied")

    report = await undo_run(host, "r2", "u1",
                      reverters={"file_write": rev_file},
                      extra={"thesis_id": "T9"})
    assert report.reverted == 1 and seen["root"] == "/data/T9"


def test_action_ids_increment_without_rescan(tmp_path):
    """id 计数器进程内递增；跨 store 实例继续（首个写时从文件播种）。"""
    store = JsonlRunStore(tmp_path)
    ids = [store.log_action("r", "u", "k", "t", None, None)
           for _ in range(50)]
    assert ids == list(range(1, 51)), "连续且无重复"

    reopened = JsonlRunStore(tmp_path)
    assert reopened.log_action("r", "u", "k", "t2", None, None) == 51
    assert len(store.list_actions("r", "u")) == 51

    conv_ids = [store.create_conversation("u", f"c{i}")["id"] for i in range(3)]
    assert conv_ids == [1, 2, 3]
    assert JsonlRunStore(tmp_path).create_conversation("u", "d")["id"] == 4


def test_large_action_values_spill_to_blobs(tmp_path):
    """old/new 超阈值外溢为内容寻址 blob；读侧透明还原，actions 文件保持小。"""
    store = JsonlRunStore(tmp_path, spill_threshold=1024)
    big_old = "x" * 20000
    big_new = "y" * 20000
    store.create_run("r3", "u1", "t")
    store.log_action("r3", "u1", "file_write", "big.txt", big_old, big_new)

    # 行本身不含 20KB 内容，只有 blob ref
    raw = (tmp_path / "actions.jsonl").read_text(encoding="utf-8")
    assert len(raw) < 2000 and "blob:sha256:" in raw
    assert len(list((tmp_path / "blobs").iterdir())) == 2

    # 读侧透明还原为完整内容（undo reverter 直接可用）
    actions = store.list_actions("r3", "u1")
    assert actions[0].old_value == big_old and actions[0].new_value == big_new

    # 小值不外溢
    store.log_action("r3", "u1", "file_write", "s.txt", "a", "b")
    assert "blob:" not in (tmp_path / "actions.jsonl").read_text().splitlines()[-1]


def test_spill_disabled_with_zero_threshold(tmp_path):
    store = JsonlRunStore(tmp_path, spill_threshold=0)
    store.create_run("r4", "u1", "t")
    store.log_action("r4", "u1", "file_write", "big.txt", "z" * 5000, "w" * 5000)
    raw = (tmp_path / "actions.jsonl").read_text(encoding="utf-8")
    assert "z" * 100 in raw, "禁用外溢时值原样落盘"


# --------------------------------------------------------------------------- #
# DictToolAdapter
# --------------------------------------------------------------------------- #
async def test_dict_tool_adapter_bridges_ctx_and_result():
    seen: dict = {}

    async def greet(ctx, args):
        seen.update(ctx)
        return {"ok": True, "summary": "greet", "content": f"hi {ctx['student_id']}"}

    specs = [{"type": "function",
              "function": {"name": "greet", "description": "g",
                           "parameters": {"type": "object", "properties": {}}}}]
    reg = DictToolAdapter(specs, {"greet": greet})

    res = await reg.dispatch("greet", {"x": 1},
                             AgentContext(run_id="r1", user_id="u42",
                                          extra={"thesis_id": 7}))
    assert res.ok and res.content == "hi u42"
    assert seen["student_id"] == "u42" and seen["user_id"] == "u42"
    assert seen["thesis_id"] == 7 and seen["run_id"] == "r1"
    # default classification is read-only → visible in anchored mode
    assert {s["function"]["name"] for s in reg.specs_for_mode("anchored")} == {"greet"}


# --------------------------------------------------------------------------- #
# assemble_messages
# --------------------------------------------------------------------------- #
def test_assemble_messages_reconciles_history():
    msgs = assemble_messages(
        "SYS",
        [{"role": "user", "content": "q"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "r"},
         {"role": "assistant", "content": "   "}],  # empty -> dropped
        "task")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[-1] == {"role": "user", "content": "task"}
    assert msgs[2]["tool_calls"][0]["id"] == "c1"


def test_assemble_messages_drops_orphan_tool_calls():
    """取消留下的孤儿 tool_call（无 tool 结果）与无主 tool 结果都不得进入
    下一次请求——否则 chat API 会 400 拒绝整个 payload。"""
    orphan_call = [{"id": "c9", "function": {"name": "f", "arguments": "{}"}}]
    msgs = assemble_messages(
        None,
        [{"role": "user", "content": "q"},
         {"role": "assistant", "content": "", "tool_calls": orphan_call},
         # run cancelled here: no tool result for c9
         {"role": "assistant", "content": "partial answer"},
         {"role": "tool", "tool_call_id": "ghost", "content": "r"}],
        "task")
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user"]
    assert "tool_calls" not in msgs[1]
    assert all(m["role"] != "tool" for m in msgs)


def test_assemble_messages_keeps_answered_calls_only():
    calls = [{"id": "a", "function": {"name": "f", "arguments": "{}"}},
             {"id": "b", "function": {"name": "g", "arguments": "{}"}}]
    msgs = assemble_messages(
        None,
        [{"role": "user", "content": "q"},
         {"role": "assistant", "content": "", "tool_calls": calls},
         {"role": "tool", "tool_call_id": "a", "content": "ra"}],
        # b never got its result (cancelled mid-step): only a survives
        "task")
    assert msgs[1]["tool_calls"] == [calls[0]]
    assert msgs[2] == {"role": "tool", "tool_call_id": "a", "content": "ra"}


# --------------------------------------------------------------------------- #
# window_with_recap
# --------------------------------------------------------------------------- #
def test_window_with_recap_summarizes_older():
    run_rows = [{"run_id": "r1", "task": "t1", "final": "a1", "status": "done"},
                {"run_id": "r2", "task": "t2", "final": "a2", "status": "done"}]
    recent = [{"role": "user", "content": "t2"},
              {"role": "assistant", "content": "a2"}]
    older_acts = [{"run_id": "r1", "kind": "file_write", "target": "a.md",
                   "status": "applied"}]

    out = window_with_recap(run_rows, recent, older_acts,
                            label_fn=lambda a: f"写入 {a['target']}", limit=1)
    assert out[0]["role"] == "assistant" and "写入 a.md" in out[0]["content"]
    assert out[-1] == {"role": "assistant", "content": "a2"}


def test_window_with_recap_no_older_no_recap():
    out = window_with_recap([{"run_id": "r1", "task": "t", "final": "a", "status": "done"}],
                            [{"role": "user", "content": "t"}], [],
                            label_fn=lambda a: None, limit=12)
    assert out == [{"role": "user", "content": "t"}]


# --------------------------------------------------------------------------- #
# run_timeline
# --------------------------------------------------------------------------- #
def _file_hooks():
    return ({"write_file": "file"},
            lambda a: "file" if a["kind"].startswith("file") else None,
            lambda a: [{"type": "file_change", "path": a["target"]}])


def test_run_timeline_meta_ui_replayed_and_action_discarded():
    run = {"task": "t"}
    messages = [
        {"role": "assistant", "content": "",
         "tool_calls": '[{"id":"c1","function":{"name":"write_file","arguments":"{\\"path\\":\\"a\\"}"}}]'},
        {"role": "tool", "tool_call_id": "c1", "tool_name": "write_file",
         "content": "ok", "meta": {"ok": True, "summary": "wrote",
                                   "ui": [{"type": "file_change", "path": "a"}]}},
        {"role": "assistant", "content": "done"},
    ]
    actions = [{"id": 1, "kind": "file_write", "target": "a", "status": "applied"}]
    tc, ac_cat, ac_ev = _file_hooks()
    evs = run_timeline(run, messages, actions, tool_category=tc,
                       action_category=ac_cat, action_events_fn=ac_ev)
    types = [e["type"] for e in evs]
    assert types[0] == "user"
    assert "tool_call" in types and "tool_result" in types
    # meta.ui replayed verbatim; the matched action is discarded -> exactly one
    assert sum(1 for e in evs if e["type"] == "file_change") == 1
    assert evs[-1] == {"type": "assistant", "text": "done"}


def test_run_timeline_legacy_rebuild_from_actions():
    run = {"task": "t"}
    messages = [
        {"role": "assistant", "content": "",
         "tool_calls": '[{"id":"c1","function":{"name":"write_file","arguments":"{}"}}]'},
        {"role": "tool", "tool_call_id": "c1", "tool_name": "write_file",
         "content": "ok", "meta": {"ok": True}},
    ]
    actions = [{"id": 1, "kind": "file_write", "target": "a", "status": "applied"}]
    tc, ac_cat, ac_ev = _file_hooks()
    evs = run_timeline(run, messages, actions, tool_category=tc,
                       action_category=ac_cat, action_events_fn=ac_ev)
    # no meta.ui → side-effect rebuilt from actions and matched to the tool call
    assert sum(1 for e in evs if e.get("type") == "file_change") == 1
    assert any(e["type"] == "tool_result" for e in evs)


def test_run_timeline_unmatched_action_flushes_at_end():
    run = {"task": "t"}
    messages = [{"role": "assistant", "content": "hi"}]  # no tool call at all
    actions = [{"id": 1, "kind": "file_write", "target": "a", "status": "applied"}]
    tc, ac_cat, ac_ev = _file_hooks()
    evs = run_timeline(run, messages, actions, tool_category=tc,
                       action_category=ac_cat, action_events_fn=ac_ev)
    assert evs[-1] == {"type": "file_change", "path": "a"}  # flushed, not lost


# -- undo_run default reverters (registry revert_kind) ---------------------------

async def test_undo_run_default_reverters_revert_file_writes(tmp_path):
    """零配置路径：register_file_tools 注册的 reverter 现在以 action 的 kind
    （file_write/file_edit）为键挂在 registry 上，undo_run 不传 reverters=
    也能真正回滚——旧实现默认表按工具名 keyed，静默空转还报 ok=True。"""
    from lithe.bundles.workspace import Workspace, register_file_tools

    store = JsonlRunStore(tmp_path)
    reg = ToolRegistry()
    ws = Workspace(tmp_path / "ws")
    register_file_tools(reg, lambda ctx: ws)

    transport = _chat_transport([
        {"choices": [{"message": {"content": "",
                                  "tool_calls": [_tc("write_file",
                                                     {"path": "a.txt",
                                                      "content": "hi"})]}}]},
        {"choices": [{"message": {"content": "done"}}]},
    ])
    host = AgentHost(reg,
                     LLMConfig(model="m", base_url="x", api_key="k",
                               transport=transport),
                     store, max_steps=5)
    events = [e async for e in host.run(AgentContext(run_id="r1", user_id="u1"),
                                        "write a.txt")]
    assert events[-1]["type"] == EventType.DONE
    assert (tmp_path / "ws" / "a.txt").read_text(encoding="utf-8") == "hi"

    report = await undo_run(host, "r1", "u1")  # 不传 reverters=
    assert report.reverted == 1 and report.ok is True
    assert not (tmp_path / "ws" / "a.txt").exists()


# -- prompt-builder failures funnel into the error envelope ----------------------

async def test_host_prompt_builder_failure_closes_run_failed(tmp_path):
    """run_start 之后、runtime 之前的崩溃（host prompt builder 抛错/盘满）
    必须走错误漏斗：error 事件 + run 落盘 failed，而不是把裸异常抛出
    生成器、run 永远卡在 running。"""
    store = JsonlRunStore(tmp_path)

    def bad_prompt(ctx, mode, anchor):
        raise RuntimeError("host prompt bug")

    host = AgentHost(_registry({}),
                     LLMConfig(model="m", base_url="x", api_key="k"),
                     store, build_system_prompt=bad_prompt, max_steps=3)
    events = [e async for e in host.run(AgentContext(run_id="r5", user_id="u1"),
                                        "go")]
    assert any(e["type"] == EventType.ERROR for e in events)
    assert events[-1]["type"] == EventType.DONE and events[-1]["status"] == "failed"
    assert store.get_run("r5", "u1").status == "failed"


def test_assemble_messages_tolerates_multimodal_history():
    """list 形态的 content（多模态块）不能让 assemble_messages 崩——
    之前的 (content or "").strip() 对 list 会 AttributeError。"""
    hist = [
        {"role": "user", "content": [{"type": "text", "text": "看看这张图"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "看到了"}]},
        {"role": "user", "content": "继续"},
    ]
    msgs = assemble_messages("SYS", hist, "next task")
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["content"] == hist[0]["content"]  # 原样转发
    assert msgs[-1] == {"role": "user", "content": "next task"}


# -- run ids are globally unique -------------------------------------------------

def test_create_run_rejects_duplicate_run_id(tmp_path):
    """重复 run_id 会让 fold 合成 chimera、list_runs 重复列出，且
    run_final 不带 user_id 时跨用户互写终态——写入侧直接拒绝。"""
    s = JsonlRunStore(tmp_path)
    s.create_run("r1", "u1", "t")
    with pytest.raises(ValueError, match="globally unique"):
        s.create_run("r1", "u1", "again")
    with pytest.raises(ValueError, match="globally unique"):
        s.create_run("r1", "u2", "other user same id")
    assert len(s.list_runs("u1")) == 1


def test_legacy_duplicate_run_headers_list_once(tmp_path):
    """旧数据文件里已存在的重复 run 头：fold 去重，list 不再双列。"""
    import json as _json
    with (tmp_path / "runs.jsonl").open("a", encoding="utf-8") as fh:
        for _ in range(2):
            fh.write(_json.dumps({"kind": "run", "run_id": "r1",
                                  "user_id": "u1", "task": "t",
                                  "status": "running"}) + "\n")
    s2 = JsonlRunStore(tmp_path)
    assert len(s2.list_runs("u1")) == 1


# -- blob refs are validated, not traversed --------------------------------------

def test_blob_load_rejects_forged_refs(tmp_path):
    s = JsonlRunStore(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("HOST SECRET", encoding="utf-8")
    with pytest.raises(ValueError):
        s.load("sha256:../../secret.txt")
    with pytest.raises(ValueError):
        s.load("sha256:../..%2fsecret")
    # a tool-supplied literal that merely looks like a ref is surfaced
    # verbatim, never rehydrated into file bytes
    forged = "blob:sha256:../../secret.txt"
    s.log_action("r1", "u1", "file_write", "a.txt", forged, "new",
                 status="applied")
    rows = s.list_actions("r1", "u1")
    assert rows[0].old_value == forged
    assert "HOST SECRET" not in str(rows[0].old_value)
