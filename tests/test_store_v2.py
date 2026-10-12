"""Store v2: timestamps, token persistence, conversation meta + summaries.

Everything here runs against ``JsonlRunStore`` in a tmp dir — no network, no
kernel runtime needed except one host round that checks token plumbing.
"""
from __future__ import annotations

import time

from lithe import AgentContext, LLMConfig, ToolRegistry
from lithe.bundles import AgentHost, JsonlRunStore, StoredMessage


def test_run_stamps_created_and_finished_times(tmp_path):
    s = JsonlRunStore(tmp_path)
    t0 = time.time()
    s.create_run("r1", "u1", "task")
    s.finish_run("r1", "done", 1, 0.0, "ok")
    run = s.get_run("r1", "u1")
    assert run.created_at is not None and run.created_at >= t0 - 1
    assert run.finished_at is not None and run.finished_at >= run.created_at


def test_finish_run_persists_token_fields(tmp_path):
    s = JsonlRunStore(tmp_path)
    s.create_run("r1", "u1", "task")
    s.finish_run("r1", "done", 2, 0.5, "ok", prompt_tokens=100,
                 completion_tokens=20, cached_tokens=8, total_tokens=120)
    run = s.get_run("r1", "u1")
    assert (run.prompt_tokens, run.completion_tokens) == (100, 20)
    assert (run.cached_tokens, run.total_tokens) == (8, 120)


def test_finish_run_persists_error_diagnostic(tmp_path):
    """失败 run 的诊断（异常类+消息）必须落进 run_final 行并能折回——
    事后排查不能依赖当时的 stderr 日志。"""
    s = JsonlRunStore(tmp_path)
    s.create_run("r1", "u1", "task")
    s.finish_run("r1", "failed", 3, 0.02, None,
                 error="ConnectError: [Errno 11001] getaddrinfo failed")
    assert s.get_run("r1", "u1").error == \
        "ConnectError: [Errno 11001] getaddrinfo failed"
    # no error → stays None (legacy rows and happy paths keep their shape)
    s.create_run("r2", "u1", "t2")
    s.finish_run("r2", "done", 1, 0.0, "ok")
    assert s.get_run("r2", "u1").error is None


def test_legacy_lines_read_back_with_none_not_zero(tmp_path):
    """A store written before v2 must fold cleanly; missing fields read as
    None (unknown), never a fabricated zero."""
    s = JsonlRunStore(tmp_path)
    # hand-write legacy lines: no created_at/finished_at/tokens
    s._append(s._runs, {"kind": "run", "run_id": "old", "user_id": "u1",
                        "conversation_id": None, "task": "t",
                        "status": "running", "model": None})
    s._append(s._runs, {"kind": "run_final", "run_id": "old",
                        "status": "done", "steps": 1, "cost": 0.1,
                        "final": "x"})
    run = s.get_run("old", "u1")
    assert run.status == "done" and run.created_at is None
    assert run.finished_at is None and run.total_tokens is None
    # a v2 write next to legacy rows keeps both worlds intact
    s.create_run("new", "u1", "t2")
    assert s.get_run("new", "u1").created_at is not None
    assert s.get_run("old", "u1").created_at is None


def test_conversation_meta_create_update_and_fold(tmp_path):
    s = JsonlRunStore(tmp_path)
    c = s.create_conversation("u1", "t", meta={"workspace": "/w",
                                               "profile": "zhipu"})
    got = s.get_conversation(c["id"], "u1")
    assert got["meta"] == {"workspace": "/w", "profile": "zhipu"}
    # meta updates merge (patch semantics), never replace wholesale
    assert s.update_conversation_meta(c["id"], "u1", {"model": "glm-4.6"}) == 1
    got = s.get_conversation(c["id"], "u1")
    assert got["meta"] == {"workspace": "/w", "profile": "zhipu",
                           "model": "glm-4.6"}
    assert s.update_conversation_meta(c["id"], "u1", {"model": "glm-4.5"}) == 1
    assert s.get_conversation(c["id"], "u1")["meta"]["model"] == "glm-4.5"
    # unknown / cross-user conversation: no-op, not an error
    assert s.update_conversation_meta(999, "u1", {"x": 1}) == 0
    assert s.update_conversation_meta(c["id"], "u2", {"x": 1}) == 0
    # meta lives through re-open (append-only fold)
    s2 = JsonlRunStore(tmp_path)
    assert s2.get_conversation(c["id"], "u1")["meta"]["profile"] == "zhipu"


def test_conversation_summaries_aggregate_and_order(tmp_path):
    s = JsonlRunStore(tmp_path)
    c1 = s.create_conversation("u1", "older")["id"]
    c2 = s.create_conversation("u1", "newer")["id"]
    s.create_run("a1", "u1", "task A1", conversation_id=c1, model="m1")
    s.finish_run("a1", "done", 1, 0.1, "a1 done",
                 prompt_tokens=50, completion_tokens=10, total_tokens=60)
    s.create_run("b1", "u1", "task B1", conversation_id=c2, model="m2")
    s.finish_run("b1", "done", 2, 0.2, "b1 done",
                 prompt_tokens=70, completion_tokens=30, total_tokens=100)
    s.create_run("a2", "u1", "task A2", conversation_id=c1, model="m1")
    s.finish_run("a2", "done", 1, 0.1, "a2 done",
                 prompt_tokens=10, completion_tokens=5, total_tokens=15)

    # user isolation
    s.create_run("x1", "u2", "other user", conversation_id=None)

    rows = s.conversation_summaries("u1")
    assert [r["id"] for r in rows] == [c1, c2]  # a2 finished last → c1 first
    c1_row = rows[0]
    assert c1_row["n_runs"] == 2 and c1_row["title"] == "older"
    assert c1_row["last_task"] == "task A2" and c1_row["last_status"] == "done"
    assert c1_row["prompt_tokens"] == 60 and c1_row["total_tokens"] == 75
    assert c1_row["total_cost"] == 0.2  # 0.1 + 0.1
    assert rows[1]["n_runs"] == 1 and rows[1]["total_tokens"] == 100
    # limit slices newest-first
    assert [r["id"] for r in s.conversation_summaries("u1", limit=1)] == [c1]


def test_messages_for_conversation_joins_runs_in_order(tmp_path):
    s = JsonlRunStore(tmp_path)
    cid = s.create_conversation("u1", "c")["id"]
    s.create_run("r1", "u1", "t1", conversation_id=cid)
    s.add_message(StoredMessage(role="user", content="q1", run_id="r1",
                                user_id="u1"))
    s.add_message(StoredMessage(role="assistant", content="a1", run_id="r1",
                                user_id="u1"))
    s.create_run("r2", "u1", "t2", conversation_id=cid)
    s.add_message(StoredMessage(role="user", content="q2", run_id="r2",
                                user_id="u1"))
    s.add_message(StoredMessage(role="assistant", content="a2", run_id="r2",
                                user_id="u1", subagent="sub"))
    # a sessionless run must not leak into the conversation
    s.create_run("solo", "u1", "solo task")
    s.add_message(StoredMessage(role="user", content="solo", run_id="solo",
                                user_id="u1"))
    # and another user's run with the same conversation id stays invisible
    s.create_run("z1", "u2", "t", conversation_id=cid)
    s.add_message(StoredMessage(role="user", content="other", run_id="z1",
                                user_id="u2"))

    msgs = s.messages_for_conversation(cid, "u1")
    assert [m["content"] for m in msgs] == ["q1", "a1", "q2"]
    everything = s.messages_for_conversation(cid, "u1",
                                             exclude_subagent=False)
    assert [m["content"] for m in everything] == ["q1", "a1", "q2", "a2"]


async def test_host_close_run_persists_tokens(tmp_path):
    """AgentHost._close_run must forward RunStats tokens into the store."""
    class _Usage:
        async def complete(self, client, **kw):
            return {"content": "答案", "tool_calls": [],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4,
                              "total_tokens": 15}}

    host = AgentHost(ToolRegistry(), LLMConfig("m", "https://x", "k",
                                               transport=_Usage()),
                     JsonlRunStore(tmp_path / "s"))
    ctx = AgentContext(run_id="tok1", user_id="u")
    async for ev in host.run(ctx, "hi"):
        assert ev.get("type") != "error"
    run = host.store.get_run("tok1", "u")
    assert run.prompt_tokens == 11 and run.completion_tokens == 4
    assert run.total_tokens == 15 and run.status == "done"
    assert run.created_at is not None and run.finished_at is not None
