"""lithe.memory: pure transforms that rebuild model context from stored
turns, with zero storage access. These pin the replay mechanics (tool-call
pairing, orphan reconciliation, head+tail truncation, recap compression)."""
from __future__ import annotations

import json

from lithe import recap_text, replay_messages, truncate_tool_result


# --- truncate_tool_result ---

def test_truncate_short_unchanged():
    assert truncate_tool_result("short") == "short"


def test_truncate_long_bounded_head_and_tail():
    big = "a" * 1000
    out = truncate_tool_result(big, cap=100)
    assert len(out) < len(big)
    assert out.startswith("a" * 50) and out.endswith("a" * 50)
    assert "已截断" in out


# --- recap_text ---

def test_recap_empty_when_no_older_runs():
    assert recap_text([], {}) == ""


def test_recap_formats_turns():
    older = [{"run_id": "r1", "task": "做什么", "final": "这样", "status": "done"}]
    recap = recap_text(older, {"r1": ["写入文件 a.md"]})
    assert "更早对话回顾" in recap
    assert "用户问：做什么" in recap
    assert "回复摘要：这样" in recap
    assert "所做改动：写入文件 a.md" in recap


def test_recap_marks_failed_turn():
    older = [{"run_id": "r1", "task": "t", "final": "", "status": "failed"}]
    assert "出错未完成" in recap_text(older, {})


# --- replay_messages ---

def test_replay_simple_user_assistant():
    rows = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    assert replay_messages(rows) == rows


def test_replay_tool_call_paired():
    rows = [
        {"role": "assistant", "content": "",
         "tool_calls": '[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]'},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    out = replay_messages(rows)
    assert out[0]["role"] == "assistant"
    assert out[0]["tool_calls"][0]["id"] == "c1"
    assert out[1] == {"role": "tool", "tool_call_id": "c1", "content": "result"}


def test_replay_rehydrates_reasoning():
    item = {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC"}
    # JSON 字符串形态（单列存储）与解析后列表形态都接受
    rows = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": json.dumps(
             [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]),
         "reasoning": json.dumps([item])},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    out = replay_messages(rows)
    assert out[1]["reasoning"] == [item]
    # 损坏/空数据降级为无 reasoning；非 reasoning 条目被丢弃
    assert "reasoning" not in replay_messages(
        [{"role": "assistant", "content": "x", "reasoning": "not json{"}])[0]
    assert "reasoning" not in replay_messages(
        [{"role": "assistant", "content": "x",
          "reasoning": json.dumps([{"type": "message", "id": "m1"}])}])[0]
    # 摘要提取：块列表与纯字符串两种 summary 形态
    from lithe.memory import reasoning_summary_text
    assert reasoning_summary_text(
        [{"type": "reasoning",
          "summary": [{"type": "summary_text", "text": "先查库"}]}]) == "先查库"
    assert reasoning_summary_text(
        [{"type": "reasoning", "summary": "直连字符串"}]) == "直连字符串"
    assert reasoning_summary_text(None) == ""


def test_replay_orphan_tool_call_falls_back_to_text():
    # assistant tool_calls with no matching tool result → tool_calls dropped,
    # text content kept (so the turn isn't lost entirely)
    rows = [{"role": "assistant", "content": "x",
             "tool_calls": '[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]'}]
    assert replay_messages(rows) == [{"role": "assistant", "content": "x"}]


def test_replay_orphan_tool_result_dropped():
    rows = [{"role": "tool", "tool_call_id": None, "content": "r"}]
    assert replay_messages(rows) == []


def test_replay_legacy_tool_pairing_by_position():
    # tool row without tool_call_id, preceded by assistant tool_calls → paired
    rows = [
        {"role": "assistant", "content": "",
         "tool_calls": '[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]'},
        {"role": "tool", "tool_call_id": None, "content": "r"},
    ]
    out = replay_messages(rows)
    assert out[1]["tool_call_id"] == "c1"


def test_replay_truncates_large_tool_result():
    big = "x" * 100
    rows = [
        {"role": "assistant", "content": "",
         "tool_calls": '[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]'},
        {"role": "tool", "tool_call_id": "c1", "content": big},
    ]
    out = replay_messages(rows, tool_result_cap=10)
    assert "已截断" in out[1]["content"]
    assert len(out[1]["content"]) < len(big)


def test_replay_prepends_recap_note():
    rows = [{"role": "user", "content": "hi"}]
    out = replay_messages(rows, recap="RECAP")
    assert out[0] == {"role": "assistant", "content": "RECAP"}
    assert out[1]["role"] == "user"


def test_replay_accepts_parsed_tool_calls_list():
    # tool_calls may already be a list (in-memory data), not just a JSON string
    rows = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    out = replay_messages(rows)
    assert out[0]["tool_calls"][0]["id"] == "c1"
