"""Mid-run context windowing: shrinking old tool outputs under a character
budget, escalating to dropping whole old exchanges under an omission note
(pairing always valid, system/user never dropped), token-calibrated budgets
from measured chars/token ratios, and message size accounting."""
from __future__ import annotations

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolCategory,
    ToolRegistry, ToolResult, ToolSpec,
)

from _fakes import FakeTransport, echo, resp, tc


# --- mid-run context budget ----------------------------------------------------

async def test_context_budget_trims_old_tool_results():
    async def big_echo(ctx, args):
        return ToolResult(True, "echoed", "x" * 3000)

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), big_echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       transport=FakeTransport([
                           resp(tool_calls=[tc(cid="c1")]),
                           resp(tool_calls=[tc(cid="c2")]),
                           resp(tool_calls=[tc(cid="c3")]),
                           resp("done")],
                       )),
        context_budget=6000)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    stats = RunStats()
    events = [e async for e in rt.run(ctx, messages, reg.specs_for_mode(),
                                      stats=stats)]
    assert any(e["type"] == "assistant" for e in events) and stats.status == "done"
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    # 最老的工具结果被压缩（保首尾），最新的原样保留
    assert len(tool_msgs[0]["content"]) < 600 and "已截断" in tool_msgs[0]["content"]
    assert len(tool_msgs[1]["content"]) == 3000
    assert len(tool_msgs[2]["content"]) == 3000
    # 结构有效性：每个 tool_call id 仍配对一个 tool 消息
    tc_ids = [tc["id"] for m in messages if m.get("role") == "assistant"
              for tc in (m.get("tool_calls") or [])]
    assert sorted(tc_ids) == sorted(m["tool_call_id"] for m in tool_msgs)


async def test_context_budget_none_disables_trimming():
    async def big_echo(ctx, args):
        return ToolResult(True, "echoed", "y" * 3000)

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), big_echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       transport=FakeTransport([
                           resp(tool_calls=[tc(cid="c1")]),
                           resp(tool_calls=[tc(cid="c2")]),
                           resp("done")])),
        context_budget=None)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    _ = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    assert all(len(m["content"]) == 3000
               for m in messages if m.get("role") == "tool")


async def test_running_context_size_stays_accurate():
    """增量计量的 chars 与全量重扫一致（含 trim 之后）。"""
    from lithe.runtime import _context_size

    async def big_echo(ctx, args):
        return ToolResult(True, "echoed", "z" * 2500)

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), big_echo)
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       transport=FakeTransport([
                           resp(tool_calls=[tc(cid="c1")]),
                           resp(tool_calls=[tc(cid="c2")]),
                           resp(tool_calls=[tc(cid="c3")]),
                           resp("done")])),
        context_budget=6000)
    ctx = AgentContext(run_id="r", user_id="u")
    messages = [{"role": "user", "content": "hi"}]
    events = [e async for e in rt.run(ctx, messages, reg.specs_for_mode())]
    # 每一步上报的 context_chars 都等于该时刻的真实全量大小
    usages = [e for e in events if e["type"] == "usage"]
    assert len(usages) == 4
    # 语义：context_chars = 本次调用发送的载荷大小（不含当步模型回复自身，
    # 那是在调用之后才 append 的），与全量重扫 messages[:-1] 一致
    assert usages[-1]["context_chars"] == _context_size(messages[:-1])
    assert usages[0]["context_chars"] < usages[-1]["context_chars"]


# --- context trim escalation: drop old exchanges when shrinking isn't enough ----

def _mk_exchange(cid: str, asst_text: str, tool_text: str = "r") -> list[dict]:
    return [
        {"role": "assistant", "content": asst_text,
         "tool_calls": [{"id": cid, "type": "function",
                         "function": {"name": "echo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": cid, "content": tool_text},
    ]


def _pairing_valid(messages: list[dict]) -> bool:
    """每个 assistant tool_call id 恰有一个配对的 tool 消息，反之亦然。"""
    call_ids = [tc.get("id") for m in messages if m.get("role") == "assistant"
                for tc in (m.get("tool_calls") or [])]
    result_ids = [m.get("tool_call_id") for m in messages
                  if m.get("role") == "tool"]
    return sorted(call_ids) == sorted(result_ids)


def test_trim_context_drops_old_exchanges_when_shrinking_insufficient():
    """超限来自不可收缩的正文（assistant 长文本）时：整轮丢弃最旧交换、
    以一条省略注记替代，配对保持有效，system/user 永不丢弃。"""
    from lithe.runtime import _OMITTED_TURNS_NOTE, _context_size, _trim_context

    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "task"}]
    for cid, ch in (("c1", "a"), ("c2", "b"), ("c3", "c"), ("c4", "d")):
        messages += _mk_exchange(cid, ch * 1500)
    assert _context_size(messages) > 4500

    result = _trim_context(messages, budget=4500)

    assert result <= 4500
    notes = [m for m in messages if m.get("content") == _OMITTED_TURNS_NOTE]
    assert len(notes) == 1 and notes[0]["role"] == "assistant"
    # 最旧两轮被整轮丢弃；最新两轮原样保留
    remaining_calls = {tc["id"] for m in messages
                       if m.get("role") == "assistant"
                       for tc in (m.get("tool_calls") or [])}
    assert remaining_calls == {"c3", "c4"}
    assert _pairing_valid(messages)
    # system / user 消息永不丢弃；注记位于保留内容之前
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
    note_idx = messages.index(notes[0])
    kept_start = next(i for i, m in enumerate(messages)
                      if m.get("tool_call_id") == "c3")
    assert note_idx < kept_start


def test_trim_context_no_drop_when_budget_unreachable():
    """即使丢光所有可丢的交换也到不了预算（超限在受保护的近期内容里）：
    不做无谓丢弃，原样返回超限总量。"""
    from lithe.runtime import _OMITTED_TURNS_NOTE, _context_size, _trim_context

    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "task"}]
    for cid, ch in (("c1", "a"), ("c2", "b"), ("c3", "c"), ("c4", "d")):
        messages += _mk_exchange(cid, ch * 1500)
    before = _context_size(messages)

    result = _trim_context(messages, budget=2000)  # 丢光两轮也远不够

    assert result == before, "无谓的丢弃不得发生"
    assert not any(m.get("content") == _OMITTED_TURNS_NOTE for m in messages)
    assert len([m for m in messages if m.get("role") == "tool"]) == 4
    assert _pairing_valid(messages)


def test_trim_context_omission_note_unique_across_passes():
    """第二轮丢弃时旧注记被移除、新注记补位：上下文里永远至多一条注记。"""
    from lithe.runtime import _OMITTED_TURNS_NOTE, _trim_context

    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "task"}]
    for cid, ch in (("c1", "a"), ("c2", "b"), ("c3", "c"), ("c4", "d")):
        messages += _mk_exchange(cid, ch * 1500)
    _trim_context(messages, budget=4500)
    assert len([m for m in messages
                if m.get("content") == _OMITTED_TURNS_NOTE]) == 1

    messages += _mk_exchange("c5", "e" * 1500)   # 又超预算
    _trim_context(messages, budget=4500)

    notes = [m for m in messages if m.get("content") == _OMITTED_TURNS_NOTE]
    assert len(notes) == 1
    remaining_calls = {tc["id"] for m in messages
                       if m.get("role") == "assistant"
                       for tc in (m.get("tool_calls") or [])}
    assert remaining_calls == {"c4", "c5"}
    assert _pairing_valid(messages)


async def test_context_escalation_drop_reaches_model_context():
    """端到端：正文膨胀且压缩无效时，后续模型调用看到的上下文恰含一条
    省略注记、配对完整，run 正常走完（而不是超窗请求）。"""
    from lithe.runtime import _OMITTED_TURNS_NOTE

    big = "x" * 1500

    class _SnapTransport:
        def __init__(self):
            self.snapshots: list[list[dict]] = []

        async def complete(self, client, **kw):
            self.snapshots.append([dict(m) for m in kw["messages"]])
            n = len(self.snapshots)
            if n >= 6:
                return resp("done")
            return resp(content=big, tool_calls=[tc(cid=f"c{n}")])

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), echo)
    transport = _SnapTransport()
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport),
                      context_budget=4500, max_steps=6)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats)]
    assert stats.final_text == "done"
    assert [e["type"] for e in events][-1] in ("assistant", "usage")

    # 第一次调用：尚无注记；任何时刻至多一条；触发丢弃后恰好一条
    assert not any(m.get("content") == _OMITTED_TURNS_NOTE
                   for m in transport.snapshots[0])
    for snap in transport.snapshots:
        assert len([m for m in snap
                    if m.get("content") == _OMITTED_TURNS_NOTE]) <= 1
        assert _pairing_valid(snap)
    assert len([m for m in transport.snapshots[-1]
                if m.get("content") == _OMITTED_TURNS_NOTE]) == 1
    # 最早的一轮确实从后续上下文中消失了
    dropped = transport.snapshots[-1]
    assert not any(m.get("tool_call_id") == "c1" for m in dropped)


async def test_token_calibrated_trim_uses_measured_ratio():
    """token 口径预算：按实测 chars/token 比率把 token 窗口换算成字符预算，
    第二步对旧工具结果触发压缩（旧 400k 字符预算下不会触发）。"""
    big = "x" * 20000

    async def _big_echo(**kw):
        return ToolResult(True, "done", big)

    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ), _big_echo)

    class _T:
        def __init__(self):
            self.calls = []

        async def complete(self, client, **kw):
            self.calls.append(kw["messages"])
            # 第一次调用报大 prompt（与字符数失配）：把 cpt 校准得很小，
            # 使 window×threshold×cpt 预算低于当前对话体积
            if len(self.calls) == 1:
                return {"content": "", "tool_calls": [tc()],
                        "usage": {"prompt_tokens": 100000}}
            return {"content": "done", "usage": {"prompt_tokens": 50000}}

    t = _T()
    rt = AgentRuntime(
        reg, LLMConfig(model="m", base_url="x", api_key="k",
                       context_window=1000, transport=t),
        context_budget=None, context_token_threshold=0.5)
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode())]
    assert [e["type"] for e in events][-1] in ("assistant", "usage")
    # 第二次调用的 messages 中，旧工具结果已被压缩（远小于原 20000 字符）
    tool_msgs = [m for m in t.calls[1] if m.get("role") == "tool"]
    assert tool_msgs and all(len(m["content"]) < 20000 for m in tool_msgs)



def test_message_size_counts_list_content():
    from lithe.runtime import _message_size

    url = "data:image/png;base64," + "A" * 100
    m = {"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}
    assert _message_size(m) == 5 + len(url)
    assert _message_size({"role": "user", "content": "hello"}) == 5
