"""The steering inbox: user text injected at the next step boundary (last
context message, user_injected event, record channel), order preserved for
multiple messages, persistence through StoreSink, stop winning over inbox,
the forced wrap-up step skipping the drain, and cross-thread queues."""
from __future__ import annotations

import asyncio
import queue as thread_queue

from lithe import (
    AgentContext, AgentRuntime, LLMConfig, RunStats, ToolCategory,
    ToolRegistry, ToolResult, ToolSpec,
)

from _fakes import Sink, resp, tc


# --- P1-4: steering inbox --------------------------------------------------------

class _SnapTransport:
    """Snapshots the messages it is called with, then plays scripted results."""

    def __init__(self, results):
        self._results = list(results)
        self.snapshots: list[list[dict]] = []

    async def complete(self, client, **kw):
        self.snapshots.append([dict(m) for m in kw["messages"]])
        r = self._results.pop(0)
        return resp(r.get("content", ""),
                     r.get("tool_calls"))


async def test_steering_inbox_injects_user_message():
    reg = ToolRegistry()
    inbox = asyncio.Queue()

    async def echo_then_steer(ctx, args):
        await inbox.put("别动 src/，只看 tests/")
        return ToolResult(True, "echoed", "ok")

    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ),
                 echo_then_steer)
    transport = _SnapTransport([
        {"tool_calls": [tc(cid="c1")]},
        {"content": "收到，改看 tests/"},
    ])
    sink = Sink()
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport),
                      sinks=[sink], max_steps=4)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats,
                                      inbox=inbox)]
    assert stats.status == "done" and stats.final_text == "收到，改看 tests/"
    # 第二次模型调用前，注入消息是上下文的最后一条
    second = transport.snapshots[1]
    assert second[-1] == {"role": "user", "content": "别动 src/，只看 tests/"}
    # 事件 + 记录双通道都看到它
    inj = [e for e in events if e["type"] == "user_injected"]
    assert len(inj) == 1 and inj[0]["text"] == "别动 src/，只看 tests/"
    types = [e["type"] for e in events]
    assert types.index("step") < types.index("user_injected") \
        < types.index("assistant")
    assert {"role": "user", "content": "别动 src/，只看 tests/"} in sink.records


async def test_steering_user_row_persists_through_storesink(tmp_path):
    """The injected line must survive persistence: StoreSink historically
    dropped user records (only assistant/tool were written), so a resumed
    conversation lost the very message that steered it."""
    from lithe.bundles import AgentHost, JsonlRunStore
    from lithe.bundles.host import StoreSink

    reg = ToolRegistry()
    inbox = asyncio.Queue()

    async def echo_then_steer(ctx, args):
        await inbox.put("改看 tests/")
        return ToolResult(True, "echoed", "ok")

    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ),
                 echo_then_steer)
    store = JsonlRunStore(tmp_path)
    host = AgentHost(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                    transport=_SnapTransport([
                                        {"tool_calls": [tc(cid="c1")]},
                                        {"content": "done"},
                                    ])), store, max_steps=4,
                     build_system_prompt=lambda ctx, mode, anchor: "sys")
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in host.run(ctx, "任务", inbox=inbox)]
    rows = store.messages_for_run("r", "u")
    injected = [r for r in rows
                if r.get("role") == "user" and r.get("content") == "改看 tests/"]
    assert len(injected) == 1
    # the host-written task row is not duplicated by the sink
    assert sum(1 for r in rows if r.get("role") == "user") == 2
    assert isinstance(StoreSink(store), StoreSink)  # import sanity


async def test_steering_multiple_messages_preserve_order():
    reg = ToolRegistry()
    inbox = asyncio.Queue()

    async def echo_then_steer(ctx, args):
        await inbox.put("第一条")
        await inbox.put("第二条")
        return ToolResult(True, "echoed", "ok")

    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ),
                 echo_then_steer)
    transport = _SnapTransport([
        {"tool_calls": [tc(cid="c1")]},
        {"content": "done"},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport))
    ctx = AgentContext(run_id="r", user_id="u")
    _ = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                 reg.specs_for_mode(), inbox=inbox)]
    user_msgs = [m for m in transport.snapshots[1] if m.get("role") == "user"]
    assert user_msgs == [{"role": "user", "content": "hi"},
                         {"role": "user", "content": "第一条"},
                         {"role": "user", "content": "第二条"}]


async def test_steering_empty_inbox_changes_nothing():
    reg = ToolRegistry()
    rt_none = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                          transport=_SnapTransport(
                                              [{"content": "done"}])))
    rt_empty = AgentRuntime(reg, LLMConfig(
        model="m", base_url="x", api_key="k",
        transport=_SnapTransport([{"content": "done"}])))
    ctx = AgentContext(run_id="r", user_id="u")
    a = [e async for e in rt_none.run(ctx, [{"role": "user", "content": "hi"}],
                                      [])]
    b = [e async for e in rt_empty.run(ctx, [{"role": "user", "content": "hi"}],
                                       [], inbox=asyncio.Queue())]
    assert [e["type"] for e in a] == [e["type"] for e in b] == \
        ["step", "assistant", "usage"]


async def test_steering_stop_wins_over_inbox():
    reg = ToolRegistry()
    inbox = asyncio.Queue()
    await inbox.put("来不及了")

    class _Never:
        async def complete(self, client, **kw):
            raise AssertionError("取消后模型不应被调用")

    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=_Never()))
    ctx = AgentContext(run_id="r", user_id="u")
    stop = asyncio.Event()
    stop.set()
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], stats=stats, stop=stop, inbox=inbox)]
    assert [e["type"] for e in events] == ["cancelled"]
    assert stats.status == "cancelled"
    assert not any(e["type"] == "user_injected" for e in events)


async def test_steering_wrapup_step_skipsdrain():
    reg = ToolRegistry()
    inbox = asyncio.Queue()

    async def echo_then_steer(ctx, args):
        await inbox.put("最后一条")
        return ToolResult(True, "echoed", "ok")

    reg.register(ToolSpec("echo", "e", category=ToolCategory.READ),
                 echo_then_steer)
    transport = _SnapTransport([
        {"tool_calls": [tc(cid="c1")]},
        {"content": "收尾"},
    ])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport), max_steps=2)
    ctx = AgentContext(run_id="r", user_id="u")
    stats = RunStats()
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      reg.specs_for_mode(), stats=stats,
                                      inbox=inbox)]
    # 第 2 步是强制收尾：不注入（工具已被扣下，注入无法被行动）
    assert stats.final_text == "收尾"
    assert not any(e["type"] == "user_injected" for e in events)
    assert not any(m.get("role") == "user" and m.get("content") == "最后一条"
                   for m in transport.snapshots[1])
    assert not inbox.empty(), "未消费的消息留在宿主队列里"


async def test_steering_accepts_cross_thread_queue():
    reg = ToolRegistry()
    inbox = thread_queue.Queue()
    inbox.put("线程投递")
    transport = _SnapTransport([{"content": "done"}])
    rt = AgentRuntime(reg, LLMConfig(model="m", base_url="x", api_key="k",
                                     transport=transport))
    ctx = AgentContext(run_id="r", user_id="u")
    events = [e async for e in rt.run(ctx, [{"role": "user", "content": "hi"}],
                                      [], inbox=inbox)]
    first = transport.snapshots[0]
    assert first == [{"role": "user", "content": "hi"},
                     {"role": "user", "content": "线程投递"}]
    assert any(e["type"] == "user_injected" for e in events)
