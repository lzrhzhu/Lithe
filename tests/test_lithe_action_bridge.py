"""StoreSink's mutation capture — the missing middle of the undo chain.

Tools emit undo-bearing UI events (``file_change`` / ``todo_change``) inside
``ToolResult.ui``; ``StoreSink.on_event`` maps them to ``store.log_action``
rows; ``undo_run`` reads the rows back and drives the kernel ``UndoEngine``
with the bundle reverters. These tests pin the whole chain, including the
``file_delete`` kind from ``apply_patch``, subagent tagging, blob spillover
of large old/new values, and the mark-reverted no-op second undo."""

from __future__ import annotations

from lithe import AgentContext, LLMConfig, ToolRegistry
from lithe.bundles.host import AgentHost, StoreSink, undo_run
from lithe.bundles.patch import register_apply_patch_tool
from lithe.bundles.store import JsonlRunStore
from lithe.bundles.workspace import Workspace, register_file_tools


def _setup(tmp_path):
    ws = Workspace(tmp_path / "ws")
    store = JsonlRunStore(tmp_path / "store")
    reg = ToolRegistry()
    reverters = register_file_tools(reg, lambda ctx: ws)
    reverters.update(register_apply_patch_tool(reg, lambda ctx: ws))
    return ws, store, reg, reverters


async def _capture(sink, ctx, result):
    """Mimic the runtime's ui forwarding (runtime.py: for ui in res.ui)."""
    for ev in result.ui:
        await sink.on_event(ctx, ev)


async def test_file_change_events_become_stored_actions(tmp_path):
    ws, store, reg, _ = _setup(tmp_path)
    sink = StoreSink(store)
    ctx = AgentContext(run_id="r1", user_id="u1")

    w = await reg.dispatch("write_file", {"path": "a.txt", "content": "v1"}, ctx)
    await _capture(sink, ctx, w)
    e = await reg.dispatch(
        "edit_file", {"path": "a.txt", "old_text": "v1", "new_text": "v2"}, ctx
    )
    await _capture(sink, ctx, e)

    rows = store.list_actions("r1", "u1")
    assert [(r.kind, r.target) for r in rows] == [
        ("file_write", "a.txt"),
        ("file_edit", "a.txt"),
    ]
    assert rows[0].old_value is None and rows[0].new_value == "v1"
    assert rows[1].old_value == "v1" and rows[1].new_value == "v2"


async def test_apply_patch_events_include_file_delete_kind(tmp_path):
    ws, store, reg, _ = _setup(tmp_path)
    sink = StoreSink(store)
    ctx = AgentContext(run_id="r1", user_id="u1")
    ws.write("d.txt", "gone\n")

    p = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": (
                "*** Begin Patch\n"
                "*** Update File: d.txt\n@@\n-gone\n+stay\n"
                "*** Delete File: d.txt\n"
                "*** End Patch"
            )
        },
        ctx,
    )
    assert p.ok
    await _capture(sink, ctx, p)

    rows = store.list_actions("r1", "u1")
    assert [r.kind for r in rows] == ["file_edit", "file_delete"]
    assert rows[1].old_value == "stay\n" and rows[1].new_value is None


async def test_subagent_tag_carried_to_action_rows(tmp_path):
    ws, store, reg, _ = _setup(tmp_path)
    sink = StoreSink(store)
    ctx = AgentContext(run_id="r2", user_id="u1", subagent="worker1")

    w = await reg.dispatch("write_file", {"path": "s.txt", "content": "x"}, ctx)
    await _capture(sink, ctx, w)

    (row,) = store.list_actions("r2", "u1")
    assert row.subagent == "worker1"


async def test_todo_change_and_unknown_events(tmp_path):
    ws, store, reg, _ = _setup(tmp_path)
    sink = StoreSink(store)
    ctx = AgentContext(run_id="r3", user_id="u1")

    await sink.on_event(
        ctx,
        {
            "type": "todo_change",
            "old": [],
            "new": [{"content": "a", "status": "pending"}],
        },
    )
    await sink.on_event(ctx, {"type": "assistant", "content": "hi"})
    await sink.on_event(ctx, {"type": "tool_result", "ok": True})

    (row,) = store.list_actions("r3", "u1")
    assert row.kind == "todo_replace" and row.target == "todos"
    assert row.old_value == [] and row.new_value is not None


async def test_capture_actions_false_and_big_values_spill(tmp_path):
    ws, store, reg, _ = _setup(tmp_path)
    off = StoreSink(store, capture_actions=False)
    ctx = AgentContext(run_id="r4", user_id="u1")
    w = await reg.dispatch(
        "write_file", {"path": "big.txt", "content": "x" * 20_000}, ctx
    )
    await _capture(off, ctx, w)
    assert store.list_actions("r4", "u1") == []

    on = StoreSink(store)  # default capture_actions=True
    await _capture(on, ctx, w)
    (row,) = store.list_actions("r4", "u1")
    # >8KB values spilled to a blob and rehydrated transparently on read
    assert row.new_value == "x" * 20_000
    assert (store.root / "blobs").is_dir()


async def test_undo_run_reverts_stored_actions_end_to_end(tmp_path):
    ws, store, reg, reverters = _setup(tmp_path)
    sink = StoreSink(store)
    ctx = AgentContext(run_id="r5", user_id="u1")
    ws.write("d.txt", "pre\n")

    w = await reg.dispatch("write_file", {"path": "a.txt", "content": "v1"}, ctx)
    await _capture(sink, ctx, w)
    e = await reg.dispatch(
        "edit_file", {"path": "a.txt", "old_text": "v1", "new_text": "v2"}, ctx
    )
    await _capture(sink, ctx, e)
    p = await reg.dispatch(
        "apply_patch",
        {
            "patch_text": (
                "*** Begin Patch\n"
                "*** Add File: n.txt\n+new\n"
                "*** Update File: d.txt\n@@\n-pre\n+post\n"
                "*** Delete File: d.txt\n"
                "*** End Patch"
            )
        },
        ctx,
    )
    assert p.ok
    await _capture(sink, ctx, p)
    assert ws.read("a.txt") == "v2" and ws.exists("n.txt")
    assert not ws.exists("d.txt")

    host = AgentHost(reg, LLMConfig(model="m", base_url="http://x", api_key="k"), store)
    report = await undo_run(host, "r5", "u1", reverters=reverters)
    assert report.ok and report.reverted == 5
    # newest-first restoration: delete restored, edit back, creations removed
    assert ws.read("d.txt") == "pre\n"
    assert not ws.exists("a.txt")  # created during the run → creation undone
    assert not ws.exists("n.txt")

    # rows were marked reverted → a second undo is a no-op
    report2 = await undo_run(host, "r5", "u1", reverters=reverters)
    assert report2.reverted == 0


async def test_agent_host_default_assembly_persists_actions(tmp_path):
    """Zero host code: AgentHost.run attaches StoreSink, the runtime forwards
    each ui event, and the write lands as an undoable stored action."""
    import json

    from lithe.runtime import EventType

    ws, store, reg, reverters = _setup(tmp_path)

    class _Transport:
        def __init__(self):
            self.calls = 0

        async def complete(self, client, **kw):
            self.calls += 1
            if self.calls == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps(
                                    {"path": "a.txt", "content": "hi"}
                                ),
                            },
                        }
                    ],
                    "usage": {},
                }
            return {"content": "done", "tool_calls": [], "usage": {}}

        async def stream(self, client, **kw):  # pragma: no cover - unused
            yield {}

    host = AgentHost(
        reg,
        LLMConfig(model="m", base_url="x", api_key="k", transport=_Transport()),
        store,
    )
    ctx = AgentContext(run_id="r6", user_id="u1")
    events = [e async for e in host.run(ctx, "write it")]
    assert any(e["type"] == EventType.DONE and e["status"] == "done" for e in events)
    assert ws.read("a.txt") == "hi"

    rows = store.list_actions("r6", "u1")
    assert [(r.kind, r.target, r.new_value) for r in rows] == [
        ("file_write", "a.txt", "hi")
    ]

    report = await undo_run(host, "r6", "u1", reverters=reverters)
    assert report.ok and report.reverted == 1
    assert not ws.exists("a.txt")
