"""lithe.bundles.todos: per-scope task list with replace semantics, a
todo_change UI contract, JSON persistence and an undo reverter — pinned with
no host application and no DB."""
from __future__ import annotations

import pytest

from lithe import Action, AgentContext, ToolRegistry, UndoEngine
from lithe.bundles.todos import (
    JsonTodoStore, TodoStore, register_todo_tools, todos_block,
)


# --- TodoStore: pure logic ---

def test_replace_returns_old_and_normalizes():
    store = TodoStore()
    old, new = store.replace([
        {"content": "write intro", "status": "in_progress"},
        {"content": "  trim figs  ", "status": "pending", "priority": "high"},
    ])
    assert old == []
    assert len(new) == 2
    assert new[0]["status"] == "in_progress"
    assert new[1]["content"] == "trim figs"          # stripped
    assert new[1]["priority"] == "high"
    assert all("id" in it for it in new)             # id assigned


def test_replace_rejects_bad_status_and_empty_content():
    store = TodoStore()
    with pytest.raises(ValueError):
        store.replace([{"content": "x", "status": "done"}])      # bad status
    with pytest.raises(ValueError):
        store.replace([{"content": "   ", "status": "pending"}])  # empty content


def test_replace_enforces_max_todos():
    store = TodoStore(max_todos=2)
    store.replace([{"content": "a", "status": "pending"},
                   {"content": "b", "status": "pending"}])
    with pytest.raises(ValueError):
        store.replace([{"content": "a", "status": "pending"},
                       {"content": "b", "status": "pending"},
                       {"content": "c", "status": "pending"}])


def test_replace_allows_at_most_one_in_progress_and_is_atomic():
    store = TodoStore([{"content": "existing", "status": "pending"}])
    with pytest.raises(ValueError, match="最多只能有一项"):
        store.replace([
            {"content": "first", "status": "in_progress"},
            {"content": "second", "status": "in_progress"},
        ])
    assert [item["content"] for item in store.list()] == ["existing"]


def test_max_todos_must_be_positive():
    with pytest.raises(ValueError, match="正整数"):
        TodoStore(max_todos=0)


def test_initial_and_restored_rows_are_sanitized_and_limited():
    rows = [
        {"id": "a", "content": "first", "status": "in_progress"},
        {"id": "b", "content": "second", "status": "in_progress"},
        {"id": "c", "content": "third", "status": "pending"},
    ]
    store = TodoStore(rows, max_todos=2)
    assert [item["status"] for item in store.list()] == ["in_progress", "pending"]
    store.restore(rows)
    assert [item["status"] for item in store.list()] == ["in_progress", "pending"]
    assert len(store.list()) == 2


def test_replace_is_total_not_delta():
    store = TodoStore()
    store.replace([{"content": "a", "status": "pending"},
                   {"content": "b", "status": "pending"},
                   {"content": "c", "status": "pending"}])
    # second call omits "b" -> it is gone, not merged
    old, new = store.replace([{"content": "a", "status": "completed"},
                              {"content": "c", "status": "pending"}])
    assert [it["content"] for it in new] == ["a", "c"]
    assert [it["content"] for it in old] == ["a", "b", "c"]


def test_restore_reverses_a_replace():
    store = TodoStore()
    store.replace([{"content": "a", "status": "pending"}])
    old, _ = store.replace([{"content": "b", "status": "in_progress"}])
    store.restore(old)
    assert [it["content"] for it in store.list()] == ["a"]


def test_empty_list_is_allowed():
    store = TodoStore([{"content": "a", "status": "pending"}])
    old, new = store.replace([])
    assert new == []
    assert len(old) == 1


def test_to_block_renders_marks_and_counts():
    store = TodoStore()
    store.replace([
        {"content": "todo", "status": "pending"},
        {"content": "now", "status": "in_progress"},
        {"content": "done", "status": "completed"},
        {"content": "nope", "status": "cancelled"},
    ])
    block = store.to_block()
    assert "[ ]" in block and "[~]" in block and "[x]" in block and "[-]" in block
    assert "1/4 已完成" in block


def test_to_block_empty_placeholder():
    assert TodoStore().to_block() == "（任务清单为空）"


def test_todos_block_helper_matches_method():
    store = TodoStore([{"content": "a", "status": "pending"}])
    assert todos_block(store) == store.to_block()


# --- JsonTodoStore: file persistence ---

def test_json_store_roundtrips(tmp_path):
    p = tmp_path / "todos.json"
    s1 = JsonTodoStore(p)
    s1.replace([{"content": "a", "status": "pending"},
                {"content": "b", "status": "completed"}])
    # a fresh instance over the same file sees the saved list
    s2 = JsonTodoStore(p)
    assert [it["content"] for it in s2.list()] == ["a", "b"]
    assert s2.list()[1]["status"] == "completed"


def test_json_store_missing_file_starts_empty(tmp_path):
    s = JsonTodoStore(tmp_path / "nope.json")
    assert s.list() == []


def test_json_store_repairs_multiple_active_items_and_caps_loaded_rows(tmp_path):
    import json

    path = tmp_path / "todos.json"
    path.write_text(json.dumps({"items": [
        {"id": "a", "content": "first", "status": "in_progress"},
        {"id": "b", "content": "second", "status": "in_progress"},
        {"id": "c", "content": "third", "status": "pending"},
    ]}), encoding="utf-8")
    store = JsonTodoStore(path, max_todos=2)
    items = store.list()
    assert [item["status"] for item in items] == ["in_progress", "pending"]
    assert len(items) == 2


def test_json_store_restore_persists(tmp_path):
    p = tmp_path / "todos.json"
    s = JsonTodoStore(p)
    s.replace([{"content": "a", "status": "pending"}])
    old, _ = s.replace([{"content": "b", "status": "in_progress"}])
    s.restore(old)
    assert [it["content"] for it in JsonTodoStore(p).list()] == ["a"]


# --- tools + undo reverter ---

def _registry_with(store):
    reg = ToolRegistry()
    reverters = register_todo_tools(reg, lambda ctx: store)
    return reg, reverters


async def test_update_todos_ui_contract_and_persistence():
    store = TodoStore()
    reg, _ = _registry_with(store)
    ctx = AgentContext(run_id="r", user_id="u")

    res = await reg.dispatch("update_todos",
                             {"todos": [{"content": "draft", "status": "in_progress"}]}, ctx)
    assert res.ok and res.ui[0]["type"] == "todo_change"
    assert res.ui[0]["old"] == []
    assert res.ui[0]["new"][0]["content"] == "draft"
    assert store.list()[0]["content"] == "draft"

    res2 = await reg.dispatch("update_todos",
                              {"todos": [{"content": "draft", "status": "completed"}]}, ctx)
    assert res2.ui[0]["old"][0]["status"] == "in_progress"
    assert res2.ui[0]["new"][0]["status"] == "completed"


async def test_update_todos_rejects_non_list():
    store = TodoStore()
    reg, _ = _registry_with(store)
    ctx = AgentContext(run_id="r", user_id="u")
    res = await reg.dispatch("update_todos", {"todos": "not a list"}, ctx)
    assert res.ok is False


async def test_update_todos_rejects_invalid_item():
    store = TodoStore()
    reg, _ = _registry_with(store)
    ctx = AgentContext(run_id="r", user_id="u")
    res = await reg.dispatch("update_todos",
                             {"todos": [{"content": "x", "status": "oops"}]}, ctx)
    assert res.ok is False and "status" in res.content


async def test_list_todos_reads_current():
    store = TodoStore([{"content": "a", "status": "pending"}])
    reg, _ = _registry_with(store)
    ctx = AgentContext(run_id="r", user_id="u")
    res = await reg.dispatch("list_todos", {}, ctx)
    assert res.ok and "a" in res.content


async def test_undo_reverter_restores_old_list():
    store = TodoStore()
    reg, reverters = _registry_with(store)
    ctx = AgentContext(run_id="r", user_id="u")
    await reg.dispatch("update_todos",
                       {"todos": [{"content": "a", "status": "pending"},
                                  {"content": "b", "status": "pending"}]}, ctx)
    # forward replace captured old=[], new=[a,b]
    await reg.dispatch("update_todos",
                       {"todos": [{"content": "a", "status": "completed"}]}, ctx)
    assert [it["content"] for it in store.list()] == ["a"]

    engine = UndoEngine(reverters)
    report = await engine.undo([
        Action(kind="todo_replace", target="todos", old_value=[], status="applied"),
        Action(kind="todo_replace", target="todos",
               old_value=[{"content": "a", "status": "pending"},
                          {"content": "b", "status": "pending"}], status="applied"),
    ], ctx)
    assert report.reverted == 2 and report.ok is True
    # newest-first: second action restored [a,b], then first restored []
    assert store.list() == []


async def test_todo_tools_filtered_by_mode():
    store = TodoStore()
    reg, _ = _registry_with(store)
    autonomous = {s["function"]["name"] for s in reg.specs_for_mode("autonomous")}
    assert autonomous == {"update_todos", "list_todos"}
    anchored = {s["function"]["name"] for s in reg.specs_for_mode("anchored")}
    assert anchored == {"list_todos"}          # write tool hidden when read-only


def test_update_todos_description_discourages_trivial_plans():
    reg, _ = _registry_with(TodoStore())
    update_spec = reg.spec("update_todos").to_openai()["function"]
    assert "单步操作" in update_spec["description"]
    assert "固定数量" in update_spec["description"]
    assert "完整" in update_spec["parameters"]["properties"]["todos"]["description"]
    assert "显示当前任务清单" in reg.spec("list_todos").to_openai()["function"]["description"]
