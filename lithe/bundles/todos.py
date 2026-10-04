"""Todos bundle: a per-scope task list the agent plans against, with undo.

Mirrors the workspace bundle's shape. A host opts in by giving a
``todos_for(ctx) -> TodoStore`` callable (it decides where each scope's list
lives and how it persists) and calling :func:`register_todo_tools`. The bundle
itself stores nothing: :class:`TodoStore` is the pure logic container and
:class:`JsonTodoStore` is the optional JSON-file persistence a host may
instantiate inside ``todos_for``.

The write tool (``update_todos``) is *replace-style*: the model rewrites the
whole list each call rather than mutating one entry. This avoids index drift
(a moved row breaks "update entry #3") and matches the TodoWrite convention.
Each call emits a ``todo_change`` UI event (old/new lists) for the frontend
and returns a ``todo_replace`` reverter for the host's
:class:`~lithe.actions.UndoEngine`.

A separate :func:`todos_block` helper renders the current list as a short
text block a host can splice into its system prompt or memory recap, so the
agent sees its own outstanding work each run instead of forgetting the list
exists.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from collections.abc import Callable

from lithe.context import AgentContext
from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec

STATUSES = ("pending", "in_progress", "completed", "cancelled")
PRIORITIES = ("low", "medium", "high")

_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]",
         "cancelled": "[-]"}

_MAX_CONTENT = 256
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _clean_content(content: str) -> str:
    """Collapse control chars/internal whitespace and cap length.

    A todo item is persisted to disk and re-rendered into the system prompt
    every run, so one item must neither break the prompt block's line
    structure (newlines/control chars) nor balloon the list size.
    """
    return " ".join(_CTRL_RE.sub(" ", content).split())[:_MAX_CONTENT]


def _norm_item(raw: dict) -> dict:
    """Validate + normalize one raw item dict from the model.

    ``id`` is (re)assigned by the store so a replace is always a clean new
    list; the model never needs to track ids. Status/priority are clamped to
    their allowed vocab; content must be a non-empty string and is
    whitespace-collapsed + capped at :data:`_MAX_CONTENT` chars.
    """
    if not isinstance(raw, dict):
        raise ValueError("每项任务必须是对象")
    content = _clean_content(str(raw.get("content", "")))
    if not content:
        raise ValueError("任务缺少 content（内容不能为空）")
    status = str(raw.get("status", "pending")).strip()
    if status not in STATUSES:
        raise ValueError(f"status 非法：{status}（应为 {'/'.join(STATUSES)}）")
    priority = str(raw.get("priority", "medium")).strip()
    if priority not in PRIORITIES:
        priority = "medium"
    return {"id": uuid.uuid4().hex[:8], "content": content,
            "status": status, "priority": priority}


def _sanitize_loaded(items: list) -> list[dict]:
    """Tolerantly load persisted rows and repair conflicting active statuses."""
    out: list[dict] = []
    has_in_progress = False
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        status = it.get("status")
        if status not in STATUSES:
            continue
        content = it.get("content")
        if not isinstance(content, str) or not _clean_content(content):
            continue
        if status == "in_progress":
            if has_in_progress:
                status = "pending"
            else:
                has_in_progress = True
        priority = it.get("priority", "medium")
        out.append({"id": it.get("id") if isinstance(it.get("id"), str)
                    and it["id"] else uuid.uuid4().hex[:8],
                    "content": _clean_content(content),
                    "status": status,
                    "priority": priority if priority in PRIORITIES
                    else "medium"})
    return out


class TodoStore:
    """Pure in-memory task-list container: validate, replace, restore, render.

    Holds a list of item dicts (``id`` / ``content`` / ``status`` /
    ``priority``). :meth:`replace` is the only mutation: it validates the new
    list and returns ``(old, new)`` so a tool can emit a UI event and a host
    can persist an undo action. :meth:`restore` reverses one such change
    (used by the ``todo_replace`` reverter). Subclasses override
    :meth:`_save` to persist (see :class:`JsonTodoStore`).
    """

    def __init__(self, items: list[dict] | None = None, *, max_todos: int = 20):
        if not isinstance(max_todos, int) or isinstance(max_todos, bool) or max_todos < 1:
            raise ValueError("max_todos 必须是正整数")
        self.max_todos = max_todos
        self._items = _sanitize_loaded(items or [])[:self.max_todos]

    def list(self) -> list[dict]:
        return [dict(i) for i in self._items]

    def replace(self, items: list[dict]) -> tuple[list[dict], list[dict]]:
        if not isinstance(items, list):
            raise ValueError("todos 必须是数组")
        if len(items) > self.max_todos:
            raise ValueError(f"任务过多：上限 {self.max_todos} 条，收到 {len(items)} 条")
        normalized = [_norm_item(it) for it in items]
        if sum(it["status"] == "in_progress" for it in normalized) > 1:
            raise ValueError("最多只能有一项任务处于 in_progress 状态")
        old = self.list()
        self._items = normalized
        self._save()
        return old, self.list()

    def restore(self, old: list[dict]) -> None:
        self._items = _sanitize_loaded(old or [])[:self.max_todos]
        self._save()

    def to_block(self) -> str:
        """Render the list as a compact text block for the system prompt.

        Empty list renders a placeholder so the agent knows the list exists
        but is clear, rather than reading nothing and assuming there is no
        such tool.
        """
        if not self._items:
            return "（任务清单为空）"
        lines = [f"{i}. {_MARK.get(it.get('status'), '[ ]')}"
                 f" {it.get('content', '')}"
                 for i, it in enumerate(self._items, 1)]
        done = sum(1 for it in self._items if it.get("status") == "completed")
        header = f"当前任务清单（{done}/{len(self._items)} 已完成）："
        legend = "图例：[ ]待办 [~]进行中 [x]完成 [-]取消"
        return header + "\n" + "\n".join(lines) + "\n" + legend

    def _save(self) -> None:
        """Persistence hook; overridden by file-backed subclasses."""


class JsonTodoStore(TodoStore):
    """:class:`TodoStore` backed by a JSON file (``{"items": [...]}``).

    Loaded once on construction; saved after every replace/restore. Missing
    or corrupt files start empty; malformed *items* (bad status, missing
    content — e.g. a partially written file or a future schema) are dropped
    on load rather than crashing every ``to_block`` render. Saves are atomic
    (temp file + ``os.replace``), so a crash mid-write can never truncate the
    previous list. A host builds one per scope inside its ``todos_for``:
    ``JsonTodoStore(storage / "agent" / uid / "todos.json")``.
    """

    def __init__(self, path: str | Path, *, max_todos: int = 20):
        self._path = Path(path)
        super().__init__(max_todos=max_todos)
        self._items = _sanitize_loaded(self._load())[:self.max_todos]

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError,
                UnicodeDecodeError):
            return []
        items = data.get("items") if isinstance(data, dict) else None
        return items if isinstance(items, list) else []

    def _save(self) -> None:
        payload = json.dumps({"items": self._items}, ensure_ascii=False,
                             indent=2)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f".{self._path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._path)
        finally:
            tmp.unlink(missing_ok=True)


def todos_block(store: TodoStore) -> str:
    """Module-level alias for :meth:`TodoStore.to_block` for prompt builders."""
    return store.to_block()


def _update_todos_params() -> dict:
    return {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": (
                    "可选的计划工具。普通问答、解释、单步操作和小改动不要使用；"
                    "仅当用户明确要求计划，或工作确有多个值得跟踪进度的独立阶段时使用。"
                    "使用时发送完整任务清单：这是整体替换而非增量，保留的任务都要包含；"
                    "条数按实际独立步骤确定，不凑固定数量。"),
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string",
                                    "description": "用一句话简述这项任务。"},
                        "status": {"type": "string", "enum": list(STATUSES),
                                   "description": (
                                       "pending = 未开始；in_progress = 正在进行"
                                       "（最多一项）；completed = 已完成；"
                                       "cancelled = 已放弃。")},
                        "priority": {"type": "string", "enum": list(PRIORITIES),
                                     "description": "可选，默认 medium。"},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["todos"],
    }


def register_todo_tools(
    registry: ToolRegistry,
    todos_for: Callable[[AgentContext], TodoStore],
) -> dict[str, Callable]:
    """Register ``update_todos`` + ``list_todos`` on *registry*.

    ``todos_for(ctx) -> TodoStore`` is host-supplied (it maps a context to
    that scope's list, deciding scope = project / student / run and where it
    is persisted). The write tool emits a ``todo_change`` UI event carrying
    old/new lists for the frontend.

    Returns ``{"todo_replace": reverter}`` for the host's
    :class:`~lithe.actions.UndoEngine` — keyed by mutation *kind* (not
    tool name), like the workspace bundle's ``file_write`` / ``file_edit``.
    The bundle stores nothing itself.
    """

    async def update_todos(ctx: AgentContext, args: dict):
        raw = args.get("todos")
        if not isinstance(raw, list):
            return ToolResult(False, "参数错误", "需要 todos 数组（完整任务列表）。")
        store = todos_for(ctx)
        try:
            old, new = store.replace(raw)
        except ValueError as exc:
            return ToolResult(False, "任务列表无效", str(exc))
        return ToolResult(
            True, f"任务清单已更新（{len(new)} 条）", store.to_block(),
            ui=[{"type": "todo_change", "old": old, "new": new}])

    async def list_todos(ctx: AgentContext, args: dict):
        return ToolResult(True, "任务清单", todos_for(ctx).to_block())

    def _revert(action, ctx: AgentContext) -> None:
        todos_for(ctx).restore(action.old_value)

    registry.register(
        ToolSpec("update_todos",
                  "可选的计划工具。普通问答、解释、单步操作和小改动不要调用；"
                  "仅当用户明确要求计划，或工作确有多个需要跟踪进度的独立阶段时使用。"
                  "替换已有清单前先用 list_todos 读取，并保留与本次无关的未完成事项；"
                  "每次发送完整清单（整体替换，不是增量）。每个独立步骤对应一条任务，"
                  "同时最多保持一项 in_progress，条数按实际步骤确定，不以 3 这类固定数量为目标。",
                  _update_todos_params(), ToolCategory.WRITE),
        update_todos, reverter=_revert, revert_kind="todo_replace")
    registry.register(
        ToolSpec("list_todos", "显示当前任务清单及各项状态。",
                 {"type": "object", "properties": {}}, ToolCategory.READ),
        list_todos)

    return {"todo_replace": _revert}
