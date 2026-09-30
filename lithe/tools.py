"""Tool registry, dispatch, and the tool result contract.

Hosts register their tools (spec + handler + optional reverter + category); the
registry produces the OpenAI function spec list for a given mode, dispatches
calls, and feeds the undo engine its reverters. The kernel never hardcodes a
toolset — a host's tool list is just a series of ``register`` calls.

A tool handler returns a :class:`ToolResult` (ok / summary / content / ui). The
``ui`` list carries arbitrary domain events (file diffs, guidance proposals,
library changes) that the runtime forwards to sinks/frontend verbatim — that is
how a host surfaces business-specific UI without the kernel knowing about it.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Awaitable, Callable

from lithe.context import AgentContext
from lithe.modes import AgentMode, ToolCategory, categories_for

# A tool handler runs against the context + parsed args and yields a result.
ToolHandler = Callable[[AgentContext, dict], Awaitable["ToolResult"]]

# A middleware runs before the handler (after argument validation). Returning
# a ToolResult short-circuits the call (deny / request confirmation / audit
# substitute); returning None lets the call proceed.
ToolMiddleware = Callable[[AgentContext, str, dict], Awaitable["ToolResult | None"]]

# JSON-Schema "type" names → python checks for minimal argument validation.
# bool is excluded from number/integer (python bools are ints — a model passing
# true where a number is required must be corrected, not silently accepted).
_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _matches_type(value: Any, want: Any) -> bool:
    if isinstance(want, list):  # e.g. ["string", "null"]
        return any(_matches_type(value, w) for w in want)
    check = _TYPE_CHECKS.get(want) if isinstance(want, str) else None
    return check(value) if check else True  # unknown/missing type: don't guess


def validate_args(parameters: dict, args: dict) -> str | None:
    """Minimal JSON-Schema check of one tool call's arguments.

    Covers exactly the two failure classes a model most often makes and can
    self-correct from feedback: a declared ``required`` property missing, and a
    top-level property whose declared ``type`` the value violates. Anything
    subtler (nested schemas, refs, unions of shapes) is the handler's business.

    Returns a human-readable error for the model, or ``None`` when valid /
    when the schema is too loose to check (non-dict, no properties).
    """
    if not isinstance(parameters, dict) or not isinstance(args, dict):
        # Non-dict args cannot be checked property-by-property — and callers
        # feed violations back to the model, so say what was wrong instead
        # of raising (dispatch treats this path as a failed call).
        if not isinstance(args, dict):
            return f"参数应为 JSON 对象，得到 {type(args).__name__}"
        return None
    problems: list[str] = []
    for name in parameters.get("required") or ():
        if isinstance(name, str) and name not in args:
            problems.append(f"缺少必填参数 {name}")
    props = parameters.get("properties")
    if isinstance(props, dict):
        for name, decl in props.items():
            if name not in args or not isinstance(decl, dict):
                continue
            want = decl.get("type")
            if want is not None and not _matches_type(args[name], want):
                problems.append(f"参数 {name} 应为 {want}，"
                                f"得到 {type(args[name]).__name__}")
    return "；".join(problems) if problems else None


@dataclass
class ToolSpec:
    """Declarative description of one tool (mirrors an OpenAI function spec)."""
    name: str
    description: str
    parameters: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}})
    category: ToolCategory = ToolCategory.READ
    # Validate dispatched args against `parameters` (required present + declared
    # top-level types) and feed violations back to the model. Set False for
    # tools with deliberately loose/hostile schemas.
    validate: bool = True
    # Per-call execution timeout in seconds. None (default) waits forever —
    # set it on any tool that can hang (external services, MCP calls) so one
    # stuck call cannot freeze the whole step; on timeout the handler task is
    # cancelled and the model receives an error tool result.
    timeout: float | None = None

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """Standard contract every tool handler returns."""
    ok: bool
    summary: str
    content: str = ""
    ui: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "summary": self.summary,
                "content": self.content, "ui": self.ui}

    @classmethod
    def from_dict(cls, d: dict) -> ToolResult:
        return cls(ok=bool(d.get("ok", True)),
                   summary=d.get("summary", ""),
                   content=d.get("content", ""),
                   ui=list(d.get("ui") or []))


def _error_result(msg: str) -> ToolResult:
    return ToolResult(ok=False, summary="工具调用失败", content=msg)


@dataclass
class _Entry:
    spec: ToolSpec
    handler: ToolHandler
    reverter: Callable | None
    revert_kind: str = ""


class ToolRegistry:
    """Holds the host's tools and drives dispatch + mode filtering + undo wiring."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._middlewares: list[ToolMiddleware] = []

    def add_middleware(self, middleware: ToolMiddleware) -> None:
        """Install a pre-dispatch middleware: ``async (ctx, name, args)``.

        Middlewares run in installation order after argument validation and
        before the handler. A returned :class:`ToolResult` short-circuits the
        call (its content goes straight back to the model); ``None`` lets the
        next middleware / the handler run. This is the hook for audit logging,
        quotas, and human-in-the-loop confirmation of WRITE tools, e.g.::

            async def confirm_writes(ctx, name, args):
                if registry.spec(name).category is ToolCategory.WRITE:
                    return ToolResult(False, "待确认",
                                      f"工具 {name} 需要用户确认后才会执行。")
                return None

            registry.add_middleware(confirm_writes)
        """
        self._middlewares.append(middleware)

    def register(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        reverter: Callable | None = None,
        revert_kind: str | None = None,
    ) -> None:
        """Register one tool.

        ``reverter`` feeds the UndoEngine. It is keyed by *undo kind* — the
        ``Action.kind`` the reverter serves — which defaults to the tool name
        but usually differs for bundled tools (``write_file``'s reverter
        serves ``file_write`` actions, because the undo record's kind comes
        from the ``file_change`` event, not the tool name). Pass
        ``revert_kind="file_write"`` style keys so the registry's default
        :meth:`reverters` map lines up with stored actions.
        """
        if spec.name in self._entries:
            raise ValueError(f"tool already registered: {spec.name}")
        self._entries[spec.name] = _Entry(spec, handler, reverter,
                                          revert_kind or spec.name)

    def unregister(self, name: str) -> bool:
        """Remove one tool (e.g. an MCP server's tools after the server was
        dropped from config). Its reverter, if any, goes with it. Returns
        True when the name was registered."""
        return self._entries.pop(name, None) is not None

    def names(self) -> list[str]:
        return list(self._entries)

    def spec(self, name: str) -> ToolSpec | None:
        entry = self._entries.get(name)
        return entry.spec if entry else None

    def reverters(self) -> dict[str, Callable]:
        """``{undo kind: reverter}`` for the UndoEngine.

        Keys are the ``revert_kind`` each tool registered with (defaulting to
        the tool name), so the map lines up with stored action kinds —
        ``register_file_tools`` registers ``file_write`` / ``file_edit``
        keys, matching the rows a ``StoreSink`` writes.
        """
        return {e.revert_kind: e.reverter
                for e in self._entries.values() if e.reverter}

    def specs_for_mode(
        self,
        mode: AgentMode | str = AgentMode.AUTONOMOUS,
        disabled: frozenset[str] | None = None,
    ) -> list[dict]:
        allowed = categories_for(mode)
        dis = disabled or frozenset()
        return [e.spec.to_openai() for name, e in self._entries.items()
                if e.spec.category in allowed and name not in dis]

    async def dispatch(self, name: str, args: dict | Any, ctx: AgentContext) -> ToolResult:
        entry = self._entries.get(name)
        if entry is None:
            return _error_result(f"未知工具：{name}")
        if name in ctx.disabled_tools:
            return _error_result(f"工具 {name} 已被停用")
        if not isinstance(args, dict):
            try:
                args = json.loads(args) if args else {}
            except Exception:  # noqa: BLE001 — unparseable → error result
                args = None
            if not isinstance(args, dict):
                # This is the public dispatch API: hosts and tests call it
                # directly with raw values. A JSON string that parses to a
                # non-object (or any junk) previously escaped as a TypeError
                # from validate_args — outside the handler try — instead of
                # the failed-tool-result contract everything else follows.
                return ToolResult(
                    False, "参数校验失败",
                    f"工具 {name} 的参数应为 JSON 对象，"
                    f"得到 {type(args).__name__ if args is not None else '无法解析的 JSON'}。"
                    f"请修正后重新调用。")
        if entry.spec.validate:
            problem = validate_args(entry.spec.parameters, args)
            if problem is not None:
                return ToolResult(False, "参数校验失败",
                                  f"工具 {name} 参数无效：{problem}。"
                                  f"请修正参数后重新调用。")
        try:
            for mw in self._middlewares:
                verdict = await mw(ctx, name, args)
                if verdict is None:
                    continue
                if isinstance(verdict, dict):
                    verdict = ToolResult.from_dict(verdict)
                return verdict
            handler = entry.handler(ctx, args)
            if entry.spec.timeout is not None:
                result = await asyncio.wait_for(handler, entry.spec.timeout)
            else:
                result = await handler
        except asyncio.TimeoutError:
            return ToolResult(
                False, "执行超时",
                f"工具 {name} 超过 {entry.spec.timeout:g}s 未返回，已被中止。"
                f"请换一种做法或稍后重试。")
        except Exception as exc:  # noqa: BLE001
            return _error_result(f"{name} 执行异常：{exc}")
        if isinstance(result, dict):
            result = ToolResult.from_dict(result)
        return result
