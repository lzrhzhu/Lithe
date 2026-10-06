"""Small protocol and control helpers used by the ReAct runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import uuid

log = logging.getLogger("lithe.runtime")


def parse_args(tc: dict) -> tuple[dict, str | None]:
    """Parse tool-call arguments to an object, returning errors for the model."""
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return {}, None
    raw = fn.get("arguments", "")
    if isinstance(raw, dict):
        return raw, None
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return {}, f"{exc}（原文片段：{raw[:120]!r}）"
    if not isinstance(parsed, dict):
        return {}, f"arguments 应为 JSON 对象，得到 {type(parsed).__name__}"
    return parsed, None


def stop_triggered(stop) -> bool:
    """Accept event-like handles, callbacks, and truthy stop values."""
    if stop is None:
        return False
    is_set = getattr(stop, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    if callable(stop):
        return bool(stop())
    return bool(stop)


async def drain_inbox(inbox) -> list[str]:
    """Drain an asyncio or thread-safe steering queue without blocking."""
    if inbox is None:
        return []
    out: list[str] = []
    while True:
        try:
            out.append(inbox.get_nowait())
        except (asyncio.QueueEmpty, queue.Empty):
            return out
        except Exception:  # noqa: BLE001 — steering must not kill the run
            log.warning("steering inbox drain failed (ignored)", exc_info=True)
            return out


def normalize_assistant(msg: dict) -> dict:
    """Keep relevant assistant fields and ensure tool calls have stable IDs."""
    out = {"role": "assistant", "content": msg.get("content") or ""}
    reasoning = msg.get("reasoning")
    if reasoning:
        out["reasoning"] = reasoning
    tcs = msg.get("tool_calls")
    if tcs:
        norm: list[dict] = []
        for tc in tcs:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict):
                continue
            tc_id = tc.get("id")
            if not isinstance(tc_id, str) or not tc_id:
                tc_id = "call_" + uuid.uuid4().hex[:20]
                tc["id"] = tc_id
            norm.append({
                "id": tc_id,
                "type": tc.get("type", "function"),
                "function": {
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments", ""),
                },
            })
        if norm:
            out["tool_calls"] = norm
    return out
