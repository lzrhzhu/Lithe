"""Conversation memory: pure transforms that turn stored turns back into model
context, with no knowledge of where they are stored.

A host implements :class:`MemoryProvider` (reading its own DB / files / vector
store) and uses these helpers to build the replay:

- :func:`replay_messages` — recent message rows → valid OpenAI chat messages
  (orphan tool calls/results reconciled, large tool outputs bounded head+tail).
- :func:`recap_text` — older turns beyond the replay window → one compressed
  assistant note, so long conversations keep their gist instead of being dropped.
- :func:`truncate_tool_result` — bound a single tool result's size.

None of these read storage; they only reshape data the host hands them.
"""
from __future__ import annotations

import json
from typing import Any, Protocol
from collections.abc import Callable

from lithe.context import AgentContext

DEFAULT_TOOL_RESULT_CAP = 8000
DEFAULT_REPLAY_LIMIT = 12


def truncate_tool_result(content: str, cap: int = DEFAULT_TOOL_RESULT_CAP) -> str:
    """Bound a tool result's size for context replay, keeping head and tail.

    Code stdout / file reads often have the meaningful bit at the end; keeping
    the tail — not just the head — lets the model recall what actually happened.
    """
    if len(content) <= cap:
        return content
    keep = cap // 2
    return (content[:keep]
            + f"\n…[已截断：保留首尾，省略中间，共 {len(content)} 字符]…\n"
            + content[-keep:])


def recap_text(older_runs: list[dict], acts_by_run: dict[str, list[str]]) -> str:
    """Compact recap of turns beyond the faithful-replay window.

    ``older_runs`` carry ``run_id`` / ``task`` / ``final`` / ``status``;
    ``acts_by_run`` maps each ``run_id`` to its pre-computed mutation labels
    (the host decides how to label actions). Each turn becomes one line; the
    whole block is meant to be prepended as a single assistant note.
    """
    if not older_runs:
        return ""
    lines = ["【更早对话回顾（已压缩，仅作记忆，无需回复）】"]
    for r in older_runs:
        task = (r.get("task") or "").strip()
        ans = (r.get("final") or "").strip()
        if not ans and r.get("status") == "failed":
            ans = "（该轮出错未完成）"
        labels = [lb for lb in acts_by_run.get(r.get("run_id"), []) if lb]
        parts = [f"用户问：{task}"]
        if ans:
            parts.append(f"回复摘要：{ans[:300]}")
        if labels:
            parts.append("所做改动：" + "；".join(labels))
        lines.append("• " + " ｜ ".join(parts))
    return "\n".join(lines)


def _parse_tool_calls(raw: Any) -> list[dict] | None:
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, list):
        return None
    return [
        {
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {
                "name": (tc.get("function") or {}).get("name"),
                "arguments": (tc.get("function") or {}).get("arguments", ""),
            },
        }
        for tc in raw if isinstance(tc, dict)
    ]


def _parse_reasoning(raw: Any) -> list[dict] | None:
    """Stored reasoning items → list of raw item dicts (``None`` when absent).

    Hosts may persist them as a JSON string (one column) or as a parsed list;
    anything malformed degrades to ``None`` — a replay without reasoning is
    valid, just chain-less. Non-dict entries are dropped, and only known
    reasoning item shapes survive (Responses ``reasoning`` items; Messages
    ``thinking`` / ``redacted_thinking`` blocks) so a corrupted row cannot
    smuggle arbitrary input items into the next request.
    """
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, list):
        return None
    items = [r for r in raw
             if isinstance(r, dict)
             and r.get("type") in ("reasoning", "thinking", "redacted_thinking")]
    return items or None


def reasoning_summary_text(reasoning: list[dict] | None) -> str:
    """Human-readable digest of reasoning output items.

    OpenAI-shaped items carry ``summary`` as a list of ``{type:
    summary_text, text}`` blocks; some gateways send a plain string. The
    encrypted chain itself is opaque — only this digest is displayable.
    Messages-protocol ``thinking`` blocks carry the chain as plain text
    under ``thinking`` (``redacted_thinking`` stays opaque); that text is
    displayable too.
    """
    parts: list[str] = []
    for item in reasoning or []:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        if isinstance(summary, str):
            if summary.strip():
                parts.append(summary.strip())
        elif isinstance(summary, list):
            for s in summary:
                if isinstance(s, dict) and (s.get("text") or "").strip():
                    parts.append(s["text"].strip())
        text = item.get("thinking")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


def replay_messages(rows: list[dict], *, recap: str = "",
                    tool_result_cap: int = DEFAULT_TOOL_RESULT_CAP) -> list[dict]:
    """Turn stored message rows into the exact OpenAI chat format the model expects.

    ``rows`` are ordered (oldest first) dicts with ``role`` / ``content`` and,
    for assistant rows, ``tool_calls`` (a JSON string or already-parsed list)
    and ``reasoning`` (raw Responses reasoning items, JSON string or list);
    for tool rows, ``tool_call_id``. Assistant entries keep their
    ``tool_calls`` / ``reasoning`` and tool results come back as
    ``{role:'tool', tool_call_id, content}``.

    Large tool results are bounded head+tail. The list is reconciled so every
    ``tool_call`` has a matching tool result (orphans are dropped), keeping the
    payload valid for the chat-completions API — including at the window's
    start: a tool row whose calling assistant turn fell outside the host's
    window is dropped rather than sent as an orphan first message the API
    would reject. When ``recap`` is given it is prepended as an assistant
    note summarizing older turns.
    """
    raw: list[dict] = []
    # Call ids emitted by assistant rows INSIDE this window: a tool row is
    # only replayable when its caller is present (legacy positional pairing
    # draws from the same set).
    seen_call_ids: set[Any] = set()
    pending_ids: list[Any] = []
    for r in rows:
        role = r.get("role")
        if role == "user":
            raw.append({"role": "user", "content": r.get("content") or ""})
            pending_ids = []
        elif role == "assistant":
            tcs = _parse_tool_calls(r.get("tool_calls"))
            msg: dict[str, Any] = {"role": "assistant", "content": r.get("content") or ""}
            reasoning = _parse_reasoning(r.get("reasoning"))
            if reasoning:
                msg["reasoning"] = reasoning
            if tcs:
                msg["tool_calls"] = tcs
                pending_ids = [tc.get("id") for tc in tcs]
                seen_call_ids.update(tc.get("id") for tc in tcs)
            else:
                pending_ids = []
            raw.append(msg)
        elif role == "tool":
            tcid = r.get("tool_call_id")
            if not tcid and pending_ids:
                tcid = pending_ids.pop(0)  # legacy rows: pair by position
            if not tcid or tcid not in seen_call_ids:
                # No id, or the calling assistant row sits outside the
                # window: an orphan tool message (possibly the very first
                # message of the replay) would 400 the whole request.
                continue
            content = truncate_tool_result(r.get("content") or "", tool_result_cap)
            raw.append({"role": "tool", "tool_call_id": tcid, "content": content})

    ids_with_result = {m["tool_call_id"] for m in raw
                       if m["role"] == "tool" and m.get("tool_call_id")}
    out: list[dict] = []
    if recap:
        out.append({"role": "assistant", "content": recap})
    for m in raw:
        if m["role"] == "tool":
            if m.get("tool_call_id"):
                out.append(m)
        elif m["role"] == "assistant" and m.get("tool_calls"):
            kept = [tc for tc in m["tool_calls"] if tc.get("id") in ids_with_result]
            if kept:
                m["tool_calls"] = kept
                out.append(m)
            elif (m["content"] or "").strip():
                out.append({"role": "assistant", "content": m["content"]})
        else:
            out.append(m)
    return out


class MemoryProvider(Protocol):
    """Host-supplied source of prior conversation turns for memory replay.

    Implementations read whatever store the host uses (a DB, files, a vector
    index) and return OpenAI-format messages to prepend as memory, or an empty
    list for a fresh single-shot call. Hosts typically compose this from
    :func:`replay_messages` + :func:`recap_text` over their stored rows.
    """

    async def history(self, ctx: AgentContext, *,
                      limit: int | None = None) -> list[dict]: ...


def window_with_recap(run_rows: list[dict], recent_msg_rows: list[dict],
                      older_action_rows: list[dict], *,
                      label_fn: Callable[[dict], str | None],
                      limit: int = DEFAULT_REPLAY_LIMIT,
                      tool_result_cap: int = DEFAULT_TOOL_RESULT_CAP,
                      ) -> list[dict]:
    """Replay the most recent ``limit`` turns faithfully, summarizing older ones.

    Mirrors a host's "conversation messages" build minus its SQL: split the
    ordered ``run_rows`` into ``recent`` (last *limit*) and ``older``; build a
    :func:`recap_text` block from the older turns' actions (labeled by the
    host-supplied ``label_fn``); then :func:`replay_messages` the recent message
    rows with that recap prepended. Pure over rows — no storage access.
    """
    older = run_rows[:-limit] if len(run_rows) > limit else []
    recap = ""
    if older:
        acts_by_run: dict[Any, list[str]] = {}
        for a in older_action_rows:
            lbl = label_fn(a)
            if lbl:
                acts_by_run.setdefault(a.get("run_id"), []).append(lbl)
        recap = recap_text(older, acts_by_run)
    return replay_messages(recent_msg_rows, recap=recap,
                           tool_result_cap=tool_result_cap)


def run_timeline(run: dict, messages: list[dict], actions: list[dict], *,
                 tool_category: dict[str, str],
                 action_category: Callable[[dict], str | None],
                 action_events_fn: Callable[[dict], list[dict]],
                 display_cap: int = 20000) -> list[dict]:
    """Reconstruct one turn's ordered event timeline from its messages + actions.

    Generic mirror of a host's "conversation events" build: emits the user
    prompt, then for each assistant step its text + tool calls, each tool's raw
    result, and the side-effect events (file/guidance/...) that tool produced.
    When a tool message carries ``meta.ui`` (captured verbatim at run time) it is
    replayed exactly and its matched action is discarded; otherwise side-effects
    are rebuilt from ``actions`` via ``action_events_fn`` and matched to the tool
    call by category. Unmatched side-effects flush at the turn's end.

    Host hooks (domain only):
      - ``tool_category``  : tool name → category bucket (e.g. ``write_file→file``)
      - ``action_category``: an action → its bucket (e.g. ``file_write→file``);
        ``None`` means "never match a tool call" (flushed at end)
      - ``action_events_fn``: an action → its UI event list (``[]`` to drop)
    """
    events: list[dict] = []
    task = (run.get("task") or "").strip()
    if task:
        events.append({"type": "user", "text": task})

    queues: dict[str | None, list[list[dict]]] = {}
    for a in actions:
        evs = action_events_fn(a)
        if not evs:
            continue
        queues.setdefault(action_category(a), []).append(evs)

    for m in messages:
        role = m.get("role")
        if role == "assistant":
            reasoning = _parse_reasoning(m.get("reasoning"))
            if reasoning:
                digest = reasoning_summary_text(reasoning)
                if digest:
                    events.append({"type": "reasoning", "text": digest})
            content = m.get("content") or ""
            if content.strip():
                events.append({"type": "assistant", "text": content})
            raw = m.get("tool_calls")
            tcs = None
            if raw:
                try:
                    tcs = json.loads(raw) if isinstance(raw, str) else raw
                except (ValueError, TypeError):
                    tcs = None
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args else {}
                        except (ValueError, TypeError):
                            args = {}
                    elif not isinstance(args, dict):
                        args = {}
                    events.append({"type": "tool_call", "id": tc.get("id"),
                                   "name": fn.get("name"), "args": args})
        elif role == "tool":
            content = m.get("content") or ""
            meta = m.get("meta")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (ValueError, TypeError):
                    meta = None
            meta = meta if isinstance(meta, dict) else None
            meta_ui = meta.get("ui") if meta else None
            has_meta_ui = isinstance(meta_ui, list) and len(meta_ui) > 0
            # Rows recorded by this kernel always carry meta.ok; a row
            # without meta predates that (or came from a foreign writer),
            # and its ok is UNKNOWN — reported as None instead of sniffing
            # the content for a particular error-message prefix, which
            # coupled replay to a display string.
            ok = meta.get("ok") if meta is not None else None
            if len(content) > display_cap:
                content = content[:display_cap] + "\n…[输出过长已截断]"
            tr: dict[str, Any] = {"type": "tool_result", "id": m.get("tool_call_id"),
                                  "name": m.get("tool_name"), "ok": ok,
                                  "content": content}
            if meta and meta.get("summary"):
                tr["summary"] = meta["summary"]
            events.append(tr)
            cat = tool_category.get(m.get("tool_name") or "")
            if has_meta_ui:
                events.extend(meta_ui)
                if cat and queues.get(cat):
                    queues[cat].pop(0)
            elif cat and ok is not False and queues.get(cat):
                # Unknown ok (legacy row) still rebuilds side-effects — the
                # pre-meta behavior — while a recorded failure still skips.
                events.extend(queues[cat].pop(0))

    for evs_list in queues.values():
        for evs in evs_list:
            events.extend(evs)
    return events
