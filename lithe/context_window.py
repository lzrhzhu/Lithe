"""Conversation sizing and safe context compaction helpers."""
from __future__ import annotations

import json

from lithe.memory import truncate_tool_result

# Default mid-run context budget (~chars of messages incl. tool results).
# Generous enough that normal runs never notice; only genuinely huge
# conversations trigger trimming. None disables trimming entirely.
DEFAULT_CONTEXT_BUDGET = 400_000

# When over budget, old tool-result contents are shrunk head+tail through
# these caps (each pass smaller) until the conversation fits. Only `content`
# strings of tool messages shrink — roles/ids stay intact, so the payload
# remains structurally valid for the chat-completions API.
_TRIM_STAGES = (1200, 400, 100)
_TRIM_FLOOR = 60
_KEEP_RECENT_TOOL_MSGS = 2

# Escalation when shrinking cannot reach the budget (the overage is prose /
# reasoning / protected-recent content, none of which may shrink): whole old
# exchanges — one assistant turn plus its tool results, always as a complete
# unit — are dropped and replaced by a single omission note, so the next
# request still goes out under budget instead of a guaranteed window
# overflow. Dropped turns are already recorded with sinks; only the live
# model context loses them. The newest exchanges are never dropped.
_KEEP_RECENT_EXCHANGES = 2
_OMITTED_TURNS_NOTE = ("（系统注：为控制上下文长度，此前若干轮工具调用过程已从"
                       "上下文中移除；这些步骤的结论已体现在其后的对话内容中，"
                       "无需重复执行。）")


def _message_size(m: dict) -> int:
    n = 0
    content = m.get("content")
    if isinstance(content, str):
        n += len(content)
    elif isinstance(content, list):
        # Multimodal content: text blocks count as their text length, image
        # blocks by their payload length (an inline data: URL IS the image,
        # in chars). Previously list content counted as 0, so a multimodal
        # conversation under-reported its size and skewed the trim budget
        # and the chars-per-token calibration.
        for part in content:
            if isinstance(part, str):
                n += len(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    n += len(text)
                iu = part.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else None
                if isinstance(url, str):
                    n += len(url)
    tcs = m.get("tool_calls")
    if tcs:
        try:
            n += len(json.dumps(tcs, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            n += 1024
    # Reasoning items count toward the conversation's size (encrypted_content
    # blobs can be kilobytes) — but are never trimmed: dropping one orphans
    # the function_call it precedes and breaks the pass-back pairing.
    rs = m.get("reasoning")
    if rs:
        try:
            n += len(json.dumps(rs, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            n += 1024
    return n


def _context_size(messages: list[dict]) -> int:
    """Rough size of the outgoing conversation (chars, incl. tool_calls JSON).
    Used for the context-fullness indication and by :func:`_trim_context`."""
    return sum(_message_size(m) for m in messages)


def _exchange_units(messages: list[dict]) -> list[tuple[int, int]]:
    """``(start, end)`` index ranges of complete exchanges — one assistant
    message carrying ``tool_calls`` plus the contiguous tool results that
    answer it. Dropping such a range never orphans a call or a result."""
    units: list[tuple[int, int]] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                j += 1
            units.append((i, j))
            i = j
        else:
            i += 1
    return units


def _trim_context(messages: list[dict], *, budget: int,
                  keep_recent: int = _KEEP_RECENT_TOOL_MSGS) -> int:
    """Best-effort mid-run context compaction, in two tiers.

    Tier 1 shrinks old tool-result contents (head+tail, like memory replay)
    until the conversation fits ``budget`` chars. The newest ``keep_recent``
    tool results are never touched — they are what the model is actively
    working from. Structural validity is preserved by construction: only
    ``content`` strings change, so every assistant ``tool_call`` keeps its
    matching tool message.

    Tier 2 (escalation) fires when tier 1 cannot reach the budget — the
    overage then lives in content tier 1 must not touch (user/assistant
    prose, reasoning items, protected-recent tool results). Complete old
    exchanges are dropped oldest-first and replaced by one omission note
    (:data:`_OMITTED_TURNS_NOTE`), keeping the payload under budget instead
    of sending a request that overflows the model's window. System and user
    messages are never dropped, the newest ``_KEEP_RECENT_EXCHANGES``
    exchanges are protected, and — to avoid losing history for nothing —
    drops are committed only when they actually reach the budget; if even
    dropping every droppable exchange would stay over, nothing is dropped.

    Returns the conversation's size in chars afterwards (also when no trim
    was needed), so callers can keep a running total without rescanning.
    """
    total = _context_size(messages)
    if total <= budget:
        return total
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    shrinkable = tool_idx[:-keep_recent] if keep_recent else list(tool_idx)
    for cap in _TRIM_STAGES:
        for i in shrinkable:
            if total <= budget:
                return total
            m = messages[i]
            c = m.get("content")
            if isinstance(c, str) and len(c) > cap:
                new = truncate_tool_result(c, cap)
                total += len(new) - len(c)
                m["content"] = new
    for i in shrinkable:  # still over: hard floor, oldest first
        if total <= budget:
            return total
        m = messages[i]
        c = m.get("content")
        if isinstance(c, str) and len(c) > _TRIM_FLOOR:
            cut = c[:_TRIM_FLOOR] + "…[已压缩]"
            total += len(cut) - len(c)
            m["content"] = cut
    if total <= budget:
        return total

    # Tier 2: simulate dropping whole old exchanges; commit only if the
    # simulation actually reaches the budget.
    protected = set(tool_idx[-keep_recent:]) if keep_recent else set()
    units = _exchange_units(messages)
    droppable = units[:-_KEEP_RECENT_EXCHANGES] if _KEEP_RECENT_EXCHANGES \
        else units
    sim = total
    drop_ranges: list[tuple[int, int]] = []
    for start, end in droppable:
        if sim <= budget:
            break
        if any(i in protected for i in range(start, end)):
            break  # protection covers a suffix of exchanges: nothing older left
        sim -= sum(_message_size(messages[i]) for i in range(start, end))
        drop_ranges.append((start, end))
    if not drop_ranges or sim > budget:
        return total  # hopeless: overage lives in undroppable content
    first_drop = drop_ranges[0][0]
    drop_idxs = {i for start, end in drop_ranges for i in range(start, end)}
    # stale omission notes from an earlier pass carry no state; remove them
    # so the fresh note below stays unique
    drop_idxs.update(i for i, m in enumerate(messages)
                     if m.get("role") == "assistant" and not m.get("tool_calls")
                     and m.get("content") == _OMITTED_TURNS_NOTE)
    kept_before = sum(1 for i in range(first_drop) if i not in drop_idxs)
    new_messages = [m for i, m in enumerate(messages) if i not in drop_idxs]
    new_messages.insert(kept_before,
                        {"role": "assistant", "content": _OMITTED_TURNS_NOTE})
    messages[:] = new_messages
    return _context_size(messages)


