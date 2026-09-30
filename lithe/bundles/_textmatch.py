"""Shared fuzzy line-matching ladder for the file-editing tools.

The ladder locates a block of expected lines inside a file, trying
progressively weaker comparisons so a model's near-miss output (a stray
trailing space, missing indentation, typographic Unicode punctuation) still
lands instead of failing the edit:

1. exact equality
2. ignore trailing whitespace (``rstrip``)
3. ignore leading/trailing whitespace (``strip``)
4. fold typographic Unicode punctuation to ASCII look-alikes, then compare

Each pass scans the whole candidate range, so a weaker comparator never
shadows a stronger hit at a later offset. ``apply_patch``'s chunk seeking and
``edit_file``'s whole-line fallback both build on these primitives.
"""
from __future__ import annotations

import operator
from collections.abc import Callable


def normalize_unicode(s: str) -> str:
    """Fold common typographic punctuation to its ASCII look-alikes."""
    for src, dst in (("\u2018", "'"), ("\u2019", "'"), ("\u201a", "'"),
                     ("\u201b", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u201e", '"'), ("\u201f", '"'), ("\u2010", "-"),
                     ("\u2011", "-"), ("\u2012", "-"), ("\u2013", "-"),
                     ("\u2014", "-"), ("\u2015", "-"), ("\u2026", "..."),
                     ("\u00a0", " ")):
        s = s.replace(src, dst)
    return s


# Ordered exact → rstrip → strip → unicode-folded.
COMPARATORS: tuple[Callable[[str, str], bool], ...] = (
    operator.eq,
    lambda a, b: a.rstrip() == b.rstrip(),
    lambda a, b: a.strip() == b.strip(),
    lambda a, b: normalize_unicode(a.strip()) == normalize_unicode(b.strip()),
)


def try_match(lines: list[str], pattern: list[str], start: int,
              compare: Callable[[str, str], bool], eof: bool = False) -> int:
    """First ladder-position match of *pattern* in *lines* at/after *start*.

    With ``eof``, a tail-anchored match is preferred (for appending at end of
    file) before the forward scan. Returns ``-1`` when *compare* finds nothing.
    """
    n = len(pattern)
    if eof:
        from_end = len(lines) - n
        if from_end >= start and all(
                compare(lines[from_end + j], pattern[j]) for j in range(n)):
            return from_end
    for i in range(start, len(lines) - n + 1):
        if all(compare(lines[i + j], pattern[j]) for j in range(n)):
            return i
    return -1


def seek_sequence(lines: list[str], pattern: list[str], start: int = 0,
                  eof: bool = False) -> int:
    """Locate *pattern* via the comparator ladder; ``-1`` when nothing hits.

    ``eof`` only *prefers* a tail-anchored hit (it then falls back to a
    forward scan) — callers that need a hard tail anchor (patch
    ``*** End of File`` chunks) use :func:`seek_tail` instead."""
    if not pattern:
        return -1
    for compare in COMPARATORS:
        found = try_match(lines, pattern, start, compare, eof)
        if found != -1:
            return found
    return -1


def seek_tail(lines: list[str], pattern: list[str]) -> int:
    """Hard tail anchor: the ONLY acceptable position is the last
    ``len(pattern)`` lines of the file, tried through the comparator ladder.

    Returns that position, or ``-1`` when no comparator matches the tail —
    a weaker forward hit elsewhere in the file is never accepted. This is
    what ``*** End of File`` patch chunks need: "append at end of file"
    must fail loudly when the tail doesn't line up, not silently edit a
    look-alike site earlier in the file."""
    if not pattern:
        return -1
    pos = len(lines) - len(pattern)
    if pos < 0:
        return -1
    for compare in COMPARATORS:
        if all(compare(lines[pos + j], pattern[j]) for j in range(len(pattern))):
            return pos
    return -1


def line_span_hits(lines: list[str], pattern: list[str]) -> list[int]:
    """All non-overlapping match starts, strongest comparator that hits.

    Returns ``[]`` when no comparator matches anywhere. Strength has priority
    over coverage: hits from a weaker comparator are never merged in.
    """
    if not pattern:
        return []
    for compare in COMPARATORS:
        hits: list[int] = []
        i = 0
        while (found := try_match(lines, pattern, i, compare)) != -1:
            hits.append(found)
            i = found + len(pattern)
        if hits:
            return hits
    return []
