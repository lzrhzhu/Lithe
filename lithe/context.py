"""Execution context passed to every tool handler and the runtime loop.

``AgentContext`` carries the few fields the kernel itself needs (``run_id``,
``user_id``, disabled tools, the current subagent tag) plus an open ``extra``
mapping for host-specific fields (a thesis app stashes ``thesis_id`` /
``thread_id`` there). It supports mapping-style access (``ctx["user_id"]``,
``ctx.get("thesis_id")``) so hosts can migrate tool code gradually from a raw
``dict`` context.

``shared`` is per-run, cross-context state, shared *by reference* with derived
contexts (a subagent's context gets the orchestrator's very same dict). It is
the vehicle for in-run coordination the kernel and bundles need to see from
every context of one run — the workspace stale-file guard's revision map, the
current run's cancellation handle — as opposed to ``extra``, which is host
data copied per context. It is runtime-internal identity, not data: it is not
part of the mapping surface and does not round-trip through ``to_dict``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_KERNEL_FIELDS = ("run_id", "user_id", "disabled_tools", "subagent")


@dataclass
class AgentContext:
    run_id: str
    user_id: str
    disabled_tools: frozenset[str] = field(default_factory=frozenset)
    subagent: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in _KERNEL_FIELDS:
            return getattr(self, key)
        return self.extra[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in _KERNEL_FIELDS or key in self.extra

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict (kernel fields + ``extra``).

        Hosts whose tool handlers still read a raw ``dict`` context (e.g.
        ``ctx["student_id"]``) use this to rebuild that dict from the kernel's
        typed context. ``student_id`` is mirrored as an alias of ``user_id`` so
        legacy dict-style handlers keep working.
        """
        out: dict[str, Any] = {
            "run_id": self.run_id, "user_id": self.user_id,
            "student_id": self.user_id,
            "disabled_tools": self.disabled_tools, "subagent": self.subagent,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentContext:
        """Build a typed context from a host dict.

        ``user_id`` is read from ``user_id`` (falling back to ``student_id``);
        every other non-kernel key is stashed in ``extra`` so nothing is lost.
        """
        user_id = d.get("user_id") or d.get("student_id") or ""
        skip = {"run_id", "user_id", "student_id", "disabled_tools", "subagent"}
        extra = {k: v for k, v in d.items() if k not in skip}
        return cls(
            run_id=d.get("run_id") or "",
            user_id=user_id,
            disabled_tools=frozenset(d.get("disabled_tools") or ()),
            subagent=d.get("subagent"),
            extra=extra,
        )
