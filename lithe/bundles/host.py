"""Host-adapter bundle: the per-host orchestration every agent app needs.

Round 1 extracted the storage-free *engine* (``AgentRuntime`` etc.); this bundle
extracts the next tier — the generic *host plumbing* that each project otherwise
hand-writes: assembling the message list from history, bridging a store into a
persistence :class:`~lithe.events.EventSink`, emitting the host-level
``run_start`` / ``done`` envelope, error funneling, and undo wiring.

A host constructs an :class:`AgentHost` with its registry + LLM config + store +
(own) system-prompt builder, then iterates :meth:`AgentHost.run`. Hosts still on
a dict-based tool system wrap it once with :class:`DictToolAdapter`.

Optional bundle — the core engine never imports this.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
from typing import Any
from collections.abc import AsyncIterator, Callable, Collection

from lithe import (
    DEFAULT_CONTEXT_BUDGET,
    DEFAULT_REPEAT_CALL_LIMIT,
    Action,
    AgentContext,
    AgentRuntime,
    EventType,
    LLMConfig,
    RunStats,
    ToolCategory,
    ToolRegistry,
    ToolSpec,
    UndoEngine,
    UndoReport,
)
from lithe.bundles.store.protocol import StoredMessage

log = logging.getLogger("lithe.host")

PromptBuilder = Callable[[AgentContext, str, "dict | None"], str]
CtxToDict = Callable[[AgentContext], dict]


async def _fanout(sinks: list, ctx: AgentContext, event: dict) -> None:
    """Best-effort delivery of a host-level envelope event to extra sinks."""
    for sink in sinks:
        try:
            await sink.on_event(ctx, event)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "host sink %s on_event failed (ignored): %s", type(sink).__name__, exc
            )


def _history_tool_calls(raw: Any) -> list[dict] | None:
    """Assistant ``tool_calls`` from a history row: an already-parsed list of
    call dicts (kept verbatim), a JSON string of one (legacy stores), or junk
    (``None`` — nothing usable to reconcile)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, list):
        return None
    calls = [c for c in raw if isinstance(c, dict)]
    return calls or None


def _has_text(content) -> bool:
    """Whether a message counts as non-empty: text is stripped, multimodal
    (list) content blocks count when any block is present — they are
    forwarded verbatim, never crashed on with ``.strip()``."""
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)


def assemble_messages(
    system_prompt: str | None, history: list[dict] | None, task: str
) -> list[dict]:
    """Build the initial OpenAI-format message list: system + reconciled history + task.

    History is reconciled exactly as a chat API needs it: assistant turns keep
    their ``tool_calls`` (empty turns with no text/calls are dropped), tool
    turns keep their ``tool_call_id``, and the new ``task`` is appended as the
    final user turn. Reconciliation is two-sided, like
    :func:`lithe.memory.replay_messages`: an assistant ``tool_call`` with
    no matching tool result (a run cancelled between the model's call and its
    execution) is dropped, and so is a tool result whose call is gone —
    either orphan makes the chat API reject the whole request, so a cancelled
    turn must never poison the next one.
    """
    hist = history or []
    calls_at: dict[int, list[dict]] = {}
    called: set[str] = set()
    for i, h in enumerate(hist):
        if h.get("role") != "assistant":
            continue
        tcs = _history_tool_calls(h.get("tool_calls"))
        if tcs:
            calls_at[i] = tcs
            called.update(c["id"] for c in tcs if isinstance(c.get("id"), str))
    answered = {
        h.get("tool_call_id")
        for h in hist
        if h.get("role") == "tool"
        and isinstance(h.get("tool_call_id"), str)
        and h.get("tool_call_id") in called
    }
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for i, h in enumerate(hist):
        role = h.get("role")
        if role == "assistant":
            kept = [
                c
                for c in calls_at.get(i, ())
                if isinstance(c.get("id"), str) and c["id"] in answered
            ]
            if not kept and not _has_text(h.get("content")):
                continue
            m = {"role": "assistant", "content": h.get("content") or ""}
            if kept:
                m["tool_calls"] = kept
            messages.append(m)
        elif role == "tool":
            tid = h.get("tool_call_id")
            if isinstance(tid, str) and tid in answered:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": h.get("content") or "",
                    }
                )
        elif role == "user" and _has_text(h.get("content")):
            messages.append({"role": "user", "content": h["content"]})
    messages.append({"role": "user", "content": task})
    return messages


class StoreSink:
    """An :class:`~lithe.events.EventSink` that persists each turn to a store.

    Display events are ignored (the caller already forwards them); only
    ``on_record`` message records are written, so the kernel never touches a
    database while the full conversation is still persisted.

    Mutation capture: ``on_event`` also maps the undo-bearing UI events tools
    emit inside ``ToolResult.ui`` into stored action rows via
    ``store.log_action`` — ``file_change`` (from ``write_file``/``edit_file``/
    ``apply_patch``, incl. the ``file_delete`` kind) and ``todo_change`` →
    ``todo_replace``. This is the missing middle of the undo chain: tools emit
    the events, :func:`undo_run` reads ``store.list_actions`` — this sink is
    what connects them, so persisted undo works with zero host code. Rows
    carry the emitting context's ``subagent`` tag, and large old/new values
    spill to blobs per the store's own policy. Construct with
    ``capture_actions=False`` when a host records actions itself (avoids
    double rows); unknown event types are ignored, never an error.
    """

    def __init__(self, store, *, capture_actions: bool = True):
        self.store = store
        self.capture_actions = capture_actions

    async def on_event(self, ctx: AgentContext, event: dict) -> None:
        if not self.capture_actions:
            return
        etype = event.get("type")
        if etype == "file_change":
            action = event.get("action")
            if action not in ("write", "edit", "delete"):
                return  # malformed/foreign emitter: skip, never a bad row
            kind, target = f"file_{action}", event.get("path", "")
        elif etype == "todo_change":
            kind, target = "todo_replace", "todos"
        else:
            return
        self.store.log_action(
            ctx.run_id,
            ctx.user_id,
            kind,
            target,
            event.get("old"),
            event.get("new"),
            subagent=ctx.subagent,
        )

    async def on_record(self, ctx: AgentContext, record: dict) -> None:
        role = record.get("role")
        if role == "assistant":
            self.store.add_message(
                StoredMessage(
                    role="assistant",
                    content=record.get("content"),
                    run_id=ctx.run_id,
                    user_id=ctx.user_id,
                    tool_calls=record.get("tool_calls"),
                    subagent=ctx.subagent,
                )
            )
        elif role == "user":
            # Steering injections (runtime._loop_step drains the inbox at
            # step boundaries): persist them like every other turn, so a
            # resumed conversation shows why the model's course changed —
            # the initial task row is written by the host directly and
            # never passes through here, so there is no double record.
            self.store.add_message(
                StoredMessage(
                    role="user",
                    content=record.get("content"),
                    run_id=ctx.run_id,
                    user_id=ctx.user_id,
                    subagent=ctx.subagent,
                )
            )
        elif role == "tool":
            self.store.add_message(
                StoredMessage(
                    role="tool",
                    content=record.get("content"),
                    run_id=ctx.run_id,
                    user_id=ctx.user_id,
                    tool_name=record.get("tool_name"),
                    tool_call_id=record.get("tool_call_id"),
                    meta=record.get("meta"),
                    subagent=ctx.subagent,
                )
            )


class AgentHost:
    """Owns one agent run's orchestration: message assembly, run envelope, store.

    Construct once per process (registry + LLM config + store + prompt builder);
    call :meth:`run` per task. ``run`` is an async generator of display events
    (``run_start`` → step/assistant/tool_call/tool_result/error → ``done``) and
    also persists every turn via an attached :class:`StoreSink` — messages,
    plus (since the action-capture round) the undo-bearing ``file_change`` /
    ``todo_change`` tool events as stored action rows, so :func:`undo_run`
    works against the store with zero host-side plumbing. A consumer
    that stops iterating mid-stream (client disconnect / task cancellation)
    still gets the run closed in the store with ``status="abandoned"``.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm_config: LLMConfig,
        store,
        *,
        build_system_prompt: PromptBuilder | None = None,
        max_steps: int = 35,
        context_budget: int | None = DEFAULT_CONTEXT_BUDGET,
        strict_records: bool = False,
        sinks: list | None = None,
        max_cost: float | None = None,
        max_total_tokens: int | None = None,
        repeat_call_limit: int | None = DEFAULT_REPEAT_CALL_LIMIT,
        http_client=None,
        capture_actions: bool = True,
    ):
        self.registry = registry
        self.llm_config = llm_config
        self.store = store
        self.build_system_prompt = build_system_prompt
        self.max_steps = max_steps
        self.context_budget = context_budget
        self.strict_records = strict_records
        # Run budgets + stuck-model guard, forwarded to every AgentRuntime
        # this host builds (see AgentRuntime for semantics).
        self.max_cost = max_cost
        self.max_total_tokens = max_total_tokens
        self.repeat_call_limit = repeat_call_limit
        # Extra EventSinks attached to every run (metrics, action capture,
        # frontend fan-out); the store's own StoreSink is always attached.
        self.sinks = list(sinks or [])
        # Shared, host-owned HTTP client forwarded to every runtime (see
        # AgentRuntime.http_client): connection pooling / limits / proxy
        # config for long-lived service hosts; None = per-run client.
        self.http_client = http_client
        # StoreSink's automatic file_change/todo_change → action capture.
        # Hosts that persist actions themselves (their own sink with custom
        # kinds) pass False to avoid double rows.
        self.capture_actions = capture_actions

    def _close_run(self, run_id: str, kstats: RunStats, stats: dict | None) -> None:
        """Persist the run's final state (store + host ``stats`` dict). Used on
        both the normal path and the abandoned path, where no ``done`` event
        can be yielded to a dead consumer anymore — synchronous only."""
        self.store.finish_run(
            run_id,
            kstats.status,
            kstats.last_step,
            kstats.total_cost,
            kstats.final_text or None,
            prompt_tokens=kstats.prompt_tokens or None,
            completion_tokens=kstats.completion_tokens or None,
            cached_tokens=kstats.cached_tokens or None,
            total_tokens=kstats.total_tokens or None,
        )
        if stats is not None:
            stats.update(
                {
                    "final_text": kstats.final_text,
                    "status": kstats.status,
                    "last_step": kstats.last_step,
                    "total_cost": kstats.total_cost,
                    "total_tokens": kstats.total_tokens,
                    "prompt_tokens": kstats.prompt_tokens,
                    "completion_tokens": kstats.completion_tokens,
                    "context_tokens": kstats.context_tokens,
                    "context_window": kstats.context_window,
                    "context_percent": kstats.context_percent,
                    "duration_s": kstats.duration_s,
                }
            )

    async def run(
        self,
        ctx: AgentContext,
        task: str,
        *,
        mode: str = "autonomous",
        history: list[dict] | None = None,
        anchor: dict | None = None,
        memory_hint: str | None = None,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        stats: dict | None = None,
        create_run: bool = True,
        stop=None,
        inbox=None,
    ) -> AsyncIterator[dict]:
        run_id, user_id = ctx.run_id, ctx.user_id
        # Stash the run's mode for derived contexts (subagent delegation
        # reads it to cap each worker's toolset to this mode's admitted
        # categories — a restricted mode must not gain powers by
        # delegating). Always set, like the stop handle below, so a reused
        # context never inherits a previous run's mode.
        ctx.shared["_host_mode"] = mode
        if create_run:
            self.store.create_run(
                run_id,
                user_id,
                task,
                conversation_id=ctx.extra.get("conversation_id"),
                model=self.llm_config.model,
            )
        kstats = RunStats()
        closed = False
        try:
            # Prompt assembly + the user turn live inside the guarded zone:
            # a crashing host prompt builder, multimodal history content, or a
            # full disk must funnel into the error envelope (error event +
            # run closed ``failed``), not strand the run at ``running`` with
            # a raw exception out of the generator. The user turn is recorded
            # before run_start is yielded so even an instant disconnect
            # persists it.
            sp = system_prompt
            if sp is None and self.build_system_prompt is not None:
                sp = self.build_system_prompt(ctx, mode, anchor)
            messages = assemble_messages(sp, history, task)
            self.store.add_message(
                StoredMessage(role="user", content=task,
                              run_id=run_id, user_id=user_id)
            )

            envelope = {
                "type": EventType.RUN_START,
                "run_id": run_id,
                "task": task,
                "model": self.llm_config.model,
                "max_steps": self.max_steps,
                "memory": memory_hint,
            }
            await _fanout(self.sinks, ctx, envelope)
            yield envelope

            # A host may hand its own tool spec list (e.g. one that conditionally
            # includes meta tools); otherwise derive it from the registry + mode.
            tool_specs = (
                tools
                if tools is not None
                else self.registry.specs_for_mode(mode, ctx.disabled_tools)
            )
            runtime = AgentRuntime(
                self.registry,
                self.llm_config,
                sinks=[StoreSink(self.store,
                                 capture_actions=self.capture_actions),
                       *self.sinks],
                max_steps=self.max_steps,
                context_budget=self.context_budget,
                strict_records=self.strict_records,
                max_cost=self.max_cost,
                max_total_tokens=self.max_total_tokens,
                repeat_call_limit=self.repeat_call_limit,
                http_client=self.http_client,
            )
            try:
                # aclosing: a disconnecting consumer must deterministically
                # close the runtime generator (and its HTTP client) instead of
                # waiting for the GC finalizer. `inbox` is the steering
                # channel: a queue of user texts drained at each step boundary.
                async with contextlib.aclosing(runtime.run(
                        ctx, messages, tool_specs, stats=kstats, stop=stop,
                        inbox=inbox)) as stream:
                    async for ev in stream:
                        yield ev
            except Exception as exc:  # noqa: BLE001
                kstats.status = "failed"
                yield {"type": EventType.ERROR, "message": f"运行出错：{exc}"}

            # Subagent spend folds into the run totals before closing: the
            # store row, host stats dict and done event then report what the
            # run actually cost including every delegation (the engine
            # accumulated it in the run's shared state).
            sub = ctx.shared.get("_subagent_usage") or {}
        except Exception as exc:  # noqa: BLE001
            # Pre-runtime failures (crashing host prompt builder, store IO,
            # fanout) funnel into the same error envelope as runtime errors —
            # error event, run closed ``failed`` — instead of escaping the
            # generator and stranding the run at ``running``.
            kstats.status = "failed"
            yield {"type": EventType.ERROR, "message": f"运行出错：{exc}"}
            sub = ctx.shared.get("_subagent_usage") or {}
        except BaseException:
            # The consumer went away mid-stream — a disconnecting SSE client
            # or a cancelled task surfaces as CancelledError / GeneratorExit,
            # both BaseException and thus invisible to ``except Exception``.
            # No event can be yielded to a dead consumer, but the run must not
            # stay ``running`` in the store forever: close it out (sync only —
            # awaiting inside cancellation handling is not safe) and re-raise
            # so the cancellation still propagates. Once the run was closed
            # normally (``closed``), a disconnect at the final ``done`` yield
            # must not rewrite an already-recorded terminal status.
            if not closed:
                kstats.status = "abandoned"
                self._close_run(run_id, kstats, stats)
            raise
        sub_cost = float(sub.get("cost") or 0.0)
        sub_tokens = int(sub.get("total_tokens") or 0)
        if sub_cost or sub_tokens:
            kstats.total_cost += sub_cost
            kstats.total_tokens += sub_tokens
            kstats.prompt_tokens += int(sub.get("prompt_tokens") or 0)
            kstats.completion_tokens += int(sub.get("completion_tokens") or 0)
        self._close_run(run_id, kstats, stats)
        closed = True
        done = {
            "type": EventType.DONE,
            "run_id": run_id,
            "steps": kstats.last_step,
            "cost": round(kstats.total_cost, 6),
            "tokens": kstats.total_tokens,
            "prompt_tokens": kstats.prompt_tokens,
            "completion_tokens": kstats.completion_tokens,
            "context_tokens": kstats.context_tokens,
            "context_window": kstats.context_window,
            "context_percent": kstats.context_percent,
            "duration_s": kstats.duration_s,
            "status": kstats.status,
        }
        if sub.get("delegations"):
            done["subagent_delegations"] = int(sub["delegations"])
            done["subagent_cost"] = round(sub_cost, 6)
            done["subagent_tokens"] = sub_tokens
        await _fanout(self.sinks, ctx, done)
        yield done


def _default_ctx_to_dict(ctx: AgentContext) -> dict:
    return ctx.to_dict()


class DictToolAdapter(ToolRegistry):
    """Adapt a host's dict-based tool system into the kernel's ToolRegistry.

    Many hosts carry tools as an OpenAI spec list + ``{name: async handler}``
    where handlers take a *dict* context and return a *dict* result. Instead of
    rewriting those tools, wrap them once: each spec becomes a registered
    :class:`~lithe.tools.ToolSpec`, and dispatch bridges the typed
    :class:`AgentContext` → dict context and normalizes the dict result into a
    :class:`ToolResult` (the base registry already does the latter).

    ``write_names`` classifies tools for mode filtering (default: all read-only).
    ``ctx_to_dict`` customizes the context bridge (default: :meth:`AgentContext.to_dict`,
    which mirrors ``user_id`` as ``student_id`` and spreads ``extra``).
    """

    def __init__(
        self,
        specs: list[dict],
        handlers: dict,
        *,
        write_names: Collection[str] = (),
        ctx_to_dict: CtxToDict | None = None,
        reverters: dict[str, Callable] | None = None,
    ):
        super().__init__()
        self._dhandlers = dict(handlers)
        self._ctx_to_dict = ctx_to_dict or _default_ctx_to_dict
        # Extra kind → reverter map surfaced through reverters(): dict-tool
        # hosts log their own action kinds, this wires them into undo_run's
        # zero-config default.
        self._extra_reverters = dict(reverters or {})
        wnames = set(write_names)
        for spec in specs:
            fn = spec.get("function", spec)
            name = fn.get("name")
            if name not in self._dhandlers:
                # A spec without a handler is almost always a typo in the
                # handler map — the tool would silently vanish from the
                # registry. Say so loudly instead.
                log.warning("DictToolAdapter: spec %r has no handler; "
                            "tool not registered", name)
                continue
            cat = ToolCategory.WRITE if name in wnames else ToolCategory.READ
            ts = ToolSpec(
                name,
                fn.get("description", ""),
                fn.get("parameters", {"type": "object", "properties": {}}),
                cat,
            )
            self.register(ts, self._bridge(name))

    def reverters(self) -> dict:
        """Base registry map (registered tools) plus the constructor's
        ``reverters`` — dict-tool hosts have no ``revert_kind`` hook, so the
        constructor map is their channel."""
        base = super().reverters()
        base.update(self._extra_reverters)
        return base

    def _bridge(self, name: str):
        async def handler(agent_ctx: AgentContext, args: dict) -> Any:
            dctx = self._ctx_to_dict(agent_ctx)
            return await self._dhandlers[name](dctx, args)

        return handler


async def undo_run(
    host: AgentHost,
    run_id: str,
    user_id: str,
    *,
    reverters: dict | None = None,
    mark_reverted: bool = True,
    extra: dict | None = None,
) -> UndoReport:
    """Revert all in-effect mutations of a run, newest-first.

    Generic mirror of a host's ``undo_run``: loads the run's revertible actions
    from the store, drives the kernel :class:`UndoEngine` with the host's
    kind-keyed reverters (sync **or** async — a reverter calling a remote API
    is awaited), and (by default) marks each reverted action row so a later
    undo is a no-op. The default reverter map is ``registry.reverters()``,
    keyed by each tool's registered ``revert_kind`` — the bundled tools
    (workspace / patch / todos) register the exact kinds their actions use
    (``file_write`` / ``file_edit`` / ``file_delete`` / ``todo_replace``), so
    the zero-configuration default reverts them. Pass ``reverters={kind: fn}``
    to override (e.g. custom kinds a DictToolAdapter host logs).

    ``extra`` populates the reverters' context ``extra`` mapping (e.g.
    ``{"thesis_id": ...}`` when ``workspace_for(ctx)`` needs it) — without it a
    reverter reading a host field off the context would KeyError.
    """
    rows = host.store.list_actions(run_id, user_id, status_in=("applied", "approved"))
    actions = [
        Action(
            kind=a.kind,
            target=a.target,
            old_value=a.old_value,
            new_value=a.new_value,
            status=a.status,
            id=a.id,
            subagent=a.subagent,
        )
        for a in rows
    ]
    base = reverters if reverters is not None else host.registry.reverters()
    if actions and not any(a.kind in base for a in actions):
        # Nothing loaded is revertible with this map: the default comes from
        # the registry (``revert_kind``-keyed), so this means custom action
        # kinds with no matching reverter — undo would be a silent no-op.
        log.warning("undo_run(%s): action kinds %s match none of the reverter "
                    "keys %s; nothing will be reverted",
                    run_id, sorted({a.kind for a in actions}), sorted(base))

    # Rows whose reversion SUCCEEDED but whose "reverted" status mark FAILED
    # (store IO error): must surface in the report — a silently unmarked row
    # re-runs its reverter on the next undo attempt, and a non-idempotent
    # reverter would then do its damage a second time.
    mark_failures: list[str] = []

    def _wrap(fn):
        async def wrapped(action: Action, ctx: AgentContext) -> None:
            result = fn(action, ctx)
            if inspect.isawaitable(result):
                await result
            if mark_reverted and action.id is not None:
                try:
                    host.store.set_action_status(action.id, ctx.user_id,
                                                 "reverted")
                except Exception as exc:  # noqa: BLE001
                    mark_failures.append(
                        f"{action.kind} {action.target}"
                        f"(id={action.id}) 已回滚但标记撤销状态失败：{exc}")

        return wrapped

    engine = UndoEngine({k: _wrap(v) for k, v in base.items()})
    report = await engine.undo(
        actions, AgentContext(run_id=run_id, user_id=user_id, extra=extra or {})
    )
    if mark_failures:
        report.errors.extend(mark_failures)
        report.ok = False
    return report


# re-exported for type hints in host code
__all__ = ["AgentHost", "DictToolAdapter", "StoreSink", "assemble_messages", "undo_run"]
