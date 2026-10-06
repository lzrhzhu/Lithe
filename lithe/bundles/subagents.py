"""Subagent bundle: focused worker agents an orchestrator delegates subtasks to.

Each subagent runs its own ReAct loop in an isolated context — a specialized
system prompt, a trimmed toolset, and any skill bodies it needs *injected* into
that prompt. Because the skills live inside the subagent's one-shot prompt (not
``load_skill``-ed into the shared conversation), they never reach the
orchestrator's replayed history: that is the whole point (the orchestrator's
context stays lean across turns).

A subagent's internal messages/actions are still recorded under the *parent*
``run_id`` but tagged with a per-delegation *instance* id
(``subagent = "<agent>:<hex8>"``), so a later undo reverts every mutation
while the orchestrator's model-context replay drops them. Instance tagging
is what makes same-agent parallel delegation safe: attribution, messages
and live-usage accounting key on the delegation instance, never on the
roster id, so two concurrent delegations of one subagent cannot
cross-claim each other's work. Delegation is exactly one level deep: no
subagent gets the ``delegate`` tool.

The *mechanism* is generic and lives here; the *roster* (subagent ids, personas,
tool lists, skill stems) is host data passed to :class:`SubagentRoster`.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import Any
from collections.abc import Callable

from lithe import (
    LLMTransport, AgentContext, AgentRuntime, LLMConfig, RunStats,
    ToolCategory, ToolResult, ToolSpec,
)
from lithe.modes import categories_for
from lithe.bundles.host import AgentHost, StoreSink
from lithe.bundles.store.protocol import StoredMessage, action_to_row

log = logging.getLogger("lithe.subagents")

# _thin_event caps: the progress callback rides a host push channel, not the
# recorded stream — a 50KB write_file args block or a fat tool summary must
# not be forwarded verbatim.
_THIN_TEXT_CAP = 300
_THIN_ARGS_CAP = 500

SkillInjector = Callable[[list[str], AgentContext], str]
ActionLabeler = Callable[[dict], "str | None"]
EnabledGate = "bool | Callable[[], bool]"
# Live progress callback: receives (subagent_ctx, {"type": "subagent_progress",
# "agent": id, "event": <thin runtime event>}). Wire it to the host's own
# push channel (websocket / SSE side-band) — tool results only land at the end.
ProgressCallback = Callable[[AgentContext, dict], Any]


@dataclass
class SubagentSpec:
    """One declarative subagent (host data).

    ``tools`` is a subset of host tool names (never includes ``delegate``).
    ``prompt`` is the persona + working rules (domain text). ``auto_skills`` are
    skill stems a host :class:`SkillInjector` inlines into that prompt.
    ``model`` / ``base_url`` / ``api_key`` / ``transport`` optionally route
    this subagent to a different endpoint or wire protocol.
    """
    id: str
    display: str
    description: str
    tools: list[str]
    prompt: str
    auto_skills: list[str] = field(default_factory=list)
    read_only: bool = False
    max_steps: int | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    transport: str | LLMTransport | None = None


class SubagentRoster:
    """Ordered collection of :class:`SubagentSpec`, keyed by id."""

    def __init__(self, specs: list[SubagentSpec]):
        self._specs: dict[str, SubagentSpec] = {}
        for s in specs:
            if s.id in self._specs:
                raise ValueError(f"duplicate subagent id: {s.id}")
            self._specs[s.id] = s

    def get(self, sub_id: str) -> SubagentSpec | None:
        return self._specs.get(sub_id)

    def ids(self) -> list[str]:
        return list(self._specs)

    def __contains__(self, sub_id: str) -> bool:
        return sub_id in self._specs

    def text(self) -> str:
        """Compact ``- id: description`` roster for the delegate description / errors."""
        return "\n".join(f"- {sid}: {s.description}" for sid, s in self._specs.items())


class SubagentEngine:
    """Runs subagents in isolation under the parent run, tagged with their id.

    Built from an :class:`AgentHost` (shares its registry/store/LLM defaults) and
    a :class:`SubagentRoster`. ``skill_injector`` inlines skill bodies into a
    subagent's prompt; ``label_fn`` labels its mutations in the summary; ``enabled``
    gates the delegate tool (bool or zero-arg callable).
    """

    def __init__(self, host: AgentHost, roster: SubagentRoster, *,
                 skill_injector: SkillInjector | None = None,
                 label_fn: ActionLabeler | None = None,
                 prompt_builder: Callable[[SubagentSpec, AgentContext], str] | None = None,
                 enabled: EnabledGate = True,
                 max_steps: int | None = None,
                 default_model: str | None = None,
                 max_parallel: int | None = None,
                 max_batch: int = 8,
                 on_subagent_event: ProgressCallback | None = None,
                 capture_actions: bool = True):
        self.host = host
        self._roster = roster
        self.skill_injector = skill_injector
        self.label_fn = label_fn
        self.prompt_builder = prompt_builder
        self.enabled = enabled
        self.max_steps = max_steps
        self.default_model = default_model
        # delegate_parallel knobs: at most max_batch tasks per call, at most
        # max_parallel subagents running at once (None = unbounded).
        self.max_parallel = max_parallel
        self.max_batch = max_batch
        # Optional live-progress hook (see ProgressCallback): without it the
        # subagent's display events are recorded (tagged) but not surfaced.
        self.on_subagent_event = on_subagent_event
        # Forwarded to the subagent run's StoreSink. Hosts whose tools log
        # their own domain actions (custom undo kinds a generic file_change
        # capture cannot know) set this False — mirroring
        # ``AgentHost(capture_actions=False)`` — so a delegation's mutations
        # are not recorded twice.
        self.capture_actions = capture_actions

    @property
    def roster(self) -> SubagentRoster:
        """The engine's roster. Read-only by design: the ``delegate`` tools'
        schema enum and validation freeze over the roster at registration
        time, so swapping it afterwards would leave "unknown subagent"
        shadows — build a new engine + re-register instead."""
        return self._roster

    def is_enabled(self) -> bool:
        return self.enabled() if callable(self.enabled) else bool(self.enabled)

    # -- prompt + tools -------------------------------------------------------
    def system_prompt(self, spec: SubagentSpec, ctx: AgentContext) -> str:
        # A host may fully own the prompt (e.g. to prepend a dynamic per-student
        # title / append a usage scope the kernel knows nothing about).
        if self.prompt_builder is not None:
            return self.prompt_builder(spec, ctx)
        parts = [spec.prompt]
        if self.skill_injector is not None and spec.auto_skills:
            injected = self.skill_injector(spec.auto_skills, ctx)
            if injected:
                parts.append(injected)
        return "\n\n".join(p for p in parts if p)

    def trimmed_tools(self, spec: SubagentSpec,
                      disabled: frozenset[str] | set[str] | None,
                      allowed_categories=None) -> list[dict]:
        """OpenAI spec list for the subagent's declared tools (minus disabled).

        ``delegate`` is never included, so delegation stays one level deep.
        ``allowed_categories`` (a category set from :func:`lithe.modes.categories_for`)
        additionally drops tools the *orchestrator's mode* does not admit —
        delegation must never widen a restricted mode's powers (an anchored /
        read-only conversation delegating to a write-capable subagent would).
        ``None`` keeps the declared list: no mode context (direct engine.run
        callers, tests).
        """
        dis = set(disabled or ())
        out: list[dict] = []
        for name in spec.tools:
            if name == "delegate" or name in dis:
                continue
            ts = self.host.registry.spec(name)
            if ts is None:
                continue
            if (allowed_categories is not None
                    and ts.category not in allowed_categories):
                continue
            out.append(ts.to_openai())
        return out

    def _llm_for(self, spec: SubagentSpec) -> LLMConfig:
        lc = self.host.llm_config
        # dataclasses.replace keeps every host-level policy field (stream,
        # context_window, retry/backoff, ...) that a manual field-by-field copy
        # has historically dropped whenever the kernel grew a new one.
        overrides: dict[str, Any] = {
            "model": spec.model or self.default_model or lc.model,
            "base_url": spec.base_url or lc.base_url,
            "api_key": spec.api_key or lc.api_key,
        }
        if spec.transport is not None:
            overrides["transport"] = spec.transport
        return replace(lc, **overrides)

    # -- execution ------------------------------------------------------------
    async def run(self, sub_id: str, task: str, parent_ctx: AgentContext, *,
                  grounding: str = "",
                  instance: str | None = None) -> tuple[RunStats, list]:
        """Run one subagent to completion under the parent run, tagged with
        a unique per-delegation instance id.

        The context / store tag is ``"<agent>:<hex8>"`` (or the caller's
        explicit ``instance``), NOT the bare roster id: messages, actions
        and the live-usage slot all key on the delegation instance, so two
        concurrent delegations of the same subagent — which the parallel
        tool now starts freely — cannot cross-claim each other's mutations
        in summaries or undo labels. Passing a previously used
        ``instance`` tag gives resume-like accounting: the high-water
        snapshot below then excludes that instance's earlier actions
        instead of starting empty.

        Returns ``(stats, sub_actions)`` where ``sub_actions`` are *this
        instance's own* mutations (new since its snapshot).
        """
        spec = self.roster.get(sub_id)
        if spec is None:
            raise KeyError(f"unknown subagent: {sub_id}")
        tag = instance or f"{sub_id}:{uuid.uuid4().hex[:8]}"
        sub_ctx = AgentContext(run_id=parent_ctx.run_id, user_id=parent_ctx.user_id,
                               disabled_tools=parent_ctx.disabled_tools,
                               subagent=tag, extra=dict(parent_ctx.extra),
                               # shared BY REFERENCE: the run's cross-context
                               # state (stale-file guard revisions, …) must be
                               # one map for the orchestrator and every
                               # subagent of the run, or two parallel
                               # subagents can clobber the same file unseen.
                               shared=parent_ctx.shared)

        # High-water snapshot of this instance's existing actions, as a SET
        # of ids: ``Action.id`` is opaque/host-assigned (ints, strings, ...),
        # so comparing it numerically — or at all — would assume store
        # internals. Membership in the pre-delegation set is the only
        # contract needed. Fresh instances start empty; a reused instance
        # tag (resume) excludes its earlier work. The instance-filtered
        # query keeps a long run's snapshots O(this delegation) instead of
        # rehydrating every earlier delegation's blobs.
        prev_ids = {a.id for a in self.host.store.list_actions(
            sub_ctx.run_id, sub_ctx.user_id, subagent=tag)
            if a.id is not None}

        user_msg = task + (f"\n\n【编排者补充上下文】\n{grounding}" if grounding else "")
        messages = [{"role": "system", "content": self.system_prompt(spec, sub_ctx)},
                    {"role": "user", "content": user_msg}]
        self.host.store.add_message(StoredMessage(
            role="user", content=user_msg, run_id=sub_ctx.run_id,
            user_id=sub_ctx.user_id, subagent=tag))

        stats = RunStats()
        steps = spec.max_steps or self.max_steps or self.host.max_steps
        # Mode fence: the orchestrator's run mode (stashed in the run's
        # shared state by AgentHost.run) caps this subagent's toolset to
        # that mode's admitted categories — an anchored (read-only)
        # conversation must not gain write powers by delegating to a
        # write-capable subagent. No stashed mode (direct engine.run
        # callers) keeps the declared roster list unchanged.
        mode = parent_ctx.shared.get("_host_mode")
        allowed = None
        if mode is not None:
            try:
                allowed = categories_for(mode)
            except ValueError:
                allowed = None  # unknown mode name: never widen, never crash
        # Budgets apply per subagent run: each delegation gets its own cap,
        # so one runaway worker cannot burn the whole ceiling unnoticed
        # (the orchestrator's own budget is separate).
        runtime = AgentRuntime(self.host.registry, self._llm_for(spec),
                               sinks=[StoreSink(self.host.store,
                                                capture_actions=self.capture_actions)],
                               max_steps=steps,
                               context_budget=self.host.context_budget,
                               strict_records=self.host.strict_records,
                               max_cost=self.host.max_cost,
                               max_total_tokens=self.host.max_total_tokens,
                               repeat_call_limit=self.host.repeat_call_limit,
                               http_client=self.host.http_client)
        # Cancellation propagates: the orchestrator's stop handle was stashed
        # in the run's shared state by AgentRuntime.run, so a user cancelling
        # the parent run also ends this subagent's loop (between its model
        # calls / tool dispatches) instead of letting it run to completion.
        parent_stop = parent_ctx.shared.get("_runtime_stop")
        # Real-time parallel-budget visibility: this delegation registers a
        # live-usage slot in the shared map; every usage event folds into it
        # as the child spends. Each concurrently running sibling's runtime
        # counts the OTHER slots against its own cap (see
        # AgentRuntime._over_budget), so N parallel workers share one
        # ceiling instead of each burning the full max_cost. The slot is
        # the delegation instance (unique per run, reused for nothing else)
        # and popped at the end.
        slot = tag
        sub_ctx.extra["_delegation_slot"] = slot
        inflight = parent_ctx.shared.setdefault("_inflight_sub_usage", {})
        inflight[slot] = {"cost": 0.0, "tokens": 0}
        # aclosing: a cancelled delegation must deterministically close the
        # inner runtime generator (its httpx client) instead of waiting for
        # the GC finalizer to notice.
        try:
            async with contextlib.aclosing(runtime.run(
                    sub_ctx, messages,
                    self.trimmed_tools(spec, sub_ctx.disabled_tools,
                                       allowed_categories=allowed),
                    stats=stats, stop=parent_stop)) as stream:
                async for ev in stream:
                    # internal display events are recorded (tagged), not
                    # forwarded to the orchestrator's stream — except through
                    # the optional live progress hook, which a host wires to
                    # its own push channel.
                    if ev.get("type") == "usage":
                        live = inflight.get(slot)
                        if live is not None:
                            live["cost"] += float(ev.get("cost") or 0.0)
                            live["tokens"] += int(ev.get("total_tokens") or 0)
                    if self.on_subagent_event is not None:
                        thin = _thin_event(ev)
                        if thin is not None:
                            try:
                                await self.on_subagent_event(
                                    sub_ctx, {"type": "subagent_progress",
                                              "agent": sub_id,
                                              "instance": tag, "event": thin})
                            except Exception as exc:  # noqa: BLE001
                                log.warning("subagent progress callback failed: %s",
                                            exc)
        finally:
            inflight.pop(slot, None)
            sub_ctx.extra.pop("_delegation_slot", None)

        # Fold this delegation's spend into the run-wide accumulator (shared
        # by reference with the orchestrator's context): without it, the
        # parent run's ``done`` event and store row report only the
        # orchestrator's own calls, under-counting a multi-delegation run's
        # true cost by everything the workers burned. Per-delegation budget
        # caps are unchanged — this is accounting, not enforcement.
        acc = parent_ctx.shared.setdefault(
            "_subagent_usage",
            {"delegations": 0, "cost": 0.0, "prompt_tokens": 0,
             "completion_tokens": 0, "total_tokens": 0})
        acc["delegations"] += 1
        acc["cost"] += stats.total_cost
        acc["prompt_tokens"] += stats.prompt_tokens
        acc["completion_tokens"] += stats.completion_tokens
        acc["total_tokens"] += stats.total_tokens

        sub_actions = [a for a in self.host.store.list_actions(
            sub_ctx.run_id, sub_ctx.user_id, subagent=tag)
            if a.id is not None and a.id not in prev_ids]
        return stats, sub_actions

    def summarize(self, spec: SubagentSpec, stats: RunStats, sub_actions: list) -> str:
        if stats.status == "failed":
            head = f"【子代理 {spec.display} 运行失败】"
        elif stats.status == "cancelled":
            head = f"【子代理 {spec.display} 已随本次运行取消】"
        elif stats.status == "budget_exceeded":
            head = f"【子代理 {spec.display} 因超出预算中止】"
        elif stats.status == "empty_response":
            head = f"【子代理 {spec.display} 未返回任何结果】"
        else:
            head = f"【子代理 {spec.display} 已完成】"
        parts = [head]
        if stats.error:
            # The diagnostic (exception class + message, set by the runtime
            # on every failed / empty model call) belongs in the summary the
            # orchestrator's model reads: without it a failed delegation
            # reads as an unexplained "运行失败", and the model cannot tell
            # a transient gateway error (worth one retry) from a task that
            # genuinely cannot run.
            parts.append(f"失败原因：{stats.error}")
        final = (stats.final_text or "").strip()
        if final:
            parts.append(final)
        if self.label_fn is not None:
            labels = [lbl for a in sub_actions if a.status != "skipped"
                      for lbl in (self.label_fn(action_to_row(a)),) if lbl]
            if labels:
                parts.append("所做改动：" + "；".join(labels))
        return "\n\n".join(parts)

    # -- delegate tool --------------------------------------------------------
    async def delegate(self, args: dict, parent_ctx: AgentContext) -> ToolResult:
        """``delegate`` tool body: validate, run a subagent, return its summary."""
        if not self.is_enabled():
            return ToolResult(False, "委派失败", "子代理委派未启用")
        sub_id = (args.get("agent") or "").strip()
        task = (args.get("task") or "").strip()
        grounding = (args.get("context") or "").strip()
        spec = self.roster.get(sub_id)
        if spec is None:
            return ToolResult(False, "委派失败",
                              "未知子代理，可选：\n" + self.roster.text())
        if not task:
            return ToolResult(False, "委派失败", "请给出子代理要执行的 task。")

        instance = f"{sub_id}:{uuid.uuid4().hex[:8]}"
        stats, sub_actions = await self.run(sub_id, task, parent_ctx,
                                            grounding=grounding,
                                            instance=instance)
        summary = self.summarize(spec, stats, sub_actions)
        # ok only when the subagent actually ran to completion (naturally or
        # via its step-cap summary); cancelled / budget-cut delegations did
        # not finish their task and must not read as success to the model.
        ok = stats.status in ("done", "max_steps")
        changes = sum(1 for a in sub_actions if a.status != "skipped")
        ui = [
            {"type": "subagent_start", "agent": sub_id, "display": spec.display,
             "task": task, "instance": instance},
            {"type": "subagent_end", "agent": sub_id, "display": spec.display,
             "status": stats.status, "steps": stats.last_step,
             "changes": changes, "ok": ok, "instance": instance},
        ]
        return ToolResult(ok, summary[:200], summary, ui=ui)


def make_delegate_tool(engine: SubagentEngine, *, name: str = "delegate",
                       timeout: float | None = None) -> tuple[ToolSpec, Any]:
    """Build the ``delegate`` ToolSpec + handler for an orchestrator registry.

    The host registers the returned pair on its orchestrator registry (the
    engine closes over it). Category is :attr:`ToolCategory.META` so it is
    available in autonomous mode but never to subagents (they get a trimmed
    toolset without ``delegate``). ``timeout`` caps one delegation's wall
    time — without it a hung subagent tool (or a wedged endpoint with no
    budget cap) holds the orchestrator's whole step forever; on timeout the
    delegation task is cancelled and the model sees an error result.
    """
    roster = engine.roster

    async def handler(ctx: AgentContext, args: dict) -> ToolResult:
        return await engine.delegate(args, ctx)

    spec = ToolSpec(
        name,
        "把一项需要专门工具的工作委派给专用子代理，在隔离上下文里完成；其加载的技能与"
        "中间结果不会留在本会话上下文里。返回子代理的完成摘要（含所做改动）。",
        {"type": "object",
         "properties": {
             "agent": {"type": "string", "enum": roster.ids(),
                       "description": "要委派的子代理"},
             "task": {"type": "string",
                      "description": "交给子代理的任务（它看不到本会话历史，要说清目标与约束）"},
             "context": {"type": "string",
                         "description": "可选：给子代理的补充上下文（slug / id / 参数等）"}},
         "required": ["agent", "task"]},
        ToolCategory.META, timeout=timeout)
    return spec, handler


def _thin_event(ev: dict) -> dict | None:
    """Slim a runtime display event for the progress callback: keep step /
    assistant (text capped) / tool_call (args capped) / tool_result (summary
    and error capped) / error / cancelled, drop the rest."""
    etype = ev.get("type")
    if etype not in ("step", "assistant", "tool_call", "tool_result",
                     "error", "cancelled"):
        return None
    thin = dict(ev)
    if etype == "assistant" and isinstance(thin.get("text"), str):
        text = thin["text"]
        thin["text"] = text[:_THIN_TEXT_CAP] + ("…" if len(text) > _THIN_TEXT_CAP
                                                else "")
    elif etype == "tool_call":
        args = thin.get("args")
        if args is not None:
            try:
                dumped = json.dumps(args, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                dumped = repr(args)
            thin["args"] = (dumped[:_THIN_ARGS_CAP]
                            + ("…" if len(dumped) > _THIN_ARGS_CAP else ""))
    elif etype == "tool_result":
        for key in ("summary", "error"):
            val = thin.get(key)
            if isinstance(val, str) and len(val) > _THIN_TEXT_CAP:
                thin[key] = val[:_THIN_TEXT_CAP] + "…"
    elif etype == "error" and isinstance(thin.get("message"), str):
        msg = thin["message"]
        thin["message"] = msg[:_THIN_TEXT_CAP] + ("…" if len(msg) > _THIN_TEXT_CAP
                                                  else "")
    return thin


def register_delegate_tool(registry, engine: SubagentEngine, *, name: str = "delegate",
                           timeout: float | None = None):
    """Register the ``delegate`` tool on *registry* (built from *engine*)."""
    spec, handler = make_delegate_tool(engine, name=name, timeout=timeout)
    registry.register(spec, handler)
    return spec


def make_parallel_delegate_tool(engine: SubagentEngine,
                                *, name: str = "delegate_parallel",
                                timeout: float | None = None
                                ) -> tuple[ToolSpec, Any]:
    """Build the ``delegate_parallel`` tool: fan several subagents out at once.

    Every task runs ``engine.delegate`` concurrently, bounded by the
    engine's ``max_parallel`` (None = unbounded) — including several tasks
    on the SAME agent. Same-agent parallelism is safe because each
    delegation is its own instance: messages, action attribution and the
    live-usage slot key on the unique instance tag
    (``SubagentEngine.run``), never on the roster id, so two concurrent
    delegations of one subagent cannot cross-claim each other's work any
    more than two different agents can. One failing subagent does not
    abort the others — the result reports per-agent blocks and ``ok`` is
    True only when all succeeded. Categories/META, like ``delegate``.

    ``timeout`` caps EACH task's wall-clock time individually (per task, not
    per batch, and counted only while running — a task queued behind the
    ``max_parallel`` semaphore does not burn it): a hung worker is cancelled
    and reported as that agent's failure block while its siblings keep
    running, mirroring ``make_delegate_tool``'s timeout semantics.
    """
    roster = engine.roster

    async def handler(ctx: AgentContext, args: dict) -> ToolResult:
        if not engine.is_enabled():
            return ToolResult(False, "委派失败", "子代理委派未启用")
        tasks = args.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            return ToolResult(False, "参数错误",
                              "需要 tasks 数组（每项含 agent 与 task）。")
        if len(tasks) > engine.max_batch:
            return ToolResult(False, "任务过多",
                              f"一次最多并行委派 {engine.max_batch} 项，"
                              f"收到 {len(tasks)} 项。")
        items: list[dict] = []
        for t in tasks:
            if not isinstance(t, dict):
                return ToolResult(False, "参数错误", "tasks 每项必须是对象。")
            agent = (t.get("agent") or "").strip()
            if agent not in roster:
                return ToolResult(False, "委派失败",
                                  f"未知子代理 {agent!r}，可选：\n" + roster.text())
            task = (t.get("task") or "").strip()
            if not task:
                return ToolResult(False, "委派失败",
                                  f"子代理 {agent} 缺少 task。")
            items.append({"agent": agent, "task": task,
                          "context": (t.get("context") or "").strip()})
        sem = asyncio.Semaphore(engine.max_parallel) \
            if engine.max_parallel else None

        async def run_one(item: dict) -> ToolResult:
            async def _run() -> ToolResult:
                if sem is not None:
                    async with sem:
                        return await engine.delegate(item, ctx)
                return await engine.delegate(item, ctx)

            if timeout is None:
                return await _run()
            try:
                return await asyncio.wait_for(_run(), timeout)
            except asyncio.TimeoutError:
                # The cancellation deterministically closes the delegation's
                # inner runtime generator (aclosing in SubagentEngine.run);
                # siblings are separate gather children and keep running.
                return ToolResult(
                    False, "执行超时",
                    f"子代理 {item['agent']} 执行超时（{timeout:g}s 内未完成），"
                    f"已被中止；其余子代理不受影响。")

        # Same-agent tasks run alongside distinct-agent tasks: delegation is
        # instance-scoped (unique tag per run), so attribution, summaries
        # and undo labels cannot cross-claim between concurrent siblings.
        # return_exceptions=True: a *crashed* delegate (store failure, bug —
        # anything escaping as an exception rather than a failed ToolResult)
        # must not abort its siblings mid-flight, which would also leave them
        # running as unawaited orphans. Crashes are reported per agent below.
        outcomes = await asyncio.gather(
            *[asyncio.ensure_future(run_one(it)) for it in items],
            return_exceptions=True)
        blocks = []
        succeeded = 0
        ui: list[dict] = []
        for item, res in zip(items, outcomes, strict=True):
            if isinstance(res, BaseException):
                log.warning("delegate_parallel: agent %s crashed: %s",
                            item["agent"], res)
                blocks.append(f"### {item['agent']}（失败）\n"
                              f"子代理运行异常：{res}")
                continue
            if res.ok:
                succeeded += 1
            blocks.append(f"### {item['agent']}（{'成功' if res.ok else '失败'}）\n"
                          f"{res.content}")
            ui.extend(res.ui)
        return ToolResult(
            succeeded == len(outcomes),
            f"并行委派 {len(outcomes)} 项，{succeeded} 成功",
            "\n\n".join(blocks), ui=ui)

    spec = ToolSpec(
        name,
        "把多项独立工作并行委派给子代理（各在隔离上下文里完成）。所有任务"
        "同时执行——包括给同一个子代理的多项任务（每次委派都是独立实例，"
        "互不共享上下文）。任务之间必须互不依赖；返回每个子代理的完成"
        "摘要。适合扇出检索/多方案生成。",
        {"type": "object",
         "properties": {
             "tasks": {
                 "type": "array",
                 "description": "要并行执行的任务列表",
                 "items": {
                     "type": "object",
                     "properties": {
                         "agent": {"type": "string",
                                   "description": "子代理 id"},
                         "task": {"type": "string",
                                  "description": "交给它的任务（它看不到本会话历史）"},
                         "context": {"type": "string",
                                     "description": "可选补充上下文"},
                     },
                     "required": ["agent", "task"],
                 },
             }},
         "required": ["tasks"]},
        ToolCategory.META)
    return spec, handler


def register_delegate_tools(registry, engine: SubagentEngine, *,
                            parallel: bool = True,
                            timeout: float | None = None) -> list[ToolSpec]:
    """Register ``delegate`` (and by default ``delegate_parallel``) on *registry*.

    ``timeout`` applies to both tools: the whole single delegation, and each
    parallel task individually."""
    specs = [register_delegate_tool(registry, engine, timeout=timeout)]
    if parallel:
        spec, handler = make_parallel_delegate_tool(engine, timeout=timeout)
        registry.register(spec, handler)
        specs.append(spec)
    return specs


__all__ = ["ActionLabeler", "ProgressCallback", "SkillInjector", "SubagentEngine",
           "SubagentRoster", "SubagentSpec", "make_delegate_tool",
           "make_parallel_delegate_tool", "register_delegate_tool",
           "register_delegate_tools"]
