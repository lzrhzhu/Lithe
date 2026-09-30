# lithe

[![PyPI](https://img.shields.io/pypi/v/lithe.svg)](https://pypi.org/project/lithe/)
[![Python](https://img.shields.io/pypi/pyversions/lithe.svg)](https://pypi.org/project/lithe/)
[![License: MIT](https://img.shields.io/pypi/l/lithe.svg)](https://pypi.org/project/lithe/)

**lithe** is a reusable, storage-free ReAct agent kernel for Python: the
engine, framework, and protocols that drive a tool-calling agent, plus
optional capability bundles. The kernel knows nothing about how runs are
stored — or even whether they are — so any application (a thesis-writing
platform, a coding agent, a support bot) supplies its own tools, system
prompt, and persistence, and reuses everything else: the loop, the event
stream, undo, and delegation.

## Highlights

- **Zero-I/O core.** `import lithe` performs no file or network I/O. Every
  module in the root package is pure engine + protocol; anything that
  touches a disk or a socket ships as an optional bundle.
- **Storage-free by design.** Persistence is a pair of host-supplied
  interfaces (an `EventSink` to write, a `MemoryProvider` to replay).
  Don't want storage? Pass nothing. Want storage? `JsonlRunStore` is the
  zero-database default.
- **An honest event stream.** Streaming `assistant_delta` events, a
  `usage` event per model call (tokens, cost, context fullness), tool
  calls announced *before* they execute, and cancellation that closes the
  in-flight model stream instead of paying for a response nobody wants.
- **Undo built in.** Every bundled write tool registers a reverter; the
  `UndoEngine` is pure and storage-free like the rest of the core, and
  `AgentHost` exposes one-call `undo_run`.
- **Delegation without context bloat.** Subagents run isolated, are
  tagged so undo still reverts their work, fan out in parallel with
  per-delegation budgets, and fold their spend back into the parent run's
  stats.
- **Any OpenAI-compatible endpoint.** Chat-completions and Responses
  protocols behind one interface, with retries, jittered backoff,
  `Retry-After`-aware 429 handling, and SSE streaming; usage is
  normalized to one shape across both.

## Install

```bash
pip install lithe
# local development (editable install + test/lint deps):
pip install -e ".[dev]"
```

Requires Python 3.10+. The only runtime dependency is `httpx`.

## Quickstart

A minimal host: sandboxed file tools, a JSONL store, and a run.

```python
from lithe import AgentContext, LLMConfig, ToolRegistry
from lithe.bundles import AgentHost, JsonlRunStore
from lithe.bundles.workspace import Workspace, register_file_tools

registry = ToolRegistry()
# workspace_for(ctx) -> Workspace (a sandboxed root per user):
register_file_tools(registry, lambda ctx: Workspace(f"/data/{ctx.user_id}"))
store = JsonlRunStore("/var/lib/myapp/agent")                       # zero-DB default
host = AgentHost(registry,
                 LLMConfig(model=..., base_url=..., api_key=...),
                 store, build_system_prompt=my_prompt_builder)

ctx = AgentContext(run_id=rid, user_id=uid)
async for event in host.run(ctx, task, history=prior_turns):
    ...  # forward run_start / step / tool_call / tool_result / done to your frontend
```

A host supplies only its **tools**, **system prompt**, and (optionally) a
store backend — the engine, persistence, run envelope, undo, and (via
`subagents`) delegation are all reused.

## Architecture

### Core — `lithe` (zero I/O, zero business logic)

| Module | What you get |
| --- | --- |
| `runtime` | `AgentRuntime`, the ReAct loop as an async event stream: streaming deltas; cancellation checked between deltas; run budgets (`max_cost` / `max_total_tokens`) ending with `status="budget_exceeded"`; a repeat-call nudge for stuck models; a forced toolless wrap-up when the step budget is hit; mid-run context trimming; error-isolated sinks; an injectable `http_client` for connection pooling. |
| `llm` | OpenAI-compatible client: retry with jittered backoff, `Retry-After`-aware 429 handling, fail-fast on fatal 4xx, response-shape validation, and SSE streaming helpers. |
| `transports` | Chat-completions and Responses transports with usage normalized to `prompt_tokens` / `completion_tokens` / `total_tokens`; reasoning items replayed within the tool loop where the protocol supports it. |
| `tools` | `ToolRegistry`: register / unregister / dispatch, READ/WRITE/META category filtering, argument validation, per-tool timeouts, and pre-dispatch **middleware** for audit, quota, or human-in-the-loop confirmation of write tools. |
| `actions` | `Action` + `UndoEngine`: pure, storage-free undo; reverters may be sync or async. |
| `memory` | `replay_messages` / `recap_text` / `window_with_recap` / `run_timeline` + the `MemoryProvider` protocol; two-sided, window-safe replay reconciliation drops orphaned tool rows instead of sending API-rejected payloads. |
| `events` | `EventSink` with separate display + record channels, and SSE serialization (`to_sse`) that degrades non-JSON values instead of crashing. |
| `context` / `modes` | `AgentContext` plus host-defined modes via `register_mode(name, categories)`; `ctx.shared` is per-run state shared *by reference* with subagent contexts — the vehicle for cross-agent coordination. |

### Bundles — `lithe.bundles` (optional capabilities)

| Bundle | What you get |
| --- | --- |
| `host` | `AgentHost`: message assembly, the `run_start`/`done` envelope, error funneling, `StoreSink`, zero-config `undo_run`, and `DictToolAdapter` for wrapping dict-based tool systems. Forwards budgets, cancellation, and a shared HTTP client to every run. |
| `store` | `RunStore` / `ConversationStore` / `BlobStore` Protocols, plus `JsonlRunStore` — a zero-database JSONL store with content-addressed blob spillover for large action values and optional `fsync` crash durability. |
| `subagents` | `SubagentEngine` + `delegate` / `delegate_parallel` tools: isolated worker agents the orchestrator hands subtasks to — tagged for undo, bounded by `max_parallel`, per-delegation budgets, cancellation propagated from the parent run, live `subagent_progress` heartbeats. |
| `admin` | Admin-panel tools: `tool_categories`, `list_tools_admin`, `list_tool_packages_admin`, `check_packages`. |
| `patch` | `apply_patch`: line-oriented multi-file edits via the Codex `*** Begin Patch` envelope, a four-pass fuzzy matcher, and all-or-nothing application against an in-memory overlay. |
| `workspace` | Sandboxed file I/O: `read_file` / `write_file` / `edit_file` / `list_files` / `search_files` / `glob_files` with undo reverters, a stale-file guard that refuses to clobber externally changed files, and symlink-safe directory walks. |
| `todos` | A per-scope task list the agent plans against: `TodoStore` + atomic JSON persistence + `update_todos` / `list_todos` tools + a `todos_block` for the system prompt. |
| `images` | `image_info` (a stdlib-only header probe: dimensions, dpi, color mode) and `analyze_image` (one vision-model call, memoized per file hash + question) — image answers never enter the main conversation. |
| `sandbox` | `run_code` / `run_file`: Python execution with bubblewrap or passthrough backends, head+tail output truncation so tracebacks stay visible. WRITE-classified, so invisible in read-only modes. |
| `command` | `run_command`: opt-in native shell execution (bash/sh on Unix-like systems, PowerShell/cmd on Windows), bounded by timeout/output limits and started in the host-selected workspace. Commands run with the host user's permissions and are not sandboxed or undoable. |
| `download` | `download_file`: streaming HTTP(S) ingress into the workspace — SSRF-guarded (every redirect hop re-validated, all resolved addresses must be globally routable), size-capped, time-budgeted, landed atomically. |
| `skills` | Markdown skill libraries: flat `SkillLibrary`, package-aware `SkillPackages` with remote registry mirrors, and the `load_skill` tool. |
| `mcp` | `MCPManager` + `MCPServerConfig`: bridge external MCP servers (stdio or streamable-http) into the registry — `tools/list` pagination followed, a safe env allow-list for spawned servers (never all of `os.environ`), self-healing sessions. |

```python
from lithe.bundles import MCPManager, MCPServerConfig

manager = MCPManager([MCPServerConfig(
    name="zai", command=["npx", "-y", "@z_ai/mcp-server"],
    env={"Z_AI_API_KEY": "...", "Z_AI_MODE": "ZHIPU"},
    default_category="read")])
await manager.attach(registry)   # registry gains zai__* tools
```

## Safety defaults

Long-lived autonomous loops need guardrails on by default:

- **Path sandboxing** — workspace tools resolve inside the sandbox root;
  directory walks never follow symlinks, so code executed by `run_code`
  cannot plant a link to a host file and read it back.
- **SSRF guard** — `download_file` accepts only globally routable
  http/https addresses, re-validating every redirect hop (cloud metadata
  endpoints included in the rejection set).
- **Secret hygiene** — spawned MCP servers get an env allow-list, never
  the parent environment; URLs log as `scheme://host` so credentials in
  query strings never reach a log line.
- **Shell execution is explicit** — `run_command` is an optional host bundle,
  not a default tool. When enabled, commands run with the host user's
  permissions; they are not sandboxed and their side effects cannot be undone.
- **Write serialization** — READ tools run in parallel, every WRITE/META
  tool runs alone in model order: no write races.
- **Run budgets** — `max_cost` / `max_total_tokens` cut a runaway run
  short with no further tool execution or model calls.

## Storage model

The kernel stores nothing. A host provides:

- an **`EventSink`** (write side) — persists message records however it
  likes (DB / file / nowhere);
- a **`MemoryProvider`** (read side, optional) — replays prior turns;
- an **`UndoEngine`** fed from wherever the host kept actions.

## More

- `examples/minimal_host.py` — a runnable, offline minimal host (tools →
  run → events → undo); runs in CI.
- `examples/live_host.py` — the same flow against a real OpenAI-compatible
  endpoint (set `LITHE_API_KEY` / `LITHE_BASE_URL` / `LITHE_MODEL`; no
  default endpoint, it never spends tokens by accident).
- `CHANGELOG.md` — what changed and when.

## License

MIT
