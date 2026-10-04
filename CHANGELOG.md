# Changelog

## 0.9.3 (2026-10-04)

Renumbered baseline: the earlier 0.9.x cadence published twenty releases
in five days, so the PyPI history was cleaned up (old releases deleted;
their version numbers and filenames are permanently retired by PyPI) and
the published line restarts here at a saner pace. This release contains
everything from the rounds previously numbered 0.9.25 and 0.9.26.

The parallel-safety round: parallel subagents can no longer silently
clobber each other's writes, `delegate_parallel` gains per-task timeouts,
and delegation can no longer widen a restricted mode's powers. Plus the
todos localization round.

- **`update_todos` / `list_todos` descriptions localized** — the tool
  descriptions and the `todos` schema texts (array guidance, item
  content/status/priority) were the only English on an otherwise
  all-Chinese tool surface; mixed languages dilute instruction weight
  for models that anchor on the system prompt's language. The guidance
  is preserved one-to-one (optional planning tool; read before replace;
  send the COMPLETE list, not a delta; at most one in_progress; never a
  fixed count) — only the language changed.
- **run-wide file-mutation lock + write generations** (`lithe.bundles.workspace`)
  — the stale-content guard was check-then-act with an `await` between the
  check and the write, so two parallel subagents writing the same file
  both passed the check and the second silently clobbered the first
  (reproduced 60/60 under plain concurrent dispatch). Fixes, both
  required: (1) `write_file` / `edit_file` / `apply_patch` now hold a
  per-run `asyncio.Lock` (stored in `ctx.shared`, shared by reference
  with every subagent context) across check → write → revision-record;
  (2) the guard gained a per-path write *generation* (bumped by every
  in-run write) plus per-context "observed-at" stamps in `ctx.extra` —
  the first writer refreshes the shared snapshot to its own stat, which a
  stat comparison alone cannot distinguish from fresh, but the sibling's
  older generation stamp can: the second writer is refused with the
  existing re-read guidance. Reads stay lock-free (READ tools remain
  parallel); `read_file` now records its snapshot stat-BEFORE-read inside
  one worker-thread unit, so a write landing between the two leaves the
  snapshot older than the content seen — the next write is
  conservatively refused instead of a blind overwrite passing a
  fresh-looking snapshot.
- **`delegate_parallel` per-task timeout** (`lithe.bundles.subagents`) —
  `make_parallel_delegate_tool(engine, timeout=...)` caps each delegated
  task's wall-clock individually (counted while running, not while queued
  behind the `max_parallel` semaphore): a hung worker is cancelled — the
  delegation's inner runtime generator closes deterministically via the
  existing `aclosing` — and reported as that agent's failure block while
  its siblings keep running. `register_delegate_tools(..., timeout=)`
  now applies to both tools; previously the parallel batch had no
  timeout at all, so one stuck subagent hung the whole gather past any
  cancellation checkpoint.
- **mode fence on delegation** (`lithe.bundles.subagents`, `host`) —
  `AgentHost.run` stashes its `mode` in `ctx.shared["_host_mode"]`, and
  `SubagentEngine.trimmed_tools` filters each worker's toolset to that
  mode's admitted categories: an anchored (read-only) conversation can
  still delegate research to read-only workers, but can no longer hand a
  write-capable coder subagent the pen. Direct `engine.run` callers
  without a stashed mode keep the declared roster unchanged.

## 0.9.24 (2026-10-04)

The dangerous-command guard: `run_command` gains a pre-execution policy
layer. Catastrophic commands never run; destructive-but-scoped ones
require a human yes first.

- **`classify_command` / `make_command_guard`** (`lithe.bundles.command`) —
  a pure, dialect-agnostic classifier (bash / PowerShell / cmd) plus a
  pre-dispatch middleware factory for `run_command`. Three verdicts:
  **"deny"** — recursive deletion aimed at a filesystem anchor (`/`, `~`,
  `.`, `..`, `*`, `$HOME`, system trees, drive roots), `mkfs`, `dd` to a
  device, `format c:`, `shutdown`, fork bombs — refused outright, no
  approver can override. **"approve"** — `rm -r` on a scoped path,
  `sudo`, `git push --force`, `git reset --hard`, `git clean -f`, broad
  kills, `curl | sh`, `chmod -R`, `crontab -r` … — runs only when the
  host-supplied async approver returns True; without an approver channel
  the call is denied with guidance to hand it to the user, so an
  unattended agent never runs destructive operations silently. Anything
  else runs as before. The recursive-deleter family (`rm` / `rd` /
  `del` / `Remove-Item` / `chmod` / `chown`) is analyzed token-wise with
  per-dialect flag syntax (`-r` vs `/s`), so `rm -rf /usr` keeps `/usr`
  a target rather than a flag, and `git push --force-with-lease` stays
  unflagged. The `run_command` tool description now tells the model the
  guard exists and not to retry a refused command verbatim.

## 0.9.23 (2026-10-04)

The workspace-tools round: `search_files` gains the options a coding
agent actually reaches for, and `read_file` stops dumping binary noise
into model context.

- **`search_files` options** — `ignore_case`, `literal`, `context=N`,
  `limit`, all optional and combinable. `literal` escapes the pattern so
  `f(` searches as text instead of failing to compile; `context` expands
  each hit to grep-`-C`-style blocks (overlapping/adjacent blocks merged,
  `--` between the rest, clamped 0–10; `context=0` output is
  byte-identical with the previous format, so `read_file` line-number
  consumers are unaffected); `limit` caps matching lines (not display
  lines) with an honest truncation note naming the effective cap.
- **`read_file` on directories** — a directory `path` now lists one level
  of entries `ls`-style: subdirectories carry a trailing `/`, symlinks
  list as opaque names (never followed), ignored names stay hidden, and
  the same `offset`/`limit` paging applies with a continue-at hint.
- **`read_file` attachment/binary guard** — a 4096-byte magic-byte sniff
  runs before any text read: images (jpeg/png/gif/webp) are refused with
  a pointer to `analyze_image`/`image_info`, PDF/OOXML likewise point at
  `analyze_document`/`document_info`, other binary content (NUL bytes /
  >30% control chars) is refused outright, and UTF-16/32 BOM text is
  reported as an encoding problem instead of passing through
  `errors="replace"` mojibake. Magic bytes decide — a text file named
  `notes.png` still reads. Without the guard, one multi-MB image read
  landed in model context as ~20k replacement-character tokens of pure
  noise.

## 0.9.22 (2026-10-03)

A one-field follow-up to 0.9.20's observability round: the *host* run
envelope now reports how long the run took.

- **`duration_s` on the host DONE event** — `AgentRuntime` already stamped
  `RunStats.duration_s` and its own envelope carried it, but `AgentHost`'s
  richer DONE event (the one CLI-style hosts actually consume) omitted the
  field, so frontends had to re-time turns themselves. The host DONE event
  and the host `stats` dict now carry `duration_s` (wall-clock seconds of
  the whole run, prompt assembly included).

## 0.9.21 (2026-10-03)

A one-fix follow-up to the steering inbox (0.9.16): injected messages
are now actually persisted.

- **StoreSink writes user rows** — `on_record` historically persisted
  only assistant and tool records, so a steering injection's `_record`
  call was silently dropped: the injected line vanished from the stored
  conversation, and a resumed session replayed model answers referencing
  an input that was no longer in the transcript. User records now land
  like every other turn (the host writes the initial task row directly,
  so nothing doubles). Caught by wiring the CLI's steering path against
  the real store.

## 0.9.20 (2026-10-03)

The hardening + observability round: an injected client can no longer
silently time out, dirty gateway usage can no longer kill a run, every
event carries attribution and order, runs report how long things took,
parallel workers share one budget ceiling, and the workspace tools keep
their disk I/O off the event loop.

- **injected `http_client` timeout upgrade** — a host-supplied
  `httpx.AsyncClient()` carries httpx's 5-second default reads, far below
  `LLMConfig.timeout` (180s) and fatal for reasoning-model calls; the
  runtime now upgrades that exact default to `cfg.timeout` before driving
  the run. A deliberately narrowed or widened timeout (or a duck-typed
  client without one) is left untouched.
- **fault-tolerant usage normalization** — `norm_usage` coerces every
  integer field through a `_to_int` helper (comma-grouped strings like
  `"1,234"` parse; unparseable values degrade to 0), matching the
  runtime's long-standing `_as_int` contract for custom transports: one
  malformed usage field from a gateway can no longer crash an otherwise
  successful multi-step run from inside the built-in transports.
- **event attribution** — every event leaving the runtime is stamped with
  its run's `run_id` and a monotonic per-run `seq` (reset at each
  `run()`), so a consumer multiplexing runs over one channel can
  attribute, order and gap-check without per-host enrichment.
- **timing metrics** — `RunStats.duration_s` (also on the runtime-emitted
  `done` envelope); `elapsed_ms` on every `tool_result` (frontends can
  render "search_files (2.3s)" from the event, not private patching);
  `ttft_ms` on the first `assistant_delta` of each streamed call, echoed
  on that call's `usage` event (`None` when not streaming).
- **parallel-delegation budget visibility** — each delegation registers a
  live-usage slot in the run's shared state and folds every usage event
  into it as the child spends; concurrently running siblings count each
  other's live spend against their own `max_cost` / `max_total_tokens`
  caps, so N parallel workers share one ceiling instead of each burning
  the full cap (previously (N+1)× the ceiling could be in flight before
  any check fired). Slots are keyed per delegation and popped at
  completion — sequential delegations keep their independent caps, and
  the run-wide accounting fold is unchanged.
- **workspace tools off the event loop** — `read_file` / `write_file` /
  `edit_file` / `list_files` / `glob_files` and the `search_files` tree
  walk now run their disk I/O through `asyncio.to_thread` (batched reads
  for the scan), the same policy the images/documents bundles already
  followed: one slow read on a cold or NFS workspace no longer stalls
  every concurrent agent run in the process. Diffs and line-delta
  computation ride along.

## 0.9.19 (2026-10-03)

The document-perception round: models that read PDFs become reachable
through the same tool-tier pattern as vision, with the wire dialect
selected per endpoint family instead of guessed from the URL.

- **new `documents` bundle** — `document_info` (a stdlib-only probe: PDF
  magic + version + best-effort `/Type /Page` count, OOXML subtype via
  the zip central directory with a name-suffix fallback) and
  `analyze_document` (one chat call carrying the document as a content
  block plus the question, memoized per file hash + question + format,
  size-capped before reading, tool-level timeout). The document never
  enters the main conversation — the kernel's context budget / trimming /
  replay machinery is untouched, mirroring the images bundle.
- **`document_format` dialects** — `inline-file` (one
  `{"type": "file", "file": {"filename", "file_data": data-URL}}` block,
  the OpenRouter family incl. self-built routers speaking its format
  behind their own base_url), `files-api` (strict OpenAI two-step:
  multipart upload to `/files` with purpose `user_data`, then a
  `file_id` block), `none` (probe only). The format follows what the
  *gateway* accepts, not what the URL looks like.
- **`LLMConfig.document_format`** — endpoint knowledge the documents
  bundle reads off its perception config; the tool loop itself ignores
  it. Provider presets contribute each family's default (openai →
  `files-api`, openrouter → `inline-file`, zai/deepseek/qwen/moonshot →
  `none`), an explicit profile field or registration argument overrides,
  and an unrecognized value raises at registration instead of surfacing
  as a mystery 400 later.
- **400 diagnostics for dialect mismatches** — an endpoint 400 on the
  analyze call (or a failed `/files` upload) returns a tool result that
  names `document_format` and the alternatives, the same
  self-correcting philosophy as `extra_body_hint`.

## 0.9.18 (2026-10-02)

The reasoning-effort release: a first-class intensity knob for reasoning
models, mapped per protocol and safe to combine with vendor budget fields.

- **`LLMConfig.reasoning_effort`** — the vendor reasoning-intensity knob
  (`minimal`/`low`/`medium`/`high`, model dependent), passed through
  verbatim: the chat transport sends OpenAI-style top-level
  `reasoning_effort`, the Responses transport sends
  `reasoning: {"effort": ...}`. `None` (default) sends nothing; an
  unsupported value surfaces through the 0.9.17 400 diagnostics instead of
  silent mangling. Usage-side nothing changes: `reasoning_tokens` was
  already normalized, accumulated and priced.
- **`_merge_extra` deep-merges dict fields per key** — the Responses
  transport's internal `reasoning: {"effort": ...}` no longer clobbers a
  host's `extra_body["reasoning"]` siblings (`max_tokens` for budget-style
  models, `exclude`, ...): dict-vs-dict collisions combine key-wise with
  the internal value winning each key. The openrouter preset's notes now
  spell out its `reasoning` object (effort/max_tokens/exclude) and the
  `/models` `supported_parameters` capability source.

## 0.9.17 (2026-10-02)

The diagnosability round: a 400 that may stem from host-supplied vendor
fields now says so, and provider presets move "which fields does this
endpoint understand?" from every user to one maintained table.

- **400s name the `extra_body` suspects** — strict gateways (OpenAI's API
  among them) reject unknown request arguments with a plain 400; a lenient
  one ignores them. When a request carried `LLMConfig.extra_body` and came
  back 400, the raised `HTTPStatusError` now carries the field names, the
  gateway's own error body (first 200 chars, where OpenAI-style 400s name
  the offending argument), and a `lithe_hint` attribute that the runtime
  appends to the run's error event — on all four paths (chat/responses ×
  stream/non-stream; the Responses 400-degradation cascade still runs
  first, unaffected). Without `extra_body` the error is byte-identical to
  before.
- **new `providers` bundle** — vendor presets as plain data:
  `get_preset(name)` / `apply_preset(name, **overrides)` /
  `known_providers()`, shipping openai / zai / deepseek / openrouter /
  qwen / moonshot. Each preset carries only what is stable (base_url,
  transport, safe-for-every-model notes on the vendor-specific knobs);
  model-specific switches stay in `notes` instead of hard-coded
  `extra_body`, because half the models behind an endpoint would break.
  `apply_preset` merges preset-under-overrides (dict fields per key),
  strips `notes`, skips `None` overrides, and never supplies credentials.
  Unknown names raise with the known list. lithe-cli profiles consume this
  via their `provider` field (see the lithe-cli 0.8.1 changelog).

## 0.9.16 (2026-10-02)

The P1 round: five host-ergonomics gaps in the zero-I/O core — vendor
field/header passthrough, self-computed cost, enum + transform tool
validation, multimodal fidelity on the Responses path, and a mid-run
steering channel.

- **`LLMConfig.extra_body` / `default_headers`** — vendor request fields the
  kernel does not model (`top_p`, `seed`, `stop`, `response_format`,
  `enable_thinking`, Responses `reasoning` effort, ...) now travel on every
  call through both transports, stream and non-stream; gateway headers
  (OpenRouter's `HTTP-Referer`/`X-Title`, `OpenAI-Organization`, ...) merge
  over the bearer/JSON defaults. Internal keys (`tools`, `tool_choice`,
  `max_tokens`, `model`, ...) win on collision — those knobs have
  first-class config members, and a payload field must not silently break
  the loop mechanics. Previously each of these meant a custom transport.
- **`LLMConfig.pricing` makes `max_cost` real without gateway cost** —
  OpenAI's API and many gateways never report `usage.cost`, so cost
  accounting and the `max_cost` budget silently read 0 there. A
  per-1M-token price table (`{"prompt", "completion", "cached_prompt"?}`,
  validated at construction) computes each call's cost when the gateway is
  silent; a reported cost still wins, and cached input defaults to the
  prompt price (over-counting is the safe direction for a budget).
- **`validate_args` checks top-level `enum`s** — the highest-value
  self-correcting failure class after required/type: the model picking a
  disallowed value now gets "只能取 [...] 之一" back as a failed tool
  result listing every legal value, instead of executing with junk.
- **`registry.add_transform(fn)`** — a new pre-*validation* argument hook,
  `async (ctx, name, args) -> args`: canonicalize paths, inject defaults,
  clamp sizes, redact secrets. Middlewares run after transforms and see the
  rewritten args (still observe-or-veto); a transform returning a non-dict
  is rejected as a failed tool result instead of smuggling junk into the
  handler.
- **Responses transport maps multimodal input instead of dropping it** —
  `_messages_to_input` now translates `image_url` blocks (remote URL or
  inline `data:` base64) to `input_image` items block-by-block; any block
  it cannot express **raises** instead of silently degrading, which used to
  have the model answer about an image it never saw. Plain-string content
  keeps its byte-identical fast path. Side fix: `_message_size` counts
  list-type (multimodal) content — text blocks by length, image blocks by
  payload — where it previously counted 0, skewing the trim budget and the
  chars-per-token calibration.
- **Steering inbox — `AgentRuntime.run(..., inbox=...)`** (forwarded by
  `AgentHost.run`): a queue of user texts (`asyncio.Queue` in-loop,
  `queue.Queue` cross-thread) drained at each step boundary; each text
  becomes a recorded, replayable user message announced via a
  `user_injected` event, so "the user typed while the agent worked"
  reaches the model on its next call. Precedence: stop and budgets are
  checked first; the forced wrap-up step skips the drain (its tools are
  withheld — an injected request could not be acted on) and unconsumed
  items stay in the host's queue. The handle is stashed in
  `ctx.shared["_runtime_inbox"]` for symmetry with `_runtime_stop`, but
  subagent runtimes deliberately do not consume it — steering addresses
  the orchestrator.

## 0.9.15 (2026-10-02)

The kernel-hardening release: three behavior-level fixes in the zero-I/O
core — a process-global flag leaking between endpoints, a stuck-model guard
that misfired on legitimate re-reads, and a context trimmer that gave up
while the next request still overflowed the window.

- **`ResponsesTransport` include-degradation memory is per instance** — the
  "gateway 400-rejected-the-`include`-field" flag was a *class* attribute:
  one gateway's 400 permanently disabled reasoning-include requests for
  every other host and endpoint in the same process. It is now instance
  state (mirroring `ChatCompletionsTransport._include_usage`); each runtime
  owns its transport, so the memory still spans that runtime's runs without
  cross-endpoint bleed.
- **the identical-repeat nudge resets after a mutating tool** — `repeat_state`
  counted every (tool, args) occurrence for the whole run, so a model doing
  the correct read → edit → read → edit cycle on one file got "repeating the
  same call" nudges from the 4th read on (the stale-file guard even demands
  re-reads). Counters are now epoch-based: every WRITE/META dispatch bumps
  the epoch and resets all read signatures — a read after a write is a fresh
  observation. A mutating tool's *own* identical retries still count (that
  re-send is exactly the stuck loop the guard exists for), and the nudge
  text now says 连续 (consecutive).
- **mid-run context trimming escalates to dropping old exchanges** — tier 1
  (head+tail shrinking of old tool results) only shrinks tool messages, so
  a conversation over budget because of long user/assistant prose or
  protected-recent results went to the API anyway, overflowing the window.
  Tier 2 now drops complete old exchanges (one assistant turn plus its tool
  results, always as a unit, so pairing stays valid) oldest-first and
  replaces them with a single omission note. System/user messages are never
  dropped, the newest exchanges are protected, drops commit only when they
  actually reach the budget (no losing history for nothing), the note stays
  unique across passes, and a still-over-budget conversation logs a warning
  instead of silently overflowing. Dropped turns remain in sinks/records —
  only the live model context loses them.

## 0.9.14 (2026-10-02)

The store-v2 release: the JSONL store learns what multi-session hosts
(lithe-cli's workbench) need — time, tokens, conversation metadata — all
additive, all backward compatible.

- **run timestamps** — `create_run` stamps `created_at` and `finish_run`
  stamps `finished_at` (unix floats, overridable for tests). Legacy rows
  read back as `None`, never a fabricated zero.
- **token persistence** — `finish_run` accepts optional
  `prompt_tokens` / `completion_tokens` / `cached_tokens` /
  `total_tokens` and `AgentHost._close_run` forwards the run's
  `RunStats` totals, so a resumed session can reconstruct its
  cumulative usage. Unknown (legacy, or a transport that reports no
  usage) stays `None` — "unknown", not "zero".
- **conversation metadata** — `create_conversation` takes a `meta`
  dict (workspace, pinned profile/model, …) and the new
  `update_conversation_meta` merges patches append-only; the fold
  survives reopens.
- **conversation summaries** — `conversation_summaries(user_id)`
  returns one aggregated row per conversation (title, meta, run count,
  last status/task/model, updated_at, cost + token totals), newest
  activity first, falling back to file order for legacy unstamped rows.
- **`messages_for_conversation`** — the store-side join hosts used to
  hand-roll (`runs_for_conversation` → `messages_for_runs`).

## 0.9.12 (2026-10-01)

The Windows release: `run_code` works outside POSIX for the first time.

- **sandbox env keeps Windows children alive** — the secret-free child
  environment now forwards `SYSTEMROOT` (plus `SYSTEMDRIVE`, `WINDIR`,
  `COMSPEC`, `PATHEXT`, `TEMP`/`TMP` and a few more runtime names) on
  Windows. Without `SYSTEMROOT` a child Python aborts before running
  any code with `Fatal Python error: _Py_HashRandomization_Init` (hash
  seeding reaches for the CryptoAPI, which resolves through it). None
  of the forwarded names can carry secrets, so the leak guarantee is
  unchanged.
- **timeout kills work on Windows** — the kill path used
  `os.killpg`/`signal.SIGKILL`, neither of which exists there, so a
  timed-out run raised `AttributeError` instead of killing. Killing now
  branches per platform: POSIX keeps the process-group kill; Windows
  walks the process tree with `taskkill /PID <pid> /T /F`, so spawned
  grandchildren die with the child instead of surviving it.

## 0.9.11 (2026-09-30)

The todo-restraint release: planning stays available but stops firing
on ordinary questions, and persisted lists get stricter integrity.

- **`update_todos` is explicitly optional** — the tool and parameter
  descriptions now say not to call it for questions, explanations,
  single-step actions or small edits, and to consult `list_todos` and
  preserve unrelated active items before replacing an existing list.
  The "one task per distinct step, never a fixed count" guidance from
  0.9.9 is retained.
- **at most one in_progress is enforced** — `TodoStore.replace` rejects
  a list with two or more active items instead of silently accepting a
  state the tool contract forbids; failures leave the previous list
  untouched.
- **load/restore paths sanitize too** — `TodoStore(...)`, `restore()`
  and `JsonTodoStore` loads repair conflicting active statuses, drop
  malformed rows and cap the list at `max_todos` (the constructor also
  validates `max_todos` is a positive integer), so undo and hand-edited
  files can no longer reintroduce an invalid list.
- **collision-safe atomic saves** — the JSON temp file is uniquely named
  per save, so two processes writing the same scope can no longer clobber
  each other's intermediate `.tmp` file.

## 0.9.10

The line-delta release: file-mutating tools now report coding-agent
style `+N -M` line counts, so frontends can show what an edit changed
at a glance.

- **`write_file` / `edit_file` / `apply_patch` summaries carry counts** —
  e.g. `写入 a.py（+2 行，新建）`, `编辑 a.py（+3 -1 行）`, `应用 2 项（+5 -2 行）`.
  Counts come from a `difflib.SequenceMatcher` over the real before/after
  line lists (deleted files count their lines as removed; identical
  rewrites add no note).
- **`file_change` UI events carry `added` / `removed`** alongside
  `old`/`new`, so hosts and sinks can render deltas without re-diffing
  full contents. Undo contracts are unchanged.

## 0.9.9

The todo-planning fix: models kept drafting exactly three tasks no
matter the work, because nothing in the tool contract said the list
should match the actual step count.

- **`update_todos` sizes the list to the work** — the tool description
  and the `todos` array description now say explicitly: one task per
  genuinely distinct step; a two-step errand gets 2 tasks and a
  ten-step build gets 10; never pad or clip to a fixed count (e.g.
  always 3). Behavior is otherwise unchanged (replace semantics,
  `max_todos=20` ceiling, undo, persistence).

## 0.9.8

The project-workspace fix: the file-count guard measured the wrong thing
and made real project directories unwritable for the agent.

- **root cause** — `Workspace.write()` refused NEW files whenever the whole
  tree held more than `max_files` (500) entries, counting everything via
  `rglob("*")`. A project workspace holding a `venv`/`node_modules`/`.git`
  (thousands of files the agent never touches), or simply a large repo,
  therefore rejected every new-file write — and `undo`'s restore path
  (`_revert_write`/`_revert_edit` → `ws.write`) failed the same way. The
  guard's stated purpose was capping *unbounded agent growth*; the size of
  the user's pre-existing project has nothing to do with that.
- **growth guard moved to the tool layer, per-run** — `Workspace.write` /
  `write_bytes` never count the tree again (hosts, reverters, undo always
  write through). The brake now measures what it meant to: how many NEW
  files this run creates through the bundled write tools
  (`write_file`, `apply_patch` adds/move-destinations, `download_file`),
  tracked in `ctx.shared` and shared by reference with subagent contexts —
  parallel workers count against the same `max_files` budget. Overwriting
  existing files never counts.
- **dependency/VCS dirs ignored** — `venv`, `node_modules`, `.git`, `.hg`,
  `.svn`, `.tox`, `.mypy_cache`, `.ruff_cache` join the default `ignored`
  set (alongside `.venv` etc.), so listings, search, glob, and `_count()`
  no longer drown in them (a `venv`-bearing workspace previously listed
  3800+ entries straight into the model's context).
- **tests** — regression cases for the venv-bearing project workspace
  (write succeeds), the per-run cap (runaway loop refused at the budget,
  overwrites free, subagents shared), direct-I/O freedom, and undo restore
  in an over-`max_files` workspace.

## 0.9.7

The shell release: an opt-in command-execution bundle closes the gap
where hosts needed the agent to run system commands (installs, builds,
git) rather than only Python.

- **`command` bundle** — `CommandRunner` + `register_command_tools`
  register a `run_command` tool that executes a shell command in the
  host-selected workspace. Shell selection is cross-platform: `auto`
  picks bash (sh fallback) on Linux/macOS and PowerShell (cmd fallback)
  on Windows; `bash` / `sh` / `powershell` / `cmd` can be forced per
  call. Bounded by a timeout (default 120s, kernel backstop +10s) and
  head+tail output caps per stream; the whole child process group is
  killed on timeout (POSIX `killpg`, `taskkill /T /F` on Windows) so
  grandchildren cannot wedge the call. The child env is a filtered
  allow-list (PATH/locale/proxy basics) — `*_API_KEY`-style secrets
  never leak into commands.
- **classification** — `run_command` is WRITE-category: invisible in
  read-only modes and serialized with other writes. Its tool description
  states plainly that commands run with the host user's permissions, are
  not sandboxed, and cannot be undone — enabling it is a host trust
  decision, never a default.
- **tests** — 10 new cases in `tests/test_lithe_command_bundle.py`:
  stdout/exit-code capture, timeout group-kill, head+tail truncation,
  stdin, empty-command refusal, WRITE classification + anchored-mode
  invisibility, env secret filtering, and Windows argv selection
  (PowerShell pick, cmd fallback) verified without needing Windows.

## 0.9.6

- **`SubagentEngine(capture_actions=False)`** — hosts whose tools log their
  own domain actions (custom undo kinds a generic `file_change` capture
  cannot know — e.g. an app-wide `chapter_create` / `guidance` vocabulary)
  could not stop the subagent run's StoreSink from ALSO capturing the
  `file_change` ui into a second action row, duplicating every delegated
  mutation. The new flag mirrors `AgentHost(capture_actions=False)` and is
  forwarded to the engine's internal StoreSink; message recording is
  unaffected.

## 0.9.5

Docs-and-rename release: the beta→lithe sweep is finished and the README is
rewritten as a proper landing page.

- **rename leftovers fixed** — README and this changelog still told hosts to
  set `BETA_API_KEY` / `BETA_BASE_URL` / `BETA_MODEL` for
  `examples/live_host.py`, which actually reads `LITHE_*`; following the docs
  failed the example's preflight check. (The `Development Status :: 4 - Beta`
  classifier is a maturity level, not the old name, and stays.)
- **README rewritten** — the per-module changelog-style walls of prose are
  replaced by a structured overview: pitch, highlights, install, quickstart,
  an architecture table per core module and per bundle, a safety-defaults
  section (path sandboxing, SSRF guard, secret hygiene, write serialization,
  run budgets), the storage model, and examples. The when/why of each change
  keeps living in this file, as before.
- packaging metadata (name, description, license) already read `lithe`;
  no code changes in this release.

## 0.9.4

- **`reasoning_scope`** — hosts choose which assistant turns replay their
  reasoning items: `"loop"` (default, previous behavior: active tool loop
  only) or `"conversation"` (every turn — cross-turn chain continuity for
  quality-over-cost policies; the kernel still never compresses reasoning).
  `LLMConfig` validates the value; both transport `complete` signatures carry
  it through.
- **400-degradation cascade** — a gateway 400 now degrades in two steps
  before failing: drop the `include` parameter (remembered process-wide),
  then drop reasoning *input* items for that call (e.g. items minted by
  another model after a host-side switch). Applies to the streamed path too,
  running only while nothing has been delivered; deltas still stream through
  unbuffered.
- **token-calibrated trim budget** — `AgentRuntime(context_token_threshold=
  0.8)` (default): when the host declared `LLMConfig.context_window`, the
  mid-run char budget derives from the real token window — `window ×
  threshold × chars_per_token`, with the ratio calibrated from each call's
  measured `prompt_tokens` (conservative 2.0 default before the first
  measurement). This supersedes the plain 400k-char `context_budget`, fixing
  the chars-vs-tokens unit mismatch; without a declared window the classic
  char budget applies unchanged.

## 0.9.3

- **reasoning pass-back within the tool loop** — reasoning models (Responses
  protocol) return their chain-of-thought as encrypted `reasoning` output
  items; multi-step tool loops must pass them back or the model re-derives
  its plan every step. `ResponsesTransport` now (a) requests them via
  `include: ["reasoning.encrypted_content"]` (a gateway 400 that looks like
  an `include` rejection is retried once without it, remembered
  process-wide), (b) captures them in `_parse_output` (unified result gains
  a `reasoning` field), and (c) re-emits stored items from `messages` into
  `input` ahead of that turn's `function_call` items — scoped to the active
  loop (assistant turns after the last user message), so reasoning from
  previous turns / a previous model is dropped automatically once a new user
  message arrives. `LLMConfig(reasoning_replay=False)` is the escape hatch
  for gateways that reject reasoning input items.
- The runtime attaches captured items to the assistant message
  (`_normalize_assistant` keeps a `reasoning` key, opaque and never trimmed —
  dropping one orphans its function_call pairing), records them on the
  persistence channel (`on_record` rows carry `reasoning` when present),
  emits a display-only `reasoning` event carrying the human-readable summary
  digest, and `replay_messages` rehydrates them from stored rows (JSON string
  or list) with strict type filtering. `run_timeline` surfaces the digest too.
- `norm_usage` also lifts `output_tokens_details.reasoning_tokens` to a
  top-level `reasoning_tokens`; `RunStats` accumulates it and `usage` events
  carry the per-call value.
- chat-completions flavor (`reasoning_content` / `<think>`) is deliberately
  NOT replayed — per-turn state by vendor contract; `include_reasoning` is a
  no-op on `ChatCompletionsTransport`.

## 0.9.2

- **usage: cached-token accounting** — `norm_usage` now lifts cached input
  tokens to a top-level `cached_tokens` from either vendor detail shape
  (`prompt_tokens_details.cached_tokens` on chat-completions,
  `input_tokens_details.cached_tokens` on Responses), `RunStats` accumulates
  them across calls, and the per-step `usage` event carries the per-call
  value. Cached tokens are a *subset* of `prompt_tokens` (not additive), so
  hosts can show "输入 X（缓存 Y）· 输出 Z" without double counting.

## 0.9.1

- **sandbox: optional `mount_proc` (default True)** — Docker containers commonly
  forbid mounting a fresh procfs even with seccomp/apparmor unconfined
  (`mount("proc")` → EPERM) while pid/mount/user namespaces work fine, so a
  host running inside such a container now constructs
  `CodeRunner(..., mount_proc=False)` instead of losing the sandbox entirely.
  Ordinary compute code (numpy/pandas parsing, plotting) doesn't need /proc;
  multiprocessing/psutil-dependent code does.

## 0.9.0

New **download bundle** — the controlled ingress the network-isolated sandbox
deliberately lacked. `register_download_tools(registry, workspace_for, *, max_bytes,
timeout, max_redirects)` registers `download_file(url, path?)`, streaming one
HTTP(S) resource into the same `Workspace` the file/code tools use, closing the
"search → download → read/parse" loop without ever handing the model a shell:

- **SSRF guard**: http/https only; redirects followed manually with *every hop*
  re-validated; all resolved addresses of every hop must be globally routable
  (`ipaddress.is_global` rejects loopback/private/link-local/CGN/reserved/
  multicast — cloud metadata included). Checking all A/AAAA records up front
  narrows DNS rebinding to the TTL window (documented, not eliminated; an
  egress proxy decides its own destinations).
- **size cap**: `Content-Length` pre-check plus a streaming accumulation
  cutoff — a lying header gets cut mid-stream and the partial `.part` file
  is removed; downloads land atomically (`.part` → rename).
- **no overwrites**: an existing target is a failed result telling the model
  to pick another name, so the tool mutates nothing undoable (no reverter).
- **defaults**: `max_bytes` 64MB, total-time budget 120s (kernel `ToolSpec`
  timeout as backstop), max 5 redirects; default landing path
  `downloads/<basename>` with a Content-Type→extension fallback.
- WRITE-classified like the other workspace-mutating tools (serialized,
  excluded from read-only modes).

## 0.8.0

Kernel robustness (P2 round), subagent/store/host integrity, tool-bundle
hardening (MCP env allow-list, pagination, credential-safe logs), packaging
fixes for the 0.7.0 PyPI page (runnable README quickstart, LICENSE file).
Behavior change: `run_code`/`run_file` are WRITE-classified (they can mutate
the sandboxed workspace).

### Kernel robustness (P2 round)

- **tool_call ids are synthesized at one point**: a non-streamed gateway that
  omits `id` previously produced `tool_call_id: null` on the paired result
  (a 400 on the next request), and the streaming assembler minted
  per-response `call_0`-style ids that collide across steps. The streamed
  side now leaves ids `None` and `_normalize_assistant` synthesizes
  collision-free ids (`call_` + uuid fragment) in place, so messages,
  records and events all agree.
- **replay window never starts with an orphan tool message**: a tool row
  whose calling assistant turn fell outside the host's window is dropped
  (its call id was never emitted in-window) instead of being sent as a
  first message the chat API rejects.
- **empty model responses are retried once and reported honestly**: a
  completely empty response (no text, no tool calls, not truncated) gets one
  retry; if it stays empty the run ends with `status="empty_response"` and
  an error event instead of "successfully" finishing with empty text.
- **synthetic run endings are recorded**: the max-step / budget / wrap-up
  fallback texts now land in `messages` and in the record channel (like any
  assistant turn), so sinks and the next run's replay see why the run ended
  — previously they were display-only.
- **`to_sse` never crashes on non-JSON values** (`default=str`): a
  `datetime` (or Path, or set) inside a tool's `ui` payload serializes
  instead of raising TypeError out of the host's SSE layer.
- **optional runtime run envelope**: `AgentRuntime(envelope=True)` emits
  `run_start`/`done` itself for hosts that drive the runtime directly (off
  by default so `AgentHost` hosts never see duplicates).
- **injectable HTTP client**: `AgentRuntime(http_client=...)` /
  `AgentHost(http_client=...)` reuse one host-owned `httpx.AsyncClient`
  across runs (never closed by the runtime) — connection pooling, limits,
  proxy/verify/redirect config for long-lived service hosts; the default
  per-run client is unchanged. Subagent runtimes forward the host's client.
- **inner generators are aclosed deterministically**: the runtime closes its
  step generator and transport streams via `contextlib.aclosing`, the host
  closes the runtime generator, and `SubagentEngine.run` closes the
  sub-runtime generator — a disconnecting consumer releases the HTTP client
  and streams immediately instead of waiting for GC finalizers.
- **`dispatch`/`validate_args` reject non-object args as failed tool
  results**: a JSON string parsing to a number/array (or any junk value)
  previously raised a bare `TypeError` from `validate_args` — outside the
  handler try — when `dispatch` is called as a public API.
- **`run_timeline` no longer sniffs error prefixes**: a tool row without
  `meta` reports `ok=None` (unknown) instead of probing the content for the
  Chinese error-message prefix, decoupling replay from a display string;
  unknown-ok legacy rows still rebuild their side-effects.

### Subagents / orchestration

- **`list_actions(subagent=...)`**: the RunStore protocol (and JsonlRunStore)
  gained a subagent filter applied during the fold, so the engine's
  per-delegation snapshots skip other rows' blob rehydration — a long
  many-delegation run no longer pays O(N²) reads over `actions.jsonl`.
- **`delegate_parallel` rejects duplicate agents in one batch**: two parallel
  tasks on the same agent share the subagent tag, so each would claim the
  other's actions in its "new since snapshot" set.
- **`make_delegate_tool(engine, timeout=...)`** (and `register_delegate_tools`)
  wire a `ToolSpec.timeout` onto delegation: a hung subagent tool can no
  longer hold the orchestrator's whole step forever.
- **`SubagentEngine.roster` is read-only** (a property): the delegate schema
  enum freezes over the roster at registration time, so reassigning it
  afterwards left "unknown subagent" shadows; build a new engine instead.
- **`_thin_event` caps what it forwards**: `tool_call` args (JSON-capped at
  500 chars) and `tool_result` summary/error join assistant text at the 300
  -char cap — a 50KB `write_file` payload no longer rides the progress
  callback that exists to be thin.

### Store / host integrity

- **`undo_run` surfaces status-mark failures**: a reversion that succeeded
  but whose `set_action_status("reverted")` failed (store IO error) now
  appends to `report.errors` and flips `ok=False` instead of being swallowed
  — an unmarked row re-runs its reverter on the next undo, and a
  non-idempotent reverter would damage twice.
- **`JsonlRunStore`: torn lines are loud, writes can fsync**: unparseable
  lines log a warning and count in `dropped_lines` (a torn line can be an
  undo-able action going missing); `fsync=True` flushes each append,
  matching the crash-safe claim.
- **todos survive corrupt files and crash mid-save**: `JsonTodoStore`
  sanitizes items on load (bad status / missing content rows are dropped
  instead of crashing `to_block` — which renders into every run's system
  prompt), catches OS/decode errors, and saves atomically (tmp +
  `os.replace`); `TodoStore.to_block` is defensive about malformed items.
- **`DictToolAdapter` warns on handler-less specs and accepts `reverters=`**:
  a typo'd handler name no longer silently drops the tool, and dict-tool
  hosts can wire their own action kinds into `undo_run`'s default map.

### Tool bundles

- **MCP: `tools/list` pagination followed** (`nextCursor`, page-capped at
  64) — paginated servers no longer silently lose half their tools.
- **MCP: stdout non-object JSON lines are ignored** instead of crashing the
  reader loop with an `AttributeError` that took the whole session down.
- **MCP: stderr tail is bounded** (`deque(maxlen=64)`) — a chatty server no
  longer grows an unbounded list for the process lifetime.
- **MCP: the timeout path's `notifications/cancelled` is bounded (1s)** — a
  server that stopped reading its stdin has a full pipe, and an unbounded
  drain hung the timeout path it was trying to escape.
- **MCP: spawned servers get a safe env allow-list** (PATH/locale/HOME/
  TMPDIR + the configured `env`), never all of `os.environ` with its
  secrets; `inherit_env=True` restores the legacy full inheritance,
  `inherit_env=False` forwards only `env`.
- **MCP: logs don't leak credentials**: connected-server logs print
  scheme://host only, and `parse_servers` errors name the offending key and
  type instead of embedding the raw spec (which can carry keys in URLs).
- **`edit_file` fuzzy path aligns trailing blanks**: a phantom trailing
  newline in `old_text` now drops its counterpart in `new_text` (mirroring
  `apply_patch`), so each fuzzy hit no longer inserts an extra blank line.
- **`apply_patch`: `*** End of File` chunks run the tail-anchored ladder at
  full strength before any forward match** — a whitespace-mismatched tail
  beats an exact look-alike earlier in the file instead of silently editing
  the wrong site (the forward fallback for non-tail patterns is kept,
  matching the codex scenario suite); two same-position pure insertions in
  one section keep document order (splices tie-break on chunk sequence —
  back-to-front application reversed them before).
- **images: stat before reading, bounded probe window, off-loop reads**:
  `image_info` reads at most a 256KB header window (constant memory for
  huge files), `analyze_image` refuses oversize files by stat instead of
  loading them first, both read off the event loop, and `detail="auto"`
  shares a cache key with an omitted detail (the API treats them
  identically — no double billing for the same question asked both ways).
- **`search_files` can't hang the loop**: lines longer than 10k chars are
  skipped (catastrophic backtracking is exponential in the subject), the
  scan carries a 10s wall budget with an honest partial-result notice, the
  handler yields to the event loop so its 30s `ToolSpec.timeout` can
  actually fire.
- **skills: failed remote refreshes clean up and stay quiet**: a failed
  refresh removes its `.staging` dir, keeps the previous cache
  authoritative (the new index signature commits only after the swap) and
  logs at debug; `_ensure_remote` no longer `pass`es in silence.
- **`Workspace.list(dirs=...)` goes through `safe_path`**: an escaping
  `dirs` entry raises the guard's clean error instead of walking host
  paths.
- **`run_code`/`run_file` descriptions state their side effects are not
  undoable** (no `file_change` events → no undo records; durable edits
  belong to the write tools).

### Facade / packaging

- **README quickstart is runnable again**: the sketch wrapped its root in a
  `Workspace` (the API takes `workspace_for(ctx) -> Workspace`, not a str)
  and imported `AgentContext` — both shipped broken on the 0.7.0 PyPI page.
- **LICENSE file added** (MIT, as the README always claimed) plus the
  `license` field and classifier in pyproject.
- **placeholder project URLs removed** (they pointed at a non-existent
  github org); restore them when the repository exists.
- **`AgentHost(capture_actions=False)`** reaches the StoreSink opt-out that
  previously required hand-building the runtime — `examples/minimal_host.py`
  uses it to stop its own ActionSink from double-capturing.
- **`lithe.skills` alias emits a `DeprecationWarning`** on import.
- **`examples/live_host.py` has no default endpoint/model** — it requires
  `LITHE_*` env vars instead of silently hitting a third-party URL.
- **CI**: Python 3.13 leg added; the offline example runs as a CI step.
- CHANGELOG fix: the stale-file guard's revision map lives in `ctx.shared`
  (the 0.7.0 entry said `ctx.extra`).

- **`tool_choice` never travels without `tools`**: both transports omit the
  field when the request carries no tools list — OpenAI-compatible endpoints
  reject the combination with a 400, which broke zero-tool hosts on their
  first call and the max-steps wrap-up (`tool_choice="none"`) exactly when
  it triggered.
- **workspace walks never follow symlinks** (`walk()` / `list_files` /
  `search_files` / `glob_files`): model-executed code (`run_code` binds the
  workspace read-write) could plant a symlink to a host file and read it
  back through the walk family, bypassing `safe_path`. Symlinks now list as
  an opaque `symlink` type instead of being followed.
- **zero-configuration `undo_run` actually reverts**: `ToolRegistry.register`
  gained `revert_kind` (the `Action.kind` a reverter serves, defaulting to
  the tool name), and the bundled tools register the exact kinds their
  stored actions use (`file_write` / `file_edit` / `file_delete` /
  `todo_replace`). The default reverter map therefore lines up with the rows
  a `StoreSink` writes — previously the fallback was keyed by tool name and
  silently reverted nothing while reporting `ok=True`. A loud warning now
  fires when no loaded action kind matches the reverter map.
- **`BlobStore.load` validates refs**: a blob ref must be
  `sha256:` + 64 hex chars; forged refs (e.g. a tool-supplied
  `old_value` of `blob:sha256:../../.ssh/id_rsa`) raise instead of reading
  arbitrary host files on rehydrate, and literal look-alike strings surface
  verbatim rather than crashing the read.
- **MCP session (re)spawns are serialized**: `revive()` and
  `_sync_registry()` share a spawn lock with an in-lock liveness re-check —
  two parallel dispatches to a dead server (READ-classified MCP tools run
  concurrently) no longer each start a subprocess and leak the loser's
  process and reader tasks.
- **malformed tool_calls and dirty usage can't kill a run**: a tool_call
  whose `function` dict lacks a usable `name` is answered as a failed tool
  result (the id still gets its tool message) instead of raising KeyError
  out of the generator with no ERROR event; non-integer usage values from
  custom transports degrade to zero instead of crashing the loop.
- **pre-runtime host failures funnel into the error envelope**: a crashing
  `build_system_prompt`, store IO error, or multimodal (list) history
  content now yields an `error` event and closes the run `failed`, instead
  of escaping the generator and stranding the run at `running` forever.
- **`JsonlRunStore.create_run` rejects duplicate run ids** (they corrupted
  the fold — chimera `StoredRun`, doubled listings, and cross-user
  `run_final` overwrites); legacy files with duplicate headers list once.
- **subagent spend folds into the parent run's accounting**: each
  delegation's cost/tokens accumulate in the run's shared state, and the
  host `done` event, stats dict and store row report the totals plus a
  `subagent_*` breakdown — a multi-delegation run no longer reports only
  the orchestrator's own spend.
- **sandbox timeouts kill the process group and reap with a deadline**:
  executions start in their own session; on timeout the whole group is
  SIGKILLed (a `Popen`-spawned grandchild holding the stdout pipe can no
  longer drag the tool far past its timeout), the post-kill drain is
  bounded at 5s, and `run_code`/`run_file` carry a kernel-level
  `ToolSpec.timeout` backstop.
- **patch chunks treat `" "` as a blank context line**: only a truly empty
  line ends a chunk — a blank context line inside an Update chunk used to
  either fail parsing or silently truncate the chunk so the fuzzy ladder
  could anchor it at the wrong site.
- **Responses-path fatal 4xx fail fast** (matching the chat path): 401/400
  class errors are not retried through the attempts budget, in both the
  plain and streaming variants; the chat `stream_options` adaptive
  downgrade now triggers only on HTTP 400 (not any 4xx) and replays once
  instead of under a fresh retry budget.
- **truncated generations are marked, not silent** (`finish_reason`
  passthrough): transports now surface `finish_reason` in the unified result
  (chat-completions natively; the Responses API's
  `status="incomplete"` + `incomplete_details.reason="max_output_tokens"`
  maps to `"length"`). When a generation was cut off by the token cap the
  runtime appends a visible truncation notice to the final text, reports
  `finish_reason` on the per-call `usage` event, and — for a tool_call whose
  arguments JSON was cut mid-string — returns "arguments truncated by
  max_tokens, re-issue the call" instead of a bare "invalid JSON" error that
  invited a byte-identical retry.
- **Responses-transport 429 handling aligned with the chat path**: it now
  retries whenever attempts remain (previously `sleep_429=0` meant no retry
  at all) and honors the server's `Retry-After` header (seconds or
  HTTP-date) over the `sleep_429` base, in both the plain and streaming
  variants.
- **`AgentContext.shared`: per-run, cross-context state.** A new field on
  the kernel context, shared *by reference* with derived subagent contexts
  (as opposed to `extra`, host data copied per context). Two consumers
  moved onto it: the workspace stale-file guard's revision map (fixing
  parallel subagents clobbering a file the orchestrator/another subagent
  read — the guard now spans the whole run) and the run's cancellation
  handle (see next item). It is runtime-internal identity, not data: not
  part of the mapping surface, does not round-trip through `to_dict`.
- **cancellation propagates into subagents**: `AgentRuntime.run` stashes
  its `stop` handle in `ctx.shared["_runtime_stop"]`; `SubagentEngine.run`
  forwards it to the sub-runtime, so cancelling the orchestrating run ends
  an in-flight subagent between its model calls / tool dispatches instead
  of letting it run (and spend) to completion. Cancelled / budget-cut
  delegations report `ok=False` with a distinct summary header, and
  `cancelled` events now flow through the `subagent_progress` heartbeat.
- **subagent action high-water is id-membership based**: the "new since this
  delegation" snapshot compared action ids numerically, assuming the
  protocol's opaque/host-assigned ids were ints — a store using string ids
  got lexicographic comparison (silent miscounts). The snapshot is now a
  set of ids; membership is the only contract.
- **`run_code` / `run_file` re-classified as WRITE tools** (were READ):
  executing model-written code can mutate the sandboxed workspace, so the
  old classification let them run in read-only agent modes and in parallel
  with other tool calls — a write race the runtime's grouping exists to
  prevent, and a mode escape for read-only hosts. Behavior change: code
  execution is invisible in `anchored` mode and serialized against other
  writes.

- **runtime: `tool_call` events now precede execution.** Previously every
  `tool_call` display event was emitted only after the whole step's tools had
  finished (they were gathered, then announced) — a slow tool left the
  frontend blind to what was pending. The runtime now announces all of the
  step's `tool_call` events before dispatching anything; `tool_result`
  events, records, and message appends still follow in model order, so turn
  reconstruction and persistence are unchanged.
- **runtime: cancellation aborts an in-flight model stream.** `stop` was only
  checked before each model call; a cancellation arriving mid-generation
  still paid for the entire response. The streaming loop now checks `stop`
  between deltas and closes the transport stream explicitly (dropping the
  underlying HTTP response); the run ends with the usual `cancelled` event
  and no assistant turn is recorded, so memory replay has nothing dangling.
- **runtime: run budgets `max_cost` / `max_total_tokens`.** A runaway loop
  previously burned money until the step cap. Crossing either budget after a
  model call now cuts the run with `status="budget_exceeded"`: the step's
  tool calls are not executed and no further model calls are made; the run
  ends with the model's own text or a budget notice. A host seeding `stats`
  with prior-conversation totals budgets across runs (the pre-call guard
  blocks the very first call). A model that finishes naturally on the
  crossing step still reports `status="done"`. `AgentHost` forwards both
  knobs; subagent runs apply them per delegation.
- **runtime: identical-repeat guard `repeat_call_limit` (default 3).** A
  stuck model reissuing the exact same (tool, args) call — the classic
  "didn't understand the error, retry harder" loop — now gets an inline
  nudge appended to that tool's result ("this is call #N with identical
  arguments; change the args or the approach"), feeding self-correction
  without touching display events. `repeat_call_limit=None` disables.

- **Undo chain completed — StoreSink captures mutation events into stored
  actions**: `StoreSink.on_event` (previously a no-op) now maps the
  undo-bearing UI events tools emit into `store.log_action` rows —
  `file_change` → `file_write` / `file_edit` / `file_delete` (covering
  `write_file`, `edit_file` and all of `apply_patch`'s event shapes) and
  `todo_change` → `todo_replace`. This was the missing middle of the
  persisted-undo chain: tools emit the events and `undo_run` reads
  `store.list_actions`, but nothing connected them, so a default assembly
  could never undo across sessions. Now it does with zero host code —
  `AgentHost.run` attaches the sink, the runtime forwards each ui event, rows
  carry the emitting context's `subagent` tag (so subagent mutations revert
  with the orchestrator's run), and large old/new values spill to blobs per
  the store's own policy (8KB default, rehydrated transparently on read).
  A `file_change` event without a recognized `action` is skipped instead of
  writing a junk `file_None` row; `StoreSink(store, capture_actions=False)`
  opts out for hosts that persist actions themselves.
- **`lithe.bundles.patch`**: `apply_patch` — a line-oriented multi-file
  patch tool speaking the OpenAI Codex `*** Begin Patch` envelope (Add /
  Update (+ `*** Move to:`) / Delete File sections, `@@` chunks of context /
  `-` / `+` lines, `*** End of File` tail anchoring). Chunk location runs a
  four-pass fuzzy ladder (exact → ignore trailing whitespace → ignore
  leading/trailing → fold typographic Unicode punctuation), so a model's
  near-miss whitespace or smart-quote output still lands. Application is
  all-or-nothing: hunks are resolved and derived hunk-over-hunk against an
  in-memory overlay (a later section of the same file sees an earlier
  section's effect; a patch may create a file and then edit it) before the
  disk is touched once per path — a bad hunk leaves the workspace untouched.
  Emits the same `file_change` UI events as `write_file`/`edit_file` and
  adds one undo kind, `file_delete` (reverters from
  `register_apply_patch_tool` merge with `register_file_tools`' map); a move
  emits one write event for the destination plus one delete event for the
  source. Parsing is strict — an unrecognized non-blank line is an error, not
  a silent skip — with blank separator lines the one tolerated deviation.
  Patches wrapped in a `cat <<'EOF'` heredoc are unwrapped automatically.
  Design references: opencode's `patch/index.ts` (MIT) and OpenAI's
  `codex-rs/apply-patch` (Apache-2.0); scenario coverage ported from the
  latter's test suite.
- **workspace P0 reliability round**: `read_file` now prefixes every line
  with its 1-based number (`42: ...`, full reads and offset/limit windows
  alike) — the coordinate system models build unique `edit_file` old_text
  spans and `apply_patch` `@@` contexts from. `edit_file` rejects
  `old_text == new_text` (was a silent no-op write) and, when the exact
  substring match finds nothing, falls back to a whole-line fuzzy match via
  the shared `_textmatch` ladder (trailing whitespace / indentation /
  typographic punctuation tolerated, `N: ` prefixes copied from numbered
  read output stripped; multi-hit ambiguity and `replace_all` behave exactly
  like the exact path). All write tools (`write_file`/`edit_file`/
  `apply_patch`) gain an optimistic-concurrency guard: `read_file` records a
  per-run `(mtime_ns, size)` snapshot in `ctx.shared`, and a write to a path
  whose stat no longer matches its last-read snapshot is refused with
  "re-read the file" instead of silently clobbering an external change;
  successful writes refresh the snapshot so chained write→edit flows never
  false-alarm. The fuzzy comparator ladder lives in
  `lithe.bundles._textmatch`, shared by `apply_patch` seeking and
  `edit_file`'s fallback.

## 0.7.0

New optional `images` bundle (tool-tier image perception, zero kernel
changes) + robustness fixes (P0 round): run lifecycle under consumer
disconnects, history reconciliation, event-loop hygiene, parallel-delegation
isolation.

- **`lithe.bundles.images`**: `image_info` — a stdlib-only
  PNG/JPEG/GIF/BMP/WEBP header probe (dimensions, dpi where the container
  carries it, color type) that answers deterministic questions with no model
  call — and `analyze_image`, a single-shot VLM request (OpenAI `image_url`
  data-URL content block + question) whose text answer is all the main
  conversation ever sees; results memoized per (file hash, question, detail)
  in-process. Deliberately not a subagent: one `chat_completion` round-trip
  instead of a full ReAct runtime. The perception model is whatever
  `LLMConfig` the host passes to `register_image_tools` — the main-loop
  config reuses the main model; a dedicated config routes vision to another
  endpoint; omitting it registers only the deterministic probe.
- **`AgentHost.run` closes abandoned runs**: a consumer that stops iterating
  mid-stream (disconnecting SSE client / cancelled task — `CancelledError` /
  `GeneratorExit`, both `BaseException`) previously bypassed the
  `except Exception` funnel, leaving the run `running` in the store forever.
  It is now closed with `status="abandoned"` (store + host `stats` dict)
  before the cancellation re-raises; a disconnect at the final `done` yield
  no longer rewrites an already-recorded terminal status.
- **`assemble_messages` reconciles orphan tool calls** (two-sided, like
  `replay_messages`): an assistant `tool_call` with no matching tool result —
  e.g. a run cancelled between the model's call and its execution — and a
  tool result whose call is gone are both dropped, so a cancelled turn can no
  longer poison the next request with an API-rejected payload. Legacy JSON
  string `tool_calls` in history rows are parsed first.
- **`load_skill` never blocks the event loop**: `register_skill_tool`'s
  handler offloads library calls (`index_text`/`resolve`/`load`) to a worker
  thread via `asyncio.to_thread` — a `SkillPackages` over `RemoteSkillSource`
  does sync network + filesystem refresh work that previously froze every
  concurrent agent run in the process for up to `(1+N) × timeout`.
  `RemoteSkillSource` TTL now uses `time.monotonic()` (immune to wall-clock
  adjustments); hosts calling the library directly (prompt builders) should
  offload the same way (documented).
- **`delegate_parallel` isolates crashed delegates**: `gather(...,
  return_exceptions=True)` — a delegate that raises a real exception (store
  failure, bug) is reported as that agent's failure block instead of
  aborting the whole step and leaving sibling subagents running as unawaited
  orphans.

## 0.6.0

Orchestration & undo depth, skills relocated, polish. **Breaking:**
`UndoEngine.undo` and `host.undo_run` are now `async` (awaiting async
reverters), and `SkillPackages`/`RemoteSkillSource` are exported from
`lithe.bundles` instead of the `lithe` root (core stays zero-I/O;
`import lithe.skills` still works as a deprecated alias).

### Subagents
- **Parallel delegation**: `delegate_parallel` tool (via `register_delegate_tools`
  or `make_parallel_delegate_tool`) fans independent tasks out to several
  subagents concurrently (`max_parallel` semaphore, `max_batch` cap); one
  failing subagent doesn't abort the others — per-agent blocks + overall ok.
- **Live progress**: `SubagentEngine(on_subagent_event=...)` forwards thin,
  tagged `subagent_progress` events (step/tool/assistant/error, text capped)
  to a host push channel; without it subagent events stay recorded-only.
- `SubagentSpec(transport=...)` routes a subagent to a different endpoint or
  wire protocol.

### Undo
- Reverters may be sync **or** async; `UndoEngine.undo` awaits whatever the
  reverter returns (`undo_run` is `async` now).

### Skills
- `SkillPackages` / `RemoteSkillSource` moved into `lithe.bundles.skills`
  (merged with `SkillLibrary`, shared frontmatter/description helpers);
  `import lithe` no longer pulls any bundle.

### Workspace / runtime polish
- `read_file` gained `offset`/`limit` line-window reads (paginate long files
  instead of head-only truncation).
- Context-fullness accounting is incremental (running char total maintained
  across steps; `_trim_context` returns the post-trim size) — no per-step
  rescan of the whole conversation.

## 0.5.0

Kernel correctness, controllability, observability and packaging — everything
since 0.4.6.

### Core runtime
- **Cancellation**: `AgentRuntime.run(..., stop=Event|callable)` → `cancelled`
  event, `status="cancelled"`; `AgentHost.run` forwards `stop=`.
- **Streaming**: `LLMConfig(stream=True)` emits `assistant_delta` events while
  the model generates (chat-completions and Responses transports; unsupported
  transports fall back silently). The full `assistant` event still follows.
- **Tool argument validation feedback**: malformed `arguments` JSON and schema
  violations (missing required / wrong declared types) come back to the model
  as failed tool results instead of executing with empty/wrong args.
  `ToolSpec(validate=False)` opts out.
- **Tool execution policy**: model order preserved; consecutive READ tools run
  in parallel, every WRITE/META tool runs alone (no write races);
  `ToolSpec(timeout=...)` cancels hung calls.
- **Max-step wrap-up**: the last step of a multi-step budget is a forced
  toolless call (`tool_choice="none"`) producing a real summary;
  `status="max_steps"` marks truncation honestly.
- **Mid-run context budget**: `context_budget` (default 400k chars) shrinks
  old tool results head+tail so long runs don't blow the context window;
  `None` disables.
- **Sink error isolation**: broken sinks log instead of killing the run;
  `strict_records=True` makes record failures fatal.
- **Usage / context observability**: per-call `usage` events
  (prompt/completion/total tokens, cost, `context_tokens`, `context_chars`,
  `context_window`, `context_percent` via `LLMConfig(context_window=...)`);
  `RunStats` and the host `done` event carry the cumulative breakdown.
  Transports normalize usage across chat-completions and Responses shapes.
- **LLMConfig**: `temperature` / `max_tokens` forwarded on every call;
  `SubagentEngine` inherits all policy fields via `dataclasses.replace`.

### LLM client
- Retry classification: fatal 4xx fail after one request (fallback still
  honored); 429 respects `Retry-After` (seconds or HTTP-date); malformed 200s
  stay retryable. Streaming follows the same policy.

### Tools & modes
- `ToolRegistry.unregister(name)`; `registry.add_middleware(...)` for audit /
  quota / human-in-the-loop confirmation of write tools.
- `register_mode(name, categories)` for host-defined modes; unknown modes now
  raise `ValueError` instead of silently degrading to read-only.

### Bundles
- `workspace`: new `search_files` (regex grep with dir/glob filters) and
  `glob_files` tools; `edit_file` rejects ambiguous `old_text` with line
  hints and supports explicit `replace_all`; edits return a compact diff.
- `sandbox`: stdout/stderr truncation keeps head+tail so tracebacks stay
  visible.
- `mcp`: JSON-RPC id-space collision fixed (server-initiated requests are
  answered with method-not-found, never matched against pending futures);
  dead sessions lazily self-heal on the next tool call (`MCPManager.revive`).
- `host`: `undo_run(..., extra=...)` seeds the reverters' context;
  `AgentHost(sinks=[...])` attaches extra EventSinks (metrics, action capture,
  frontend fan-out) that observe the full run lifecycle including
  `run_start`/`done`.
- `store` (JsonlRunStore): id counters are in-memory (no per-append rescan);
  action values larger than `spill_threshold` (default 8KB) externalize to
  content-addressed blobs and rehydrate transparently on read.

### Packaging
- `py.typed` (PEP 561); version single-sourced from `lithe.__version__`
  (`import` it, or read `lithe.__version__`); ruff target py310 with
  B/UP/ASYNC rules; CHANGELOG, CI workflow and a runnable example.

## 0.4.x

Initial extraction rounds: storage-free kernel (runtime/llm/tools/actions/
memory/events/context/modes/transports) + optional bundles (host, store,
subagents, admin, workspace, sandbox, skills, mcp, todos).
