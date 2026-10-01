"""MCP bundle: bridge external MCP (Model Context Protocol) servers into a
lithe ``ToolRegistry``.

A host declares servers, the manager keeps each server session alive across
registry rebuilds (registries are cheap and per-turn in most hosts; an MCP
subprocess / HTTP session is not), and ``attach`` translates the server's
``tools/list`` into ``ToolSpec`` entries whose handlers call ``tools/call``.
This gives any lithe host one-call access to the growing ecosystem of MCP
servers (vision analysis, web search, databases, ...).

Two transports, chosen per server by config:
  - stdio (``command`` given): spawn a local subprocess and speak
    newline-delimited JSON-RPC 2.0 over its stdin/stdout. Self-contained
    client — stdlib only.
  - streamable-http (``url`` given): POST JSON-RPC to a remote MCP endpoint;
    responses come back as ``application/json`` or an SSE stream
    (``text/event-stream``). The server-assigned ``Mcp-Session-Id`` header
    (returned on initialize, when the server uses sessions) is echoed on
    subsequent requests and DELETEd on close. Auth headers (``headers``)
    stay host-side and are never logged.

Only the three calls a tool bridge needs are implemented (``initialize``
handshake, ``tools/list``, ``tools/call``).

Mapping rules:
  - tool name      -> ``{prefix}{tool}`` (default prefix ``{server}__``) so
                      servers never collide with host tools or each other;
                      colliding names are skipped with a warning, never raised.
  - inputSchema    -> ``ToolSpec.parameters`` verbatim (both JSON Schema).
  - annotations    -> ``readOnlyHint`` maps to ``ToolCategory.READ``; tools
                      without it fall back to ``default_category`` (WRITE by
                      default — fail-safe for read-only agent modes).
  - result content -> text items joined; ``isError`` maps to ``ok=False``.
  - undo           -> MCP tools carry no reverters (not undoable).

Optional bundle — the core engine never imports this.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from lithe import __version__
from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec

log = logging.getLogger("lithe.mcp")

DEFAULT_PROTOCOL_VERSION = "2025-06-18"
_CLIENT_INFO = {"name": "lithe-mcp", "version": __version__}

# Parent-process vars safe to forward to a spawned MCP server when
# ``inherit_env=None`` (the default): pure runtime config, never secrets —
# the same allow-list policy the sandbox bundle uses. ``inherit_env=True``
# forwards all of os.environ (legacy behavior); ``False`` forwards none.
_SAFE_ENV_NAMES = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "HOME", "TMPDIR")
_WINDOWS_SAFE_ENV_NAMES = (
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
    "OS", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)


@dataclass
class MCPServerConfig:
    """One MCP server declaration: stdio (``command``) or streamable-http
    (``url``); exactly one of the two must be set.

    stdio: ``command`` is argv, e.g. ``["npx", "-y", "@z_ai/mcp-server"]``;
    ``env`` carries server secrets (API keys) and is merged over
    ``os.environ`` for the subprocess only. http: ``url`` is the MCP endpoint
    (e.g. ``https://open.bigmodel.cn/api/mcp/web_search_prime/mcp``) and
    ``headers`` carries auth (``{"Authorization": "Bearer ..."}``) plus any
    extra headers; both are sent verbatim and never logged.

    ``tool_allowlist``/``tool_blocklist`` filter by the server-side
    (unprefixed) tool name; ``None`` allowlist means all tools.
    """

    name: str
    command: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    protocol_version: str = DEFAULT_PROTOCOL_VERSION
    timeout: float = 60.0
    startup_timeout: float = 30.0
    prefix: str | None = None
    tool_allowlist: frozenset[str] | None = None
    tool_blocklist: frozenset[str] = frozenset()
    default_category: ToolCategory = ToolCategory.WRITE
    # None: forward only the safe allow-list (PATH/locale/HOME/…) plus `env`;
    # True: forward all of os.environ plus `env` (legacy behavior); False:
    # forward only `env`. Secrets in os.environ never reach the subprocess
    # unless a host explicitly opts in with True.
    inherit_env: bool | None = None

    def tool_prefix(self) -> str:
        return self.prefix if self.prefix is not None else f"{self.name}__"

    def __post_init__(self) -> None:
        if isinstance(self.command, str):
            self.command = self.command.split()
        if bool(self.command) == bool(self.url):
            raise ValueError(
                f"MCP 服务器 {self.name} 需要且仅需要 command(stdio) 或 url(http) 之一")

    @classmethod
    def from_dict(cls, data: dict) -> MCPServerConfig:
        """Build from a plain dict (env/config-file friendly).

        Required: ``name`` + (``command`` xor ``url``). A string ``command``
        is split on spaces. ``tool_allowlist``/``tool_blocklist`` accept
        lists; ``default_category`` accepts the enum value string
        ("read"/"write"/"meta").
        """
        command = data.get("command")
        if isinstance(command, str):
            command = command.split()
        allow = data.get("tool_allowlist")
        category = data.get("default_category", ToolCategory.WRITE)
        inherit_env = data.get("inherit_env")
        return cls(
            name=data["name"],
            command=list(command) if command else None,
            env=dict(data.get("env") or {}),
            cwd=data.get("cwd"),
            url=data.get("url"),
            headers=dict(data.get("headers") or {}),
            protocol_version=data.get("protocol_version", DEFAULT_PROTOCOL_VERSION),
            timeout=float(data.get("timeout", 60.0)),
            startup_timeout=float(data.get("startup_timeout", 30.0)),
            prefix=data.get("prefix"),
            tool_allowlist=frozenset(allow) if allow is not None else None,
            tool_blocklist=frozenset(data.get("tool_blocklist") or ()),
            default_category=ToolCategory(category),
            inherit_env=None if inherit_env is None else bool(inherit_env),
        )


def parse_servers(spec: str | list | dict) -> list[MCPServerConfig]:
    """Parse server declarations from JSON text (list of specs, or a
    ``{name: spec}`` mapping) into configs. Unknown shapes raise ValueError."""
    if isinstance(spec, str):
        spec = json.loads(spec) if spec.strip() else []
    if isinstance(spec, dict):
        for key, value in spec.items():
            if not isinstance(value, dict):
                # Never embed the raw value: a config mistake then leaks the
                # whole server spec (url with credentials, env with API keys)
                # into an exception that ends up in logs.
                raise ValueError(
                    f"MCP_SERVERS 对象形式的值必须是配置对象：{key} -> "
                    f"{type(value).__name__}")
        spec = [{**v, "name": k} for k, v in spec.items()]
    if not isinstance(spec, list):
        raise ValueError(f"MCP_SERVERS 需为 JSON 数组或对象，得到 {type(spec).__name__}")
    return [MCPServerConfig.from_dict(item) for item in spec]


def _log_host(url: str | None) -> str:
    """Just scheme://host[:port] for logs — the URL may carry credentials
    (basic-auth userinfo, query-string API keys) that must never reach a
    log line."""
    if not url:
        return "?"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "?"
    host = parts.hostname
    if not host:
        return url[:32]
    if parts.port:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}" if parts.scheme else host


async def _list_all_tools(session, *, page_timeout: float) -> list[dict]:
    """Collect a server's full tool list, following ``nextCursor`` pages.

    The MCP ``tools/list`` result may be paginated; a server with more tools
    than its page size silently returned a truncated list when the cursor was
    ignored. A defensive page cap stops a buggy server from looping forever.
    """
    tools: list[dict] = []
    cursor: str | None = None
    for _ in range(64):
        params = {"cursor": cursor} if cursor else None
        result = await session.call("tools/list", params,
                                    timeout=page_timeout)
        tools.extend(result.get("tools") or [])
        cursor = result.get("nextCursor")
        if not cursor:
            break
    else:
        log.warning("tools/list exceeded 64 pages; list may be incomplete")
    return tools


def _parse_sse_messages(body: str) -> list[dict]:
    """Parse an SSE stream body into JSON messages (data fields only)."""
    messages: list[dict] = []
    data_lines: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            if data_lines:
                text = "\n".join(data_lines)
                try:
                    messages.append(json.loads(text))
                except ValueError:
                    pass
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
    if data_lines:
        try:
            messages.append(json.loads("\n".join(data_lines)))
        except ValueError:
            pass
    return messages


class _StdioSession:
    """One live MCP server subprocess with request/response correlation."""

    def __init__(self, cfg: MCPServerConfig):
        self.cfg = cfg
        self.proc: asyncio.subprocess.Process | None = None
        self.alive = False
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        # Bounded: a chatty server must not grow an unbounded list for the
        # process lifetime; the tail is only used for startup-failure reports.
        self._stderr_tail: deque[str] = deque(maxlen=64)

    async def start(self) -> None:
        env = self._spawn_env()
        process_options = (
            {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        self.proc = await asyncio.create_subprocess_exec(
            *self.cfg.command, env=env, cwd=self.cfg.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_options,
        )
        self._reader = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            result = await self._request(
                "initialize",
                {"protocolVersion": self.cfg.protocol_version,
                 "capabilities": {}, "clientInfo": _CLIENT_INFO},
                timeout=self.cfg.startup_timeout)
            await self._notify("notifications/initialized")
        except Exception as exc:
            await self.close()
            tail = "\n".join(list(self._stderr_tail)[-5:])
            raise RuntimeError(
                f"MCP 服务器 {self.cfg.name} 启动握手失败：{exc}\n{tail}") from exc
        self.alive = True
        log.info("MCP server %s started (pid %s, protocol %s)",
                 self.cfg.name, self.proc.pid, result.get("protocolVersion"))

    def _spawn_env(self, *, windows: bool | None = None) -> dict[str, str]:
        """Build a secret-conscious child environment with required OS runtime vars."""
        is_windows = os.name == "nt" if windows is None else windows
        if self.cfg.inherit_env is True:
            base = dict(os.environ)
        elif self.cfg.inherit_env is False:
            base = {}
        else:
            safe_names = _SAFE_ENV_NAMES + (
                _WINDOWS_SAFE_ENV_NAMES if is_windows else ()
            )
            base = {k: v for k, v in os.environ.items()
                    if k.upper() in safe_names}
        if is_windows and self.cfg.inherit_env is not True:
            base.update(
                (key, value)
                for key, value in os.environ.items()
                if key.upper() in _WINDOWS_SAFE_ENV_NAMES
            )
        return {**base, **self.cfg.env}

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except ValueError:
                    log.debug("[%s] non-JSON stdout line ignored", self.cfg.name)
                    continue
                if not isinstance(msg, dict):
                    # A bare JSON scalar/line (some servers print arrays or
                    # banner strings on stdout) must not kill the reader with
                    # an AttributeError — taking the whole session down.
                    log.debug("[%s] non-object stdout line ignored", self.cfg.name)
                    continue
                if "method" in msg and "result" not in msg and "error" not in msg:
                    # Server-initiated message (a request carrying its own id,
                    # or a notification). JSON-RPC lets client and server id
                    # spaces collide, so this must NEVER be matched against
                    # ``_pending`` — resolving our future with it corrupts
                    # request/response correlation. Requests are answered with
                    # a method-not-found error so the server is not left
                    # hanging; notifications are just logged.
                    if msg.get("id") is not None:
                        log.debug("[%s] server request %s unsupported",
                                  self.cfg.name, msg["method"])
                        await self._respond_error(
                            msg["id"], -32601,
                            f"Method not found: {msg.get('method')}")
                    else:
                        log.debug("[%s] notification %s",
                                  self.cfg.name, msg["method"])
                    continue
                if msg.get("id") is not None:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
        finally:
            self.alive = False
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_result({"error": {"message": "MCP 服务器连接已断开"}})
            self._pending.clear()

    async def _respond_error(self, req_id, code: int, message: str) -> None:
        """Answer a server-initiated request with a JSON-RPC error reply."""
        if not self.proc or not self.proc.stdin or self.proc.stdin.is_closing():
            return
        payload = {"jsonrpc": "2.0", "id": req_id,
                   "error": {"code": code, "message": message}}
        try:
            self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            await self.proc.stdin.drain()
        except Exception:  # noqa: BLE001 — best effort, reply must not kill the loop
            pass

    async def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            self._stderr_tail.append(line.decode("utf-8", errors="replace").strip())

    async def _notify(self, method: str, params: dict | None = None) -> None:
        assert self.proc and self.proc.stdin
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()

    async def _request(self, method: str, params: dict | None = None,
                       *, timeout: float) -> dict:
        if (not self.proc or self.proc.stdin.is_closing()
                or not self.alive and method != "initialize"):
            raise RuntimeError(f"MCP 服务器 {self.cfg.name} 未连接")
        self._next_id += 1
        req_id = self._next_id
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()
        try:
            msg = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            # Tell the server we gave up — but bounded: a server that stopped
            # reading its stdin has a full pipe, and an unbounded drain here
            # would hang the timeout path we are already trying to escape.
            try:
                await asyncio.wait_for(
                    self._notify("notifications/cancelled",
                                 {"requestId": req_id}),
                    timeout=1.0)
            except Exception:  # noqa: BLE001 — best-effort notice only
                log.debug("[%s] sending notifications/cancelled failed",
                          self.cfg.name, exc_info=True)
            raise
        if "error" in msg:
            raise RuntimeError(f"{method} 错误：{msg['error'].get('message')}")
        return msg.get("result") or {}

    async def call(self, method: str, params: dict | None = None,
                   *, timeout: float | None = None) -> dict:
        return await self._request(
            method, params, timeout=timeout if timeout is not None else self.cfg.timeout)

    async def close(self, grace: float = 5.0) -> None:
        """Shut the server down. MCP stdio convention first: closing stdin asks
        the server to exit on its own (node/npx wrappers die cleanly this way);
        after ``grace`` seconds escalate to SIGKILL on the whole process group
        (``start_new_session`` puts the server in its own group, so wrapper
        children like the node process npx forks are not orphaned)."""
        self.alive = False
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                if proc.stdin and not proc.stdin.is_closing():
                    proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace)
            except asyncio.TimeoutError:
                await self._kill_group(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        log.warning("MCP server %s did not exit after forced termination",
                                    self.cfg.name)
        for task in (self._reader, self._stderr_task):
            if task:
                task.cancel()
        self._reader = None
        self._stderr_task = None

    async def _kill_group(self, proc: asyncio.subprocess.Process) -> None:
        if os.name == "nt":
            taskkill = shutil.which("taskkill")
            if taskkill:
                try:
                    killer = await asyncio.create_subprocess_exec(
                        taskkill, "/PID", str(proc.pid), "/T", "/F",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.wait_for(killer.wait(), timeout=3)
                    if killer.returncode == 0:
                        return
                except (OSError, asyncio.TimeoutError):
                    pass
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            proc.kill()
        except ProcessLookupError:
            pass


class _HTTPSession:
    """One streamable-http MCP session (remote endpoint, optional server
    session id). Holds a persistent ``httpx.AsyncClient``; ``alive`` flips off
    on transport errors so the manager restarts on the next attach."""

    def __init__(self, cfg: MCPServerConfig):
        self.cfg = cfg
        self.proc = None
        self.alive = False
        self._client: httpx.AsyncClient | None = None
        self._next_id = 0
        self._session_id: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            **self.cfg.headers,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _post(self, payload: dict, *, timeout: float) -> httpx.Response:
        assert self._client
        return await self._client.post(
            self.cfg.url or "", json=payload, headers=self._headers(),
            timeout=timeout)

    def _decode_response(self, resp: httpx.Response, method: str,
                         req_id: int) -> dict:
        if resp.status_code >= 400:
            self.alive = False
            raise RuntimeError(
                f"{method} HTTP {resp.status_code}：{resp.text[:200]}")
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            messages = _parse_sse_messages(resp.text)
            msg = next((m for m in messages if m.get("id") == req_id
                        and ("result" in m or "error" in m)), None)
            if msg is None:
                raise RuntimeError(f"{method} 响应流中没有对应请求 {req_id} 的消息")
        else:
            try:
                msg = json.loads(resp.text) if resp.text.strip() else None
            except ValueError:
                raise RuntimeError(
                    f"{method} 响应不是合法 JSON：{resp.text[:200]}") from None
            if msg is None:
                raise RuntimeError(f"{method} 收到空响应")
        if "error" in msg:
            raise RuntimeError(f"{method} 错误：{msg['error'].get('message')}")
        return msg.get("result") or {}

    async def _request(self, method: str, params: dict | None = None,
                       *, timeout: float) -> dict:
        self._next_id += 1
        req_id = self._next_id
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        try:
            resp = await self._post(payload, timeout=timeout)
        except httpx.HTTPError as exc:
            self.alive = False
            raise RuntimeError(f"MCP 服务器 {self.cfg.name} 请求失败：{exc}") from exc
        return self._decode_response(resp, method, req_id)

    async def _notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        try:
            resp = await self._post(payload, timeout=self.cfg.timeout)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"MCP 服务器 {self.cfg.name} 通知失败：{exc}") from exc
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{method} 通知 HTTP {resp.status_code}：{resp.text[:200]}")

    async def start(self) -> None:
        self._client = httpx.AsyncClient()
        try:
            self._next_id += 1
            req_id = self._next_id
            payload = {"jsonrpc": "2.0", "id": req_id, "method": "initialize",
                       "params": {"protocolVersion": self.cfg.protocol_version,
                                  "capabilities": {}, "clientInfo": _CLIENT_INFO}}
            try:
                resp = await self._post(payload, timeout=self.cfg.startup_timeout)
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"MCP 服务器 {self.cfg.name} 请求失败：{exc}") from exc
            self._session_id = resp.headers.get("Mcp-Session-Id")
            result = self._decode_response(resp, "initialize", req_id)
            await self._notify("notifications/initialized")
        except Exception as exc:
            await self.close()
            raise RuntimeError(
                f"MCP 服务器 {self.cfg.name} 启动握手失败：{exc}") from exc
        self.alive = True
        log.info("MCP server %s connected (host %s, protocol %s, session %s)",
                 self.cfg.name, _log_host(self.cfg.url),
                 result.get("protocolVersion"),
                 "yes" if self._session_id else "no")

    async def call(self, method: str, params: dict | None = None,
                   *, timeout: float | None = None) -> dict:
        return await self._request(
            method, params, timeout=timeout if timeout is not None else self.cfg.timeout)

    async def close(self) -> None:
        self.alive = False
        if self._session_id and self._client:
            try:
                await self._client.delete(
                    self.cfg.url or "",
                    headers={**self.cfg.headers, "Mcp-Session-Id": self._session_id},
                    timeout=5.0)
            except httpx.HTTPError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None


class MCPManager:
    """Keeps MCP server sessions alive across registry rebuilds.

    Hosts typically build a fresh ``ToolRegistry`` per turn; spawning an MCP
    subprocess (npx cold start can take seconds) each turn would dominate
    latency. The manager owns the long-lived sessions keyed by server name and
    re-attaches specs to any fresh registry. Failed servers degrade gracefully:
    ``attach`` logs and reports them, never raises.
    """

    def __init__(self, configs: list[MCPServerConfig]):
        by_name: dict[str, MCPServerConfig] = {}
        for cfg in configs:
            if cfg.name in by_name:
                raise ValueError(f"duplicate MCP server name: {cfg.name}")
            by_name[cfg.name] = cfg
        self.configs = by_name
        self._sessions: dict[str, _StdioSession | _HTTPSession] = {}
        # Serializes session (re)spawns across _sync_registry and revive:
        # MCP read tools run in parallel, so two dispatches to a dead server
        # would otherwise both start a subprocess and leak the loser (its
        # process and reader tasks would never be closed).
        self._spawn_lock = asyncio.Lock()

    def _build_session(self, cfg: MCPServerConfig) -> _StdioSession | _HTTPSession:
        return _HTTPSession(cfg) if cfg.url else _StdioSession(cfg)

    async def attach(self, registry: ToolRegistry) -> dict[str, list[str] | str]:
        """Start (or reuse) every server session and register its tools.

        Returns a report ``{server: [tool names]}``; failed servers map to an
        error string instead of a list. Collisions with already-registered
        names are skipped with a warning.
        """
        return await self._sync_registry(registry, warn_collision=True)

    async def ensure(self, registry: ToolRegistry) -> dict[str, list[str] | str]:
        """Idempotent variant of :meth:`attach` for hosts that cache their
        registry across runs: only (re)starts dead sessions and registers
        tools whose prefixed names are not already in ``registry``. Safe to
        call before every agent run."""
        return await self._sync_registry(registry, warn_collision=False)

    async def _sync_registry(self, registry: ToolRegistry,
                             *, warn_collision: bool) -> dict[str, list[str] | str]:
        report: dict[str, list[str] | str] = {}
        taken = set(registry.names())
        for name, cfg in self.configs.items():
            session = self._sessions.get(name)
            if session is None or not session.alive:
                async with self._spawn_lock:
                    session = self._sessions.get(name)  # re-check: a
                    # concurrent revive/_sync may have spawned it already
                    if session is None or not session.alive:
                        if session is not None:
                            await session.close()
                        session = self._build_session(cfg)
                        try:
                            await session.start()
                        except Exception as exc:
                            log.warning("MCP server %s failed to start: %s",
                                        name, exc)
                            self._sessions.pop(name, None)
                            report[name] = f"启动失败：{exc}"
                            continue
                        self._sessions[name] = session
            try:
                tools = await _list_all_tools(session, page_timeout=cfg.timeout)
            except Exception as exc:
                log.warning("MCP server %s tools/list failed: %s", name, exc)
                report[name] = f"工具列表获取失败：{exc}"
                continue
            registered: list[str] = []
            for tool in tools:
                raw = tool.get("name", "")
                if not raw:
                    continue
                if cfg.tool_allowlist is not None and raw not in cfg.tool_allowlist:
                    continue
                if raw in cfg.tool_blocklist:
                    continue
                full = f"{cfg.tool_prefix()}{raw}"
                if full in taken:
                    if warn_collision:
                        log.warning("MCP tool %s collides with an existing tool; skipped",
                                    full)
                    continue
                taken.add(full)
                registry.register(self._make_spec(cfg, tool, full),
                                  self._make_handler(cfg, raw))
                registered.append(full)
            report[name] = registered
        return report

    def _make_spec(self, cfg: MCPServerConfig, tool: dict, full_name: str) -> ToolSpec:
        annotations = tool.get("annotations") or {}
        if annotations.get("readOnlyHint"):
            category = ToolCategory.READ
        else:
            category = cfg.default_category
        description = (tool.get("description") or
                       f"MCP tool {tool.get('name')} from {cfg.name}")
        source = f"[{cfg.name}] " if not description.startswith("[") else ""
        return ToolSpec(full_name, source + description,
                        tool.get("inputSchema")
                        or {"type": "object", "properties": {}},
                        category)

    async def revive(self, name: str) -> bool:
        """Best-effort lazy restart of one dead server session.

        Handlers resolve their session by name at call time, so cached
        registry entries start working again the moment the session is back —
        no re-attach needed. Returns True when a live session exists
        afterwards; a failed restart is logged, never raised.
        """
        cfg = self.configs.get(name)
        if cfg is None:
            return False
        async with self._spawn_lock:
            session = self._sessions.get(name)
            if session is not None and session.alive:
                return True
            if session is not None:
                await session.close()
            session = self._build_session(cfg)
            try:
                await session.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("MCP server %s lazy restart failed: %s", name, exc)
                self._sessions.pop(name, None)
                return False
            self._sessions[name] = session
            return True

    def _make_handler(self, cfg: MCPServerConfig, raw_name: str):
        async def handler(ctx, args):
            live = self._sessions.get(cfg.name)
            if live is None or not live.alive:
                # Lazy self-heal: hosts cache registries across turns while the
                # subprocess may have died in between; restart once instead of
                # failing every call until the next external ensure()/attach().
                if not await self.revive(cfg.name):
                    return ToolResult(False, "MCP 服务器已断开",
                                      f"MCP 服务器 {cfg.name} 连接已断开，工具 {raw_name} 不可用。")
                live = self._sessions[cfg.name]
            try:
                result = await live.call(
                    "tools/call", {"name": raw_name, "arguments": args},
                    timeout=cfg.timeout)
            except asyncio.TimeoutError:
                return ToolResult(False, "MCP 调用超时",
                                  f"工具 {raw_name} 超过 {cfg.timeout:.0f}s 未返回。")
            except Exception as exc:  # noqa: BLE001
                return ToolResult(False, "MCP 调用失败",
                                  f"{cfg.name}·{raw_name} 调用异常：{exc}")
            content = result.get("content") or []
            text = "\n".join(item.get("text", "") for item in content
                             if item.get("type") == "text")
            ok = not result.get("isError", False)
            return ToolResult(
                ok=ok,
                summary=f"{cfg.name}·{raw_name}" + ("" if ok else "（失败）"),
                content=text if text else ("调用成功" if ok else "调用失败"))

        return handler

    async def close(self) -> None:
        for session in self._sessions.values():
            await session.close()
        self._sessions.clear()

    def status(self) -> dict[str, dict]:
        """Live-ness snapshot: ``{server: {"alive": bool, "pid": int|None}}``."""
        return {
            name: {"alive": s.alive, "pid": s.proc.pid if s.proc else None}
            for name, s in self._sessions.items()
        }
