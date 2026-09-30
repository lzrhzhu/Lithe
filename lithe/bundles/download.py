"""Download bundle: fetch remote content into the workspace.

``download_file(url, path?)`` streams one HTTP(S) resource into the agent's
workspace (default under ``downloads/``) — the controlled ingress the
network-isolated sandbox deliberately lacks: *fetching* is a kernel-side,
bounded, audited tool, while *executing* model code stays offline. A host
binds it to the same ``Workspace`` the file/code tools use, so the flow
"search (MCP) → download → read/parse" closes without ever handing the
model a shell.

Hard boundaries (per call, host-tunable via ``register_download_tools``):

- scheme allow-list http/https; redirects are followed manually and **every
  hop** is re-validated;
- SSRF: every resolved address of every hop must be globally routable
  (``ipaddress`` ``is_global`` rejects loopback / private / link-local /
  CGN / reserved / multicast — cloud metadata ``169.254.169.254`` included).
  Checking all A/AAAA records up front narrows DNS rebinding to the TTL
  window; it cannot eliminate it (a name may re-resolve between check and
  connect). Environments that egress through an HTTP proxy should note the
  proxy, not this check, decides the real destination;
- size cap: ``Content-Length`` pre-check where given, plus a streaming
  accumulation cutoff — a lying header still gets cut off mid-stream and
  the partial ``.part`` file is removed;
- no overwrites: an existing target is an error (the model picks a new
  name), so the tool has no side effect to undo and needs no reverter;
- total-time budget wraps the whole fetch, with the ``ToolSpec`` timeout as
  the kernel-level backstop that cancels a wedged handler.

WRITE-classified like the other workspace-mutating tools: serialized against
concurrent write tools, excluded from read-only agent modes.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import socket
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from lithe import __version__
from lithe.context import AgentContext
from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec
from lithe.bundles.workspace import Workspace, _check_run_growth, _note_created

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_REDIRECTS = 5

_HEADERS = {"User-Agent": f"lithe-download/{__version__}"}

# Content types worth mapping to an extension when the URL basename carries
# none — enough to keep read_file/run_code from staring at "download_1".
_EXT_BY_TYPE = {
    "application/pdf": ".pdf",
    "application/json": ".json",
    "application/zip": ".zip",
    "application/x-tar": ".tar",
    "application/gzip": ".gz",
    "text/html": ".html",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/markdown": ".md",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
    "audio/mpeg": ".mp3",
    "video/mp4": ".mp4",
}


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}GB"


def _safe_name(name: str, *, max_len: int = 96) -> str:
    """URL basename → workspace-safe file name (no separators, no traversal)."""
    base = Path(name or "").name.strip()
    cleaned = "".join(ch if ch.isprintable() and ch not in '/\\:*?"<>|' else "_"
                      for ch in base).strip("._") or "download"
    return cleaned[:max_len]


def _check_host(host: str) -> None:
    """Raise ``ValueError`` unless *host* resolves to globally routable
    addresses only. Callable standalone so hosts (and tests) can reuse the
    exact policy the tool enforces."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"主机名解析失败：{host}") from exc
    import ipaddress
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if not addr.is_global or addr.is_multicast:
            raise ValueError(
                f"目标地址 {addr} 非公网地址，已拒绝（SSRF 防护）")


def _guess_target(ws: Workspace, url: str, path: str | None,
                  content_type: str | None) -> Path:
    """Decide the on-disk target: explicit path wins; otherwise
    ``downloads/<basename>`` with an extension inferred from Content-Type."""
    rel = (path or "").strip()
    if not rel:
        name = _safe_name(Path(urlsplit(url).path).name)
        base = Path(name)
        if not base.suffix:
            ext = _EXT_BY_TYPE.get((content_type or "").split(";")[0].strip().lower())
            if ext:
                name = base.with_suffix(ext).name
        rel = f"downloads/{name}"
    # safe_path raises PermissionError on traversal — surfaced as a failed
    # tool result by the handler.
    return ws.safe_path(rel)


async def download_to_workspace(
    ws: Workspace, url: str, path: str | None, *,
    max_bytes: int, timeout: float, max_redirects: int,
) -> tuple[Path, int, str | None, str]:
    """Fetch *url* into *ws*; returns ``(target, size, content_type, sha256)``.

    Raises ``ValueError`` on policy violations (scheme/SSRF/size/redirects/
    exists) and lets httpx errors/``asyncio.TimeoutError`` propagate to the
    handler's generic branch.
    """
    async def _validate_hop(u: str) -> httpx.URL:
        parsed = httpx.URL(u)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"仅支持 http/https，得到 {parsed.scheme}://")
        host = parsed.host
        if not host:
            raise ValueError("URL 缺少主机名")
        # DNS off the event loop: a slow resolver must not stall the run.
        await asyncio.to_thread(_check_host, host)
        return parsed

    async with httpx.AsyncClient(
            headers=_HEADERS, follow_redirects=False,
            timeout=httpx.Timeout(15.0, read=60.0, write=15.0, pool=15.0),
    ) as client:
        current = await _validate_hop(url)
        resp: httpx.Response | None = None
        for _ in range(max_redirects + 1):
            # stream=True: headers arrive, the body stays unread — the
            # Content-Length pre-check must run before any byte is fetched
            # (client.get() would slurp the whole body first).
            resp = await client.send(client.build_request("GET", current),
                                     stream=True)
            if resp.status_code // 100 == 3 and "location" in resp.headers:
                # Relative Location headers resolve against the current URL.
                nxt = current.join(resp.headers["location"])
                with contextlib.suppress(httpx.HTTPError):
                    await resp.aclose()
                current = await _validate_hop(str(nxt))
                continue
            resp.raise_for_status()
            break
        else:
            if resp is not None:
                with contextlib.suppress(httpx.HTTPError):
                    await resp.aclose()
            raise ValueError(f"重定向超过 {max_redirects} 跳，已放弃")

        declared = resp.headers.get("content-length")
        if declared is not None and int(declared) > max_bytes:
            # Close before raising: leaving an unread body makes the client's
            # exit path drain a response the server may never finish, and that
            # secondary error would mask this refusal. A failed close just
            # drops the connection — irrelevant to the refusal.
            with contextlib.suppress(httpx.HTTPError):
                await resp.aclose()
            raise ValueError(
                f"文件声明大小 {_human_size(int(declared))}，"
                f"超过上限 {_human_size(max_bytes)}")

        content_type = resp.headers.get("content-type")
        target = _guess_target(ws, str(current), path, content_type)
        if target.exists():
            raise ValueError(f"目标文件已存在：{target.relative_to(ws.root)}，"
                             "请换一个保存路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")

        hasher = hashlib.sha256()
        size = 0
        try:
            with part.open("wb") as fh:
                async for chunk in resp.aiter_bytes(64 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(
                            f"下载超过上限 {_human_size(max_bytes)}，已中断"
                            "（服务端未如实申报大小）")
                    hasher.update(chunk)
                    # The loop yields per chunk, keeping the event loop live;
                    # blocking disk writes at 64KB granularity are bounded.
                    fh.write(chunk)
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
        return target, size, content_type, hasher.hexdigest()


def register_download_tools(
    registry: ToolRegistry,
    workspace_for: Callable[[AgentContext], Workspace],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> None:
    """Register ``download_file`` bound to a host-supplied
    ``workspace_for(ctx) -> Workspace`` (same pattern as the file tools)."""

    async def download_file(ctx, args):
        url = (args.get("url") or "").strip()
        if not url:
            return ToolResult(False, "缺少 url", "download_file 需要 url。")
        ws = workspace_for(ctx)
        # The target path is only known after redirect/content-type resolution,
        # so the growth check runs against the run's created-count alone: a
        # full budget refuses one more file whatever it will be called.
        guard = _check_run_growth(ctx, ws, [args.get("path") or "!new"])
        if guard is not None:
            return guard
        started = time.time()
        try:
            target, size, content_type, sha = await asyncio.wait_for(
                download_to_workspace(
                    ws, url, args.get("path"),
                    max_bytes=max_bytes, timeout=timeout,
                    max_redirects=max_redirects),
                timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResult(False, "下载超时",
                              f"下载超过 {timeout:.0f}s 总预算，已取消。")
        except ValueError as exc:
            return ToolResult(False, "下载被拒绝", str(exc))
        except PermissionError as exc:
            return ToolResult(False, "非法路径", str(exc))
        except httpx.HTTPError as exc:
            return ToolResult(False, "下载失败",
                              f"网络错误：{type(exc).__name__}: {exc}")
        rel = target.relative_to(ws.root).as_posix()
        _note_created(ctx, ws, rel)
        type_note = f"，{content_type.split(';')[0]}" if content_type else ""
        return ToolResult(
            True, f"下载 {rel}",
            f"已保存 {rel}（{_human_size(size)}{type_note}，"
            f"sha256 {sha[:16]}…，用时 {time.time() - started:.1f}s）。")

    registry.register(
        ToolSpec("download_file",
                 "从 http/https URL 下载一个文件到工作区（默认 downloads/ 目录，"
                 "流式落盘、限制单文件大小、禁止覆盖已有文件、拒绝内网/环回地址）。"
                 "下载后用 read_file（文本）或 run_code（解析/统计）处理内容。",
                 {"type": "object",
                  "properties": {
                      "url": {"type": "string", "description": "要下载的完整 URL"},
                      "path": {"type": "string",
                               "description": "工作区内相对保存路径（可选，"
                                              "默认 downloads/<文件名>）"}},
                  "required": ["url"]},
                 ToolCategory.WRITE,
                 # Kernel-level backstop past the handler's own total budget,
                 # so DNS wedges / read stalls can never freeze the step.
                 timeout=timeout + 15),
        download_file)
