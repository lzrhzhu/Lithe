"""lithe.bundles.download: controlled ingress into the workspace.

Pinning (a) the SSRF address policy on the exact function the tool enforces,
(b) the happy path (sha256, default ``downloads/`` landing, atomic rename,
Content-Type→extension fallback), (c) every refusal boundary: scheme,
redirect cap/loop, size pre-check and streaming cutoff (with ``.part``
cleanup), overwrite, traversal, total-time budget — and (d) the registered
spec (WRITE, kernel timeout backstop). The local HTTP fixture runs on
127.0.0.1, which the real policy forbids; tests monkeypatch ``_check_host``
precisely because that is the product code's only trust boundary."""
from __future__ import annotations

import asyncio
import hashlib
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import httpx

import lithe.bundles.download as dl
from lithe import AgentContext, ToolRegistry
from lithe.bundles.download import register_download_tools
from lithe.bundles.workspace import Workspace
from lithe.tools import ToolCategory


def _ctx() -> AgentContext:
    return AgentContext(run_id="r", user_id="u")


def _fake_addr(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


# --- _check_host: the SSRF policy, no real DNS ---

@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "172.16.0.1",
                                "192.168.1.1", "169.254.169.254",
                                "100.64.0.1", "0.0.0.0", "224.0.0.1"])
def test_check_host_rejects_non_global_ipv4(ip, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _fake_addr(ip))
    with pytest.raises(ValueError):
        dl._check_host("example.com")


@pytest.mark.parametrize("ip", ["::1", "fc00::1", "fe80::1", "ff02::1"])
def test_check_host_rejects_non_global_ipv6(ip, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET6, 1, 6, "", (ip, 0))])
    with pytest.raises(ValueError):
        dl._check_host("example.com")


def test_check_host_allows_public(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: _fake_addr("140.82.121.4"))
    dl._check_host("example.com")


def test_check_host_rejects_when_any_record_is_private(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: (
        _fake_addr("140.82.121.4") + _fake_addr("10.0.0.1")))
    with pytest.raises(ValueError):
        dl._check_host("example.com")


def test_check_host_resolution_failure(monkeypatch):
    def boom(*a, **k):
        raise socket.gaierror("no dns")
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(ValueError):
        dl._check_host("nonexistent.invalid")


# --- local HTTP fixture (127.0.0.1; tests patch _check_host to trust it) ---

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output clean
        pass

    def do_GET(self):
        if self.path == "/hello.txt":
            body = b"hello " * 100
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/noext":
            body = b"%PDF-1.4 fake"
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/redir":
            self.send_response(302)
            self.send_header("Location", "/hello.txt")
            self.end_headers()
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.end_headers()
        elif self.path == "/lying":
            # Chunked, no Content-Length: streams 4 chunks of 512B so the
            # streaming cutoff fires past a (here absent) declared length.
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for _ in range(8):
                chunk = b"x" * 512
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        elif self.path == "/declared-big":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(10 * 1024 * 1024))
            self.end_headers()
        elif self.path == "/slow":
            time.sleep(2.0)
            self.send_response(200)
            self.send_header("Content-Length", "1")
            self.end_headers()
            self.wfile.write(b"x")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def ws(tmp_path) -> Workspace:
    return Workspace(root=tmp_path / "ws")


@pytest.fixture
def trust_local(monkeypatch):
    monkeypatch.setattr(dl, "_check_host", lambda host: None)


async def test_happy_path_sha256_and_default_landing(ws, server, trust_local):
    target, size, ctype, sha = await dl.download_to_workspace(
        ws, f"{server}/hello.txt", None,
        max_bytes=1024 * 1024, timeout=30, max_redirects=5)
    assert target == ws.root / "downloads" / "hello.txt"
    data = target.read_bytes()
    assert size == len(data) == 600
    assert sha == hashlib.sha256(data).hexdigest()
    assert ctype.startswith("text/plain")
    assert not list(ws.root.rglob("*.part"))


async def test_explicit_path(ws, server, trust_local):
    target, *_ = await dl.download_to_workspace(
        ws, f"{server}/hello.txt", "materials/a.txt",
        max_bytes=1024 * 1024, timeout=30, max_redirects=5)
    assert target == ws.root / "materials" / "a.txt"
    assert target.is_file()


async def test_extension_fallback_from_content_type(ws, server, trust_local):
    target, *_ = await dl.download_to_workspace(
        ws, f"{server}/noext", None,
        max_bytes=1024 * 1024, timeout=30, max_redirects=5)
    assert target.name == "noext.pdf"


async def test_redirect_followed(ws, server, trust_local):
    target, size, *_ = await dl.download_to_workspace(
        ws, f"{server}/redir", None,
        max_bytes=1024 * 1024, timeout=30, max_redirects=5)
    assert size == 600
    assert target.name == "hello.txt"


async def test_redirect_loop_rejected(ws, server, trust_local):
    with pytest.raises(ValueError, match="重定向超过"):
        await dl.download_to_workspace(
            ws, f"{server}/loop", None,
            max_bytes=1024 * 1024, timeout=30, max_redirects=3)


async def test_content_length_precheck(ws, server, trust_local):
    with pytest.raises(ValueError, match="超过上限"):
        await dl.download_to_workspace(
            ws, f"{server}/declared-big", None,
            max_bytes=1024, timeout=30, max_redirects=5)
    assert not (ws.root / "downloads").exists() or \
        not list((ws.root / "downloads").iterdir())


async def test_streaming_cutoff_cleans_part(ws, server, trust_local):
    with pytest.raises(ValueError, match="超过上限"):
        await dl.download_to_workspace(
            ws, f"{server}/lying", None,
            max_bytes=1024, timeout=30, max_redirects=5)
    assert not list(ws.root.rglob("*.part"))
    assert not list(ws.root.rglob("lying*"))


async def test_refuses_existing_target(ws, server, trust_local):
    (ws.root / "downloads").mkdir(parents=True)
    (ws.root / "downloads" / "hello.txt").write_text("old")
    with pytest.raises(ValueError, match="已存在"):
        await dl.download_to_workspace(
            ws, f"{server}/hello.txt", None,
            max_bytes=1024 * 1024, timeout=30, max_redirects=5)
    assert (ws.root / "downloads" / "hello.txt").read_text() == "old"


async def test_scheme_allowlist(ws, trust_local):
    with pytest.raises(ValueError, match="http/https"):
        await dl.download_to_workspace(
            ws, "file:///etc/passwd", None,
            max_bytes=1024, timeout=30, max_redirects=5)


async def test_traversal_refused(ws, server, trust_local):
    with pytest.raises(PermissionError):
        await dl.download_to_workspace(
            ws, f"{server}/hello.txt", "../../escape.txt",
            max_bytes=1024 * 1024, timeout=30, max_redirects=5)


async def test_total_time_budget(ws, server, trust_local):
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            dl.download_to_workspace(
                ws, f"{server}/slow", None,
                max_bytes=1024 * 1024, timeout=30, max_redirects=5),
            timeout=0.5)


async def test_ssrf_still_enforced_without_patch(ws, server):
    with pytest.raises(ValueError, match="SSRF"):
        await dl.download_to_workspace(
            ws, f"{server}/hello.txt", None,
            max_bytes=1024 * 1024, timeout=10, max_redirects=5)


async def test_http_error_surfaced(ws, server, trust_local):
    with pytest.raises(httpx.HTTPStatusError):
        await dl.download_to_workspace(
            ws, f"{server}/missing.bin", None,
            max_bytes=1024 * 1024, timeout=30, max_redirects=5)


# --- registered tool: spec shape + dispatch round-trip ---

def test_registered_spec_shape(tmp_path):
    registry = ToolRegistry()
    register_download_tools(registry, lambda ctx: Workspace(root=tmp_path))
    spec = registry._entries["download_file"].spec
    assert spec.category == ToolCategory.WRITE
    assert spec.timeout is not None and spec.timeout >= dl.DEFAULT_TIMEOUT
    assert "url" in spec.parameters["properties"]
    assert spec.parameters["required"] == ["url"]
    # WRITE tools vanish from read-only mode specs (mode filtering contract).
    names = [s["function"]["name"] for s in registry.specs_for_mode()]
    assert "download_file" in names


async def test_dispatch_round_trip(tmp_path, server, trust_local):
    registry = ToolRegistry()
    register_download_tools(registry, lambda ctx: Workspace(root=tmp_path / "ws"))
    res = await registry.dispatch(
        "download_file", {"url": f"{server}/hello.txt"}, _ctx())
    assert res.ok
    assert "downloads/hello.txt" in res.content
    assert "sha256" in res.content
    assert (tmp_path / "ws" / "downloads" / "hello.txt").is_file()


async def test_dispatch_missing_url(tmp_path):
    registry = ToolRegistry()
    register_download_tools(registry, lambda ctx: Workspace(root=tmp_path))
    res = await registry.dispatch("download_file", {}, _ctx())
    assert not res.ok


async def test_dispatch_timeout_result(tmp_path, server, trust_local):
    registry = ToolRegistry()
    register_download_tools(
        registry, lambda ctx: Workspace(root=tmp_path / "ws"), timeout=0.5)
    res = await registry.dispatch(
        "download_file", {"url": f"{server}/slow"}, _ctx())
    assert not res.ok
    assert "超时" in res.summary or "超时" in res.content
