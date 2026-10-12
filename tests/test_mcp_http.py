"""lithe.bundles.mcp (streamable-http transport): JSON and SSE responses,
server ``Mcp-Session-Id`` echo, auth headers passthrough, DELETE on close,
JSON-vs-SSE decoding, and config command/url exclusivity — against a stdlib
fake HTTP MCP server, no real network."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lithe import AgentContext, ToolRegistry
from lithe.bundles.mcp import (MCPManager, MCPServerConfig, _parse_sse_messages,
                                  parse_servers)

SEEN: dict[str, str] = {}
DELETED = {"flag": False}


class _FakeMCPServer(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _reply(self, status: int, body: str, *,
               ctype: str = "application/json", extra: dict | None = None):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        if self.path != "/mcp":
            self._reply(404, json.dumps({"error": "not found"}))
            return
        length = int(self.headers.get("Content-Length", 0))
        msg = json.loads(self.rfile.read(length))
        auth = self.headers.get("Authorization", "")
        sid = self.headers.get("Mcp-Session-Id", "")
        method = msg.get("method")
        if method == "initialize":
            result = {"protocolVersion": msg["params"]["protocolVersion"],
                      "capabilities": {}, "serverInfo": {"name": "fake-http"}}
            self._reply(200, json.dumps(
                {"jsonrpc": "2.0", "id": msg["id"], "result": result}),
                extra={"Mcp-Session-Id": "sess-42"})
        elif method == "notifications/initialized":
            SEEN["sid_on_init_notification"] = sid
            SEEN["auth"] = auth
            self._reply(202, "")
        elif method == "tools/list":
            SEEN["sid_on_request"] = sid
            tools = [{"name": "echo", "description": "回显",
                      "inputSchema": {"type": "object",
                                      "properties": {"text": {"type": "string"}}},
                      "annotations": {"readOnlyHint": True}}]
            stream = ("event: message\n"
                      f"data: {json.dumps({'jsonrpc': '2.0', 'method': 'ping'})}\n\n"
                      "event: message\n"
                      f"data: {json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'tools': tools}})}\n\n")
            self._reply(200, stream, ctype="text/event-stream")
        elif method == "tools/call":
            text = (msg["params"].get("arguments") or {}).get("text", "")
            SEEN["auth_on_call"] = auth
            result = {"content": [{"type": "text", "text": f"echo:{text}"}]}
            self._reply(200, json.dumps(
                {"jsonrpc": "2.0", "id": msg["id"], "result": result}))
        else:
            self._reply(400, json.dumps(
                {"jsonrpc": "2.0", "id": msg.get("id"),
                 "error": {"code": -32601, "message": f"unknown {method}"}}))

    def do_DELETE(self):  # noqa: N802
        DELETED["flag"] = self.headers.get("Mcp-Session-Id") == "sess-42"
        self._reply(204, "")


@pytest.fixture()
def http_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeMCPServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    server.shutdown()
    thread.join()


def _cfg(url: str) -> MCPServerConfig:
    return MCPServerConfig(name="fake", url=url,
                           headers={"Authorization": "Bearer test-key"},
                           timeout=5, startup_timeout=10)


async def test_http_attach_dispatch_session_and_delete(http_url):
    mgr = MCPManager([_cfg(http_url)])
    reg = ToolRegistry()
    try:
        report = await mgr.attach(reg)
        assert report["fake"] == ["fake__echo"]
        ctx = AgentContext(run_id="r", user_id="u")
        res = await reg.dispatch("fake__echo", {"text": "hi"}, ctx)
        assert res.ok and res.content == "echo:hi"
        assert res.ui == [], "ui 必须是列表（字符串会被 runtime 逐字符当事件喷出）"
        assert res.summary.startswith("fake·echo")
        assert SEEN["auth"] == "Bearer test-key"
        assert SEEN["sid_on_init_notification"] == "sess-42"
        assert SEEN["sid_on_request"] == "sess-42"
        assert SEEN["auth_on_call"] == "Bearer test-key"
    finally:
        await mgr.close()
    assert DELETED["flag"], "close 应带 Mcp-Session-Id 发 DELETE"


async def test_http_session_reused_across_attaches(http_url):
    mgr = MCPManager([_cfg(http_url)])
    try:
        await mgr.attach(ToolRegistry())
        first = mgr.status()["fake"]
        await mgr.attach(ToolRegistry())
        assert mgr.status()["fake"]["alive"] and first["alive"]
    finally:
        await mgr.close()


async def test_http_bad_endpoint_degrades(http_url):
    mgr = MCPManager([MCPServerConfig(name="dead", url=http_url + "/nope",
                                      startup_timeout=5, timeout=5)])
    reg = ToolRegistry()
    try:
        report = await mgr.attach(reg)
        assert isinstance(report["dead"], str) and "启动失败" in report["dead"]
    finally:
        await mgr.close()


def test_sse_parser_multiline_and_noise():
    body = ("event: message\ndata: {\"a\": 1}\n\n"
            ": keep-alive comment\n\n"
            "data: {\"b\":\ndata: 2}\n\n")
    assert _parse_sse_messages(body) == [{"a": 1}, {"b": 2}]


def test_config_command_url_exclusivity():
    with pytest.raises(ValueError):
        MCPServerConfig(name="x", command=["ls"], url="http://x/mcp")
    with pytest.raises(ValueError):
        MCPServerConfig(name="x")
    cfg = parse_servers('{"h": {"url": "http://x/mcp", "headers": '
                        '{"Authorization": "Bearer k"}}}')[0]
    assert cfg.url == "http://x/mcp" and cfg.command is None
    assert cfg.headers["Authorization"] == "Bearer k"
