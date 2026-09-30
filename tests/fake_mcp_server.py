"""Fake MCP stdio server for tests: newline-delimited JSON-RPC 2.0.

Protocol surface: initialize handshake, tools/list, tools/call. Behaviour is
env-driven so one script covers many test cases:

  FAKE_MCP_LOG     append one line per request ("<pid> <method>") for liveness
                   assertions (session reuse / respawn). Client replies to our
                   pushed requests are logged as "<pid> reply <id> error|result".
  FAKE_SECRET      echoed back by the ``echo_env`` tool (env passthrough check).
  FAKE_SLEEP       seconds the ``slow`` tool sleeps before answering.
  FAKE_PUSH_REQUEST  when replying to ``tools/list``, first push a
                   server-initiated request whose id (2) collides with the
                   client's own tools/list request id — exercising the JSON-RPC
                   id-space separation in the client's read loop.
"""
from __future__ import annotations

import json
import os
import sys
import time

_PUSHED = False

TOOLS = [
    {
        "name": "echo",
        "description": "回显文本",
        "inputSchema": {"type": "object",
                        "properties": {"text": {"type": "string"}}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "boom",
        "description": "总是失败的工具",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "echo_env",
        "description": "回显 FAKE_SECRET 环境变量",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "slow",
        "description": "慢工具（FAKE_SLEEP 控制延迟）",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
]


def log_request(method: str) -> None:
    path = os.environ.get("FAKE_MCP_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} {method}\n")


def log_reply(msg: dict) -> None:
    path = os.environ.get("FAKE_MCP_LOG")
    if path:
        kind = "error" if "error" in msg else "result"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} reply {msg.get('id')} {kind}\n")


def maybe_push_colliding_request() -> None:
    """Push a server-initiated request whose id collides with the client's
    pending tools/list id (2) — before replying to tools/list, so the client's
    read loop sees it while its own request is still pending."""
    global _PUSHED
    if os.environ.get("FAKE_PUSH_REQUEST") and not _PUSHED:
        _PUSHED = True
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0", "id": 2,
            "method": "sampling/createMessage", "params": {},
        }) + "\n")
        sys.stdout.flush()


def handle(method: str, params: dict) -> dict:
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp", "version": "0.0.1"}}
    if method == "tools/list":
        maybe_push_colliding_request()
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params.get("name")
        if name == "echo":
            text = (params.get("arguments") or {}).get("text", "")
            return {"content": [{"type": "text", "text": f"echo:{text}"}]}
        if name == "echo_env":
            return {"content": [{"type": "text",
                                 "text": os.environ.get("FAKE_SECRET", "")}]}
        if name == "slow":
            time.sleep(float(os.environ.get("FAKE_SLEEP", "2")))
            return {"content": [{"type": "text", "text": "finally"}]}
        if name == "boom":
            return {"isError": True,
                    "content": [{"type": "text", "text": "炸了"}]}
        raise ValueError(f"unknown tool {name}")
    raise ValueError(f"unknown method {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if "method" not in msg:
            # a reply from the client (e.g. the method-not-found answer to our
            # pushed request) — log it, never treat it as a request
            if msg.get("id") is not None:
                log_reply(msg)
            continue
        if "id" not in msg:
            continue  # notification — nothing to log or reply to
        log_request(msg["method"])
        try:
            result = handle(msg["method"], msg.get("params") or {})
        except Exception as exc:  # noqa: BLE001
            reply = {"jsonrpc": "2.0", "id": msg["id"],
                     "error": {"code": -32603, "message": str(exc)}}
        else:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    if os.environ.get("FAKE_STUBBORN"):
        time.sleep(600)


if __name__ == "__main__":
    main()
