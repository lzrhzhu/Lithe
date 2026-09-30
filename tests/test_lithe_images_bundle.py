"""lithe.bundles.images: tool-tier image perception.

Pinning (a) the stdlib-only header probe for every supported container
(PNG/JPEG/GIF/BMP/WEBP), (b) the analyze_image wire shape (OpenAI image_url
data-URL block + text), (c) the (hash, question, detail) answer cache, and
(d) the degradation paths — non-image, oversized, no-llm_config registration.
All VLM calls are faked; no network."""
from __future__ import annotations

import struct

import lithe.bundles.images as images_mod
from lithe import AgentContext, LLMConfig, ToolRegistry
from lithe.bundles.images import probe_image, register_image_tools
from lithe.bundles.workspace import Workspace


# --- hand-built minimal images (headers only; probe never decodes pixels) ---

def _png(w: int = 3, h: int = 2, ppu: int | None = None) -> bytes:
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # RGB, 8bit
    out = b"\x89PNG\r\n\x1a\n"
    out += struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 4
    if ppu:
        phys = struct.pack(">IIB", ppu, ppu, 1)
        out += struct.pack(">I", 9) + b"pHYs" + phys + b"\x00" * 4
    return out


def _jpeg(w: int = 640, h: int = 480, dpi: int = 300) -> bytes:
    jfif = (b"JFIF\x00" + b"\x01\x02" + b"\x01"
            + struct.pack(">HH", dpi, dpi) + b"\x00\x00")
    sof = b"\x08" + struct.pack(">HH", h, w) + b"\x03" + (
        b"\x01\x22\x00\x02\x11\x01\x03\x11\x01")
    return (b"\xff\xd8"
            + b"\xff\xe0" + struct.pack(">H", len(jfif) + 2) + jfif
            + b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof
            + b"\xff\xd9")


def _gif(w: int = 10, h: int = 20) -> bytes:
    return b"GIF89a" + struct.pack("<HH", w, h) + b"\x00" * 3


def _bmp(w: int = 8, h: int = -6) -> bytes:  # negative height = top-down
    return b"BM" + b"\x00" * 16 + struct.pack("<ii", w, h) + b"\x00" * 8


def _webp_vp8l(w: int = 5, h: int = 7) -> bytes:
    bits = (w - 1) | ((h - 1) << 14)
    payload = b"\x2f" + struct.pack("<I", bits)
    body = b"WEBP" + b"VP8L" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _ctx() -> AgentContext:
    return AgentContext(run_id="r", user_id="u")


# --- probe_image ---

def test_probe_png():
    info = probe_image(_png(1024, 768, ppu=11811))  # 11811 ppm ≈ 300 dpi
    assert info["format"] == "PNG"
    assert (info["width"], info["height"]) == (1024, 768)
    assert info["color_type"] == "RGB"
    assert info["bit_depth"] == 8
    assert info["dpi"] == (300.0, 300.0)


def test_probe_jpeg():
    info = probe_image(_jpeg(640, 480, dpi=300))
    assert info["format"] == "JPEG"
    assert (info["width"], info["height"]) == (640, 480)
    assert info["dpi"] == (300.0, 300.0)
    assert info["progressive"] is False


def test_probe_gif_bmp_webp():
    assert probe_image(_gif(10, 20))["width"] == 10
    bmp = probe_image(_bmp(8, -6))
    assert (bmp["width"], bmp["height"]) == (8, 6)
    wl = probe_image(_webp_vp8l(5, 7))
    assert (wl["format"], wl["width"], wl["height"], wl["lossless"]) == \
        ("WEBP", 5, 7, True)


def test_probe_unknown():
    assert probe_image(b"hello world, definitely not an image")["format"] is None
    assert probe_image(b"")["format"] is None


# --- image_info tool ---

async def test_image_info_roundtrip(tmp_path):
    ws = Workspace(tmp_path)
    ws.write_bytes("fig.png", _png(1024, 768, ppu=11811))
    reg = ToolRegistry()
    register_image_tools(reg, lambda ctx: ws)
    res = await reg.dispatch("image_info", {"path": "fig.png"}, _ctx())
    assert res.ok
    assert "PNG" in res.content and "1024×768" in res.content
    assert "300×300 dpi" in res.content


async def test_image_info_errors(tmp_path):
    ws = Workspace(tmp_path)
    ws.write_bytes("x.txt", b"plain text")
    reg = ToolRegistry()
    register_image_tools(reg, lambda ctx: ws)
    missing = await reg.dispatch("image_info", {"path": "nope.png"}, _ctx())
    assert not missing.ok and "不存在" in missing.content
    binary = await reg.dispatch("image_info", {"path": "x.txt"}, _ctx())
    assert not binary.ok and "图片格式" in binary.content
    escape = await reg.dispatch("image_info", {"path": "../etc"}, _ctx())
    assert not escape.ok


# --- analyze_image tool ---

def _fake_chat(calls: list):
    async def fake(client, *, base_url, api_key, model, messages, **kw):
        calls.append({"base_url": base_url, "model": model,
                      "messages": messages})
        return {"choices": [{"message": {
            "content": "<think>hidden</think>图中是一条上升的折线。"}}]}
    return fake


def _registry_with_vision(ws, monkeypatch, calls, **kw):
    monkeypatch.setattr(images_mod, "chat_completion", _fake_chat(calls))
    reg = ToolRegistry()
    register_image_tools(reg, lambda ctx: ws, llm_config=LLMConfig(
        model="vision-model", base_url="http://vlm.test/v1",
        api_key="k"), **kw)
    return reg


async def test_analyze_image_wire_shape_and_think_strip(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    ws.write_bytes("fig.png", _png())
    calls: list = []
    reg = _registry_with_vision(ws, monkeypatch, calls)
    res = await reg.dispatch(
        "analyze_image", {"path": "fig.png", "question": "图里是什么",
                          "detail": "low"}, _ctx())
    assert res.ok
    assert res.content == "图中是一条上升的折线。"  # <think> stripped
    sent = calls[0]["messages"][0]
    assert sent["role"] == "user"
    assert [b["type"] for b in sent["content"]] == ["image_url", "text"]
    assert sent["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,")
    assert sent["content"][0]["image_url"]["detail"] == "low"
    assert sent["content"][1]["text"] == "图里是什么"
    assert calls[0]["model"] == "vision-model"


async def test_analyze_image_cache(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    ws.write_bytes("fig.png", _png())
    calls: list = []
    reg = _registry_with_vision(ws, monkeypatch, calls)
    same = {"path": "fig.png", "question": "图里是什么"}
    r1 = await reg.dispatch("analyze_image", same, _ctx())
    r2 = await reg.dispatch("analyze_image", same, _ctx())
    assert len(calls) == 1  # second hit served from cache
    assert r1.content == r2.content
    await reg.dispatch(
        "analyze_image", {"path": "fig.png", "question": "换个问题"}, _ctx())
    assert len(calls) == 2  # new question misses
    await reg.dispatch(
        "analyze_image", {"path": "fig.png", "question": "图里是什么",
                          "detail": "high"}, _ctx())
    assert len(calls) == 3  # different detail misses


async def test_analyze_image_guards(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    ws.write_bytes("big.png", _png())
    ws.write_bytes("x.txt", b"text")
    calls: list = []
    reg = _registry_with_vision(ws, monkeypatch, calls, max_image_bytes=8)
    big = await reg.dispatch(
        "analyze_image", {"path": "big.png", "question": "?"}, _ctx())
    assert not big.ok and "图片过大" in big.summary
    nonimg = await reg.dispatch(
        "analyze_image", {"path": "x.txt", "question": "?"}, _ctx())
    assert not nonimg.ok and "非图片" in nonimg.summary
    noq = await reg.dispatch("analyze_image", {"path": "big.png"}, _ctx())
    assert not noq.ok
    assert calls == []  # every guard fired before any HTTP request


async def test_no_llm_config_registers_info_only(tmp_path):
    ws = Workspace(tmp_path)
    reg = ToolRegistry()
    register_image_tools(reg, lambda ctx: ws)
    assert "image_info" in reg.names()
    assert "analyze_image" not in reg.names()
    res = await reg.dispatch("analyze_image", {"path": "x"}, _ctx())
    assert not res.ok  # unknown tool
