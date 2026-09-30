"""Images bundle: tool-tier image perception, no kernel changes.

Two tools, per the "perception stays in tools" design:

- ``image_info`` — zero-model, stdlib-only header probe (PNG/JPEG/GIF/BMP/
  WEBP): dimensions, dpi where the container carries it, color type. Answers
  every deterministic question ("多大？是不是 300dpi？RGB 还是调色板？") with
  no model call at all.
- ``analyze_image`` — the only perception path: reads the file, sends it as an
  OpenAI ``image_url`` data-URL content block together with the question in
  ONE chat call, and returns the description text. The image itself never
  enters the main conversation, so the kernel's context budget / trimming /
  replay machinery is untouched; answers are memoized per
  (file hash, question, detail) in-process.

This is deliberately NOT a subagent: a full ReAct runtime (tools, system
prompt, step loop) is overkill for one look-at-image-answer-question call —
a single :func:`lithe.llm.chat_completion` round-trip is the whole story.
The model used is whatever ``llm_config`` the host passes at registration:
the same ``LLMConfig`` as the main loop reuses the main model; a dedicated
one routes vision to a different endpoint; omitting it registers only the
deterministic ``image_info``.

Optional bundle — stdlib + httpx only (like the rest of lithe); Pillow is
NOT required. Headers are parsed, pixels never decoded, so a crafted file
cannot burn CPU here. Hosts wanting downscaling should pre-process uploads
or use the ``detail`` argument.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import struct
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import httpx

from lithe.context import AgentContext
from lithe.llm import chat_completion, first_content, strip_think
from lithe.runtime import LLMConfig
from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec
from lithe.bundles.workspace import Workspace

DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_CACHE_SIZE = 64
DEFAULT_ANALYZE_TIMEOUT = 120.0
# image_info reads at most this many bytes: dimensions/dpi live in header
# chunks well before the pixel bulk, so a multi-hundred-MB image probes in
# constant memory instead of a full read.
_PROBE_WINDOW = 256 * 1024

_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif",
         "BMP": "image/bmp", "WEBP": "image/webp"}

_PNG_COLOR = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+Alpha", 6: "RGBA"}

# SOF0-SOF15 minus DHT(0xC4)/JPG(0xC8)/DAC(0xCC) carry frame dimensions.
_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


# --------------------------------------------------------------------------- #
# stdlib-only format probing
# --------------------------------------------------------------------------- #

def _png_dpi(data: bytes) -> tuple[float, float] | None:
    """Scan PNG chunks up to IDAT for a ``pHYs`` (pixels-per-meter) block."""
    off = 8
    while off + 12 <= len(data):
        (length,) = struct.unpack(">I", data[off:off + 4])
        ctype = data[off + 4:off + 8]
        if ctype == b"pHYs" and length >= 9 and off + 8 + length <= len(data):
            ppu_x, ppu_y, unit = struct.unpack(">IIB", data[off + 8:off + 17])
            if unit == 1 and ppu_x:
                return (round(ppu_x * 0.0254, 1), round(ppu_y * 0.0254, 1))
            return None
        if ctype == b"IDAT":
            break
        off += 12 + length
    return None


def _probe_jpeg(data: bytes) -> dict:
    """Walk JPEG segments for JFIF density (APP0) and frame size (SOFn)."""
    out: dict = {"format": "JPEG"}
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # no payload
            i += 2
            continue
        (seg_len,) = struct.unpack(">H", data[i + 2:i + 4])
        if marker == 0xE0 and data[i + 4:i + 9] == b"JFIF\x00":
            if i + 16 <= len(data):
                units = data[i + 11]
                xd, yd = struct.unpack(">HH", data[i + 12:i + 16])
                if xd:
                    if units == 1:
                        out["dpi"] = (float(xd), float(yd))
                    elif units == 2:  # dots per cm
                        out["dpi"] = (round(xd * 2.54, 1), round(yd * 2.54, 1))
        if marker in _JPEG_SOF and i + 9 <= len(data):
            out["height"], out["width"] = struct.unpack(">HH", data[i + 5:i + 9])
            out["progressive"] = marker == 0xC2
            break
        i += 2 + seg_len
    return out


def _probe_webp(data: bytes) -> dict:
    """Parse the first RIFF chunk of a WEBP (VP8 / VP8L / VP8X)."""
    out: dict = {"format": "WEBP"}
    off = 12
    while off + 8 <= len(data):
        fourcc = data[off:off + 4]
        (size,) = struct.unpack("<I", data[off + 4:off + 8])
        body = data[off + 8:off + 8 + size]
        if fourcc == b"VP8X" and len(body) >= 10:
            out["width"] = int.from_bytes(body[4:7], "little") + 1
            out["height"] = int.from_bytes(body[7:10], "little") + 1
            break
        if fourcc == b"VP8 " and len(body) >= 10:  # lossy: skip tag+sync
            w, h = struct.unpack("<HH", body[6:10])
            out["width"], out["height"] = w & 0x3FFF, h & 0x3FFF
            break
        if fourcc == b"VP8L" and len(body) >= 5:  # lossless
            bits = int.from_bytes(body[1:5], "little")
            out["width"] = (bits & 0x3FFF) + 1
            out["height"] = ((bits >> 14) & 0x3FFF) + 1
            out["lossless"] = True
            break
        off += 8 + size + (size & 1)  # RIFF chunks pad to even sizes
    return out


def probe_image(data: bytes) -> dict:
    """Best-effort header probe: format / dimensions / dpi / mode extras.

    Returns ``{"format", "width", "height", "dpi", ...extras}``; ``format`` is
    ``None`` when the magic bytes match no supported container. Only headers
    are read — no pixel decoding, no external dependency.
    """
    out: dict = {"format": None, "width": None, "height": None, "dpi": None}
    if len(data) >= 29 and data[:8] == b"\x89PNG\r\n\x1a\n":
        out["format"] = "PNG"
        out["width"] = struct.unpack(">I", data[16:20])[0]
        out["height"] = struct.unpack(">I", data[20:24])[0]
        out["bit_depth"] = data[24]
        out["color_type"] = _PNG_COLOR.get(data[25], f"未知({data[25]})")
        out["interlaced"] = data[28] == 1
        out["dpi"] = _png_dpi(data)
    elif len(data) >= 10 and data[:4] == b"GIF8":
        out["format"] = "GIF"
        out["width"], out["height"] = struct.unpack("<HH", data[6:10])
    elif len(data) >= 26 and data[:2] == b"BM":
        out["format"] = "BMP"
        w, h = struct.unpack("<ii", data[18:26])  # negative height = top-down
        out["width"], out["height"] = abs(w), abs(h)
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        out.update(_probe_webp(data))
    elif len(data) >= 4 and data[:2] == b"\xff\xd8":
        out.update(_probe_jpeg(data))
    return out


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def _resolve(ws: Workspace, rel: str) -> tuple[ToolResult | None, Path | None]:
    """Shared path resolution for both tools: (error_result | None, target)."""
    try:
        target = ws.safe_path(rel)
    except PermissionError as exc:
        return ToolResult(False, "非法路径", str(exc)), None
    if not target.is_file():
        return ToolResult(False, "文件不存在", f"文件不存在：{rel}"), None
    return None, target


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #

def _path_params() -> dict:
    return {"type": "object",
            "properties": {"path": {"type": "string",
                                    "description": "工作区内相对路径"}},
            "required": ["path"]}


def _analyze_params() -> dict:
    p = _path_params()
    p["properties"]["question"] = {
        "type": "string",
        "description": "想从图片中了解什么；越具体越好（如：图中曲线在哪个区间上升）"}
    p["properties"]["detail"] = {
        "type": "string", "enum": ["auto", "low", "high"],
        "description": "分析精度：low 便宜快速，high 精细；默认 auto"}
    p["required"].append("question")
    return p


def register_image_tools(
    registry: ToolRegistry,
    workspace_for: Callable[[AgentContext], Workspace],
    *,
    llm_config: LLMConfig | None = None,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    cache_size: int = DEFAULT_CACHE_SIZE,
    analyze_timeout: float = DEFAULT_ANALYZE_TIMEOUT,
) -> None:
    """Register ``image_info`` (always) and ``analyze_image`` (when configured).

    ``workspace_for`` is the same host-supplied callable the workspace bundle
    uses. ``llm_config`` selects the perception model: pass the host's main
    ``LLMConfig`` to reuse the main model, a dedicated one to route vision to
    another endpoint, or ``None`` (default) to register only the deterministic
    probe. ``max_image_bytes`` guards the VLM payload; ``cache_size`` bounds
    the in-process (file-hash, question, detail) → answer memo; 
    ``analyze_timeout`` is the tool-level execution cap.
    """
    cache: OrderedDict[tuple, str] = OrderedDict()

    async def image_info(ctx: AgentContext, args: dict) -> ToolResult:
        rel = (args.get("path") or "").strip()
        if not rel:
            return ToolResult(False, "缺少 path", "缺少 path 参数。")
        err, target = _resolve(workspace_for(ctx), rel)
        if err is not None:
            return err
        # Read a bounded header window off the event loop: headers (magic,
        # dimensions, dpi chunks) sit at the front, so a huge file probes in
        # constant memory instead of an unbounded read_bytes on the loop.
        size = (await asyncio.to_thread(target.stat)).st_size
        with target.open("rb") as fh:
            data = await asyncio.to_thread(fh.read, _PROBE_WINDOW)
        info = probe_image(data)
        if info["format"] is None:
            return ToolResult(
                False, "非图片文件",
                f"{rel} 不是可识别的图片格式（PNG/JPEG/GIF/BMP/WEBP），"
                f"共 {_human_size(size)}。二进制处理可用 run_code。")
        body = (f"{rel}：{info['format']}，{info['width']}×{info['height']}"
                f"（{(info['width'] or 0) * (info['height'] or 0) / 1e6:.1f} 百万像素）")
        extra: list[str] = []
        if info["format"] == "PNG":
            extra.append(f"{info.get('color_type', '?')} {info['bit_depth']}bit")
            if info.get("interlaced"):
                extra.append("隔行扫描")
        elif info["format"] == "JPEG":
            extra.append("渐进式" if info.get("progressive") else "基线")
        elif info["format"] == "WEBP" and info.get("lossless"):
            extra.append("无损")
        if extra:
            body += "，" + " ".join(extra)
        dpi = info.get("dpi")
        if dpi:
            body += f"，{dpi[0]:g}×{dpi[1]:g} dpi"
        body += f"，{_human_size(size)}。"
        return ToolResult(True, f"探测 {rel}", body)

    registry.register(
        ToolSpec("image_info",
                 "探测工作区图片文件的确定信息：格式、像素尺寸、dpi、色彩模式"
                 "（PNG/JPEG/GIF/BMP/WEBP）。不调用模型；回答“这张图多大/"
                 "是否 300dpi/什么格式”这类问题应优先用它，再按需用 "
                 "analyze_image 做视觉分析。",
                 _path_params(), ToolCategory.READ),
        image_info)

    if llm_config is None:
        return
    cfg = llm_config

    async def analyze_image(ctx: AgentContext, args: dict) -> ToolResult:
        rel = (args.get("path") or "").strip()
        question = (args.get("question") or "").strip()
        if not rel:
            return ToolResult(False, "缺少 path", "缺少 path 参数。")
        if not question:
            return ToolResult(False, "缺少 question",
                              "analyze_image 需要一个明确的问题"
                              "（想从图中了解什么）。")
        detail = args.get("detail")
        if detail not in ("auto", "low", "high"):
            detail = None
        if detail == "auto":
            # The API treats "auto" and an omitted field identically;
            # normalizing keeps them ONE cache key (no double billing for
            # the same question asked both ways).
            detail = None
        err, target = _resolve(workspace_for(ctx), rel)
        if err is not None:
            return err
        # Size check BEFORE reading: a 2GB upload must be refused by stat,
        # not by loading 2GB into memory first.
        size = (await asyncio.to_thread(target.stat)).st_size
        if size > max_image_bytes:
            return ToolResult(
                False, "图片过大",
                f"{rel} 为 {_human_size(size)}，超过上限 "
                f"{_human_size(max_image_bytes)}。请先用 run_code 压缩或"
                f"降分辨率后再分析。")
        data = await asyncio.to_thread(target.read_bytes)
        mime = _MIME.get(probe_image(data)["format"] or "")
        if mime is None:
            return ToolResult(
                False, "非图片文件",
                f"{rel} 不是受支持的图片格式（PNG/JPEG/GIF/BMP/WEBP）。")
        key = (hashlib.sha256(data).hexdigest(), question, detail)
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return ToolResult(True, f"已缓存的分析（{rel}）", hit)
        image_url: dict = {"url": "data:" + mime + ";base64,"
                          + base64.b64encode(data).decode("ascii")}
        if detail:
            image_url["detail"] = detail
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": image_url},
            {"type": "text", "text": question},
        ]}]
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                resp = await chat_completion(
                    client, base_url=cfg.base_url, api_key=cfg.api_key,
                    model=cfg.model, messages=messages,
                    attempts=cfg.attempts, sleep_429=cfg.sleep_429,
                    sleep_err=cfg.sleep_err, log_name="analyze-image")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                False, "模型请求失败",
                f"图像分析请求失败：{exc}。可稍后重试，或先用 image_info "
                f"查看确定元信息。")
        answer = strip_think(first_content(resp))
        if not answer:
            return ToolResult(False, "空回复",
                              "图像模型未返回内容，请稍后重试或换一个问题。")
        cache[key] = answer
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return ToolResult(True, f"分析 {rel}", answer)

    registry.register(
        ToolSpec("analyze_image",
                 "用视觉模型分析工作区内一张图片，回答关于它的具体问题"
                 "（如：图中画了什么/数据走势/文字内容）。每次调用是一次"
                 "独立的视觉问答；相同图片与问题会命中缓存。确定性问题"
                 "（尺寸/dpi/格式）请先用 image_info。",
                 _analyze_params(), ToolCategory.READ,
                 timeout=analyze_timeout),
        analyze_image)
