"""Documents bundle: tool-tier document (PDF/OOXML) perception, no kernel
changes.

Two tools, same "perception stays in tools" design as the images bundle:

- ``document_info`` — zero-model, stdlib-only probe: format (PDF / DOCX /
  XLSX / PPTX), PDF version, best-effort page count, size. Answers every
  deterministic question with no model call at all.
- ``analyze_document`` — the only document-perception path: reads the file,
  sends it as ONE content block (plus the question) in a single chat call,
  and returns the answer text. The document itself never enters the main
  conversation, so the kernel's context budget / trimming / replay
  machinery is untouched; answers are memoized per
  (file hash, question, format) in-process.

Document block dialects (``document_format``) — the wire shape varies by
gateway family, and the format follows what the *gateway* accepts, not
what the URL looks like:

- ``"inline-file"`` — OpenRouter family (and most OpenAI-compatible relays
  fronting multimodal models): one content part
  ``{"type": "file", "file": {"filename": ..., "file_data":
  "data:<mime>;base64,..."}}`` in a single chat call. Self-built routers
  speaking the OpenRouter format behind their own base_url stay here.
- ``"files-api"`` — strict OpenAI chat-completions: upload the file to
  ``{base_url}/files`` first (multipart, purpose ``user_data``), then send
  ``{"type": "file", "file": {"file_id": ...}}``. Two requests.
- ``"none"`` (or omitting ``document_format`` / ``llm_config``) — registers
  only the deterministic probe.

The provider preset (``lithe.bundles.providers``) carries each family's
default ``document_format``; a profile field or registration argument
overrides it — presets are defaults under overrides, never truth. A 400
from the endpoint surfaces with a hint pointing at a dialect mismatch,
mirroring ``extra_body_hint`` in ``lithe.llm``: the model self-corrects by
switching format or falling back to ``run_code`` text extraction.

Optional bundle — stdlib + httpx only (like the rest of lithe).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import re
import zipfile
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import httpx

from lithe.context import AgentContext
from lithe.llm import chat_completion, first_content, strip_think
from lithe.runtime import LLMConfig
from lithe.tools import ToolCategory, ToolRegistry, ToolResult, ToolSpec
from lithe.bundles.workspace import Workspace

DEFAULT_MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
DEFAULT_CACHE_SIZE = 16
DEFAULT_ANALYZE_TIMEOUT = 180.0
# document_info reads at most this much for probing: magic bytes and PDF
# page objects sit scattered through the body, so a generous window keeps
# the probe informative on big files without an unbounded read. Files up to
# _PROBE_FULL_LIMIT are read whole (OOXML subtype detection needs the zip
# central directory, which lives at the END of the file).
_PROBE_WINDOW = 256 * 1024
_PROBE_FULL_LIMIT = 8 * 1024 * 1024

_MIME = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation",
}

# Zip-name markers that identify an OOXML package subtype.
_OOXML_MARKERS = (
    ("word/document.xml", "docx"),
    ("xl/workbook.xml", "xlsx"),
    ("ppt/presentation.xml", "pptx"),
)

# Suffix fallback when the zip central directory is outside the probe data.
_OOXML_SUFFIX = {".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx"}

# A PDF page object declares /Type /Page — the negative lookahead keeps
# /Type /Pages (the page-tree node) from counting as a page.
_PDF_PAGE_RE = re.compile(rb"/Type\s*/Page(?![s])")


# --------------------------------------------------------------------------- #
# stdlib-only format probing
# --------------------------------------------------------------------------- #

def _probe_ooxml(data: bytes) -> str | None:
    """Identify the OOXML subtype from the zip central directory.

    Works only when *data* carries the whole package (the directory lives
    at the end); a truncated probe returns ``None`` so the caller falls
    back to the file-name suffix. Extraction never happens — namelist
    reads the directory only, so a crafted archive cannot burn CPU here.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
    except (zipfile.BadZipFile, OSError, ValueError):
        return None
    for marker, subtype in _OOXML_MARKERS:
        if marker in names:
            return subtype
    return None


def probe_document(data: bytes, name: str = "") -> dict:
    """Best-effort header probe: format / PDF version / OOXML subtype.

    Returns ``{"format": "pdf"|"docx"|"xlsx"|"pptx"|None, "version": ...}``.
    ``format`` is ``None`` when the magic bytes match no supported
    container. OOXML subtyping uses the zip directory when fully present,
    else the file-name suffix (``name`` should be the bare file name).
    """
    out: dict = {"format": None, "version": None}
    if len(data) >= 5 and data[:5] == b"%PDF-":
        out["format"] = "pdf"
        raw = data[5:8].decode("ascii", "replace").strip()
        if re.fullmatch(r"\d\.\d", raw):
            out["version"] = raw
        return out
    if len(data) >= 4 and data[:4] == b"PK\x03\x04":
        subtype = _probe_ooxml(data)
        if subtype is None:
            subtype = _OOXML_SUFFIX.get(Path(name or "").suffix.lower())
        if subtype:
            out["format"] = subtype
    return out


def _pdf_page_estimate(data: bytes) -> int:
    """Count ``/Type /Page`` declarations — an estimate, not a parse:
    compressed object streams hide their pages from a raw scan, so the
    number is a lower bound. Callers label it as such."""
    return len(_PDF_PAGE_RE.findall(data))


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
        "description": "想从文档中了解什么；越具体越好（如：第二节的核心结论是什么）"}
    p["required"].append("question")
    return p


async def _upload_file(client: httpx.AsyncClient, cfg: LLMConfig,
                       filename: str, data: bytes, mime: str) -> str:
    """POST ``{base_url}/files`` (multipart, purpose ``user_data``) → id.

    The strict-OpenAI two-step: the returned id goes into the chat call's
    ``file_id`` block. multipart bodies pick their own content-type, so
    only the bearer header is set here (a JSON content-type would corrupt
    the upload).
    """
    resp = await client.post(
        f"{cfg.base_url}/files",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        data={"purpose": "user_data"},
        files={"file": (filename, data, mime)},
    )
    resp.raise_for_status()
    payload = resp.json()
    fid = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(fid, str) or not fid:
        raise ValueError(f"/files 响应缺少 id：{str(payload)[:200]}")
    return fid


def register_document_tools(
    registry: ToolRegistry,
    workspace_for: Callable[[AgentContext], Workspace],
    *,
    llm_config: LLMConfig | None = None,
    document_format: str | None = None,
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
    cache_size: int = DEFAULT_CACHE_SIZE,
    analyze_timeout: float = DEFAULT_ANALYZE_TIMEOUT,
) -> None:
    """Register ``document_info`` (always) and ``analyze_document`` (when a
    model call is both configured and possible).

    ``document_format`` selects the wire dialect (see the module docstring):
    ``"inline-file"`` / ``"files-api"`` enable ``analyze_document`` with the
    matching content-block shape; ``"none"`` / ``None`` register only the
    deterministic probe. When the argument is ``None`` it falls back to
    ``llm_config.document_format`` — the provider preset contributes that
    default, so a host passing its main ``LLMConfig`` needs no extra wiring.
    An unrecognized value raises :class:`ValueError`: a config typo must
    fail loudly at startup, not as a mystery 400 later.
    """
    raw = document_format if document_format is not None \
        else (getattr(llm_config, "document_format", None) or "")
    fmt = raw.strip().lower()
    if fmt not in ("", "none", "inline-file", "files-api"):
        raise ValueError(
            f"未知 document_format：{raw!r}"
            f"（可用：inline-file / files-api / none）")

    cache: OrderedDict[tuple, str] = OrderedDict()

    async def document_info(ctx: AgentContext, args: dict) -> ToolResult:
        rel = (args.get("path") or "").strip()
        if not rel:
            return ToolResult(False, "缺少 path", "缺少 path 参数。")
        err, target = _resolve(workspace_for(ctx), rel)
        if err is not None:
            return err
        size = (await asyncio.to_thread(target.stat)).st_size
        # Whole-file read only under the probe limit (OOXML subtyping needs
        # the tail-side zip directory); bigger files get a head window that
        # still carries the magic bytes and any early PDF objects.
        if size <= _PROBE_FULL_LIMIT:
            with target.open("rb") as fh:
                data = await asyncio.to_thread(fh.read)
        else:
            with target.open("rb") as fh:
                data = await asyncio.to_thread(fh.read, _PROBE_WINDOW)
        info = probe_document(data, target.name)
        if info["format"] is None:
            return ToolResult(
                False, "非文档文件",
                f"{rel} 不是可识别的文档格式（PDF/DOCX/XLSX/PPTX），"
                f"共 {_human_size(size)}。文本提取可用 run_code。")
        fmt_name = info["format"].upper()
        bits = [fmt_name]
        if info["version"]:
            bits.append(info["version"])
        if info["format"] == "pdf" and size <= _PROBE_FULL_LIMIT:
            pages = _pdf_page_estimate(data)
            if pages:
                bits.append(f"约 {pages} 页（扫描估计）")
        body = f"{rel}：" + "，".join(bits) + f"，{_human_size(size)}。"
        return ToolResult(True, f"探测 {rel}", body)

    registry.register(
        ToolSpec("document_info",
                 "探测工作区文档文件的确定信息：格式（PDF/DOCX/XLSX/PPTX）、"
                 "PDF 版本、页数（尽力估计）、大小。不调用模型；回答“这个文件"
                 "是什么格式/多少页”这类问题应优先用它，需要理解内容时再用 "
                 "analyze_document。",
                 _path_params(), ToolCategory.READ),
        document_info)

    if llm_config is None or fmt in ("", "none"):
        return
    cfg = llm_config

    async def analyze_document(ctx: AgentContext, args: dict) -> ToolResult:
        rel = (args.get("path") or "").strip()
        question = (args.get("question") or "").strip()
        if not rel:
            return ToolResult(False, "缺少 path", "缺少 path 参数。")
        if not question:
            return ToolResult(False, "缺少 question",
                              "analyze_document 需要一个明确的问题"
                              "（想从文档中了解什么）。")
        err, target = _resolve(workspace_for(ctx), rel)
        if err is not None:
            return err
        # Size check BEFORE reading: a 2GB upload must be refused by stat,
        # not by loading 2GB into memory first.
        size = (await asyncio.to_thread(target.stat)).st_size
        if size > max_document_bytes:
            return ToolResult(
                False, "文档过大",
                f"{rel} 为 {_human_size(size)}，超过上限 "
                f"{_human_size(max_document_bytes)}。请先用 run_code 提取或"
                f"拆分出需要的部分再分析。")
        data = await asyncio.to_thread(target.read_bytes)
        info = probe_document(data, target.name)
        mime = _MIME.get(info["format"] or "")
        if mime is None:
            return ToolResult(
                False, "非文档文件",
                f"{rel} 不是受支持的文档格式（PDF/DOCX/XLSX/PPTX）。"
                f"文本提取可用 run_code。")
        key = (hashlib.sha256(data).hexdigest(), question, fmt)
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return ToolResult(True, f"已缓存的分析（{rel}）", hit)
        name = target.name
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                if fmt == "files-api":
                    try:
                        fid = await _upload_file(client, cfg, name, data, mime)
                    except Exception as exc:  # noqa: BLE001
                        return ToolResult(
                            False, "文档上传失败",
                            f"向 {cfg.base_url}/files 上传失败：{exc}。"
                            f"该网关可能不支持 files-api 两步上传，可尝试将 "
                            f"document_format 改为 inline-file，或用 run_code "
                            f"提取文本。")
                    block: dict = {"type": "file",
                                   "file": {"file_id": fid}}
                else:  # inline-file
                    block = {"type": "file",
                             "file": {"filename": name,
                                      "file_data": "data:" + mime + ";base64,"
                                      + base64.b64encode(data).decode("ascii")}}
                messages = [{"role": "user", "content": [
                    block, {"type": "text", "text": question}]}]
                try:
                    resp = await chat_completion(
                        client, base_url=cfg.base_url, api_key=cfg.api_key,
                        model=cfg.model, messages=messages,
                        attempts=cfg.attempts, sleep_429=cfg.sleep_429,
                        sleep_err=cfg.sleep_err, log_name="analyze-document")
                except Exception as exc:  # noqa: BLE001
                    hint = ""
                    if (isinstance(exc, httpx.HTTPStatusError)
                            and exc.response.status_code == 400):
                        hint = (f"；该端点可能不接受当前文档块方言"
                                f"（document_format={fmt}，可选 "
                                f"inline-file / files-api），可在配置中改写后"
                                f"重试")
                    return ToolResult(
                        False, "模型请求失败",
                        f"文档分析请求失败：{exc}{hint}。可稍后重试，或先用 "
                        f"document_info 查看确定元信息，或用 run_code 提取文本。")
        except Exception as exc:  # noqa: BLE001 — client construction etc.
            return ToolResult(
                False, "模型请求失败", f"文档分析请求失败：{exc}。可稍后重试。")
        answer = strip_think(first_content(resp))
        if not answer:
            return ToolResult(False, "空回复",
                              "模型未返回内容，请稍后重试或换一个问题。")
        cache[key] = answer
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return ToolResult(True, f"分析 {rel}", answer)

    registry.register(
        ToolSpec("analyze_document",
                 "让模型直接阅读工作区内一个文档（PDF/DOCX/XLSX/PPTX）并回答"
                 "关于它的具体问题（如：核心结论/第几页讲了什么/表格数据）。"
                 "每次调用是一次独立的文档问答；相同文档与问题会命中缓存。"
                 "需要端点支持文档输入；确定性信息（格式/页数）请先用 "
                 "document_info，纯文本提取可用 run_code。",
                 _analyze_params(), ToolCategory.READ,
                 timeout=analyze_timeout),
        analyze_document)
