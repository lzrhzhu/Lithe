"""lithe.bundles.documents: tool-tier document (PDF/OOXML) perception.

Pinning (a) the stdlib-only probe (PDF magic + version + page estimate,
OOXML subtype via the zip directory / suffix fallback), (b) the
analyze_document wire shapes per dialect (inline-file block vs files-api
file_id block), (c) the (hash, question, format) answer cache, and (d) the
degradation paths — non-document, oversized, upload failure, 400 dialect
hint, unconfigured registration. All model calls are faked; no network."""
from __future__ import annotations

import io
import zipfile

import httpx
import pytest

import lithe.bundles.documents as documents_mod
from lithe import AgentContext, LLMConfig, ToolRegistry
from lithe.bundles.documents import probe_document, register_document_tools
from lithe.bundles.workspace import Workspace


# --- hand-built minimal documents (probe never parses page content) ---

def _pdf(pages: int = 3, version: bytes = b"1.4") -> bytes:
    body = b"%PDF-" + version + b"\n"
    for _ in range(pages):
        body += b"<< /Type /Page /MediaBox [0 0 595 842] >>\n"
    body += b"<< /Type /Pages /Kids [] /Count 0 >>\n"
    return body + b"%%EOF"


def _oozip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _ctx() -> AgentContext:
    return AgentContext(run_id="r", user_id="u")


# --- probe_document ---

def test_probe_pdf():
    info = probe_document(_pdf(pages=5, version=b"1.7"))
    assert info["format"] == "pdf" and info["version"] == "1.7"


def test_probe_ooxml_subtypes():
    assert probe_document(
        _oozip({"word/document.xml": b"<w/>"}), "a.docx")["format"] == "docx"
    assert probe_document(
        _oozip({"xl/workbook.xml": b"<x/>"}), "a.xlsx")["format"] == "xlsx"
    assert probe_document(
        _oozip({"ppt/presentation.xml": b"<p/>"}), "a.pptx")["format"] == "pptx"


def test_probe_ooxml_suffix_fallback_on_truncated_zip():
    # Only the local-file header survives the window: the zip directory is
    # out of reach, so the suffix speaks.
    data = _oozip({"word/document.xml": b"<w/>", "junk": b"z" * 4096})
    assert probe_document(data[:64], "report.docx")["format"] == "docx"
    # and without a telling suffix it stays unknown
    assert probe_document(data[:64], "blob")["format"] is None


def test_probe_unknown():
    assert probe_document(b"plain text, not a document")["format"] is None
    assert probe_document(b"")["format"] is None


def test_pdf_page_estimate_excludes_pages_tree():
    from lithe.bundles.documents import _pdf_page_estimate
    assert _pdf_page_estimate(_pdf(pages=3)) == 3  # /Type /Pages not counted


# --- document_info tool ---

async def test_document_info_roundtrip(tmp_path):
    ws = Workspace(tmp_path)
    ws.write_bytes("report.pdf", _pdf(pages=3, version=b"1.4"))
    reg = ToolRegistry()
    register_document_tools(reg, lambda ctx: ws)
    res = await reg.dispatch("document_info", {"path": "report.pdf"}, _ctx())
    assert res.ok
    assert "PDF" in res.content and "1.4" in res.content
    assert "3 页" in res.content


async def test_document_info_ooxml_and_errors(tmp_path):
    ws = Workspace(tmp_path)
    ws.write_bytes("a.docx", _oozip({"word/document.xml": b"<w/>"}))
    ws.write_bytes("x.txt", b"plain text")
    reg = ToolRegistry()
    register_document_tools(reg, lambda ctx: ws)
    ok = await reg.dispatch("document_info", {"path": "a.docx"}, _ctx())
    assert ok.ok and "DOCX" in ok.content
    missing = await reg.dispatch("document_info", {"path": "nope.pdf"}, _ctx())
    assert not missing.ok and "不存在" in missing.content
    nontdoc = await reg.dispatch("document_info", {"path": "x.txt"}, _ctx())
    assert not nontdoc.ok and "文档格式" in nontdoc.content
    escape = await reg.dispatch("document_info", {"path": "../etc"}, _ctx())
    assert not escape.ok


# --- analyze_document tool ---

def _fake_chat(calls: list):
    async def fake(client, *, base_url, api_key, model, messages, **kw):
        calls.append({"base_url": base_url, "model": model,
                      "messages": messages})
        return {"choices": [{"message": {
            "content": "<think>hidden</think>文档核心结论在第二节。"}}]}
    return fake


def _registry_with_documents(ws, monkeypatch, calls, fmt, **kw):
    monkeypatch.setattr(documents_mod, "chat_completion", _fake_chat(calls))
    reg = ToolRegistry()
    register_document_tools(reg, lambda ctx: ws, llm_config=LLMConfig(
        model="doc-model", base_url="http://relay.test/v1",
        api_key="k"), document_format=fmt, **kw)
    return reg


async def test_analyze_document_inline_file_wire_shape(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    ws.write_bytes("plan.pdf", _pdf(pages=1))
    calls: list = []
    reg = _registry_with_documents(ws, monkeypatch, calls, "inline-file")
    res = await reg.dispatch(
        "analyze_document", {"path": "plan.pdf", "question": "结论是什么"},
        _ctx())
    assert res.ok
    assert res.content == "文档核心结论在第二节。"  # <think> stripped
    sent = calls[0]["messages"][0]
    assert sent["role"] == "user"
    assert [b["type"] for b in sent["content"]] == ["file", "text"]
    block = sent["content"][0]["file"]
    assert block["filename"] == "plan.pdf"
    assert block["file_data"].startswith("data:application/pdf;base64,")
    assert sent["content"][1]["text"] == "结论是什么"
    assert calls[0]["model"] == "doc-model"


async def test_analyze_document_files_api_wire_shape(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    ws.write_bytes("plan.pdf", _pdf(pages=1))
    calls: list = []

    async def fake_upload(client, cfg, filename, data, mime):
        assert filename == "plan.pdf" and mime == "application/pdf"
        return "file-abc123"

    monkeypatch.setattr(documents_mod, "_upload_file", fake_upload)
    reg = _registry_with_documents(ws, monkeypatch, calls, "files-api")
    res = await reg.dispatch(
        "analyze_document", {"path": "plan.pdf", "question": "?"}, _ctx())
    assert res.ok
    block = calls[0]["messages"][0]["content"][0]
    assert block == {"type": "file", "file": {"file_id": "file-abc123"}}


async def test_analyze_document_upload_failure_hints_inline(tmp_path,
                                                             monkeypatch):
    ws = Workspace(tmp_path)
    ws.write_bytes("plan.pdf", _pdf(pages=1))
    calls: list = []

    async def bad_upload(client, cfg, filename, data, mime):
        raise httpx.HTTPStatusError(
            "404", request=httpx.Request("POST", "http://relay.test/v1/files"),
            response=httpx.Response(404))

    monkeypatch.setattr(documents_mod, "_upload_file", bad_upload)
    reg = _registry_with_documents(ws, monkeypatch, calls, "files-api")
    res = await reg.dispatch(
        "analyze_document", {"path": "plan.pdf", "question": "?"}, _ctx())
    assert not res.ok and "上传失败" in res.summary
    assert "inline-file" in res.content
    assert calls == []  # no chat call happened


async def test_analyze_document_400_hints_dialect(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    ws.write_bytes("plan.pdf", _pdf(pages=1))
    req = httpx.Request("POST", "http://relay.test/v1/chat/completions")

    async def rejecting(client, **kw):
        raise httpx.HTTPStatusError("400 Bad Request",
                                    request=req, response=httpx.Response(400))

    monkeypatch.setattr(documents_mod, "chat_completion", rejecting)
    reg = ToolRegistry()
    register_document_tools(reg, lambda ctx: ws, llm_config=LLMConfig(
        model="m", base_url="http://relay.test/v1", api_key="k"),
        document_format="inline-file")
    res = await reg.dispatch(
        "analyze_document", {"path": "plan.pdf", "question": "?"}, _ctx())
    assert not res.ok
    assert "document_format" in res.content


async def test_analyze_document_cache(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    ws.write_bytes("plan.pdf", _pdf(pages=1))
    calls: list = []
    reg = _registry_with_documents(ws, monkeypatch, calls, "inline-file")
    same = {"path": "plan.pdf", "question": "结论是什么"}
    r1 = await reg.dispatch("analyze_document", same, _ctx())
    r2 = await reg.dispatch("analyze_document", same, _ctx())
    assert len(calls) == 1  # second hit served from cache
    assert r1.content == r2.content
    await reg.dispatch(
        "analyze_document", {"path": "plan.pdf", "question": "换个问题"}, _ctx())
    assert len(calls) == 2  # new question misses


async def test_analyze_document_guards(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    ws.write_bytes("big.pdf", _pdf(pages=1))
    ws.write_bytes("x.txt", b"text")
    calls: list = []
    reg = _registry_with_documents(ws, monkeypatch, calls, "inline-file",
                                   max_document_bytes=8)
    big = await reg.dispatch(
        "analyze_document", {"path": "big.pdf", "question": "?"}, _ctx())
    assert not big.ok and "文档过大" in big.summary
    nondoc = await reg.dispatch(
        "analyze_document", {"path": "x.txt", "question": "?"}, _ctx())
    assert not nondoc.ok and "非文档" in nondoc.summary
    assert "格式" in nondoc.content
    noq = await reg.dispatch("analyze_document", {"path": "big.pdf"}, _ctx())
    assert not noq.ok
    assert calls == []  # every guard fired before any HTTP request


# --- registration gates ---

async def test_unconfigured_registration_is_probe_only(tmp_path):
    ws = Workspace(tmp_path)
    reg = ToolRegistry()
    register_document_tools(reg, lambda ctx: ws)  # no llm_config
    assert "document_info" in reg.names()
    assert "analyze_document" not in reg.names()
    reg2 = ToolRegistry()
    register_document_tools(reg2, lambda ctx: ws, llm_config=LLMConfig(
        model="m", base_url="http://x/v1", api_key="k"),
        document_format="none")
    assert "analyze_document" not in reg2.names()


def test_unknown_document_format_raises(tmp_path):
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="document_format"):
        register_document_tools(
            reg, lambda ctx: Workspace(tmp_path), llm_config=LLMConfig(
                model="m", base_url="http://x/v1", api_key="k"),
            document_format="anthropic-document")
