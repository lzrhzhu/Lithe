"""lithe.bundles.providers: preset lookup and preset-under-overrides merge.
Pure data + pure logic; no I/O anywhere."""
from __future__ import annotations

import pytest

from lithe.bundles.providers import (
    PRESETS, apply_preset, get_preset, known_providers,
)


def test_known_providers_cover_documented_set():
    names = known_providers()
    for expected in ("openai", "anthropic", "zai", "deepseek", "openrouter",
                     "qwen", "moonshot"):
        assert expected in names
    assert names == sorted(names)


def test_get_preset_returns_config_keys_only():
    preset = get_preset("zai")
    assert preset["base_url"].startswith("https://")
    assert preset["transport"] == "chat"
    assert "notes" not in preset, "notes 是给人看的，不进 LLMConfig"


def test_anthropic_preset_selects_messages_transport():
    """厂商→协议映射只发生在 preset 的 transport 字段：anthropic 厂商
    用 messages 协议，档案里选了它 LLMConfig 就直接可跑。"""
    kwargs = apply_preset("anthropic", model="claude-x", api_key="sk-x")
    assert kwargs["base_url"] == "https://api.anthropic.com/v1"
    assert kwargs["transport"] == "messages"
    from lithe import LLMConfig
    from lithe.transports import MessagesTransport, make_transport
    cfg = LLMConfig(**kwargs)
    assert isinstance(make_transport(cfg.transport), MessagesTransport)


def test_get_preset_unknown_raises_with_known_list():
    with pytest.raises(ValueError, match="openai"):
        get_preset("nonexistent")
    with pytest.raises(ValueError):
        get_preset("")
    # 名字大小写与空白宽容，避免无谓的失败
    assert get_preset(" ZAI ")["transport"] == "chat"


def test_apply_preset_scalar_override_wins():
    merged = apply_preset("zai", base_url="https://my-proxy.example/v1",
                          model="glm-4.6", api_key="sk-x")
    assert merged["base_url"] == "https://my-proxy.example/v1"
    assert merged["model"] == "glm-4.6" and merged["api_key"] == "sk-x"
    assert "notes" not in merged


def test_apply_preset_dict_fields_merge_per_key():
    # preset 带字典字段的合并语义：覆盖按键合并，不清空其余键
    PRESETS["demo"] = {"base_url": "https://demo.example/v1",
                       "extra_body": {"keep": 1, "over": 1}}
    try:
        merged = apply_preset("demo", extra_body={"over": 2, "added": 3})
        assert merged["extra_body"] == {"keep": 1, "over": 2, "added": 3}
    finally:
        del PRESETS["demo"]


def test_apply_preset_skips_none_overrides():
    merged = apply_preset("openai", model=None, api_key=None,
                          base_url="https://api.openai.com/v1")
    assert "model" not in merged and "api_key" not in merged
    assert merged["base_url"] == "https://api.openai.com/v1"


def test_apply_preset_feeds_llmconfig():
    from lithe import LLMConfig

    kwargs = apply_preset("zai", model="glm-4.6", api_key="sk-x")
    cfg = LLMConfig(**kwargs)
    assert cfg.model == "glm-4.6"
    assert cfg.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert cfg.transport == "chat"
