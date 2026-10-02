"""Provider presets: known-good ``LLMConfig`` fields per OpenAI-compatible
vendor, as reusable data.

The kernel deliberately knows nothing about vendors — ``extra_body`` /
``default_headers`` are honest pipes, and deciding *which* fields a gateway
understands is the host's job. This bundle is where that job is done once,
for every host: each preset is a plain dict of ``LLMConfig`` constructor
kwargs (endpoint credentials excluded), maintained as community knowledge
that can expire — presets are **defaults under overrides, never truth**.

What a preset deliberately does and does not carry:

- ``base_url`` — the vendor's usual chat-completions root (the kernel
  appends ``/chat/completions``; ``responses`` presets carry the full
  endpoint URL instead, matching ``ResponsesTransport``).
- ``transport`` — usually ``"chat"``; a ``"responses"`` preset names it.
- ``extra_body`` / ``default_headers`` — only fields that are safe for
  *every* model behind that endpoint. Model-specific switches (Qwen's
  ``enable_thinking``, GLM's ``thinking`` object, OpenRouter routing) stay
  in ``notes``: hard-coding them would break the other half of the models.
- ``pricing`` — omitted: prices vary per model, not per provider; set it
  on the profile that also pins the model.

Usage::

    from lithe.bundles.providers import apply_preset

    kwargs = apply_preset("zai", model="glm-4.6", api_key=KEY,
                          extra_body={"thinking": {"type": "enabled"}})
    cfg = LLMConfig(**kwargs)
"""
from __future__ import annotations

from typing import Any

# Keys that are LLMConfig fields vs. this module's own metadata.
_PRESET_META_KEYS = ("notes",)
_PRESET_CONFIG_KEYS = ("base_url", "transport", "extra_body",
                       "default_headers", "pricing")

PRESETS: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "transport": "chat",
        "notes": "官方端点对未知参数严格：extra_body 里只放确定支持的字段"
                 "（top_p/seed/response_format 等）。推理模型可改用 responses "
                 "transport（base_url 需为完整 /responses 端点 URL）。",
    },
    "zai": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "transport": "chat",
        "notes": "智谱开放平台。GLM 思考模型的 thinking 开关（如 "
                 '{"thinking": {"type": "enabled"}}）经 extra_body 按需开启；'
                 "非思考模型不要带。",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "transport": "chat",
        "notes": "deepseek-reasoner 的 reasoning_content 是每轮状态，内核仅在"
                 "显示层使用、不回传（chat 协议的既定行为），无需配置。",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "transport": "chat",
        "notes": "归因头 HTTP-Referer / X-Title 经 default_headers 自带值设置；"
                 "路由（route/provider 等字段）与用量统计字段按需经 extra_body。",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "transport": "chat",
        "notes": "阿里云百炼 OpenAI 兼容模式。Qwen3 系列的 enable_thinking 等"
                 "开关因模型而异，经 extra_body 按所用模型设置。",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "transport": "chat",
        "notes": "Kimi 系列模型。",
    },
}


def known_providers() -> list[str]:
    """Sorted preset names (for wizards, completion, and error messages)."""
    return sorted(PRESETS)


def get_preset(name: str) -> dict:
    """The preset for *name* (config keys only, ``notes`` stripped).

    Raises :class:`ValueError` for an unknown name — a typo'd provider must
    fail loudly, not silently run without its defaults.
    """
    preset = PRESETS.get((name or "").strip().lower())
    if preset is None:
        raise ValueError(
            f"未知 provider：{name!r}（可用：{', '.join(known_providers())}）")
    return {k: v for k, v in preset.items() if k in _PRESET_CONFIG_KEYS}


def apply_preset(provider: str, **overrides: Any) -> dict[str, Any]:
    """Merge a provider preset under host overrides → ``LLMConfig`` kwargs.

    Scalar fields (``base_url``, ``transport``, ...): an override wins
    outright. Dict fields (``extra_body``, ``default_headers``): merged
    key-wise with the override winning per key, so a preset can contribute
    a baseline and the host add one field without clobbering the rest.
    ``None`` overrides are skipped (unset), and ``notes`` never leaks into
    the result. Credentials (``model`` / ``api_key``) always come from the
    caller — a preset describes an endpoint, not an account.
    """
    merged = get_preset(provider)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            combined = dict(merged[key])
            combined.update(value)
            merged[key] = combined
        else:
            merged[key] = value
    return merged


__all__ = ["PRESETS", "apply_preset", "get_preset", "known_providers"]
