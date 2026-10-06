"""Usage normalization and cost accounting helpers for the runtime."""

from __future__ import annotations


def as_int(val) -> int:
    """Coerce a possibly dirty usage value; failures count as zero."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def gateway_cost(data: dict) -> float | None:
    """Return provider-reported cost, preserving the distinction from absent."""
    usage = data.get("usage") or {}
    breakdown = usage.get("cost_breakdown") or {}
    for src in (breakdown.get("total_cost"), usage.get("cost"), data.get("cost")):
        if src is not None:
            try:
                return float(src)
            except (TypeError, ValueError):
                continue
    return None


def computed_cost(pricing: dict | None, usage: dict) -> float:
    """Calculate cost from per-million-token prices, including cached input."""
    if not pricing:
        return 0.0
    prompt_price = pricing.get("prompt")
    completion_price = pricing.get("completion")
    if prompt_price is None and completion_price is None:
        return 0.0
    prompt_tokens = as_int(usage.get("prompt_tokens"))
    completion_tokens = as_int(usage.get("completion_tokens"))
    cached_tokens = min(as_int(usage.get("cached_tokens")), prompt_tokens)
    cost = 0.0
    if prompt_price is not None:
        cached_price = pricing.get("cached_prompt", prompt_price)
        cost += ((prompt_tokens - cached_tokens) / 1e6) * prompt_price
        cost += (cached_tokens / 1e6) * cached_price
    if completion_price is not None:
        cost += (completion_tokens / 1e6) * completion_price
    return cost


def call_cost(cfg, usage: dict) -> float:
    """Prefer gateway cost, otherwise use the host's configured price table."""
    reported = gateway_cost({"usage": usage})
    if reported is not None:
        return reported
    return computed_cost(cfg.pricing, usage)
