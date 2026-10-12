"""Transport-shared contract: usage normalization across gateway shapes
(chat / responses, cached-token details, dirty values) and the
make_transport factory."""
from __future__ import annotations

import pytest

from lithe.transports import (
    ChatCompletionsTransport, ResponsesTransport, make_transport,
)


# -- usage normalization --------------------------------------------------------


def test_norm_usage_maps_responses_shape():
    from lithe.transports import norm_usage

    u = norm_usage(
        {"input_tokens": 7, "output_tokens": 3, "cost_breakdown": {"total_cost": 0.01}}
    )
    assert u["prompt_tokens"] == 7 and u["completion_tokens"] == 3
    assert u["total_tokens"] == 10  # computed when the vendor omits it
    assert u["cost_breakdown"] == {"total_cost": 0.01}  # extras pass through


def test_norm_usage_completes_chat_shape():
    from lithe.transports import norm_usage

    u = norm_usage({"prompt_tokens": 5, "completion_tokens": 2})
    assert u["total_tokens"] == 7
    assert norm_usage(None) == {}
    assert norm_usage({}) == {}


def test_norm_usage_lifts_cached_tokens():
    from lithe.transports import norm_usage

    # chat-completions shape
    u = norm_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 80},
        }
    )
    assert u["cached_tokens"] == 80
    # responses shape
    u = norm_usage(
        {
            "input_tokens": 50,
            "output_tokens": 2,
            "input_tokens_details": {"cached_tokens": 30},
        }
    )
    assert u["prompt_tokens"] == 50 and u["completion_tokens"] == 2
    assert u["total_tokens"] == 52
    assert u["cached_tokens"] == 30
    # absent details → no cached_tokens key (not a zero that lies)
    assert "cached_tokens" not in norm_usage({"input_tokens": 5, "output_tokens": 1})


def test_norm_usage_tolerates_dirty_gateway_values():
    from lithe.transports import norm_usage

    # comma-grouped strings, whitespace, floats-as-str and outright garbage
    # degrade to coerced/0 values instead of raising out of the transport
    u = norm_usage(
        {"prompt_tokens": "1,234", "completion_tokens": " 56 ", "total_tokens": "1,290"}
    )
    assert u["prompt_tokens"] == 1234 and u["completion_tokens"] == 56
    assert u["total_tokens"] == 1290
    u = norm_usage({"input_tokens": "junk", "output_tokens": [3]})
    assert u["prompt_tokens"] == 0 and u["completion_tokens"] == 0
    assert u["total_tokens"] == 0
    u = norm_usage(
        {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": "1,0"},
            "output_tokens_details": {"reasoning_tokens": "x"},
        }
    )
    assert u["cached_tokens"] == 10 and u["reasoning_tokens"] == 0


# -- factory ------------------------------------------------------------------


def test_make_transport_builtins_and_custom():
    assert isinstance(make_transport("chat"), ChatCompletionsTransport)
    assert isinstance(make_transport("responses"), ResponsesTransport)
    custom = ChatCompletionsTransport()
    assert make_transport(custom) is custom
    with pytest.raises(ValueError):
        make_transport("bogus")
