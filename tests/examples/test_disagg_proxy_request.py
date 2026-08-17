# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import Path
import sys

import pytest


EXAMPLE_DIR = Path(__file__).parents[2] / "examples" / "disagg_prefill"
sys.path.insert(0, str(EXAMPLE_DIR))

from disagg_proxy_request import (  # noqa: E402
    InvalidTokenBudget,
    build_phase_requests,
)


@pytest.mark.parametrize(
    ("request_data", "expected_decode_tokens"),
    [
        ({"max_completion_tokens": 150}, 149),
        ({"max_tokens": 120}, 119),
        ({"max_tokens": 120, "max_completion_tokens": 150}, 149),
    ],
)
def test_chat_budget_is_normalized_for_internal_completion_requests(
    request_data, expected_decode_tokens
):
    request_data.update(
        {
            "model": "MiniMax-M2.7",
            "messages": [{"role": "user", "content": "hello"}],
            "stream_options": {"include_usage": True},
            "temperature": 0.7,
        }
    )
    original = deepcopy(request_data)

    prefill_request, decode_request = build_phase_requests(
        request_data, [10, 20, 30], is_chat=True
    )

    assert request_data == original
    assert prefill_request["max_tokens"] == 1
    assert decode_request["max_tokens"] == expected_decode_tokens
    assert "max_completion_tokens" not in prefill_request
    assert "max_completion_tokens" not in decode_request
    assert "stream_options" not in prefill_request
    assert decode_request["stream_options"] == {"include_usage": True}
    assert prefill_request["stream"] is False
    assert decode_request["stream"] is True
    assert prefill_request["temperature"] == 0.7
    assert decode_request["temperature"] == 0.7

    decode_request["prompt"].append(40)
    assert prefill_request["prompt"] == [10, 20, 30]
    assert decode_request["prompt"] == [10, 20, 30, 40]


def test_chat_without_token_budget_is_rejected():
    with pytest.raises(InvalidTokenBudget, match="max_completion_tokens.*max_tokens"):
        build_phase_requests(
            {"messages": [{"role": "user", "content": "hello"}]},
            [10, 20],
            is_chat=True,
        )


def test_completion_without_max_tokens_uses_vllm_default():
    request_data = {
        "model": "MiniMax-M2.7",
        "prompt": "hello",
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    original = deepcopy(request_data)

    prefill_request, decode_request = build_phase_requests(
        request_data, [10, 20], is_chat=False
    )

    assert request_data == original
    assert prefill_request == {
        "model": "MiniMax-M2.7",
        "prompt": [10, 20],
        "ignore_eos": True,
        "max_tokens": 1,
        "stream": False,
    }
    assert decode_request == {
        "model": "MiniMax-M2.7",
        "prompt": [10, 20],
        "ignore_eos": True,
        "max_tokens": 15,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
