# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


EXAMPLE_DIR = Path(__file__).parents[2] / "examples" / "disagg_prefill"
sys.path.insert(0, str(EXAMPLE_DIR))

import disagg_proxy_request as request_helpers  # noqa: E402


build_phase_requests = request_helpers.build_phase_requests


@pytest.mark.parametrize(
    "request_data",
    [
        {"tools": []},
        {"tools": [], "tool_choice": None},
        {"tools": [], "tool_choice": "none"},
    ],
)
def test_normalize_chat_request_removes_semantically_empty_tools(request_data):
    request_data.update(
        {
            "model": "MiniMax-M2.7",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    original = deepcopy(request_data)

    normalized = request_helpers.normalize_chat_request(request_data)

    assert "tools" not in normalized
    assert request_data == original


@pytest.mark.parametrize("tool_choice", ["auto", "required", {"type": "function"}])
def test_normalize_chat_request_preserves_invalid_tool_choice_for_validation(
    tool_choice,
):
    request_data = {"tools": [], "tool_choice": tool_choice}

    normalized = request_helpers.normalize_chat_request(request_data)

    assert normalized == request_data
    assert normalized is not request_data


@pytest.mark.parametrize(
    ("request_data", "resolved_max_tokens", "expected_decode_tokens"),
    [
        ({"max_completion_tokens": 150}, 140, 139),
        ({"max_tokens": 120}, 110, 109),
        ({"max_tokens": 120, "max_completion_tokens": 150}, 140, 139),
    ],
)
def test_chat_budget_is_normalized_for_internal_completion_requests(
    request_data, resolved_max_tokens, expected_decode_tokens
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
        request_data,
        [10, 20, 30],
        is_chat=True,
        resolved_max_tokens=resolved_max_tokens,
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


def test_chat_without_token_budget_uses_rendered_context_limit():
    prefill_request, decode_request = build_phase_requests(
        {"messages": [{"role": "user", "content": "hello"}]},
        [10, 20],
        is_chat=True,
        resolved_max_tokens=196569,
    )

    assert prefill_request["max_tokens"] == 1
    assert decode_request["max_tokens"] == 196568


def test_parse_chat_render_output_returns_tokens_and_effective_budget():
    prompt_token_ids, resolved_max_tokens = request_helpers.parse_chat_render_output(
        {
            "token_ids": [10, 20, 30],
            "sampling_params": {"max_tokens": 196569, "temperature": 1.0},
        }
    )

    assert prompt_token_ids == [10, 20, 30]
    assert resolved_max_tokens == 196569


def test_upstream_service_error_preserves_response_details():
    response = SimpleNamespace(
        status_code=400,
        request=SimpleNamespace(
            url="http://7.150.2.142:7100/v1/chat/completions/render"
        ),
        content=b'{"error":{"message":"Input length exceeds model limit"}}',
        headers={"content-type": "application/json"},
    )

    error = request_helpers.upstream_service_error_from_response(response)

    assert error.status_code == 400
    assert error.url == "http://7.150.2.142:7100/v1/chat/completions/render"
    assert error.body == response.content
    assert error.content_type == "application/json"
    assert "Input length exceeds model limit" in str(error)


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
