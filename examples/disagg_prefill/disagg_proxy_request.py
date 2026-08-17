# SPDX-License-Identifier: Apache-2.0

from typing import Any


DEFAULT_COMPLETION_MAX_TOKENS = 16


class InvalidTokenBudget(ValueError):
    """Raised when a request does not provide a usable output-token budget."""


def build_phase_requests(
    request_data: dict[str, Any],
    prompt_token_ids: list[int],
    *,
    is_chat: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build independent Completion payloads for the prefill and decode phases."""
    normalized = request_data.copy()

    if is_chat:
        max_tokens = normalized.pop("max_completion_tokens", None)
        if max_tokens is None:
            max_tokens = normalized.get("max_tokens")
        if max_tokens is None:
            raise InvalidTokenBudget(
                "Chat Completions requests must provide max_completion_tokens "
                "or max_tokens"
            )
    else:
        max_tokens = normalized.get("max_tokens", DEFAULT_COMPLETION_MAX_TOKENS)

    stream_options = normalized.pop("stream_options", None)
    normalized["prompt"] = list(prompt_token_ids)

    prefill_request = normalized.copy()
    prefill_request["prompt"] = list(prompt_token_ids)
    prefill_request["max_tokens"] = 1
    prefill_request["stream"] = False

    decode_request = normalized.copy()
    decode_request["prompt"] = list(prompt_token_ids)
    decode_request["max_tokens"] = max_tokens - 1
    decode_request["stream"] = True
    if stream_options is not None:
        decode_request["stream_options"] = stream_options

    return prefill_request, decode_request
