"""Shared helpers: a scripted client and builders for OpenRouter-shaped responses.

Helpers and opt-in fixtures only; nothing here is autouse.
"""

import copy
import json
import time

import pytest

# A real usage payload and its response metadata, from `.experiments/output/usage_check.txt` (turn 2).
REAL_USAGE = {
    "prompt_tokens": 1433,
    "completion_tokens": 2,
    "total_tokens": 1435,
    "cost": 2.799e-05,
    "is_byok": False,
    "prompt_tokens_details": {"cached_tokens": 1280, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0},
    "cost_details": {
        "upstream_inference_cost": 2.799e-05,
        "upstream_inference_prompt_cost": 2.679e-05,
        "upstream_inference_completions_cost": 1.2e-06,
    },
    "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 0, "audio_tokens": 0},
}
REAL_META = {
    "id": "gen-1789748403-OPdrI8Y5SuMpxM9FVbYd",
    "provider": "DeepSeek",
    "model": "deepseek/deepseek-v4.1-flash",
    "created": 1789748403,
}


class FakeClient:
    """Returns scripted responses in order, like `ChatClient.complete`; a scripted exception is raised instead.

    `requests` keeps a deep copy of every request. `credits()` returns `credit_data`, or raises it if it's an exception.
    """

    def __init__(self, responses, credit_data=None):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.credit_data = credit_data

    def complete(self, request: dict) -> dict:
        self.requests.append(copy.deepcopy(request))
        if not self.responses:
            raise AssertionError(f"no scripted response left for request {len(self.requests)}")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)

    def credits(self) -> dict:
        if isinstance(self.credit_data, Exception):
            raise self.credit_data
        return self.credit_data


def usage(prompt=100, completion=10, cached=0, reasoning=0, cost=0.001) -> dict:
    """A usage payload in the real OpenRouter shape, with the given counts."""
    u = copy.deepcopy(REAL_USAGE)
    u.update(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion, cost=cost)
    u["prompt_tokens_details"]["cached_tokens"] = cached
    u["completion_tokens_details"]["reasoning_tokens"] = reasoning
    u["cost_details"]["upstream_inference_cost"] = cost
    return u


default_usage = usage  # `response()` takes a `usage` argument, which hides the function
_ids = iter(range(1, 1_000_000))


def response(content="", tool_calls=None, usage=None, id=None, created=1789748403, no_usage=False) -> dict:
    """A whole non-streaming response. `usage` defaults to `usage()`; `no_usage=True` leaves the key out."""
    message = {"role": "assistant", "content": content, "refusal": None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    r = {
        "id": id or f"gen-{next(_ids)}",
        "provider": "DeepSeek",
        "model": "deepseek/deepseek-v4.1-flash",
        "created": created,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
    }
    if not no_usage:
        r["usage"] = usage if usage is not None else default_usage()
    return r


def call(name: str, id: str | None = None, **args) -> dict:
    """A tool call with JSON-string arguments, as the API sends it."""
    return {"id": id or f"call-{next(_ids)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def utc(monkeypatch):
    """Format local times in UTC for the test, then restore the process's time zone."""
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()
