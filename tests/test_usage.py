"""Tests for `usage.py`: the `ModelCall` record, the log and the one-line summary. `AgentNode` is in `test_budget.py`.

Each test names the bug it would catch.
"""

import pytest

from conftest import REAL_META, REAL_USAGE, response, usage
from jean_code.core.llm import parse_completion
from jean_code.core.usage import ModelCall, append_log, human, usage_line


def model_call(**kwargs) -> ModelCall:
    """A booked call with the given usage, as `Agent._book` builds it."""
    return ModelCall.from_completion(parse_completion(response(usage=usage(**kwargs))), path="main")


def real_call(**kwargs) -> ModelCall:
    completion = parse_completion({**response(), **REAL_META, "usage": REAL_USAGE})
    return ModelCall.from_completion(completion, path="main", **kwargs)


# ---------------------------------------------------------------- ModelCall


LOG_LINE = (
    '{"time": "2026-09-18T16:20:03+0000", "generation_id": "gen-1789748403-OPdrI8Y5SuMpxM9FVbYd", '
    '"provider": "DeepSeek", "model": "deepseek/deepseek-v4.1-flash", "prompt_tokens": 1433, '
    '"cached_tokens": 1280, "completion_tokens": 2, "reasoning_tokens": 0, "cost": 2.799e-05, '
    '"budget": "main"}'
)


def test_log_line_of_the_real_payload(utc):
    # Catches a renamed, moved or reordered field in logs/usage.jsonl, which other sessions append to.
    assert real_call().log_line() == LOG_LINE


def test_finish_reason_is_kept_but_not_logged(utc):
    # Catches a reply cut off at "length" looking like an answer in a trace, or a new key in logs/usage.jsonl.
    assert real_call().finish_reason == "stop"
    assert '"finish_reason"' not in real_call().log_line()


def test_kind_is_logged_only_when_set(utc):
    # Catches `"kind": null` on every ordinary call, which the log never had.
    assert '"kind": "compaction"' in real_call(kind="compaction").log_line()
    assert '"kind"' not in real_call().log_line()


@pytest.mark.parametrize(
    "raw",
    [
        {"prompt_tokens_details": None, "completion_tokens_details": None},
        {},
        {
            "prompt_tokens": None,
            "completion_tokens": None,
            "prompt_tokens_details": {"cached_tokens": None},
            "completion_tokens_details": {"reasoning_tokens": None},
        },
    ],
)
def test_null_or_missing_counts_read_as_zero(raw):
    # Catches a crash after a paid call when a provider sends nulls or leaves the details out.
    c = ModelCall.from_completion(parse_completion(response(usage=raw)), path="main")
    assert (c.prompt_tokens, c.cached_tokens, c.completion_tokens, c.reasoning_tokens) == (0, 0, 0, 0)
    assert c.cost is None
    usage_line([c])  # sums without a TypeError


# ---------------------------------------------------------------- append_log


def test_append_log_with_nothing_creates_no_file(tmp_path):
    path = tmp_path / "usage.jsonl"
    append_log(path, [])
    assert not path.exists()


def test_append_log_creates_dirs_and_appends(tmp_path):
    # Catches "w" mode wiping the cross-session history, and a crash on the first run.
    path = tmp_path / "logs" / "nested" / "usage.jsonl"
    first, second = [model_call(cost=0.001), model_call(cost=0.002)], [model_call(cost=0.003), model_call(cost=0.004)]
    append_log(path, first)
    append_log(path, second)
    assert path.read_text().splitlines() == [c.log_line() for c in first + second]


# ---------------------------------------------------------------- usage_line


def test_usage_line_sums_calls():
    # Catches showing only the last call, or a cached share with the wrong denominator.
    calls = [
        model_call(prompt=1400, cached=1000, completion=100, cost=0.0005),
        model_call(prompt=2000, cached=1788, completion=110, cost=0.0007),
    ]
    assert usage_line(calls) == "$0.0012 · 3.4k in (82% cached) · 210 out"


def test_usage_line_with_no_prompt_tokens():
    # Catches a ZeroDivisionError on an empty turn or one with 0 prompt tokens.
    assert usage_line([]) == "$0.0000 · 0 in · 0 out"
    assert usage_line([model_call(prompt=0, completion=0, cost=0.0)]) == "$0.0000 · 0 in · 0 out"


def test_usage_line_marks_unknown_cost():
    # Catches an unknown cost shown as free.
    known, unknown = model_call(prompt=100, cost=0.0005), model_call(prompt=100, cost=None)
    assert usage_line([known, unknown]).startswith("$0.0005+? · ")
    assert usage_line([unknown]).startswith("$? · ")


def test_human_boundary():
    assert human(999) == "999"
    assert human(1000) == "1.0k"
