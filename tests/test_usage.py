"""Tests for `usage.py`: the `ModelCall` record, the log and the one-line summary. `Budget` is in `test_budget.py`.

Each test names the bug it would catch.
"""

import json

import pytest

from conftest import REAL_META, REAL_USAGE, usage
from jean_code.core.llm import parse_receipt
from jean_code.core.usage import ModelCall, append_log, human, usage_line


def model_call(**kwargs) -> ModelCall:
    """A booked call with the given usage, as `Agent._book` builds it."""
    receipt, _ = parse_receipt(
        {"id": "gen-x", "provider": "DeepSeek", "model": "m", "created": 1789748403, "usage": usage(**kwargs)}
    )
    return ModelCall.from_receipt(receipt, budget="main")


def real_call(**kwargs) -> ModelCall:
    receipt, _ = parse_receipt({**REAL_META, "usage": REAL_USAGE})
    return ModelCall.from_receipt(receipt, budget="main", **kwargs)


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


def test_kind_is_logged_only_when_set(utc):
    # Catches `"kind": null` on every ordinary call, which the log never had.
    assert '"kind": "compaction"' in real_call(kind="compaction").log_line()
    assert '"kind"' not in real_call().log_line()


def test_odd_metadata_still_logs(utc):
    # Catches an odd id or timestamp from a fallback provider crashing the log write after a paid call.
    receipt, _ = parse_receipt({"id": 7, "created": "x", "usage": {"cost": 0.001}})
    line = json.loads(ModelCall.from_receipt(receipt, budget="main").log_line())
    assert line["time"] is None and line["generation_id"] is None and line["cost"] == 0.001


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
    c = ModelCall.from_receipt(parse_receipt({"usage": raw})[0], budget="main")
    assert (c.prompt_tokens, c.cached_tokens, c.completion_tokens, c.reasoning_tokens) == (0, 0, 0, 0)
    assert c.cost is None
    usage_line([c])  # sums without a TypeError


def test_without_created_there_is_no_time():
    # Catches time.localtime(None), which silently writes the current time.
    assert ModelCall.from_receipt(parse_receipt({"usage": {"prompt_tokens": 1}})[0], budget="main").time is None


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
