"""Tests for `usage.py`: records, the log, the one-line summary and `logged_calls`.

Each test names the bug it would catch.
"""

import json

import pytest

from conftest import REAL_META, REAL_USAGE, usage
from usage import append_log, human, logged_calls, record, usage_line

REAL_ENTRY = {**REAL_USAGE, **REAL_META}  # an entry of `agent.usage`, as `Agent.step()` builds it


def entry(**kwargs) -> dict:
    return {**usage(**kwargs), "id": "gen-x", "provider": "DeepSeek", "model": "m", "created": 1789748403}


# ---------------------------------------------------------------- record


def test_record_of_the_real_payload(utc):
    # Catches a renamed or moved field (e.g. cached_tokens read from the top level) silently becoming 0.
    assert record(REAL_ENTRY) == {
        "time": "2026-09-18T16:20:03+0000",
        "generation_id": "gen-1789748403-OPdrI8Y5SuMpxM9FVbYd",
        "provider": "DeepSeek",
        "model": "deepseek/deepseek-v4.1-flash",
        "prompt_tokens": 1433,
        "cached_tokens": 1280,
        "completion_tokens": 2,
        "reasoning_tokens": 0,
        "cost": 2.799e-05,
    }


@pytest.mark.parametrize("raw", [
    {"prompt_tokens_details": None, "completion_tokens_details": None},
    {},
    {"prompt_tokens": None, "completion_tokens": None,
     "prompt_tokens_details": {"cached_tokens": None}, "completion_tokens_details": {"reasoning_tokens": None}},
])
def test_record_reads_null_or_missing_counts_as_zero(raw):
    # Catches a crash after a paid call when a provider sends nulls or leaves the details out.
    r = record(raw)
    assert (r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"], r["reasoning_tokens"]) == (0, 0, 0, 0)
    assert r["cost"] is None
    usage_line([raw])  # sums without a TypeError


def test_record_without_created_has_no_time():
    # Catches time.localtime(None), which silently writes the current time.
    assert record({"prompt_tokens": 1})["time"] is None


# ---------------------------------------------------------------- append_log


def test_append_log_with_nothing_creates_no_file(tmp_path):
    path = tmp_path / "usage.jsonl"
    append_log(path, [])
    assert not path.exists()


def test_append_log_creates_dirs_and_appends(tmp_path):
    # Catches "w" mode wiping the cross-session history, and a crash on the first run.
    path = tmp_path / "logs" / "nested" / "usage.jsonl"
    first, second = [entry(cost=0.001), entry(cost=0.002)], [entry(cost=0.003), entry(cost=0.004)]
    append_log(path, first)
    append_log(path, second)
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [record(raw) for raw in first + second]


# ---------------------------------------------------------------- usage_line


def test_usage_line_sums_calls():
    # Catches showing only the last call, or a cached share with the wrong denominator.
    raws = [entry(prompt=1400, cached=1000, completion=100, cost=0.0005),
            entry(prompt=2000, cached=1788, completion=110, cost=0.0007)]
    assert usage_line(raws) == "$0.0012 · 3.4k in (82% cached) · 210 out"


def test_usage_line_with_no_prompt_tokens():
    # Catches a ZeroDivisionError on an empty turn or one with 0 prompt tokens.
    assert usage_line([]) == "$0.0000 · 0 in · 0 out"
    assert usage_line([entry(prompt=0, completion=0, cost=0.0)]) == "$0.0000 · 0 in · 0 out"


def test_usage_line_marks_unknown_cost():
    # Catches an unknown cost shown as free.
    known, unknown = entry(prompt=100, cost=0.0005), entry(prompt=100, cost=None)
    assert usage_line([known, unknown]).startswith("$0.0005+? · ")
    assert usage_line([unknown]).startswith("$? · ")


def test_human_boundary():
    assert human(999) == "999"
    assert human(1000) == "1.0k"


# ---------------------------------------------------------------- logged_calls


def test_logged_calls_logs_only_the_calls_made_inside(tmp_path):
    # Catches the slice taken at the wrong time, which logs turn 1 again in turn 2.
    path = tmp_path / "usage.jsonl"
    agent_usage = [entry(cost=0.001)]
    with logged_calls(agent_usage, path) as calls:
        agent_usage += [entry(cost=0.002), entry(cost=0.003)]
    assert calls == agent_usage[1:]
    assert [json.loads(line) for line in path.read_text().splitlines()] == [record(raw) for raw in agent_usage[1:]]


def test_logged_calls_logs_on_error_and_reraises(tmp_path):
    # Catches a crash mid-turn dropping calls that were already paid for, or swallowing the error.
    path = tmp_path / "usage.jsonl"
    agent_usage: list[dict] = []
    with pytest.raises(RuntimeError, match="boom"):
        with logged_calls(agent_usage, path):
            agent_usage.append(entry(cost=0.002))
            raise RuntimeError("boom")
    assert len(path.read_text().splitlines()) == 1


def test_logged_calls_without_a_path(tmp_path):
    agent_usage: list[dict] = []
    with logged_calls(agent_usage, None) as calls:
        agent_usage.append(entry())
    assert calls == agent_usage
    assert list(tmp_path.iterdir()) == []
