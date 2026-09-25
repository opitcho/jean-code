"""Accounting for model calls: what each call used and cost, shown to the user and saved to a log.

The agent appends one raw entry per model call to `agent.usage` (see `Agent._record_usage`): the API's
`usage` object (token counts, `cost`) plus the response's `id`, `provider`, `model` and `created`, and a
`kind` for calls that aren't normal steps, such as "compaction". This module turns those raw entries into
flat records, writes them to a JSON Lines log (`config.USAGE_LOG`) so spend can be summed across sessions,
and sums them into the one-line summary the REPL prints after each turn.
"""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

COST_UNKNOWN = "[stopped: cost of the last call is unknown, so the budget can't be enforced]"


def record(raw: dict) -> dict:
    """Flatten one raw entry of `agent.usage` into the record written to the log.

    Token counts that are missing or null read as 0. A missing cost stays None, meaning unknown,
    so it is never mistaken for a free call.

    Args:
        raw: One entry of `agent.usage`: the API's usage object plus the response's `id`, `provider`,
            `model` and `created` (Unix seconds), and `kind` if the call wasn't a normal step.

    Returns:
        A flat dict with `time` (local ISO 8601, or None), `generation_id`, `provider`, `model`,
        `prompt_tokens`, `cached_tokens`, `completion_tokens`, `reasoning_tokens` and `cost`
        (dollars, or None if unknown), plus `kind` when the entry has one.
    """
    created = raw.get("created")
    kind = {"kind": raw["kind"]} if raw.get("kind") else {}
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(created)) if created else None,
        "generation_id": raw.get("id"),
        "provider": raw.get("provider"),
        "model": raw.get("model"),
        "prompt_tokens": raw.get("prompt_tokens") or 0,
        "cached_tokens": (raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        "completion_tokens": raw.get("completion_tokens") or 0,
        "reasoning_tokens": (raw.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
        "cost": raw.get("cost"),
        **kind,
    }


def append_log(path: str | Path, raws: list[dict]) -> None:
    """Append model calls to a JSON Lines log, one `record` per line.

    The log only grows, so summing its `cost` fields gives the spend across all sessions.
    The file and its parent folders are created if missing. Nothing is written if `raws` is empty.

    Args:
        path: The log file, usually `config.USAGE_LOG`.
        raws: Entries of `agent.usage` to log, in order.
    """
    if not raws:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for raw in raws:
            f.write(json.dumps(record(raw)) + "\n")


@contextmanager
def logged_calls(usage: list[dict], path: str | Path | None) -> Iterator[list[dict]]:
    """Log the model calls made inside a `with` block, such as one turn of the agent.

    On entry it notes how many entries `usage` already has. On exit, even if the block raised, it
    collects the entries appended since then and appends them to the log at `path`, so calls that
    were already paid for are logged when a turn fails. The error, if any, is re-raised.

        with logged_calls(agent.usage, USAGE_LOG) as calls:
            agent.run_turn(text)
        print(usage_line(calls))

    Args:
        usage: The list the agent appends each call to, `agent.usage`. It must only be appended to
            while the block runs.
        path: The log file to append to, or None to collect the calls without writing anything.

    Yields:
        A list that is empty inside the block and holds the block's entries of `usage` once it exits.
    """
    start = len(usage)
    calls: list[dict] = []
    try:
        yield calls
    finally:
        calls.extend(usage[start:])
        if path:
            append_log(path, calls)


def human(n: int) -> str:
    """Format a token count briefly: `999` stays `999`, `3400` becomes `3.4k`."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def usage_line(raws: list[dict]) -> str:
    """Sum some model calls into one line for the user, e.g. `$0.0012 · 3.4k in (82% cached) · 210 out`.

    Calls with an unknown cost are left out of the dollar amount and flagged: `+?` after the known
    amount, or `$?` if no call's cost is known.

    Args:
        raws: Entries of `agent.usage` to sum, usually the calls of one turn from `logged_calls`.

    Returns:
        The total cost, the prompt tokens with the share read from cache, and the completion tokens.
    """
    records = [record(raw) for raw in raws]
    known = [r["cost"] for r in records if r["cost"] is not None]
    if records and not known:
        cost = "$?"
    else:
        cost = f"${sum(known):.4f}" + ("+?" if len(known) < len(records) else "")
    prompt = sum(r["prompt_tokens"] for r in records)
    cached = sum(r["cached_tokens"] for r in records)
    out = sum(r["completion_tokens"] for r in records)
    share = f" ({cached / prompt:.0%} cached)" if prompt else ""
    return f"{cost} · {human(prompt)} in{share} · {human(out)} out"
