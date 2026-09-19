"""Accounting for model calls: turns the usage the agent keeps (`agent.usage`) into records, lines and a log."""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

COST_UNKNOWN = "[stopped: cost of the last call is unknown, so the budget can't be enforced]"


def record(raw: dict) -> dict:
    """One flat record from an entry of `agent.usage` (the API's usage plus the response's id, provider, model, created).

    Token counts that are missing or null read as 0; a missing cost stays None, meaning unknown.
    A call that wasn't a normal step (`kind`, e.g. "compaction") keeps its kind.
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
    """Append one JSON line per model call, to sum spend across sessions."""
    if not raws:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for raw in raws:
            f.write(json.dumps(record(raw)) + "\n")


@contextmanager
def logged_calls(usage: list[dict], path: str | Path | None) -> Iterator[list[dict]]:
    """Collect the entries appended to `usage` inside the block, and append them to `path` when it exits, even on error.

    The yielded list is filled in on exit. With `path=None` nothing is written.
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
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def usage_line(raws: list[dict]) -> str:
    """One line summing some entries of `agent.usage`: `$0.0012 · 3.4k in (82% cached) · 210 out`.

    Calls with an unknown cost show as `+?` after the known amount, or `$?` if no cost is known.
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
