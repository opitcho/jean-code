"""Accounting for model calls: turns the usage the agent keeps (`agent.usage`) into records, lines and a log."""

import json
import time
from pathlib import Path


def record(raw: dict) -> dict:
    """One flat record from an entry of `agent.usage` (the API's usage plus the response's id, provider, model, created)."""
    created = raw.get("created")
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(created)) if created else None,
        "generation_id": raw.get("id"),
        "provider": raw.get("provider"),
        "model": raw.get("model"),
        "prompt_tokens": raw.get("prompt_tokens", 0),
        "cached_tokens": (raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "completion_tokens": raw.get("completion_tokens", 0),
        "reasoning_tokens": (raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        "cost": raw.get("cost"),
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


def human(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def usage_line(raws: list[dict]) -> str:
    """One line summing some entries of `agent.usage`: `$0.0012 · 3.4k in (82% cached) · 210 out`."""
    records = [record(raw) for raw in raws]
    cost = sum(r["cost"] or 0 for r in records)
    prompt = sum(r["prompt_tokens"] for r in records)
    cached = sum(r["cached_tokens"] for r in records)
    out = sum(r["completion_tokens"] for r in records)
    share = f" ({cached / prompt:.0%} cached)" if prompt else ""
    return f"${cost:.4f} · {human(prompt)} in{share} · {human(out)} out"
