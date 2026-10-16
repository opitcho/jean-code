"""Accounting for model calls: what each call used and cost.

`ModelCall` is one paid model call as the agent books it, `append_log` writes calls to a JSON Lines log
(`config.USAGE_LOG`), and `usage_line` sums them into the line the REPL prints after each turn. The agent tree that
charges them, with its limits, is in node.py.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from jean_code.core.llm import Completion, TokenDetails, Usage


@dataclass(frozen=True)
class ModelCall:
    """One paid model call, as `agent.usage`, the agent's node and the usage log hold it. Unknown token counts read
    as 0; an unknown cost stays None, so it's never mistaken for a free call."""

    generation_id: str
    provider: str | None
    model: str
    created: int  # Unix seconds
    cost: float | None
    path: str  # path of the node charged, e.g. "main/1"
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    kind: str | None = None  # "compaction" for summary calls

    @classmethod
    def from_completion(cls, completion: Completion, path: str, kind: str | None = None) -> "ModelCall":
        """The call `completion` answered, charged to the node at `path`."""
        usage = completion.usage or Usage()
        return cls(
            completion.generation_id,
            completion.provider,
            completion.model,
            completion.created,
            usage.cost,
            path,
            prompt_tokens=usage.prompt_tokens or 0,
            cached_tokens=(usage.prompt_tokens_details or TokenDetails()).cached_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            reasoning_tokens=(usage.completion_tokens_details or TokenDetails()).reasoning_tokens or 0,
            kind=kind,
        )

    @property
    def time(self) -> str:
        """Local ISO 8601 time of the call, from `created`."""
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.created))

    def log_line(self) -> str:
        """The usage log's JSON line: the keys, order and values the log has always had, so old and new lines sum
        together. `kind` and `budget` (the path) are left out when empty."""
        line = {
            "time": self.time,
            "generation_id": self.generation_id,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
        }
        # The path keeps its old key, "budget", so old and new lines still sum together.
        line |= {key: value for key, value in (("kind", self.kind), ("budget", self.path)) if value}
        return json.dumps(line)


def append_log(path: str | Path, calls: list[ModelCall]) -> None:
    """Append model calls to a JSON Lines log, one `ModelCall.log_line` per line.

    The log only grows, so summing its `cost` fields gives the spend across all sessions.
    The file and its parent folders are created if missing. Nothing is written if `calls` is empty.
    """
    if not calls:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for call in calls:
            f.write(call.log_line() + "\n")


def human(n: int) -> str:
    """Format a token count briefly: `999` stays `999`, `3400` becomes `3.4k`."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def usage_line(calls: list[ModelCall]) -> str:
    """Sum some model calls into one line for the user, e.g. `$0.0012 · 3.4k in (82% cached) · 210 out`.

    Calls with an unknown cost are left out of the dollar amount and flagged: `+?` after the known
    amount, or `$?` if no call's cost is known.

    Args:
        calls: Model calls to sum, usually the calls of one turn: a slice of `AgentNode.journal`.

    Returns:
        The total cost, the prompt tokens with the share read from cache, and the completion tokens.
    """
    known = [c.cost for c in calls if c.cost is not None]
    if calls and not known:  # noqa: SIM108 - a ternary of these two would be harder to read
        cost = "$?"
    else:
        cost = f"${sum(known):.4f}" + ("+?" if len(known) < len(calls) else "")
    prompt = sum(c.prompt_tokens for c in calls)
    cached = sum(c.cached_tokens for c in calls)
    out = sum(c.completion_tokens for c in calls)
    share = f" ({cached / prompt:.0%} cached)" if prompt else ""
    return f"{cost} · {human(prompt)} in{share} · {human(out)} out"


def breakdown_lines(rows: list[dict]) -> list[str]:
    """`AgentNode.breakdown()` rows as a table, most expensive first: which agent took the money, and why."""
    lines = [
        f"{'path':<12} {'task':<40} {'calls':>5} {'prompt':>7} {'cached':>6} {'peak ctx':>8} "
        f"{'out':>6} {'reason':>6} {'cost':>9}"
    ]
    for r in sorted(rows, key=lambda r: r["cost"], reverse=True):
        cached = f"{r['cached_tokens'] / r['prompt_tokens']:.0%}" if r["prompt_tokens"] else "-"
        cost = f"${r['cost']:.4f}" + ("+?" if r["unknown_costs"] else "")
        task = " ".join(r["task"].split())[:40] or "-"
        lines.append(
            f"{r['path']:<12} {task:<40} {r['calls']:>5} {human(r['prompt_tokens']):>7} {cached:>6} "
            f"{human(r['peak_prompt_tokens']):>8} {human(r['completion_tokens']):>6} "
            f"{human(r['reasoning_tokens']):>6} {cost:>9}"
        )
    return lines
