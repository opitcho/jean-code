"""Accounting for model calls: what each call used and cost.

`ModelCall` is one paid model call. `usage_line` sums calls into a line for the user; `breakdown_row` sums one
agent's calls into a row of a breakdown, and `breakdown_lines` formats those rows as a table.
"""

import json
import time
from dataclasses import dataclass

from jean_code.core.llm import Completion, TokenDetails, Usage


@dataclass(frozen=True)
class ModelCall:
    """One paid model call: what it used and what it cost.

    Unknown token counts read as 0; an unknown cost stays None, so it's never mistaken for a free call.

    Attributes:
        generation_id: The response's id.
        provider: Who served it, if reported.
        model: The model that answered.
        created: When it was made, in Unix seconds.
        cost: What it cost, in USD; None if unknown.
        path: The path of the node charged, e.g. "main/1".
        prompt_tokens: Tokens in the prompt.
        cached_tokens: Prompt tokens read from the cache.
        completion_tokens: Tokens in the reply.
        reasoning_tokens: Reply tokens spent reasoning.
        kind: "compaction" for summary calls; None for the others.
        finish_reason: Why the model stopped, as sent; None for a cut-off stream.
        ttft: As `Completion.ttft`.
        duration: As `Completion.duration`.
    """

    generation_id: str
    provider: str | None
    model: str
    created: int
    cost: float | None
    path: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    kind: str | None = None
    finish_reason: str | None = None
    ttft: float | None = None
    duration: float | None = None

    @classmethod
    def from_completion(cls, completion: Completion, path: str, kind: str | None = None) -> "ModelCall":
        """Returns the call `completion` answered, charged to the node at `path`.

        Args:
            completion: The parsed response.
            path: The node's path.
            kind: "compaction" for summary calls.
        """
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
            finish_reason=completion.finish_reason,
            ttft=completion.ttft,
            duration=completion.duration,
        )

    @property
    def time(self) -> str:
        """Local ISO 8601 time of the call, from `created`."""
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.created))

    def log_line(self) -> str:
        """Returns the call as the usage log's JSON line.

        The keys the log has always had keep their order and values, so old and new lines sum together. `kind` and
        `budget` (the path) are left out when empty, and `ttft` and `duration` (seconds, to the millisecond) when
        the call wasn't timed.
        """
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
        timing = (("ttft", self.ttft), ("duration", self.duration))
        line |= {key: round(value, 3) for key, value in timing if value is not None}
        return json.dumps(line)


def human(n: int) -> str:
    """Formats a token count briefly: `999` stays `999`, `3400` becomes `3.4k`.

    Args:
        n: The token count.
    """
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def usage_line(calls: list[ModelCall]) -> str:
    """Sums model calls into one line for the user.

    For example `$0.0012 · 3.4k in (82% cached) · 210 out`. Calls with an unknown cost are left out of the dollar
    amount and flagged: `+?` after the known amount, or `$?` if no call's cost is known.

    Args:
        calls: Model calls to sum, usually the calls of one turn: a slice of `TreeState.calls`.

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


def breakdown_row(path: str, task: str, calls: list[ModelCall]) -> dict:
    """Returns one agent's row of a spend breakdown.

    The row has `path` and `task`, and from `calls`: their count, token totals, peak prompt, known cost, and how
    many had no cost.

    Args:
        path: The agent's path.
        task: What it was asked to do.
        calls: Its own model calls.
    """
    known = [c.cost for c in calls if c.cost is not None]
    return {
        "path": path,
        "task": task,
        "calls": len(calls),
        "prompt_tokens": sum(c.prompt_tokens for c in calls),
        "cached_tokens": sum(c.cached_tokens for c in calls),
        "peak_prompt_tokens": max((c.prompt_tokens for c in calls), default=0),
        "completion_tokens": sum(c.completion_tokens for c in calls),
        "reasoning_tokens": sum(c.reasoning_tokens for c in calls),
        "cost": sum(known),
        "unknown_costs": len(calls) - len(known),
    }


def breakdown_lines(rows: list[dict]) -> list[str]:
    """Formats `breakdown_row` rows as a table, most expensive first.

    The table shows which agent took the money, and why.

    Args:
        rows: The rows.

    Returns:
        The table's lines, header first.
    """
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
