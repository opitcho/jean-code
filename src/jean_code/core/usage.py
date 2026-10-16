"""Accounting for model calls: what each call used and cost.

`ModelCall` is one paid model call as the agent books it, `append_log` writes calls to a JSON Lines log
(`config.USAGE_LOG`), and `usage_line` sums them into the line the REPL prints after each turn. `Budget` limits
spend across an agent and its subagents.
"""

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from jean_code.core.llm import Completion, TokenDetails, Usage


@dataclass(frozen=True)
class ModelCall:
    """One paid model call, as `agent.usage`, the budget and the usage log hold it. Unknown token counts read as 0;
    an unknown cost stays None, so it's never mistaken for a free call."""

    generation_id: str
    provider: str | None
    model: str
    created: int  # Unix seconds
    cost: float | None
    budget: str  # path of the Budget node charged, e.g. "main/1"
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    kind: str | None = None  # "compaction" for summary calls

    @classmethod
    def from_completion(cls, completion: Completion, budget: str, kind: str | None = None) -> "ModelCall":
        """The call `completion` answered, charged to the node at `budget`."""
        usage = completion.usage or Usage()
        return cls(
            completion.generation_id,
            completion.provider,
            completion.model,
            completion.created,
            usage.cost,
            budget,
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
        together. `kind` and `budget` are left out when empty."""
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
        line |= {key: value for key, value in (("kind", self.kind), ("budget", self.budget)) if value}
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
        calls: Model calls to sum, usually the calls of one turn: a slice of `Budget.journal`.

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


class Budget:
    """One agent's cost ledger, in a tree that mirrors the agents: a subagent's budget is a child of its parent's.

    `limit` (USD) applies to this node's subtree: an agent may call the model only while no node on its chain
    is `exhausted`. Entries stay at the node that made the call, so any subtree's spend, and where it went,
    can be read at any time. `journal` is the whole tree's calls in charge order, each tagged with the path
    of its node. With `log_path` set on the root, each call is also appended to that file as it's charged,
    whichever agent made it and whenever.

    `cancel()` stops a subtree's work now: its agents make no further call. Unlike an agent's interrupt, which
    every turn clears, nothing clears it during a turn, so a cancel sent before a turn starts still holds.
    `reopen()` clears it between turns, for a subagent that is resumed.

    Thread-safe: one lock per tree, shared by all its nodes.
    """

    def __init__(
        self,
        limit: float | None = None,
        parent: "Budget | None" = None,
        name: str = "main",
        task: str = "",
        log_path: str | Path | None = None,
    ):
        self.limit = limit  # settable: the REPL and tests raise it mid-session
        self.parent = parent
        self.name = name
        self.task = task  # what this node's agent was asked to do, for breakdowns
        self.log_path = log_path  # read at the root only; settable, e.g. by a UI once it has built the agent
        self.children: list[Budget] = []
        self.entries: list[ModelCall] = []  # this node's own calls: its agent's usage
        self._cancelled = False  # set by cancel(), cleared only by reopen()
        self._lock: threading.Lock = parent._lock if parent else threading.Lock()
        self._journal: list[ModelCall] = parent._journal if parent else []

    @property
    def path(self) -> str:
        """The node's place in the tree, e.g. "main/1/2"."""
        return f"{self.parent.path}/{self.name}" if self.parent else self.name

    @property
    def journal(self) -> list[ModelCall]:
        """The whole tree's calls in charge order, each carrying its node's path. Append-only, so
        `journal[start:]` is every call made in the tree since `len(journal)` was `start`."""
        return self._journal

    def child(self, limit: float | None, name: str, task: str = "") -> "Budget":
        """A new node under this one, for a subagent: its spend counts against this node and every ancestor."""
        with self._lock:
            node = Budget(limit, parent=self, name=name, task=task)
            self.children.append(node)
        return node

    def charge(self, call: ModelCall) -> None:
        """Record one model call at this node and in the tree's journal, and append it to the root's `log_path`, if
        set. The file is written outside the lock.

        Raises:
            ValueError: The call is tagged with another node's path, so its spend would land in the wrong subtree.
        """
        if call.budget != self.path:
            raise ValueError(f"call for {call.budget!r} charged to {self.path!r}")
        with self._lock:
            self.entries.append(call)
            self._journal.append(call)
        if path := self._chain()[-1].log_path:
            append_log(path, [call])

    def cancel(self) -> None:
        """No more model calls in this subtree, until `reopen()`. Stops work from outside it: a parent dropping a
        run, an interrupted wait, a close."""
        self._cancelled = True

    def reopen(self) -> None:
        """Clear this node's cancel, so a resumed run can call the model again. Only this node: a cancelled
        ancestor still stops it. Call it only between turns, so a cancel sent to a turn holds for that turn."""
        self._cancelled = False

    @property
    def spent(self) -> float:
        """USD spent by this node's subtree; unknown costs count as 0."""
        with self._lock:
            return sum(call.cost or 0 for node in self._subtree() for call in node.entries)

    @property
    def cancelled(self) -> bool:
        """True when this node or an ancestor was cancelled."""
        return any(node._cancelled for node in self._chain())

    @property
    def exhausted(self) -> bool:
        """True when this node or an ancestor was cancelled or has spent its limit: no more model calls."""
        return self.cancelled or any(node.limit is not None and node.spent >= node.limit for node in self._chain())

    @property
    def remaining(self) -> float | None:
        """What this node may still spend: the smallest `limit - spent` on its chain (at least 0), or None when no
        node on it has a limit."""
        left = [node.limit - node.spent for node in self._chain() if node.limit is not None]
        return max(0.0, min(left)) if left else None

    def nodes(self) -> list["Budget"]:
        """This node and its descendants, depth first."""
        with self._lock:
            return self._subtree()

    def breakdown(self) -> list[dict]:
        """One row per node of the subtree, from its own calls: the usual cost drivers, to see where spend went."""
        with self._lock:
            snapshot = [(node.path, node.task, list(node.entries)) for node in self._subtree()]
        rows = []
        for path, task, calls in snapshot:
            known = [c.cost for c in calls if c.cost is not None]
            rows.append(
                {
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
            )
        return rows

    def _chain(self) -> list["Budget"]:
        """This node and its ancestors, ending at the root."""
        chain: list[Budget] = [self]
        while chain[-1].parent is not None:
            chain.append(chain[-1].parent)
        return chain

    def _subtree(self) -> list["Budget"]:
        """This node and its descendants, depth first. The caller holds the lock."""
        nodes: list[Budget] = [self]
        for child in self.children:
            nodes += child._subtree()
        return nodes


def breakdown_lines(rows: list[dict]) -> list[str]:
    """`Budget.breakdown()` rows as a table, most expensive first: which agent took the money, and why."""
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
