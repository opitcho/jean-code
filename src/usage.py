"""Accounting for model calls: turns the usage the agent keeps (`agent.usage`) into records, lines and a log,
and keeps a `Budget` per agent, arranged as a tree, that limits spend and shows where it went."""

import json
import threading
import time
from pathlib import Path

COST_UNKNOWN = "[stopped: cost of the last call is unknown, so the budget can't be enforced]"


def record(raw: dict) -> dict:
    """One flat record from an entry of `agent.usage` (the API's usage plus the response's id, provider, model, created).

    Token counts that are missing or null read as 0; a missing cost stays None, meaning unknown.
    A call that wasn't a normal step (`kind`, e.g. "compaction") keeps its kind, and a call charged to a
    `Budget` keeps the path of the node it was charged to (`budget`, e.g. "main/1").
    """
    created = raw.get("created")
    tags = {key: raw[key] for key in ("kind", "budget") if raw.get(key)}
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
        **tags,
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


class Budget:
    """One agent's cost ledger, in a tree that mirrors the agents: a subagent's budget is a child of its parent's.

    `limit` (USD) applies to this node's subtree: an agent may call the model only while no node on its chain
    is `exhausted`. Entries stay at the node that made the call, so any subtree's spend, and where it went,
    can be read at any time. `journal` is the whole tree's calls in charge order, each tagged with the path
    of its node. With `log_path` set on the root, each call is also appended to that file as it's charged,
    whichever agent made it and whenever.

    `cancel()` stops a subtree for good: its agents make no further call. Unlike an agent's interrupt, which
    every turn clears, nothing clears it, so a cancel sent before a turn starts still holds.

    Thread-safe: one lock per tree, shared by all its nodes, since parallel agents charge concurrently. It's
    never held around anything that blocks.
    """

    def __init__(self, limit: float | None = None, parent: "Budget | None" = None, name: str = "main", task: str = "",
                 log_path: str | Path | None = None):
        self.limit = limit  # settable: the REPL and tests raise it mid-session
        self.parent = parent
        self.name = name
        self.task = task  # what this node's agent was asked to do, for breakdowns
        self.log_path = log_path  # read at the root only; settable, e.g. by a UI once it has built the agent
        self.children: list[Budget] = []
        self.entries: list[dict] = []  # this node's own calls: its agent's usage entries
        self._cancelled = False  # set once by cancel(), never cleared
        self._lock: threading.Lock = parent._lock if parent else threading.Lock()
        self._journal: list[dict] = parent._journal if parent else []

    @property
    def path(self) -> str:
        """The node's place in the tree, e.g. "main/1/2"."""
        return f"{self.parent.path}/{self.name}" if self.parent else self.name

    @property
    def journal(self) -> list[dict]:
        """The whole tree's calls in charge order, each tagged `{"budget": path}`. Append-only, so
        `journal[start:]` is every call made in the tree since `len(journal)` was `start`."""
        return self._journal

    def child(self, limit: float | None, name: str, task: str = "") -> "Budget":
        """A new node under this one, for a subagent: its spend counts against this node and every ancestor."""
        with self._lock:
            node = Budget(limit, parent=self, name=name, task=task)
            self.children.append(node)
        return node

    def charge(self, entry: dict) -> None:
        """Record one model call's usage entry at this node and in the tree's journal, and append it to the root's
        `log_path`, if set. The file is written outside the lock."""
        tagged = {**entry, "budget": self.path}
        with self._lock:
            self.entries.append(entry)
            self._journal.append(tagged)
        if path := self._chain()[-1].log_path:
            append_log(path, [tagged])

    def cancel(self) -> None:
        """No more model calls in this subtree. Never cleared: a cancelled node stays cancelled."""
        self._cancelled = True

    @property
    def spent(self) -> float:
        """USD spent by this node's subtree; unknown costs count as 0."""
        with self._lock:
            return sum(entry.get("cost") or 0 for node in self._subtree() for entry in node.entries)

    @property
    def cancelled(self) -> bool:
        """True when this node or an ancestor was cancelled."""
        return any(node._cancelled for node in self._chain())

    @property
    def exhausted(self) -> bool:
        """True when this node or an ancestor was cancelled or has spent its limit: no more model calls."""
        return self.cancelled or any(node.limit is not None and node.spent >= node.limit for node in self._chain())

    @property
    def limited(self) -> bool:
        """True when this node or an ancestor has a limit, so a call of unknown cost makes it unenforceable."""
        return any(node.limit is not None for node in self._chain())

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
        for path, task, entries in snapshot:
            records = [record(entry) for entry in entries]
            known = [r["cost"] for r in records if r["cost"] is not None]
            rows.append({
                "path": path,
                "task": task,
                "calls": len(records),
                "prompt_tokens": sum(r["prompt_tokens"] for r in records),
                "cached_tokens": sum(r["cached_tokens"] for r in records),
                "peak_prompt_tokens": max((r["prompt_tokens"] for r in records), default=0),
                "completion_tokens": sum(r["completion_tokens"] for r in records),
                "reasoning_tokens": sum(r["reasoning_tokens"] for r in records),
                "cost": sum(known),
                "unknown_costs": len(records) - len(known),
            })
        return rows

    def _chain(self) -> list["Budget"]:
        """This node and its ancestors, ending at the root."""
        chain: list[Budget] = [self]
        while chain[-1].parent is not None:
            chain.append(chain[-1].parent)
        return chain

    def _subtree(self) -> list["Budget"]:
        """This node and its descendants, depth first. The caller holds the lock."""
        nodes = [self]
        for child in self.children:
            nodes += child._subtree()
        return nodes


def breakdown_lines(rows: list[dict]) -> list[str]:
    """`Budget.breakdown()` rows as a table, most expensive first: which agent took the money, and why."""
    lines = [f"{'path':<12} {'task':<40} {'calls':>5} {'prompt':>7} {'cached':>6} {'peak ctx':>8} "
             f"{'out':>6} {'reason':>6} {'cost':>9}"]
    for r in sorted(rows, key=lambda r: r["cost"], reverse=True):
        cached = f"{r['cached_tokens'] / r['prompt_tokens']:.0%}" if r["prompt_tokens"] else "-"
        cost = f"${r['cost']:.4f}" + ("+?" if r["unknown_costs"] else "")
        task = " ".join(r["task"].split())[:40] or "-"
        lines.append(f"{r['path']:<12} {task:<40} {r['calls']:>5} {human(r['prompt_tokens']):>7} {cached:>6} "
                     f"{human(r['peak_prompt_tokens']):>8} {human(r['completion_tokens']):>6} "
                     f"{human(r['reasoning_tokens']):>6} {cost:>9}")
    return lines
