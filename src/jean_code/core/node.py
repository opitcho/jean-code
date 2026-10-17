"""The agent tree: one `AgentNode` per agent, with its path, its task, its spend and limit, and its cancel."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from jean_code.core.events import AgentEvent, Charged
from jean_code.core.usage import ModelCall, append_log

if TYPE_CHECKING:
    from jean_code.core.agent import Agent


class AgentNode:
    """One agent's node in the agent tree: a subagent's node is a child of its parent's.

    It holds what applies to an agent and everything below it:
    - where it is: `path` ("main/1/2") and `task`, what it was asked to do
    - what it may spend and has spent: `limit` (USD) applies to the subtree, and an agent may call the model only
      while no node on its chain is `exhausted`. Entries stay at the node that made the call, so any subtree's
      spend, and where it went, can be read at any time; `journal` is the whole tree's calls in charge order
    - whether it may go on: `cancel()` stops the subtree until `reopen()`
    - where its calls are logged: with `log_path` set on the root, each call is appended to that file as it's
      charged, whichever agent made it and whenever
    - who hears about it: with `listener` set on the root, every agent in the tree emits its events (events.py) to
      it through `emit`. It's the tree's one hook: the listener records events and returns, nothing more

    Thread-safe: one lock per tree, shared by all its nodes.
    """

    def __init__(
        self,
        limit: float | None = None,
        parent: AgentNode | None = None,
        name: str = "main",
        task: str = "",
        log_path: str | Path | None = None,
        profile: str = "",
        listener: Callable[[AgentEvent], None] | None = None,
    ):
        self.limit = limit  # settable: the REPL and tests raise it mid-session
        self.parent = parent
        self.name = name
        self.task = task  # what this node's agent was asked to do, for breakdowns
        self.log_path = log_path  # read at the root only; settable, e.g. by a UI once it has built the agent
        self.profile = profile  # what kind of agent this is, for clients; "" for the root
        self.listener = listener  # read at the root only, like log_path
        self.agent: Agent | None = None  # set by Agent.__init__, so a client can find any agent by its path
        self.children: list[AgentNode] = []
        self.entries: list[ModelCall] = []  # this node's own calls: its agent's usage
        self._cancelled = False  # set by cancel(), cleared only by reopen()
        self._lock: threading.Lock = parent._lock if parent else threading.Lock()
        self._journal: list[ModelCall] = parent._journal if parent else []

    @property
    def path(self) -> str:
        """The node's place in the tree, e.g. "main/1/2"."""
        return f"{self.parent.path}/{self.name}" if self.parent else self.name

    @property
    def root(self) -> AgentNode:
        """The root of this node's agent tree."""
        return self._chain()[-1]

    @property
    def journal(self) -> list[ModelCall]:
        """The whole tree's calls in charge order, each carrying its node's path. Append-only, so
        `journal[start:]` is every call made in the tree since `len(journal)` was `start`."""
        return self._journal

    def child(self, limit: float | None, name: str, task: str = "", profile: str = "") -> AgentNode:
        """A new node under this one, for a subagent: its spend counts against this node and every ancestor."""
        with self._lock:
            node = AgentNode(limit, parent=self, name=name, task=task, profile=profile)
            self.children.append(node)
        return node

    def charge(self, call: ModelCall) -> None:
        """Record one model call at this node and in the tree's journal, append it to the root's `log_path`, if
        set, and emit it as `Charged`. The file is written and the event emitted outside the lock.

        Raises:
            ValueError: The call is tagged with another node's path, so its spend would land in the wrong subtree.
        """
        if call.path != self.path:
            raise ValueError(f"call for {call.path!r} charged to {self.path!r}")
        with self._lock:
            self.entries.append(call)
            self._journal.append(call)
        if path := self.root.log_path:
            append_log(path, [call])
        self.emit(Charged(path=self.path, call=call))

    def emit(self, event: AgentEvent) -> None:
        """Hand `event` to the root's listener, if any. Called on the thread doing the work, never under the tree's
        lock. The listener must not block and must not touch any agent: it records the event and returns."""
        if (listener := self.root.listener) is not None:
            listener(event)

    def cancel(self) -> None:
        """No more model calls in this subtree, until `reopen()`. Stops work from outside it: a parent dropping a
        run, an interrupted wait, a close. Unlike an agent's interrupt, which every turn clears, nothing clears it
        during a turn, so a cancel sent before a turn starts still holds."""
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

    def nodes(self) -> list[AgentNode]:
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

    def _chain(self) -> list[AgentNode]:
        """This node and its ancestors, ending at the root."""
        chain: list[AgentNode] = [self]
        while chain[-1].parent is not None:
            chain.append(chain[-1].parent)
        return chain

    def _subtree(self) -> list[AgentNode]:
        """This node and its descendants, depth first. The caller holds the lock."""
        nodes: list[AgentNode] = [self]
        for child in self.children:
            nodes += child._subtree()
        return nodes
