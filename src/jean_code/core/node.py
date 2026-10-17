"""The agent tree: one `AgentNode` per agent."""

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
    """One agent's place in the agent tree.

    A node holds what applies to its agent and every agent below it: where it is, what it may spend and has
    spent, and whether it may go on. A subagent's node is a child of its parent's. Thread-safe. The arguments it
    keeps are attributes of the same name.

    Args:
        limit: Most the subtree may spend, in USD; None for no limit. Settable at any time.
        parent: The parent's node; None for the root.
        name: The last part of `path`.
        task: What this node's agent was asked to do.
        log_path: Root only: the file every model call in the tree is appended to as it's charged. Settable.
        profile: What kind of agent this is; "" for the root.
        listener: Root only: receives every event the tree's agents `emit`. Settable.

    Attributes:
        agent: The agent built on this node, once there is one.
        children: The subagents' nodes, in creation order.
        entries: This node's own model calls, in charge order.
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
        self.limit = limit
        self.parent = parent
        self.name = name
        self.task = task
        self.log_path = log_path
        self.profile = profile
        self.listener = listener
        self.agent: Agent | None = None
        self.children: list[AgentNode] = []
        self.entries: list[ModelCall] = []
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
        """The whole tree's model calls, in charge order.

        Each carries its node's path. Append-only, so `journal[start:]` is every call made in the tree since
        `len(journal)` was `start`.
        """
        return self._journal

    def child(self, limit: float | None, name: str, task: str = "", profile: str = "") -> AgentNode:
        """Creates a node under this one, for a subagent.

        The new node's spend counts against this node and every ancestor.

        Args:
            limit: Most its subtree may spend, in USD; None for no limit.
            name: The last part of its path.
            task: What its agent is asked to do.
            profile: What kind of agent it is.

        Returns:
            The new node.
        """
        with self._lock:
            node = AgentNode(limit, parent=self, name=name, task=task, profile=profile)
            node._cancelled = self._cancelled  # a child made after a cancel is part of the stopped subtree
            self.children.append(node)
        return node

    def charge(self, call: ModelCall) -> None:
        """Records one model call at this node, and emits it as `Charged`.

        The call also goes into the tree's `journal`, and into the root's `log_path` file if one is set. The file
        is written, and the event emitted, without holding the tree's lock.

        Args:
            call: The call, tagged with this node's path.

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
        """Hands `event` to the root's listener, if there is one.

        The listener runs on the calling thread, never under the tree's lock. It must not block and must not touch
        any agent: it records the event and returns.

        Args:
            event: What the agent did.
        """
        if (listener := self.root.listener) is not None:
            listener(event)

    def cancel(self) -> None:
        """Stops model calls in this subtree.

        Every node in the subtree stays cancelled until it's reopened itself: reopening this node later doesn't
        reopen the ones below it. Thread-safe.
        """
        with self._lock:
            for node in self._subtree():
                node._cancelled = True

    def reopen(self) -> None:
        """Lets this node's agent call the model again.

        Only this node: its descendants stay cancelled, and a cancelled ancestor still stops it. The caller that
        starts the agent's next turn reopens it, before that turn starts, so a cancel sent to a turn holds for it.
        """
        self._cancelled = False

    @property
    def spent(self) -> float:
        """USD spent by this node's subtree; unknown costs count as 0."""
        with self._lock:
            return sum(call.cost or 0 for node in self._subtree() for call in node.entries)

    @property
    def cancelled(self) -> bool:
        """Whether this node or an ancestor was cancelled."""
        return any(node._cancelled for node in self._chain())

    @property
    def exhausted(self) -> bool:
        """Whether this node or an ancestor is cancelled or has spent its limit.

        Then no more model calls may be made.
        """
        return self.cancelled or any(node.limit is not None and node.spent >= node.limit for node in self._chain())

    @property
    def remaining(self) -> float | None:
        """What this node may still spend; None when no node on its chain has a limit.

        It's the smallest `limit - spent` on the chain, at least 0.
        """
        left = [node.limit - node.spent for node in self._chain() if node.limit is not None]
        return max(0.0, min(left)) if left else None

    def nodes(self) -> list[AgentNode]:
        """Returns this node and its descendants, depth first."""
        with self._lock:
            return self._subtree()

    def breakdown(self) -> list[dict]:
        """Returns one row per node of the subtree, to see where spend went.

        Each row has the node's `path` and `task`, and from its own calls: their count, token totals, peak
        prompt, known cost, and how many had no cost.
        """
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
        """Returns this node and its ancestors, ending at the root."""
        chain: list[AgentNode] = [self]
        while chain[-1].parent is not None:
            chain.append(chain[-1].parent)
        return chain

    def _subtree(self) -> list[AgentNode]:
        """Returns this node and its descendants, depth first; the caller holds the lock."""
        nodes: list[AgentNode] = [self]
        for child in self.children:
            nodes += child._subtree()
        return nodes
