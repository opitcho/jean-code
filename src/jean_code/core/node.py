"""The work tree: `Node`, its kinds `AgentNode` and `GroupNode`, and `UnknownNode`."""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

from jean_code.core.events import AgentEvent, Charged, ErrorCaught, GroupAdded, LimitSet, error_line
from jean_code.core.usage import ModelCall, breakdown_row

if TYPE_CHECKING:
    from jean_code.core.agent import Agent


class UnknownNode(LookupError):
    """No node in the tree has this path."""


class Node:
    """A place in the work tree: an agent's, or a group's.

    A group holds the nodes under it. A node holds what applies to it and everything below it: where it is, what it
    may spend and has spent, and whether it may go on. Thread-safe. The arguments it keeps are attributes of the same
    name.

    Args:
        limit: Most the subtree may spend, in USD; None for no limit. Settable at any time; a change is emitted as
            `LimitSet`.
        parent: The parent's node; None for the root.
        name: The last part of `path`.
        task: What this node was made for.
        profile: What kind of agent or group this is; "" for the root.
        listener: Root only: called with every event any node of the tree emits, on the thread that emits it, and
            sometimes while the tree is locked. It must not call into the tree, and must take no lock that anything
            holds while calling into the tree. Settable.

    Attributes:
        children: The nodes under this one, in creation order.
    """

    def __init__(
        self,
        limit: float | None = None,
        parent: Node | None = None,
        name: str = "main",
        task: str = "",
        profile: str = "",
        listener: Callable[[AgentEvent], None] | None = None,
    ):
        self._limit = limit  # not through the setter: nothing has announced the node yet for a LimitSet to follow
        self.parent = parent
        self.name = name
        self.task = task
        self.profile = profile
        self.listener = listener
        self.children: list[Node] = []
        self._cancelled = False  # set by cancel(), cleared only by AgentNode.reopen()
        self._lock: threading.Lock = parent._lock if parent else threading.Lock()

    @property
    def limit(self) -> float | None:
        """Most the subtree may spend, in USD; None for no limit."""
        return self._limit

    @limit.setter
    def limit(self, value: float | None) -> None:
        with self._lock:  # one hold for both: with two writers, the last LimitSet still matches the limit
            self._limit = value
            self.emit(LimitSet(path=self.path, limit=value))

    @property
    def path(self) -> str:
        """The node's place in the tree, e.g. "main/1/2"."""
        return f"{self.parent.path}/{self.name}" if self.parent else self.name

    @property
    def root(self) -> Node:
        """The root of this node's tree."""
        return self._chain()[-1]

    def add_agent(self, limit: float | None, task: str = "", profile: str = "") -> AgentNode:
        """Creates a node under this one, for an agent.

        It's named with the next number among this node's children, of either kind. The new node's spend counts
        against this node and every ancestor.

        Args:
            limit: Most its subtree may spend, in USD; None for no limit.
            task: What its agent is asked to do.
            profile: What kind of agent it is.

        Returns:
            The new node.
        """
        with self._lock:
            return self._add_locked(AgentNode, limit, task, profile)

    def add_group(self, limit: float | None, task: str = "", profile: str = "") -> GroupNode:
        """Creates a group node under this one, and emits `GroupAdded`.

        It's named as `add_agent` names a node.

        Args:
            limit: Most the group's subtree may spend, in USD; None for no limit.
            task: What the group was made for.
            profile: What kind of group it is.

        Returns:
            The new node.
        """
        with self._lock:
            group = self._add_locked(GroupNode, limit, task, profile)
            # In the same hold: nothing can find the group, and emit its LimitSet, before its GroupAdded
            self.emit(GroupAdded(path=group.path, profile=profile, task=task, limit=limit))
        return group

    def find(self, path: str) -> Node:
        """Returns the node at `path` in this subtree.

        Args:
            path: The node's full path, e.g. "main/1/2".

        Raises:
            UnknownNode: No node in this subtree has `path`.
        """
        node = next((n for n in self.nodes() if n.path == path), None)
        if node is None:
            raise UnknownNode(path)
        return node

    def emit(self, event: AgentEvent) -> None:
        """Calls the root's listener with `event`, if any, and returns once it has.

        Raises what the listener raises.

        Args:
            event: What happened.
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

    @property
    def spent(self) -> float:
        """USD spent by this node's subtree; unknown costs count as 0."""
        with self._lock:
            return sum(call.cost or 0 for node in self._agent_nodes() for call in node.entries)

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

    def nodes(self) -> list[Node]:
        """Returns this node and its descendants, depth first."""
        with self._lock:
            return self._subtree()

    def agents(self) -> list[AgentNode]:
        """Returns the nodes of this subtree that carry an agent, depth first."""
        with self._lock:
            return self._agents_locked()

    def breakdown(self) -> list[dict]:
        """Returns one `breakdown_row` per node in `agents()`, to see where spend went.

        Each row sums the node's own calls.
        """
        with self._lock:
            rows = [(node.path, node.task, list(node.entries)) for node in self._agents_locked()]
        return [breakdown_row(path, task, calls) for path, task, calls in rows]

    def _add_locked[N: Node](self, kind: type[N], limit: float | None, task: str, profile: str) -> N:
        node = kind(limit, parent=self, name=str(len(self.children) + 1), task=task, profile=profile)
        node._cancelled = self._cancelled  # a child made after a cancel is part of the stopped subtree
        self.children.append(node)
        return node

    def _agent_nodes(self) -> list[AgentNode]:
        """Returns the subtree's `AgentNode`s, agent or not; the caller holds the lock."""
        return [node for node in self._subtree() if isinstance(node, AgentNode)]

    def _agents_locked(self) -> list[AgentNode]:
        return [node for node in self._agent_nodes() if node.agent is not None]

    def _chain(self) -> list[Node]:
        """Returns this node and its ancestors, ending at the root."""
        chain: list[Node] = [self]
        while chain[-1].parent is not None:
            chain.append(chain[-1].parent)
        return chain

    def _subtree(self) -> list[Node]:
        """Returns this node and its descendants, depth first; the caller holds the lock."""
        nodes: list[Node] = [self]
        for child in self.children:
            nodes += child._subtree()
        return nodes


class AgentNode(Node):
    """A node that carries one agent.

    Its own model calls are charged to it. The root of a tree is always one. Takes the same arguments as `Node`.

    Attributes:
        agent: The agent built on this node, once there is one.
        entries: This node's own model calls, in charge order.
    """

    def __init__(
        self,
        limit: float | None = None,
        parent: Node | None = None,
        name: str = "main",
        task: str = "",
        profile: str = "",
        listener: Callable[[AgentEvent], None] | None = None,
    ):
        super().__init__(limit, parent, name, task, profile, listener)
        self.agent: Agent | None = None
        self.entries: list[ModelCall] = []

    def charge(self, call: ModelCall) -> None:
        """Records one model call at this node, and emits it as `Charged`.

        Args:
            call: The call, tagged with this node's path.

        Raises:
            ValueError: The call is tagged with another node's path, so its spend would land in the wrong subtree.
        """
        if call.path != self.path:
            raise ValueError(f"call for {call.path!r} charged to {self.path!r}")
        with self._lock:
            self.entries.append(call)
        self.emit(Charged(path=self.path, call=call))

    def record_error(self, error: BaseException, tool_call_id: str | None = None) -> str:
        """Emits `error` as `ErrorCaught`, with its traceback, and returns it as one line.

        Only an `Exception` is emitted: `KeyboardInterrupt` and `SystemExit` are the process stopping. Never raises,
        unless the listener does.

        Args:
            error: The exception caught.
            tool_call_id: The tool call that raised it, if one did.

        Returns:
            The exception as one line, as `error_line` gives it.
        """
        line = error_line(error)
        # Nothing here may raise: agents call it inside the handlers that must still end their turn.
        if isinstance(error, Exception):
            trace = "".join(traceback.format_exception(error))
            self.emit(ErrorCaught(path=self.path, error=line, traceback=trace, tool_call_id=tool_call_id))
        return line

    def reopen(self) -> None:
        """Lets this node's agent call the model again.

        Only this node is reopened: its descendants stay cancelled, and a cancelled ancestor still stops it.
        """
        self._cancelled = False


class GroupNode(Node):
    """A node that holds other nodes and runs nothing itself.

    The nodes under it share its limit, and a cancel of it stops them all. Once cancelled, it stays cancelled: a new
    run takes a new group. Takes the same arguments as `Node`.
    """
