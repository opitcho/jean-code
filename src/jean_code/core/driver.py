"""Running an agent from another thread: `Driver`, and what its commands raise."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from pathlib import Path

from jean_code.core.agent import NOT_RUN, Agent
from jean_code.core.events import (
    Approval,
    ApprovalRequested,
    ApprovalResolved,
    JobEnded,
    JobStarted,
    ViewEvent,
)
from jean_code.core.node import AgentNode
from jean_code.core.view import AgentView

Approve = Callable[[str, dict], str | None]
Build = Callable[[AgentNode, Approve], Agent]
LIVE_SETTINGS = ("model", "reasoning", "max_cost", "compact_at")
REASONING = (None, "low", "medium", "high")


class Busy(RuntimeError):
    """A turn or a compaction is already running."""


class UnknownAgent(LookupError):
    """No agent, or for a cancel no node, has this path."""


def denied(reason: str = "") -> str:
    """Returns the result the model gets for a call the user refused.

    Args:
        reason: Why, in the user's words; "" for none.
    """
    return f"Denied by user. {reason}".strip()


class Driver:
    """One agent tree, run off the caller's thread, one job at a time.

    Each job (a turn or a compaction) runs on a thread of its own. The driver makes the tree's root node, with
    `view` listening to it, and has `build` put the agent on it. Its own events (jobs, approvals) go to `view` as
    well, so `view` is everything a client reads. Commands are thread-safe and return at once; what they cause
    arrives as events.

    Args:
        build: Builds the agent on the node it's given, with the `approve` function it's given.
        max_cost: The most the whole tree may spend, in USD; None for no limit.
        log_path: The file every model call in the tree is appended to; None for no log.

    Attributes:
        view: The tree's state, for clients to read and subscribe to.
        agent: The agent the driver runs: the root of the tree.
    """

    def __init__(self, build: Build, max_cost: float | None = None, log_path: str | Path | None = None):
        self.view = AgentView()
        self._lock = threading.Lock()  # guards the fields below; never held while an agent works
        self._busy = False
        self._job: threading.Thread | None = None
        self._approval: Approval | None = None
        self._answered = threading.Event()
        self._refusal: str | None = None
        self._ids = itertools.count(1)
        root = AgentNode(max_cost, log_path=log_path, listener=self.view.listen)
        self.agent = build(root, self._approve)

    # ---------------------------------------------------------------- commands

    def send(self, text: str) -> None:
        """Starts a turn on `text`.

        Args:
            text: The user's message.

        Raises:
            Busy: A job is running.
        """
        self._start("turn", lambda: self.agent.run_turn(text), text)

    def compact(self) -> None:
        """Starts compacting the context, whatever its size.

        Raises:
            Busy: A job is running.
        """
        self._start("compact", lambda: self.agent.compact(force=True))

    def cancel(self, path: str = "main") -> None:
        """Stops the agent at `path`, and every agent below it, before its next model call.

        For the root, the job ends "cancelled" and a waiting approval is refused. Any other agent stays cancelled
        until it's resumed.

        Args:
            path: The agent's path, e.g. "main/1".

        Raises:
            UnknownAgent: No node in the tree has `path`.
        """
        if path == self.agent.node.path:
            self._stop()
        else:
            self._node(path).cancel()

    def answer(self, approval_id: int, refusal: str | None = None) -> None:
        """Answers the tool call waiting for approval.

        An id that isn't the one waiting is ignored.

        Args:
            approval_id: The approval's id.
            refusal: None to run the call; otherwise the result the model gets instead.
        """
        with self._lock:
            if self._approval is None or self._approval.id != approval_id:
                return
            self._approval, self._refusal = None, refusal
            self._emit(ApprovalResolved(id=approval_id, refusal=refusal))
            self._answered.set()

    def configure(self, **settings) -> None:
        """Changes settings the agent reads on every request.

        All are checked before any applies.

        Args:
            **settings: Any of `model`, `reasoning` ("low", "medium", "high" or None), `max_cost` (USD, positive,
                or None) and `compact_at` (tokens, at most half the context window, or None).

        Raises:
            ValueError: A setting is unknown, or its value is one the agent would refuse.
        """
        if unknown := settings.keys() - set(LIVE_SETTINGS):
            raise ValueError(f"not a live setting: {', '.join(sorted(unknown))}")
        if settings.get("reasoning") not in REASONING:
            raise ValueError(f"reasoning must be one of {REASONING}, got {settings['reasoning']!r}")
        if (cost := settings.get("max_cost")) is not None and cost <= 0:
            raise ValueError(f"max_cost must be positive, got {cost}")
        if (at := settings.get("compact_at")) is not None and at > self.agent.context_window // 2:
            raise ValueError(
                f"compact_at must leave room for the summary call: at most {self.agent.context_window // 2}"
            )
        for name, value in settings.items():
            setattr(self.agent, name, value)

    def close(self) -> None:
        """Stops whatever runs, waits for it, and releases what the agents' tools hold.

        It can be called at any time.
        """
        self._stop()
        if self._job is not None:
            self._job.join()
        self.agent.close()
        self.agent.node.listener = None  # a run that winds down later can't reach the view

    # ---------------------------------------------------------------- reads

    def agent_at(self, path: str) -> Agent:
        """Returns the agent at `path`, for an inspector to read.

        Drive it only through this driver.

        Args:
            path: The agent's path, e.g. "main/1".

        Raises:
            UnknownAgent: No agent in the tree has `path`.
        """
        agent = self._node(path).agent
        if agent is None:
            raise UnknownAgent(f"{path} has no agent")
        return agent

    # ---------------------------------------------------------------- internals

    def _emit(self, event: ViewEvent) -> None:
        self.view.listen(event)

    def _start(self, kind, work: Callable[[], str], text: str = "") -> None:
        with self._lock:
            if self._busy:
                raise Busy("a job is running; wait for it to end, or stop it")
            self._busy = True
            # Under the lock a stop takes: a stop lands before this (it was for the last job) or after (it holds).
            self.agent.node.reopen()
            self._emit(JobStarted(kind=kind, text=text))
        self._job = threading.Thread(target=self._run, args=(kind, work), name=f"driver {kind}", daemon=True)
        self._job.start()

    def _run(self, kind, work: Callable[[], str]) -> None:
        result, error = None, None
        try:
            result = work()
        except Exception as e:  # noqa: BLE001 - reported in JobEnded; the driver stays usable
            error = f"{type(e).__name__}: {e}"
        with self._lock:  # busy clears in the same step JobEnded is seen, so a client can send again at once
            self._busy = False
            self._emit(JobEnded(kind=kind, result=result, error=error))

    def _approve(self, name: str, arguments: dict) -> str | None:
        with self._lock:
            # A stop that came before this question can't answer it, so it's answered here.
            if self.agent.node.cancelled:
                return NOT_RUN
            self._answered.clear()
            self._refusal = None
            self._approval = Approval(id=next(self._ids), path=self.agent.node.path, tool=name, arguments=arguments)
            self._emit(ApprovalRequested(approval=self._approval))
        self._answered.wait()
        return self._refusal

    def _stop(self) -> None:
        # A stop alone doesn't wake a turn that waits for the user, so a waiting approval is refused too.
        with self._lock:
            self.agent.interrupt()
            approval = self._approval
        if approval is not None:
            self.answer(approval.id, NOT_RUN)

    def _node(self, path: str) -> AgentNode:
        node = next((n for n in self.agent.node.nodes() if n.path == path), None)
        if node is None:
            raise UnknownAgent(path)
        return node
