"""Running an agent from another thread: `Driver`, and what its commands raise."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from jean_code.core.agent import NOT_RUN, Agent
from jean_code.core.events import (
    Approval,
    ApprovalRequested,
    ApprovalResolved,
    JobEnded,
    JobKind,
    JobStarted,
    Queued,
    ViewEvent,
    error_line,
)
from jean_code.core.node import AgentNode
from jean_code.core.view import AgentView

Approve = Callable[[str, dict], str | None]
Build = Callable[[AgentNode, Approve], Agent]


@dataclass(frozen=True)
class _Turn:
    """A turn on the user's message `text`."""

    kind: ClassVar[JobKind] = "turn"
    text: str


@dataclass(frozen=True)
class _Compaction:
    """A compaction of the context, whatever its size."""

    kind: ClassVar[JobKind] = "compact"


_Job = _Turn | _Compaction


class Busy(RuntimeError):
    """The driver can't take this command now.

    A job runs, or for a message, another message already waits for the running job to stop.
    """


def denied(reason: str = "") -> str:
    """Returns the result the model gets for a call the user refused.

    Args:
        reason: Why, in the user's words; "" for none.
    """
    return f"Denied by user. {reason}".strip()


class Driver:
    """One agent tree, run off the caller's thread, one job at a time.

    Each job (a turn or a compaction) runs on a thread of its own; a message sent while one runs stops it, and its
    turn runs next on the same thread. The driver makes the tree's root node, with `view` listening to it, and has
    `build` put the agent on it. Its own events (jobs, approvals) go to `view` as
    well, so `view` is everything a client reads. Commands are thread-safe and return at once; what they cause
    arrives as events.

    Args:
        build: Builds the agent on the node it's given, with the `approve` function it's given.
        max_cost: The most the whole tree may spend, in USD; None for no limit.
        view: The view the tree's events go to; a new one if None. Pass one to subscribe to it before the agent is
            built, and so get every event.

    Attributes:
        view: The tree's state, for clients to read and subscribe to: `view` if given, else a new one.
    """

    def __init__(self, build: Build, max_cost: float | None = None, view: AgentView | None = None):
        self.view = view or AgentView()
        self._lock = threading.Lock()  # guards the fields below; never held while an agent works
        self._busy = False
        self._job: threading.Thread | None = None
        self._queued: _Turn | None = None
        self._approval: Approval | None = None
        self._answered = threading.Event()
        self._refusal: str | None = None
        self._ids = itertools.count(1)
        root = AgentNode(max_cost, listener=self.view.listen)
        self._agent = build(root, self._approve)

    # ---------------------------------------------------------------- commands

    def send(self, text: str) -> None:
        """Starts a turn on `text`, or redirects the running job to it.

        While a job runs, it's stopped as `cancel()` stops the root, and the turn on `text` starts once it has
        ended.

        Args:
            text: The user's message.

        Raises:
            Busy: A job is running, and a message already waits for it to stop.
        """
        with self._lock:
            if self._busy:
                if self._queued is not None:
                    raise Busy("a message already waits for the running job to stop")
                self._queued = _Turn(text)
                self._emit(Queued(text=text))
                self._stop_locked()
                return
            self._begin_locked(job := _Turn(text))
        self._spawn(job)

    def compact(self) -> None:
        """Starts compacting the context, whatever its size.

        Raises:
            Busy: A job is running.
        """
        with self._lock:
            if self._busy:
                raise Busy("a job is running; wait for it to end, or stop it")
            self._begin_locked(job := _Compaction())
        self._spawn(job)

    def cancel(self, path: str = "main") -> None:
        """Stops every agent at or below the node at `path` before its next model call.

        For the root, the job ends "cancelled" and a waiting approval is refused. A subagent stays cancelled until
        it's resumed, and a group for good.

        Args:
            path: The node's path, e.g. "main/1".

        Raises:
            UnknownNode: No node in the tree has `path`.
        """
        if path == self._agent.node.path:
            with self._lock:
                self._stop_locked()
        else:
            self._agent.node.find(path).cancel()

    def answer(self, approval_id: int, refusal: str | None = None) -> None:
        """Answers the tool call waiting for approval.

        An id that isn't the one waiting is ignored.

        Args:
            approval_id: The approval's id.
            refusal: None to run the call; otherwise the result the model gets instead.
        """
        with self._lock:
            if self._approval is not None and self._approval.id == approval_id:
                self._resolve_locked(self._approval, refusal)

    def configure(self, **settings) -> None:
        """Changes settings the agent reads on every request.

        All are checked before any applies. A change to `model` or `reasoning` invalidates the whole prompt cache, so
        the next request pays the uncached price for every token of the prompt.

        Args:
            **settings: Any of `model`, `reasoning` ("low", "medium", "high" or None) and `compact_at` (tokens, at
                most half the context window, or None).

        Raises:
            ValueError: A setting is unknown, or its value is one the agent would refuse.
        """
        self._agent.configure(**settings)

    def set_limit(self, path: str, limit: float | None) -> None:
        """Sets the most the node at `path` and everything below it may spend.

        The limit holds until something replaces it. Resuming a subagent does: its limit becomes what it has spent
        plus its profile's default, which can be lower than one set here.

        Args:
            path: The node's path, e.g. "main" or "main/1".
            limit: USD, positive; None for no limit.

        Raises:
            ValueError: `limit` isn't positive.
            UnknownNode: No node in the tree has `path`.
        """
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        self._agent.node.find(path).limit = limit

    def close(self) -> None:
        """Stops whatever runs, waits for it, and releases what the agents' tools hold.

        It can be called at any time. A message waiting for the job to stop is dropped.
        """
        with self._lock:
            self._queued = None
            self._stop_locked()
        if self._job is not None:
            self._job.join()
        self._agent.close()
        self._agent.node.listener = None  # a run that winds down later can't reach the view

    # ---------------------------------------------------------------- internals

    def _emit(self, event: ViewEvent) -> None:
        self.view.listen(event)

    def _begin_locked(self, job: _Job) -> None:
        self._busy = True
        # Under the lock a stop takes: a stop lands before this (it was for the last job) or after (it holds).
        self._agent.node.reopen()
        self._emit(JobStarted(kind=job.kind, text=job.text if isinstance(job, _Turn) else ""))

    def _spawn(self, job: _Job) -> None:
        self._job = threading.Thread(target=self._run, args=(job,), name=f"driver {job.kind}", daemon=True)
        self._job.start()

    def _run(self, job: _Job) -> None:
        while True:
            result, error = None, None
            try:
                match job:
                    case _Turn(text=text):
                        result = self._agent.run_turn(text)
                    case _Compaction():
                        result = self._agent.compact(force=True)
            except Exception as e:  # noqa: BLE001 - reported in JobEnded; the driver stays usable
                error = error_line(e)  # the agent recorded its stack as it left
            with self._lock:  # busy clears in the same step JobEnded is seen, so a client can send again at once
                self._busy = False
                self._emit(JobEnded(kind=job.kind, result=result, error=error))
                if self._queued is None:
                    return
                job, self._queued = self._queued, None
                self._begin_locked(job)

    def _approve(self, name: str, arguments: dict) -> str | None:
        with self._lock:
            # A stop that came before this question can't answer it, so it's answered here.
            if self._agent.node.cancelled:
                return NOT_RUN
            self._answered.clear()
            self._refusal = None
            self._approval = Approval(id=next(self._ids), path=self._agent.node.path, tool=name, arguments=arguments)
            self._emit(ApprovalRequested(approval=self._approval))
        self._answered.wait()
        return self._refusal

    def _stop_locked(self) -> None:
        self._agent.interrupt()
        # A stop alone doesn't wake a turn that waits for the user, so a waiting approval is refused too.
        if self._approval is not None:
            self._resolve_locked(self._approval, NOT_RUN)

    def _resolve_locked(self, approval: Approval, refusal: str | None) -> None:
        self._emit(ApprovalResolved(id=approval.id, refusal=refusal))
        self._approval, self._refusal = None, refusal
        self._answered.set()
