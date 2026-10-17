"""The one interface clients drive an agent through.

`Session` holds the agent; a client sends it commands and follows it through a `Subscription`. `build_agent` is how
a session builds its agent from `Settings` by default.

Typical usage:

    session = Session(Settings(kind="coding"))
    with session.subscribe() as (state, events):
        session.send("Fix the failing test")
        for event in events:
            if isinstance(event, JobEnded):
                break
    session.close()
"""

import itertools
import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from jean_code.agents.coding import coding_agent
from jean_code.agents.experts import orchestrator_agent
from jean_code.config import USAGE_LOG
from jean_code.core.agent import NOT_RUN, Agent
from jean_code.core.llm import render_request
from jean_code.core.node import AgentNode
from jean_code.session.contract import (
    LIVE_SETTINGS,
    AgentDetails,
    Approval,
    ApprovalRequested,
    ApprovalResolved,
    Busy,
    JobEnded,
    JobStarted,
    Reset,
    SessionEvent,
    SessionState,
    Settings,
    SettingsChanged,
    UnknownAgent,
    apply,
    snapshot,
)

Approve = Callable[[str, dict], str | None]
Build = Callable[[Settings, AgentNode, Approve], Agent]


def denied(reason: str = "") -> str:
    """The result the model gets for a call the user refused."""
    return f"Denied by user. {reason}".strip()


def build_agent(settings: Settings, node: AgentNode, approve: Approve) -> Agent:
    """The agent `settings` describe, on `node`: the orchestrator, or the coding agent with its shell where
    `settings.sandbox` says."""
    common: dict[str, Any] = {"reasoning": settings.reasoning, "compact_at": settings.compact_at, "node": node}
    if settings.kind == "orchestrator":  # nothing it calls needs approval
        return orchestrator_agent(settings.model, network=settings.network, **common)
    return coding_agent(settings.model, sandbox=settings.sandbox, network=settings.network, approve=approve, **common)


class Subscription:
    """A client's view: the state when it subscribed, then every event after it. Use it as
    `with session.subscribe() as (snapshot, events):`, so it's removed when the client is done."""

    def __init__(self, session: "Session", state: SessionState):
        self.session = session
        self.state = state
        self.queue: queue.Queue[SessionEvent] = queue.Queue()

    def get(self, timeout: float | None = None) -> SessionEvent:
        """The next event; raises `queue.Empty` after `timeout` seconds (None: wait as long as it takes)."""
        return self.queue.get(timeout=timeout)

    def __iter__(self) -> Iterator[SessionEvent]:
        while True:
            yield self.get()

    def close(self) -> None:
        self.session._unsubscribe(self)

    def __enter__(self) -> tuple[SessionState, "Subscription"]:
        return self.state, self

    def __exit__(self, *exc) -> None:
        self.close()


class Session:
    """One agent, with its subagents, shared by any number of clients.

    Commands can be called from any thread and return at once; what they cause reaches every subscriber as events.
    A turn or a compaction is a job: one runs at a time, between `JobStarted` and `JobEnded`, and starting another
    meanwhile raises `Busy`. A tool call that needs approval waits until a client calls `answer`.

    Args:
        settings: What to build the agent from.
        build: Builds the agent on the node it's given, with the `approve` function it's given (`build_agent`).
        log_path: Where every model call in the tree is logged; None for no log.
    """

    def __init__(
        self, settings: Settings | None = None, build: Build = build_agent, log_path: str | Path | None = USAGE_LOG
    ):
        self._build = build
        self._log_path = log_path
        self._lock = threading.Lock()  # guards the state and the subscribers; never held while calling an agent
        self._state = SessionState(settings or Settings())
        self._subscribers: set[Subscription] = set()
        self._job: threading.Thread | None = None
        self._ids = itertools.count(1)
        self._answered = threading.Event()
        self._refusal: str | None = None
        self._stopped = False  # main was stopped during this job: an approval it asks for now is refused at once
        self._agent = self._new_agent()

    # ---------------------------------------------------------------- commands

    def send(self, text: str) -> None:
        """Start a turn on `text`. Raises `Busy` while a job runs."""
        self._start("turn", lambda: self._agent.run_turn(text), text)

    def compact(self) -> None:
        """Compact the context now, whatever its size. Raises `Busy` while a job runs."""
        self._start("compact", lambda: self._agent.compact(force=True))

    def cancel(self, path: str = "main") -> None:
        """Stop the agent at `path` after the call it's making, and everything below it. The main agent's job ends
        "interrupted"; any other path is cancelled until it's resumed. Raises `UnknownAgent` for a path not in the
        tree."""
        if path == self._agent.node.path:
            self._stop_main()
        else:
            self._node(path).cancel()

    def answer(self, approval_id: int, refusal: str | None = None) -> None:
        """Answer the call waiting for approval: None runs it, otherwise `refusal` is its result. An id that isn't the
        one waiting (answered already, or stopped) is ignored."""
        with self._lock:
            if self._state.approval is None or self._state.approval.id != approval_id:
                return
            self._refusal = refusal
            self._publish_locked(ApprovalResolved(id=approval_id, refusal=refusal))
            self._answered.set()

    def configure(self, patch: dict[str, Any]) -> None:
        """Change settings: the whole result is validated before anything applies (`ValidationError` otherwise).
        Live settings apply to the running agent; any other change starts a new chat (`Busy` while a job runs)."""
        current = self._state.settings
        settings = Settings.model_validate(current.model_dump() | patch)
        changed = {name for name in Settings.model_fields if getattr(settings, name) != getattr(current, name)}
        if changed - set(LIVE_SETTINGS):
            self.reset(settings)
            return
        agent = self._agent
        agent.model, agent.reasoning = settings.model, settings.reasoning
        agent.max_cost, agent.compact_at = settings.max_cost, settings.compact_at
        self._publish(SettingsChanged(settings=settings))

    def reset(self, settings: Settings | None = None) -> None:
        """A new chat: close the agent tree and build a new one, from `settings` if given. Raises `Busy` while a job
        runs."""
        if self._state.busy:
            raise Busy("a job is running; stop it first")
        self._close_tree()
        self._publish(Reset(settings=settings or self._state.settings))
        self._agent = self._new_agent()

    def close(self) -> None:
        """Stop whatever runs, wait for it, and release what the agents' tools hold. Any time."""
        self._stop_main()
        if self._job is not None:
            self._job.join()
        self._close_tree()

    # ---------------------------------------------------------------- reads

    def subscribe(self) -> Subscription:
        """The state now, plus a queue of every event after it: taken under the publishing lock, so nothing is
        missed or seen twice in between."""
        with self._lock:
            subscription = Subscription(self, snapshot(self._state))
            self._subscribers.add(subscription)
        return subscription

    def state(self) -> SessionState:
        with self._lock:
            return snapshot(self._state)

    def pending(self) -> int:
        """Background subagent results the main agent's model hasn't received: no event marks a run finishing, so
        this reads the agent."""
        subagents = self._agent.subagents
        return subagents.pending if subagents else 0

    def credits(self) -> dict | None:
        """The API key's usage and limit, if the agent's client reports them (OpenRouter does); None if it can't.
        Raises what the client raises."""
        credits = getattr(self._agent.client, "credits", None)
        return credits() if credits else None

    def breakdown(self) -> list[dict]:
        """One row per agent, from its own calls: the cost drivers `AgentNode.breakdown` reports, from the state."""
        rows = []
        for agent in self.state().agents.values():
            calls = agent.calls
            known = [c.cost for c in calls if c.cost is not None]
            rows.append(
                {
                    "path": agent.path,
                    "task": agent.task,
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

    def inspect(self, path: str) -> AgentDetails:
        """What an inspector shows about the agent at `path` beyond its state. Raises `UnknownAgent`."""
        agent = self._agent_at(path)
        return AgentDetails(
            model=agent.model,
            reasoning=agent.reasoning,
            max_tool_rounds=agent.max_tool_rounds,
            max_cost=agent.max_cost,
            needs_approval=sorted(agent.needs_approval),
            tools=list(agent._tools),
            skills=sorted(agent.skills.catalog),
            loaded_skills=list(agent.skills.loaded),
            compact_at=agent.compact_at,
            keep_first=agent.keep_first,
            keep_last=agent.keep_last,
            context_window=agent.context_window,
            head_end=agent.head_end,
        )

    def last_request(self, path: str) -> dict | None:
        """The last request the agent at `path` sent, as the wire dict; None before its first. Raises
        `UnknownAgent`."""
        request = self._agent_at(path).last_request
        return render_request(request) if request else None

    # ---------------------------------------------------------------- internals

    def _new_agent(self) -> Agent:
        root = AgentNode(self._state.settings.max_cost, log_path=self._log_path, listener=self._publish)
        return self._build(self._state.settings, root, self._approve)

    def _start(self, kind, work: Callable[[], str], text: str = "") -> None:
        with self._lock:
            if self._state.busy:
                raise Busy("a job is running; wait for it to end, or stop it")
            self._stopped = False
            self._publish_locked(JobStarted(kind=kind, text=text))
        self._job = threading.Thread(target=self._run_job, args=(kind, work), name=f"session {kind}", daemon=True)
        self._job.start()

    def _run_job(self, kind, work: Callable[[], str]) -> None:
        """The job thread: run the work, then end the job however it went. Its end clears `busy` in the same
        publish, so a client that sees `JobEnded` can start the next job at once."""
        result, error = None, None
        try:
            result = work()
        except Exception as e:  # noqa: BLE001 - reported in JobEnded; the session stays usable
            error = f"{type(e).__name__}: {e}"
        self._publish(JobEnded(kind=kind, result=result, error=error))

    def _approve(self, name: str, arguments: dict) -> str | None:
        """The agent's `approve`, on the job thread: wait until a client answers, or the main agent is stopped."""
        with self._lock:
            # A stop that came before this question can't answer it, so it's answered here.
            if self._stopped:
                return NOT_RUN
            self._answered.clear()
            self._refusal = None
            approval = Approval(id=next(self._ids), path=self._agent.node.path, tool=name, arguments=arguments)
            self._publish_locked(ApprovalRequested(approval=approval))
        self._answered.wait()
        return self._refusal

    def _stop_main(self) -> None:
        """Interrupt the main agent and answer a waiting approval with NOT_RUN: an interrupt alone doesn't wake a turn
        that waits for the user."""
        with self._lock:
            self._stopped = True
            approval = self._state.approval
        self._agent.interrupt()
        if approval is not None:
            self.answer(approval.id, NOT_RUN)

    def _close_tree(self) -> None:
        """Close the agent (it waits for its runs) and stop listening to its tree: a run that winds down later
        can't reach the next tree's state."""
        self._agent.close()
        self._agent.node.listener = None

    def _node(self, path: str) -> AgentNode:
        node = next((n for n in self._agent.node.nodes() if n.path == path), None)
        if node is None:
            raise UnknownAgent(path)
        return node

    def _agent_at(self, path: str) -> Agent:
        agent = self._node(path).agent
        if agent is None:
            raise UnknownAgent(f"{path} has no agent")
        return agent

    def _publish(self, event: SessionEvent) -> None:
        """The root node's listener, and the only writer of the state."""
        with self._lock:
            self._publish_locked(event)

    def _publish_locked(self, event: SessionEvent) -> None:
        apply(self._state, event)
        for subscription in self._subscribers:
            subscription.queue.put_nowait(event)

    def _unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._subscribers.discard(subscription)
