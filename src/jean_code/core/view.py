"""The state of an agent tree, built from its events, and its `AgentView`.

`apply` folds one event into a `TreeState`, so a subscriber that applies every event after its snapshot holds the
same state as `AgentView.state()`. `breakdown` sums a state's spend per agent.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field, replace

from jean_code.core.events import (
    AgentAdded,
    Approval,
    ApprovalRequested,
    ApprovalResolved,
    Charged,
    Compaction,
    Configured,
    Delivered,
    Delta,
    ErrorCaught,
    GroupAdded,
    JobEnded,
    JobStarted,
    LimitSet,
    MessageAdded,
    Queued,
    RunEnded,
    Settings,
    SkillLoaded,
    TurnEnd,
    TurnEnded,
    TurnStarted,
    ViewEvent,
)
from jean_code.core.llm import Message
from jean_code.core.usage import ModelCall, breakdown_row


@dataclass(frozen=True)
class Item:
    """A message an agent has had, under a uid that a compaction doesn't change.

    Attributes:
        uid: Its number in the agent's timeline; never reused.
        message: The message.
        in_context: Whether it's still in the agent's `messages`.
    """

    uid: int
    message: Message
    in_context: bool = True


@dataclass
class AgentState:
    """One agent, as its events describe it.

    Attributes:
        path: The agent's node, e.g. "main/1".
        profile: What kind of agent it is; "" for the root.
        task: What its node was created for.
        settings: Its settings, as of its last `Configured`.
        limit: Its node's limit in USD; None for none.
        tools: The tool schemas it sends with every request.
        needs_approval: The names of the tools whose calls wait for approval, sorted.
        skills: The names of the skills it can load.
        loaded_skills: The skills it has loaded, first load first.
        outcome: How its latest task as a subagent ended; None until one has.
        delivered: Whether its parent's model has received that outcome.
        items: Every message it has had, in timeline order.
        context: The uids of its messages now, in order.
        draft: A reply streaming in: "content" and "reasoning" so far.
        running: Whether a turn of its is running.
        end: How its last turn ended.
        error: What its last turn raised, if it did.
        calls: Its own model calls.
        spent: USD spent by it and every agent below it.
        compactions: Its compaction attempts.
        caught: Every exception its code caught, with its traceback, in order; tools' included. Not `error`,
            which is only what its last turn raised.
        context_tokens: Its context size, as of its last event.
    """

    path: str
    profile: str
    task: str
    settings: Settings
    limit: float | None
    tools: list[dict] = field(default_factory=list)
    needs_approval: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    loaded_skills: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    context: list[int] = field(default_factory=list)
    draft: dict[str, str] = field(default_factory=dict)
    running: bool = False
    end: TurnEnd | None = None
    error: str | None = None
    calls: list[ModelCall] = field(default_factory=list)
    spent: float = 0.0
    compactions: list[Compaction] = field(default_factory=list)
    caught: list[ErrorCaught] = field(default_factory=list)
    context_tokens: int = 0
    outcome: RunEnded | None = None
    delivered: bool = False

    def messages(self) -> tuple[Message, ...]:
        """Returns its messages now, in order: what its next request sends."""
        by_uid = {i.uid: i.message for i in self.items}
        return tuple(by_uid[uid] for uid in self.context)


@dataclass
class GroupState:
    """One group node, as its events describe it.

    Attributes:
        path: The group's node, e.g. "main/1".
        profile: What kind of group it is.
        task: What it was made for.
        limit: Its node's limit in USD; None for none.
        spent: USD spent by every agent below it.
    """

    path: str
    profile: str
    task: str
    limit: float | None
    spent: float = 0.0


@dataclass
class TreeState:
    """An agent tree and its driver, as their events describe them.

    Attributes:
        agents: Every agent, by path, in the order they were added.
        groups: Every group, by path, in the order they were added.
        busy: Whether the driver runs a job.
        approval: The tool call waiting for the user, if any.
        last_job: How the last job ended.
        queued: The message waiting for the running job to stop, if any.
        calls: Every model call in the tree, in charge order.
    """

    agents: dict[str, AgentState] = field(default_factory=dict)
    groups: dict[str, GroupState] = field(default_factory=dict)
    busy: bool = False
    approval: Approval | None = None
    last_job: JobEnded | None = None
    queued: str | None = None
    calls: list[ModelCall] = field(default_factory=list)

    def waiting(self, path: str) -> int:
        """Returns how many subagent results the agent at `path` hasn't received yet.

        Only the subagents directly below `path` count: a nested one's result goes to its own parent.

        Args:
            path: The parent's path, e.g. "main".
        """
        return sum(
            a.outcome is not None and not a.delivered for p, a in self.agents.items() if p.rpartition("/")[0] == path
        )


def apply(state: TreeState, event: ViewEvent) -> None:
    """Folds one event into `state`, in place.

    Args:
        state: The state to update.
        event: An agent's or a driver's event.
    """
    match event:
        case AgentAdded():
            state.agents[event.path] = AgentState(
                event.path,
                event.profile,
                event.task,
                event.settings,
                event.limit,
                tools=event.tools,
                needs_approval=event.needs_approval,
                skills=event.skills,
            )
        case GroupAdded():
            state.groups[event.path] = GroupState(event.path, event.profile, event.task, event.limit)
        case Configured():
            state.agents[event.path].settings = event.settings
        case SkillLoaded():
            state.agents[event.path].loaded_skills.append(event.name)
        case RunEnded():
            agent = state.agents[event.path]
            agent.outcome, agent.delivered = event, False
        case Delivered():
            state.agents[event.path].delivered = True
        case LimitSet():
            # A spawn whose start failed leaves a node with neither state
            if (node := state.agents.get(event.path) or state.groups.get(event.path)) is not None:
                node.limit = event.limit
        case MessageAdded():
            agent = state.agents[event.path]
            uid = max((i.uid for i in agent.items), default=0) + 1
            agent.items.append(Item(uid, event.message))
            agent.context.append(uid)
            agent.draft = {}
            agent.context_tokens = event.context_tokens
        case Delta():
            draft = state.agents[event.path].draft
            draft[event.kind] = draft.get(event.kind, "") + event.text
        case TurnStarted():
            agent = state.agents[event.path]
            agent.running, agent.end, agent.error = True, None, None
        case TurnEnded():
            agent = state.agents[event.path]
            agent.running, agent.end, agent.error, agent.draft = False, event.end, event.error, {}
            agent.context_tokens = event.context_tokens
        case ErrorCaught():
            state.agents[event.path].caught.append(event)
        case Compaction():
            agent = state.agents[event.path]
            agent.compactions.append(event)
            agent.context_tokens = event.context_tokens
            if event.status == "compacted" and event.summary is not None:
                _splice(agent, event.start, event.removed, event.summary)
        case Charged():
            state.calls.append(event.call)
            state.agents[event.path].calls.append(event.call)
            for path, node in [*state.agents.items(), *state.groups.items()]:  # it counts toward every ancestor
                if event.path == path or event.path.startswith(path + "/"):
                    node.spent += event.call.cost or 0
        case JobStarted():
            state.busy, state.queued = True, None
        case Queued():
            state.queued = event.text
        case JobEnded():
            state.busy, state.last_job = False, event
        case ApprovalRequested():
            state.approval = event.approval
        case ApprovalResolved():
            if state.approval is not None and state.approval.id == event.id:
                state.approval = None


def _splice(agent: AgentState, start: int, removed: int, summary: Message) -> None:
    # The replaced items stay, out of the context; the summary goes right after them, where the turns it replaced were.
    gone = agent.context[start : start + removed]
    uid = max(i.uid for i in agent.items) + 1
    agent.items = [replace(i, in_context=False) if i.uid in gone else i for i in agent.items]
    last = max(n for n, i in enumerate(agent.items) if i.uid in gone)
    agent.items.insert(last + 1, Item(uid, summary))
    agent.context[start : start + removed] = [uid]


def breakdown(state: TreeState) -> list[dict]:
    """Returns one `breakdown_row` per agent, to see where spend went.

    Each row sums the agent's own calls.

    Args:
        state: The tree's state.
    """
    return [breakdown_row(a.path, a.task, a.calls) for a in state.agents.values()]


def snapshot(state: TreeState) -> TreeState:
    """Returns a copy of `state` that lateDeltar events don't change.

    Args:
        state: The state to copy.
    """
    agents = {
        path: replace(
            a,
            items=list(a.items),
            context=list(a.context),
            draft=dict(a.draft),
            loaded_skills=list(a.loaded_skills),
            calls=list(a.calls),
            compactions=list(a.compactions),
            caught=list(a.caught),
        )
        for path, a in state.agents.items()
    }
    groups = {path: replace(g) for path, g in state.groups.items()}
    return replace(state, agents=agents, groups=groups, calls=list(state.calls))


class Subscription:
    """The events a subscriber gets after its snapshot.

    Use it as `with view.subscribe() as (snapshot, events):`, so it's removed when the subscriber is done.

    Args:
        view: The view it subscribes to.
        state: The state when it subscribed.
    """

    def __init__(self, view: AgentView, state: TreeState):
        self.view = view
        self.state = state
        self.queue: queue.SimpleQueue[ViewEvent] = queue.SimpleQueue()

    def get(self, timeout: float | None = None) -> ViewEvent:
        """Returns the next event, waiting for it.

        Args:
            timeout: Seconds to wait; None to wait as long as it takes.

        Raises:
            queue.Empty: No event arrived within `timeout`.
        """
        return self.queue.get(timeout=timeout)

    def close(self) -> None:
        """Stops the events; the subscription receives nothing more."""
        self.view._unsubscribe(self)

    def __enter__(self) -> tuple[TreeState, Subscription]:
        return self.state, self

    def __exit__(self, *exc) -> None:
        self.close()


class AgentView:
    """The state of an agent tree, built from its events, for any number of readers.

    Thread-safe: events may arrive from every agent's thread at once, and `listen` may be called under any lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = TreeState()
        self._subscribers: set[Subscription] = set()

    def listen(self, event: ViewEvent) -> None:
        """Folds `event` into the state and passes it to every subscriber.

        Holds the view's lock only while it updates the state and queues the event for subscribers, and takes no other
        lock, so it may be called while holding any other lock.

        Args:
            event: An agent's or a driver's event.
        """
        with self._lock:
            apply(self._state, event)
            for subscription in self._subscribers:
                subscription.queue.put_nowait(event)

    def state(self) -> TreeState:
        """Returns a copy of the state now."""
        with self._lock:
            return snapshot(self._state)

    def subscribe(self) -> Subscription:
        """Returns the state now and a queue of every event after it.

        No event is missed or repeated in between.
        """
        with self._lock:  # the lock `listen` takes, so no event falls between the snapshot and the queue
            subscription = Subscription(self, snapshot(self._state))
            self._subscribers.add(subscription)
        return subscription

    def _unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._subscribers.discard(subscription)
