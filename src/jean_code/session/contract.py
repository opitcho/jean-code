"""What clients and a `Session` exchange: settings, the session's own events, the state, and the errors.

The state is a fold of the events (`apply`): a subscriber that applies every event after its snapshot holds the same
state as `Session.state()`. No field of it is read from an agent.
"""

from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jean_code.core.agent import CONTEXT_WINDOW, DEFAULT_MODEL
from jean_code.core.events import (
    AgentAdded,
    Charged,
    Compaction,
    Delta,
    MessageAdded,
    TurnEnd,
    TurnEnded,
    TurnStarted,
)
from jean_code.core.llm import Message
from jean_code.core.usage import ModelCall

# ---------------------------------------------------------------- settings


class Settings(BaseModel):
    """What the agent is built from. `model`, `reasoning`, `max_cost` and `compact_at` apply to a running agent;
    `kind`, `sandbox` and `network` decide which agent gets built, so changing one starts a new chat."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["coding", "orchestrator"] = "coding"
    sandbox: bool = False  # coding agent: run bash in a Docker container
    network: bool = True  # whether containers get a network
    model: str = DEFAULT_MODEL
    reasoning: Literal["low", "medium", "high"] | None = "medium"  # None sends no reasoning setting
    max_cost: float | None = None  # USD, for the whole tree
    compact_at: int | None = 256_000  # context tokens that trigger compaction after a turn

    # The checks Agent.__init__ runs, here too so a value it would refuse is never applied live.
    @field_validator("max_cost")
    @classmethod
    def _positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError(f"max_cost must be positive, got {value}")
        return value

    @field_validator("compact_at")
    @classmethod
    def _room_for_the_summary(cls, value: int | None) -> int | None:
        if value is not None and value > CONTEXT_WINDOW // 2:
            raise ValueError(f"compact_at must leave room for the summary call: at most {CONTEXT_WINDOW // 2}")
        return value


LIVE_SETTINGS = ("model", "reasoning", "max_cost", "compact_at")  # read on every request: no rebuild needed


# ---------------------------------------------------------------- the session's own events


class Approval(BaseModel):
    """A tool call waiting for the user: answer it with `Session.answer(id, refusal)`."""

    model_config = ConfigDict(frozen=True)
    id: int
    path: str
    tool: str
    arguments: dict[str, Any]


class JobStarted(BaseModel):
    """The session started a turn (`text` is the user's message) or a compaction; it's busy until `JobEnded`."""

    model_config = ConfigDict(frozen=True)
    type: Literal["job_started"] = "job_started"
    kind: Literal["turn", "compact"]
    text: str = ""


class JobEnded(BaseModel):
    """The job is over and the session is idle. `result` is how the turn ended, or what `compact` returned; None
    when it raised, with `error` saying what."""

    model_config = ConfigDict(frozen=True)
    type: Literal["job_ended"] = "job_ended"
    kind: Literal["turn", "compact"]
    result: str | None = None
    error: str | None = None


class ApprovalRequested(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["approval_requested"] = "approval_requested"
    approval: Approval


class ApprovalResolved(BaseModel):
    """The approval was answered: `refusal` None ran the call, otherwise it's the call's result."""

    model_config = ConfigDict(frozen=True)
    type: Literal["approval_resolved"] = "approval_resolved"
    id: int
    refusal: str | None


class SettingsChanged(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["settings_changed"] = "settings_changed"
    settings: Settings


class Reset(BaseModel):
    """A new chat: everything before it is gone, and the new agent's `AgentAdded` follows."""

    model_config = ConfigDict(frozen=True)
    type: Literal["reset"] = "reset"
    settings: Settings


SessionEvent = Annotated[
    AgentAdded
    | MessageAdded
    | Delta
    | TurnStarted
    | TurnEnded
    | Compaction
    | Charged
    | JobStarted
    | JobEnded
    | ApprovalRequested
    | ApprovalResolved
    | SettingsChanged
    | Reset,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------- the state


@dataclass(frozen=True)
class Item:
    """A message an agent has had, with a uid that stays the same when a compaction takes it out of the context."""

    uid: int
    message: Message
    in_context: bool = True


@dataclass
class AgentState:
    """One agent, as the events describe it."""

    path: str
    profile: str
    task: str
    model: str
    items: list[Item] = field(default_factory=list)  # every message it has had, in timeline order
    context: list[int] = field(default_factory=list)  # uids of its messages now, in order: its `messages`
    draft: dict[str, str] = field(default_factory=dict)  # a reply streaming in: "content" and "reasoning" so far
    running: bool = False  # between TurnStarted and TurnEnded
    end: TurnEnd | None = None  # how its last turn ended
    error: str | None = None  # what its last turn raised
    calls: list[ModelCall] = field(default_factory=list)  # its own calls
    spent: float = 0.0  # USD, its own calls and its subtree's
    compactions: list[Compaction] = field(default_factory=list)
    context_tokens: int = 0

    def item(self, uid: int) -> Item:
        return next(i for i in self.items if i.uid == uid)


@dataclass
class SessionState:
    settings: Settings
    agents: dict[str, AgentState] = field(default_factory=dict)  # by path, in the order they were added
    busy: bool = False  # between JobStarted and JobEnded
    approval: Approval | None = None  # the call waiting for the user
    last_job: JobEnded | None = None
    calls: list[ModelCall] = field(default_factory=list)  # every call in the tree, in charge order


def apply(state: SessionState, event: SessionEvent) -> None:
    """Fold one event into the state, in place."""
    match event:
        case AgentAdded():
            state.agents[event.path] = AgentState(event.path, event.profile, event.task, event.model)
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
        case Compaction():
            agent = state.agents[event.path]
            agent.compactions.append(event)
            agent.context_tokens = event.context_tokens
            if event.status == "compacted" and event.summary is not None:
                _splice(agent, event.start, event.removed, event.summary)
        case Charged():
            state.calls.append(event.call)
            state.agents[event.path].calls.append(event.call)
            for path, agent in state.agents.items():  # the call counts toward every ancestor
                if event.path == path or event.path.startswith(path + "/"):
                    agent.spent += event.call.cost or 0
        case JobStarted():
            state.busy = True
        case JobEnded():
            state.busy, state.last_job = False, event
        case ApprovalRequested():
            state.approval = event.approval
        case ApprovalResolved():
            if state.approval is not None and state.approval.id == event.id:
                state.approval = None
        case SettingsChanged():
            state.settings = event.settings
        case Reset():
            state.agents, state.busy, state.approval, state.last_job, state.calls = {}, False, None, None, []
            state.settings = event.settings


def _splice(agent: AgentState, start: int, removed: int, summary: Message) -> None:
    """`context[start:start + removed]` was replaced by `summary`. The replaced items stay, out of the context, and
    the summary's item goes right after them, so the timeline shows it where the turns it replaced were."""
    gone = agent.context[start : start + removed]
    uid = max(i.uid for i in agent.items) + 1
    agent.items = [replace(i, in_context=False) if i.uid in gone else i for i in agent.items]
    last = max(n for n, i in enumerate(agent.items) if i.uid in gone)
    agent.items.insert(last + 1, Item(uid, summary))
    agent.context[start : start + removed] = [uid]


def snapshot(state: SessionState) -> SessionState:
    """A copy a client can keep: the lists and dicts are copied, the models and items, which never change, aren't."""
    agents = {
        path: replace(
            a,
            items=list(a.items),
            context=list(a.context),
            draft=dict(a.draft),
            calls=list(a.calls),
            compactions=list(a.compactions),
        )
        for path, a in state.agents.items()
    }
    return replace(state, agents=agents, calls=list(state.calls))


@dataclass(frozen=True)
class AgentDetails:
    """What an inspector shows about one agent beyond its state, read from the agent when asked: nothing announces
    when these change."""

    model: str
    reasoning: str | None
    max_tool_rounds: int
    max_cost: float | None
    needs_approval: list[str]
    tools: list[str]
    skills: list[str]  # the skills it may load
    loaded_skills: list[str]
    compact_at: int | None
    keep_first: int
    keep_last: int
    context_window: int
    head_end: int | None


class Busy(RuntimeError):
    """A turn or a compaction is running; wait for `JobEnded`."""


class UnknownAgent(LookupError):
    """No agent (or, for `cancel`, no node) has this path."""
