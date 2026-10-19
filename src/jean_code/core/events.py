"""The events an agent emits as it works, and those a `Driver` emits around it.

The agent's events are enough to rebuild its transcript: `AgentAdded` starts it, `MessageAdded` appends to it, and a
`Compaction` with status "compacted" replaces part of it with a summary. `ViewEvent` is every event of both kinds.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jean_code.core.llm import Message, UserMessage
from jean_code.core.usage import ModelCall

# How a turn ended. The values are the text shown for it, and keys once stored: don't reword them.
TurnEnd = Literal["answered", "budget spent", "cancelled", "context full", "out of tool rounds"]

# What a driver's job is: a turn, or a compaction.
JobKind = Literal["turn", "compact"]


class Event(BaseModel):
    """Something an agent did, as it reports it.

    Attributes:
        path: The node of the agent the event is about, e.g. "main/1".
    """

    model_config = ConfigDict(frozen=True)
    path: str


class Settings(BaseModel):
    """What an agent reads on every request.

    Attributes:
        model: Its model id.
        reasoning: Its reasoning effort: "low", "medium" or "high"; None sends no setting.
        compact_at: The context size in tokens at which it compacts after a turn; None for never on its own.
        max_tool_rounds: A turn makes at most 1 + this many model calls.
        keep_first: Turns at the start that a compaction keeps verbatim.
        keep_last: Turns at the end that a compaction keeps verbatim.
        context_window: The model's context window, in tokens.
    """

    model_config = ConfigDict(frozen=True)
    model: str
    reasoning: str | None
    compact_at: int | None
    max_tool_rounds: int
    keep_first: int
    keep_last: int
    context_window: int


class AgentAdded(Event):
    """An agent was built on the node at `path`.

    It comes before any other event from that path.

    Attributes:
        profile: What kind of agent it is; "" for the root.
        task: What its node was created for; "" for the root.
        settings: Its settings.
        limit: Its node's limit in USD; None for none.
        tools: The tool schemas it sends with every request.
        needs_approval: The names of the tools whose calls wait for approval, sorted.
        skills: The names of the skills it can load.
    """

    type: Literal["agent_added"] = "agent_added"
    profile: str
    task: str
    settings: Settings
    limit: float | None
    tools: list[dict]
    needs_approval: list[str]
    skills: list[str]


class Configured(Event):
    """The agent's settings changed; `settings` is all of them now."""

    type: Literal["configured"] = "configured"
    settings: Settings


class LimitSet(Event):
    """The limit of the node at `path` changed.

    Attributes:
        limit: The most its subtree may spend now, in USD; None for no limit.
    """

    type: Literal["limit_set"] = "limit_set"
    limit: float | None


class SkillLoaded(Event):
    """The agent loaded a skill for the first time.

    Attributes:
        name: The skill's name.
    """

    type: Literal["skill_loaded"] = "skill_loaded"
    name: str


class MessageAdded(Event):
    """`message` was appended to the agent's transcript.

    Attributes:
        message: The message appended.
        context_tokens: The agent's context size with the message in it.
    """

    type: Literal["message"] = "message"
    message: Message
    context_tokens: int


class Delta(Event):
    """A piece of a reply as it streams in.

    The whole reply follows as a `MessageAdded`, unless the call fails or is cut off before any text; the turn's
    `TurnEnded` comes after either way.

    Attributes:
        kind: Whether the piece is reply text or reasoning.
        text: The piece.
    """

    type: Literal["delta"] = "delta"
    kind: Literal["content", "reasoning"]
    text: str


class TurnStarted(Event):
    """A turn started.

    Attributes:
        text: The user's input, before any command expands it.
    """

    type: Literal["turn_started"] = "turn_started"
    text: str


class TurnEnded(Event):
    """A turn ended; every `TurnStarted` gets exactly one, however the turn ends.

    Attributes:
        end: How the turn ended; None when it raised.
        error: What it raised, if it did.
        context_tokens: The agent's context size at the end.
    """

    type: Literal["turn_ended"] = "turn_ended"
    end: TurnEnd | None
    error: str | None = None
    context_tokens: int


class Compaction(Event):
    """A compaction attempt that got past the checks.

    Only status "compacted" changed the transcript: `messages[start:start + removed]` was replaced by `summary`.

    Attributes:
        status: "compacted", or why the transcript was left unchanged.
        start: The index of the first message replaced.
        removed: How many messages were replaced.
        summary: The summary message, if compacted.
        tokens_before: The context size before.
        tokens_after: The context size after, if compacted.
        cost: The summary call's cost, if the attempt made one and it reported a cost.
        context_tokens: The agent's context size now.
    """

    type: Literal["compaction"] = "compaction"
    status: str
    start: int
    removed: int
    summary: UserMessage | None = None
    tokens_before: int
    tokens_after: int | None = None
    cost: float | None = None
    context_tokens: int


def error_line(error: BaseException) -> str:
    """Returns `error` as one line, "<type>: <message>"."""
    return f"{type(error).__name__}: {error}"


class ErrorCaught(Event):
    """An exception an agent's code caught, with its traceback.

    It's emitted once per exception, where the agent swallowed it or where it left the agent. Not every one is a
    bug: the model's malformed tool arguments, and refusals a tool raises on purpose, are recorded too. The model
    doesn't see it: a tool's result still holds only `error`.

    Attributes:
        error: The exception as one line, as `error_line` gives it.
        traceback: The whole traceback, chained causes included, from where the agent caught it down to where it
            was raised.
        tool_call_id: The tool call that raised it, if one did; its result follows.
    """

    type: Literal["error"] = "error"
    error: str
    traceback: str
    tool_call_id: str | None = None


class RunEnded(Event):
    """The subagent at `path` finished a task; its parent can now receive the result.

    Attributes:
        status: "done", "stopped" or "failed".
        reason: Why it stopped, for "stopped".
    """

    type: Literal["run_ended"] = "run_ended"
    status: Literal["done", "stopped", "failed"]
    reason: TurnEnd | None = None


class Delivered(Event):
    """The parent's model received the latest result of the subagent at `path`."""

    type: Literal["delivered"] = "delivered"


class Charged(Event):
    """A paid model call, charged to the node at `path`.

    Attributes:
        call: What the call used and cost.
    """

    type: Literal["charged"] = "charged"
    call: ModelCall


AgentEvent = Annotated[
    AgentAdded
    | Configured
    | LimitSet
    | SkillLoaded
    | MessageAdded
    | Delta
    | TurnStarted
    | TurnEnded
    | Compaction
    | ErrorCaught
    | RunEnded
    | Delivered
    | Charged,
    Field(discriminator="type"),
]


class Approval(BaseModel):
    """A tool call waiting for the user's yes or no.

    Attributes:
        id: Identifies this approval among all those the driver asked.
        path: The agent asking.
        tool: The tool's name.
        arguments: The call's arguments.
    """

    model_config = ConfigDict(frozen=True)
    id: int
    path: str
    tool: str
    arguments: dict[str, Any]


class JobStarted(BaseModel):
    """A driver started a turn or a compaction; it's busy until `JobEnded`.

    Attributes:
        kind: A turn, or a compaction.
        text: The user's message, for a turn.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["job_started"] = "job_started"
    kind: JobKind
    text: str = ""


class Queued(BaseModel):
    """A message came in while a job ran; its turn starts once the job has stopped.

    Attributes:
        text: The user's message.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["queued"] = "queued"
    text: str


class JobEnded(BaseModel):
    """A driver's job is over, and the driver is idle.

    Attributes:
        kind: A turn, or a compaction.
        result: How the turn ended, or what the compaction returned; None when the job raised.
        error: What the job raised, if it did.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["job_ended"] = "job_ended"
    kind: JobKind
    result: str | None = None
    error: str | None = None


class ApprovalRequested(BaseModel):
    """A tool call waits for the user.

    Attributes:
        approval: The call waiting.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["approval_requested"] = "approval_requested"
    approval: Approval


class ApprovalResolved(BaseModel):
    """A waiting tool call was answered.

    Attributes:
        id: The approval answered.
        refusal: None when the call may run; otherwise the result the model gets instead.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["approval_resolved"] = "approval_resolved"
    id: int
    refusal: str | None


ViewEvent = Annotated[
    AgentAdded
    | Configured
    | LimitSet
    | SkillLoaded
    | MessageAdded
    | Delta
    | TurnStarted
    | TurnEnded
    | Compaction
    | ErrorCaught
    | RunEnded
    | Delivered
    | Charged
    | JobStarted
    | Queued
    | JobEnded
    | ApprovalRequested
    | ApprovalResolved,
    Field(discriminator="type"),
]
