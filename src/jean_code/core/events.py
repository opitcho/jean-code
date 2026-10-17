"""What agents tell the outside world: typed events, emitted through `AgentNode.emit` to the root node's listener.

The listener is the one hook an agent tree has (see node.py). Every agent in the tree emits through its own node,
so a client that listens at the root sees all of them, each event carrying the emitting agent's `path`.

Events are enough to rebuild each agent's transcript: `AgentAdded` starts it, `MessageAdded` appends to it, and a
`Compaction` with status "compacted" replaces `messages[start:start + removed]` with its `summary`.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from jean_code.core.llm import Message, UserMessage
from jean_code.core.usage import ModelCall

# How a turn ended. The values are the text shown for it, and keys once stored: don't reword them.
TurnEnd = Literal["answered", "interrupted", "budget spent", "cancelled", "context full", "out of tool rounds"]


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str  # the emitting agent's node, e.g. "main/1"


class AgentAdded(Event):
    """An agent was built on the node at `path`. Comes before any other event from that path."""

    type: Literal["agent_added"] = "agent_added"
    profile: str  # what kind of agent: "" for the root
    task: str  # what its node was created for: "" for the root
    model: str


class MessageAdded(Event):
    """`message` was appended to the agent's transcript."""

    type: Literal["message"] = "message"
    message: Message
    context_tokens: int  # the agent's context size with the message in it


class Delta(Event):
    """A piece of a reply as it streams in; the reply itself follows as a `MessageAdded`."""

    type: Literal["delta"] = "delta"
    kind: Literal["content", "reasoning"]
    text: str


class TurnStarted(Event):
    type: Literal["turn_started"] = "turn_started"
    text: str  # the user's input, before any command expands it


class TurnEnded(Event):
    """Every `TurnStarted` gets exactly one, however the turn ended. `end` is None when the turn raised."""

    type: Literal["turn_ended"] = "turn_ended"
    end: TurnEnd | None
    error: str | None = None
    context_tokens: int


class Compaction(Event):
    """A compaction attempt that `agent.compactions` recorded. Only status "compacted" changes the transcript:
    `messages[start:start + removed]` was replaced by `summary`."""

    type: Literal["compaction"] = "compaction"
    status: str
    start: int
    removed: int
    summary: UserMessage | None = None
    tokens_before: int
    tokens_after: int | None = None
    context_tokens: int


class Charged(Event):
    """A paid model call, charged to the node at `path`."""

    type: Literal["charged"] = "charged"
    call: ModelCall


AgentEvent = Annotated[
    AgentAdded | MessageAdded | Delta | TurnStarted | TurnEnded | Compaction | Charged, Field(discriminator="type")
]
