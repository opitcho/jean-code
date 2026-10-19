"""An agent with `stream=True`: its replies arrive as `Delta`s, and a stop cuts off the reply in flight.

Each test names the bug it would catch.
"""

import pytest

from conftest import FakeClient, call, response
from jean_code.core.agent import Agent
from jean_code.core.events import Compaction, Delta, MessageAdded
from jean_code.core.llm import AssistantMessage, ToolMessage
from jean_code.core.tool_schema import tool

REPLY = "the answer is forty-two"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


class StopAt(FakeClient):
    """A FakeClient whose stream stops the agent after `items` streamed items of the call numbered `at`."""

    def __init__(self, responses, at: int, items: int):
        super().__init__(responses)
        self.at, self.items = at, items
        self.agent: Agent | None = None

    def stream(self, request, stop=None):
        number = len(self.requests)
        inner = super().stream(request, stop)
        count = 0
        while True:
            if number == self.at and count == self.items:
                assert self.agent is not None
                self.agent.interrupt()
            try:
                item = next(inner)
            except StopIteration:
                return
            count += 1
            yield item


def streaming_agent(client: FakeClient, **kwargs) -> tuple[Agent, list]:
    agent = Agent(client=client, tools={"add": add}, stream=True, **kwargs)
    if isinstance(client, StopAt):
        client.agent = agent
    events: list = []
    agent.node.listener = events.append
    return agent, events


def test_a_turn_reply_streams_as_deltas_before_its_message():
    # Catches streamed text that arrives after its reply, or a reply that never streams.
    agent, events = streaming_agent(FakeClient([response(REPLY, reasoning="let me think")]))
    assert agent.run_turn("question") == "answered"
    deltas = [e for e in events if isinstance(e, Delta)]
    reply = next(i for i, e in enumerate(events) if isinstance(e, MessageAdded) and e.message.role == "assistant")
    assert "".join(d.text for d in deltas if d.kind == "reasoning") == "let me think"
    assert "".join(d.text for d in deltas if d.kind == "content") == REPLY
    assert len(deltas) == 6 and all(events.index(d) < reply for d in deltas)
    assert agent.messages[-1] == AssistantMessage(content=REPLY, reasoning="let me think")


def test_a_compaction_streams_nothing_to_show():
    # Catches the summary call's text shown as if it were the agent's reply.
    script = [response("one"), response("two"), response("the summary")]
    agent, events = streaming_agent(FakeClient(script), keep_first=0, keep_last=1)
    agent.run_turn("first")
    agent.run_turn("second")
    events.clear()
    assert agent.compact(force=True) == "compacted"
    assert not [e for e in events if isinstance(e, Delta)]
    assert [e.status for e in events if isinstance(e, Compaction)] == ["compacted"]


def test_a_reply_cut_off_mid_stream_keeps_its_text_and_runs_none_of_its_calls():
    # Catches a half-written tool call run, a cut reply lost, or a transcript the next request can't send.
    cut = response(REPLY, tool_calls=[call("add", a=1, b=2)])
    client = StopAt([cut, response("ok")], at=0, items=2)
    agent, _ = streaming_agent(client)
    assert agent.run_turn("go") == "cancelled"
    last = agent.messages[-1]
    assert isinstance(last, AssistantMessage) and last.tool_calls == ()
    assert last.content and REPLY.startswith(last.content) and last.content != REPLY
    assert len(agent.node.entries) == 1 and agent.node.entries[0].cost is None
    agent.node.reopen()
    assert agent.run_turn("again") == "answered"
    sent = client.requests[-1]["messages"]
    assert sent[-3] == {"role": "assistant", "content": last.content}


def test_a_stop_before_the_first_chunk_books_nothing():
    # Catches an empty cut-off reply booked as a call, or appended as an empty message.
    agent, _ = streaming_agent(StopAt([response(REPLY)], at=0, items=0))
    before = agent.messages
    assert agent.run_turn("go") == "cancelled"
    assert agent.node.entries == [] and agent.messages[len(before) :][-1].role == "user"


def test_a_reply_that_arrived_whole_runs_its_calls_and_then_stops():
    # Catches a stop that drops the calls of a reply that had fully arrived, which a non-streamed reply would run.
    client = FakeClient([response(tool_calls=[call("add", a=1, b=2)]), response("never")])
    agent, events = streaming_agent(client)

    def stop_on_reply(event) -> None:
        events.append(event)
        if isinstance(event, MessageAdded) and isinstance(event.message, AssistantMessage):
            agent.interrupt()

    agent.node.listener = stop_on_reply
    assert agent.run_turn("go") == "cancelled"
    assert isinstance(agent.messages[-1], ToolMessage) and agent.messages[-1].content == "3"
    assert len(client.requests) == 1


def test_a_compaction_cut_off_mid_stream_changes_nothing():
    # Catches half a summary replacing the middle of the transcript.
    script = [response("one"), response("two"), response("the summary, cut short")]
    agent, _ = streaming_agent(StopAt(script, at=2, items=1), keep_first=0, keep_last=1)
    agent.run_turn("first")
    agent.run_turn("second")
    before = agent.messages
    assert agent.compact(force=True) == "failed: interrupted"
    assert agent.messages == before


def test_a_compaction_stopped_before_its_first_chunk_fails_as_interrupted():
    # Catches a stop before the summary's first chunk reported as an error.
    script = [response("one"), response("two"), response("the summary")]
    agent, _ = streaming_agent(StopAt(script, at=2, items=0), keep_first=0, keep_last=1)
    agent.run_turn("first")
    agent.run_turn("second")
    assert agent.compact(force=True) == "failed: interrupted"


def test_streaming_needs_a_client_that_can_stream():
    # Catches a misbuilt streaming agent that only fails on its first turn.
    class CompleteOnly:
        def complete(self, request):
            raise AssertionError("never called")

    with pytest.raises(ValueError, match="can't stream"):
        Agent(client=CompleteOnly(), stream=True)  # type: ignore[arg-type]
