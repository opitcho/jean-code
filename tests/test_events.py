"""Tests for the agent's events (`core/events.py`): what every agent in a tree emits through the root node's
listener, and that a client can rebuild each agent's transcript from those events alone.

Each test names the bug it would catch.
"""

import threading

import pytest

from conftest import FakeClient, call, response, scripted, usage
from jean_code.core.agent import Agent
from jean_code.core.events import AgentAdded, Charged, Compaction, MessageAdded, TurnEnded, TurnStarted
from jean_code.core.llm import UserMessage
from jean_code.core.node import AgentNode
from jean_code.core.subagents import agent_profile, agent_tree
from jean_code.core.tool_schema import tool

TIMEOUT = 5  # seconds any wait in these tests may take before it counts as a hang
SMALL = usage(prompt=100, completion=5, cost=0.001)
BIG = usage(prompt=5000, completion=5, cost=0.001)
BULK = " filler" * 2000


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@tool
def rm(path: str) -> str:
    """Remove a file.

    Args:
        path: The file.
    """
    return f"removed {path}"


class Recorder(list):
    """The root listener: keeps every event, from any thread, in the order they arrive."""

    def __call__(self, event) -> None:
        self.append(event)

    def of(self, path: str) -> list:
        return [e for e in self if e.path == path]


def rebuild(events) -> dict[str, list]:
    """Every path's transcript, from events alone: what a client's reducer does."""
    transcripts: dict[str, list] = {}
    for e in events:
        if isinstance(e, AgentAdded):
            transcripts[e.path] = []
        elif isinstance(e, MessageAdded):
            transcripts[e.path].append(e.message)
        elif isinstance(e, Compaction) and e.status == "compacted":
            transcripts[e.path][e.start : e.start + e.removed] = [e.summary]
    return transcripts


def check_replay(root_agent: Agent, events: Recorder) -> None:
    """The invariants every scenario must keep: each agent's transcript rebuilds exactly from the events, each path
    starts with AgentAdded, each started turn ends once, spend and context sizes match the agent's own."""
    tree = agent_tree(root_agent)
    rebuilt = rebuild(events)
    for path, agent in tree.items():
        assert tuple(rebuilt[path]) == agent.messages, path
        assert isinstance(events.of(path)[0], AgentAdded), path
        turns = [e for e in events.of(path) if isinstance(e, TurnStarted | TurnEnded)]
        assert [type(e) for e in turns] == [TurnStarted, TurnEnded] * (len(turns) // 2), path
        sized = [e for e in events.of(path) if isinstance(e, MessageAdded | TurnEnded | Compaction)]
        assert sized[-1].context_tokens == agent.context_tokens, path
    charged = sum(e.call.cost or 0 for e in events if isinstance(e, Charged))
    assert charged == pytest.approx(root_agent.node.spent)


def root(events: Recorder, **kwargs) -> AgentNode:
    return AgentNode(listener=events, **kwargs)


def make_agent(events: Recorder, responses, **kwargs) -> Agent:
    return Agent(
        client=FakeClient(responses), system_prompt="sys", tools={"add": add, "rm": rm}, node=root(events), **kwargs
    )


def wait_for(predicate) -> None:
    for _ in range(TIMEOUT * 100):
        if predicate():
            return
        threading.Event().wait(0.01)
    raise AssertionError("timed out")


# ---------------------------------------------------------------- one agent


def test_a_plain_answer_rebuilds_and_announces_the_agent_first():
    # Catches a transcript a client can't rebuild, or a client that sees messages before it knows the agent.
    events = Recorder()
    agent = make_agent(events, [response("hi")])
    assert agent.run_turn("hello") == "answered"
    check_replay(agent, events)
    added = events[0]
    assert isinstance(added, AgentAdded) and added.path == "main" and added.model == agent.model
    assert [type(e).__name__ for e in events] == [
        "AgentAdded",
        "MessageAdded",  # the system message
        "TurnStarted",
        "MessageAdded",  # the user's
        "Charged",
        "MessageAdded",  # the reply
        "TurnEnded",
    ]
    assert events[-1].end == "answered" and events[-1].error is None


def test_two_rounds_of_tool_calls_rebuild():
    # Catches a tool result appended without its event.
    events = Recorder()
    agent = make_agent(
        events,
        [
            response(tool_calls=[call("add", a=1, b=2), call("add", a=3, b=4)]),
            response(tool_calls=[call("add", a=5, b=6)]),
            response("done"),
        ],
    )
    assert agent.run_turn("go") == "answered"
    check_replay(agent, events)


def test_a_denied_call_rebuilds():
    # Catches the refusal's result missing from the events.
    events = Recorder()
    agent = make_agent(
        events,
        [response(tool_calls=[call("rm", path="x")]), response("ok")],
        needs_approval={"rm"},
        approve=lambda n, a: "no",
    )
    assert agent.run_turn("delete x") == "answered"
    check_replay(agent, events)


def test_a_stop_mid_reply_rebuilds_and_ends_cancelled():
    # Catches a result appended without its event when a stop lands mid-reply, and a TurnEnded without its end.
    events = Recorder()

    @tool
    def stop() -> str:
        """Stop the turn."""
        agent.interrupt()
        return "stopped"

    agent = Agent(
        client=FakeClient([response(tool_calls=[call("stop"), call("add", a=1, b=2)]), response("never")]),
        tools={"stop": stop, "add": add},
        node=root(events),
    )
    assert agent.run_turn("go") == "cancelled"
    check_replay(agent, events)
    assert agent.messages[-1].content == "3"  # the reply's calls finish; the stop holds before the next model call
    assert events[-1] == TurnEnded(path="main", end="cancelled", context_tokens=agent.context_tokens)


def test_a_turn_that_raises_still_ends_once_with_the_error():
    # Catches a TurnStarted with no TurnEnded, so a client shows the agent running forever.
    events = Recorder()
    agent = make_agent(events, [RuntimeError("HTTP 502")])
    with pytest.raises(RuntimeError):
        agent.run_turn("go")
    check_replay(agent, events)
    assert events[-1].end is None and events[-1].error == "RuntimeError: HTTP 502"


def test_a_command_rebuilds_with_its_output(tmp_path):
    # Catches the expanded user message (the command's output) differing from what the events carry.
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "SKILL.md").write_text("---\nname: demo\ndescription: A demo.\n---\nDEMO BODY\n")
    events = Recorder()
    agent = make_agent(events, [response("done")], skills_dir=tmp_path)
    agent.run_turn("/demo do it")
    check_replay(agent, events)
    assert "DEMO BODY" in agent.messages[-2].content
    assert next(e for e in events if isinstance(e, TurnStarted)).text == "/demo do it"


# ---------------------------------------------------------------- compaction


def compacting_agent(events: Recorder, **kwargs) -> Agent:
    agent = make_agent(events, [], compact_at=1000, keep_first=2, keep_last=2, **kwargs)
    for i in range(1, 5):  # turns 1-2 the head, 3 the middle (bulky), 4 the tail
        scripted(agent).responses.append(response(f"reply {i}" + (BULK if i == 3 else ""), usage=SMALL))
        agent.run_turn(f"turn {i}")
    return agent


def test_a_forced_compaction_rebuilds_through_the_splice():
    # Catches a splice at the wrong index, which would show the wrong turns as compacted.
    events = Recorder()
    agent = compacting_agent(events)
    scripted(agent).responses += [
        response("reply 5", usage=SMALL),  # a fifth turn: with two kept at each end, turn 3 is the middle
        response("SUMMARY", usage=usage(prompt=99_999, completion=300, cost=0.002)),
    ]
    agent.run_turn("turn 5")
    assert agent.compact(force=True) == "compacted"
    check_replay(agent, events)
    compaction = events[-1]
    assert isinstance(compaction, Compaction) and compaction.summary is agent.compactions[-1]["summary"]
    assert compaction.tokens_after == agent.compactions[-1]["tokens_after"]
    assert compaction.cost == agent.compactions[-1]["call"].cost == 0.002


def test_an_automatic_compaction_rebuilds():
    # Catches the compaction at the end of a turn left out of the events.
    events = Recorder()
    agent = compacting_agent(events)
    scripted(agent).responses += [
        response("reply 5", usage=BIG),
        response("SUMMARY", usage=usage(prompt=99_999, completion=300, cost=0.002)),
    ]
    agent.run_turn("turn 5")
    assert agent.compactions[-1]["status"] == "compacted"
    check_replay(agent, events)


def test_a_failed_compaction_is_reported_and_changes_nothing():
    # Catches a failed attempt spliced into the transcript, or not reported at all.
    events = Recorder()
    agent = compacting_agent(events)
    scripted(agent).responses += [response("reply 5", usage=BIG), RuntimeError("HTTP 502")]
    agent.run_turn("turn 5")
    check_replay(agent, events)
    compaction = next(e for e in events if isinstance(e, Compaction))
    assert compaction.status.startswith("failed") and compaction.summary is None


def test_an_attempt_that_records_nothing_emits_nothing():
    # Catches a Compaction event for "below threshold", which no record holds either.
    events = Recorder()
    agent = make_agent(events, [response("hi")], compact_at=1000)
    agent.run_turn("go")
    assert agent.compact() == "below threshold"
    assert not [e for e in events if isinstance(e, Compaction)]


# ---------------------------------------------------------------- subagents and other agents on child nodes


def echo_profile():
    def build(task: str, node: AgentNode) -> Agent:
        return Agent(client=FakeClient([response(f"answer to {task}")]), system_prompt="sub", node=node)

    return agent_profile("answers at once", build, max_cost=1.0)


def test_a_foreground_subagent_rebuilds_under_its_path_with_its_profile():
    # Catches a subagent's events lost (its root has no listener) or announced without its profile and task.
    events = Recorder()
    agent = make_agent(
        events,
        [response(tool_calls=[call("spawn", profile="echo", task="t1")]), response("done")],
        subagent_profiles={"echo": echo_profile()},
    )
    assert agent.run_turn("go") == "answered"
    check_replay(agent, events)
    added = events.of("main/1")[0]
    child = agent.node.children[0].agent
    tools = list(child.last_request.tools)
    assert added == AgentAdded(path="main/1", profile="echo", task="t1", model=agent.model, tools=tools)


def test_a_background_subagent_delivered_by_claim_rebuilds():
    # Catches events from the run's own thread lost, or the claimed result's user message out of the events.
    events = Recorder()
    agent = make_agent(
        events,
        [response(tool_calls=[call("spawn", profile="echo", task="t1", background=True)]), response("started")],
        subagent_profiles={"echo": echo_profile()},
    )
    agent.run_turn("go")
    wait_for(lambda: agent.subagents.pending == 1)
    scripted(agent).responses.append(response("got it"))
    agent.run_turn("next")
    check_replay(agent, events)
    assert "answer to t1" in agent.messages[-2].content


def test_a_nested_subagent_rebuilds_under_its_grandparent_root():
    # Catches events of a grandchild not reaching the root's listener.
    def lead_build(task: str, node: AgentNode) -> Agent:
        client = FakeClient([response(tool_calls=[call("spawn", profile="echo", task="inner")]), response("lead done")])
        return Agent(client=client, node=node, subagent_profiles={"echo": echo_profile()})

    events = Recorder()
    agent = make_agent(
        events,
        [response(tool_calls=[call("spawn", profile="lead", task="outer")]), response("done")],
        subagent_profiles={"lead": agent_profile("leads", lead_build, max_cost=1.0)},
    )
    agent.run_turn("go")
    check_replay(agent, events)
    assert set(rebuild(events)) == {"main", "main/1", "main/1/1"}


def test_an_agent_built_on_any_child_node_is_seen():
    # Catches agents from something other than spawn (the engine, later) invisible to the root's listener.
    events = Recorder()
    group = root(events).child(None, name="wf1", task="build it")
    agent = Agent(client=FakeClient([response("planned")]), node=group.child(None, name="plan", profile="planner"))
    agent.run_turn("plan it")
    added = events[0]
    tools = list(agent.last_request.tools)
    assert added == AgentAdded(path="main/wf1/plan", profile="planner", task="", model=agent.model, tools=tools)
    assert tuple(rebuild(events)["main/wf1/plan"]) == agent.messages
    assert agent.node.agent is agent


def test_agent_added_carries_the_tools_it_sends():
    # Catches a trace that can't rebuild a request: the schemas are sent with every call but emitted nowhere.
    events = Recorder()
    agent = make_agent(events, [response("hi")])
    agent.run_turn("go")
    names = [schema["function"]["name"] for schema in events[0].tools]
    assert {"add", "rm"} <= set(names)
    assert events[0].tools == list(agent.last_request.tools)


def test_no_listener_no_events_and_nothing_breaks():
    # Catches emit failing on a tree with no listener: the REPL and Streamlit build agents that way until they move.
    agent = Agent(client=FakeClient([response("hi")]), system_prompt="sys")
    assert agent.run_turn("go") == "answered"
    assert agent.messages[-2] == UserMessage(content="go")


def test_messages_cannot_be_appended_to_directly():
    # Catches a code path that grows the transcript without going through `_append`, so without an event.
    agent = Agent(client=FakeClient([]))
    with pytest.raises(AttributeError):
        agent.messages.append(UserMessage(content="x"))  # type: ignore[attr-defined]
