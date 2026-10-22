"""Tests for `ErrorCaught`: an agent records each exception its code caught, with its traceback, once.

Each test names the bug it would catch.
"""

import json

import httpx
import pytest

from conftest import FakeClient, call, response, scripted
from jean_code.core import agent as agent_module
from jean_code.core.agent import Agent
from jean_code.core.driver import Driver
from jean_code.core.events import Compaction, ErrorCaught, JobEnded, MessageAdded, TurnEnded, error_line
from jean_code.core.llm import MalformedResponse, ToolMessage, render_request
from jean_code.core.node import AgentNode
from jean_code.core.subagents import agent_profile
from jean_code.core.tool_schema import tool
from test_compaction import BIG, SMALL, compactions, is_summary_request, run_small_turns, summary
from test_compaction import make_agent as compacting_agent
from test_driver import next_of
from test_events import Recorder as Events


@tool
def lookup(key: str) -> str:
    """Looks a key up.

    Args:
        key: The key.
    """
    return {}[key]


def errors(events: Events, path: str | None = None) -> list[ErrorCaught]:
    return [e for e in events if isinstance(e, ErrorCaught) and path in (None, e.path)]


def tool_agent(events: Events, responses, **kwargs) -> Agent:
    return Agent(client=FakeClient(responses), tools={"lookup": lookup}, node=AgentNode(listener=events), **kwargs)


def bad_summary(completion, stopped):
    raise ValueError("the summary check broke")


# ---------------------------------------------------------------- where the agent swallows it


def test_a_failing_tool_is_recorded_with_its_stack():
    # Catches a tool's exception reaching the trace as one line, with no stack, or unlinked from its call.
    events = Events()
    agent = tool_agent(events, [response(tool_calls=[call("lookup", key="url")]), response("done")])
    agent.run_turn("go")
    [error] = errors(events)
    result = next(m for m in agent.messages if isinstance(m, ToolMessage))
    assert error.tool_call_id == result.tool_call_id
    assert error.error == "KeyError: 'url'" and result.content == "Error: " + error.error
    assert "in lookup" in error.traceback and error.traceback.rstrip().endswith("KeyError: 'url'")
    after = events[events.index(error) + 1]
    assert isinstance(after, MessageAdded) and after.message == result


def test_the_model_never_sees_the_traceback():
    # Catches the stack leaking into the context, where it would cost tokens on every later call.
    events = Events()
    agent = tool_agent(events, [response(tool_calls=[call("lookup", key="url")]), response("done")])
    agent.run_turn("go")
    assert errors(events)
    assert "Traceback (most recent call last)" not in json.dumps(scripted(agent).requests)


def test_bad_arguments_are_recorded():
    # Catches the model's malformed arguments missing from the trace.
    bad = {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{not json"}}
    events = Events()
    tool_agent(events, [response(tool_calls=[bad]), response("done")]).run_turn("go")
    [error] = errors(events)
    assert error.tool_call_id == "c1" and error.error.startswith("JSONDecodeError")


def test_a_failed_summary_is_recorded():
    # Catches a failed summary call leaving only its one-line status.
    events = Events()
    agent, client = compacting_agent([], node=AgentNode(listener=events))
    run_small_turns(agent, 5)
    client.responses.append(RuntimeError("provider down"))
    assert agent.compact(force=True) == "failed: RuntimeError: provider down"
    [error] = errors(events)
    [compaction] = [e for e in events if isinstance(e, Compaction)]
    assert compaction.status == "failed: " + error.error


DROP = httpx.ConnectError("[Errno -3] Temporary failure in name resolution")


@pytest.mark.parametrize("stream", [False, True])
def test_a_connection_that_fails_to_open_is_retried_and_recorded(monkeypatch, stream):
    # Catches a one-second Wi-Fi drop ending the agent's turn, or a retry the trace never shows.
    monkeypatch.setattr(agent_module, "NETWORK_RETRY_WAITS", (0, 0))
    events = Events()
    agent = Agent(client=FakeClient([DROP, DROP, response("done")]), node=AgentNode(listener=events), stream=stream)
    assert agent.run_turn("go") == "answered"
    assert [e.error for e in errors(events)] == [error_line(DROP)] * 2
    assert all("ConnectError" in e.traceback and e.tool_call_id is None for e in errors(events))


def test_a_connection_that_stays_down_ends_the_turn_once_the_retries_run_out(monkeypatch):
    # Catches an endless retry: an agent that hangs without a word while the network is down.
    monkeypatch.setattr(agent_module, "NETWORK_RETRY_WAITS", (0, 0))
    events = Events()
    client = FakeClient([DROP] * 4)
    agent = Agent(client=client, node=AgentNode(listener=events))
    with pytest.raises(httpx.ConnectError):
        agent.run_turn("go")
    assert len(client.requests) == 3
    assert len(errors(events)) == 3  # two retried, and the last one where it leaves the agent


def test_a_stop_while_waiting_to_retry_ends_the_turn_at_once(monkeypatch):
    # Catches a cancel that sits out the whole backoff, or ends the turn as an error.
    client = FakeClient([DROP, response("done")])
    agent = Agent(client=client, node=AgentNode())
    monkeypatch.setattr(agent_module.time, "sleep", lambda s: agent.node.cancel())
    assert agent.run_turn("go") == "cancelled"
    assert len(client.requests) == 1


def test_a_connection_lost_mid_request_is_not_retried(monkeypatch):
    # Catches a retry of a request the provider may already be answering, or whose reply was partly shown.
    monkeypatch.setattr(agent_module, "NETWORK_RETRY_WAITS", (0, 0))
    agent = Agent(client=FakeClient([httpx.ReadError("connection reset"), response("done")]), node=AgentNode())
    with pytest.raises(httpx.ReadError):
        agent.run_turn("go")


# ---------------------------------------------------------------- where it leaves the agent


def test_a_raising_turn_is_recorded_by_the_agent_before_its_end():
    # Catches a turn's exception left for the caller to record, or recorded after the turn's end.
    events = Events()
    agent = tool_agent(events, [MalformedResponse("garbled")])
    with pytest.raises(MalformedResponse):
        agent.run_turn("go")
    [error] = errors(events)
    after = events[events.index(error) + 1]
    assert isinstance(after, TurnEnded) and after.error == error.error == "MalformedResponse: garbled"
    assert "in run_turn" in error.traceback


def test_the_driver_adds_nothing():
    # Catches the driver recording the turn's exception a second time.
    driver = Driver(lambda node, approve: Agent(client=FakeClient([MalformedResponse("garbled")]), node=node))
    with driver.view.subscribe() as (_, events):
        driver.send("go")
        job = next_of(events, JobEnded)
    [error] = driver.view.state().agents["main"].caught
    assert job.error == error.error == "MalformedResponse: garbled"
    driver.close()


def test_a_compaction_raising_inside_a_turn_is_recorded_once(monkeypatch):
    # Catches an exception leaving through both compact and run_turn, so it's recorded twice.
    events = Events()
    agent, client = compacting_agent([], node=AgentNode(listener=events))
    run_small_turns(agent, 4)
    client.responses += [response("reply 5", usage=BIG), summary()]
    monkeypatch.setattr(agent_module, "summary_failure", bad_summary)
    with pytest.raises(ValueError, match="summary check broke"):
        agent.run_turn("turn 5")
    [error] = errors(events)
    assert error.error == "ValueError: the summary check broke"


def test_a_compact_job_that_raises_is_recorded(monkeypatch):
    # Catches a compaction's exception lost when it leaves the agent through compact, as a /compact job does.
    script = [response(f"reply {i}", usage=SMALL) for i in range(1, 6)] + [summary()]
    driver = Driver(lambda node, approve: compacting_agent(script, node=node)[0])
    with driver.view.subscribe() as (_, events):
        for i in range(1, 6):
            driver.send(f"turn {i}")
            next_of(events, JobEnded)
        monkeypatch.setattr(agent_module, "summary_failure", bad_summary)
        driver.compact()
        job = next_of(events, JobEnded)
    [error] = driver.view.state().agents["main"].caught
    assert job.error == error.error == "ValueError: the summary check broke"
    driver.close()


def test_a_subagent_error_lands_in_its_own_stream_once():
    # Catches a subagent's exception recorded on its parent, twice, or not at all.
    def build(task: str, node: AgentNode) -> Agent:
        return Agent(client=FakeClient([MalformedResponse("garbled")]), node=node)

    events = Events()
    parent = Agent(
        client=FakeClient([response(tool_calls=[call("spawn", profile="boom", task="t")]), response("done")]),
        node=AgentNode(listener=events),
        subagent_profiles={"boom": agent_profile("fails at once", build, max_cost=1.0)},
    )
    parent.run_turn("go")
    assert [e.error for e in errors(events)] == ["MalformedResponse: garbled"]
    assert errors(events, "main/1") and not errors(events, "main")
    assert 'status="failed"' in next(m.content for m in parent.messages if isinstance(m, ToolMessage))


def test_a_failing_close_is_recorded():
    # Catches a cleanup failure reaching the caller with no stack in the trace.
    def unmount() -> None:
        raise OSError("device busy")

    events = Events()
    agent = Agent(client=FakeClient([]), cleanup=[unmount], node=AgentNode(listener=events))
    with pytest.raises(OSError):
        agent.close()
    [error] = errors(events)
    assert error.error == "OSError: device busy" and "in unmount" in error.traceback


# ---------------------------------------------------------------- what isn't recorded


def test_a_stop_records_nothing():
    # Catches a stop (a cancel, an interrupted summary) recorded as if something broke.
    events = Events()
    cancelled = tool_agent(events, [response("never")])
    cancelled.node.cancel()
    assert cancelled.run_turn("go") == "cancelled"

    agent, _ = compacting_agent([], node=AgentNode(listener=events))
    run_small_turns(agent, 4)

    class InterruptingSummary(FakeClient):
        def complete(self, request):
            if is_summary_request(render_request(request)):
                agent.interrupt()
            return super().complete(request)

    agent.client = InterruptingSummary([response("reply 5", usage=BIG), summary()])
    agent.run_turn("turn 5")
    assert compactions(agent)[-1].status == "failed: interrupted"
    assert errors(events) == []


def test_record_error_skips_the_process_stopping():
    # Catches Ctrl+C recorded as an agent's error.
    events = Events()
    node = AgentNode(listener=events)
    stop = KeyboardInterrupt("stop")
    assert node.record_error(stop) == error_line(stop) == "KeyboardInterrupt: stop"
    assert events == []
