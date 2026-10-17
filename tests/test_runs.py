"""Tests for the subagents' Python API: `Profile`, `Run` and the `Outcome` a run ends with. No tools involved,
except where a run's own subagents are the point.

Each test names the bug it would catch.
"""

import threading
import time

import pytest

from conftest import FakeClient, call, response, scripted, usage
from jean_code.core.agent import Agent
from jean_code.core.node import AgentNode
from jean_code.core.subagents import Outcome, Profile, Run, agent_profile
from jean_code.core.tool_schema import tool

TIMEOUT = 5  # seconds any wait in these tests may take before it counts as a hang


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@tool
def slow() -> str:
    """Wait a moment."""
    time.sleep(0.1)
    return "ok"


def tool_reply(cost=0.001, content="") -> dict:
    return response(content, tool_calls=[call("add", a=1, b=2)], usage=usage(cost=cost))


def make_agent(responses, node=None, **kwargs) -> Agent:
    return Agent(client=FakeClient(responses), tools={"add": add}, node=node or AgentNode(1.0), **kwargs)


def gated(agent: Agent) -> threading.Event:
    """Hold a run whose task starts with `/gate` just before `run_turn` clears the interrupt flag, until set."""
    gate = threading.Event()

    def hold(rest: str) -> None:  # None: the task goes on as typed
        gate.wait(TIMEOUT)

    agent.commands["gate"] = hold
    return gate


def done(text: str) -> Outcome:
    return Outcome("done", text)


# ---------------------------------------------------------------- profiles and outcomes


def test_each_start_is_a_new_run():
    # Catches a profile that shares an agent (or its state) between runs.
    built = []

    def build(task: str, node: AgentNode) -> Agent:
        built.append(make_agent([response(f"answer to {task}")], node))
        return built[-1]

    profile = agent_profile("helper", build, max_cost=1.0)
    root = AgentNode()
    a = profile.start("a", root.child(1.0, "1"))
    b = profile.start("b", root.child(1.0, "2"))
    assert (a.outcome(TIMEOUT), b.outcome(TIMEOUT)) == (done("answer to a"), done("answer to b"))
    assert a.agent is not b.agent and len(built) == 2
    assert (a.path, b.path) == ("main/1", "main/2")


def test_the_answer_is_the_last_reply_or_the_report():
    # Catches a report being ignored, or the default handing back the wrong message.
    plain = Run.start(make_agent([tool_reply(content="let me add"), response("it's 3")]), "add 1 and 2")
    assert plain.outcome(TIMEOUT) == done("it's 3")
    reported = Run.start(make_agent([response("done, see the notebook")]), "go", report=lambda: "the notebook")
    assert reported.outcome(TIMEOUT) == done("the notebook")


def test_no_outcome_while_running():
    agent = make_agent([response("later")])
    gate = gated(agent)
    run = Run.start(agent, "/gate go")
    assert run.outcome(0) is None and not run.done()
    gate.set()
    assert run.outcome(TIMEOUT) == done("later")


def test_profile_limits_are_checked():
    # Catches an unlimited profile (no default) or a cap that lowers the default.
    def start(task: str, node: AgentNode) -> Run:
        raise AssertionError("no run starts in this test")

    with pytest.raises(ValueError, match="positive"):
        Profile("p", start, max_cost=0)
    with pytest.raises(ValueError, match="at least"):
        Profile("p", start, max_cost=1.0, max_cost_cap=0.5)
    assert Profile("p", start, max_cost=1.0).cap == 1.0
    assert Profile("p", start, max_cost=1.0, max_cost_cap=3.0).cap == 3.0


def test_a_subagent_that_would_need_approval_is_refused():
    # Catches a run that stops at its first `bash` call waiting for an approval nobody can give.
    with pytest.raises(ValueError, match="approval: add"):
        Run.start(make_agent([], needs_approval={"add"}, approve=lambda name, arguments: None), "go")


# ---------------------------------------------------------------- stopped, not done


def test_budget_spent_is_stopped_with_the_last_remark():
    # Catches a mid-task remark handed back as the answer after the budget ran out.
    agent = make_agent([tool_reply(cost=0.004, content="let me check"), response("never")], AgentNode(0.004))
    run = Run.start(agent, "go")
    assert run.outcome(TIMEOUT) == Outcome("stopped", "let me check", "budget spent")


def test_an_answer_that_spends_the_last_of_the_budget_is_done():
    # Catches a finished answer labelled "budget spent" because the answering call used the budget up.
    run = Run.start(make_agent([response("the answer", usage=usage(cost=0.004))], AgentNode(0.004)), "go")
    assert run.outcome(TIMEOUT) == done("the answer")


def test_unknown_cost_does_not_fail_the_run():
    # Catches a call of unknown cost failing the run instead of only being warned about.
    unknown = response(tool_calls=[call("add", a=1, b=2)], no_usage=True)
    run = Run.start(make_agent([unknown, response("the answer")]), "go")
    assert run.outcome(TIMEOUT) == done("the answer")


def test_a_full_context_is_stopped():
    outcome = Run.start(make_agent([], context_window=10, compact_at=None), "go").outcome(TIMEOUT)
    assert (outcome.status, outcome.reason) == ("stopped", "context full")


def test_running_out_of_tool_rounds_is_stopped():
    # Catches the tool-round limit, which never interrupts, reported as an answer.
    run = Run.start(make_agent([tool_reply(), tool_reply()], max_tool_rounds=1), "go")
    assert run.outcome(TIMEOUT) == Outcome("stopped", "(no reply)", "out of tool rounds")


# ---------------------------------------------------------------- failed


def test_a_crash_fails_the_run():
    # Catches a crashed run that stays "running" forever, or whose error escapes as an exception.
    run = Run.start(make_agent([RuntimeError("HTTP 500")]), "go")
    assert run.outcome(TIMEOUT) == Outcome("failed", "RuntimeError: HTTP 500")


def test_a_failing_report_fails_the_run():
    run = Run.start(make_agent([response("hi")]), "go", report=lambda: str(1 / 0))
    assert run.outcome(TIMEOUT) == Outcome("failed", "ZeroDivisionError: division by zero")


def test_the_report_still_runs_when_the_agent_raises():
    # Catches a research run that hits a 5xx losing its report, though its notebook holds everything found so far.
    run = Run.start(make_agent([RuntimeError("HTTP 500")]), "go", report=lambda: "the notebook")
    assert run.outcome(TIMEOUT) == Outcome("failed", "RuntimeError: HTTP 500\n\nthe notebook")
    both = Run.start(make_agent([RuntimeError("HTTP 500")]), "go", report=lambda: str(1 / 0))
    assert both.outcome(TIMEOUT) == Outcome(
        "failed", "RuntimeError: HTTP 500\n\nThe report failed too: ZeroDivisionError: division by zero"
    )


def test_a_failing_cleanup_still_completes_the_run():
    # Catches `close()` raising outside the try, which would leave the run "running" forever.
    def boom():
        raise OSError("container already gone")

    run = Run.start(make_agent([response("hi")], cleanup=[boom]), "go")
    assert run.outcome(TIMEOUT) == Outcome("failed", "OSError: container already gone")


# ---------------------------------------------------------------- cancelling


def test_a_cancel_before_the_turn_starts_is_not_lost():
    # Catches a stop cleared when the turn starts, so a cancel sent just before it is lost.
    agent = make_agent([response("never")])
    gate = gated(agent)
    run = Run.start(agent, "/gate go")  # held just before run_turn clears the flag
    run.cancel()
    gate.set()
    assert run.outcome(TIMEOUT) == Outcome("stopped", "(no reply)", "cancelled")
    assert scripted(agent).requests == []


def test_a_cancel_after_the_run_finished_changes_nothing():
    # Catches a finished run relabelled "cancelled" by a cancel that came too late.
    run = Run.start(make_agent([response("the answer")]), "go")
    assert run.outcome(TIMEOUT) == done("the answer")
    run.cancel()
    assert run.outcome(TIMEOUT) == done("the answer")


def test_a_finished_run_stops_the_runs_it_started():
    # Catches a run that finishes while its own background subagents keep spending.
    grandchild_client = FakeClient([response(tool_calls=[call("slow")])] * 20 + [response("dug")])
    grandchild = agent_profile(
        "digger",
        lambda task, node: Agent(client=grandchild_client, tools={"slow": slow}, node=node),
        max_cost=1.0,
    )
    child = Agent(
        client=FakeClient(
            [
                response(tool_calls=[call("spawn", profile="digger", task="dig", background=True)]),
                response("handed off"),
            ]
        ),
        node=AgentNode(1.0),
        subagent_profiles={"digger": grandchild},
    )
    run = Run.start(child, "go")
    assert run.outcome(TIMEOUT) == done("handed off")
    ((_, _, dig),) = child.subagents.snapshot()
    assert dig.done()  # closed before the run completed: nothing below it still running
    assert (dig.outcome(0).status, dig.outcome(0).reason) == ("stopped", "cancelled")
    assert len(grandchild_client.requests) < 21


def test_a_malformed_response_fails_the_run():
    # Catches a subagent's broken response escaping the run.
    bad = response("x", usage=usage(cost=0.005))
    bad["choices"] = []
    root = AgentNode(1.0)
    run = Run.start(make_agent([bad], root.child(0.5, "1")), "go")
    outcome = run.outcome(TIMEOUT)
    assert outcome is not None and outcome.status == "failed" and outcome.text.startswith("MalformedResponse:")


# ---------------------------------------------------------------- resuming


def test_resume_continues_the_same_agent():
    # Catches a follow-up that starts a fresh agent, a new node, or drops the earlier turn from the context.
    agent = make_agent([response("first")])
    run = Run.start(agent, "go")
    assert run.outcome(TIMEOUT) == done("first")
    scripted(agent).responses.append(response("second"))
    again = run.resume("more")
    assert again.outcome(TIMEOUT) == done("second")
    assert again is not run and again.agent is agent and again.path == run.path
    assert (run.task, again.task) == ("go", "more")
    contents = [message.content for message in agent.messages]
    assert contents[-2:] == ["more", "second"] and "first" in contents[:-2]  # a second turn adds the head marker
    sent = [message["content"] for message in scripted(agent).requests[-1]["messages"]]
    assert "first" in sent and sent[-1] == "more"  # the follow-up's call saw turn one


def test_resume_refuses_a_running_run():
    # Catches a second turn started on an agent that is still in its first one.
    agent = make_agent([response("later")])
    gate = gated(agent)
    run = Run.start(agent, "/gate go")
    with pytest.raises(ValueError, match="still running"):
        run.resume("more")
    gate.set()
    assert run.outcome(TIMEOUT) == done("later")
    assert len(scripted(agent).requests) == 1


def failed_run() -> Run:
    return Run.start(make_agent([RuntimeError("HTTP 500")]), "go")


def cancelled_run() -> Run:
    agent = make_agent([])
    gate = gated(agent)
    run = Run.start(agent, "/gate go")
    run.cancel()
    gate.set()
    return run


def budget_spent_run() -> Run:
    return Run.start(make_agent([tool_reply(cost=0.004, content="let me check")], AgentNode(0.004)), "go")


def tool_rounds_run() -> Run:
    return Run.start(make_agent([tool_reply(), tool_reply()], max_tool_rounds=1), "go")


@pytest.mark.parametrize(
    ("start", "first"),
    [
        (failed_run, ("failed", None)),
        (cancelled_run, ("stopped", "cancelled")),
        (budget_spent_run, ("stopped", "budget spent")),
        (tool_rounds_run, ("stopped", "out of tool rounds")),
    ],
)
def test_resume_after_every_ending(start, first):
    # Catches a run that can't be continued because of how it stopped: nothing about an ending is final.
    run = start()
    outcome = run.outcome(TIMEOUT)
    assert (outcome.status, outcome.reason) == first
    run.agent.max_cost = run.agent.node.spent + 1.0  # a spent budget is topped up, as Subagents.resume does
    scripted(run.agent).responses.append(response("carried on"))
    again = run.resume("carry on")
    assert again.outcome(TIMEOUT) == done("carried on")
    assert not run.agent.node.cancelled


def test_resume_after_a_full_context_stops_again():
    # Catches a resumed full context that calls the model anyway, or isn't reported as still full.
    run = Run.start(make_agent([], context_window=10, compact_at=None), "go")
    assert run.outcome(TIMEOUT).reason == "context full"
    again = run.resume("carry on")
    assert (again.outcome(TIMEOUT).status, again.outcome(TIMEOUT).reason) == ("stopped", "context full")
    assert scripted(run.agent).requests == []


def test_a_resumed_run_keeps_its_report():
    # Catches a resumed research lead answering with its last reply instead of its notebook.
    agent = make_agent([response("see the notebook"), response("updated the notebook")])
    run = Run.start(agent, "go", report=lambda: "the notebook")
    assert run.outcome(TIMEOUT) == done("the notebook")
    assert run.resume("dig deeper").outcome(TIMEOUT) == done("the notebook")
