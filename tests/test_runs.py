"""Tests for the subagents' Python API: `Profile`, `Run` and the `Outcome` a run ends with. No tools involved,
except where a run's own subagents are the point.

Each test names the bug it would catch.
"""

import threading
import time

import pytest

from agent import Agent
from conftest import FakeClient, call, response, usage
from subagents import Outcome, Profile, Run, agent_profile
from usage import Budget

TIMEOUT = 5  # seconds any wait in these tests may take before it counts as a hang


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def slow() -> str:
    """Wait a moment."""
    time.sleep(0.1)
    return "ok"


def tool_reply(cost=0.001, content="") -> dict:
    return response(content, tool_calls=[call("add", a=1, b=2)], usage=usage(cost=cost))


def make_agent(responses, budget=None, **kwargs) -> Agent:
    return Agent(client=FakeClient(responses), tools={"add": add}, budget=budget or Budget(1.0), **kwargs)


def gated(agent: Agent) -> threading.Event:
    """Hold a run whose task starts with `/gate` just before `run_turn` clears the interrupt flag, until set."""
    gate = threading.Event()
    agent.commands["gate"] = lambda rest: gate.wait(TIMEOUT) and None  # None: the task goes on as typed
    return gate


def done(text: str) -> Outcome:
    return Outcome("done", text)


# ---------------------------------------------------------------- profiles and outcomes


def test_each_start_is_a_new_run():
    # Catches a profile that shares an agent (or its state) between runs.
    built = []

    def build(task: str, budget: Budget) -> Agent:
        built.append(make_agent([response(f"answer to {task}")], budget))
        return built[-1]

    profile = agent_profile("helper", build, max_cost=1.0)
    root = Budget()
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
    start = lambda task, budget: None  # noqa: E731
    with pytest.raises(ValueError, match="positive"):
        Profile("p", start, max_cost=0)
    with pytest.raises(ValueError, match="at least"):
        Profile("p", start, max_cost=1.0, max_cost_cap=0.5)
    assert Profile("p", start, max_cost=1.0).cap == 1.0
    assert Profile("p", start, max_cost=1.0, max_cost_cap=3.0).cap == 3.0


def test_a_subagent_that_would_need_approval_is_refused():
    # Catches a run that stops at its first `bash` call waiting for an approval nobody can give.
    with pytest.raises(ValueError, match="approval: add"):
        Run.start(make_agent([], needs_approval={"add"}), "go")


# ---------------------------------------------------------------- stopped, not done


def test_budget_spent_is_stopped_with_the_last_remark():
    # Catches a mid-task remark handed back as the answer after the budget ran out.
    run = Run.start(make_agent([tool_reply(cost=0.004, content="let me check"), response("never")], Budget(0.004)), "go")
    assert run.outcome(TIMEOUT) == Outcome("stopped", "let me check", "budget spent")


def test_an_answer_that_spends_the_last_of_the_budget_is_done():
    # Catches a finished answer labelled "budget spent" because the answering call used the budget up.
    run = Run.start(make_agent([response("the answer", usage=usage(cost=0.004))], Budget(0.004)), "go")
    assert run.outcome(TIMEOUT) == done("the answer")


def test_unknown_cost_is_stopped():
    run = Run.start(make_agent([response(tool_calls=[call("add", a=1, b=2)], no_usage=True)]), "go")
    outcome = run.outcome(TIMEOUT)
    assert outcome.status == "stopped" and outcome.reason.startswith("a call's cost was unknown")


def test_a_full_context_is_stopped():
    outcome = Run.start(make_agent([], context_window=10, compact_at=None), "go").outcome(TIMEOUT)
    assert outcome.status == "stopped" and outcome.reason.startswith("the context is nearly full")


def test_running_out_of_tool_rounds_is_stopped():
    # Catches the tool-round limit, which never interrupts, reported as an answer.
    run = Run.start(make_agent([tool_reply(), tool_reply()], max_tool_rounds=1), "go")
    assert run.outcome(TIMEOUT) == Outcome("stopped", "(no reply)", "out of tool rounds (1)")


# ---------------------------------------------------------------- failed


def test_a_crash_fails_the_run():
    # Catches a crashed run that stays "running" forever, or whose error escapes as an exception.
    run = Run.start(make_agent([RuntimeError("HTTP 500")]), "go")
    assert run.outcome(TIMEOUT) == Outcome("failed", "RuntimeError: HTTP 500")


def test_a_failing_report_fails_the_run():
    run = Run.start(make_agent([response("hi")]), "go", report=lambda: 1 / 0)
    assert run.outcome(TIMEOUT) == Outcome("failed", "ZeroDivisionError: division by zero")


def test_a_failing_cleanup_still_completes_the_run():
    # Catches `close()` raising outside the try, which would leave the run "running" forever.
    def boom():
        raise OSError("container already gone")

    run = Run.start(make_agent([response("hi")], cleanup=[boom]), "go")
    assert run.outcome(TIMEOUT) == Outcome("failed", "OSError: container already gone")


# ---------------------------------------------------------------- cancelling


def test_a_cancel_before_the_turn_starts_is_not_lost():
    # Catches cancel() relying on the interrupt flag, which `run_turn` clears when it starts.
    agent = make_agent([response("never")])
    gate = gated(agent)
    run = Run.start(agent, "/gate go")  # held just before run_turn clears the flag
    run.cancel()
    gate.set()
    assert run.outcome(TIMEOUT) == Outcome("stopped", "(no reply)", "cancelled")
    assert agent.client.requests == []


def test_the_gate_reaches_the_window_an_interrupt_would_lose():
    # The control for the test above: an interrupt sent at the same moment is cleared, and the run answers.
    agent = make_agent([response("answered anyway")])
    gate = gated(agent)
    run = Run.start(agent, "/gate go")
    agent.interrupt()
    gate.set()
    assert run.outcome(TIMEOUT) == done("answered anyway")


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
        "digger", lambda task, budget: Agent(client=grandchild_client, tools={"slow": slow}, budget=budget), max_cost=1.0)
    child = Agent(
        client=FakeClient([response(tool_calls=[call("spawn", profile="digger", task="dig", background=True)]),
                           response("handed off")]),
        budget=Budget(1.0), subagent_profiles={"digger": grandchild})
    run = Run.start(child, "go")
    assert run.outcome(TIMEOUT) == done("handed off")
    (_, _, dig), = child.subagents.snapshot()
    assert dig.done()  # closed before the run completed: nothing below it still running
    assert (dig.outcome(0).status, dig.outcome(0).reason) == ("stopped", "cancelled")
    assert len(grandchild_client.requests) < 21
