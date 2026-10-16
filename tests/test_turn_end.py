"""Tests for what `run_turn` returns (how the turn ended), and for the calls that need approval.

Each test names the bug it would catch.
"""

from typing import get_args

import pytest

from conftest import FakeClient, call, response, usage
from jean_code.core.agent import NOT_RUN, Agent, TurnEnd, TurnStopped
from jean_code.core.llm import tool
from jean_code.core.usage import Budget


class AskingAgent(Agent):
    asked: list[tuple[str, dict]]  # the calls `approve` got


def make_agent(responses, client_class=FakeClient, **kwargs) -> AskingAgent:
    """An agent with `add`, `rm` (list it in `needs_approval` to have `approve` asked: by default it says yes)
    and `stop`, which interrupts the turn from inside a tool call. `agent.asked` lists the calls `approve` got."""
    agent: AskingAgent
    asked: list[tuple[str, dict]] = []
    answers = iter(kwargs.pop("answers", ()))  # what `approve` returns, call by call; None once they run out

    def approve(name: str, arguments: dict) -> str | None:
        asked.append((name, arguments))
        answer = next(answers, None)
        if isinstance(answer, Exception):
            raise answer
        return answer

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

    @tool
    def stop() -> str:
        """Interrupt the turn, as the user would while this call runs."""
        agent.interrupt()
        return "stopped"

    agent = AskingAgent(
        client=client_class(responses), tools={"add": add, "rm": rm, "stop": stop}, approve=approve, **kwargs
    )
    agent.asked = asked
    return agent


def results(agent: Agent) -> list[str]:
    return [m["content"] for m in agent.messages if m["role"] == "tool"]


def tool_reply(name="add", cost=0.004, prompt=100, **args) -> dict:
    """A reply that calls a tool, so the loop keeps going."""
    args = args or ({"a": 1, "b": 2} if name == "add" else {})
    return response(tool_calls=[call(name, **args)], usage=usage(cost=cost, prompt=prompt))


def cancelled_agent() -> Agent:
    budget = Budget().child(1.0, "1")
    budget.cancel()
    return make_agent([response("never")], budget=budget)


# What each scenario's turn must return. The keys are checked against `TurnEnd` below.
ENDINGS = {
    "answered": lambda: make_agent([response("hi")]),
    "interrupted": lambda: make_agent([tool_reply("stop"), response("never")]),
    "budget spent": lambda: make_agent([tool_reply() for _ in range(11)], max_cost=0.01),
    "cancelled": cancelled_agent,
    "context full": lambda: make_agent([tool_reply(prompt=1900)], compact_at=None, context_window=2000),
    "out of tool rounds": lambda: make_agent([tool_reply() for _ in range(5)], max_tool_rounds=4),
}


@pytest.mark.parametrize("ending", ENDINGS)
def test_every_ending(ending):
    # Catches a turn that ends one way and is reported as another, or as nothing.
    assert ENDINGS[ending]().run_turn("go") == ending


def test_the_endings_tested_are_the_declared_ones():
    # Catches an ending added to or dropped from `TurnEnd` without a scenario, and a misspelled one: nothing checks
    # the Literal at runtime.
    assert set(ENDINGS) == set(get_args(TurnEnd))


# ---------------------------------------------------------------- when two endings apply


def test_an_answer_that_spends_the_budget_is_answered():
    # Catches a finished answer reported as a budget stop because the answering call used the budget up.
    agent = make_agent([response("hi", usage=usage(cost=0.02))], max_cost=0.01)
    assert agent.run_turn("go") == "answered"


def test_the_budget_comes_before_the_round_limit():
    # Catches "out of tool rounds" for a turn that more rounds wouldn't help: its last allowed call spent the budget.
    agent = make_agent([tool_reply(cost=0.02)], max_cost=0.01, max_tool_rounds=0)
    assert agent.run_turn("go") == "budget spent"


def test_a_full_context_comes_before_the_round_limit():
    # Catches the same for the context: it filled up on the last allowed call.
    agent = make_agent([tool_reply(prompt=1900)], compact_at=None, context_window=2000, max_tool_rounds=0)
    assert agent.run_turn("go") == "context full"


def test_an_interrupt_comes_before_the_budget():
    # Catches the user's stop reported as a budget stop when the interrupted call also spent the budget.
    agent = make_agent([tool_reply("stop", cost=0.02), response("never")], max_cost=0.01)
    assert agent.run_turn("go") == "interrupted"


def test_an_interrupt_during_the_answer_is_answered():
    # Catches a complete answer reported as interrupted because the user's stop arrived while it did.
    class Interrupting(FakeClient):
        def complete(self, request):
            agent.interrupt()
            return super().complete(request)

    agent = make_agent([response("hi")], client_class=Interrupting)
    assert agent.run_turn("go") == "answered"


def test_no_call_runs_after_an_interrupt():
    # Catches a call run, or the approval prompt shown, after the user said stop: the reply's first call interrupts,
    # and the calls after it are not run, each with a result so the next request is valid.
    reply = response(tool_calls=[call("stop"), call("add", a=1, b=2), call("rm", path="x")])
    agent = make_agent([reply, response("never")], needs_approval={"rm"})
    assert agent.run_turn("go") == "interrupted"
    assert agent.asked == [] and results(agent) == ["stopped", NOT_RUN, NOT_RUN]


def test_an_interrupt_during_the_model_call_runs_none_of_its_tool_calls():
    # Catches the reply's calls run although the stop came before any of them started.
    class Interrupting(FakeClient):
        def complete(self, request):
            agent.interrupt()
            return super().complete(request)

    agent = make_agent([tool_reply(), response("never")], client_class=Interrupting)
    assert agent.run_turn("go") == "interrupted"
    assert results(agent) == [NOT_RUN]


# ---------------------------------------------------------------- calls that need approval


def test_every_call_that_needs_approval_is_asked_about_and_the_turn_goes_on():
    # Catches a call run unasked, one asked about twice, and a turn that ends at the question.
    reply = response(tool_calls=[call("rm", path="x"), call("add", a=1, b=2), call("rm", path="y")])
    agent = make_agent([reply, response("done")], needs_approval={"rm"})
    assert agent.run_turn("delete x and y") == "answered"
    assert agent.asked == [("rm", {"path": "x"}), ("rm", {"path": "y"})]
    assert results(agent) == ["removed x", "3", "removed y"]


def test_a_refusal_is_the_calls_result_and_the_tool_does_not_run():
    # Catches a refused call that runs anyway, and a refusal that ends the turn or the reply's other calls.
    reply = response(tool_calls=[call("rm", path="x"), call("rm", path="y")])
    agent = make_agent([reply, response("ok")], needs_approval={"rm"}, answers=["Denied by user. no"])
    assert agent.run_turn("delete x and y") == "answered"
    assert results(agent) == ["Denied by user. no", "removed y"]


def test_approve_can_stop_the_turn():
    # Catches a stop raised from inside `approve` lost, or leaving calls without results (a 400 on the next request).
    reply = response(tool_calls=[call("add", a=1, b=2), call("rm", path="x"), call("rm", path="y")])
    agent = make_agent([reply, response("never")], needs_approval={"rm"}, answers=[TurnStopped("interrupted")])
    assert agent.run_turn("go") == "interrupted"
    assert agent.asked == [("rm", {"path": "x"})] and results(agent) == ["3", NOT_RUN, NOT_RUN]


def test_approve_failing_leaves_a_valid_transcript():
    # Catches the same after a bug in the front-end's `approve`: the error surfaces, and the next turn still works.
    agent = make_agent([tool_reply("rm", path="x"), response("ok")], needs_approval={"rm"}, answers=[OSError("tty")])
    with pytest.raises(OSError, match="tty"):
        agent.run_turn("delete x")
    assert results(agent) == [NOT_RUN]
    assert agent.run_turn("never mind") == "answered"


def test_a_call_with_invalid_arguments_is_not_asked_about():
    # Catches a question about a call that can't run: its error is the result, as for any tool.
    reply = response(tool_calls=[call("rm", path="x")])
    reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    agent = make_agent([reply, response("ok")], needs_approval={"rm"})
    assert agent.run_turn("go") == "answered"
    assert agent.asked == [] and results(agent)[0].startswith("Error: JSONDecodeError")


def test_tools_that_need_approval_need_an_approve_function():
    # Catches an agent whose gated calls would crash mid-turn, when nobody is there to ask.
    with pytest.raises(ValueError, match=r"no `approve` function.*rm"):
        Agent(client=FakeClient([]), needs_approval={"rm"})
