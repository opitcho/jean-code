"""Tests for how the agent records usage and enforces `max_cost`, and for `ChatClient`'s error handling.

Each test names the bug it would catch.
"""

import httpx
import pytest

from conftest import FakeClient, call, response, usage
from jean_code.core import llm
from jean_code.core.agent import Agent
from jean_code.core.llm import ChatClient, ChatRequest, MalformedResponse, UserMessage
from jean_code.core.tool_schema import tool
from jean_code.core.usage import ModelCall


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


class RecordingAgent(Agent):
    ran: list[dict]  # calls of `rm` that actually ran


def make_agent(responses, **kwargs) -> tuple[RecordingAgent, FakeClient]:
    client = FakeClient(responses)
    ran: list[dict] = []

    @tool
    def rm(path: str) -> str:
        """Remove a file.

        Args:
            path: The file.
        """
        ran.append({"path": path})
        return f"removed {path}"

    kwargs.setdefault("tools", {"add": add, "rm": rm})
    kwargs.setdefault("approve", lambda name, arguments: None)  # yes to every call that needs approval
    agent = RecordingAgent(client=client, **kwargs)
    agent.ran = ran
    return agent, client


def tool_reply(cost=0.004, **kwargs) -> dict:
    """A reply that calls `add`, so the loop keeps going."""
    return response(tool_calls=[call("add", a=1, b=2)], usage=usage(cost=cost), **kwargs)


# ---------------------------------------------------------------- capture in step()


def test_usage_entry_has_the_response_metadata():
    # Catches losing id/provider/model/created, so the log can't be matched against the OpenRouter dashboard.
    u = usage(prompt=500, cost=0.002)
    agent, _ = make_agent([response("hi", usage=u, id="gen-abc", created=1789748403)])
    agent.run_turn("hello")
    assert agent.node.entries == [
        ModelCall(
            "gen-abc",
            "DeepSeek",
            "deepseek/deepseek-v4.1-flash",
            1789748403,
            0.002,
            "main",
            prompt_tokens=500,
            completion_tokens=10,
            finish_reason="stop",
        )
    ]


def test_every_call_of_a_turn_is_recorded_in_order():
    # Catches keeping usage only for the final call of a turn.
    agent, client = make_agent(
        [
            tool_reply(cost=0.001, id="gen-1"),
            tool_reply(cost=0.002, id="gen-2"),
            response("done", usage=usage(cost=0.003), id="gen-3"),
        ]
    )
    agent.run_turn("go")
    assert len(client.requests) == 3
    assert [c.generation_id for c in agent.node.entries] == ["gen-1", "gen-2", "gen-3"]
    assert agent.node.spent == pytest.approx(0.006)


def test_response_without_usage_is_recorded_with_unknown_cost(caplog):
    # Catches a call without usage leaving no trace, or counting as free without a warning.
    agent, _ = make_agent([response("hi", no_usage=True, id="gen-x")])
    assert agent.run_turn("hello") == "answered"
    assert "gen-x reported no cost" in caplog.text
    assert len(agent.node.entries) == 1
    assert agent.node.entries[0].generation_id == "gen-x"
    assert agent.node.entries[0].cost is None


def test_malformed_response_raises_and_books_nothing():
    # Catches a response that couldn't be read being booked from fields nobody checked.
    bad = response("x", usage=usage(cost=0.002))
    bad["choices"][0]["message"]["content"] = [{"type": "text", "text": "hi"}]
    agent, _ = make_agent([bad])
    with pytest.raises(MalformedResponse, match=r"choices\.0\.message\.content"):
        agent.run_turn("hello")
    assert agent.node.entries == []


@pytest.mark.parametrize("refusal", [None, "Denied by user."], ids=["approve", "deny"])
def test_calls_after_an_approval_decision_are_recorded(refusal):
    # Catches usage lost for the calls made after a question was answered.
    agent, _ = make_agent(
        [response(tool_calls=[call("rm", path="x")], id="gen-1"), response("ok", id="gen-2")],
        needs_approval={"rm"},
        approve=lambda name, arguments: refusal,
    )
    agent.run_turn("delete x")
    assert [c.generation_id for c in agent.node.entries] == ["gen-1", "gen-2"]


# ---------------------------------------------------------------- budget
# Every scripted call costs $0.004 and calls a tool, so only the budget or the round limit stops the loop.


def test_budget_is_checked_before_every_call():
    # Catches checking the budget once per turn: then all 11 calls would run.
    agent, client = make_agent([tool_reply() for _ in range(11)], max_cost=0.01)
    end = agent.run_turn("go")
    assert len(client.requests) == 3  # spent before each: 0, 0.004, 0.008 < 0.01; then 0.012 stops
    assert end == "budget spent" and agent.node.exhausted


def test_spending_exactly_the_budget_stops():
    # Catches `>=` changed to `>`. Sums of 0.004 are exact (.experiments/output/budget_plan_check.txt).
    agent, client = make_agent([tool_reply() for _ in range(11)], max_cost=0.008)
    agent.run_turn("go")
    assert len(client.requests) == 2


def test_no_call_once_over_budget():
    # Catches the check moved after step(): that sends one more call (budget_plan_check.txt).
    agent, client = make_agent([response("hi", usage=usage(cost=0.02)), tool_reply()], max_cost=0.01)
    agent.run_turn("first")
    end = agent.run_turn("second")
    assert len(client.requests) == 1
    assert agent.messages[-1] == UserMessage(content="second")
    assert end == "budget spent"


def test_approval_over_budget_runs_the_tool_but_calls_no_model():
    # Catches an approved call skipped, or the model called again, when the reply that asked spent the budget.
    agent, client = make_agent(
        [response(tool_calls=[call("rm", path="x")], usage=usage(cost=0.02)), tool_reply()],
        needs_approval={"rm"},
        max_cost=0.01,
    )
    end = agent.run_turn("delete x")
    assert agent.ran == [{"path": "x"}]
    assert len(client.requests) == 1
    assert end == "budget spent"


def test_no_budget_runs_to_the_round_limit():
    # Catches a None comparison raising TypeError.
    agent, client = make_agent([tool_reply() for _ in range(5)], max_tool_rounds=4)
    end = agent.run_turn("go")
    assert len(client.requests) == 5
    assert end == "out of tool rounds"


@pytest.mark.parametrize(
    "unknown",
    [
        response(tool_calls=[call("add", a=1, b=2)], no_usage=True),
        response(tool_calls=[call("add", a=1, b=2)], usage=usage(cost=None)),
    ],
    ids=["no usage", "cost None"],
)
@pytest.mark.parametrize("max_cost", [1.0, None], ids=["budget", "no budget"])
def test_unknown_cost_warns_and_the_turn_carries_on(unknown, max_cost, caplog):
    # Catches a call of unknown cost stopping the turn, or passing without a warning, with or without a budget.
    agent, client = make_agent([tool_reply(), unknown, response("done")], max_cost=max_cost)
    assert agent.run_turn("go") == "answered"
    assert "reported no cost" in caplog.text
    assert len(client.requests) == 3
    assert len(agent.node.entries) == 3 and agent.node.entries[1].cost is None


@pytest.mark.parametrize("max_cost", [0, -1])
def test_non_positive_budget_is_rejected(max_cost):
    # Catches the app dividing by a zero budget on every rerun.
    with pytest.raises(ValueError):
        make_agent([], max_cost=max_cost)


def test_turn_after_raising_the_budget_runs():
    # Catches a budget stop that outlasts the budget being raised.
    agent, client = make_agent([tool_reply() for _ in range(3)] + [response("done")], max_cost=0.01)
    assert agent.run_turn("go") == "budget spent"
    agent.node.limit = 1.0
    assert agent.run_turn("more") == "answered"
    assert len(client.requests) == 4


# ---------------------------------------------------------------- ChatClient


def mock_client(monkeypatch, *replies: httpx.Response) -> ChatClient:
    """A ChatClient whose HTTP calls return `replies` in order, with no backoff sleeps."""
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    queue = list(replies)
    client = ChatClient("https://example.test/api", "key", retries=3)
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: queue.pop(0)))
    return client


def test_retry_returns_the_successful_body(monkeypatch):
    # Catches the failed attempt's body being returned as the response.
    ok = response("hi", id="gen-ok")
    client = mock_client(
        monkeypatch,
        httpx.Response(429, json={"id": "gen-rate-limited", "usage": usage()}),
        httpx.Response(200, json=ok),
    )
    assert client.complete(ChatRequest("m", (), (), None)).generation_id == "gen-ok"


def test_error_body_with_status_200_raises_and_records_nothing(monkeypatch):
    # Catches an error body being recorded as a call.
    client = mock_client(monkeypatch, httpx.Response(200, json={"error": {"message": "provider down"}}))
    agent = Agent(client=client)
    with pytest.raises(RuntimeError, match="provider down"):
        agent.run_turn("hello")
    assert agent.node.entries == []


def test_not_json_books_nothing(monkeypatch):
    # Catches a non-JSON body (a proxy's error page) crashing with a JSONDecodeError, or booking a call never made.
    client = mock_client(monkeypatch, httpx.Response(200, text="<html>bad gateway</html>"))
    agent = Agent(client=client)
    with pytest.raises(MalformedResponse, match="not JSON"):
        agent.run_turn("hello")
    assert agent.node.entries == []
