"""Tests for how the agent records usage and enforces `max_cost`, and for `ChatClient`'s error handling.

Each test names the bug it would catch.
"""

import httpx
import pytest

import llm
from agent import Agent
from conftest import FakeClient, call, response, usage
from llm import ChatClient


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def make_agent(responses, **kwargs) -> tuple[Agent, FakeClient]:
    client = FakeClient(responses)
    ran: list[dict] = []

    def rm(path: str) -> str:
        """Remove a file.

        Args:
            path: The file.
        """
        ran.append({"path": path})
        return f"removed {path}"

    kwargs.setdefault("tools", {"add": add, "rm": rm})
    agent = Agent(client=client, **kwargs)
    agent.ran = ran  # calls of `rm` that actually ran
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
    assert agent.usage == [{**u, "id": "gen-abc", "provider": "DeepSeek",
                            "model": "deepseek/deepseek-v4.1-flash", "created": 1789748403}]


def test_every_call_of_a_turn_is_recorded_in_order():
    # Catches keeping usage only for the final call of a turn.
    agent, client = make_agent([tool_reply(cost=0.001, id="gen-1"), tool_reply(cost=0.002, id="gen-2"),
                                response("done", usage=usage(cost=0.003), id="gen-3")])
    agent.run_turn("go")
    assert len(client.requests) == 3
    assert [e["id"] for e in agent.usage] == ["gen-1", "gen-2", "gen-3"]
    assert agent.total_cost == pytest.approx(0.006)


def test_response_without_usage_is_recorded_with_unknown_cost():
    # Catches a call without usage leaving no trace and counting as free.
    agent, _ = make_agent([response("hi", no_usage=True, id="gen-x")])
    agent.run_turn("hello")
    assert len(agent.usage) == 1
    assert agent.usage[0]["id"] == "gen-x"
    assert agent.usage[0]["cost"] is None


def test_usage_is_recorded_before_a_malformed_reply_raises():
    # Catches a paid call being lost when its reply can't be read.
    bad = response("x", usage=usage(cost=0.005))
    bad["choices"] = []
    agent, _ = make_agent([bad])
    with pytest.raises(IndexError):
        agent.run_turn("hello")
    assert agent.total_cost == pytest.approx(0.005)


@pytest.mark.parametrize("decide", ["approve", "deny"])
def test_calls_after_an_approval_decision_are_recorded(decide):
    # Catches usage recorded only on the run_turn path.
    agent, _ = make_agent(
        [response(tool_calls=[call("rm", path="x")], id="gen-1"), response("ok", id="gen-2")],
        needs_approval={"rm"},
    )
    agent.run_turn("delete x")
    assert agent.pending_calls
    getattr(agent, decide)()
    assert [e["id"] for e in agent.usage] == ["gen-1", "gen-2"]


# ---------------------------------------------------------------- budget
# Every scripted call costs $0.004 and calls a tool, so only the budget or the round limit stops the loop.


def test_budget_is_checked_before_every_call():
    # Catches checking the budget once per turn: then all 11 calls would run.
    agent, client = make_agent([tool_reply() for _ in range(11)], max_cost=0.01)
    agent.run_turn("go")
    assert len(client.requests) == 3  # spent before each: 0, 0.004, 0.008 < 0.01; then 0.012 stops
    assert agent.interrupted and agent.over_budget


def test_spending_exactly_the_budget_stops():
    # Catches `>=` changed to `>`. Sums of 0.004 are exact (.experiments/output/budget_plan_check.txt).
    agent, client = make_agent([tool_reply() for _ in range(11)], max_cost=0.008)
    agent.run_turn("go")
    assert len(client.requests) == 2


def test_no_call_once_over_budget():
    # Catches the check moved after step(): that sends one more call (budget_plan_check.txt).
    agent, client = make_agent([response("hi", usage=usage(cost=0.02)), tool_reply()], max_cost=0.01)
    agent.run_turn("first")
    agent.run_turn("second")
    assert len(client.requests) == 1
    assert agent.messages[-1] == {"role": "user", "content": "second"}
    assert agent.interrupted


def test_approval_over_budget_runs_the_tool_but_calls_no_model():
    # Catches the approval path skipping the budget check.
    agent, client = make_agent(
        [response(tool_calls=[call("rm", path="x")], usage=usage(cost=0.02)), tool_reply()],
        needs_approval={"rm"}, max_cost=0.01,
    )
    agent.run_turn("delete x")
    agent.approve()
    assert agent.ran == [{"path": "x"}]
    assert len(client.requests) == 1
    assert agent.interrupted


def test_no_budget_runs_to_the_round_limit():
    # Catches a None comparison raising TypeError.
    agent, client = make_agent([tool_reply() for _ in range(5)], max_tool_rounds=4)
    agent.run_turn("go")
    assert len(client.requests) == 5
    assert not agent.interrupted


@pytest.mark.parametrize("unknown", [
    response(tool_calls=[call("add", a=1, b=2)], no_usage=True),
    response(tool_calls=[call("add", a=1, b=2)], usage=usage(cost=None)),
], ids=["no usage", "cost None"])
def test_unknown_cost_stops_the_turn_when_there_is_a_budget(unknown):
    # Catches the budget failing open, or one unknown cost stopping every later turn.
    agent, client = make_agent([tool_reply(), unknown, tool_reply(), response("done")], max_cost=1.0)
    agent.run_turn("go")
    assert len(client.requests) == 2
    assert agent.interrupted and agent.last_cost_unknown and not agent.over_budget

    agent.run_turn("continue")
    assert len(client.requests) == 4
    assert not agent.interrupted


def test_unknown_cost_without_a_budget_does_not_stop():
    agent, client = make_agent([tool_reply(), response(tool_calls=[call("add", a=1, b=2)], no_usage=True),
                                response("done")])
    agent.run_turn("go")
    assert len(client.requests) == 3
    assert not agent.interrupted and not agent.last_cost_unknown


@pytest.mark.parametrize("max_cost", [0, -1])
def test_non_positive_budget_is_rejected(max_cost):
    # Catches the app dividing by a zero budget on every rerun.
    with pytest.raises(ValueError):
        make_agent([], max_cost=max_cost)


def test_turn_after_raising_the_budget_runs():
    # Catches the budget stop leaving the interrupt set forever.
    agent, client = make_agent([tool_reply() for _ in range(3)] + [response("done")], max_cost=0.01)
    agent.run_turn("go")
    assert agent.interrupted
    agent.max_cost = 1.0
    agent.run_turn("more")
    assert len(client.requests) == 4
    assert not agent.interrupted


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
    client = mock_client(monkeypatch, httpx.Response(429, json={"id": "gen-rate-limited", "usage": usage()}),
                         httpx.Response(200, json=ok))
    assert client.complete({"messages": []}) == ok


def test_error_body_with_status_200_raises_and_records_nothing(monkeypatch):
    # Catches an error body being recorded as a call.
    client = mock_client(monkeypatch, httpx.Response(200, json={"error": {"message": "provider down"}}))
    agent = Agent(client=client)
    with pytest.raises(RuntimeError, match="provider down"):
        agent.run_turn("hello")
    assert agent.usage == []
