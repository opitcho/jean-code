"""Tests for what the REPL prints about spend and stops. Logging itself is tested through `logged_calls`.

Each test names the bug it would catch.
"""

from agent import Agent
from conftest import FakeClient, call, response, usage
from repl import print_cost, run_and_print
from usage import COST_UNKNOWN


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def tool_reply(cost=0.004, **kwargs) -> dict:
    return response(tool_calls=[call("add", a=1, b=2)], usage=usage(cost=cost), **kwargs)


def make_agent(responses, credit_data=None, **kwargs) -> Agent:
    return Agent(client=FakeClient(responses, credit_data), tools={"add": add}, **kwargs)


def test_budget_stop_message(capsys, tmp_path):
    # Catches the wrong message on a budget stop.
    agent = make_agent([tool_reply() for _ in range(11)], max_cost=0.01)
    run_and_print(agent, tmp_path / "usage.jsonl", agent.run_turn, "go")
    out = capsys.readouterr().out
    assert "[budget of $0.01 reached]" in out
    assert "[interrupted]" not in out


def test_unknown_cost_message(capsys, tmp_path):
    # Catches the unknown-cost stop missing its message or showing the budget one.
    agent = make_agent([response("hi", no_usage=True)], max_cost=1.0)
    run_and_print(agent, tmp_path / "usage.jsonl", agent.run_turn, "go")
    out = capsys.readouterr().out
    assert COST_UNKNOWN in out
    assert "budget of" not in out and "[interrupted]" not in out


def test_run_writes_one_log_line_per_call(capsys, tmp_path):
    # Catches `logged_calls` not being wired into the REPL.
    path = tmp_path / "usage.jsonl"
    agent = make_agent([tool_reply(), tool_reply(), response("done")])
    run_and_print(agent, path, agent.run_turn, "go")
    assert len(path.read_text().splitlines()) == 3
    assert "$0.0090 · " in capsys.readouterr().out  # 0.004 + 0.004 + 0.001


def test_print_cost_with_a_limit(capsys):
    agent = make_agent([response("hi", usage=usage(cost=0.0021))],
                       credit_data={"usage": 1.5, "limit": 10.0, "limit_remaining": 8.5})
    agent.run_turn("hello")
    print_cost(agent)
    assert capsys.readouterr().out.splitlines() == [
        "session: $0.0021 over 1 model calls",
        "API key: $1.50 used of $10.00 ($8.50 left)",
    ]


def test_print_cost_without_a_limit(capsys):
    # Catches formatting a None limit.
    agent = make_agent([], credit_data={"usage": None, "limit": None, "limit_remaining": None})
    print_cost(agent)
    assert capsys.readouterr().out.splitlines() == ["session: $0.0000 over 0 model calls", "API key: $0.00 used"]


def test_print_cost_when_credits_fail(capsys):
    # Catches /cost crashing when OpenRouter is down.
    agent = make_agent([], credit_data=RuntimeError("HTTP 503"))
    print_cost(agent)
    assert capsys.readouterr().out.splitlines()[-1] == "credits unavailable: RuntimeError: HTTP 503"
