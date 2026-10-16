"""Tests for what the REPL prints about spend and stops, and that it wires the usage log.
Logging itself is tested through `Budget` in `test_budget.py`.

Each test names the bug it would catch.
"""

from conftest import FakeClient, call, response, scripted, usage
from jean_code.core.agent import NOT_RUN, Agent
from jean_code.core.llm import tool
from jean_code.ui.approval import Approvals
from jean_code.ui.repl import print_cost, repl, run_and_print


@tool
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


def scripted_input(lines):
    """A stand-in for `input` that returns `lines` in order, then acts like Ctrl+D."""
    lines = iter(lines)

    def fake(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    return fake


def test_budget_stop_message(capsys):
    # Catches the wrong message on a budget stop.
    agent = make_agent([tool_reply() for _ in range(11)], max_cost=0.01)
    run_and_print(agent, "go")
    out = capsys.readouterr().out
    assert "[budget spent]" in out
    assert "[interrupted]" not in out


def test_round_limit_stop_message(capsys):
    # Catches a turn that ran out of tool rounds ending without a word, as if the model had answered.
    agent = make_agent([tool_reply(), tool_reply()], max_tool_rounds=1)
    run_and_print(agent, "go")
    assert "[out of tool rounds]" in capsys.readouterr().out


def test_an_answered_turn_prints_no_stop_message(capsys):
    # Catches the ending printed for every turn, not only for the ones that stopped early.
    agent = make_agent([response("hi")])
    assert run_and_print(agent, "go") == "answered"
    assert "[answered]" not in capsys.readouterr().out


def test_run_and_print_returns_none_when_the_turn_raises(capsys):
    # Catches the REPL asking for approvals, or crashing, after a turn that failed.
    agent = make_agent([RuntimeError("boom")])
    assert run_and_print(agent, "go") is None
    assert "[error] RuntimeError: boom" in capsys.readouterr().out


def gated_agent(responses) -> tuple[Agent, Approvals, list[str]]:
    """An agent whose `rm` needs approval through an `Approvals`, and the paths `rm` ran on."""
    ran = []

    @tool
    def rm(path: str) -> str:
        """Remove a file.

        Args:
            path: The file.
        """
        ran.append(path)
        return f"removed {path}"

    approvals = Approvals()
    agent = Agent(client=FakeClient(responses), tools={"rm": rm}, needs_approval={"rm"}, approve=approvals.ask)
    return agent, approvals, ran


def test_repl_asks_for_approval_and_runs_the_call(monkeypatch, capsys):
    # Catches the question never asked: the turn would wait forever. And the call printed after its question.
    agent, approvals, ran = gated_agent([response(tool_calls=[call("rm", path="x")]), response("done")])
    monkeypatch.setattr("builtins.input", scripted_input(["delete x", "y"]))
    repl(agent, approvals, usage_log=None)
    out = capsys.readouterr().out
    assert ran == ["x"] and len(scripted(agent).requests) == 2
    assert out.index("tool> rm(") < out.index("result> removed x") < out.index("agent> done")


def test_a_reason_typed_at_the_approval_prompt_denies_the_call(monkeypatch, capsys):
    # Catches anything but y/yes running the call, and the reason not reaching the model.
    agent, approvals, ran = gated_agent([response(tool_calls=[call("rm", path="x")]), response("ok")])
    monkeypatch.setattr("builtins.input", scripted_input(["delete x", "keep it"]))
    repl(agent, approvals, usage_log=None)
    assert ran == [] and "result> Denied by user. keep it" in capsys.readouterr().out


def test_ctrl_d_at_the_approval_prompt_stops_the_turn(monkeypatch, capsys):
    # Catches a turn left waiting forever, or a second question, after the user walked away from the first.
    reply = response(tool_calls=[call("rm", path="x"), call("rm", path="y")])
    agent, approvals, ran = gated_agent([reply, response("never")])
    monkeypatch.setattr("builtins.input", scripted_input(["delete x and y"]))
    repl(agent, approvals, usage_log=None)
    out = capsys.readouterr().out
    assert ran == [] and out.count(f"result> {NOT_RUN}") == 2 and "[interrupted]" in out


def test_unknown_cost_message(capsys):
    # Catches the unknown-cost error missing from the output or shown as a budget stop.
    agent = make_agent([response("hi", no_usage=True)], max_cost=1.0)
    run_and_print(agent, "go")
    out = capsys.readouterr().out
    assert "[error] RuntimeError: model call" in out and "reported no cost" in out
    assert "budget of" not in out and "[interrupted]" not in out


def test_the_usage_line_sums_the_turns_calls(capsys):
    # Catches the per-turn line counting earlier turns, or missing calls.
    agent = make_agent([response("hi", usage=usage(cost=0.002)), tool_reply(), tool_reply(), response("done")])
    agent.run_turn("earlier turn")
    run_and_print(agent, "go")
    assert "$0.0090 · " in capsys.readouterr().out  # 0.004 + 0.004 + 0.001, not the earlier 0.002


def test_repl_logs_every_call(monkeypatch, capsys, tmp_path):
    # Catches the REPL not wiring the usage log into the agent's budget.
    path = tmp_path / "usage.jsonl"
    agent = make_agent([tool_reply(), tool_reply(), response("done")])
    monkeypatch.setattr("builtins.input", scripted_input(["go"]))
    repl(agent, usage_log=path)
    assert len(path.read_text().splitlines()) == 3


def test_print_cost_with_a_limit(capsys):
    agent = make_agent(
        [response("hi", usage=usage(cost=0.0021))], credit_data={"usage": 1.5, "limit": 10.0, "limit_remaining": 8.5}
    )
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
