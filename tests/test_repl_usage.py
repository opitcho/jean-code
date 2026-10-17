"""Tests for what the REPL prints (spend, stops, subagents, compaction) and how it answers approvals, driven through a
`Driver` built on a scripted agent. Logging itself is tested through `AgentNode` in `test_budget.py`.

Each test names the bug it would catch.
"""

import threading

from conftest import FakeClient, call, response, usage
from jean_code.core.agent import NOT_RUN, Agent
from jean_code.core.driver import Driver
from jean_code.core.events import JobEnded
from jean_code.core.node import AgentNode
from jean_code.core.subagents import agent_profile
from jean_code.core.tool_schema import tool
from jean_code.ui.repl import print_cost, repl, run_and_print

TIMEOUT = 5  # seconds any wait in these tests may take before it counts as a hang


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


def make_session(responses, credit_data=None, max_cost: float | None = None, log_path=None, **agent_kwargs):
    """A driver whose agent answers from `responses`, and the list it builds that agent into."""
    built: list[Agent] = []

    def build(node: AgentNode, approve) -> Agent:
        tools = agent_kwargs.pop("tools", {"add": add})
        agent = Agent(
            client=FakeClient(responses, credit_data), tools=tools, node=node, approve=approve, **agent_kwargs
        )
        built.append(agent)
        return agent

    return Driver(build, max_cost=max_cost, log_path=log_path), built


def run(session: Driver, line: str) -> JobEnded:
    with session.view.subscribe() as (_, events):
        return run_and_print(session, events, line)


def scripted_input(lines):
    """A stand-in for `input` that returns `lines` in order, then acts like Ctrl+D."""
    lines = iter(lines)

    def fake(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    return fake


# ---------------------------------------------------------------- how a turn ends


def test_budget_stop_message(capsys):
    # Catches the wrong message on a budget stop.
    session, _ = make_session([tool_reply() for _ in range(11)], max_cost=0.01)
    run(session, "go")
    out = capsys.readouterr().out
    assert "[budget spent]" in out
    assert "[cancelled]" not in out


def test_round_limit_stop_message(capsys):
    # Catches a turn that ran out of tool rounds ending without a word, as if the model had answered.
    session, _ = make_session([tool_reply(), tool_reply()], max_tool_rounds=1)
    run(session, "go")
    assert "[out of tool rounds]" in capsys.readouterr().out


def test_an_answered_turn_prints_no_stop_message(capsys):
    # Catches the ending printed for every turn, not only for the ones that stopped early.
    session, _ = make_session([response("hi")])
    assert run(session, "go").result == "answered"
    assert "[answered]" not in capsys.readouterr().out


def test_run_and_print_reports_a_turn_that_raises(capsys):
    # Catches the REPL crashing, or waiting forever, after a turn that failed.
    session, _ = make_session([RuntimeError("boom")])
    ended = run(session, "go")
    assert ended.result is None and ended.error == "RuntimeError: boom"
    assert "[error] RuntimeError: boom" in capsys.readouterr().out


def test_unknown_cost_message(capsys):
    # Catches an unknown cost ending the turn, or shown as free or as a budget stop.
    session, _ = make_session([response("hi", no_usage=True)], max_cost=1.0)
    assert run(session, "go").result == "answered"
    out = capsys.readouterr().out
    assert "$?" in out and "[error]" not in out
    assert "budget of" not in out and "[cancelled]" not in out


def test_the_usage_line_sums_the_turns_calls(capsys):
    # Catches the per-turn line counting earlier turns, or missing calls.
    session, _ = make_session([response("hi", usage=usage(cost=0.002)), tool_reply(), tool_reply(), response("done")])
    run(session, "earlier turn")
    capsys.readouterr()
    run(session, "go")
    assert "$0.0090 · " in capsys.readouterr().out  # 0.004 + 0.004 + 0.001, not the earlier 0.002


# ---------------------------------------------------------------- approvals


def gated_session(responses) -> tuple[Driver, list[str]]:
    """A session whose `rm` needs approval, and the paths `rm` ran on."""
    ran = []

    @tool
    def rm(path: str) -> str:
        """Remove a file.

        Args:
            path: The file.
        """
        ran.append(path)
        return f"removed {path}"

    session, _ = make_session(responses, tools={"rm": rm}, needs_approval={"rm"})
    return session, ran


def test_repl_asks_for_approval_and_runs_the_call(monkeypatch, capsys):
    # Catches the question never asked: the turn would wait forever. And the call printed after its question.
    session, ran = gated_session([response(tool_calls=[call("rm", path="x")]), response("done")])
    monkeypatch.setattr("builtins.input", scripted_input(["delete x", "y"]))
    repl(session)
    out = capsys.readouterr().out
    assert ran == ["x"] and len(session.view.state().calls) == 2
    assert out.index("tool> rm(") < out.index("result> removed x") < out.index("agent> done")


def test_a_reason_typed_at_the_approval_prompt_denies_the_call(monkeypatch, capsys):
    # Catches anything but y/yes running the call, and the reason not reaching the model.
    session, ran = gated_session([response(tool_calls=[call("rm", path="x")]), response("ok")])
    monkeypatch.setattr("builtins.input", scripted_input(["delete x", "keep it"]))
    repl(session)
    assert ran == [] and "result> Denied by user. keep it" in capsys.readouterr().out


def test_ctrl_d_at_the_approval_prompt_stops_the_turn(monkeypatch, capsys):
    # Catches a turn left waiting forever, or a second question, after the user walked away from the first.
    reply = response(tool_calls=[call("rm", path="x"), call("rm", path="y")])
    session, ran = gated_session([reply, response("never")])
    monkeypatch.setattr("builtins.input", scripted_input(["delete x and y"]))
    repl(session)
    out = capsys.readouterr().out
    assert ran == [] and out.count(f"result> {NOT_RUN}") == 2 and "[cancelled]" in out


def test_repl_logs_every_call(monkeypatch, capsys, tmp_path):
    # Catches the session not wiring the usage log into the root node.
    path = tmp_path / "usage.jsonl"
    session, _ = make_session([tool_reply(), tool_reply(), response("done")], log_path=path)
    monkeypatch.setattr("builtins.input", scripted_input(["go"]))
    repl(session)
    assert len(path.read_text().splitlines()) == 3


# ---------------------------------------------------------------- subagents and compaction


def echo_profile(release: threading.Event | None = None):
    @tool
    def hold() -> str:
        """Wait until released."""
        if release:
            release.wait(TIMEOUT)
        return "released"

    def build(task: str, node: AgentNode) -> Agent:
        responses = [response(tool_calls=[call("hold")]), response(f"answer to {task}")]
        return Agent(client=FakeClient(responses), tools={"hold": hold}, node=node)

    return agent_profile("answers", build, max_cost=1.0)


def test_a_foreground_subagent_gets_one_started_and_one_ended_line(capsys):
    # Catches a subagent's tool output printed among the main agent's, or its start or end not shown at all.
    session, _ = make_session(
        [response(tool_calls=[call("spawn", profile="echo", task="t")]), response("done")],
        subagent_profiles={"echo": echo_profile()},
    )
    run(session, "go")
    out = capsys.readouterr().out
    assert out.count("[main/1 echo: started]") == 1 and out.count("[main/1: answered]") == 1
    assert "hold" not in out and "answer to t" in out  # its result, as main's tool result


def test_background_events_while_the_prompt_waits_are_not_printed_at_the_next_turn(monkeypatch, capsys):
    # Catches the next turn opening with a dump of a background run's output from while the prompt waited.
    release = threading.Event()
    session, _ = make_session(
        [
            response(tool_calls=[call("spawn", profile="echo", task="t", background=True)]),
            response("started"),
            response("got it"),
        ],
        subagent_profiles={"echo": echo_profile(release)},
    )
    lines = iter(["go", "next"])

    def fake_input(prompt=""):
        line = next(lines, None)
        if line is None:
            raise EOFError
        if line == "next":  # the run finishes while this prompt waits
            release.set()
            for _ in range(TIMEOUT * 100):
                if session.agent.subagents.pending:
                    break
                threading.Event().wait(0.01)
        return line

    monkeypatch.setattr("builtins.input", fake_input)
    repl(session)
    out = capsys.readouterr().out
    assert out.count("[main/1 echo: started]") == 1 and "[main/1: answered]" not in out
    assert "agent> got it" in out and "answer to t" not in out  # its result went to the model, not the screen


def test_waiting_results_line_before_the_prompt(monkeypatch, capsys):
    # Catches a finished background result the user is never told about.
    session, _ = make_session(
        [response(tool_calls=[call("spawn", profile="echo", task="t", background=True)]), response("started")],
        subagent_profiles={"echo": echo_profile()},
    )
    run(session, "go")
    for _ in range(TIMEOUT * 100):
        if session.agent.subagents.pending:
            break
        threading.Event().wait(0.01)
    capsys.readouterr()
    monkeypatch.setattr("builtins.input", scripted_input([]))
    repl(session)
    out = capsys.readouterr().out
    assert "[1 subagent result waiting; your next message delivers it]" in out
    assert "[main/1: answered]" not in out  # the run's end arrived while no turn ran: not printed


def test_compact_on_a_fresh_chat_says_nothing_to_compact(monkeypatch, capsys):
    # Catches /compact that changes nothing getting no reply at all.
    session, _ = make_session([])
    monkeypatch.setattr("builtins.input", scripted_input(["/compact"]))
    repl(session)
    assert "[compaction nothing to compact]" in capsys.readouterr().out


# ---------------------------------------------------------------- /cost


def test_print_cost_with_a_limit(capsys):
    session, _ = make_session(
        [response("hi", usage=usage(cost=0.0021))], credit_data={"usage": 1.5, "limit": 10.0, "limit_remaining": 8.5}
    )
    run(session, "hello")
    capsys.readouterr()
    print_cost(session)
    assert capsys.readouterr().out.splitlines() == [
        "session: $0.0021 over 1 model calls",
        "API key: $1.50 used of $10.00 ($8.50 left)",
    ]


def test_print_cost_without_a_limit(capsys):
    # Catches formatting a None limit.
    session, _ = make_session([], credit_data={"usage": None, "limit": None, "limit_remaining": None})
    print_cost(session)
    assert capsys.readouterr().out.splitlines() == ["session: $0.0000 over 0 model calls", "API key: $0.00 used"]


def test_print_cost_when_credits_fail(capsys):
    # Catches /cost crashing when OpenRouter is down.
    session, _ = make_session([], credit_data=RuntimeError("HTTP 503"))
    print_cost(session)
    assert capsys.readouterr().out.splitlines()[-1] == "credits unavailable: RuntimeError: HTTP 503"


def test_print_cost_breaks_down_a_tree(capsys):
    # Catches the breakdown missing once there are subagents, which is when it matters.
    session, _ = make_session(
        [response(tool_calls=[call("spawn", profile="echo", task="t")]), response("done")],
        credit_data=RuntimeError("no key"),
        subagent_profiles={"echo": echo_profile()},
    )
    run(session, "go")
    capsys.readouterr()
    print_cost(session)
    out = capsys.readouterr().out
    assert "main/1" in out and "path" in out.splitlines()[1]
