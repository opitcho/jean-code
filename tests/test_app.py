"""Tests for the Streamlit app (ui/app.py), run with Streamlit's AppTest on a `Session` whose agent is scripted.
The session is put in `st.session_state` before the first run, so the app draws it instead of building its own.

Each test names the bug it would catch.
"""

import threading
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import jean_code
from conftest import FakeClient, call, response, usage
from jean_code.core.agent import Agent
from jean_code.core.node import AgentNode
from jean_code.core.subagents import agent_profile
from jean_code.core.tool_schema import tool
from jean_code.session import Session, Settings

APP = Path(jean_code.__file__).parent / "ui" / "app.py"
TIMEOUT = 5  # seconds any wait in these tests may take before it counts as a hang
SMALL = usage(prompt=100, completion=5, cost=0.001)
BULK = " filler" * 2000


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def make_session(responses, tools=None, **agent_kwargs) -> tuple[Session, list[Agent]]:
    built: list[Agent] = []

    def build(settings: Settings, node: AgentNode, approve) -> Agent:
        agent = Agent(
            client=FakeClient(responses), tools=tools or {"add": add}, node=node, approve=approve, **agent_kwargs
        )
        built.append(agent)
        return agent

    return Session(Settings(), build=build, log_path=None), built


def app(session: Session) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=10)
    at.session_state["session"] = session
    return at.run()


def eventually(predicate) -> None:
    for _ in range(TIMEOUT * 100):
        if predicate():
            return
        threading.Event().wait(0.01)
    raise AssertionError("timed out")


def idle(session: Session) -> None:
    eventually(lambda: not session.state().busy)


def htmls(at: AppTest) -> list[str]:
    return [e.proto.body for e in at.get("html")]


def no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def test_a_turn_with_a_tool_call_renders_one_block_per_message_and_its_result():
    # Catches a timeline that misses messages, or a call whose result the inspector can't find.
    session, _ = make_session([response(tool_calls=[call("add", a=1, b=2)]), response("three")])
    at = app(session)
    at.chat_input[0].set_value("add 1 and 2").run()
    idle(session)
    at.run()
    no_errors(at)
    assert sorted(b.key for b in at.button if b.key and b.key.startswith("hit-")) == [
        "hit-1",
        "hit-2",
        "hit-3",
        "hit-4",
    ]
    at.button(key="hit-2").click().run()  # the tool call
    no_errors(at)
    assert any("add()" in h for h in htmls(at)) and "3" in [c.value for c in at.code]


@tool
def rm(path: str) -> str:
    """Remove a file.

    Args:
        path: The file.
    """
    return f"removed {path}"


@pytest.mark.parametrize("approve", [True, False])
def test_a_pending_approval_is_answered_from_its_card(approve):
    # Catches an approval card that doesn't show, or whose buttons don't reach the waiting turn.
    session, built = make_session(
        [response(tool_calls=[call("rm", path="x")]), response("ok")], tools={"rm": rm}, needs_approval={"rm"}
    )
    at = app(session)
    at.chat_input[0].set_value("delete x").run()
    eventually(lambda: session.state().approval is not None)
    at.run()
    no_errors(at)
    if approve:
        next(b for b in at.button if b.label == "Approve").click().run()
    else:
        at.text_input(key="deny_reason").set_value("keep it")
        next(b for b in at.button if b.label == "Deny").click().run()
    idle(session)
    expected = "removed x" if approve else "Denied by user. keep it"
    assert built[-1].messages[-2].content == expected


def test_compacted_messages_are_dimmed_under_a_divider():
    # Catches compacted messages dropped from the timeline, or shown as if still in context.
    session, _ = make_session([], system_prompt="sys", compact_at=1000, keep_first=2, keep_last=2)
    client = session._agent.client
    assert isinstance(client, FakeClient)
    for i in range(1, 6):
        client.responses.append(response(f"reply {i}" + (BULK if i == 3 else ""), usage=SMALL))
        session.send(f"turn {i}")
        idle(session)
    client.responses.append(response("SUMMARY", usage=usage(prompt=99_999, completion=300, cost=0.002)))
    session.compact()
    idle(session)
    at = app(session)
    no_errors(at)
    cards = [h for h in htmls(at) if 'class="tl-card' in h]
    assert sum(" out" in h.split('"')[1] for h in cards) == 2  # turn 3's message and reply
    assert any("2 messages compacted" in h for h in htmls(at))


def test_a_subagent_appears_in_the_picker_with_how_its_turn_ended():
    # Catches subagents missing from the picker, or labelled with a status their turn didn't have.
    def build(task: str, node: AgentNode) -> Agent:
        return Agent(client=FakeClient([response(f"answer to {task}")]), node=node)

    session, _ = make_session(
        [response(tool_calls=[call("spawn", profile="echo", task="t")]), response("done")],
        subagent_profiles={"echo": agent_profile("answers", build, 1.0)},
    )
    at = app(session)
    at.chat_input[0].set_value("go").run()
    idle(session)
    at.run()
    no_errors(at)
    picker = at.radio(key="view")
    assert picker.options == ["**main**", "↳ **#1** echo"]
    assert picker.proto.captions[1].startswith(":violet[●] answered · 1 calls")
    picker.set_value("main/1").run()
    no_errors(at)
    assert any("answer to t" in h for h in htmls(at))


def test_while_busy_the_input_and_the_rebuilding_settings_are_disabled():
    # Catches a message or a new chat started under a running turn, which Session would refuse with Busy.
    release = threading.Event()

    @tool
    def hold() -> str:
        """Wait until released."""
        release.wait(TIMEOUT)
        return "released"

    session, _ = make_session([response(tool_calls=[call("hold")]), response("done")], tools={"hold": hold})
    at = app(session)
    at.chat_input[0].set_value("go").run()
    eventually(lambda: session.state().agents["main"].running)
    at.run()
    no_errors(at)
    try:
        assert at.chat_input[0].proto.disabled
        assert at.selectbox(key="agent_kind").proto.disabled and at.toggle(key="sandbox").proto.disabled
        assert any(b.label == "Stop" for b in at.button)
    finally:
        release.set()
        idle(session)


def test_stop_ends_the_turn():
    # Catches a Stop button that doesn't reach the session.
    release = threading.Event()

    @tool
    def hold() -> str:
        """Wait until released."""
        release.wait(TIMEOUT)
        return "released"

    session, _ = make_session(
        [response(tool_calls=[call("hold"), call("hold")]), response("never")], tools={"hold": hold}
    )
    at = app(session)
    at.chat_input[0].set_value("go").run()
    eventually(lambda: session.state().agents["main"].running)
    at.run()
    next(b for b in at.button if b.label == "Stop").click().run()
    release.set()
    idle(session)
    assert session.state().last_job.result == "cancelled"


def test_compact_that_changes_nothing_says_so():
    # Catches a /compact that leaves no record getting no reply at all.
    session, _ = make_session([])
    at = app(session)
    at.chat_input[0].set_value("/compact").run()
    idle(session)
    toasts = [t.value for t in at.toast]
    at.run()
    toasts += [t.value for t in at.toast]
    assert toasts.count("[compaction nothing to compact]") == 1
