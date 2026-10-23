"""Tests for the Streamlit app (ui/app.py), run with Streamlit's AppTest on a `Driver` whose agent is scripted.
The driver is put in `st.session_state` before the first run, so the app draws it instead of building its own.

Each test names the bug it would catch.
"""

import threading
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import jean_code
import jean_code.config
import jean_code.logs
from conftest import FakeClient, call, response, usage
from jean_code.core.agent import Agent
from jean_code.core.driver import Driver
from jean_code.core.events import Delta, MessageAdded
from jean_code.core.llm import AssistantMessage
from jean_code.core.node import AgentNode
from jean_code.core.subagents import agent_profile
from jean_code.core.tool_schema import tool
from jean_code.logs import ProcessLogs

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


def make_session(responses, tools=None, **agent_kwargs) -> tuple[Driver, list[Agent]]:
    built: list[Agent] = []

    def build(node: AgentNode, approve) -> Agent:
        agent = Agent(
            client=FakeClient(responses), tools=tools or {"add": add}, node=node, approve=approve, **agent_kwargs
        )
        built.append(agent)
        return agent

    return Driver(build), built


def app(session: Driver) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=10)
    at.session_state["driver"] = session
    return at.run()


def eventually(predicate) -> None:
    for _ in range(TIMEOUT * 100):
        if predicate():
            return
        threading.Event().wait(0.01)
    raise AssertionError("timed out")


def idle(session: Driver) -> None:
    eventually(lambda: not session.view.state().busy)


def htmls(at: AppTest) -> list[str]:
    return [e.proto.body for e in at.get("html")]


def markdowns(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown]


def block_keys(at: AppTest) -> list[str]:
    """Returns the key of every keyed container in the main area."""
    keys, todo = [], [at.main]
    while todo:
        node = todo.pop()
        todo.extend(getattr(node, "children", {}).values())
        block_id = getattr(getattr(node, "proto", None), "id", "")
        if block_id.startswith("$$ID-"):
            keys.append(block_id.split("-", 2)[2])
    return keys


def no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def test_a_turn_with_a_tool_call_shows_the_call_with_its_result_and_opens_to_them():
    # Catches a chat that misses messages, or a call whose result its row can't find.
    session, _ = make_session([response(tool_calls=[call("add", a=1, b=2)]), response("three")])
    at = app(session)
    at.chat_input[0].set_value("add 1 and 2").run()
    idle(session)
    at.run()
    no_errors(at)
    assert {"add 1 and 2", "three"} <= set(markdowns(at))
    assert [b.key for b in at.button if b.key and b.key.startswith("hit-")] == ["hit-c2-0"]  # one row: call and result
    assert any("add" in h and "✓ 3" in h for h in htmls(at))
    at.button(key="hit-c2-0").click().run()
    no_errors(at)
    assert "3" in [c.value for c in at.code]


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
    eventually(lambda: session.view.state().approval is not None)
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


def test_the_inspector_draws_the_next_request_and_the_settings_from_the_view():
    # Catches an inspector that crashes without the live agent, or shows a setting the view never received.
    session, _ = make_session([response("hi")], needs_approval={"add"})
    session.send("hello")
    idle(session)
    session.configure(reasoning="high")
    session.set_limit("main", 3.0)
    at = app(session)
    at.toggle(key="inspector").set_value(True).run()
    no_errors(at)
    assert [t.label for t in at.tabs] == ["Next request", "Context", "Agent state"]
    page = "".join(htmls(at))
    assert "high" in page and "3.0" in page and ">add<" in page
    assert any(j.value and '"hello"' in j.value for j in at.json)  # the transcript is the payload


def test_compacted_messages_are_dimmed_under_a_divider():
    # Catches compacted messages dropped from the timeline, or shown as if still in context.
    session, built = make_session([], system_prompt="sys", compact_at=1000, keep_first=2, keep_last=2)
    client = built[-1].client
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
    assert sum("-out-" in k for k in block_keys(at)) == 2  # turn 3's message and reply
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
    assert "answer to t" in markdowns(at)


def held_session(release: threading.Event, *script) -> Driver:
    """A driver whose first turn holds in a tool call until `release` is set."""

    @tool
    def hold() -> str:
        """Wait until released."""
        release.wait(TIMEOUT)
        return "released"

    script = script or (response(tool_calls=[call("hold")]), response("done"))
    session, _ = make_session(list(script), tools={"hold": hold})
    return session


def test_while_busy_the_rebuilding_settings_are_disabled_but_the_input_is_not():
    # Catches a new chat started under a running turn, or no way to redirect the agent while it works.
    release = threading.Event()
    session = held_session(release)
    at = app(session)
    at.chat_input[0].set_value("go").run()
    eventually(lambda: session.view.state().agents["main"].running)
    at.run()
    no_errors(at)
    try:
        assert not at.chat_input[0].proto.disabled
        assert at.chat_input[0].proto.placeholder == "Send to stop and redirect…"
        assert at.selectbox(key="agent_kind").proto.disabled and at.toggle(key="sandbox").proto.disabled
        assert any(b.label == "Stop" for b in at.button)
    finally:
        release.set()
        idle(session)


def test_a_message_while_busy_waits_on_a_card_then_runs():
    # Catches a redirect that never shows, a second message accepted over the first, or the queued turn not run.
    release = threading.Event()
    session = held_session(release, response(tool_calls=[call("hold")]), response("redirected"))
    at = app(session)
    at.chat_input[0].set_value("go").run()
    eventually(lambda: session.view.state().agents["main"].running)
    at.run()
    at.chat_input[0].set_value("do this instead").run()
    no_errors(at)
    try:
        assert "*stopping… then:* do this instead" in markdowns(at)
        assert at.chat_input[0].proto.disabled and at.chat_input[0].proto.placeholder == "Stopping…"
    finally:
        release.set()
    idle(session)
    at.run()  # the poll that sees the stopped job has ended sends the message
    eventually(lambda: (job := session.view.state().last_job) is not None and job.result == "answered")
    at.run()
    assert not any("stopping…" in m for m in markdowns(at))
    assert session.view.state().agents["main"].items[-1].message.content == "redirected"


def test_a_reply_streaming_in_shows_at_the_end_of_the_chat():
    # Catches streamed text that never reaches the screen until the reply lands.
    release = threading.Event()
    session = held_session(release)
    at = app(session)
    at.chat_input[0].set_value("go").run()
    eventually(lambda: session.view.state().agents["main"].running)
    session.view.listen(Delta(path="main", kind="reasoning", text="weighing it"))
    session.view.listen(Delta(path="main", kind="content", text="half an answer"))
    at.run()
    no_errors(at)
    try:
        assert any("stream" in h and "weighing it" in h for h in htmls(at))
        assert "half an answer" in markdowns(at)
    finally:
        release.set()
        idle(session)


def test_a_thought_streaming_in_opens_in_full_and_stays_open_when_it_lands():
    # Catches a live thought clipped to its tail when opened, or closing on its own when the reply lands.
    release = threading.Event()
    session = held_session(release)
    at = app(session)
    at.chat_input[0].set_value("go").run()
    eventually(lambda: session.view.state().agents["main"].running)
    thought = "first idea. " + "x" * 500 + " last idea."
    session.view.listen(Delta(path="main", kind="reasoning", text=thought))
    at.run()
    try:
        hit = next(b for b in at.button if b.key and b.key.startswith("hit-r"))
        assert not any("first idea" in h for h in htmls(at))  # closed, it shows the latest words only
        hit.click().run()
        no_errors(at)
        assert thought in markdowns(at)
        uids = [i.uid for i in session.view.state().agents["main"].items]
        session.view.listen(
            MessageAdded(path="main", message=AssistantMessage(content="ok", reasoning=thought), context_tokens=1)
        )
        at.run()
        assert session.view.state().agents["main"].items[-1].uid == max(uids) + 1
        assert thought in markdowns(at)
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
    eventually(lambda: session.view.state().agents["main"].running)
    at.run()
    next(b for b in at.button if b.label == "Stop").click().run()
    release.set()
    idle(session)
    assert session.view.state().last_job.result == "cancelled"


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


@pytest.mark.parametrize("kind, tools", [("Chat", set()), ("Web search", {"web_search", "fetch_page"})])
def test_the_agent_select_builds_the_agent_it_names(kind, tools, monkeypatch, tmp_path):
    # Catches a kind the select offers that the build doesn't know, which would silently build the coding agent.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # builds a client; nothing here sends a request
    monkeypatch.setattr(jean_code.logs, "process_logs", ProcessLogs(tmp_path))  # the new chat's trace
    monkeypatch.setattr(jean_code.config, "USAGE_LOG", tmp_path / "usage.jsonl")
    session, _ = make_session([])
    at = app(session)
    at.session_state["built"] = ("Coding", False)  # what the app takes the scripted session for
    at.selectbox(key="agent_kind").set_value(kind).run()
    driver = at.session_state["driver"]
    try:
        no_errors(at)
        assert {t["function"]["name"] for t in driver.view.state().agents["main"].tools} == tools
        assert not [t for t in at.toggle if t.key == "sandbox"]  # no shell, so no sandbox toggle
    finally:
        driver.close()
        at.session_state["recorder"].close()
