"""Tests for `Session` (session/): the one thing clients drive the agent through. It owns the job thread and the
approval handshake, and folds the agent tree's events into a state that subscribers see exactly once.

Each test names the bug it would catch.
"""

import copy
import queue
import threading

import pytest
from pydantic import ValidationError

from conftest import FakeClient, call, response, usage
from jean_code.core.agent import NOT_RUN, Agent
from jean_code.core.events import AgentAdded, Charged, Compaction, MessageAdded, TurnEnded
from jean_code.core.llm import AssistantMessage, UserMessage
from jean_code.core.node import AgentNode
from jean_code.core.subagents import agent_profile
from jean_code.core.tool_schema import tool
from jean_code.core.usage import ModelCall
from jean_code.session import (
    ApprovalRequested,
    ApprovalResolved,
    Busy,
    JobEnded,
    JobStarted,
    Reset,
    Session,
    SessionState,
    Settings,
    SettingsChanged,
    UnknownAgent,
    apply,
)

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
def rm(path: str) -> str:
    """Remove a file.

    Args:
        path: The file.
    """
    return f"removed {path}"


class Builder:
    """Builds each agent on a FakeClient with the given scripts, one script per build, and keeps what it built."""

    def __init__(self, *scripts, **agent_kwargs):
        self.scripts = list(scripts)
        self.agent_kwargs = agent_kwargs
        self.built: list[Agent] = []

    def __call__(self, settings: Settings, node: AgentNode, approve) -> Agent:
        agent = Agent(
            model=settings.model,
            client=FakeClient(self.scripts.pop(0) if self.scripts else []),
            tools={"add": add, "rm": rm},
            approve=approve,
            reasoning=settings.reasoning,
            compact_at=settings.compact_at,
            node=node,
            **self.agent_kwargs,
        )
        self.built.append(agent)
        return agent

    @property
    def agent(self) -> Agent:
        return self.built[-1]


def make_session(*scripts, settings: Settings | None = None, **agent_kwargs) -> tuple[Session, Builder]:
    build = Builder(*scripts, **agent_kwargs)
    return Session(settings or Settings(), build=build, log_path=None), build


def next_of(events, kind, timeout: float = TIMEOUT):
    """The next event of type `kind` from a subscription, skipping the others."""
    while True:
        event = events.get(timeout=timeout)
        if isinstance(event, kind):
            return event


def drain(events) -> list:
    out = []
    while True:
        try:
            out.append(events.get(timeout=0))
        except queue.Empty:
            return out


def eventually(predicate) -> None:
    for _ in range(TIMEOUT * 100):
        if predicate():
            return
        threading.Event().wait(0.01)
    raise AssertionError("timed out")


def blocking_tool(release: threading.Event):
    @tool
    def hold() -> str:
        """Wait until released."""
        release.wait(TIMEOUT)
        return "released"

    return hold


# ---------------------------------------------------------------- jobs


def held_builder(release: threading.Event, *script):
    hold = blocking_tool(release)

    def build(settings, node, approve):
        responses = list(script) or [response(tool_calls=[call("hold")]), response("done")]
        return Agent(client=FakeClient(responses), tools={"hold": hold}, node=node, approve=approve)

    return build


def test_a_turn_is_wrapped_in_job_events_and_its_messages_land_in_the_state():
    # Catches a turn whose start or end no client can see, or a state that misses the transcript.
    session, build = make_session([response("hi")])
    with session.subscribe() as (snapshot, events):
        assert list(snapshot.agents) == ["main"] and not snapshot.busy
        session.send("hello")
        started = events.get(timeout=TIMEOUT)
        ended = next_of(events, JobEnded)
    assert started == JobStarted(kind="turn", text="hello")
    assert ended == JobEnded(kind="turn", result="answered")
    main = session.state().agents["main"]
    assert [item.message for item in main.items] == list(build.agent.messages)
    assert main.end == "answered" and not main.running and not session.state().busy


def test_send_while_busy_raises_busy():
    # Catches a second turn started on an agent whose turn is still running on another thread.
    release = threading.Event()
    session = Session(Settings(), build=held_builder(release), log_path=None)
    with session.subscribe() as (_, events):
        session.send("go")
        with pytest.raises(Busy):
            session.send("again")
        with pytest.raises(Busy):
            session.compact()
        release.set()
        next_of(events, JobEnded)


def test_send_the_moment_the_job_ends_is_never_busy():
    # Catches busy cleared after JobEnded is published, so a client acting on JobEnded races the job thread.
    session, _ = make_session([response(f"reply {i}") for i in range(20)])
    with session.subscribe() as (_, events):
        for i in range(20):
            session.send(f"turn {i}")
            next_of(events, JobEnded)


def test_compact_is_a_job_and_reports_what_it_did():
    # Catches /compact with no start or end (no Stop shown while it runs) or no reply when it changed nothing.
    session, _ = make_session([])
    with session.subscribe() as (_, events):
        session.compact()
        assert events.get(timeout=TIMEOUT) == JobStarted(kind="compact")
        assert next_of(events, JobEnded) == JobEnded(kind="compact", result="nothing to compact")


def test_a_turn_that_raises_ends_the_job_with_the_error():
    # Catches a session left busy forever after the turn's thread died.
    session, _ = make_session([RuntimeError("HTTP 502")])
    with session.subscribe() as (_, events):
        session.send("go")
        ended = next_of(events, JobEnded)
    assert ended.result is None and ended.error == "RuntimeError: HTTP 502"
    assert not session.state().busy


# ---------------------------------------------------------------- approvals and stopping


def test_an_approval_round_trip_and_a_stale_answer_is_ignored():
    # Catches an answer meant for an earlier question approving the call that waits now.
    session, build = make_session([response(tool_calls=[call("rm", path="x")]), response("ok")], needs_approval={"rm"})
    with session.subscribe() as (_, events):
        session.send("delete x")
        approval = next_of(events, ApprovalRequested).approval
        assert (approval.path, approval.tool, approval.arguments) == ("main", "rm", {"path": "x"})
        assert session.state().approval == approval
        session.answer(approval.id + 1, None)  # stale: ignored
        assert session.state().approval == approval
        session.answer(approval.id, None)
        assert next_of(events, ApprovalResolved) == ApprovalResolved(id=approval.id, refusal=None)
        assert next_of(events, JobEnded).result == "answered"
    assert build.agent.messages[-2].content == "removed x"
    assert session.state().approval is None


def test_a_refusal_is_the_calls_result():
    # Catches the refusal text lost between the client and the model.
    session, build = make_session([response(tool_calls=[call("rm", path="x")]), response("ok")], needs_approval={"rm"})
    with session.subscribe() as (_, events):
        session.send("delete x")
        session.answer(next_of(events, ApprovalRequested).approval.id, "Denied by user. not that one")
        next_of(events, JobEnded)
    assert build.agent.messages[-2].content == "Denied by user. not that one"


def test_cancel_while_an_approval_waits_ends_the_job():
    # Catches a Stop that interrupts but leaves the turn waiting on an approval nobody will answer.
    session, build = make_session(
        [response(tool_calls=[call("rm", path="x")]), response("never")], needs_approval={"rm"}
    )
    with session.subscribe() as (_, events):
        session.send("delete x")
        next_of(events, ApprovalRequested)
        session.cancel()
        assert next_of(events, JobEnded, timeout=1).result == "interrupted"
    assert build.agent.messages[-1].content == NOT_RUN and session.state().approval is None


def test_cancel_a_subagent_path_stops_only_that_run():
    # Catches a cancel by path that misses the run, or stops its parent.
    release = threading.Event()
    held = held_builder(release)
    profile = agent_profile("blocks", lambda task, node: held(Settings(), node, None), max_cost=1.0)
    session, _ = make_session(
        [response(tool_calls=[call("spawn", profile="held", task="t")]), response("done")],
        subagent_profiles={"held": profile},
    )
    with session.subscribe() as (_, events):
        session.send("go")
        eventually(lambda: "main/1" in session.state().agents)
        session.cancel("main/1")
        release.set()
        assert next_of(events, JobEnded).result == "answered"
    assert session.state().agents["main/1"].end == "cancelled"
    assert session.state().agents["main"].end == "answered"


def test_any_node_in_the_tree_can_be_cancelled_and_inspected_by_path():
    # Catches agents built outside spawn (the engine, later) that clients see but can't stop or inspect.
    session, build = make_session([])
    group = build.agent.node.child(None, name="wf1")
    planner = Agent(client=FakeClient([]), node=group.child(None, name="plan", profile="planner"))
    assert session.state().agents["main/wf1/plan"].profile == "planner"
    assert session.inspect("main/wf1/plan").model == planner.model
    session.cancel("main/wf1")
    assert planner.node.cancelled and not build.agent.node.cancelled
    with pytest.raises(UnknownAgent):
        session.cancel("main/nope")
    with pytest.raises(UnknownAgent):
        session.inspect("main/wf1")  # a node with no agent


# ---------------------------------------------------------------- the state and subscriptions


def background_session():
    profile = agent_profile(
        "answers", lambda task, node: Agent(client=FakeClient([response(f"answer to {task}")]), node=node), 1.0
    )
    script = [
        response(tool_calls=[call("spawn", profile="echo", task=f"t{i}", background=True) for i in range(3)]),
        response("started them"),
    ]
    return make_session(script, subagent_profiles={"echo": profile})


@pytest.mark.parametrize("attempt", range(10))
def test_snapshot_plus_events_is_the_state(attempt):
    # Catches an event missed or seen twice between a subscriber's snapshot and its queue, with events arriving
    # from several threads (the turn's and three subagent runs') while it subscribes.
    session, build = background_session()
    session.send("go")
    with session.subscribe() as (snapshot, events):
        eventually(lambda: not session.state().busy and build.agent.subagents.running() == 0)
        replayed = copy.deepcopy(snapshot)
        for event in drain(events):
            apply(replayed, event)
    assert replayed == session.state()
    assert set(replayed.agents) == {"main", "main/1", "main/2", "main/3"}


def test_a_closed_subscription_receives_nothing_more():
    # Catches a client that's gone leaving a queue that collects every event until the process exits.
    session, _ = make_session([response("hi")])
    with session.subscribe() as (_, events):
        pass
    assert session._subscribers == set()
    session.send("go")
    eventually(lambda: not session.state().busy)
    assert drain(events) == []


def test_a_state_handed_out_does_not_change_afterwards():
    # Catches a snapshot that shares lists with the live state, so a client's copy changes under it.
    session, _ = make_session([response("hi")])
    before = session.state()
    items = list(before.agents["main"].items)
    session.send("go")
    eventually(lambda: not session.state().busy)
    assert before.agents["main"].items == items and not before.busy


def test_spend_counts_toward_every_ancestor_and_the_tree_log():
    # Catches a subagent's spend missing from its parent's total, which the header shows.
    profile = agent_profile(
        "answers",
        lambda task, node: Agent(client=FakeClient([response("a", usage=usage(cost=0.25))]), node=node),
        1.0,
    )
    session, build = make_session(
        [response(tool_calls=[call("spawn", profile="echo", task="t")], usage=usage(cost=0.5)), response("done")],
        subagent_profiles={"echo": profile},
    )
    with session.subscribe() as (_, events):
        session.send("go")
        next_of(events, JobEnded)
    state = session.state()
    assert state.agents["main/1"].spent == pytest.approx(0.25)
    assert state.agents["main"].spent == pytest.approx(build.agent.node.spent)
    assert [c.path for c in state.calls] == ["main", "main/1", "main"]


def test_pending_counts_background_results_waiting():
    # Catches the "results waiting" line never shown, since no event marks a run's future completing.
    session, _ = background_session()
    session.send("go")
    eventually(lambda: session.pending() == 3)


# ---------------------------------------------------------------- the reducer


def call_for(path: str, cost: float) -> ModelCall:
    return ModelCall("gen", None, "m", 0, cost, path)


def test_the_reducer_splices_a_compaction_and_keeps_the_replaced_items():
    # Catches compacted messages dropped from the timeline, or the summary shown after the turns that followed it.
    state = SessionState(settings=Settings())
    apply(state, AgentAdded(path="main", profile="", task="", model="m"))
    for text in ("t1", "t2", "t3", "t4"):
        apply(state, MessageAdded(path="main", message=UserMessage(content=text), context_tokens=10))
    summary = UserMessage(content="SUMMARY")
    apply(
        state,
        Compaction(
            path="main", status="compacted", start=1, removed=2, summary=summary, tokens_before=40, context_tokens=20
        ),
    )
    main = state.agents["main"]
    assert [(i.message.content, i.in_context) for i in main.items] == [
        ("t1", True),
        ("t2", False),
        ("t3", False),
        ("SUMMARY", True),
        ("t4", True),
    ]
    assert [main.item(uid).message.content for uid in main.context] == ["t1", "SUMMARY", "t4"]
    assert main.context_tokens == 20 and len(main.compactions) == 1


def test_the_reducer_drops_a_draft_once_its_reply_lands():
    # Catches streamed text shown twice: as the draft and as the reply it became.
    from jean_code.core.events import Delta, TurnStarted

    state = SessionState(settings=Settings())
    apply(state, AgentAdded(path="main", profile="", task="", model="m"))
    apply(state, TurnStarted(path="main", text="go"))
    apply(state, Delta(path="main", kind="reasoning", text="hm"))
    apply(state, Delta(path="main", kind="content", text="he"))
    apply(state, Delta(path="main", kind="content", text="llo"))
    assert state.agents["main"].draft == {"reasoning": "hm", "content": "hello"} and state.agents["main"].running
    apply(state, MessageAdded(path="main", message=AssistantMessage(content="hello"), context_tokens=5))
    assert state.agents["main"].draft == {}
    apply(state, Charged(path="main", call=call_for("main", 0.1)))
    apply(state, TurnEnded(path="main", end="answered", context_tokens=5))
    assert not state.agents["main"].running and state.agents["main"].end == "answered"


# ---------------------------------------------------------------- settings, reset, close


def test_live_settings_apply_to_the_running_agent():
    # Catches a reasoning change that needs a new chat, or one that the agent never reads.
    session, build = make_session([])
    with session.subscribe() as (_, events):
        session.configure({"reasoning": "high", "compact_at": 10_000, "max_cost": 2.0})
        changed = next_of(events, SettingsChanged)
    assert changed.settings.reasoning == "high"
    assert (build.agent.reasoning, build.agent.compact_at, build.agent.max_cost) == ("high", 10_000, 2.0)
    assert len(build.built) == 1


def test_a_change_of_agent_kind_rebuilds_the_agent():
    # Catches the old agent kept, or its events still reaching the new state.
    session, build = make_session([], [])
    old = build.agent
    with session.subscribe() as (_, events):
        session.configure({"kind": "orchestrator"})
        assert next_of(events, Reset).settings.kind == "orchestrator"
        assert next_of(events, AgentAdded).path == "main"
    assert len(build.built) == 2 and old.node.listener is None
    assert session.state().settings.kind == "orchestrator"


@pytest.mark.parametrize("patch", [{"compact_at": 10_000_000}, {"max_cost": 0}, {"reasoning": "extreme"}])
def test_invalid_settings_are_refused_with_nothing_applied(patch):
    # Catches half-applied settings, or a value Agent.__init__ would refuse accepted live.
    session, build = make_session([])
    before = session.state().settings
    with pytest.raises(ValidationError):
        session.configure(patch)
    assert session.state().settings == before and build.agent.compact_at == before.compact_at


def test_rebuilding_while_busy_raises_busy():
    # Catches a new chat started under a running turn, whose thread would keep writing to the old agent.
    release = threading.Event()
    session = Session(Settings(), build=held_builder(release), log_path=None)
    with session.subscribe() as (_, events):
        session.send("go")
        with pytest.raises(Busy):
            session.reset()
        with pytest.raises(Busy):
            session.configure({"kind": "orchestrator"})
        release.set()
        next_of(events, JobEnded)


def test_reset_starts_an_empty_state_with_a_new_main():
    # Catches the old transcript left in the state after a new chat.
    session, _ = make_session([response("hi")], [])
    with session.subscribe() as (_, events):
        session.send("go")
        next_of(events, JobEnded)
        session.reset()
        next_of(events, Reset)
        next_of(events, AgentAdded)
    assert session.state().agents["main"].items == [] and session.state().last_job is None


def test_close_stops_a_running_turn_and_its_runs():
    # Catches a process that can't exit because a run is still going.
    release = threading.Event()
    session = Session(Settings(), build=held_builder(release), log_path=None)
    with session.subscribe() as (_, events):
        session.send("go")
        eventually(lambda: session.state().agents["main"].running)
        release.set()
        session.close()
        assert next_of(events, JobEnded).result in ("answered", "interrupted")


def test_inspect_and_last_request():
    # Catches the inspector panels' reads failing on the root.
    session, build = make_session([response("hi")])
    assert session.last_request("main") is None
    session.send("go")
    eventually(lambda: not session.state().busy)
    details = session.inspect("main")
    assert details.tools == ["add", "rm"] and details.reasoning == build.agent.reasoning
    request = session.last_request("main")
    assert request is not None and request["messages"][-1] == {"role": "user", "content": "go"}


def test_an_approval_asked_after_a_stop_is_refused_at_once():
    # Catches a turn that passed its interrupt check just before the Stop, then waits forever on an approval
    # that _stop_main never saw.
    session, _ = make_session([])
    session.cancel()
    answers = []
    asking = threading.Thread(target=lambda: answers.append(session._approve("rm", {"path": "x"})), daemon=True)
    asking.start()
    asking.join(1)
    assert answers == [NOT_RUN] and session.state().approval is None
