"""Tests for `Driver`: one agent's jobs run off the caller's thread, its approvals, stopping, and the view it
keeps, whose state subscribers see exactly once.

Each test names the bug it would catch.
"""

import copy
import queue
import threading

import pytest

from conftest import FakeClient, call, response, usage
from jean_code.core.agent import NOT_RUN, Agent
from jean_code.core.driver import Busy, Driver, UnknownAgent
from jean_code.core.events import ApprovalRequested, ApprovalResolved, Configured, JobEnded, JobStarted, Queued
from jean_code.core.node import AgentNode
from jean_code.core.subagents import agent_profile
from jean_code.core.tool_schema import tool
from jean_code.core.view import AgentView, apply

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

    def __call__(self, node: AgentNode, approve) -> Agent:
        agent = Agent(
            client=FakeClient(self.scripts.pop(0) if self.scripts else []),
            tools={"add": add, "rm": rm},
            approve=approve,
            node=node,
            **self.agent_kwargs,
        )
        self.built.append(agent)
        return agent

    @property
    def agent(self) -> Agent:
        return self.built[-1]


def make_driver(*scripts, max_cost: float | None = None, **agent_kwargs) -> tuple[Driver, Builder]:
    build = Builder(*scripts, **agent_kwargs)
    return Driver(build, max_cost=max_cost), build


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

    def build(node, approve):
        responses = list(script) or [response(tool_calls=[call("hold")]), response("done")]
        return Agent(client=FakeClient(responses), tools={"hold": hold}, node=node, approve=approve)

    return build


def test_a_turn_is_wrapped_in_job_events_and_its_messages_land_in_the_state():
    # Catches a turn whose start or end no client can see, or a state that misses the transcript.
    driver, build = make_driver([response("hi")])
    with driver.view.subscribe() as (snapshot, events):
        assert list(snapshot.agents) == ["main"] and not snapshot.busy
        driver.send("hello")
        started = events.get(timeout=TIMEOUT)
        ended = next_of(events, JobEnded)
    assert started == JobStarted(kind="turn", text="hello")
    assert ended == JobEnded(kind="turn", result="answered")
    main = driver.view.state().agents["main"]
    assert [item.message for item in main.items] == list(build.agent.messages)
    assert main.end == "answered" and not main.running and not driver.view.state().busy


def gated(agent: Agent, gate: threading.Event) -> None:
    """Adds a `/gate` command that holds the turn before its first model call until `gate` is set."""

    def hold(rest: str) -> None:
        gate.wait(TIMEOUT)

    agent.commands["gate"] = hold


def test_send_while_busy_stops_the_job_and_runs_the_message_next():
    # Catches a message refused or lost while the agent works, or a queued turn hit by the stop that queued it.
    gate = threading.Event()
    driver, build = make_driver([response("instead, done")])
    gated(build.agent, gate)
    with driver.view.subscribe() as (_, events):
        driver.send("/gate go")
        driver.send("do this instead")
        assert driver.view.state().queued == "do this instead"
        gate.set()
        jobs = [next_of(events, (JobStarted, Queued, JobEnded)) for _ in range(5)]
    assert jobs == [
        JobStarted(kind="turn", text="/gate go"),
        Queued(text="do this instead"),
        JobEnded(kind="turn", result="cancelled"),
        JobStarted(kind="turn", text="do this instead"),
        JobEnded(kind="turn", result="answered"),
    ]
    assert len(build.agent.client.requests) == 1  # type: ignore[attr-defined]
    assert driver.view.state().queued is None and not driver.view.state().busy


def test_a_second_message_while_one_waits_is_busy_and_compact_always_is():
    # Catches a second redirect overwriting the first message, or a compaction started on a running agent.
    release = threading.Event()
    driver = Driver(held_builder(release))
    with driver.view.subscribe() as (_, events):
        driver.send("go")
        with pytest.raises(Busy):
            driver.compact()
        driver.send("first")
        with pytest.raises(Busy):
            driver.send("second")
        release.set()
        assert next_of(events, JobStarted, timeout=TIMEOUT) == JobStarted(kind="turn", text="go")
        assert next_of(events, JobStarted) == JobStarted(kind="turn", text="first")
        next_of(events, JobEnded)


def test_a_message_while_an_approval_waits_refuses_it_and_runs_next():
    # Catches a redirect that leaves the turn waiting on an approval nobody will answer.
    driver, build = make_driver(
        [response(tool_calls=[call("rm", path="x")]), response("ok, not deleting")], needs_approval={"rm"}
    )
    with driver.view.subscribe() as (_, events):
        driver.send("delete x")
        approval = next_of(events, ApprovalRequested).approval
        driver.send("no, keep it")
        assert next_of(events, ApprovalResolved) == ApprovalResolved(id=approval.id, refusal=NOT_RUN)
        assert next_of(events, JobEnded).result == "cancelled"
        assert next_of(events, JobEnded).result == "answered"
    assert build.agent.messages[-1].content == "ok, not deleting"


def test_close_drops_a_waiting_message():
    # Catches a queued turn started on an agent that's being closed.
    gate = threading.Event()
    driver, build = make_driver([response("never")])
    gated(build.agent, gate)
    with driver.view.subscribe() as (_, events):
        driver.send("/gate go")
        driver.send("queued")
        closing = threading.Thread(target=driver.close, daemon=True)
        closing.start()
        gate.set()
        closing.join(TIMEOUT)
        assert next_of(events, JobEnded).result == "cancelled"
        assert not [e for e in drain(events) if isinstance(e, JobStarted)]
    assert build.agent.client.requests == []  # type: ignore[attr-defined]


def test_send_the_moment_the_job_ends_is_never_busy():
    # Catches busy cleared after JobEnded is published, so a client acting on JobEnded races the job thread.
    driver, _ = make_driver([response(f"reply {i}") for i in range(20)])
    with driver.view.subscribe() as (_, events):
        for i in range(20):
            driver.send(f"turn {i}")
            next_of(events, JobEnded)


def test_compact_is_a_job_and_reports_what_it_did():
    # Catches /compact with no start or end (no Stop shown while it runs) or no reply when it changed nothing.
    driver, _ = make_driver([])
    with driver.view.subscribe() as (_, events):
        driver.compact()
        assert events.get(timeout=TIMEOUT) == JobStarted(kind="compact")
        assert next_of(events, JobEnded) == JobEnded(kind="compact", result="nothing to compact")


def test_a_turn_that_raises_ends_the_job_with_the_error():
    # Catches a driver left busy forever after the turn's thread died.
    driver, _ = make_driver([RuntimeError("HTTP 502")])
    with driver.view.subscribe() as (_, events):
        driver.send("go")
        ended = next_of(events, JobEnded)
    assert ended.result is None and ended.error == "RuntimeError: HTTP 502"
    assert not driver.view.state().busy


# ---------------------------------------------------------------- approvals and stopping


def test_an_approval_round_trip_and_a_stale_answer_is_ignored():
    # Catches an answer meant for an earlier question approving the call that waits now.
    driver, build = make_driver([response(tool_calls=[call("rm", path="x")]), response("ok")], needs_approval={"rm"})
    with driver.view.subscribe() as (_, events):
        driver.send("delete x")
        approval = next_of(events, ApprovalRequested).approval
        assert (approval.path, approval.tool, approval.arguments) == ("main", "rm", {"path": "x"})
        assert driver.view.state().approval == approval
        driver.answer(approval.id + 1, None)  # stale: ignored
        assert driver.view.state().approval == approval
        driver.answer(approval.id, None)
        assert next_of(events, ApprovalResolved) == ApprovalResolved(id=approval.id, refusal=None)
        assert next_of(events, JobEnded).result == "answered"
    assert build.agent.messages[-2].content == "removed x"
    assert driver.view.state().approval is None


def test_a_refusal_is_the_calls_result():
    # Catches the refusal text lost between the client and the model.
    driver, build = make_driver([response(tool_calls=[call("rm", path="x")]), response("ok")], needs_approval={"rm"})
    with driver.view.subscribe() as (_, events):
        driver.send("delete x")
        driver.answer(next_of(events, ApprovalRequested).approval.id, "Denied by user. not that one")
        next_of(events, JobEnded)
    assert build.agent.messages[-2].content == "Denied by user. not that one"


def test_cancel_while_an_approval_waits_ends_the_job():
    # Catches a Stop that interrupts but leaves the turn waiting on an approval nobody will answer.
    driver, build = make_driver([response(tool_calls=[call("rm", path="x")]), response("never")], needs_approval={"rm"})
    with driver.view.subscribe() as (_, events):
        driver.send("delete x")
        next_of(events, ApprovalRequested)
        driver.cancel()
        assert next_of(events, JobEnded, timeout=1).result == "cancelled"
    assert build.agent.messages[-1].content == NOT_RUN and driver.view.state().approval is None


def test_a_stop_right_after_send_holds_for_that_job():
    # Catches the job's turn clearing a stop that arrived before its thread got going.
    gate = threading.Event()
    driver, build = make_driver([response("never")])

    def hold(rest: str) -> None:  # holds the turn before its first model call
        gate.wait(TIMEOUT)

    build.agent.commands["gate"] = hold
    with driver.view.subscribe() as (_, events):
        driver.send("/gate go")
        driver.cancel()
        gate.set()
        assert next_of(events, JobEnded).result == "cancelled"
    assert build.agent.client.requests == []  # type: ignore[attr-defined]


def test_a_stop_between_jobs_does_not_reach_the_next_one():
    # Catches a Stop pressed while idle cancelling the user's next message.
    driver, _ = make_driver([response("hi")])
    driver.cancel()
    with driver.view.subscribe() as (_, events):
        driver.send("go")
        assert next_of(events, JobEnded).result == "answered"


def test_cancel_a_subagent_path_stops_only_that_run():
    # Catches a cancel by path that misses the run, or stops its parent.
    release = threading.Event()
    held = held_builder(release)
    profile = agent_profile("blocks", lambda task, node: held(node, None), max_cost=1.0)
    driver, _ = make_driver(
        [response(tool_calls=[call("spawn", profile="held", task="t")]), response("done")],
        subagent_profiles={"held": profile},
    )
    with driver.view.subscribe() as (_, events):
        driver.send("go")
        eventually(lambda: "main/1" in driver.view.state().agents)
        driver.cancel("main/1")
        release.set()
        assert next_of(events, JobEnded).result == "answered"
    assert driver.view.state().agents["main/1"].end == "cancelled"
    assert driver.view.state().agents["main"].end == "answered"


def test_any_node_in_the_tree_can_be_cancelled_and_inspected_by_path():
    # Catches agents built outside spawn (the engine, later) that clients see but can't stop or inspect.
    driver, build = make_driver([])
    group = build.agent.node.child(None, name="wf1")
    planner = Agent(client=FakeClient([]), node=group.child(None, name="plan", profile="planner"))
    assert driver.view.state().agents["main/wf1/plan"].profile == "planner"
    assert driver.agent_at("main/wf1/plan") is planner
    driver.cancel("main/wf1")
    assert planner.node.cancelled and not build.agent.node.cancelled
    with pytest.raises(UnknownAgent):
        driver.cancel("main/nope")
    with pytest.raises(UnknownAgent):
        driver.agent_at("main/wf1")  # a node with no agent


# ---------------------------------------------------------------- the state and subscriptions


def background_driver():
    profile = agent_profile(
        "answers", lambda task, node: Agent(client=FakeClient([response(f"answer to {task}")]), node=node), 1.0
    )
    script = [
        response(tool_calls=[call("spawn", profile="echo", task=f"t{i}", background=True) for i in range(3)]),
        response("started them"),
    ]
    return make_driver(script, subagent_profiles={"echo": profile})


@pytest.mark.parametrize("attempt", range(10))
def test_snapshot_plus_events_is_the_state(attempt):
    # Catches an event missed or seen twice between a subscriber's snapshot and its queue, with events arriving
    # from several threads (the turn's and three subagent runs') while it subscribes.
    driver, build = background_driver()
    driver.send("go")
    with driver.view.subscribe() as (snapshot, events):
        eventually(lambda: not driver.view.state().busy and build.agent.subagents.running() == 0)
        replayed = copy.deepcopy(snapshot)
        for event in drain(events):
            apply(replayed, event)
    assert replayed == driver.view.state()
    assert set(replayed.agents) == {"main", "main/1", "main/2", "main/3"}


def test_a_closed_subscription_receives_nothing_more():
    # Catches a client that's gone leaving a queue that collects every event until the process exits.
    driver, _ = make_driver([response("hi")])
    with driver.view.subscribe() as (_, events):
        pass
    assert driver.view._subscribers == set()
    driver.send("go")
    eventually(lambda: not driver.view.state().busy)
    assert drain(events) == []


def test_a_state_handed_out_does_not_change_afterwards():
    # Catches a snapshot that shares lists with the live state, so a client's copy changes under it.
    driver, _ = make_driver([response("hi")])
    before = driver.view.state()
    items = list(before.agents["main"].items)
    driver.send("go")
    eventually(lambda: not driver.view.state().busy)
    assert before.agents["main"].items == items and not before.busy


def test_spend_counts_toward_every_ancestor_and_the_tree_log():
    # Catches a subagent's spend missing from its parent's total, which the header shows.
    profile = agent_profile(
        "answers",
        lambda task, node: Agent(client=FakeClient([response("a", usage=usage(cost=0.25))]), node=node),
        1.0,
    )
    driver, build = make_driver(
        [response(tool_calls=[call("spawn", profile="echo", task="t")], usage=usage(cost=0.5)), response("done")],
        subagent_profiles={"echo": profile},
    )
    with driver.view.subscribe() as (_, events):
        driver.send("go")
        next_of(events, JobEnded)
    state = driver.view.state()
    assert state.agents["main/1"].spent == pytest.approx(0.25)
    assert state.agents["main"].spent == pytest.approx(build.agent.node.spent)
    assert [c.path for c in state.calls] == ["main", "main/1", "main"]


def test_pending_counts_background_results_waiting():
    # Catches the "results waiting" line never shown, since no event marks a run's future completing.
    driver, build = background_driver()
    driver.send("go")
    eventually(lambda: build.agent.subagents is not None and build.agent.subagents.pending == 3)


# ---------------------------------------------------------------- settings, reset, close


def test_close_stops_a_running_turn_and_its_runs():
    # Catches a process that can't exit because a run is still going.
    release = threading.Event()
    driver = Driver(held_builder(release))
    with driver.view.subscribe() as (_, events):
        driver.send("go")
        eventually(lambda: driver.view.state().agents["main"].running)
        release.set()
        driver.close()
        assert next_of(events, JobEnded).result in ("answered", "cancelled")


def test_an_approval_asked_after_a_stop_is_refused_at_once():
    # Catches a turn that passed its interrupt check just before the Stop, then waits forever on an approval
    # that _stop_main never saw.
    driver, _ = make_driver([])
    driver.cancel()
    answers = []
    asking = threading.Thread(target=lambda: answers.append(driver._approve("rm", {"path": "x"})), daemon=True)
    asking.start()
    asking.join(1)
    assert answers == [NOT_RUN] and driver.view.state().approval is None


def test_live_settings_apply_to_the_agent_and_reach_the_view():
    # Catches a reasoning change that needs a new chat, one the agent never reads, or a view left on the old model.
    driver, build = make_driver([])
    driver.configure(model="other", reasoning="high", compact_at=10_000, max_cost=2.0)
    settings = build.agent.settings
    assert (settings.model, settings.reasoning, settings.compact_at, build.agent.max_cost) == (
        "other",
        "high",
        10_000,
        2.0,
    )
    main = driver.view.state().agents["main"]
    assert (main.settings, main.limit) == (settings, 2.0)


def test_configure_emits_one_configured_with_every_setting():
    # Catches a change the view never hears about, or one announced as a partial update.
    driver, build = make_driver([])
    with driver.view.subscribe() as (_, events):
        driver.configure(reasoning="high")
        assert events.get(timeout=TIMEOUT) == Configured(path="main", settings=build.agent.settings)
        with pytest.raises(queue.Empty):
            events.get(timeout=0)


@pytest.mark.parametrize(
    "changes",
    [{"compact_at": 10_000_000}, {"max_cost": 0}, {"reasoning": "extreme"}, {"x": 1}, {"keep_first": 0}],
)
def test_invalid_settings_are_refused_with_nothing_applied(changes):
    # Catches half-applied settings, a value Agent.__init__ would refuse accepted live, or a setting fixed at
    # construction changed mid-run.
    driver, build = make_driver([])
    before = (build.agent.settings, build.agent.max_cost)
    with driver.view.subscribe() as (_, events):
        with pytest.raises(ValueError):
            driver.configure(**({"reasoning": "low"} | changes))  # the good change must not apply either
        with pytest.raises(queue.Empty):
            events.get(timeout=0)
    assert (build.agent.settings, build.agent.max_cost) == before


def test_the_root_carries_the_budget():
    # Catches a driver whose max_cost never reaches the root node, or the view.
    driver = Driver(Builder([response("hi")]), max_cost=0.5)
    assert driver.agent.max_cost == driver.view.state().agents["main"].limit == 0.5


def test_the_view_given_is_the_one_that_listens():
    # Catches a driver that makes its own view, so a recorder subscribed beforehand gets nothing.
    view = AgentView()
    driver = Driver(Builder([]), view=view)
    assert driver.view is view
    assert list(view.state().agents) == ["main"]


def test_close_stops_listening_to_the_tree():
    # Catches a run that winds down after close still writing into a view nobody reads any more.
    driver, build = make_driver([])
    driver.close()
    assert build.agent.node.listener is None
