"""Tests for the subagents' tool boundary (`Subagents`: `spawn`, `subagent`, delivery, waits, limits) and how
`Agent` wires it in.

Each test names the bug it would catch.
"""

import threading
import time

import pytest

from conftest import FakeClient, call, response, usage
from jean_code.core.agent import Agent
from jean_code.core.events import Delivered, LimitSet, RunEnded
from jean_code.core.node import AgentNode
from jean_code.core.subagents import (
    FOLLOW_UP,
    MAX_RUNNING,
    MAX_WAIT,
    Profile,
    Subagents,
    agent_profile,
    agent_tree,
)
from jean_code.core.tool_schema import tool
from jean_code.core.view import AgentView

TIMEOUT = 5  # seconds any wait in these tests may take before it counts as a hang


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def echo_profile(max_cost=1.0, max_cost_cap=None):
    """Answers "answer to <task>" in one call."""

    def build(task: str, node: AgentNode) -> Agent:
        return Agent(client=FakeClient([response(f"answer to {task}")]), node=node)

    return agent_profile("answers at once", build, max_cost, max_cost_cap)


def held_profile(release: threading.Event):
    """Calls `hold`, which blocks until `release` is set, then answers "did <task>"."""

    @tool
    def hold() -> str:
        """Wait until released."""
        release.wait(TIMEOUT)
        return "released"

    def build(task: str, node: AgentNode) -> Agent:
        client = FakeClient([response(tool_calls=[call("hold")]), response(f"did {task}")])
        return Agent(client=client, tools={"hold": hold}, node=node)

    return agent_profile("blocks until released", build, max_cost=1.0)


def scripted_profile(responses):
    def build(task: str, node: AgentNode) -> Agent:
        return Agent(client=FakeClient(list(responses)), tools={"add": add}, node=node)

    return agent_profile("scripted", build, max_cost=1.0)


def make_subagents(node=None, **profiles) -> Subagents:
    return Subagents(profiles or {"echo": echo_profile()}, node or AgentNode())


def run_of(subagents: Subagents, run_id: int):
    return subagents.runs[run_id][1]


def eventually(predicate) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def tool_results(agent: Agent) -> list[str]:
    return [m.content for m in agent.messages if m.role == "tool"]


# ---------------------------------------------------------------- spawn and subagent


def test_foreground_spawn_returns_the_result_block():
    assert make_subagents().spawn("echo", "x") == (
        '<subagent-result run="1" profile="echo" status="done">\nanswer to x\n</subagent-result>'
    )


def test_a_stopped_run_is_labelled_with_its_reason():
    # Catches the model receiving a mid-task remark as status="done".
    sub = make_subagents(
        tiny=scripted_profile([response("let me check", tool_calls=[call("add", a=1, b=2)], usage=usage(cost=1.0))])
    )
    assert sub.spawn("tiny", "go") == (
        '<subagent-result run="1" profile="tiny" status="stopped" reason="budget spent">\nlet me check\n'
        "</subagent-result>"
    )


def test_a_failed_run_is_labelled_with_its_error():
    sub = make_subagents(broken=scripted_profile([RuntimeError("HTTP 500")]))
    assert 'status="failed">\nRuntimeError: HTTP 500\n' in sub.spawn("broken", "go")


def test_background_spawn_returns_at_once_and_subagent_collects():
    release = threading.Event()
    sub = make_subagents(held=held_profile(release))
    assert sub.spawn("held", "x", background=True) == "started run 1 with $1.00"
    assert sub.subagent(1).startswith("run 1 is still running: ")
    release.set()
    assert 'status="done">\ndid x\n' in sub.subagent(1, wait=TIMEOUT)
    assert 'status="done">\ndid x\n' in sub.subagent(1)  # asking again returns it again


def test_ids_count_up_and_a_failed_start_skips_its_id():
    # Catches reused ids, or a half-registered run after its start failed.
    def broken_start(task, node):
        raise RuntimeError("no model")

    sub = make_subagents(echo=echo_profile(), broken=Profile("broken", broken_start, max_cost=1.0))
    sub.spawn("echo", "a")
    with pytest.raises(RuntimeError, match="no model"):
        sub.spawn("broken", "b")
    sub.spawn("echo", "c")
    assert list(sub.runs) == [1, 3]


def test_unknown_names_list_the_valid_ones():
    sub = make_subagents()
    with pytest.raises(ValueError, match="unknown profile 'nope'; profiles: echo"):
        sub.spawn("nope", "x")
    sub.spawn("echo", "x")
    with pytest.raises(ValueError, match="unknown run 7; runs: 1"):
        sub.subagent(7)


def test_a_profile_that_ignores_its_node_is_refused():
    # Catches a run whose spend escapes the tree: no limit, missing from `spent`, `breakdown()` and the log.
    rogue = agent_profile(
        "rogue", lambda task, node: Agent(client=FakeClient([response("hi")]), node=AgentNode()), max_cost=1.0
    )
    sub = make_subagents(rogue=rogue)
    with pytest.raises(ValueError, match="didn't build its agent on the node it was given"):
        sub.spawn("rogue", "x")
    assert sub.runs == {}


# ---------------------------------------------------------------- limits


def test_max_cost_defaults_is_clamped_and_states_what_applies():
    # Catches a limit above the cap, or the model told it has more than its parent has left.
    sub = make_subagents(AgentNode(0.3), echo=echo_profile(max_cost=0.5, max_cost_cap=1.0))
    assert sub.spawn("echo", "a", background=True) == "started run 1 with $0.30"  # the parent's 0.3 applies
    sub.spawn("echo", "b", background=True, max_cost=5.0)
    assert [run_of(sub, i).agent.node.limit for i in (1, 2)] == [0.5, 1.0]
    with pytest.raises(ValueError, match="positive"):
        sub.spawn("echo", "c", max_cost=0)


def test_at_most_max_running_runs_at_once():
    # Catches an unbounded fan-out, whose overshoot past a limit is unbounded too.
    release = threading.Event()
    sub = make_subagents(held=held_profile(release))
    for i in range(MAX_RUNNING):
        sub.spawn("held", str(i), background=True)
    with pytest.raises(ValueError, match=f"{MAX_RUNNING} subagents are running"):
        sub.spawn("held", "one too many", background=True)
    release.set()
    eventually(lambda: sub.running() == 0)
    assert sub.spawn("held", "now there's room", background=True).startswith("started run")


def test_waits_are_capped(monkeypatch):
    sub = make_subagents()
    sub.spawn("echo", "x", background=True)
    waits = []
    monkeypatch.setattr(sub, "_wait", lambda run, timeout: waits.append(timeout) or True)
    sub.subagent(1, wait=5000)
    sub.subagent(1, wait=-3)
    assert waits == [MAX_WAIT, 0]


# ---------------------------------------------------------------- delivery


def test_a_background_result_is_delivered_once_with_the_next_message():
    # Catches a result injected twice, or never.
    parent = Agent(
        client=FakeClient(
            [
                response(tool_calls=[call("spawn", profile="echo", task="x", background=True)]),
                response("started it"),
                response("got it"),
                response("fine"),
            ]
        ),
        subagent_profiles={"echo": echo_profile()},
    )
    parent.run_turn("start something")
    eventually(lambda: parent.subagents.pending == 1)
    parent.run_turn("any news?")
    assert parent.messages[-2].content == (
        'any news?\n\n<subagent-result run="1" profile="echo" status="done">\nanswer to x\n</subagent-result>'
    )
    assert parent.subagents.pending == 0
    parent.run_turn("and now?")
    assert parent.messages[-2].content == "and now?"


def test_a_result_read_with_subagent_is_not_injected_again():
    sub = make_subagents()
    sub.spawn("echo", "x", background=True)
    assert "answer to x" in sub.subagent(1, wait=TIMEOUT)
    assert sub.pending == 0 and sub.claim() == []


def outcomes(events: list) -> list:
    return [e for e in events if isinstance(e, RunEnded | Delivered)]


def test_a_foreground_result_is_delivered_before_spawn_returns():
    # Catches a foreground result the view counts as waiting for the next message forever.
    events = []
    sub = make_subagents(AgentNode(listener=events.append))
    sub.spawn("echo", "x")
    assert outcomes(events) == [RunEnded(path="main/1", status="done"), Delivered(path="main/1")]


def test_a_background_result_ends_then_is_delivered_when_claimed():
    # Catches a finished background run the view never counts, or one it counts after its result was taken.
    view = AgentView()
    sub = make_subagents(AgentNode(listener=view.listen), held=held_profile(release := threading.Event()))
    sub.spawn("held", "x", background=True)
    assert view.state().waiting("main") == sub.pending == 0  # still running
    release.set()
    eventually(lambda: sub.pending == 1)
    assert view.state().waiting("main") == 1
    sub.claim()
    assert view.state().waiting("main") == sub.pending == 0


def test_the_view_agrees_with_pending_after_each_tool_call_and_turn():
    # Catches the view's count and Subagents.pending drifting apart at the points a client reads it.
    view = AgentView()
    parent = Agent(
        client=FakeClient(
            [
                response(tool_calls=[call("spawn", profile="echo", task="bg", background=True)]),
                response(tool_calls=[call("spawn", profile="echo", task="fg")]),
                response("both started"),
                response("got it"),
            ]
        ),
        subagent_profiles={"echo": echo_profile()},
        node=AgentNode(listener=view.listen),
    )
    parent.run_turn("start two")
    eventually(lambda: parent.subagents.pending == 1)
    assert view.state().waiting("main") == parent.subagents.pending == 1
    parent.run_turn("any news?")
    assert view.state().waiting("main") == parent.subagents.pending == 0


def test_running_and_foreground_runs_are_never_pending():
    release = threading.Event()
    sub = make_subagents(echo=echo_profile(), held=held_profile(release))
    sub.spawn("echo", "foreground")
    sub.spawn("held", "background", background=True)
    assert sub.pending == 0 and sub.claim() == []
    release.set()
    eventually(lambda: sub.pending == 1)
    assert [id for id, _, _ in sub.claim()] == [2]


def test_fan_out_runs_in_parallel():
    # Catches background spawns that run one after another: each waits for the other two at the barrier.
    barrier = threading.Barrier(3, timeout=TIMEOUT)

    @tool
    def meet() -> str:
        """Wait for the others."""
        barrier.wait()
        return "met"

    def build(task: str, node: AgentNode) -> Agent:
        return Agent(
            client=FakeClient([response(tool_calls=[call("meet")]), response(f"{task} done")]),
            tools={"meet": meet},
            node=node,
        )

    parent = Agent(
        client=FakeClient(
            [
                response(tool_calls=[call("spawn", profile="p", task=t, background=True) for t in "abc"]),
                response(tool_calls=[call("subagent", run_id=i, wait=TIMEOUT) for i in (1, 2, 3)]),
                response("all done"),
            ]
        ),
        subagent_profiles={"p": agent_profile("meets", build, max_cost=1.0)},
    )
    parent.run_turn("fan out")
    assert [r.split(">\n")[1].split("\n")[0] for r in tool_results(parent)[3:]] == ["a done", "b done", "c done"]
    # The runs answer even when the barrier breaks (the tool error goes back to them as text), so the proof is
    # inside: every `meet` returned. Run one after another, each gets "Error: BrokenBarrierError"
    # (.experiments/output/subagents_fanout_control.txt).
    assert [tool_results(run.agent) for _, _, run in parent.subagents.snapshot()] == [["met"]] * 3


# ---------------------------------------------------------------- through the agent


def test_the_tools_take_loose_json_and_answer_errors_as_text():
    # Catches a string id rejected, or a bad call crashing the turn instead of informing the model.
    parent = Agent(
        client=FakeClient(
            [
                response(tool_calls=[call("spawn", profile="echo", task="x", background=True)]),
                response(
                    tool_calls=[
                        call("subagent", run_id="1", wait=TIMEOUT),
                        call("subagent", run_id=7),
                        call("spawn", profile="nope", task="y"),
                    ]
                ),
                response("done"),
            ]
        ),
        subagent_profiles={"echo": echo_profile()},
    )
    parent.run_turn("go")
    started, by_string, unknown_id, unknown_profile = tool_results(parent)
    assert started == "started run 1 with $1.00"
    assert 'status="done"' in by_string
    assert unknown_id == "Error: ValueError: unknown run 7; runs: 1"
    assert unknown_profile == "Error: ValueError: unknown profile 'nope'; profiles: echo"


def test_the_listing_and_tools_come_only_with_profiles():
    # Catches management methods exposed as tools, or tools and a listing for an agent without profiles.
    agent = Agent(client=FakeClient([]), system_prompt="Be brief.", subagent_profiles={"echo": echo_profile()})
    assert agent.messages[0].content.startswith("Be brief.\n\n## Subagents\n")
    assert "- echo: answers at once" in agent.messages[0].content
    assert set(agent._tools) == {"spawn", "subagent", "resume"}
    plain = Agent(client=FakeClient([]))
    assert plain.subagents is None and plain._tools == {} and plain.messages == ()


def test_agent_tree_follows_the_budget_paths():
    # Catches the tree and the nodes' breakdown disagreeing on a run's path.
    def lead(task: str, node: AgentNode) -> Agent:
        return Agent(
            client=FakeClient([response(tool_calls=[call("spawn", profile="echo", task="dig")]), response("led")]),
            node=node,
            subagent_profiles={"echo": echo_profile()},
        )

    parent = Agent(
        client=FakeClient([response(tool_calls=[call("spawn", profile="lead", task="lead it")]), response("done")]),
        subagent_profiles={"lead": agent_profile("leads", lead, max_cost=1.0)},
    )
    parent.run_turn("go")
    assert list(agent_tree(parent)) == ["main", "main/1", "main/1/1"]
    assert [row["path"] for row in parent.node.breakdown()] == ["main", "main/1", "main/1/1"]


# ---------------------------------------------------------------- stops and close


def test_a_stop_ends_a_wait_and_stops_a_background_run():
    # Catches Ctrl+C doing nothing while the parent waits up to 10 minutes on subagent(id, wait=600), and a
    # background run that keeps working and spending after the user said stop.
    release = threading.Event()
    sub = make_subagents(held=held_profile(release))
    sub.spawn("held", "x", background=True)
    answers = []
    waiter = threading.Thread(target=lambda: answers.append(sub.subagent(1, wait=MAX_WAIT)))
    waiter.start()
    time.sleep(0.1)
    sub.node.cancel()
    waiter.join(1)
    assert not waiter.is_alive() and answers[0].startswith("run 1 is still running")
    assert run_of(sub, 1).agent.node.cancelled
    release.set()  # the call in flight ends; the run makes no further call
    assert run_of(sub, 1).outcome(TIMEOUT).reason == "cancelled"


def test_a_stop_stops_a_foreground_run():
    release = threading.Event()
    sub = make_subagents(held=held_profile(release))
    answers = []
    spawner = threading.Thread(target=lambda: answers.append(sub.spawn("held", "x")))
    spawner.start()
    eventually(lambda: 1 in sub.runs)
    sub.node.cancel()
    eventually(lambda: run_of(sub, 1).agent.node.cancelled)
    release.set()  # the call in flight (the hold) ends; the run makes no further call
    spawner.join(TIMEOUT)
    assert answers == [
        '<subagent-result run="1" profile="held" status="stopped" reason="cancelled">\n(no reply)\n</subagent-result>'
    ]


def test_close_stops_every_run_and_waits_for_them():
    # Catches a run outliving its parent.
    release = threading.Event()
    parent = Agent(client=FakeClient([]), subagent_profiles={"held": held_profile(release)})
    for task in ("a", "b"):
        parent.subagents.spawn("held", task, background=True)
    closer = threading.Thread(target=parent.close)
    closer.start()
    time.sleep(0.1)
    assert closer.is_alive()  # waiting for the calls in flight
    release.set()
    closer.join(TIMEOUT)
    assert not closer.is_alive() and parent.subagents.running() == 0
    assert all(run.agent.node.cancelled for _, _, run in parent.subagents.snapshot())


def test_a_tool_named_like_spawn_is_refused():
    # Catches a passed tool silently replaced by the agent's own `spawn`, so the model never reaches it.
    @tool
    def spawn(task: str) -> str:
        """Start something."""
        return task

    with pytest.raises(ValueError, match="named like the agent's own: spawn"):
        Agent(client=FakeClient([]), tools={"spawn": spawn}, subagent_profiles={"echo": echo_profile()})


def test_a_tool_named_like_resume_is_refused():
    # Catches a passed tool silently replaced by the agent's own `resume`.
    @tool
    def resume(run_id: int) -> str:
        """Resume something."""
        return str(run_id)

    with pytest.raises(ValueError, match="named like the agent's own: resume"):
        Agent(client=FakeClient([]), tools={"resume": resume}, subagent_profiles={"echo": echo_profile()})


# ---------------------------------------------------------------- resume


def turns_profile(*replies, max_cost=1.0, max_cost_cap=None, tools=None):
    """Each agent gets the same scripted replies, one or more per turn."""

    def build(task: str, node: AgentNode) -> Agent:
        return Agent(client=FakeClient(list(replies)), tools=tools or {"add": add}, node=node)

    return agent_profile("scripted turns", build, max_cost, max_cost_cap)


def blocking_tool(release: threading.Event):
    @tool
    def hold() -> str:
        """Wait until released."""
        release.wait(TIMEOUT)
        return "released"

    return hold


def test_resume_returns_the_result_under_the_same_id():
    # Catches a follow-up that gets a new id, a new agent, or arrives without saying it's a follow-up.
    sub = make_subagents(chat=turns_profile(response("first"), response("second")))
    sub.spawn("chat", "go")
    agent = run_of(sub, 1).agent
    assert sub.resume(1, "more") == (
        '<subagent-result run="1" profile="chat" status="done">\nsecond\n</subagent-result>'
    )
    assert list(sub.runs) == [1] and run_of(sub, 1).agent is agent
    assert agent.messages[-2].content == f"{FOLLOW_UP}more"


def test_a_resumed_background_run_is_delivered_once():
    # Catches a follow-up's result never delivered (its id was already marked), or delivered twice.
    sub = make_subagents(chat=turns_profile(response("first"), response("second")))
    sub.spawn("chat", "go", background=True)
    assert "first" in sub.subagent(1, wait=TIMEOUT)
    assert sub.resume(1, "more", background=True) == "resumed run 1 with $1.00"
    eventually(lambda: sub.pending == 1)
    ((id, _, run),) = sub.claim()
    assert id == 1 and run.outcome(0).text == "second"
    assert sub.pending == 0 and sub.claim() == []
    assert "second" in sub.subagent(1)  # the latest result, and it isn't pending again


def test_resume_refuses_an_undelivered_result():
    # Catches a follow-up replacing a background result the model never read.
    sub = make_subagents(chat=turns_profile(response("first"), response("second")))
    sub.spawn("chat", "go", background=True)
    eventually(lambda: sub.pending == 1)
    before = run_of(sub, 1)
    with pytest.raises(ValueError, match=r"read it with subagent\(1\) first"):
        sub.resume(1, "more")
    assert run_of(sub, 1) is before and sub.pending == 1


def test_resume_refuses_a_running_run():
    # Catches a second turn started on an agent that is still in its first one.
    release = threading.Event()
    sub = make_subagents(held=held_profile(release))
    sub.spawn("held", "x", background=True)
    with pytest.raises(ValueError, match="still running"):
        sub.resume(1, "more")
    release.set()
    assert "did x" in sub.subagent(1, wait=TIMEOUT)


def test_a_cancelled_run_can_be_resumed():
    # Catches a cancel that is still permanent: the resumed run would stop at its first call.
    release = threading.Event()
    profile = turns_profile(
        response(tool_calls=[call("hold")]), response("picked up"), tools={"hold": blocking_tool(release)}
    )
    sub = make_subagents(held=profile)
    sub.spawn("held", "x", background=True)
    sub.subagent(1, cancel=True)
    release.set()
    assert 'reason="cancelled"' in sub.subagent(1, wait=TIMEOUT)
    assert sub.resume(1, "carry on") == (
        '<subagent-result run="1" profile="held" status="done">\npicked up\n</subagent-result>'
    )
    assert not run_of(sub, 1).agent.node.cancelled


def test_resume_tops_up_the_budget():
    # Catches a follow-up to a spent run that stops at once, a top-up past the profile's cap, or one the view
    # never hears about.
    spend = response("let me check", tool_calls=[call("add", a=1, b=2)], usage=usage(cost=0.4))
    events = []
    sub = make_subagents(
        AgentNode(1.0, listener=events.append),
        tiny=turns_profile(spend, response("finished"), max_cost=0.4, max_cost_cap=1.0),
    )
    assert 'reason="budget spent"' in sub.spawn("tiny", "go")
    node = run_of(sub, 1).agent.node
    assert sub.resume(1, "carry on", max_cost=5.0, background=True) == "resumed run 1 with $0.60"
    assert node.limit == pytest.approx(1.4)  # spent + the cap; the parent's $0.60 left is what binds
    assert [e for e in events if isinstance(e, LimitSet)] == [LimitSet(path="main/1", limit=node.limit)]
    assert 'status="done">\nfinished\n' in sub.subagent(1, wait=TIMEOUT)


def test_resume_refuses_a_non_positive_max_cost():
    # Catches a limit set below what was spent, which would stop the follow-up before its first call.
    sub = make_subagents(chat=turns_profile(response("first"), response("second")))
    sub.spawn("chat", "go")
    node = run_of(sub, 1).agent.node
    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            sub.resume(1, "more", max_cost=bad)
    assert node.limit == 1.0 and run_of(sub, 1).outcome(0).text == "first"


def test_an_interrupt_stops_a_foreground_resume():
    # Catches Ctrl+C not stopping a follow-up, or the stopped follow-up unable to be resumed again.
    release = threading.Event()
    profile = turns_profile(
        response("first"),
        response(tool_calls=[call("hold")]),
        response("picked up"),
        tools={"hold": blocking_tool(release)},
    )
    sub = make_subagents(held=profile)
    sub.spawn("held", "go")
    first = run_of(sub, 1)
    answers = []
    resumer = threading.Thread(target=lambda: answers.append(sub.resume(1, "more")))
    resumer.start()
    eventually(lambda: run_of(sub, 1) is not first)
    sub.node.cancel()
    eventually(lambda: run_of(sub, 1).agent.node.cancelled)
    release.set()
    resumer.join(TIMEOUT)
    assert answers == [
        '<subagent-result run="1" profile="held" status="stopped" reason="cancelled">\n(no reply)\n</subagent-result>'
    ]
    sub.node.reopen()  # the parent's next turn; resume reopens the run itself
    assert 'status="done">\npicked up\n' in sub.resume(1, "carry on")


def test_a_resumed_agent_can_resume_its_own_runs():
    # Catches a grandchild left unresumable by its parent's close, which cancels every child after each turn.
    digger = turns_profile(response("dug"), response("dug more"))

    def lead(task: str, node: AgentNode) -> Agent:
        client = FakeClient(
            [
                response(tool_calls=[call("spawn", profile="digger", task="dig")]),
                response("led"),
                response(tool_calls=[call("resume", run_id=1, message="dig more")]),
                response("led again"),
            ]
        )
        return Agent(client=client, node=node, subagent_profiles={"digger": digger})

    parent = Agent(
        client=FakeClient(
            [
                response(tool_calls=[call("spawn", profile="lead", task="lead it")]),
                response("done"),
                response(tool_calls=[call("resume", run_id=1, message="again")]),
                response("done again"),
            ]
        ),
        subagent_profiles={"lead": agent_profile("leads", lead, max_cost=1.0)},
    )
    parent.run_turn("go")
    tree = agent_tree(parent)
    assert tree["main/1/1"].node.cancelled  # the lead's close after its turn
    parent.run_turn("again")
    assert 'status="done">\ndug more\n' in tool_results(tree["main/1"])[-1]
    assert 'status="done">\nled again\n' in tool_results(parent)[-1]
    assert list(agent_tree(parent)) == ["main", "main/1", "main/1/1"]
