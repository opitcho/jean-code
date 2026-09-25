"""Tests for the subagents' tool boundary (`Subagents`: `spawn`, `subagent`, delivery, waits, limits) and how
`Agent` wires it in.

Each test names the bug it would catch.
"""

import threading
import time

import pytest

from agent import Agent
from conftest import FakeClient, call, response, usage
from subagents import MAX_RUNNING, MAX_WAIT, Profile, Subagents, agent_profile, agent_tree
from usage import Budget

TIMEOUT = 5  # seconds any wait in these tests may take before it counts as a hang


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def echo_profile(max_cost=1.0, max_cost_cap=None):
    """Answers "answer to <task>" in one call."""
    def build(task: str, budget: Budget) -> Agent:
        return Agent(client=FakeClient([response(f"answer to {task}")]), budget=budget)
    return agent_profile("answers at once", build, max_cost, max_cost_cap)


def held_profile(release: threading.Event):
    """Calls `hold`, which blocks until `release` is set, then answers "did <task>"."""
    def hold() -> str:
        """Wait until released."""
        release.wait(TIMEOUT)
        return "released"

    def build(task: str, budget: Budget) -> Agent:
        client = FakeClient([response(tool_calls=[call("hold")]), response(f"did {task}")])
        return Agent(client=client, tools={"hold": hold}, budget=budget)
    return agent_profile("blocks until released", build, max_cost=1.0)


def scripted_profile(responses):
    def build(task: str, budget: Budget) -> Agent:
        return Agent(client=FakeClient(list(responses)), tools={"add": add}, budget=budget)
    return agent_profile("scripted", build, max_cost=1.0)


def make_subagents(budget=None, **profiles) -> Subagents:
    return Subagents(profiles or {"echo": echo_profile()}, budget or Budget(), threading.Event())


def run_of(subagents: Subagents, run_id: int):
    return subagents.runs[run_id][1]


def eventually(predicate) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def tool_results(agent: Agent) -> list[str]:
    return [m["content"] for m in agent.messages if m["role"] == "tool"]


# ---------------------------------------------------------------- spawn and subagent


def test_foreground_spawn_returns_the_result_block():
    assert make_subagents().spawn("echo", "x") == (
        '<subagent-result run="1" profile="echo" status="done">\nanswer to x\n</subagent-result>')


def test_a_stopped_run_is_labelled_with_its_reason():
    # Catches the model receiving a mid-task remark as status="done".
    sub = make_subagents(tiny=scripted_profile(
        [response("let me check", tool_calls=[call("add", a=1, b=2)], usage=usage(cost=1.0))]))
    assert sub.spawn("tiny", "go") == (
        '<subagent-result run="1" profile="tiny" status="stopped" reason="budget spent">\nlet me check\n'
        '</subagent-result>')


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
    def broken_start(task, budget):
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


def test_a_profile_that_ignores_its_budget_is_refused():
    # Catches a run whose spend escapes the tree: no limit, missing from `spent`, `breakdown()` and the log.
    rogue = agent_profile("rogue", lambda task, budget: Agent(client=FakeClient([response("hi")]), budget=Budget()),
                          max_cost=1.0)
    sub = make_subagents(rogue=rogue)
    with pytest.raises(ValueError, match="didn't build its agent on the budget it was given"):
        sub.spawn("rogue", "x")
    assert sub.runs == {}


# ---------------------------------------------------------------- limits


def test_max_cost_defaults_is_clamped_and_states_what_applies():
    # Catches a limit above the cap, or the model told it has more than its parent has left.
    sub = make_subagents(Budget(0.3), echo=echo_profile(max_cost=0.5, max_cost_cap=1.0))
    assert sub.spawn("echo", "a", background=True) == "started run 1 with $0.30"  # the parent's 0.3 applies
    sub.spawn("echo", "b", background=True, max_cost=5.0)
    assert [run_of(sub, i).agent.budget.limit for i in (1, 2)] == [0.5, 1.0]
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
    parent = Agent(client=FakeClient([
        response(tool_calls=[call("spawn", profile="echo", task="x", background=True)]),
        response("started it"),
        response("got it"),
        response("fine"),
    ]), subagent_profiles={"echo": echo_profile()})
    parent.run_turn("start something")
    eventually(lambda: parent.subagents.pending == 1)
    parent.run_turn("any news?")
    assert parent.messages[-2]["content"] == (
        'any news?\n\n<subagent-result run="1" profile="echo" status="done">\nanswer to x\n</subagent-result>')
    assert parent.subagents.pending == 0
    parent.run_turn("and now?")
    assert parent.messages[-2]["content"] == "and now?"


def test_a_result_read_with_subagent_is_not_injected_again():
    sub = make_subagents()
    sub.spawn("echo", "x", background=True)
    assert "answer to x" in sub.subagent(1, wait=TIMEOUT)
    assert sub.pending == 0 and sub.claim() == []


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

    def meet() -> str:
        """Wait for the others."""
        barrier.wait()
        return "met"

    def build(task: str, budget: Budget) -> Agent:
        return Agent(client=FakeClient([response(tool_calls=[call("meet")]), response(f"{task} done")]),
                     tools={"meet": meet}, budget=budget)

    parent = Agent(client=FakeClient([
        response(tool_calls=[call("spawn", profile="p", task=t, background=True) for t in "abc"]),
        response(tool_calls=[call("subagent", run_id=i, wait=TIMEOUT) for i in (1, 2, 3)]),
        response("all done"),
    ]), subagent_profiles={"p": agent_profile("meets", build, max_cost=1.0)})
    parent.run_turn("fan out")
    assert [r.split(">\n")[1].split("\n")[0] for r in tool_results(parent)[3:]] == ["a done", "b done", "c done"]
    # The runs answer even when the barrier breaks (the tool error goes back to them as text), so the proof is
    # inside: every `meet` returned. Run one after another, each gets "Error: BrokenBarrierError"
    # (.experiments/output/subagents_fanout_control.txt).
    assert [tool_results(run.agent) for _, _, run in parent.subagents.snapshot()] == [["met"]] * 3


# ---------------------------------------------------------------- through the agent


def test_the_tools_take_loose_json_and_answer_errors_as_text():
    # Catches a string id rejected, or a bad call crashing the turn instead of informing the model.
    parent = Agent(client=FakeClient([
        response(tool_calls=[call("spawn", profile="echo", task="x", background=True)]),
        response(tool_calls=[call("subagent", run_id="1", wait=TIMEOUT), call("subagent", run_id=7),
                             call("spawn", profile="nope", task="y")]),
        response("done"),
    ]), subagent_profiles={"echo": echo_profile()})
    parent.run_turn("go")
    started, by_string, unknown_id, unknown_profile = tool_results(parent)
    assert started == "started run 1 with $1.00"
    assert 'status="done"' in by_string
    assert unknown_id == "Error: ValueError: unknown run 7; runs: 1"
    assert unknown_profile == "Error: ValueError: unknown profile 'nope'; profiles: echo"


def test_the_listing_and_tools_come_only_with_profiles():
    # Catches management methods exposed as tools, or tools and a listing for an agent without profiles.
    agent = Agent(client=FakeClient([]), system_prompt="Be brief.", subagent_profiles={"echo": echo_profile()})
    assert agent.messages[0]["content"].startswith("Be brief.\n\n## Subagents\n")
    assert "- echo: answers at once" in agent.messages[0]["content"]
    assert set(agent._tools()) == {"spawn", "subagent"}
    plain = Agent(client=FakeClient([]))
    assert plain.subagents is None and plain._tools() == {} and plain.messages == []


def test_agent_tree_follows_the_budget_paths():
    # Catches the tree and the budget disagreeing on a run's path.
    def lead(task: str, budget: Budget) -> Agent:
        return Agent(client=FakeClient([response(tool_calls=[call("spawn", profile="echo", task="dig")]),
                                        response("led")]),
                     budget=budget, subagent_profiles={"echo": echo_profile()})

    parent = Agent(client=FakeClient([response(tool_calls=[call("spawn", profile="lead", task="lead it")]),
                                      response("done")]),
                   subagent_profiles={"lead": agent_profile("leads", lead, max_cost=1.0)})
    parent.run_turn("go")
    assert list(agent_tree(parent)) == ["main", "main/1", "main/1/1"]
    assert [row["path"] for row in parent.budget.breakdown()] == ["main", "main/1", "main/1/1"]


# ---------------------------------------------------------------- interrupts and close


def test_an_interrupt_ends_a_wait_and_leaves_the_run_going():
    # Catches Ctrl+C doing nothing while the parent waits up to 10 minutes on subagent(id, wait=600).
    release = threading.Event()
    sub = make_subagents(held=held_profile(release))
    sub.spawn("held", "x", background=True)
    answers = []
    waiter = threading.Thread(target=lambda: answers.append(sub.subagent(1, wait=MAX_WAIT)))
    waiter.start()
    time.sleep(0.1)
    sub._interrupted.set()
    waiter.join(1)
    assert not waiter.is_alive() and answers[0].startswith("run 1 is still running")
    assert not run_of(sub, 1).agent.budget.cancelled  # background runs keep going, like shell jobs
    release.set()


def test_an_interrupt_stops_a_foreground_run():
    release = threading.Event()
    sub = make_subagents(held=held_profile(release))
    answers = []
    spawner = threading.Thread(target=lambda: answers.append(sub.spawn("held", "x")))
    spawner.start()
    eventually(lambda: 1 in sub.runs)
    sub._interrupted.set()
    eventually(lambda: run_of(sub, 1).agent.budget.cancelled)  # spawn noticed within a poll, and cancelled its run
    release.set()  # the call in flight (the hold) ends; the run makes no further call
    spawner.join(TIMEOUT)
    assert answers == ['<subagent-result run="1" profile="held" status="stopped" reason="cancelled">\n(no reply)\n'
                       '</subagent-result>']


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
    assert all(run.agent.budget.cancelled for _, _, run in parent.subagents.snapshot())
