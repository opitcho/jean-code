"""Tests for `Budget`: the cost tree, its limits, journal and breakdown, and how `Agent` and the REPL use it.

Each test names the bug it would catch.
"""

import json
import threading

import pytest

from agent import Agent
from conftest import FakeClient, call, response, usage
from repl import print_cost
from test_compaction import compacted_session
from usage import Budget, breakdown_lines, record


def entry(cost=0.001, prompt=100, completion=10, cached=0, reasoning=0) -> dict:
    """A usage entry as `Agent._record_usage` stores it."""
    return {**usage(prompt=prompt, completion=completion, cached=cached, reasoning=reasoning, cost=cost),
            "id": "gen-1", "provider": "DeepSeek", "model": "m", "created": 1789748403}


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def tool_reply(cost=0.004) -> dict:
    return response(tool_calls=[call("add", a=1, b=2)], usage=usage(cost=cost))


# ---------------------------------------------------------------- the tree


def test_a_childs_spend_counts_against_its_ancestors_and_stays_at_its_node():
    # Catches a parent that doesn't see its subagents' spend, or entries filed at the wrong node.
    root = Budget()
    lead = root.child(0.5, "1")
    researcher = lead.child(0.1, "1")
    root.charge(entry(0.001))
    lead.charge(entry(0.002))
    researcher.charge(entry(0.004))
    assert researcher.spent == pytest.approx(0.004)
    assert lead.spent == pytest.approx(0.006)
    assert root.spent == pytest.approx(0.007)
    assert [len(node.entries) for node in (root, lead, researcher)] == [1, 1, 1]


def test_exhausted_follows_the_chain():
    # Catches a subagent that keeps spending after its parent's (or the session's) limit is reached.
    root = Budget(1.0)  # amounts exact in binary floating point, so the sum reaches the limit exactly
    lead = root.child(None, "1")
    researcher = lead.child(5.0, "1")
    researcher.charge(entry(0.75))
    assert not researcher.exhausted
    lead.charge(entry(0.25))  # the root's limit is now spent, by its descendants
    assert researcher.exhausted and lead.exhausted and root.exhausted


def test_exhausted_at_exactly_the_limit_and_not_before():
    # Catches an off-by-one in the limit check, which `max_cost` has always treated as `>=`.
    budget = Budget(0.002)
    budget.charge(entry(0.001))
    assert not budget.exhausted
    budget.charge(entry(0.001))
    assert budget.exhausted


def test_unknown_costs_count_as_zero():
    # Catches a TypeError when a response had no usage.
    budget = Budget(1.0)
    budget.charge(entry(cost=None))
    assert budget.spent == 0


def test_paths_and_the_journal():
    # Catches paths that don't match the eval harness's "main/1/2", or a journal per node instead of per tree.
    root = Budget()
    lead = root.child(None, "1")
    researcher = lead.child(None, "2")
    assert researcher.path == "main/1/2"
    researcher.charge(entry(0.004))
    root.charge(entry(0.001))
    assert researcher.journal is root.journal
    assert [e["budget"] for e in root.journal] == ["main/1/2", "main"]
    assert "budget" not in researcher.entries[0]  # the node's own entries are the agent's, untouched


def test_nodes_are_depth_first():
    root = Budget()
    a = root.child(None, "1")
    a.child(None, "1")
    root.child(None, "2")
    assert [n.path for n in root.nodes()] == ["main", "main/1", "main/1/1", "main/2"]


def test_concurrent_charges_are_all_counted():
    # Catches lost updates when parallel subagents charge one tree without the lock.
    root = Budget()
    nodes = [root.child(None, str(i)) for i in range(8)]
    start = threading.Barrier(len(nodes))

    def charge_many(node: Budget) -> None:
        start.wait()
        for _ in range(1000):
            node.charge(entry(0.001))

    threads = [threading.Thread(target=charge_many, args=(node,)) for node in nodes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(root.journal) == 8000
    assert sum(len(n.entries) for n in nodes) == 8000
    assert root.spent == pytest.approx(8.0)


# ---------------------------------------------------------------- breakdown and records


def test_breakdown_has_one_row_per_node_with_its_own_numbers():
    # Catches rows that mix a node's calls with its children's, or a wrong peak context.
    root = Budget(task="")
    lead = root.child(None, "1", task="Compare uv, Poetry and PDM")
    lead.charge(entry(0.002, prompt=1000, cached=800, completion=50, reasoning=20))
    lead.charge(entry(None, prompt=3000, cached=600, completion=70, reasoning=30))
    rows = {row["path"]: row for row in root.breakdown()}
    assert rows["main"]["calls"] == 0 and rows["main"]["cost"] == 0
    assert rows["main/1"] == {
        "path": "main/1", "task": "Compare uv, Poetry and PDM", "calls": 2,
        "prompt_tokens": 4000, "cached_tokens": 1400, "peak_prompt_tokens": 3000,
        "completion_tokens": 120, "reasoning_tokens": 50, "cost": pytest.approx(0.002), "unknown_costs": 1,
    }


def test_breakdown_lines_put_the_most_expensive_first():
    # Catches a table that hides the subagent that took the money.
    root = Budget()
    root.child(None, "1", task="cheap").charge(entry(0.001))
    root.child(None, "2", task="expensive").charge(entry(0.009))
    header, *lines = breakdown_lines(root.breakdown())
    assert header.split()[:2] == ["path", "task"]
    assert [line.split()[0] for line in lines] == ["main/2", "main/1", "main"]
    assert lines[0].split()[-1] == "$0.0090"


def test_breakdown_lines_mark_unknown_costs():
    root = Budget()
    root.charge(entry(None))
    assert breakdown_lines(root.breakdown())[1].split()[-1] == "$0.0000+?"


def test_record_keeps_the_budget_path():
    # Catches the log losing which agent made a call.
    root = Budget()
    root.child(None, "1").charge(entry(0.001))
    assert record(root.journal[0])["budget"] == "main/1"
    assert "budget" not in record(entry())  # entries without a path are recorded as before


# ---------------------------------------------------------------- Agent


def make_agent(responses, **kwargs) -> Agent:
    return Agent(client=FakeClient(responses), tools={"add": add}, **kwargs)


def test_max_cost_is_a_view_of_the_budget_limit():
    # Catches `max_cost` and the budget disagreeing, e.g. after the REPL raises the limit mid-session.
    agent = make_agent([], max_cost=0.01)
    assert agent.budget.limit == 0.01
    agent.max_cost = 1.0
    assert agent.budget.limit == 1.0 and agent.max_cost == 1.0
    assert make_agent([]).budget.limit is None


def test_max_cost_and_budget_together_are_rejected():
    # Catches two limits for one agent, where one would be silently ignored.
    with pytest.raises(ValueError, match="not both"):
        make_agent([], max_cost=1.0, budget=Budget(2.0))


def test_every_call_is_charged_and_usage_stays_the_agents_own():
    # Catches calls missing from the budget, or subagent entries leaking into an agent's `usage`.
    root = Budget()
    parent = make_agent([response("hi", usage=usage(cost=0.001))], budget=root)
    child = make_agent([tool_reply(), response("done", usage=usage(cost=0.002))], budget=root.child(None, "1"))
    parent.run_turn("hello")
    child.run_turn("go")
    assert parent.budget.entries == parent.usage and len(parent.usage) == 1
    assert child.budget.entries == child.usage and len(child.usage) == 2
    assert root.spent == pytest.approx(0.007)
    assert parent.total_cost == pytest.approx(0.001)  # its own calls only, as before
    assert [e["budget"] for e in root.journal] == ["main", "main/1", "main/1"]


def test_a_subagent_makes_no_call_once_its_parent_budget_is_spent():
    # Catches a subagent spending past the session's limit.
    root = Budget(0.004)
    parent = make_agent([tool_reply(cost=0.004)], budget=root)
    parent.run_turn("go")  # spends the root's whole limit; its next call is refused
    client = FakeClient([response("never")])
    child = Agent(client=client, budget=root.child(1.0, "1"))
    child.run_turn("hello")
    assert client.requests == []
    assert child.over_budget and child.interrupted


def test_compaction_calls_are_charged():
    # Catches the summary call, often a session's most expensive, missing from the budget.
    agent, _ = compacted_session()
    assert agent.budget.entries[-1]["kind"] == "compaction"
    assert agent.budget.spent == pytest.approx(agent.total_cost)


# ---------------------------------------------------------------- the REPL


def test_cost_prints_the_breakdown_once_there_are_several_agents(capsys):
    # Catches /cost hiding where the money went once subagents exist.
    agent = Agent(client=FakeClient([], credit_data={"usage": 0, "limit": None, "limit_remaining": None}))
    agent.budget.charge(entry(0.001))
    agent.budget.child(None, "1", task="look something up").charge(entry(0.004))
    print_cost(agent)
    out = capsys.readouterr().out
    assert "session: $0.0050 over 2 model calls" in out
    assert "main/1" in out and "look something up" in out


# ---------------------------------------------------------------- cancel, limits along the chain, logging


def test_cancel_stops_the_subtree_for_good():
    # Catches a cancel that misses descendants, reaches the parent, or wears off.
    root = Budget()
    run = root.child(1.0, "1")
    below = run.child(None, "1")
    run.cancel()
    assert run.exhausted and below.exhausted and below.cancelled
    assert not root.exhausted and not root.cancelled
    run.limit = 100.0  # nothing about the limit brings it back
    assert run.exhausted


def test_an_agent_on_a_cancelled_budget_makes_no_call():
    # Catches the cancel being cleared at the start of a turn, as the interrupt flag is.
    budget = Budget().child(1.0, "1")
    client = FakeClient([response("never")])
    agent = Agent(client=client, budget=budget)
    budget.cancel()
    agent.run_turn("hello")
    assert client.requests == []
    assert agent.over_budget


def test_remaining_is_the_tightest_limit_on_the_chain():
    # Catches a subagent told it has more than an ancestor has left.
    root = Budget(1.0)
    run = root.child(0.5, "1")
    assert run.remaining == pytest.approx(0.5)
    root.charge(entry(0.75))  # the root has 0.25 left, less than the run's own 0.5
    assert run.remaining == pytest.approx(0.25)
    root.charge(entry(0.5))  # overspent: never negative
    assert run.remaining == 0.0
    assert Budget().child(None, "1").remaining is None


def test_unknown_cost_stops_an_agent_whose_limit_is_an_ancestors():
    # Catches a child with no limit of its own counting unknown costs as $0 against the session's limit.
    root = Budget(2.0)
    client = FakeClient([response(tool_calls=[call("add", a=1, b=2)], no_usage=True), response("never")])
    child = Agent(client=client, tools={"add": add}, budget=root.child(None, "1"))
    child.run_turn("go")
    assert child.last_cost_unknown and child.interrupted
    assert len(client.requests) == 1


def test_unknown_cost_without_any_limit_changes_nothing():
    # Catches the chain check stopping agents that have no limit anywhere, which never stopped before.
    agent = make_agent([tool_reply(), response("done", no_usage=True)])
    agent.run_turn("go")
    assert not agent.last_cost_unknown and not agent.interrupted


def test_every_charge_is_logged_at_the_roots_path(tmp_path):
    # Catches calls missing from the usage log: a subagent's, or one made between two turns.
    path = tmp_path / "usage.jsonl"
    root = Budget(log_path=path)
    root.charge(entry(0.001))
    root.child(None, "1").child(None, "1").charge(entry(0.002))  # e.g. a background run, while the REPL is idle
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["budget"] for line in lines] == ["main", "main/1/1"]
    assert lines[1]["cost"] == 0.002


def test_no_log_without_a_path(tmp_path):
    root = Budget()
    root.child(None, "1").charge(entry())
    assert list(tmp_path.iterdir()) == []
