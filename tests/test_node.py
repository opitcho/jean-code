"""Tests for the work tree: `Node`'s two kinds, how children are numbered and found, and what a group shares with the
agents under it. Spend and limits along the chain are tested in `test_budget.py`.

Each test names the bug it would catch.
"""

import pytest

from conftest import FakeClient, call, response, usage
from jean_code.core.agent import Agent
from jean_code.core.events import GroupAdded, LimitSet
from jean_code.core.node import AgentNode, Node, UnknownNode
from jean_code.core.tool_schema import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def agent_on(node: Node, *responses) -> Agent:
    return Agent(client=FakeClient(list(responses)), tools={"add": add}, node=node.add_agent(None))


def test_children_are_numbered_across_both_kinds():
    # Catches a spawn and a group under one parent both taking "main/1".
    root = AgentNode()
    children = [root.add_agent(None), root.add_group(None), root.add_agent(None)]
    assert [n.path for n in children] == ["main/1", "main/2", "main/3"]


def test_nodes_are_depth_first():
    root = AgentNode()
    a = root.add_agent(None)
    a.add_agent(None)
    root.add_group(None)
    assert [n.path for n in root.nodes()] == ["main", "main/1", "main/1/1", "main/2"]


def test_root():
    # Catches a node that doesn't lead back to its tree's root: its events would reach no listener.
    root = AgentNode()
    grand = root.add_group(None).add_agent(None)
    assert root.root is root and grand.root is root


def test_find_reaches_any_depth_within_its_subtree():
    # Catches a cancel or a limit by path that can't reach a node deep in the tree, or lands on another one.
    root = AgentNode()
    group = root.add_group(None)
    deep = group.add_agent(None).add_agent(None)
    assert root.find("main/1/1/1") is deep and root.find("main") is root and group.find("main/1") is group
    for path, node in [("main/2", root), ("main", group)]:
        with pytest.raises(UnknownNode):
            node.find(path)


def test_add_group_announces_the_group_before_its_limit_changes():
    # Catches a group the view never hears about, so a later LimitSet for it is skipped.
    events = []
    group = AgentNode(listener=events.append).add_group(0.5, task="compare", profile="pair")
    group.limit = 0.2
    assert events == [
        GroupAdded(path="main/1", profile="pair", task="compare", limit=0.5),
        LimitSet(path="main/1", limit=0.2),
    ]


def test_a_groups_limit_stops_every_agent_under_it_and_not_main():
    # Catches agents in a group that together spend past its limit, or a group's limit that stops main too.
    root = AgentNode(5.0)
    main = Agent(client=FakeClient([response("hi")]), node=root)
    group = root.add_group(0.10)
    a = agent_on(group, response("first", usage=usage(cost=0.0625)), response("never"))
    b = agent_on(group, response(tool_calls=[call("add", a=1, b=2)], usage=usage(cost=0.0625)), response("never"))
    assert a.run_turn("go") == "answered"
    assert b.run_turn("go") == "budget spent"  # its first call takes the group past $0.10
    assert a.run_turn("again") == "budget spent"
    assert group.spent == pytest.approx(0.125)
    assert main.run_turn("hello") == "answered"


def test_cancelling_a_group_stops_the_agents_under_it_and_leaves_main_open():
    # Catches a cancel of a group that misses an agent under it, even one added later, or reaches main.
    root = AgentNode()
    group = root.add_group(None)
    a, b = group.add_agent(None), group.add_agent(None)
    group.cancel()
    assert a.cancelled and b.cancelled and group.add_agent(None).cancelled
    assert not root.cancelled


def test_cancelling_main_reaches_the_agents_in_its_groups():
    # Catches a stop of the whole tree that an agent in a group runs through.
    root = AgentNode()
    agent = root.add_group(None).add_agent(None)
    root.cancel()
    root.reopen()  # main's next turn
    assert agent.cancelled and not root.cancelled


def test_agents_leaves_out_groups_and_a_node_whose_start_failed():
    # Catches a breakdown row for something that never ran a model call.
    root = AgentNode()
    Agent(client=FakeClient([]), node=root)
    inner = agent_on(root.add_group(None)).node
    root.add_agent(None)  # a start that failed: the node, but no agent
    assert root.agents() == [root, inner]
    assert [row["path"] for row in root.breakdown()] == ["main", "main/1/1"]
