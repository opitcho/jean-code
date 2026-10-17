"""Tests for `ProcessLogs` (log_paths.py): the process's log folder, one folder per agent tree, one file per agent.

Each test names the bug it would catch.
"""

import gc
import os
import re
import threading

from conftest import FakeClient, call, response
from jean_code.core.agent import Agent
from jean_code.core.log_paths import ProcessLogs
from jean_code.core.node import AgentNode
from jean_code.core.subagents import agent_profile, agent_tree
from test_subagents import echo_profile


def test_nothing_is_created_until_asked(tmp_path):
    # Catches a process that never logs leaving an empty folder behind.
    ProcessLogs(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_one_process_folder(tmp_path):
    # Catches a process whose agents write to two folders, or a name that doesn't sort by start time.
    logs = ProcessLogs(tmp_path)
    assert re.fullmatch(rf"\d{{8}}-\d{{6}}-{os.getpid()}", logs.dir.name)
    assert logs.dir == logs.dir
    assert list(tmp_path.iterdir()) == [logs.dir]


def test_subagents_share_their_roots_tree(tmp_path):
    # Catches a subagent filed outside its parent's tree, or a file name that loses part of the path.
    logs = ProcessLogs(tmp_path)
    root = AgentNode()
    child = root.child(None, "1")
    grand = child.child(None, "2")
    assert [logs.agent_file(node) for node in (root, child, grand)] == [
        logs.dir / "1" / "main.jsonl",
        logs.dir / "1" / "main-1.jsonl",
        logs.dir / "1" / "main-1-2.jsonl",
    ]


def test_each_tree_gets_its_own_folder(tmp_path):
    # Catches two root agents, both "main", writing to the same file: two chats in the app, two eval trials.
    logs = ProcessLogs(tmp_path)
    first, second = AgentNode(), AgentNode()
    assert logs.tree_dir(second).name == "1"  # numbered by first ask, not by creation
    assert logs.tree_dir(first).name == "2"
    assert logs.agent_file(first) != logs.agent_file(second)


def test_concurrent_first_asks(tmp_path):
    # Catches a race on a tree's first ask: two folders for one tree, or one number for two trees.
    def ask_all(logs: ProcessLogs, nodes: list[AgentNode]) -> list:
        barrier = threading.Barrier(len(nodes))
        found = [None] * len(nodes)

        def ask(i: int) -> None:
            barrier.wait()
            found[i] = logs.tree_dir(nodes[i])

        threads = [threading.Thread(target=ask, args=(i,)) for i in range(len(nodes))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return found

    one_tree = ProcessLogs(tmp_path / "one")
    root = AgentNode()
    assert len(set(ask_all(one_tree, [root.child(None, str(i)) for i in range(8)]))) == 1
    assert [p.name for p in one_tree.dir.iterdir()] == ["1"]

    many_trees = ProcessLogs(tmp_path / "many")
    found = ask_all(many_trees, [AgentNode() for _ in range(8)])
    assert sorted(int(p.name) for p in found) == list(range(1, 9))


def test_a_dropped_tree_is_forgotten(tmp_path):
    # Catches the mapping keeping every finished tree alive, or a new tree reusing a dropped tree's folder.
    logs = ProcessLogs(tmp_path)
    root = AgentNode()
    logs.tree_dir(root)
    del root
    gc.collect()
    assert len(logs._trees) == 0
    assert logs.tree_dir(AgentNode()).name == "2"


def test_spawned_agents_land_in_the_tree(tmp_path):
    # Catches an agent built by a profile, rather than by the caller, filed outside its parent's tree.
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
    logs = ProcessLogs(tmp_path)
    files = {path: logs.agent_file(agent.node) for path, agent in agent_tree(parent).items()}
    assert files == {
        "main": logs.dir / "1" / "main.jsonl",
        "main/1": logs.dir / "1" / "main-1.jsonl",
        "main/1/1": logs.dir / "1" / "main-1-1.jsonl",
    }
