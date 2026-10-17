"""Tests for `logs.py`: the process's log folder, the recorder's trace and usage log, and reading a trace back.

Each test names the bug it would catch.
"""

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from conftest import FakeClient, agent_added, call, response
from jean_code.core.agent import Agent
from jean_code.core.driver import Driver
from jean_code.core.events import (
    ApprovalRequested,
    Configured,
    Delivered,
    JobEnded,
    LimitSet,
    MessageAdded,
    RunEnded,
    SkillLoaded,
)
from jean_code.core.llm import UserMessage
from jean_code.core.node import AgentNode
from jean_code.core.subagents import agent_profile
from jean_code.core.view import AgentView
from jean_code.logs import ProcessLogs, Recorder, read_trace, replay
from test_driver import Builder, eventually, next_of
from test_subagents import echo_profile

# ---------------------------------------------------------------- the process folder


def test_nothing_is_created_until_asked(tmp_path):
    # Catches a process that never logs leaving an empty folder behind.
    ProcessLogs(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_one_process_folder_that_says_what_ran(tmp_path):
    # Catches a process whose trees land in two folders, a name that doesn't sort by start time, or a trace that
    # can't be matched to the command and code that made it.
    logs = ProcessLogs(tmp_path)
    assert re.fullmatch(rf"\d{{8}}-\d{{6}}-{os.getpid()}", logs.dir.name)
    assert logs.dir == logs.dir
    assert list(tmp_path.iterdir()) == [logs.dir]
    info = json.loads((logs.dir / "process.json").read_text())
    assert info["argv"] == sys.argv
    assert info["pid"] == os.getpid()
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4}", info["started"])
    assert set(info["git"]) == {"sha", "dirty"}


def test_trees_are_numbered_chats_made_on_first_write(tmp_path):
    # Catches two root agents, both "main", sharing a folder, or a chat that never wrote leaving an empty folder.
    logs = ProcessLogs(tmp_path)
    first, second = logs.new_tree(), logs.new_tree()
    assert (first, second) == (logs.dir / "chat-1", logs.dir / "chat-2")
    assert not first.exists() and not second.exists()
    Recorder(AgentView(), second).close()
    assert sorted(p.name for p in logs.dir.iterdir()) == ["process.json"]


def test_concurrent_new_trees_get_different_folders(tmp_path):
    # Catches a race between two new trees: one number for two trees, or a second process folder.
    logs = ProcessLogs(tmp_path)
    barrier = threading.Barrier(8)
    found = []

    def make() -> None:
        barrier.wait()
        found.append(logs.new_tree())

    threads = [threading.Thread(target=make) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(int(p.name.removeprefix("chat-")) for p in found) == list(range(1, 9))
    assert len(list(tmp_path.iterdir())) == 1


# ---------------------------------------------------------------- the recorder


def lead_profile():
    """A lead that spawns an echo, then answers."""

    def build(task: str, node: AgentNode) -> Agent:
        client = FakeClient([response(tool_calls=[call("spawn", profile="echo", task="dig")]), response("led")])
        return Agent(client=client, node=node, subagent_profiles={"echo": echo_profile()})

    return agent_profile("leads", build, max_cost=1.0)


def traced(build, tmp_path, usage_log=None, **driver_kwargs) -> tuple[Driver, Recorder]:
    """A driver whose view a recorder writes to `tmp_path / "trace"`, subscribed before the agent is built."""
    view = AgentView()
    recorder = Recorder(view, tmp_path / "trace", usage_log)
    return Driver(build, view=view, **driver_kwargs), recorder


def turn(driver: Driver, text: str) -> None:
    with driver.view.subscribe() as (_, events):
        driver.send(text)
        next_of(events, JobEnded)


def finish(driver: Driver, recorder: Recorder) -> None:
    driver.close()
    recorder.close()


def lines(file: Path) -> list[dict]:
    return [json.loads(line) for line in file.read_text().splitlines()]


def spawning_driver(tmp_path, usage_log=None) -> tuple[Driver, Recorder]:
    build = Builder(
        [response(tool_calls=[call("spawn", profile="lead", task="lead it")]), response("done")],
        subagent_profiles={"lead": lead_profile()},
    )
    driver, recorder = traced(build, tmp_path, usage_log)
    turn(driver, "go")
    finish(driver, recorder)
    return driver, recorder


def test_folders_follow_the_tree(tmp_path):
    # Catches a subagent filed outside its parent's folder, an event written to another agent's file, or folders
    # named by bare run numbers that can't be told from the tree's own number.
    spawning_driver(tmp_path)
    trace = tmp_path / "trace"
    paths = {
        str(f.relative_to(trace)): {line["path"] for line in lines(f) if "path" in line}
        for f in trace.rglob("events.jsonl")
    }
    assert paths == {
        "events.jsonl": {"main"},
        "1-lead/events.jsonl": {"main/1"},
        "1-lead/1-echo/events.jsonl": {"main/1/1"},
    }


def test_a_node_without_an_agent_gets_a_plain_folder(tmp_path):
    # Catches an agent under a grouping node (the engine's, later) with no folder for the group, or a crash.
    view = AgentView()
    recorder = Recorder(view, tmp_path / "trace")
    view.listen(agent_added("main"))
    view.listen(agent_added("main/wf1/plan", profile="planner"))
    recorder.close()
    assert (tmp_path / "trace" / "wf1" / "plan-planner" / "events.jsonl").is_file()
    assert recorder.errors == []


def test_replay_matches_the_view(tmp_path):
    # Catches an event the recorder drops, or one that doesn't survive the trip through JSON.
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: A demo.\n---\nDEMO BODY\n")
    load = {"id": "l1", "type": "function", "function": {"name": "load_skill", "arguments": '{"name": "demo"}'}}
    script = [
        response(tool_calls=[call("spawn", profile="echo", task="t1")]),
        response("spawned"),
        response(tool_calls=[load]),
        response("loaded"),
        response(tool_calls=[call("spawn", profile="echo", task="t2", background=True)]),
        response("started"),
        response(tool_calls=[call("rm", path="x")]),
        response("removed"),
        response("three"),
        response("four"),
        response("the summary"),
    ]
    build = Builder(
        script, needs_approval={"rm"}, subagent_profiles={"echo": echo_profile()}, skills_dir=tmp_path / "skills"
    )
    driver, recorder = traced(build, tmp_path)
    turn(driver, "spawn one")
    turn(driver, "load the skill")
    driver.configure(reasoning="high", max_cost=5.0)
    turn(driver, "spawn one in the background")
    eventually(lambda: driver.view.state().waiting("main") == 1)
    with driver.view.subscribe() as (_, events):
        driver.send("remove x")  # also delivers the background result
        driver.answer(next_of(events, ApprovalRequested).approval.id)
        next_of(events, JobEnded)
    turn(driver, "three")
    turn(driver, "four")
    with driver.view.subscribe() as (_, events):
        driver.compact()
        assert next_of(events, JobEnded).result == "compacted"
    finish(driver, recorder)
    kinds = {type(event) for event in read_trace(tmp_path / "trace")}
    assert {Configured, LimitSet, SkillLoaded, RunEnded, Delivered} <= kinds
    assert replay(tmp_path / "trace") == driver.view.state()


def test_seq_is_one_timeline(tmp_path):
    # Catches files that can't be merged back into one order: a gap, a repeat, or a file out of order.
    spawning_driver(tmp_path)
    by_file = [[line["seq"] for line in lines(f)] for f in (tmp_path / "trace").rglob("events.jsonl")]
    assert all(seqs == sorted(seqs) for seqs in by_file)
    assert sorted(seq for seqs in by_file for seq in seqs) == list(range(1, sum(map(len, by_file)) + 1))


def test_deltas_are_not_written(tmp_path):
    # Catches a trace made mostly of reply pieces that the whole message repeats.
    driver, recorder = traced(Builder([response("a streamed reply")], stream=True), tmp_path)
    turn(driver, "go")
    finish(driver, recorder)
    types = [line["type"] for line in lines(tmp_path / "trace" / "events.jsonl")]
    assert "delta" not in types
    assert driver.view.state().agents["main"].items[-1].message.content == "a streamed reply"
    assert types.count("message") == 2  # the user's, and the reply


def test_driver_events_go_to_the_root(tmp_path):
    # Catches approvals and jobs missing from the trace, or written somewhere no reader looks.
    build = Builder([response(tool_calls=[call("rm", path="x")]), response("removed")], needs_approval={"rm"})
    driver, recorder = traced(build, tmp_path)
    with driver.view.subscribe() as (_, events):
        driver.send("remove x")
        driver.answer(next_of(events, ApprovalRequested).approval.id)
        next_of(events, JobEnded)
    finish(driver, recorder)
    types = [line["type"] for line in lines(tmp_path / "trace" / "events.jsonl")]
    for kind in ("job_started", "approval_requested", "approval_resolved", "job_ended"):
        assert kind in types


def test_usage_log_gets_every_call(tmp_path):
    # Catches calls missing from the spend log, a subagent's above all, or a line in a new format.
    usage_log = tmp_path / "logs" / "usage.jsonl"
    driver, _ = spawning_driver(tmp_path, usage_log)
    calls = driver.view.state().calls
    assert {c.path for c in calls} == {"main", "main/1", "main/1/1"}
    assert usage_log.read_text().splitlines() == [c.log_line() for c in calls]


def test_late_subscribe_is_refused(tmp_path):
    # Catches a trace that silently lacks its start: the root's agent_added and system message.
    driver = Driver(Builder([]))
    with pytest.raises(ValueError, match="before the agent is built"):
        Recorder(driver.view, tmp_path / "trace")
    driver.close()


def test_close_writes_what_is_queued(tmp_path):
    # Catches close() returning, or the process going on, before everything already emitted is on disk.
    view = AgentView()
    recorder = Recorder(view, tmp_path / "trace")
    view.listen(agent_added("main"))
    for i in range(500):
        view.listen(MessageAdded(path="main", message=UserMessage(content=str(i)), context_tokens=0))
    recorder.close()
    assert len(read_trace(tmp_path / "trace")) == 501


def test_a_failed_trace_keeps_the_usage_log(tmp_path, caplog):
    # Catches a broken trace silencing the spend log, failing the turn, or failing without a word.
    (tmp_path / "trace").write_text("a file where the tree's folder should be")
    usage_log = tmp_path / "usage.jsonl"
    build = Builder([response(tool_calls=[call("add", a=1, b=2)]), response("3")])
    driver, recorder = traced(build, tmp_path, usage_log)
    turn(driver, "add")
    finish(driver, recorder)
    assert driver.view.state().agents["main"].end == "answered"
    assert len(recorder.errors) == 1 and recorder.errors[0].startswith("trace: ")
    assert recorder.trace_dir is None and recorder.usage_log == usage_log
    assert "recorder trace stopped" in caplog.text
    assert len(usage_log.read_text().splitlines()) == 2


def test_a_cut_trace_replays_busy(tmp_path):
    # Catches a trace cut off mid-job replaying as a finished one, which a reader would trust.
    driver, recorder = traced(Builder([response("hi")]), tmp_path)
    turn(driver, "go")
    finish(driver, recorder)
    trace = tmp_path / "trace"
    finished = replay(trace)
    assert not finished.busy and not finished.agents["main"].running
    file = trace / "events.jsonl"
    kept = file.read_text().splitlines(keepends=True)
    cut = next(n for n, line in enumerate(kept) if '"turn_started"' in line) + 1
    file.write_text("".join(kept[:cut]))
    state = replay(trace)
    assert state.busy and state.agents["main"].running


def test_unfinished_last_line_is_skipped(tmp_path):
    # Catches a killed process's half-written line making its whole trace unreadable, or a bad line hidden.
    driver, recorder = traced(Builder([response("hi")]), tmp_path)
    turn(driver, "go")
    finish(driver, recorder)
    trace = tmp_path / "trace"
    file = trace / "events.jsonl"
    whole = len(read_trace(trace))
    file.write_text(file.read_text() + '{"seq": 999, "type": "mess')
    assert len(read_trace(trace)) == whole
    file.write_text('{"not": "an event"}\n' + file.read_text())
    with pytest.raises(ValueError):
        read_trace(trace)


def test_unclosed_recorder_is_drained_at_exit(tmp_path):
    # Catches a process that exits without closing its recorder losing the events still queued.
    added = agent_added("main").model_dump_json()
    code = f"""
from pathlib import Path
from jean_code.core.events import AgentAdded, MessageAdded
from jean_code.core.llm import UserMessage
from jean_code.core.view import AgentView
from jean_code.logs import Recorder

view = AgentView()
Recorder(view, Path({str(tmp_path / "trace")!r}))
view.listen(AgentAdded.model_validate_json({added!r}))
for i in range(3000):
    view.listen(MessageAdded(path="main", message=UserMessage(content=str(i)), context_tokens=0))
"""  # exits right after emitting, without closing the recorder
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)  # noqa: S603 - this Python, fixed code
    assert len(read_trace(tmp_path / "trace")) == 3001
