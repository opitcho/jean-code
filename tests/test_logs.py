"""Tests for `logs.py`: the process's log folder, the recorder's trace and usage log, and reading a trace back.

Each test names the bug it would catch.
"""

import json
import os
import re
import sys
import threading

from jean_code.logs import ProcessLogs

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


def test_trees_are_numbered_in_order(tmp_path):
    # Catches two root agents, both "main", sharing a folder: two chats in the app, two eval trials.
    logs = ProcessLogs(tmp_path)
    first, second = logs.new_tree(), logs.new_tree()
    assert (first, second) == (logs.dir / "1", logs.dir / "2")
    assert first.is_dir() and second.is_dir()


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
    assert sorted(int(p.name) for p in found) == list(range(1, 9))
    assert len(list(tmp_path.iterdir())) == 1
