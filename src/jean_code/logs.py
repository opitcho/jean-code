"""Logging agent trees to disk.

`ProcessLogs` is a process's log folder, with a numbered folder per agent tree in it; `process_logs` is this
process's.
"""

import itertools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from jean_code.config import PROCESS_LOGS_DIR, ROOT


class ProcessLogs:
    """A process's log folder, with one numbered folder per agent tree in it.

    The folder is `<root>/<YYYYmmdd-HHMMSS>-<pid>/`, with a `process.json` that says what ran: argv, pid, start
    time, and the git commit and whether the checkout had changes. Trees are numbered from 1 in the order they're
    made. Nothing is created until a folder is asked for. Thread-safe.

    Args:
        root: The folder the process folders go in; kept as `root`.
    """

    def __init__(self, root: Path):
        self.root = root
        self._dir: Path | None = None
        self._lock = threading.Lock()
        self._numbers = itertools.count(1)

    @property
    def dir(self) -> Path:
        """The process's folder, created on first use."""
        with self._lock:
            return self._process_dir()

    def new_tree(self) -> Path:
        """Creates the next agent tree's folder, and returns it."""
        with self._lock:
            tree = self._process_dir() / str(next(self._numbers))  # never reused: no tree lands in an old folder
            tree.mkdir()
            return tree

    def _process_dir(self) -> Path:
        """Returns the process's folder, created on first use; the caller holds the lock."""
        if self._dir is None:
            dir = self.root / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
            dir.mkdir(parents=True)  # no exist_ok: a second owner of this process's folder is a bug
            info = {
                "argv": sys.argv,
                "pid": os.getpid(),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "git": {"sha": _git("rev-parse", "HEAD"), "dirty": _dirty()},
            }
            (dir / "process.json").write_text(json.dumps(info, indent=2) + "\n")
            self._dir = dir
        return self._dir


def _git(*args: str) -> str | None:
    """Returns git's output in the repo, stripped; None if git fails."""
    try:
        done = subprocess.run(  # noqa: S603 - fixed git commands
            ["git", *args],  # noqa: S607 - git is found on PATH
            check=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _dirty() -> bool | None:
    status = _git("status", "--porcelain")
    return None if status is None else status != ""


process_logs = ProcessLogs(PROCESS_LOGS_DIR)  # the one every caller uses; tests build their own
