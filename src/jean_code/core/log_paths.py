"""Where agents' logs go on disk.

There's a folder per process, a folder per agent tree in it, and a file per agent; `process_logs` is this
process's. Nothing is created until a path is asked for, so a process that logs nothing leaves no folder.
"""

import itertools
import os
import threading
import time
import weakref
from pathlib import Path

from jean_code.config import PROCESS_LOGS_DIR
from jean_code.core.node import AgentNode


class ProcessLogs:
    """A process's log folder, with one numbered folder per agent tree in it.

    The folder is `<root>/<YYYYmmdd-HHMMSS>-<pid>/`. An agent tree is a root agent and every subagent below it;
    trees are numbered from 1 in the order they first ask for a path. Thread-safe.

    Args:
        root: The folder the process folders go in; kept as `root`.
    """

    def __init__(self, root: Path):
        self.root = root
        self._dir: Path | None = None
        self._lock = threading.Lock()
        self._numbers = itertools.count(1)
        # Root node → its tree's folder, by identity. Weak, so closed chats' and finished trials' trees can be freed.
        self._trees: weakref.WeakKeyDictionary[AgentNode, Path] = weakref.WeakKeyDictionary()

    @property
    def dir(self) -> Path:
        """The process's folder, created on first use."""
        with self._lock:
            return self._process_dir()

    def tree_dir(self, node: AgentNode) -> Path:
        """Returns the folder of the agent tree `node` is in, created on first use.

        Args:
            node: Any node of the tree.
        """
        root = node.root
        with self._lock:
            if (tree := self._trees.get(root)) is None:
                tree = self._process_dir() / str(next(self._numbers))  # never reused: no tree lands in an old folder
                tree.mkdir()
                self._trees[root] = tree
            return tree

    def agent_file(self, node: AgentNode, suffix: str = ".jsonl") -> Path:
        """Returns the log file of the agent at `node`, e.g. `3/main-1-2.jsonl`.

        The file is in its tree's folder, which exists; creating the file is the caller's job.

        Args:
            node: The agent's node.
            suffix: The file's extension.
        """
        return self.tree_dir(node) / (node.path.replace("/", "-") + suffix)

    def _process_dir(self) -> Path:
        """Returns the process's folder, created on first use; the caller holds the lock."""
        if self._dir is None:
            dir = self.root / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
            dir.mkdir(parents=True)  # no exist_ok: a second owner of this process's folder is a bug
            self._dir = dir
        return self._dir


process_logs = ProcessLogs(PROCESS_LOGS_DIR)  # the one every caller uses; tests build their own
