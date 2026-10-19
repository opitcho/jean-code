"""Logging agent trees to disk, from their views.

`Recorder` writes what an `AgentView` receives: the trace, one `events.jsonl` per agent in folders nested like the
tree, and the usage log. `read_trace` and `replay` read a trace back. `ProcessLogs` is a process's log folder, with
a numbered folder per agent tree in it; `process_logs` is this process's. `append_log` appends model calls to a usage
log.

Example:
    view = AgentView()
    recorder = Recorder(view, process_logs.new_tree(), USAGE_LOG)
    driver = Driver(build, view=view)
    ...
    driver.close()
    recorder.close()
"""

import atexit
import itertools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from pydantic import TypeAdapter

from jean_code.config import PROCESS_LOGS_DIR, ROOT
from jean_code.core.events import Charged, Delta, ViewEvent
from jean_code.core.usage import ModelCall
from jean_code.core.view import AgentView, TreeState, apply

log = logging.getLogger(__name__)
EVENTS: TypeAdapter[ViewEvent] = TypeAdapter(ViewEvent)
_STOP = object()  # ends the recorder's loop; put on the queue after everything it must write


class Recorder:
    """Writes everything an agent tree's view receives, on a thread of its own.

    - Each node's events go to its own `events.jsonl`: the root's in `trace_dir`, any other node's (a subagent's or a
      group's) in a folder under its parent's named `<name>-<profile>`, e.g. `1-research/2-researcher/events.jsonl`.
      A folder is made when its first line is written.
    - Each line is an event's JSON plus `seq`, its place among the lines written, and `at`, the Unix time it was
      written. Streamed deltas aren't written.
    - A write that fails turns off its sink (the trace or the usage log), sets that sink's argument to None and adds
      to `errors`. The agents never see the failure.
    - A recorder that isn't closed is closed at exit.

    The arguments it keeps are attributes of the same name.

    Args:
        view: The view to subscribe to; it must have received no event yet.
        trace_dir: The tree's folder; None for no trace.
        usage_log: The file each model call is appended to as `ModelCall.log_line`; None for none.

    Attributes:
        errors: Why a sink stopped, one entry per failed sink, e.g. "trace: [Errno 28] No space left on device".

    Raises:
        ValueError: The view has already received events, so the trace would lack its start.
    """

    def __init__(self, view: AgentView, trace_dir: Path | None, usage_log: Path | None = None):
        self._subscription = view.subscribe()
        if self._subscription.state.agents or self._subscription.state.busy:
            self._subscription.close()
            raise ValueError("subscribe the recorder before the agent is built, or its trace lacks the start")
        self.trace_dir = trace_dir
        self.usage_log = usage_log
        self.errors: list[str] = []
        self._seq = 0
        self._root: str | None = None
        self._folders: dict[str, Path] = {}  # node path -> its folder, fixed by the node's first event
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        atexit.register(self.close)  # needs the daemon: exit joins non-daemon threads before atexit runs

    def close(self) -> None:
        """Writes what's still queued, then stops; waits until it has.

        Call it once the tree's driver is closed, so no event comes after it. Calling it again does nothing.
        """
        atexit.unregister(self.close)  # a closed recorder isn't kept alive until exit
        self._subscription.close()
        self._subscription.queue.put_nowait(_STOP)  # pyright: ignore[reportArgumentType]
        self._thread.join()

    def _run(self) -> None:
        for event in iter(self._subscription.get, _STOP):
            if isinstance(event, Delta):
                continue
            if self.trace_dir is not None:
                try:
                    self._trace(self.trace_dir, event)
                except Exception as e:  # noqa: BLE001 - a broken trace stops only the trace
                    self._fail("trace", e)
                    self.trace_dir = None
            if self.usage_log is not None and isinstance(event, Charged):
                try:
                    append_log(self.usage_log, [event.call])
                except Exception as e:  # noqa: BLE001 - a broken usage log stops only the usage log
                    self._fail("usage log", e)
                    self.usage_log = None

    def _trace(self, trace_dir: Path, event: ViewEvent) -> None:
        path = getattr(event, "path", None) or self._root  # a driver event belongs to the root
        if path is None:
            raise ValueError(f"a {event.type} event came before any agent's, so it has no folder")
        self._root = self._root or path  # the first event is the root's AgentAdded
        folder = self._folder(trace_dir, path, getattr(event, "profile", ""))
        folder.mkdir(parents=True, exist_ok=True)
        self._seq += 1
        line = {"seq": self._seq, "at": round(time.time(), 3)} | event.model_dump(mode="json")
        with (folder / "events.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")

    def _folder(self, trace_dir: Path, path: str, profile: str = "") -> Path:
        if (folder := self._folders.get(path)) is None:
            parent, _, name = path.rpartition("/")
            if not parent:
                folder = trace_dir  # the tree's folder is the root agent's
            else:  # a node's first event is its AgentAdded or GroupAdded, so its profile names it
                folder = self._folder(trace_dir, parent) / (f"{name}-{profile}" if profile else name)
            self._folders[path] = folder
        return folder

    def _fail(self, sink: str, error: Exception) -> None:
        self.errors.append(f"{sink}: {error}")
        log.warning("recorder %s stopped: %s", sink, error)


def read_trace(dir: Path) -> list[ViewEvent]:
    """Returns every event in a tree's trace, in the order the view received them.

    Args:
        dir: The tree's folder, as a `Recorder` wrote it.

    Raises:
        ValueError: A line other than a file's unfinished last line isn't an event.
    """
    lines = [
        json.loads(raw)
        for file in dir.rglob("events.jsonl")
        for raw in file.read_text().split("\n")[:-1]  # an unfinished last line is from a killed process
    ]
    lines.sort(key=lambda line: line.get("seq", 0))
    return [EVENTS.validate_python(line) for line in lines]


def replay(dir: Path) -> TreeState:
    """Returns the tree's state as its trace describes it, as its `AgentView` held it.

    A trace that ends mid-job replays with `busy` set.

    Args:
        dir: The tree's folder, as a `Recorder` wrote it.

    Raises:
        ValueError: A line other than a file's unfinished last line isn't an event.
    """
    state = TreeState()
    for event in read_trace(dir):
        apply(state, event)
    return state


class ProcessLogs:
    """A process's log folder, with one numbered folder per agent tree in it.

    The folder is `<root>/<YYYYmmdd-HHMMSS>-<pid>/`, with a `process.json` that says what ran: argv, pid, start
    time, and the git commit and whether the checkout had changes. Tree folders are `chat-1`, `chat-2`, … in the
    order they're asked for. Nothing is created until a folder is asked for. Thread-safe.

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
        """Returns the next agent tree's folder, in the process's folder.

        The process's folder is created; the tree's folder isn't, so a tree that never writes leaves nothing.
        """
        with self._lock:
            # Never reused, though not yet created: no tree lands in another's folder.
            return self._process_dir() / f"chat-{next(self._numbers)}"

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


def append_log(path: str | Path, calls: list[ModelCall]) -> None:
    """Appends model calls to a JSON Lines log, one `ModelCall.log_line` per line.

    The log only grows, so summing its `cost` fields gives the spend across all sessions. The file and its parent
    folders are created if missing.

    Args:
        path: The log file.
        calls: The calls to append; nothing is written if it's empty.
    """
    if not calls:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for call in calls:
            f.write(call.log_line() + "\n")


process_logs = ProcessLogs(PROCESS_LOGS_DIR)  # the one every caller uses; tests build their own
