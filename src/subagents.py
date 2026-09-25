"""Subagents: letting an agent start other agents, wait for them, or hand work off and carry on.

Two layers. The Python API is for code: a `Profile` is the spec of one kind of subagent, and
`profile.start(task, budget)` returns a `Run`, one subagent working on one task on its own thread, whose result
is a future. `Subagents` is the boundary with a model: the `spawn` and `subagent` tools, which give runs small
ids and turn results into text, plus the methods the agent and UIs use to manage those runs.

A run's result is the text its agent last replied (or a profile's report). A run that ended before its agent
answered (budget spent, cancelled, context full, cost unknown, out of tool rounds) completes with `Stopped`
instead, so a mid-task remark is never mistaken for an answer.
"""

from __future__ import annotations

import itertools
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from usage import Budget

if TYPE_CHECKING:
    from agent import Agent

MAX_RUNNING = 8  # runs per parent at once: bounds the overshoot past a limit, and parallel requests to the provider
MAX_WAIT = 600  # seconds `subagent(wait=…)` may block, like `job`
POLL_SECONDS = 0.2  # how often a wait checks whether the parent was interrupted


@dataclass(frozen=True)
class Profile:
    """The spec of one kind of subagent: what it's for, what it may spend, and how to start one on a task."""

    description: str  # one line, listed in the parent's system prompt
    start: Callable[[str, Budget], Run]  # (task, budget) → a new running Run; each call is a new run
    max_cost: float  # default limit per run, in USD; required, so no run is unlimited
    max_cost_cap: float | None = None  # the most a caller may give one run; None → max_cost

    def __post_init__(self):
        if self.max_cost <= 0:
            raise ValueError(f"max_cost must be positive, got {self.max_cost}")
        if self.max_cost_cap is not None and self.max_cost_cap < self.max_cost:
            raise ValueError(f"max_cost_cap ({self.max_cost_cap}) must be at least max_cost ({self.max_cost})")

    @property
    def cap(self) -> float:
        return self.max_cost if self.max_cost_cap is None else self.max_cost_cap


def agent_profile(description: str, build: Callable[[str, Budget], Agent], max_cost: float,
                  max_cost_cap: float | None = None) -> Profile:
    """A profile for the common case: build a fresh agent on the given budget, run it on the task, and hand back its
    last reply. `build(task, budget)` must pass `budget=budget` to the `Agent`."""
    return Profile(description, lambda task, budget: Run.start(build(task, budget), task), max_cost, max_cost_cap)


class Stopped(Exception):
    """A run ended before its agent answered. `reason` says why; `text` is what it last said (or its report),
    which may be incomplete but can still help."""

    def __init__(self, reason: str, text: str | None):
        super().__init__(reason)
        self.reason = reason
        self.text = text


@dataclass(frozen=True, eq=False)
class Run:
    """One subagent working on one task: the running agent and the future of its result.

    Only `Run.start` creates one, and it starts the run's thread. Nothing about a Run changes afterwards except its
    future, which the standard library completes exactly once: with the text to hand back, with `Stopped`, or with
    the exception that ended the run.
    """

    agent: Agent
    task: str
    _future: futures.Future = field(default_factory=futures.Future, repr=False)

    @classmethod
    def start(cls, agent: Agent, task: str, report: Callable[[], str] | None = None) -> Run:
        """Start `agent` on `task` on a new thread, and return at once.

        `report`, if given, produces the text to hand back instead of the agent's last reply: for profiles whose
        real output is somewhere else, such as a notebook the agent wrote to.
        """
        if agent.needs_approval:  # nobody can approve a subagent's calls: refuse rather than stop at the first one
            raise ValueError(f"a subagent can't have tools that need approval: {', '.join(sorted(agent.needs_approval))}")
        run = cls(agent, task)
        # A daemon thread, like the REPL's worker: close() is the orderly stop, and a crashed process doesn't hang on it.
        threading.Thread(target=run._work, args=(report,), name=f"subagent {agent.budget.path}", daemon=True).start()
        return run

    def _work(self, report: Callable[[], str] | None) -> None:
        """The run's own thread: get the outcome, close the agent, then publish the outcome."""
        try:
            outcome: str | BaseException = self._outcome(report)
        except BaseException as e:
            outcome = e
        try:
            self.agent.close()  # before publishing: a finished run leaves nothing running below it
        except BaseException as e:
            if not isinstance(outcome, BaseException):  # keep the first error if there are two
                outcome = e
        if isinstance(outcome, BaseException):  # always completed, so a run never looks "running" forever
            self._future.set_exception(outcome)
        else:
            self._future.set_result(outcome)

    def _outcome(self, report: Callable[[], str] | None) -> str | Stopped:
        self.agent.run_turn(self.task)
        reason = why_stopped(self.agent)  # read now: a cancel() after the turn mustn't relabel a finished run
        text = report() if report else last_reply(self.agent)
        if reason:
            return Stopped(reason, text)
        return text or "(the subagent's final reply was empty)"

    @property
    def path(self) -> str:
        """The run's place in the budget tree, e.g. "main/1/2": unique in the tree."""
        return self.agent.budget.path

    def done(self) -> bool:
        return self._future.done()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run is done or `timeout` seconds pass; return whether it's done."""
        futures.wait([self._future], timeout=timeout)
        return self.done()

    def result(self, timeout: float | None = None) -> str:
        """The agent's answer, waiting for it. Raises `Stopped` if it ended before answering, or the run's error."""
        return self._future.result(timeout)

    def cancel(self) -> None:
        """No more model calls in this run or below it: it stops after the call in flight. Can't be lost or undone."""
        self.agent.budget.cancel()


def why_stopped(agent: Agent) -> str | None:
    """Why the agent's turn ended before it answered, or None when its last message is an answer."""
    last = agent.messages[-1]
    if last["role"] == "assistant" and not last.get("tool_calls"):
        return None  # it answered, even if that call spent the last of the budget
    if agent.budget.cancelled:
        return "cancelled"
    if agent.over_budget:
        return "budget spent"
    if agent.last_cost_unknown:
        return "a call's cost was unknown, so the budget couldn't be enforced"
    if agent.stop_reason:
        return agent.stop_reason  # e.g. the context is nearly full
    return f"out of tool rounds ({agent.max_tool_rounds})"


def last_reply(agent: Agent) -> str | None:
    """The text of the agent's last assistant message, or None if it has none."""
    for message in reversed(agent.messages):
        if message["role"] == "assistant":
            return (message.get("content") or "").strip() or None
    return None


def run_status(run: Run) -> tuple[str, str | None, str]:
    """(status, reason, text) of a run: "running"; "done" with its answer; "stopped" with why and what it last said;
    or "failed" with the error."""
    if not run.done():
        return "running", None, ""
    try:
        return "done", None, run.result(timeout=0)
    except Stopped as s:
        return "stopped", s.reason, s.text or "(no reply)"
    except BaseException as e:
        return "failed", None, f"{type(e).__name__}: {e}"


def result_block(id: int, profile: str, run: Run) -> str:
    """A finished run's result as text for the model: done (its answer), stopped (why, and what it last said) or
    failed (the error)."""
    status, reason, text = run_status(run)
    attrs = f'run="{id}" profile="{profile}" status="{status}"' + (f' reason="{reason}"' if reason else "")
    return f"<subagent-result {attrs}>\n{text}\n</subagent-result>"


def subagents_prompt(profiles: dict[str, Profile]) -> str:
    """The subagents listing for the system prompt: names and descriptions, like the skills listing."""
    if not profiles:
        return ""
    lines = [
        "## Subagents",
        "A subagent is a separate agent you start with spawn to work on one task. It sees only the task you give "
        "it, so put everything it needs in the task. Its final reply comes back to you as the result.",
        "",
    ]
    lines += [f"- {name}: {profile.description}" for name, profile in profiles.items()]
    return "\n".join(lines)


def agent_tree(agent: Agent) -> dict[str, Agent]:
    """Every agent under `agent`, itself included, keyed by budget path ("main", "main/1", "main/1/2"): for
    transcripts, evals and UIs."""
    tree = {agent.budget.path: agent}
    if agent.subagents:
        for _, _, run in agent.subagents.snapshot():
            tree |= agent_tree(run.agent)
    return tree


class Subagents:
    """An agent's subagents: the two tools its model calls, and the methods the agent and UIs use to manage its runs.

    Tools (the model calls these with JSON arguments and reads their text results):
        spawn     start a subagent, and wait for its result or let it run in the background
        subagent  check on a run, wait for it, or cancel it

    Management (code calls these; they are never tools):
        claim     take the finished background results nobody has seen, once each (`Agent.run_turn` injects them)
        pending   how many finished results wait to be claimed (UIs)
        running   how many runs haven't finished (spawn's cap, UIs)
        snapshot  every run so far, by id (UIs, `agent_tree`)
        close     cancel every run and wait for them (`Agent.close`)

    Only here do runs have ids and results become text. Waits end early when the agent is interrupted (its
    `interrupted` event is the agent's own); background runs keep going.

    A background result is delivered once: a finished background run is pending until `claim` or `subagent`
    marks it delivered. Pending is derived from the runs' futures, not recorded by a callback, since a future
    wakes its waiters before it runs its callbacks: a result read in between would be delivered twice.
    """

    def __init__(self, profiles: dict[str, Profile], budget: Budget, interrupted: threading.Event):
        self.profiles = profiles
        self.budget = budget
        self._interrupted = interrupted  # the parent agent's interrupt flag: waits end when it's set
        self.runs: dict[int, tuple[str, Run]] = {}  # id → (profile name, run); append-only; only spawn inserts
        self._ids = itertools.count(1)  # only spawn uses it, on the parent's thread
        self._background: set[int] = set()  # ids of runs started in the background
        self._delivered: set[int] = set()  # ids whose result the model has received; never removed
        self._lock = threading.Lock()  # guards runs, _background and _delivered

    # ---------------------------------------------------------------- tools

    def spawn(self, profile: str, task: str, background: bool = False, max_cost: float | None = None) -> str:
        """Start a new subagent: a separate agent that works on `task` and hands back one text result.

        Args:
            profile: Name of the kind of subagent to start, exactly as listed under "## Subagents" in your
                instructions. It fixes the subagent's model, tools and default budget. An unknown name starts
                nothing and returns an error listing the valid names.
            task: The prompt the subagent receives. It becomes the subagent's first and only user message: it sees
                none of this conversation, none of your tool results and none of the files you read. Put in every
                fact, file path and constraint it needs, and say what its answer should contain, because its final
                reply is what you get back.
            background: False (the default): this call blocks until the subagent finishes and returns its result;
                if the user interrupts you, the subagent is stopped too. True: this call returns at once with the
                run's id while the subagent keeps working; get its result with subagent(run_id), or it is appended
                to the next user message after it finishes. To run several subagents in parallel, make several
                background spawn calls in one reply. At most 8 of your subagents can run at once; past that, spawn
                returns an error.
            max_cost: The most, in USD, this subagent may spend, including any subagents it starts itself. Must be
                positive. If omitted, the profile's default is used. A value above the profile's cap is lowered to
                the cap, and it can't exceed what you have left yourself. When the limit is reached, the subagent
                stops at once, and its result says status="stopped".

        Returns:
            Foreground: a <subagent-result run="N" profile="…" status="…"> block. status="done" holds the
            subagent's final answer; status="stopped" means it ended before answering (reason="…" says why) and
            holds what it last said, which may be incomplete; status="failed" holds the error. Background:
            "started run N with $X", where N is the run_id for subagent() and $X is the limit that applies.
        """
        spec = self.profiles.get(profile)
        if spec is None:
            raise ValueError(f"unknown profile '{profile}'; profiles: {', '.join(self.profiles)}")
        if max_cost is not None and max_cost <= 0:
            raise ValueError(f"max_cost must be positive, got {max_cost}")
        if (running := self.running()) >= MAX_RUNNING:
            raise ValueError(f"{running} subagents are running, the most at once; "
                             "collect one with subagent(run_id, wait=…) first")
        id = next(self._ids)  # taken before start: a start that fails skips its id
        limit = min(spec.max_cost if max_cost is None else max_cost, spec.cap)
        node = self.budget.child(limit, name=str(id), task=task)
        run = spec.start(task, node)
        if run.agent.budget is not node:  # otherwise its spend would escape this tree, its limit and the log
            run.cancel()
            raise ValueError(f"profile '{profile}' didn't build its agent on the budget it was given")
        with self._lock:
            self.runs[id] = (profile, run)
            if background:
                self._background.add(id)
        if background:
            return f"started run {id} with ${node.remaining:.2f}"
        if not self._wait(run, None):  # interrupted: stop the run, then wait for the call it's making
            run.cancel()
            run.wait()
        return result_block(id, profile, run)

    def subagent(self, run_id: int, wait: int = 0, cancel: bool = False) -> str:
        """Check on a subagent you started with spawn: how far it has got, or its result once it has finished.

        Args:
            run_id: The number spawn gave the run ("started run 3 …" → 3). Numbers start at 1 in each conversation
                and are never reused. An unknown number returns an error listing the valid ones.
            wait: How many seconds to block waiting for the run to finish before answering. 0 (the default)
                answers at once; values above 600 are lowered to 600. If the run finishes within that time you get
                its result; otherwise you get a progress line (model calls made and USD spent so far) and the run
                keeps going. If the user interrupts you, the wait ends early.
            cancel: True stops the run: it ends after the model call it is making now, and any subagents it started
                are stopped too. Its result then says status="stopped". Pass wait as well to receive that result
                in this call; otherwise it arrives like any finished background run.

        Returns:
            Finished: the same <subagent-result …> block spawn would have returned; asking again returns it again.
            Still running: the progress line.
        """
        with self._lock:
            entry = self.runs.get(run_id)
            ids = ", ".join(map(str, self.runs)) or "none"
        if entry is None:
            raise ValueError(f"unknown run {run_id}; runs: {ids}")
        profile, run = entry
        if cancel:
            run.cancel()
        if not self._wait(run, min(max(wait, 0), MAX_WAIT)):
            return (f"run {run_id} is still running: {len(run.agent.usage)} model calls, "
                    f"${run.agent.budget.spent:.4f} spent so far")
        self.claim(run_id)
        return result_block(run_id, profile, run)

    def _wait(self, run: Run, timeout: float | None) -> bool:
        """Wait until the run is done, the parent is interrupted, or `timeout` seconds pass (None: no timeout).
        Returns whether the run is done."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not run.done() and not self._interrupted.is_set():
            left = POLL_SECONDS if deadline is None else min(POLL_SECONDS, deadline - time.monotonic())
            if left <= 0:
                break
            run.wait(left)
        return run.done()

    # ---------------------------------------------------------------- management

    def claim(self, run_id: int | None = None) -> list[tuple[int, str, Run]]:
        """Take the pending background results, in id order: all of them, or only `run_id`'s, and mark them
        delivered. Each run is returned by one claim at most; reading a result again with `subagent` needs none."""
        with self._lock:
            ids = self._pending_ids() if run_id is None else [i for i in self._pending_ids() if i == run_id]
            self._delivered.update(ids)
            return [(i, *self.runs[i]) for i in ids]

    @property
    def pending(self) -> int:
        """Finished background results the model hasn't received yet."""
        with self._lock:
            return len(self._pending_ids())

    def _pending_ids(self) -> list[int]:
        """Finished background runs not yet delivered, in id order. The caller holds the lock."""
        return sorted(i for i in self._background - self._delivered if self.runs[i][1].done())

    def running(self) -> int:
        """Runs that haven't finished."""
        return sum(not run.done() for _, _, run in self.snapshot())

    def snapshot(self) -> list[tuple[int, str, Run]]:
        """Every run so far, as (id, profile name, run), in id order."""
        with self._lock:
            return [(i, name, run) for i, (name, run) in self.runs.items()]

    def close(self) -> None:
        """Cancel every run, then wait for all of them: each stops after the model call it is making."""
        runs = [run for _, _, run in self.snapshot()]
        for run in runs:
            run.cancel()
        for run in runs:
            run.wait()
