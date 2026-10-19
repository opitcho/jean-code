"""Subagents: agents an agent starts, waits for, or hands work off to.

Two layers. For code: a `Profile` is the spec of one kind of subagent, and `profile.start(task, node)` returns a
`Run`, one subagent working on one task on its own thread, which ends with an `Outcome`. For a model: `Subagents`
holds an agent's runs and gives its model the `spawn`, `subagent` and `resume` tools, which give runs small ids
and turn results into text.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent import futures
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, overload

from jean_code.core.events import Delivered, RunEnded, error_line
from jean_code.core.llm import AssistantMessage
from jean_code.core.node import AgentNode
from jean_code.core.tool_schema import tool

if TYPE_CHECKING:
    from jean_code.core.agent import Agent, TurnEnd

MAX_RUNNING = 8  # runs per parent at once: bounds the overshoot past a limit, and parallel requests to the provider
MAX_WAIT = 600  # seconds `subagent(wait=…)` may block, like `job`
POLL_SECONDS = 0.2  # how often a wait checks whether the parent was stopped
FOLLOW_UP = "Follow-up from the agent that started you:\n\n"  # starts a resumed run's message: its note says one task


# TODO: Customized tools per profile
@dataclass(frozen=True)
class Profile:
    """The spec of one kind of subagent: its purpose, its limits, and how to start one.

    Attributes:
        description: One line, listed in the parent's system prompt.
        start: `(task, node)` → a new running `Run`; each call is a new run.
        max_cost: The default limit per run, in USD; required, so no run is unlimited.
        max_cost_cap: The most a caller may give one run (a spawn or a follow-up); None means `max_cost`.

    Raises:
        ValueError: `max_cost` isn't positive, or `max_cost_cap` is below it.
    """

    description: str
    start: Callable[[str, AgentNode], Run]
    max_cost: float
    max_cost_cap: float | None = None

    def __post_init__(self):
        if self.max_cost <= 0:
            raise ValueError(f"max_cost must be positive, got {self.max_cost}")
        if self.max_cost_cap is not None and self.max_cost_cap < self.max_cost:
            raise ValueError(f"max_cost_cap ({self.max_cost_cap}) must be at least max_cost ({self.max_cost})")

    @property
    def cap(self) -> float:
        """The most a caller may give one run."""
        return self.max_cost if self.max_cost_cap is None else self.max_cost_cap


def agent_profile(
    description: str, build: Callable[[str, AgentNode], Agent], max_cost: float, max_cost_cap: float | None = None
) -> Profile:
    """Returns the common profile: a fresh agent per task, handing back its last reply.

    Args:
        description: As `Profile.description`.
        build: `(task, node)` → the agent; it must pass `node=node` to the `Agent`.
        max_cost: As `Profile.max_cost`.
        max_cost_cap: As `Profile.max_cost_cap`.
    """
    return Profile(description, lambda task, node: Run.start(build(task, node), task), max_cost, max_cost_cap)


@dataclass(frozen=True)
class Outcome:
    """How a run ended.

    - "done": the agent answered; `text` is its answer (or the profile's report).
    - "stopped": it ended before answering; `reason` says why, and `text` is what it last said (or the report),
      which may be incomplete but can still help.
    - "failed": the agent, its report or its cleanup raised; `text` is the error, followed by the report when the
      agent raised and the profile has one.

    However a run ended, cancelled included, its agent can be resumed with a follow-up.

    Attributes:
        status: "done", "stopped" or "failed", as above.
        text: The answer, the report or the error, as above.
        reason: Why it stopped, for "stopped".
    """

    status: Literal["done", "stopped", "failed"]
    text: str
    reason: TurnEnd | None = None


@dataclass(frozen=True, eq=False)
class Run:
    """One task a subagent worked on: its agent, and the task's `Outcome` once it ends.

    A resumed subagent has one Run per task, all on the same agent. Only `Run.start` creates one. Nothing about a
    Run changes afterwards except whether it's done.

    Attributes:
        agent: The subagent.
        task: This turn's user message: the spawn task, or a follow-up.
        report: Produces the text handed back instead of the agent's last reply; None hands back the reply.
    """

    agent: Agent
    task: str
    report: Callable[[], str] | None = None  # kept on the run, so a resumed run reports the same way
    _future: futures.Future[Outcome] = field(default_factory=futures.Future, repr=False)

    @classmethod
    def start(cls, agent: Agent, task: str, report: Callable[[], str] | None = None) -> Run:
        """Starts `agent` on `task` on a new thread, and returns at once.

        Args:
            agent: The subagent.
            task: Its next user message.
            report: Produces the text to hand back instead of the agent's last reply, for profiles whose real
                output is elsewhere, such as a notebook the agent wrote to.

        Returns:
            The new run.

        Raises:
            ValueError: The agent has tools that need approval, which nobody could approve.
        """
        if agent.needs_approval:  # nobody can approve a subagent's calls: refuse rather than wait at the first one
            names = ", ".join(sorted(agent.needs_approval))
            raise ValueError(f"a subagent can't have tools that need approval: {names}")
        run = cls(agent, task, report)
        # A daemon thread, like the REPL's worker: close() is the orderly stop, and a crashed process doesn't hang
        # on it.
        threading.Thread(target=run._work, name=f"subagent {agent.node.path}", daemon=True).start()
        return run

    def resume(self, message: str) -> Run:
        """Starts the agent's next turn, on `message`, however this run ended.

        Clears a cancel on the agent's own node first; topping up its limit is the caller's job.

        Args:
            message: The agent's next user message.

        Returns:
            A new Run on the same agent, already started.

        Raises:
            ValueError: This run is still running: the agent takes one turn at a time.
        """
        if not self.done():
            raise ValueError(f"run {self.path} is still running; wait for it to finish first")
        # Between turns, on the caller's thread: a cancel sent once the new turn exists still holds for it
        self.agent.node.reopen()
        return Run.start(self.agent, message, self.report)

    def _work(self) -> None:
        """Runs the agent on the run's own thread, closes it, then sets the outcome.

        The one place a run's errors are caught: whatever the agent, its report or its cleanup raises (code this
        module doesn't control) becomes a "failed" outcome, so the future is always completed, with a value.
        """
        try:
            outcome = self._finish()
        except BaseException as e:  # noqa: BLE001 - the future must always get an outcome
            outcome = Outcome("failed", error_line(e))
        try:
            self.agent.close()  # before publishing: a finished run leaves nothing running below it
        except BaseException as e:  # noqa: BLE001 - the future must always get an outcome
            if outcome.status != "failed":  # keep the first error if there are two
                outcome = Outcome("failed", error_line(e))
        # Before the future: a result is delivered only once the run is done, so RunEnded comes before its Delivered.
        try:
            self.agent.node.emit(RunEnded(path=self.path, status=outcome.status, reason=outcome.reason))
        finally:  # a listener that raises must not leave the parent waiting on a future that never completes
            self._future.set_result(outcome)

    def _finish(self) -> Outcome:
        """Runs the agent's turn and returns how it ended.

        If there's a report, its text replaces the agent's last reply.
        """
        report = self.report
        try:
            end = self.agent.run_turn(self.task)
        except BaseException as e:
            if report is None:
                raise
            # A report hands back what the agent left elsewhere (a notebook), which an error doesn't undo.
            error = error_line(e)
            try:
                return Outcome("failed", f"{error}\n\n{report()}")
            except BaseException as r:  # noqa: BLE001 - the outcome reports both errors
                return Outcome("failed", f"{error}\n\nThe report failed too: {error_line(r)}")
        text = report() if report else last_reply(self.agent)
        if end == "answered":
            return Outcome("done", text or "(the subagent's final reply was empty)")
        return Outcome("stopped", text or "(no reply)", end)

    @property
    def path(self) -> str:
        """The run's place in the agent tree, e.g. "main/1/2": unique in the tree."""
        return self.agent.node.path

    def done(self) -> bool:
        """Returns whether the run has ended."""
        return self._future.done()

    def wait(self, timeout: float | None = None) -> bool:
        """Blocks until the run is done or `timeout` seconds pass.

        Args:
            timeout: Seconds to wait; None to wait until it's done.

        Returns:
            Whether the run is done.
        """
        futures.wait([self._future], timeout=timeout)
        return self.done()

    @overload
    def outcome(self, timeout: None = None) -> Outcome: ...
    @overload
    def outcome(self, timeout: float) -> Outcome | None: ...
    def outcome(self, timeout: float | None = None) -> Outcome | None:
        """Returns how the run ended, waiting up to `timeout` seconds.

        Args:
            timeout: Seconds to wait; None to wait until it ends.

        Returns:
            The outcome, or None if the run is still running.
        """
        return self._future.result() if self.wait(timeout) else None

    def cancel(self) -> None:
        """Stops model calls in this run and below it, after the call in flight.

        It holds until the run is resumed, which clears it.
        """
        self.agent.node.cancel()


def last_reply(agent: Agent) -> str | None:
    """Returns the text of the agent's last reply, or None if it has none.

    Args:
        agent: The agent.
    """
    for message in reversed(agent.messages):
        if isinstance(message, AssistantMessage):
            return message.content.strip() or None
    return None


def result_block(id: int, profile: str, outcome: Outcome) -> str:
    """Returns how a run ended, as text for the model.

    The text holds its status, why it stopped if it did, and the outcome's text.

    Args:
        id: The run's id.
        profile: The run's profile name.
        outcome: How it ended.
    """
    attrs = f'run="{id}" profile="{profile}" status="{outcome.status}"'
    if outcome.reason:
        attrs += f' reason="{outcome.reason}"'
    return f"<subagent-result {attrs}>\n{outcome.text}\n</subagent-result>"


def subagents_prompt(profiles: dict[str, Profile]) -> str:
    """Returns the subagents listing for the system prompt: names and descriptions.

    It's empty without profiles.

    Args:
        profiles: The kinds of subagent, by name.
    """
    if not profiles:
        return ""
    lines = [
        "## Subagents",
        "A subagent is a separate agent you start with spawn to work on one task. It sees only the task you give "
        "it, so put everything it needs in the task. Its final reply comes back to you as the result. Once it has "
        "stopped, however it stopped, you can send it a follow-up with resume; it keeps everything it saw and did.",
        "",
    ]
    lines += [f"- {name}: {profile.description}" for name, profile in profiles.items()]
    return "\n".join(lines)


def agent_tree(agent: Agent) -> dict[str, Agent]:
    """Returns every agent under `agent`, itself included, by path.

    Paths look like "main", "main/1", "main/1/2".

    Args:
        agent: The top of the tree to list.
    """
    return {node.path: node.agent for node in agent.node.agents() if node.agent is not None}


class Subagents:
    """An agent's subagents: their runs, and the tools its model manages them with.

    Tools:
        spawn     start a subagent, and wait for its result or let it run in the background
        subagent  check on a run, wait for it, or cancel it
        resume    send a stopped run's agent a follow-up: its next turn, under the same id

    It numbers the runs from 1, and turns their outcomes into the text the model reads. Waits end early when the
    parent agent is stopped, and stopping the parent stops its runs too, background ones included.

    A background run's result is delivered once: when the run finishes, its result is pending until `claim` or
    `subagent` marks it delivered. A resumed run has a new result to deliver, under the same id. The arguments it
    keeps are attributes of the same name.

    Args:
        profiles: The kinds of subagent the model may start, by name.
        node: The parent agent's node; each run gets a child of it.

    Attributes:
        runs: Each id's profile name and latest run; spawn adds ids, and resume replaces an id's run.
    """

    def __init__(self, profiles: dict[str, Profile], node: AgentNode):
        self.profiles = profiles
        self.node = node
        self.runs: dict[int, tuple[str, Run]] = {}
        self._background: set[Run] = set()  # runs started in the background; never removed
        self._delivered: set[Run] = set()  # by Run, not id, since resume gives an id a new Run; never removed
        self._lock = threading.Lock()  # guards runs, _background and _delivered

    # ---------------------------------------------------------------- tools

    @tool
    def spawn(self, profile: str, task: str, background: bool = False, max_cost: float | None = None) -> str:
        """Start a new subagent: a separate agent that works on `task` and hands back one text result.

        Args:
            profile: Name of the kind of subagent to start, exactly as listed under "## Subagents" in your
                instructions. It fixes the subagent's model, tools and default budget. An unknown name starts
                nothing and returns an error listing the valid names.
            task: The prompt the subagent receives. It becomes the subagent's first user message: it sees
                none of this conversation, none of your tool results and none of the files you read. Put in every
                fact, file path and constraint it needs, and say what its answer should contain, because its final
                reply is what you get back.
            background: False (the default): this call blocks until the subagent finishes and returns its result.
                True: this call returns at once with the run's id while the subagent keeps working; get its result
                with subagent(run_id), or it is appended to the next user message after it finishes. Either way, if
                the user stops you, the subagent is stopped too. To run several subagents in parallel, make several
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
            raise ValueError(
                f"{running} subagents are running, the most at once; collect one with subagent(run_id, wait=…) first"
            )
        limit = min(spec.max_cost if max_cost is None else max_cost, spec.cap)
        child = self.node.add_agent(limit, task=task, profile=profile)
        id = int(child.name)  # numbered before start: a start that fails skips its id
        run = spec.start(task, child)
        if run.agent.node is not child:  # otherwise its spend would escape this tree, its limit and the log
            run.cancel()
            raise ValueError(f"profile '{profile}' didn't build its agent on the node it was given")
        with self._lock:
            self.runs[id] = (profile, run)
            if background:
                self._background.add(run)
        if background:
            return f"started run {id} with ${child.remaining:.2f}"
        if not self._wait(run, None):  # stopped: the run ends after the call it's making
            run.cancel()
        outcome = run.outcome()
        self._deliver([run])
        return result_block(id, profile, outcome)

    @tool
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
                are stopped too. Its result then says status="stopped" reason="cancelled". Pass wait as well to
                receive that result in this call; otherwise it arrives like any finished background run. A
                cancelled run can still be resumed later.

        Returns:
            Finished: the same <subagent-result …> block spawn would have returned, for the run's latest task;
            asking again returns it again. Still running: the progress line.
        """
        with self._lock:
            entry = self.runs.get(run_id)
            ids = ", ".join(map(str, self.runs)) or "none"
        if entry is None:
            raise ValueError(f"unknown run {run_id}; runs: {ids}")
        profile, run = entry
        if cancel:
            run.cancel()
        self._wait(run, min(max(wait, 0), MAX_WAIT))
        outcome = run.outcome(0)
        if outcome is None:
            return (
                f"run {run_id} is still running: {len(run.agent.node.entries)} model calls, "
                f"${run.agent.node.spent:.4f} spent so far"
            )
        self.claim(run_id)
        return result_block(run_id, profile, outcome)

    @tool
    def resume(self, run_id: int, message: str, background: bool = False, max_cost: float | None = None) -> str:
        """Send a subagent you started with spawn a follow-up: it carries on with its whole conversation so far
        (everything it read, did and answered), and `message` becomes its next user message.

        Args:
            run_id: The number spawn gave the run. It keeps that number: the follow-up's result comes back under it.
            message: Its next user message. It already has everything from its earlier task, so say only what's
                new: what to fix, what to add, or to carry on. This works however the run ended, including a run
                that was cancelled, failed or ran out of budget.
            background: As for spawn. False (the default) blocks until the follow-up finishes and returns its
                result; True returns at once, and the result arrives like any background run's.
            max_cost: Extra USD for this follow-up, on top of what the subagent has spent so far. Must be positive.
                If omitted, the profile's default is used; a value above the profile's cap is lowered to the cap,
                and it can't exceed what you have left yourself.

        Returns:
            Foreground: the same <subagent-result run="N" …> block as spawn, for the follow-up. Background:
            "resumed run N with $X". An error, and nothing started, if the run is still running (wait for it with
            subagent first) or has a background result you haven't read yet (read it with subagent first).
        """  # noqa: D205 - a tool docstring is a prompt: rewording it changes what the model sees
        with self._lock:
            entry = self.runs.get(run_id)
            ids = ", ".join(map(str, self.runs)) or "none"
        if entry is None:
            raise ValueError(f"unknown run {run_id}; runs: {ids}")
        profile, run = entry
        if max_cost is not None and max_cost <= 0:
            raise ValueError(f"max_cost must be positive, got {max_cost}")
        if not run.done():
            raise ValueError(f"run {run_id} is still running; wait for it with subagent({run_id}, wait=…) first")
        with self._lock:
            unread = run_id in self._pending_ids()
        if unread:  # resuming would replace a result the model never saw
            raise ValueError(f"run {run_id} has a result you haven't read; read it with subagent({run_id}) first")
        if (running := self.running()) >= MAX_RUNNING:
            raise ValueError(
                f"{running} subagents are running, the most at once; collect one with subagent(run_id, wait=…) first"
            )
        spec = self.profiles[profile]
        node = run.agent.node
        # Before the new thread starts, so its first budget check sees the new limit; every ancestor still caps it
        node.limit = node.spent + min(spec.max_cost if max_cost is None else max_cost, spec.cap)
        new = run.resume(FOLLOW_UP + message)
        with self._lock:
            self.runs[run_id] = (profile, new)
            if background:
                self._background.add(new)
        if background:
            return f"resumed run {run_id} with ${node.remaining:.2f}"
        if not self._wait(new, None):  # stopped: the run ends after the call it's making
            new.cancel()
        outcome = new.outcome()
        self._deliver([new])
        return result_block(run_id, profile, outcome)

    def _wait(self, run: Run, timeout: float | None) -> bool:
        """Waits until the run is done, the parent is stopped, or `timeout` passes.

        Args:
            run: The run to wait for.
            timeout: Seconds to wait; None for no timeout.

        Returns:
            Whether the run is done.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not run.done() and not self.node.cancelled:
            left = POLL_SECONDS if deadline is None else min(POLL_SECONDS, deadline - time.monotonic())
            if left <= 0:
                break
            run.wait(left)
        return run.done()

    # ---------------------------------------------------------------- management

    def claim(self, run_id: int | None = None) -> list[tuple[int, str, Run]]:
        """Takes the pending background results, in id order, and marks them delivered.

        Each run is returned by one claim at most; reading a result again with `subagent` needs none.

        Args:
            run_id: Take only this run's result; None takes all.

        Returns:
            `(id, profile name, run)` for each result taken.
        """
        with self._lock:
            ids = self._pending_ids() if run_id is None else [i for i in self._pending_ids() if i == run_id]
            self._delivered.update(self.runs[i][1] for i in ids)
            claimed = [(i, *self.runs[i]) for i in ids]
        self._deliver([run for _, _, run in claimed])
        return claimed

    def _deliver(self, runs: list[Run]) -> None:
        """Emits `Delivered` for each run whose result the model now receives."""
        for run in runs:
            self.node.emit(Delivered(path=run.path))

    @property
    def pending(self) -> int:
        """Finished background results the model hasn't received yet."""
        with self._lock:
            return len(self._pending_ids())

    def _pending_ids(self) -> list[int]:
        """Returns the ids of finished background runs not yet delivered, in id order.

        The caller holds the lock.
        """
        # Derived from the runs' futures, not recorded by a callback: a future wakes its waiters before it runs its
        # callbacks, so a result read in between would be delivered twice.
        return sorted(
            i
            for i, (_, run) in self.runs.items()
            if run in self._background and run not in self._delivered and run.done()
        )

    def running(self) -> int:
        """Returns how many runs haven't finished."""
        return sum(not run.done() for _, _, run in self.snapshot())

    def snapshot(self) -> list[tuple[int, str, Run]]:
        """Returns every run so far, as `(id, profile name, run)`, in id order."""
        with self._lock:
            return [(i, name, run) for i, (name, run) in self.runs.items()]

    def close(self) -> None:
        """Cancels every run, then waits for all of them.

        Each stops after the model call it is making.
        """
        runs = [run for _, _, run in self.snapshot()]
        for run in runs:
            run.cancel()
        for run in runs:
            run.wait()
