"""The tool-calling agent.

`Agent` runs turns: it calls the model, and the tools the model asks for, until the model answers. `TurnStopped`
ends a turn early. `estimate_tokens` and `message_tokens` estimate prompt sizes.
"""

import itertools
import json
import logging
from collections.abc import Callable
from pathlib import Path

from pydantic import validate_call

from jean_code.core.compaction import (
    HEAD_MARKER,
    SUMMARY_MESSAGE,
    head_end,
    middle,
    summary_failure,
    summary_prompt,
    turn_starts,
)
from jean_code.core.events import (
    AgentAdded,
    Compaction,
    Configured,
    Delta,
    MessageAdded,
    Settings,
    TurnEnd,
    TurnEnded,
    TurnStarted,
)
from jean_code.core.llm import (
    AssistantMessage,
    ChatBackend,
    ChatRequest,
    Completion,
    Message,
    StreamingBackend,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
    openrouter,
    render_message,
)
from jean_code.core.node import AgentNode
from jean_code.core.skills import Skills
from jean_code.core.subagents import Profile, Subagents, result_block, subagents_prompt
from jean_code.core.tool_schema import is_tool, tool_schema
from jean_code.core.usage import ModelCall

log = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
CONTEXT_WINDOW = 1_048_576  # tokens, for DEFAULT_MODEL (OpenRouter's /models)
HARD_LIMIT = 0.9  # a turn stops before its context passes this share of the window
NOT_RUN = "Not run: the turn was stopped."  # the result of a call the turn ended before running
LIVE_SETTINGS = ("model", "reasoning", "compact_at")  # what `configure` may change; the rest is fixed at construction
REASONING = (None, "low", "medium", "high")


class TurnStopped(Exception):
    """A turn was stopped before the model answered.

    Args:
        reason: How the turn ended; kept as `reason`.
    """

    def __init__(self, reason: TurnEnd):
        super().__init__(reason)
        self.reason: TurnEnd = reason


def parse_arguments(arguments: str) -> dict:
    """Parses a tool call's arguments, which the model sends as a JSON string.

    Args:
        arguments: The JSON text; empty means no arguments.

    Returns:
        The arguments by name.

    Raises:
        ValueError: `arguments` isn't a JSON object.
    """
    value = json.loads(arguments or "{}")
    if not isinstance(value, dict):
        raise ValueError(f"arguments must be a JSON object, got {arguments!r}")
    return value


def estimate_tokens(value) -> int:
    """Returns a high estimate of the tokens `value` takes in a prompt.

    The estimate is the UTF-8 bytes of its JSON, divided by 3: bytes rather than characters, so text in scripts
    like Chinese isn't underestimated.

    Args:
        value: Anything `json.dumps` accepts.
    """
    return -(-len(json.dumps(value, ensure_ascii=False).encode()) // 3)


def message_tokens(message: Message) -> int:
    """Returns a high estimate of the tokens `message` takes in a prompt, as sent.

    Args:
        message: The message to measure.
    """
    return estimate_tokens(render_message(message))


class Agent:
    """A tool-calling agent: a conversation with a model, and the tools it may call.

    Drive it with `run_turn(text)`, which returns how the turn ended: "answered", or why it stopped early. Call
    `close()` at the end. Everything the agent does is also emitted as an `AgentEvent` through its node. The
    arguments it keeps are attributes of the same name, except the settings it reads on every request, which are
    `settings`.

    Args:
        model: OpenRouter model id.
        system_prompt: Goes first; the skills and subagents listings are appended to it.
        tools: Functions marked `@tool`, by the name the model calls them.
        needs_approval: Names of the tools whose calls are run only if `approve` allows them.
        approve: Asked before each such call, on the turn's thread, with the tool's name and arguments; the turn
            waits for it. Returns None to run the call, or the text the model gets as its result instead.
        cleanup: Functions `close()` runs, to release what the tools hold. A subagent is closed after every turn
            and may be resumed, so what they release must come back on next use.
        reasoning: Effort: "low", "medium" or "high". None sends no reasoning setting.
        max_tool_rounds: A turn makes at most 1 + this many model calls.
        client: Where requests are sent. OpenRouter by default.
        stream: Stream every reply: a turn's reply is emitted piece by piece as `Delta`s, and a stop cuts off the
            reply in flight. `client` must then be a `StreamingBackend`.
        skills_dir: Folder of skills the model can load.
        max_cost: Limit in USD for this agent and its subagents. Not together with `node`.
        node: This agent's node in an existing agent tree. Subagents get one from their parent.
        subagent_profiles: The kinds of subagent the model may start, by name.
        compact_at: Context size in tokens at which the agent compacts after a turn. None: never on its own.
        keep_first: Turns at the start that a compaction keeps verbatim.
        keep_last: Turns at the end that a compaction keeps verbatim.
        context_window: The model's context window, in tokens.

    Attributes:
        settings: `model`, `reasoning`, `compact_at`, `max_tool_rounds`, `keep_first`, `keep_last` and
            `context_window`, as given or as `configure` last changed them.
        tool_map: The `tools` argument.
        skills: The skills the model can load, from `skills_dir`.
        commands: What the user runs by starting a message with `/name`: name → a function of the rest of the line.
        node: Its place in the agent tree: `node` if given, else the root of a new tree.
        subagents: The kinds of subagent it may start, and their runs; None if `subagent_profiles` is empty.
        messages: The transcript, oldest first, with each reply as the provider sent it.
        usage: One `ModelCall` per paid model call, in call order.
        compactions: One record per compaction attempt that got past the checks.
        last_request: The last request sent to the model; None before the first.

    Raises:
        ValueError: An argument is out of range; `max_cost` and `node` are both given; some tools need approval
            but there's no `approve`; `stream` is set but `client` can't stream; or a tool isn't marked `@tool`,
            or is named like one of the agent's own.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        tools: dict[str, Callable] | None = None,
        needs_approval: set[str] | None = None,
        approve: Callable[[str, dict], str | None] | None = None,
        cleanup: list[Callable[[], None]] | None = None,
        reasoning: str | None = "medium",
        max_tool_rounds: int = 10,
        client: ChatBackend | None = None,
        stream: bool = False,
        skills_dir: str | Path | None = None,
        max_cost: float | None = None,
        node: AgentNode | None = None,
        subagent_profiles: dict[str, Profile] | None = None,
        compact_at: int | None = 256_000,
        keep_first: int = 1,
        keep_last: int = 2,
        context_window: int = CONTEXT_WINDOW,
    ):
        self.system_prompt = system_prompt
        self.client: ChatBackend = client or openrouter()
        if stream and not isinstance(self.client, StreamingBackend):
            raise ValueError(f"stream is set, but the client ({type(self.client).__name__}) can't stream")
        self.stream = stream
        self.tool_map: dict[str, Callable] = dict(tools or {})
        self.needs_approval: set[str] = set(needs_approval or ())
        if self.needs_approval and approve is None:
            raise ValueError(
                f"no `approve` function for the tools that need approval: {', '.join(sorted(self.needs_approval))}"
            )
        self.approve = approve
        self.cleanup: list[Callable[[], None]] = list(cleanup or [])

        if max_cost is not None and max_cost <= 0:
            raise ValueError(f"max_cost must be positive, got {max_cost}")
        if max_cost is not None and node is not None:
            raise ValueError("pass max_cost or node, not both: a given node carries its own limit")
        self.node = node or AgentNode(max_cost)
        self.usage: list[ModelCall] = []

        self.skills = Skills(skills_dir, self.node)
        # Currently only skills included, optional second argument is never used.
        self.commands: dict[str, Callable[[str], str | None]] = self.skills.commands()

        listings = (self.system_prompt, self.skills.prompt(), subagents_prompt(subagent_profiles or {}))
        system = "\n\n".join(part for part in listings if part).strip()

        if compact_at is not None and compact_at > context_window // 2:
            raise ValueError(f"compact_at must leave room for the summary call: at most {context_window // 2}")
        if keep_first < 0 or keep_last < 1:
            raise ValueError("keep_first must be >= 0 and keep_last >= 1")
        self._settings = Settings(
            model=model,
            reasoning=reasoning,
            compact_at=compact_at,
            max_tool_rounds=max_tool_rounds,
            keep_first=keep_first,
            keep_last=keep_last,
            context_window=context_window,
        )
        self.compactions: list[dict] = []
        self._measured: tuple[int, int] | None = None  # (len(messages), tokens) after the last measured call

        self.last_request: ChatRequest | None = None
        # Runs exist only once the model calls spawn.
        self.subagents = Subagents(subagent_profiles, self.node) if subagent_profiles else None
        # Every tool the model can call, as (validating wrapper, schema) by name. Should never be changed: the schemas
        # sit near the top of the prompt, so any change would invalidate the cache.
        self._tools: dict[str, tuple[Callable, dict]] = {
            name: (validate_call(fn), tool_schema(name, fn)) for name, fn in self._all_tools().items()
        }

        # Last: the system message's event carries the context size, which needs the tool schemas above.
        self.node.agent = self
        self.node.emit(
            AgentAdded(
                path=self.node.path,
                profile=self.node.profile,
                task=self.node.task,
                settings=self._settings,
                limit=self.node.limit,
                tools=self._tool_schemas(),
                needs_approval=sorted(self.needs_approval),
                skills=list(self.skills.catalog),
            )
        )
        self.messages: tuple[Message, ...] = ()
        if system:
            self._append(SystemMessage(content=system))

    @property
    def settings(self) -> Settings:
        """What the agent reads on every request."""
        return self._settings

    def configure(self, **settings) -> None:
        """Changes settings the agent reads on every request, and emits `Configured`.

        All are checked before any applies. A change to `model` or `reasoning` invalidates the whole prompt cache, so
        the next request pays the uncached price for every token of the prompt.

        Args:
            **settings: Any of `model`, `reasoning` ("low", "medium", "high" or None) and `compact_at` (tokens, at
                most half the context window, or None).

        Raises:
            ValueError: A setting is unknown, or its value is one the agent would refuse.
        """
        if unknown := settings.keys() - set(LIVE_SETTINGS):
            raise ValueError(f"not a live setting: {', '.join(sorted(unknown))}")
        if settings.get("reasoning") not in REASONING:
            raise ValueError(f"reasoning must be one of {REASONING}, got {settings['reasoning']!r}")
        window = self._settings.context_window
        if (at := settings.get("compact_at")) is not None and at > window // 2:
            raise ValueError(f"compact_at must leave room for the summary call: at most {window // 2}")
        self._settings = self._settings.model_copy(update=settings)
        self.node.emit(Configured(path=self.node.path, settings=self._settings))

    @property
    def total_cost(self) -> float:
        """USD this agent has spent on model calls."""
        return sum(call.cost or 0 for call in self.usage)

    @property
    def max_cost(self) -> float | None:
        """This agent's limit in USD, which is its node's."""
        return self.node.limit

    @property
    def over_budget(self) -> bool:
        """Whether this agent's node, or an ancestor's, has spent its limit."""
        return self.node.exhausted

    @property
    def context_tokens(self) -> int:
        """Tokens the next request's prompt will take.

        It's the last call's prompt and completion as measured, plus an estimate for the messages added since.
        """
        measured, messages = self._measured, self.messages  # read once: another thread may be running the turn
        if measured is None:  # no call yet: estimate the whole transcript and the tool schemas
            return estimate_tokens(self._tool_schemas()) + sum(message_tokens(m) for m in messages)
        count, tokens = measured
        return tokens + sum(message_tokens(m) for m in messages[count:])

    @property
    def context_history(self) -> list[tuple[Message, ...]]:
        """Every context the agent has had: one per compaction, then the current one."""
        return [c["before"] for c in self.compactions if c["status"] == "compacted"] + [self.messages]

    def interrupt(self):
        """Stops this agent and every agent below it before their next model call.

        A reply that has arrived still has its tool calls run to the end; a streamed reply still arriving is cut
        off, keeping its text and running none of its tool calls. The turn then ends "cancelled". The agent stays
        stopped, also for its next turn, until its node is reopened. Safe to call from another thread or a signal
        handler.
        """
        self.node.cancel()

    def close(self) -> None:
        """Stops every subagent run and waits for them, then runs `cleanup`.

        `cleanup` releases what the tools hold, such as background jobs and containers.
        """
        if self.subagents:  # first: no run outlives its parent
            self.subagents.close()
        for fn in self.cleanup:
            fn()

    def run_turn(self, user_input: str) -> TurnEnd:
        """Runs one turn on `user_input`, until the model answers or the turn stops.

        Adds the user's message, with a command's output if it starts with `/name` and any background subagent
        results not yet delivered, then calls the model until it answers. Emits `TurnStarted`, then exactly one
        `TurnEnded`, also when the turn raises.

        Args:
            user_input: The user's message.

        Returns:
            "answered", or why the turn stopped early.

        Raises:
            MalformedResponse: A model response couldn't be used.
        """
        self.node.emit(TurnStarted(path=self.node.path, text=user_input))
        end: TurnEnd | None = None
        error: str | None = None
        try:
            end = self._turn(user_input)
            return end
        except BaseException as e:  # like Run._finish: whatever stopped the turn, its end is reported
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self.node.emit(TurnEnded(path=self.node.path, end=end, error=error, context_tokens=self.context_tokens))

    def _turn(self, user_input: str) -> TurnEnd:
        content = self._run_command(user_input)
        claimed = self.subagents.claim() if self.subagents else []  # each result is taken, so delivered, once
        if claimed:
            blocks = "\n\n".join(result_block(id, profile, run.outcome()) for id, profile, run in claimed)
            content = f"{content}\n\n{blocks}" if content else blocks
        if head_end(self.messages) is None and len(turn_starts(self.messages)) == self.settings.keep_first:
            self._append(UserMessage(content=HEAD_MARKER))
        self._append(UserMessage(content=content))
        try:
            self._loop()
        except TurnStopped as stop:
            return stop.reason
        return "answered"

    def step(self) -> None:
        """Makes one model call on the current context, then runs the reply's tool calls.

        The reply and each tool result are appended to `messages`.

        Raises:
            MalformedResponse: The response couldn't be used.
            TurnStopped: The agent was stopped while the reply streamed in, or `approve` stopped the turn.
        """
        completion = self._call(self._build_request())
        if completion.finish_reason is None and self.node.cancelled:  # cut off mid-stream: its tool calls never run
            message = completion.message.model_copy(update={"tool_calls": ()})
            if message.content or message.reasoning:
                self._append(message)
            raise TurnStopped("cancelled")
        usage = completion.usage
        # The prompt plus the reply: measured before the append, so the reply's event carries the measured size.
        if usage is not None and usage.prompt_tokens is not None:
            self._measured = (len(self.messages) + 1, usage.prompt_tokens + (usage.completion_tokens or 0))
        self._append(completion.message)
        self._run_tool_calls(completion.message.tool_calls)

    def compact(self, force: bool = False) -> str:
        """Replaces the middle of the transcript with a summary the model writes.

        The first `keep_first` and last `keep_last` turns are kept as they are. Call it between turns;
        `interrupt()` stops it.

        Args:
            force: Compact even if the context is under `compact_at`, or if compacting couldn't bring it under.

        Returns:
            "compacted", or why the transcript was left unchanged.
        """
        if self.over_budget:
            return "refused: the budget is spent"
        tokens_before = self.context_tokens
        compact_at = self.settings.compact_at
        if not force and (compact_at is None or tokens_before < compact_at):
            return "below threshold"
        messages = self.messages
        region = middle(messages, head_end(messages), self.settings.keep_last)
        if region is None:
            return "nothing to compact"
        start, end = region
        record = {"status": None, "removed": end - start, "tokens_before": tokens_before}
        self.compactions.append(record)
        try:
            return self._compact(record, messages, region, force)
        finally:
            if record["status"]:  # a raise past the summary call leaves no status, and no event
                self.node.emit(
                    Compaction(
                        path=self.node.path,
                        status=record["status"],
                        start=start,
                        removed=record["removed"],
                        summary=record.get("summary"),
                        tokens_before=tokens_before,
                        tokens_after=record.get("tokens_after"),
                        cost=record["call"].cost if record.get("call") else None,
                        context_tokens=self.context_tokens,
                    )
                )

    def _compact(self, record: dict, messages: tuple[Message, ...], region: tuple[int, int], force: bool) -> str:
        """Runs a recorded compaction attempt; sets `record["status"]` and returns it."""
        start, end = region
        head, replaced, tail = messages[:start], messages[start:end], messages[end:]
        tokens_before = record["tokens_before"]

        # Check that compacting can bring the context under compact_at.
        schemas = estimate_tokens(self._tool_schemas())
        kept_estimate = schemas + sum(message_tokens(m) for m in head + tail)
        kept = round(tokens_before * kept_estimate / (kept_estimate + sum(message_tokens(m) for m in replaced)))
        compact_at = self.settings.compact_at
        if not force and compact_at is not None and kept >= compact_at:
            record["status"] = f"skipped: the kept turns alone take ~{kept} tokens, over compact_at"
            return record["status"]

        request = self._build_request(UserMessage(content=summary_prompt(replaced, tail, self.settings.keep_last)))
        # No `tool_choice: "none"`: DeepSeek then renders the prompt without tools, which drops every reasoning
        # block and misses the cache (.experiments/output/compaction_check_run1_tool_choice_none.txt).
        # A reply with tool calls counts as a failure instead.
        record["request"] = request
        booked = len(self.usage)
        try:
            completion = self._call(request, kind="compaction")
        except TurnStopped:  # stopped before the summary's first chunk
            record["status"] = "failed: interrupted"
            return record["status"]
        except Exception as e:  # noqa: BLE001 - a failed summary leaves the transcript as it was
            record["status"] = f"failed: {type(e).__name__}: {e}"
            return record["status"]
        finally:
            record["call"] = self.usage[-1] if len(self.usage) > booked else None  # booked even when it failed
        if failure := summary_failure(completion, self.node.cancelled):
            record["status"] = failure
            return failure

        summary = completion.message.content.strip()
        record["summary"] = UserMessage(content=SUMMARY_MESSAGE.format(summary=summary))
        record["before"] = messages  # the old tuple itself: it's replaced, never modified
        self.messages = (*head, record["summary"], *tail)
        # The old measurement described the longer list. Scale it by the estimates instead of taking the raw
        # estimate, which runs high; the next call measures the real size.
        new_estimate = kept_estimate + message_tokens(record["summary"])
        old_estimate = kept_estimate + sum(message_tokens(m) for m in replaced)
        self._measured = (len(self.messages), round(tokens_before * new_estimate / old_estimate))
        record["tokens_after"] = self.context_tokens
        record["status"] = "compacted"
        return record["status"]

    def _call(self, request: ChatRequest, kind: str | None = None) -> Completion:
        """Calls the model; the only place the agent does.

        Books the call, and logs a warning if a reply that arrived whole reported no cost. A streamed reply is cut
        off once the agent is stopped, and comes back with no `finish_reason` or usage.

        Raises:
            MalformedResponse: The response couldn't be used. The call isn't booked.
            TurnStopped: The agent was stopped before the reply's first chunk. Nothing is booked.
        """
        self.last_request = request
        if self.stream and isinstance(self.client, StreamingBackend):  # __init__ refuses a stream without one
            completion = self._stream(self.client, request, show=kind is None)
        else:
            completion = self.client.complete(request)
        call = self._book(completion, kind)
        if call.cost is None and completion.finish_reason is not None:  # a cut-off stream never gets its usage
            log.warning("model call %s reported no cost, so spend can't be tracked", call.generation_id)
        return completion

    def _stream(self, client: StreamingBackend, request: ChatRequest, show: bool) -> Completion:
        """Streams the reply, emitting its pieces as `Delta`s if `show`.

        Raises:
            TurnStopped: The agent was stopped before the reply's first chunk.
        """
        completion = None
        for item in client.stream(request, stop=lambda: self.node.cancelled):
            if isinstance(item, Completion):
                completion = item
            elif show:
                self.node.emit(Delta(path=self.node.path, kind=item.kind, text=item.text))
        if completion is None:
            raise TurnStopped("cancelled")
        return completion

    def _book(self, completion: Completion, kind: str | None) -> ModelCall:
        """Records a paid call in `usage` and charges it to the node."""
        call = ModelCall.from_completion(completion, path=self.node.path, kind=kind)
        self.usage.append(call)
        self.node.charge(call)
        return call

    def _all_tools(self) -> dict[str, Callable]:
        """Returns `tool_map` plus the agent's own tools.

        Raises:
            ValueError: A tool isn't marked `@tool`, or is named like one of the agent's own, which it would
                otherwise silently replace.
        """
        own: dict[str, Callable] = {}
        own |= self.skills.tools()
        if self.subagents:
            own |= {
                "spawn": self.subagents.spawn,
                "subagent": self.subagents.subagent,
                "resume": self.subagents.resume,
            }
        if clash := sorted(self.tool_map.keys() & own.keys()):
            raise ValueError(f"tools named like the agent's own: {', '.join(clash)}")
        if unmarked := [name for name, fn in self.tool_map.items() if not is_tool(fn)]:
            raise ValueError(
                f"tools not marked @tool: {', '.join(unmarked)}; mark them, so it's clear where they're "
                "defined that a model sees their name, parameters and docstring"
            )
        return self.tool_map | own

    def _build_request(self, *extra: Message) -> ChatRequest:
        """Builds the request for the transcript, plus `extra` for this call only."""
        settings = self.settings
        return ChatRequest(settings.model, (*self.messages, *extra), tuple(self._tool_schemas()), settings.reasoning)

    def _tool_schemas(self) -> list[dict]:
        return [schema for _, schema in self._tools.values()]

    def _run_command(self, user_input: str) -> str:
        """Returns the message, with a command's output appended if it starts with one."""
        first, _, rest = user_input.strip().partition(" ")
        command = self.commands.get(first[1:]) if first.startswith("/") else None
        if command is None:  # not a command, or an unknown `/foo`: plain text
            return user_input
        output = command(rest.strip())
        if output is None:
            return user_input
        return f'{user_input}\n\n<command name="{first[1:]}">\n{output}\n</command>'

    def _loop(self) -> None:
        """Calls the model until it answers, running the tools it calls.

        Once the model has answered, compacts the context if it's over `compact_at`.

        Raises:
            TurnStopped: The turn has to end before the model answers.
        """
        for calls in itertools.count():
            if self.node.cancelled:  # before the budget: a cancelled one also counts as spent
                raise TurnStopped("cancelled")
            if self.over_budget:
                raise TurnStopped("budget spent")
            if self.context_tokens >= HARD_LIMIT * self.settings.context_window:
                raise TurnStopped("context full")
            if calls > self.settings.max_tool_rounds:
                raise TurnStopped("out of tool rounds")
            self.step()
            last = self.messages[-1]
            if isinstance(last, AssistantMessage) and not last.tool_calls:  # an answer, even if cut off or empty
                if not self.node.cancelled:  # no paid summary after a stop
                    self.compact()
                return

    def _run_tool_calls(self, calls: tuple[ToolCall, ...]) -> None:
        """Appends one result per call, in call order.

        If a call raises, the calls left get `NOT_RUN`: a request can't hold a call without a result.

        Raises:
            TurnStopped: `approve` stopped the turn.
        """
        done = 0
        try:
            # No stop check here: a stop lets the reply's calls finish (a researcher's last record_fact lands).
            for call in calls:
                self._add_result(call, self._run_tool(call))
                done += 1
        finally:
            for call in calls[done:]:
                self._add_result(call, NOT_RUN)

    def _add_result(self, call: ToolCall, content: str) -> None:
        self._append(ToolMessage(tool_call_id=call.id, content=content))

    def _append(self, message: Message) -> None:
        """Appends `message` to `messages` and emits it as `MessageAdded`.

        Use it for every append, so the events stay enough to rebuild the transcript.
        """
        self.messages = (*self.messages, message)
        self.node.emit(MessageAdded(path=self.node.path, message=message, context_tokens=self.context_tokens))

    def _run_tool(self, call: ToolCall) -> str:
        """Runs one tool call and returns its result.

        If the tool needs approval, `approve` may give the result instead. Errors (bad JSON, invalid arguments,
        failures) are returned as the result too, so the model can retry.

        Raises:
            RuntimeError: The tool needs approval but the agent has no `approve`.
        """
        name = call.function.name
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        fn, _ = tool
        try:
            arguments = parse_arguments(call.function.arguments)
        except ValueError as e:
            return f"Error: {type(e).__name__}: {e}"
        if name in self.needs_approval:
            if self.approve is None:  # needs_approval was changed after the agent was built
                raise RuntimeError(f"tool '{name}' needs approval, but the agent has no `approve` function")
            if (refusal := self.approve(name, arguments)) is not None:
                return refusal
        try:
            return str(fn(**arguments))
        except Exception as e:  # noqa: BLE001 - any tool failure goes back to the model as its result
            return f"Error: {type(e).__name__}: {e}"
