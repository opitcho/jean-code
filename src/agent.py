import itertools
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import validate_call

from llm import ChatBackend, is_tool, openrouter, tool_schema
from prompts import COMPACTION_PROMPT, EARLIER_SUMMARY_NOTE, HEAD_MARKER, HEAD_MARKER_TAG, SUMMARY_MESSAGE, SUMMARY_TAG
from skills import Skills
from subagents import Profile, Subagents, result_block, subagents_prompt
from usage import Budget

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
CONTEXT_WINDOW = 1_048_576  # tokens, for DEFAULT_MODEL (OpenRouter's /models)
HARD_LIMIT = 0.9  # a turn stops before its context passes this share of the window
# How a turn ended. The values are the text shown for it, and keys once stored: don't reword them.
TurnEnd = Literal["answered", "interrupted", "budget spent", "cancelled", "context full", "out of tool rounds"]
NOT_RUN = "Not run: the turn was stopped."  # the result of a call the turn ended before running


class TurnStopped(Exception):
    """Raised anywhere inside a turn to end it before the model answers; `run_turn` returns its `reason`."""

    def __init__(self, reason: TurnEnd):
        super().__init__(reason)
        self.reason: TurnEnd = reason


def parse_arguments(arguments: str) -> dict:
    """A tool call's arguments, which the model sends as a JSON string. Raises ValueError if not a JSON object."""
    value = json.loads(arguments or "{}")
    if not isinstance(value, dict):
        raise ValueError(f"arguments must be a JSON object, got {arguments!r}")
    return value


def estimate_tokens(value) -> int:
    """A high estimate of the tokens `value` takes in a prompt: the UTF-8 bytes of its JSON, divided by 3.

    Bytes rather than characters, so text in scripts like Chinese isn't underestimated.
    """
    return -(-len(json.dumps(value, ensure_ascii=False).encode()) // 3)


def is_synthetic(message: dict) -> bool:
    """True for the user messages the agent writes itself: the head marker and summaries."""
    content = message.get("content")
    return message["role"] == "user" and isinstance(content, str) and content.startswith((HEAD_MARKER_TAG, SUMMARY_TAG))


def ordinal(n: int) -> str:
    return "last" if n == 1 else f"{n}{ {2: 'nd', 3: 'rd'}.get(n, 'th')}-to-last"


class Agent:
    """A tool-calling agent: one conversation with a model, and the tools the model may call.

    Drive it with `run_turn(text)`. It returns how the turn ended (a `TurnEnd`): "answered", or why it stopped
    early. Call `close()` at the end.

    Callers read its state directly. `messages` is the transcript in the OpenAI chat format, with each reply stored
    as the API returned it. `usage` has one entry per model call, and `budget` holds the spend.

    Args:
        model: OpenRouter model id.
        system_prompt: Goes first; the skills and subagents listings are appended to it.
        tools: Functions marked `@tool` (llm.py), by the name the model calls them.
        needs_approval: Names of the tools whose calls are run only if `approve` allows them.
        approve: Asked before each such call, on the turn's thread, with the tool's name and arguments; the turn
            waits for it. Returns None to run the call, or the text the model gets as its result instead.
        cleanup: Functions `close()` runs, to release what the tools hold.
        reasoning: Effort: "low", "medium" or "high". None sends no reasoning setting.
        max_tool_rounds: A turn makes at most 1 + this many model calls.
        client: Where requests are sent. OpenRouter by default.
        skills_dir: Folder of skills the model can load
        max_cost: Limit in USD for this agent and its subagents. Not together with `budget`.
        budget: This agent's node in an existing budget tree (usage.py). Subagents get one from their parent.
        subagent_profiles: The kinds of subagent the model may start, by name (subagents.py).
        compact_at: Context size in tokens at which the agent compacts after a turn. None: never on its own.
        keep_first: Turns at the start that a compaction keeps verbatim.
        keep_last: Turns at the end that a compaction keeps verbatim.
        context_window: The model's context window, in tokens.
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
        skills_dir: str | Path | None = None,
        max_cost: float | None = None,
        budget: Budget | None = None,
        subagent_profiles: dict[str, Profile] | None = None,
        compact_at: int | None = 256_000,
        keep_first: int = 1,
        keep_last: int = 2,
        context_window: int = CONTEXT_WINDOW,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.reasoning = reasoning
        self.max_tool_rounds = max_tool_rounds
        self.client: ChatBackend = client or openrouter()
        self.tool_map: dict[str, Callable] = dict(tools or {})
        self.needs_approval: set[str] = set(needs_approval or ())
        if self.needs_approval and approve is None:
            raise ValueError(f"no `approve` function for the tools that need approval: "
                             f"{', '.join(sorted(self.needs_approval))}")
        self.approve = approve
        self.cleanup: list[Callable[[], None]] = list(cleanup or [])

        self.skills = Skills(skills_dir)
        # What the user runs by starting a message with `/name`: name → a function of the rest of the line.
        # Currently only skills included, optional second argument is never used.
        self.commands: dict[str, Callable[[str], str | None]] = self.skills.commands()

        self.messages: list[dict] = []
        listings = (self.system_prompt, self.skills.prompt(), subagents_prompt(subagent_profiles or {}))
        system = "\n\n".join(part for part in listings if part).strip()
        if system:
            self.messages.append({"role": "system", "content": system})

        if max_cost is not None and max_cost <= 0:
            raise ValueError(f"max_cost must be positive, got {max_cost}")
        if max_cost is not None and budget is not None:
            raise ValueError("pass max_cost or budget, not both: a given budget carries its own limit")
        self.budget = budget or Budget(max_cost)  # this agent's node; a new root unless a parent's tree is given
        self.usage: list[dict] = []  # per model call: the API's usage, plus the response's id, provider, model, created

        if compact_at is not None and compact_at > context_window // 2:
            raise ValueError(f"compact_at must leave room for the summary call: at most {context_window // 2}")
        if keep_first < 0 or keep_last < 1:
            raise ValueError("keep_first must be >= 0 and keep_last >= 1")
        self.compact_at = compact_at
        self.keep_first = keep_first
        self.keep_last = keep_last
        self.context_window = context_window
        self.head_end: int | None = None  # index just after the head marker; nothing before it is ever removed
        self.compactions: list[dict] = []  # one record per compaction attempt that got past the checks
        self._measured: tuple[int, int] | None = None  # (len(messages), tokens) after the last measured call

        self.last_request: dict | None = None
        self.last_response: dict | None = None  # the last response: id, provider, choices, usage
        self._interrupt = threading.Event()
        # The kinds of subagent it may start; runs exist only once the model calls spawn. Waits watch _interrupt.
        self.subagents = Subagents(subagent_profiles, self.budget, self._interrupt) if subagent_profiles else None
        # Every tool the model can call, as (validating wrapper, schema) by name. Should never be changed: the schemas
        # sit near the top of the prompt, so any change would invalidate the cache.
        self._tools: dict[str, tuple[Callable, dict]] = {
            name: (validate_call(fn), tool_schema(name, fn)) for name, fn in self._all_tools().items()
        }

    @property
    def total_cost(self) -> float:
        """USD spent on model calls this session."""
        return sum(entry.get("cost") or 0 for entry in self.usage)

    @property
    def max_cost(self) -> float | None:
        """This agent's limit in USD: its budget node's. Settable, e.g. to raise it mid-session."""
        return self.budget.limit

    @max_cost.setter
    def max_cost(self, value: float | None) -> None:
        self.budget.limit = value

    @property
    def over_budget(self) -> bool:
        """True once this agent's budget node, or any ancestor's, has spent its limit."""
        return self.budget.exhausted

    @property
    def context_tokens(self) -> int:
        """Tokens the next request's prompt will take: the last call's prompt and completion, plus an estimate
        for the messages added since."""
        measured, messages = self._measured, self.messages  # read once: another thread may be running the turn
        if measured is None:  # no call yet: estimate the whole transcript and the tool schemas
            return estimate_tokens(self._tool_schemas()) + sum(estimate_tokens(m) for m in messages)
        count, tokens = measured
        return tokens + sum(estimate_tokens(m) for m in messages[count:])

    @property
    def context_history(self) -> list[list[dict]]:
        """Every context the session has had: the one before each compaction, then the current one."""
        return [c["before"] for c in self.compactions if c["status"] == "compacted"] + [self.messages]

    def interrupt(self):
        """Stop the turn once the model call or tool call in flight returns. The reply's tool calls that haven't
        started are not run.

        Safe to call from another thread or a signal handler.
        """
        self._interrupt.set()

    def close(self) -> None:
        """Stop every subagent run and wait for them, then release what the tools hold (background jobs,
        containers) by running `cleanup`."""
        if self.subagents:  # first: no run outlives its parent
            self.subagents.close()
        for fn in self.cleanup:
            fn()

    def run_turn(self, user_input: str) -> TurnEnd:
        """Add the user's message (with a command's output, if it starts with one, and any background subagent
        results nobody has read), then call the model until it answers. Returns how the turn ended."""
        content = self._run_command(user_input)
        claimed = self.subagents.claim() if self.subagents else []  # each result is taken, so delivered, once
        if claimed:
            blocks = "\n\n".join(result_block(id, profile, run.outcome()) for id, profile, run in claimed)
            content = f"{content}\n\n{blocks}" if content else blocks
        self._interrupt.clear()
        if self.head_end is None and len(self._turn_starts()) == self.keep_first:
            self.messages.append({"role": "user", "content": HEAD_MARKER})
            self.head_end = len(self.messages)
        self.messages.append({"role": "user", "content": content})
        try:
            self._loop()
        except TurnStopped as stop:
            return stop.reason
        return "answered"

    def step(self) -> None:
        """One model call on the current context: append its reply, then run its tool calls."""
        request = self._build_request()
        self.last_request = request
        self.last_response = None
        response = self.client.complete(request)
        self.last_response = response
        usage = self._record_usage(response)
        message = response["choices"][0]["message"]
        reply = {k: v for k, v in message.items() if v is not None}  # drop unset fields like `refusal: null`
        reply.setdefault("content", "")
        self.messages.append(reply)
        if usage.get("prompt_tokens") is not None:  # the prompt plus the reply just appended
            self._measured = (len(self.messages), usage["prompt_tokens"] + (usage.get("completion_tokens") or 0))
        self._run_tool_calls(reply.get("tool_calls", []))

    def compact(self, force: bool = False) -> str:
        """Replace the middle of the conversation with a summary written by the model. The first `keep_first` and
        last `keep_last` turns are kept as they are. Call it between turns.

        Does nothing if the context is under `compact_at`, or if compacting couldn't bring it under; `force` skips
        both checks. Returns "compacted", or why the transcript was left unchanged. `interrupt()` cancels it.
        """
        self._interrupt.clear()
        if self.over_budget:
            return "refused: the budget is spent"
        tokens_before = self.context_tokens
        if not force and (self.compact_at is None or tokens_before < self.compact_at):
            return "below threshold"
        region = self._middle()
        if region is None:
            return "nothing to compact"
        messages = self.messages
        start, end = region
        head, middle, tail = messages[:start], messages[start:end], messages[end:]
        record = {"status": None, "removed": len(middle), "tokens_before": tokens_before}
        self.compactions.append(record)

        # Check that compacting can bring the context under compact_at.
        schemas = estimate_tokens(self._tool_schemas())
        kept_estimate = schemas + sum(estimate_tokens(m) for m in head + tail)
        kept = round(tokens_before * kept_estimate / (kept_estimate + sum(estimate_tokens(m) for m in middle)))
        if not force and self.compact_at is not None and kept >= self.compact_at:
            record["status"] = f"skipped: the kept turns alone take ~{kept} tokens, over compact_at"
            return record["status"]

        request = self._build_request()
        request["messages"].append({"role": "user", "content": COMPACTION_PROMPT.format(
            head_tag=HEAD_MARKER_TAG,
            ordinal=ordinal(self.keep_last),
            tail_quote=" ".join(tail[0]["content"].split())[:80],
            earlier_summary=EARLIER_SUMMARY_NOTE if is_synthetic(middle[0]) else "",
        )})
        # No `tool_choice: "none"`: DeepSeek then renders the prompt without tools, which drops every reasoning
        # block and misses the cache (.experiments/output/compaction_check_run1_tool_choice_none.txt).
        # A reply with tool calls counts as a failure instead.
        record["request"] = request
        try:
            response = self.client.complete(request)
            record["response"] = response
            self._record_usage(response, kind="compaction")
            choice = response["choices"][0]
            summary = (choice["message"].get("content") or "").strip()
        except Exception as e:  # noqa: BLE001 - a failed summary leaves the transcript as it was
            record["status"] = f"failed: {type(e).__name__}: {e}"
            return record["status"]
        if choice["message"].get("tool_calls"):
            record["status"] = "failed: the model called a tool instead of summarizing"
        elif choice.get("finish_reason") == "length":
            record["status"] = "failed: the summary was cut off at the length limit"
        elif not summary:
            record["status"] = "failed: the summary was empty"
        elif self._interrupt.is_set():
            record["status"] = "failed: interrupted"
        if record["status"]:
            return record["status"]

        record["summary"] = {"role": "user", "content": SUMMARY_MESSAGE.format(summary=summary)}
        record["before"] = messages  # the old list itself: it's replaced, never modified
        self.messages = [*head, record["summary"], *tail]
        # The old measurement described the longer list. Scale it by the estimates instead of taking the raw
        # estimate, which runs high; the next call measures the real size.
        new_estimate = kept_estimate + estimate_tokens(record["summary"])
        old_estimate = kept_estimate + sum(estimate_tokens(m) for m in middle)
        self._measured = (len(self.messages), round(tokens_before * new_estimate / old_estimate))
        record["tokens_after"] = self.context_tokens
        record["status"] = "compacted"
        return record["status"]

    def _turn_starts(self) -> list[int]:
        """Indices of the user's own messages, each of which starts a turn."""
        return [i for i, m in enumerate(self.messages) if m["role"] == "user" and not is_synthetic(m)]

    def _middle(self) -> tuple[int, int] | None:
        """(start, end) of the messages a compaction would replace, or None if they hold no turn of the user's."""
        starts = self._turn_starts()
        if self.head_end is None or len(starts) < self.keep_last:
            return None
        end = starts[-self.keep_last]
        if not any(self.head_end <= i < end for i in starts):
            return None  # nothing new: at most an earlier summary
        return self.head_end, end

    def _record_usage(self, response: dict, kind: str | None = None) -> dict:
        """Append the response's usage to `self.usage`, charge it to `self.budget`, and return it (empty if the
        response had none).

        Called before the reply is read, so a paid call counts even if its reply is malformed.
        Without usage the entry still marks the call, with `cost: None`.

        Raises:
            RuntimeError: The response reported no cost, so spend can't be tracked. The entry is recorded first.
        """
        usage = response.get("usage") or {}
        entry = {"cost": None, **usage, **{k: response.get(k) for k in ("id", "provider", "model", "created")}}
        if kind:
            entry["kind"] = kind
        self.usage.append(entry)
        self.budget.charge(entry)
        if entry["cost"] is None:
            raise RuntimeError(f"model call {entry['id']} reported no cost, so spend can't be tracked")
        return usage

    def _all_tools(self) -> dict[str, Callable]:
        """`tool_map` plus the agent's own tools. Raises on a tool not marked `@tool`, and on one named like one of
        the agent's, which would otherwise silently replace it."""
        own: dict[str, Callable] = {}
        own |= self.skills.tools()
        if self.subagents:
            own |= {"spawn": self.subagents.spawn, "subagent": self.subagents.subagent}
        if clash := sorted(self.tool_map.keys() & own.keys()):
            raise ValueError(f"tools named like the agent's own: {', '.join(clash)}")
        if unmarked := [name for name, fn in self.tool_map.items() if not is_tool(fn)]:
            raise ValueError(f"tools not marked @tool: {', '.join(unmarked)}; mark them, so it's clear where they're "
                             "defined that a model sees their name, parameters and docstring")
        return self.tool_map | own

    def _build_request(self) -> dict:
        """The chat request for the current transcript."""
        request: dict = {
            "model": self.model,
            "messages": list(self.messages),  # a copy, so last_request doesn't grow with the transcript
        }
        if self.reasoning:
            request["reasoning"] = {"effort": self.reasoning}
        tools = self._tool_schemas()
        if tools:
            request["tools"] = tools
        return request

    def _tool_schemas(self) -> list[dict]:
        return [schema for _, schema in self._tools.values()]

    def _run_command(self, user_input: str) -> str:
        """The user's message; if it starts with `/name` for a command, run it and append its output."""
        first, _, rest = user_input.strip().partition(" ")
        command = self.commands.get(first[1:]) if first.startswith("/") else None
        if command is None:  # not a command, or an unknown `/foo`: plain text
            return user_input
        output = command(rest.strip())
        if output is None:
            return user_input
        return f'{user_input}\n\n<command name="{first[1:]}">\n{output}\n</command>'

    def _loop(self) -> None:
        """Call the model until it answers, running the tools it calls. Raises `TurnStopped` if the turn has to end
        first.

        Once the model has answered, compact the context if it's over `compact_at`.
        """
        for calls in itertools.count():
            if self.budget.cancelled:  # before the budget: a cancelled one also counts as spent
                raise TurnStopped("cancelled")
            if self.over_budget:
                raise TurnStopped("budget spent")
            if self.context_tokens >= HARD_LIMIT * self.context_window:
                raise TurnStopped("context full")
            if calls > self.max_tool_rounds:
                raise TurnStopped("out of tool rounds")
            self.step()
            last = self.messages[-1]
            if last["role"] == "assistant" and not last.get("tool_calls"):  # an answer, even if cut off or empty
                if not self._interrupt.is_set():  # compact() clears the flag, and the user said stop
                    self.compact()
                return
            if self._interrupt.is_set():
                raise TurnStopped("interrupted")

    def _run_tool_calls(self, calls: list[dict]) -> None:
        """Append one result per call, in call order. If the turn is stopped part-way, the calls left get `NOT_RUN`:
        a request can't hold a call without a result."""
        done = 0
        try:
            for call in calls:
                if self._interrupt.is_set():
                    raise TurnStopped("interrupted")
                self._add_result(call, self._run_tool(call))
                done += 1
        finally:
            for call in calls[done:]:
                self._add_result(call, NOT_RUN)

    def _add_result(self, call: dict, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

    def _run_tool(self, call: dict) -> str:
        """A call's result: the tool's output, or what `approve` gave instead if the tool needs approval. Errors
        (bad JSON, invalid arguments, failures) are returned too, so the model can retry."""
        name = call["function"]["name"]
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        fn, _ = tool
        try:
            arguments = parse_arguments(call["function"]["arguments"])
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
