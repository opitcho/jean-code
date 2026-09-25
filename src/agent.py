import json
import threading
from pathlib import Path
from typing import Callable

from pydantic import validate_call

import skills
from llm import ChatClient, openrouter, tool_schema
from prompts import (COMPACTION_PROMPT, EARLIER_SUMMARY_NOTE, HEAD_MARKER, HEAD_MARKER_TAG, SUMMARY_MESSAGE,
                     SUMMARY_TAG)
from usage import Budget

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
CONTEXT_WINDOW = 1_048_576  # tokens, for DEFAULT_MODEL (OpenRouter's /models)
HARD_LIMIT = 0.9  # a turn stops before its context passes this share of the window


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
    """Tool-calling agent that chats with a user through a model on OpenRouter.

    The transcript in `self.messages` is in the OpenAI chat format. It stores each model reply
    as the API returned it and sends it back unchanged: one assistant message (content, plus
    reasoning, reasoning_details and tool calls when present), then one tool message per call,
    matched to its call by `tool_call_id`. Tools are typed, documented functions passed as `tools`;
    their schemas are built from the signature and docstring.

    A UI shows a turn by reading `messages` as it grows; the agent knows nothing about UIs.

    Tools named in `needs_approval` wait for the user: the turn stops before running
    one, and resumes when a UI calls `approve()` or `deny()`. A call waiting for approval is
    simply a call without a result yet, so `pending_calls` is read from the transcript.

    Skills in `skills_dir` load in three levels, so the model only pays for what it uses:
    names and descriptions in the system prompt, a SKILL.md body via `load_skill` (or the
    user typing `/name`), and bundled files via `read_skill_file`. The tool list and system
    prompt never change once the first request is sent, so loading a skill doesn't invalidate the cache.

    User commands are agent methods the user calls by starting a message with `/name`, as tools
    are the ones the model calls. `self.commands` maps a name to a function of the rest of the
    line; its output goes into the user's message, so the model sees what the command did.

    Each model call appends the API's usage to `self.usage` (this agent's own calls) and charges it to
    `self.budget`, the agent's node in a tree of budgets that mirrors its subagents. Once this node or an
    ancestor has spent its limit (`max_cost` is this node's), turns stop as if interrupted. With a limit, a
    call whose cost is unknown also stops the turn, since the budget can't be enforced past it. `close()`
    runs `cleanup`, releasing what the tools hold.

    Context compaction: `context_tokens` is the size of the context, measured by the last call plus an
    estimate for what came after it. When a turn ends with the context at `compact_at` tokens or more,
    `compact()` has the model summarize the middle of the conversation, and replaces the middle with that
    summary. The first `keep_first` turns and the last `keep_last` turns stay verbatim. It only happens between
    turns, so a tool loop's reasoning is never cut. A marker message (`head_end`) ends the kept head. Each
    attempt is recorded in `compactions`; `context_history` lists every context the session has had.
    A turn that would pass `HARD_LIMIT` of the window stops, with the reason in `stop_reason`.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        tools: dict[str, Callable] | None = None,
        needs_approval: set[str] | None = None,
        cleanup: list[Callable[[], None]] | None = None,
        reasoning: str | None = "medium",
        max_tool_rounds: int = 10,
        client: ChatClient | None = None,
        skills_dir: str | Path | None = None,
        max_cost: float | None = None,
        budget: Budget | None = None,
        compact_at: int | None = 256_000,
        keep_first: int = 1,
        keep_last: int = 2,
        context_window: int = CONTEXT_WINDOW,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.reasoning = reasoning  # effort: "low", "medium" or "high"; None sends no reasoning setting
        self.max_tool_rounds = max_tool_rounds
        self.client = client or openrouter()
        self.tool_map: dict[str, Callable] = dict(tools or {})
        self.needs_approval: set[str] = set(needs_approval or ())  # tool names the user must approve before each call
        self.cleanup: list[Callable[[], None]] = list(cleanup or [])
        self._registered: dict[str, tuple[Callable, dict]] | None = None  # built at the first request

        self.skills = skills.discover_skills(skills_dir) if skills_dir else {}
        self.loaded_skills: list[str] = []
        self.commands: dict[str, Callable[[str], str | None]] = {
            name: lambda rest, name=name: self.load_skill(name) for name in self.skills
        }

        self.messages: list[dict] = []
        system = f"{self.system_prompt}\n\n{skills.skills_prompt(self.skills)}".strip()
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
        self.compact_at = compact_at  # None: never compact on its own
        self.keep_first = keep_first
        self.keep_last = keep_last
        self.context_window = context_window
        self.head_end: int | None = None  # index just after the head marker; nothing before it is ever removed
        self.compactions: list[dict] = []  # one record per compaction attempt that got past the checks
        self.stop_reason: str | None = None  # why the last turn stopped early, when it wasn't the user or the budget
        self._measured: tuple[int, int] | None = None  # (len(messages), tokens) after the last measured call

        self.last_request: dict | None = None
        self.last_response: dict | None = None  # the last response: id, provider, choices, usage
        self._interrupt = threading.Event()

    @property
    def awaiting_user(self) -> bool:
        """True when the model has nothing left to respond to, a call waits for approval, or it was interrupted."""
        if self.interrupted or self.pending_calls:
            return True
        return not self.messages or self.messages[-1]["role"] not in ("user", "tool")

    @property
    def interrupted(self) -> bool:
        """True once `interrupt()` was called, the budget ran out or a cost was unknown, until the next turn or approval decision."""
        return self._interrupt.is_set()

    @property
    def pending_calls(self) -> list[tuple[str, dict]]:
        """Calls from the model's last reply that have no result yet: they wait for the user's approval."""
        pending = []
        for call in self._unanswered_calls():
            try:
                arguments = parse_arguments(call["function"]["arguments"])
            except ValueError:
                arguments = {"<invalid JSON>": call["function"]["arguments"]}
            pending.append((call["function"]["name"], arguments))
        return pending

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
    def last_cost_unknown(self) -> bool:
        """True when there is a budget and the last model call reported no cost, so `total_cost` is too low."""
        return self.max_cost is not None and bool(self.usage) and self.usage[-1].get("cost") is None

    @property
    def context_tokens(self) -> int:
        """Tokens the next request's prompt will take: the last call's prompt and completion, plus an estimate
        for the messages added since. Before the first call, the whole transcript and the tool schemas are
        estimated; right after a compaction, the old measurement scaled down by the estimated share kept."""
        measured, messages = self._measured, self.messages  # read once: another thread may be running the turn
        if measured is None:
            return estimate_tokens(self._tool_schemas()) + sum(estimate_tokens(m) for m in messages)
        count, tokens = measured
        return tokens + sum(estimate_tokens(m) for m in messages[count:])

    @property
    def context_history(self) -> list[list[dict]]:
        """Every context the session has had: the one before each compaction, then the current one."""
        return [c["before"] for c in self.compactions if c["status"] == "compacted"] + [self.messages]

    def approve(self) -> None:
        """Run the first pending call, then continue the turn until it ends or another call needs approval."""
        call, *rest = self._pending_or_raise()
        self._append_result(call["id"], self._run_tool(call))
        self._resume(rest)

    def deny(self, reason: str = "") -> None:
        """Refuse the first pending call; the model gets the refusal (and reason) as its result and continues."""
        call, *rest = self._pending_or_raise()
        self._append_result(call["id"], f"Denied by user. {reason}".strip())
        self._resume(rest)

    def interrupt(self):
        """Stop the turn once the model call in flight returns; that reply's tool calls still run.

        Safe to call from another thread or a signal handler.
        """
        self._interrupt.set()

    def close(self) -> None:
        """Release what the tools hold (background jobs, containers) by running `cleanup`."""
        for fn in self.cleanup:
            fn()

    def run_turn(self, user_input: str) -> None:
        """Add the user's message (with a command's output, if it starts with one), then call the model until it answers."""
        content = self._run_command(user_input)
        self._interrupt.clear()
        self.stop_reason = None
        for call in self._unanswered_calls():  # every call needs a result, so answer the ones left waiting
            self._append_result(call["id"], "Not run: the user sent a new message instead.")
        if self.head_end is None and len(self._turn_starts()) == self.keep_first:
            self.messages.append({"role": "user", "content": HEAD_MARKER})
            self.head_end = len(self.messages)
        self.messages.append({"role": "user", "content": content})
        self._loop()

    def step(self) -> None:
        """One model call on the current context: append its reply, then run its tool calls."""
        request = self._build_request()
        self.last_request = request
        self.last_response = None
        response = self.client.complete(request)
        self.last_response = response
        # Recorded before the reply is read, so a paid call counts even if its reply is malformed.
        # Without usage the entry still marks the call, with `cost: None`.
        usage = self._record_usage(response)
        message = response["choices"][0]["message"]
        reply = {k: v for k, v in message.items() if v is not None}  # drop unset fields like `refusal: null`
        reply.setdefault("content", "")
        self.messages.append(reply)
        if usage.get("prompt_tokens") is not None:  # the prompt plus the reply just appended
            self._measured = (len(self.messages), usage["prompt_tokens"] + (usage.get("completion_tokens") or 0))
        self._run_tool_calls(reply.get("tool_calls", []))

    def compact(self, force: bool = False) -> str:
        """Replace the middle of the conversation with a summary the model writes, keeping the first
        `keep_first` and last `keep_last` turns. Call it between turns.

        Without `force`, only when `context_tokens` is at `compact_at` or more, and only if keeping the head and
        tail alone gets under it. Returns what happened: "compacted", "below threshold", "nothing to compact",
        "skipped: …", "refused: …" or "failed: …". The transcript is unchanged unless it's "compacted";
        `interrupt()` during the summary call cancels it.
        """
        self._interrupt.clear()
        if self._unanswered_calls():
            return "refused: a tool call is waiting for approval"
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

        # The kept part's share of the estimate, applied to the measured size: the estimator's bias cancels out.
        schemas = estimate_tokens(self._tool_schemas())
        kept_estimate = schemas + sum(estimate_tokens(m) for m in head + tail)
        kept = round(tokens_before * kept_estimate / (kept_estimate + sum(estimate_tokens(m) for m in middle)))
        if not force and kept >= self.compact_at:
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
        except Exception as e:
            record["status"] = f"failed: {type(e).__name__}: {e}"
            return record["status"]
        if choice["message"].get("tool_calls"):
            record["status"] = "failed: the model called a tool instead of summarizing"
        elif choice.get("finish_reason") == "length":
            record["status"] = "failed: the summary was cut off at the length limit"
        elif not summary:
            record["status"] = "failed: the summary was empty"
        elif self.interrupted:
            record["status"] = "failed: interrupted"
        if record["status"]:
            return record["status"]

        record["summary"] = {"role": "user", "content": SUMMARY_MESSAGE.format(summary=summary)}
        record["before"] = messages  # the old list itself: it's replaced, never modified
        self.messages = head + [record["summary"]] + tail
        # The old measurement described the longer list. Scale it by the estimates instead of taking the raw
        # estimate, which runs high; the next call measures the real size.
        new_estimate = kept_estimate + estimate_tokens(record["summary"])
        old_estimate = kept_estimate + sum(estimate_tokens(m) for m in middle)
        self._measured = (len(self.messages), round(tokens_before * new_estimate / old_estimate))
        self.loaded_skills = [name for name in self.loaded_skills if self._skill_in_context(name)]
        record["tokens_after"] = self.context_tokens
        record["status"] = "compacted"
        return record["status"]

    def load_skill(self, name: str) -> str:
        """Load a skill's instructions. Call this before starting a task that matches a skill.

        Args:
            name: Name of the skill, from the list of available skills.

        Returns:
            The skill's instructions, followed by the other files it provides.
        """
        skill = self._get_skill(name)
        if name in self.loaded_skills:
            return f"Skill '{name}' is already loaded; follow the instructions above."
        body = skills.instructions(skill)
        self.loaded_skills.append(name)
        return body

    def read_skill_file(self, name: str, path: str) -> str:
        """Read a file bundled with a skill, such as a reference document or a script.

        Args:
            name: Name of the skill the file belongs to.
            path: Path of the file relative to the skill's folder, as listed by load_skill.

        Returns:
            The file's text.
        """
        return skills.read_file(self._get_skill(name), path)

    def _skill_in_context(self, name: str) -> bool:
        body = skills.instructions(self._get_skill(name))
        return any(isinstance(m.get("content"), str) and body in m["content"] for m in self.messages)

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
        """
        usage = response.get("usage") or {}
        entry = {"cost": None, **usage, **{k: response.get(k) for k in ("id", "provider", "model", "created")}}
        if kind:
            entry["kind"] = kind
        self.usage.append(entry)
        self.budget.charge(entry)
        return usage

    def _get_skill(self, name: str) -> skills.Skill:
        if name not in self.skills:
            raise ValueError(f"unknown skill '{name}'; available: {', '.join(self.skills)}")
        return self.skills[name]

    def _tools(self) -> dict[str, tuple[Callable, dict]]:
        """Every tool the model can call, as (validating wrapper, schema) by name.

        Built once, at the first request, and never changed after: the schemas sit near the top
        of the prompt, so any change would invalidate the cache.
        """
        if self._registered is None:
            tools = dict(self.tool_map)
            if self.skills:
                tools |= {"load_skill": self.load_skill, "read_skill_file": self.read_skill_file}
            self._registered = {name: (validate_call(fn), tool_schema(name, fn)) for name, fn in tools.items()}
        return self._registered

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
        return [schema for _, schema in self._tools().values()]

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
        """Call the model until it answers, a call needs approval, the turn is interrupted or the budget is spent.

        Once the model has answered, compact the context if it's over `compact_at`.
        """
        for _ in range(1 + self.max_tool_rounds):
            if self.over_budget:
                self._interrupt.set()
                break
            if self.context_tokens >= HARD_LIMIT * self.context_window:
                self.stop_reason = f"the context is nearly full (~{self.context_tokens} tokens), so the turn stopped"
                self._interrupt.set()
                break
            self.step()
            if self.last_cost_unknown:
                self._interrupt.set()
                break
            if self.awaiting_user:
                break
        if self.awaiting_user and not self.interrupted and not self._unanswered_calls():
            self.compact()
            if self.last_cost_unknown:
                self._interrupt.set()

    def _resume(self, rest: list[dict]) -> None:
        """After an approval decision: run the reply's remaining calls, then continue if none is waiting."""
        self._interrupt.clear()
        self.stop_reason = None
        self._run_tool_calls(rest)
        if not self.pending_calls:
            self._loop()

    def _unanswered_calls(self) -> list[dict]:
        """Calls in the last assistant message without a tool result yet, in call order."""
        last = next((i for i in reversed(range(len(self.messages))) if self.messages[i]["role"] == "assistant"), None)
        if last is None:
            return []
        answered = {m["tool_call_id"] for m in self.messages[last + 1 :] if m["role"] == "tool"}
        return [call for call in self.messages[last].get("tool_calls", []) if call["id"] not in answered]

    def _pending_or_raise(self) -> list[dict]:
        pending = self._unanswered_calls()
        if not pending:
            raise ValueError("no tool call is waiting for approval")
        return pending

    def _run_tool_calls(self, calls: list[dict]) -> None:
        """Append one result per call, in call order.

        Stops before the first call that needs approval; it and the calls after it stay pending.
        """
        for call in calls:
            if call["function"]["name"] in self.needs_approval:
                return
            self._append_result(call["id"], self._run_tool(call))

    def _append_result(self, call_id: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def _run_tool(self, call: dict) -> str:
        """Run a registered tool; errors (bad JSON, invalid arguments, failures) are returned so the model can retry."""
        name = call["function"]["name"]
        tool = self._tools().get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        fn, _ = tool
        try:
            return str(fn(**parse_arguments(call["function"]["arguments"])))
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
