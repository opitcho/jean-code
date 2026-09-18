import json
import threading
from pathlib import Path
from typing import Callable

from pydantic import validate_call

import skills
from llm import ChatClient, openrouter, tool_schema

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"


def parse_arguments(arguments: str) -> dict:
    """A tool call's arguments, which the model sends as a JSON string. Raises ValueError if not a JSON object."""
    value = json.loads(arguments or "{}")
    if not isinstance(value, dict):
        raise ValueError(f"arguments must be a JSON object, got {arguments!r}")
    return value


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

    Each model call appends the API's usage to `self.usage`. Once `total_cost` reaches
    `max_cost`, turns stop as if interrupted. `close()` runs `cleanup`, releasing what the tools hold.
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

        self.max_cost = max_cost
        self.usage: list[dict] = []  # per model call: the API's usage, plus the response's id, provider, model, created

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
        """True once `interrupt()` was called or the budget ran out, until the next turn or approval decision."""
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
    def over_budget(self) -> bool:
        return self.max_cost is not None and self.total_cost >= self.max_cost

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
        for call in self._unanswered_calls():  # every call needs a result, so answer the ones left waiting
            self._append_result(call["id"], "Not run: the user sent a new message instead.")
        self.messages.append({"role": "user", "content": content})
        self._loop()

    def step(self) -> None:
        """One model call on the current context: append its reply, then run its tool calls."""
        request = self._build_request()
        self.last_request = request
        self.last_response = None
        response = self.client.complete(request)
        self.last_response = response
        message = response["choices"][0]["message"]
        reply = {k: v for k, v in message.items() if v is not None}  # drop unset fields like `refusal: null`
        reply.setdefault("content", "")
        if usage := response.get("usage"):
            self.usage.append({**usage, **{k: response.get(k) for k in ("id", "provider", "model", "created")}})
        self.messages.append(reply)
        self._run_tool_calls(reply.get("tool_calls", []))

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
        tools = self._tools()
        if tools:
            request["tools"] = [schema for _, schema in tools.values()]
        return request

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
        """Call the model until it answers, a call needs approval, the turn is interrupted or the budget is spent."""
        for _ in range(1 + self.max_tool_rounds):
            if self.over_budget:
                self._interrupt.set()
                break
            self.step()
            if self.awaiting_user:
                break

    def _resume(self, rest: list[dict]) -> None:
        """After an approval decision: run the reply's remaining calls, then continue if none is waiting."""
        self._interrupt.clear()
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
