import argparse
import re
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ollama
import yaml

from shell import DockerShell, LocalShell, Shell
from tools import add, sub, web_search, fetch_page

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass
class Skill:
    """A skill folder: SKILL.md metadata plus the folder holding its instructions and files."""

    name: str
    description: str
    path: Path


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a SKILL.md into its YAML frontmatter and its body."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)(.*)", text, re.DOTALL)
    if match is None:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, match.group(2).lstrip()


def skill_files(skill_dir: Path) -> list[Path]:
    """Files bundled with a skill, relative to its folder, skipping SKILL.md and hidden or dunder paths."""
    files = []
    for path in sorted(skill_dir.rglob("*")):
        rel = path.relative_to(skill_dir)
        hidden = any(part.startswith((".", "__")) for part in rel.parts)
        if path.is_file() and rel != Path("SKILL.md") and not hidden:
            files.append(rel)
    return files


def assistant_turn(
    content: str | None, thinking: str | None = None, calls: list[tuple[str, dict]] | None = None
) -> dict:
    """An assistant message as Ollama expects it; `thinking` and `tool_calls` only when present."""
    turn: dict = {"role": "assistant", "content": content or ""}
    if thinking:
        turn["thinking"] = thinking  # kept so later rounds see the reasoning behind the calls
    if calls:
        turn["tool_calls"] = [{"function": {"name": name, "arguments": args}} for name, args in calls]
    return turn


def discover_skills(skills_dir: str | Path) -> dict[str, Skill]:
    """Read the metadata of every `<skills_dir>/<name>/SKILL.md`, keyed by skill name."""
    skills = {}
    for skill_md in sorted(Path(skills_dir).glob("*/SKILL.md")):
        meta, _ = split_frontmatter(skill_md.read_text())
        name = meta.get("name", skill_md.parent.name)
        skills[name] = Skill(name, meta.get("description", ""), skill_md.parent)
    return skills


class BaseAgent:
    """Tool-calling agent that chats with a user through an Ollama model.

    The transcript in `self.messages` stores each model reply as Ollama returns it and
    sends it back as-is: one assistant message (content, plus thinking and tool calls when
    present), then one tool message per call in call order. Results are matched to calls
    by position, so the same tool can be called several times in one reply. Subclasses
    register tools by filling `self.tool_map`.

    Tools named in `self.needs_approval` wait for the user: the turn stops before running
    one, and resumes when a UI calls `approve()` or `deny()`. A call waiting for approval is
    simply a call without a result yet, so `pending_calls` is read from the transcript.

    Skills in `skills_dir` load in three levels, so the model only pays for what it uses:
    names and descriptions in the system prompt, a SKILL.md body via `load_skill` (or the
    user typing `/name`), and bundled files via `read_skill_file`. The tool list and system
    prompt never change after construction, so loading a skill doesn't invalidate the KV cache.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        think: bool | str = True,
        max_tool_rounds: int = 10,
        client: ollama.Client | None = None,
        skills_dir: str | Path | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.think = think
        self.max_tool_rounds = max_tool_rounds
        self.client = client or ollama.Client()
        self.tool_map: dict[str, Callable] = {}
        self.needs_approval: set[str] = set()  # tool names the user must approve before each call

        self.skills = discover_skills(skills_dir) if skills_dir else {}
        self.loaded_skills: list[str] = []

        self.messages: list[dict] = []
        system = f"{self.system_prompt}\n\n{self._skills_prompt()}".strip()
        if system:
            self.messages.append({"role": "system", "content": system})

        self.last_request: dict | None = None
        self.last_response = None
        self._interrupt = threading.Event()

    @property
    def awaiting_user(self) -> bool:
        """True when the model has nothing left to respond to, a call waits for approval, or it was interrupted."""
        if self._interrupt.is_set() or self.pending_calls:
            return True
        return not self.messages or self.messages[-1]["role"] not in ("user", "tool")

    @property
    def pending_calls(self) -> list[tuple[str, dict]]:
        """Calls from the model's last reply that have no result yet: they wait for the user's approval."""
        last = next((i for i in reversed(range(len(self.messages))) if self.messages[i]["role"] == "assistant"), None)
        if last is None:
            return []
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in self.messages[last].get("tool_calls", [])]
        answered = len(self.messages) - 1 - last
        return calls[answered:]

    def approve(self) -> None:
        """Run the first pending call, then continue the turn until it ends or another call needs approval."""
        (name, arguments), *rest = self._pending_or_raise()
        self._append_result(name, self._run_tool(name, arguments))
        self._resume(rest)

    def deny(self, reason: str = "") -> None:
        """Refuse the first pending call; the model gets the refusal (and reason) as its result and continues."""
        (name, _), *rest = self._pending_or_raise()
        self._append_result(name, f"Denied by user. {reason}".strip())
        self._resume(rest)

    def interrupt(self):
        """Stop the turn once the current model call and its tool calls finish.

        Safe to call from another thread or a signal handler.
        """
        self._interrupt.set()

    def run_turn(self, user_input: str) -> None:
        """Add the user's message, then call the model until it answers (or is interrupted)."""
        self._interrupt.clear()
        for name, _ in self.pending_calls:  # every call needs a result, so answer the ones left waiting
            self._append_result(name, "Not run: the user sent a new message instead.")
        self.messages.append({"role": "user", "content": user_input})
        self._load_skill_command(user_input)
        self._loop()

    def step(self) -> None:
        """One model call on the current context: append its reply, then run its tool calls."""
        request = self._build_request()
        self.last_request = request
        self.last_response = self.client.chat(**request)

        reply = self.last_response.message
        calls = [(call.function.name, dict(call.function.arguments)) for call in reply.tool_calls or []]
        self.messages.append(assistant_turn(reply.content, reply.thinking, calls))
        self._run_tool_calls(calls)

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
        _, body = split_frontmatter((skill.path / "SKILL.md").read_text())
        files = skill_files(skill.path)
        if files:
            listing = "\n".join(f"- {rel}" for rel in files)
            body = f"{body.rstrip()}\n\nFiles in this skill (read with read_skill_file):\n{listing}"
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
        root = self._get_skill(name).path.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"'{path}' is not a file in skill '{name}'")
        return target.read_text()

    def _get_skill(self, name: str) -> Skill:
        if name not in self.skills:
            raise ValueError(f"unknown skill '{name}'; available: {', '.join(self.skills)}")
        return self.skills[name]

    def _skills_prompt(self) -> str:
        """The level-1 skills listing for the system prompt: names and descriptions only."""
        if not self.skills:
            return ""
        lines = [
            "## Skills",
            "Skills hold instructions for specific tasks. When a task matches a skill, call "
            "load_skill with its name once; its instructions then stay in the conversation.",
            "",
        ]
        lines += [f"- {skill.name}: {skill.description}" for skill in self.skills.values()]
        return "\n".join(lines)

    def _tools(self) -> dict[str, Callable]:
        """Every tool the model can call; fixed after construction so the prompt prefix stays cached."""
        return {**self.tool_map}

    def _build_request(self) -> dict:
        """The chat request for the current transcript."""
        request: dict = {
            "model": self.model,
            "messages": list(self.messages),  # a copy, so last_request doesn't grow with the transcript
            "think": self.think,
            "stream": False,
        }
        tools = self._tools()
        if tools:
            request["tools"] = list(tools.values())
        return request

    def _load_skill_command(self, user_input: str) -> None:
        """If the user typed `/name` for a skill, load it as if the model had called load_skill."""
        words = user_input.split()
        if not words or not words[0].startswith("/"):
            return
        name = words[0][1:]
        if name in self.skills:
            calls = [("load_skill", {"name": name})]
            self.messages.append(assistant_turn("", calls=calls))
            self._run_tool_calls(calls)

    def _loop(self) -> None:
        """Call the model until it answers, a call needs approval, or the turn is interrupted."""
        for _ in range(1 + self.max_tool_rounds):
            self.step()
            if self.awaiting_user:
                break

    def _resume(self, rest: list[tuple[str, dict]]) -> None:
        """After an approval decision: run the reply's remaining calls, then continue if none is waiting."""
        self._interrupt.clear()
        self._run_tool_calls(rest)
        if not self.pending_calls:
            self._loop()

    def _pending_or_raise(self) -> list[tuple[str, dict]]:
        pending = self.pending_calls
        if not pending:
            raise ValueError("no tool call is waiting for approval")
        return pending

    def _run_tool_calls(self, calls: list[tuple[str, dict]]) -> None:
        """Append one result per call, in call order: Ollama matches results to calls by position.

        Stops before the first call that needs approval; it and the calls after it stay pending.
        """
        for name, arguments in calls:
            if name in self.needs_approval:
                return
            self._append_result(name, self._run_tool(name, arguments))

    def _append_result(self, name: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_name": name, "content": content})

    def _run_tool(self, name: str, arguments: dict) -> str:
        """Run a registered tool; errors are returned as the result so the model can recover."""
        fn = self._tools().get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            return str(fn(**arguments))
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def run(self):
        """Interactive REPL: read a line, run a turn, print its turns, and ask about calls needing approval.

        Ctrl+C mid-turn interrupts once the current model call finishes.
        """
        print(f"Chatting with {self.model}. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            self._run_and_print(self.run_turn, user_input)
            while self.pending_calls and not self._interrupt.is_set():
                name, arguments = self.pending_calls[0]
                try:
                    answer = input(f"approve {name}({arguments})? [y/N, or a reason to deny] ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break  # leave it pending; the next message answers it
                if answer.lower() in {"y", "yes"}:
                    self._run_and_print(self.approve)
                else:
                    self._run_and_print(self.deny, "" if answer.lower() in {"", "n", "no"} else answer)

    def _run_and_print(self, fn: Callable, *args) -> None:
        """Run `fn` with Ctrl+C mapped to interrupt(), then print the turns it added."""
        start = len(self.messages)
        previous = signal.signal(signal.SIGINT, lambda signum, frame: self.interrupt())
        try:
            fn(*args)
        finally:
            signal.signal(signal.SIGINT, previous)

        for msg in self.messages[start:]:
            if msg["role"] != "user":  # the user just typed it
                print_turn(msg)
        if self._interrupt.is_set():
            print("[interrupted]")


def print_turn(msg: dict):
    """Print one transcript turn for the REPL; a model reply can hold thinking, content and calls."""
    if msg["role"] == "tool":
        print(f"result> {msg['content']}")
        return
    if msg.get("thinking"):
        print(f"\033[2mthinking> {msg['thinking']}\033[0m")
    if msg.get("content"):
        print(f"agent> {msg['content']}")
    for call in msg.get("tool_calls", []):
        print(f"tool> {call['function']['name']}({call['function']['arguments']})")


class MathAgent(BaseAgent):
    """Example specialized agent with math and web tools."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_map = {"add": add, "sub": sub, "web_search": web_search, "fetch_page": fetch_page}


class CodingAgent(BaseAgent):
    """Agent with a shell (on the host or in a Docker sandbox) and web tools.

    Whether `bash` needs approval comes from the shell: yes on the host, no in the sandbox.
    """

    def __init__(self, model: str, shell: Shell, system_prompt: str = "", **kwargs):
        super().__init__(model, system_prompt=f"{system_prompt}\n\n{shell.describe()}".strip(), **kwargs)
        self.shell = shell
        self.tool_map = {"bash": shell.bash, "job": shell.job, "web_search": web_search, "fetch_page": fetch_page}
        self.needs_approval = {"bash"} if shell.requires_approval else set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a coding agent that has a shell.")
    parser.add_argument("--sandbox", action="store_true", help="run commands in a Docker container")
    parser.add_argument("--no-network", action="store_true", help="with --sandbox: no network in the container")
    cli = parser.parse_args()

    shell = DockerShell(network=not cli.no_network) if cli.sandbox else LocalShell()
    try:
        CodingAgent(model="qwen3.6", shell=shell, skills_dir=SKILLS_DIR).run()
    finally:
        shell.close()
