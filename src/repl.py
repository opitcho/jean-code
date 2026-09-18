"""Terminal chat with the coding agent.

Run with: uv run --env-file .env python src/repl.py [--sandbox]
"""

import argparse
import signal
import threading
from pathlib import Path
from typing import Callable

from agent import Agent, DEFAULT_MODEL
from coding import coding_agent
from config import USAGE_LOG
from usage import append_log, usage_line


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


class TurnPrinter:
    """Prints a turn for the REPL as it happens, by polling the agent's `messages`."""

    def __init__(self, agent: Agent, start: int):
        self.agent = agent
        self.next = start  # index of the next message to print

    def poll(self) -> None:
        messages = self.agent.messages
        while self.next < len(messages):
            self._print_message(messages[self.next])
            self.next += 1

    def _print_message(self, msg: dict) -> None:
        if msg["role"] == "tool":
            print(f"result> {msg['content']}")
        elif msg["role"] == "assistant":
            if msg.get("reasoning"):
                print(dim(f"thinking> {msg['reasoning']}"))
            if msg.get("content"):
                print(f"agent> {msg['content']}")
            for call in msg.get("tool_calls", []):
                print(f"tool> {call['function']['name']}({call['function']['arguments']})")
        # user messages: the user just typed them


def repl(agent: Agent, usage_log: str | Path | None = USAGE_LOG) -> None:
    """Read a line, run a turn while printing it live, and ask about calls needing approval.

    Ctrl+C mid-turn interrupts. `/cost` shows the session's spend and the API key's credits.
    Each model call is appended to `usage_log`, to sum spend across sessions.
    """
    print(f"Chatting with {agent.model}. Type 'exit' or 'quit' to stop, '/cost' for spend.")
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
        if user_input == "/cost":
            print_cost(agent)
            continue

        run_and_print(agent, usage_log, agent.run_turn, user_input)
        while agent.pending_calls and not agent.interrupted:
            name, arguments = agent.pending_calls[0]
            try:
                answer = input(f"approve {name}({arguments})? [y/N, or a reason to deny] ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break  # leave it pending; the next message answers it
            if answer.lower() in {"y", "yes"}:
                run_and_print(agent, usage_log, agent.approve)
            else:
                run_and_print(agent, usage_log, agent.deny, "" if answer.lower() in {"", "n", "no"} else answer)


def run_and_print(agent: Agent, usage_log: str | Path | None, fn: Callable, *args) -> None:
    """Run `fn` in a thread with Ctrl+C mapped to interrupt(), printing each message as it lands."""
    printer = TurnPrinter(agent, start=len(agent.messages))
    first_call = len(agent.usage)
    errors: list[Exception] = []

    def target():
        try:
            fn(*args)
        except Exception as e:
            errors.append(e)

    worker = threading.Thread(target=target, daemon=True)
    previous = signal.signal(signal.SIGINT, lambda signum, frame: agent.interrupt())
    try:
        worker.start()
        while worker.is_alive():
            printer.poll()
            worker.join(0.05)
    finally:
        signal.signal(signal.SIGINT, previous)
    printer.poll()

    for e in errors:
        print(f"[error] {type(e).__name__}: {e}")
    if agent.interrupted:
        print(f"[budget of ${agent.max_cost} reached]" if agent.over_budget else "[interrupted]")
    calls = agent.usage[first_call:]
    if calls:
        print(dim(usage_line(calls)))
        if usage_log:
            append_log(usage_log, calls)


def print_cost(agent: Agent) -> None:
    print(f"session: ${agent.total_cost:.4f} over {len(agent.usage)} model calls")
    try:
        key = agent.client.credits()
    except Exception as e:
        print(f"credits unavailable: {type(e).__name__}: {e}")
        return
    line = f"API key: ${key.get('usage') or 0:.2f} used"
    if key.get("limit") is not None:
        line += f" of ${key['limit']:.2f} (${key.get('limit_remaining') or 0:.2f} left)"
    print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a coding agent that has a shell.")
    parser.add_argument("--sandbox", action="store_true", help="run commands in a Docker container")
    parser.add_argument("--no-network", action="store_true", help="with --sandbox: no network in the container")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--reasoning", default="medium", help="reasoning effort: low, medium or high")
    parser.add_argument("--max-cost", type=float, default=None, help="stop once the session has spent this many USD")
    cli = parser.parse_args()

    agent = coding_agent(
        cli.model, sandbox=cli.sandbox, network=not cli.no_network, reasoning=cli.reasoning, max_cost=cli.max_cost
    )
    try:
        repl(agent)
    finally:
        agent.close()
