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
from usage import COST_UNKNOWN, human, logged_calls, usage_line


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


class TurnPrinter:
    """Prints a turn for the REPL as it happens, by polling the agent's `messages` and `compactions`.

    Messages are matched by identity, not position, because a compaction replaces the list.
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.seen = list(agent.messages)  # references keep the ids from being reused
        self.seen_ids = {id(m) for m in self.seen}
        self.compactions = len(agent.compactions)

    def poll(self) -> None:
        for msg in list(self.agent.messages):
            if id(msg) not in self.seen_ids:
                self.seen.append(msg)
                self.seen_ids.add(id(msg))
                self._print_message(msg)
        compactions = self.agent.compactions
        while self.compactions < len(compactions) and compactions[self.compactions]["status"]:
            print(dim(compaction_line(compactions[self.compactions])))
            self.compactions += 1

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
        # user messages: the user just typed them; the marker and summaries show as compaction lines


def compaction_line(record: dict) -> str:
    """`[compacted 38 messages · ~210.0k → ~24.0k tokens]`, or why a compaction didn't happen."""
    if record["status"] == "compacted":
        return (f"[compacted {record['removed']} messages · ~{human(record['tokens_before'])} → "
                f"~{human(record['tokens_after'])} tokens]")
    return f"[compaction {record['status']}]"


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
    printer = TurnPrinter(agent)
    errors: list[Exception] = []

    def target():
        try:
            fn(*args)
        except Exception as e:
            errors.append(e)

    with logged_calls(agent.usage, usage_log) as calls:
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
        if agent.over_budget:
            print(f"[budget of ${agent.max_cost} reached]")
        elif agent.last_cost_unknown:
            print(COST_UNKNOWN)
        elif agent.stop_reason:
            print(f"[{agent.stop_reason}]")
        else:
            print("[interrupted]")
    if calls:
        print(dim(f"{usage_line(calls)} · context ~{human(agent.context_tokens)}"))


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
    parser.add_argument("--compact-at", type=int, default=256_000, help="compact the context after a turn that ends over this many tokens")
    cli = parser.parse_args()

    agent = coding_agent(
        cli.model, sandbox=cli.sandbox, network=not cli.no_network, reasoning=cli.reasoning, max_cost=cli.max_cost,
        compact_at=cli.compact_at,
    )
    try:
        repl(agent)
    finally:
        agent.close()
