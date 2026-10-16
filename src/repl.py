"""Terminal chat with the coding agent.

Run with: uv run --env-file .env python src/repl.py [--sandbox]
"""

import argparse
import signal
import threading
from pathlib import Path

from agent import DEFAULT_MODEL, NOT_RUN, Agent, TurnEnd
from approval import Approvals, denied
from coding import coding_agent
from config import USAGE_LOG
from llm import HasCredits
from usage import breakdown_lines, human, usage_line


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


def repl(agent: Agent, approvals: Approvals | None = None, usage_log: str | Path | None = USAGE_LOG) -> None:
    """Read a line, then run a turn while printing it live and asking about the calls that need approval.

    `approvals` is the one whose `ask` the agent was built with, if any of its tools need approval.
    Ctrl+C mid-turn interrupts. `/cost` shows the session's spend and the API key's credits.
    Each model call in the agent's tree is appended to `usage_log` as it's made, to sum spend across sessions:
    background subagents' calls between turns included.
    """
    if usage_log:
        agent.budget.log_path = usage_log
    print(f"Chatting with {agent.model}. Type 'exit' or 'quit' to stop, '/cost' for spend.")
    announced = 0  # finished background results already announced
    while True:
        pending = agent.subagents.pending if agent.subagents else 0
        if pending and pending != announced:
            print(dim(f"[{pending} subagent result{'s' if pending > 1 else ''} waiting; your next message delivers "
                      f"{'them' if pending > 1 else 'it'}]"))
        announced = pending
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

        run_and_print(agent, user_input, approvals)


def ask_approval(agent: Agent, name: str, arguments: dict) -> str | None:
    """Ask on the terminal whether a call may run: None for yes, otherwise the result the model gets instead.
    Ctrl+C or Ctrl+D stops the turn."""
    try:
        answer = input(f"approve {name}({arguments})? [y/N, or a reason to deny] ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        agent.interrupt()
        return NOT_RUN
    if answer.lower() in {"y", "yes"}:
        return None
    return denied("" if answer.lower() in {"", "n", "no"} else answer)


def run_and_print(agent: Agent, user_input: str, approvals: Approvals | None = None) -> TurnEnd | None:
    """Run a turn and print it as it happens.

    The turn runs on a worker thread while this thread prints each new message, asks about each call that waits in
    `approvals`, and maps Ctrl+C to `agent.interrupt()`. Afterwards prints the error if the turn raised, why it
    stopped if it didn't end with an answer, and a usage line for the model calls made meanwhile in the agent's
    tree (subagents included).

    Returns how the turn ended, or None if it raised.
    """
    printer = TurnPrinter(agent)
    errors: list[Exception] = []
    ends: list[TurnEnd] = []

    def target():
        try:
            ends.append(agent.run_turn(user_input))
        except Exception as e:  # noqa: BLE001 - printed after the turn; the REPL carries on
            errors.append(e)

    def on_sigint(signum, frame):
        agent.interrupt()

    journal = agent.budget.journal
    start = len(journal)
    worker = threading.Thread(target=target, daemon=True)
    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        worker.start()
        while worker.is_alive():
            waiting = approvals.waiting if approvals else None  # read before printing, so its call is printed first
            printer.poll()
            if waiting and approvals:
                signal.signal(signal.SIGINT, previous)  # at the prompt, Ctrl+C is a KeyboardInterrupt again
                try:
                    approvals.answer(ask_approval(agent, *waiting))
                finally:
                    signal.signal(signal.SIGINT, on_sigint)
            worker.join(0.05)
    finally:
        signal.signal(signal.SIGINT, previous)
    printer.poll()
    calls = journal[start:]

    for e in errors:
        print(f"[error] {type(e).__name__}: {e}")
    end = ends[0] if ends else None
    if end not in (None, "answered"):
        print(f"[{end}]")
    if calls:
        print(dim(f"{usage_line(calls)} · context ~{human(agent.context_tokens)}"))
    return end


def print_cost(agent: Agent) -> None:
    """The session's spend (the agent and any subagents), where it went once there are several agents, and the
    key's credits."""
    budget = agent.budget
    print(f"session: ${budget.spent:.4f} over {len(budget.journal)} model calls")
    if len(budget.nodes()) > 1:
        for line in breakdown_lines(budget.breakdown()):
            print(dim(line))
    if not isinstance(agent.client, HasCredits):  # only OpenRouter reports credits
        return
    try:
        key = agent.client.credits()
    except Exception as e:  # noqa: BLE001 - the credits line is optional
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
    parser.add_argument("--compact-at", type=int, default=256_000,
                        help="compact the context after a turn that ends over this many tokens")
    cli = parser.parse_args()

    approvals = Approvals()
    agent = coding_agent(
        cli.model, sandbox=cli.sandbox, network=not cli.no_network, reasoning=cli.reasoning, max_cost=cli.max_cost,
        compact_at=cli.compact_at, approve=approvals.ask,
    )
    try:
        repl(agent, approvals)
    finally:
        if agent.subagents and (running := agent.subagents.running()):
            print(dim(f"[stopping {running} subagent{'s' if running > 1 else ''}…]"))
        agent.close()
