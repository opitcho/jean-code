"""Terminal chat with the coding agent or the orchestrator.

Run with: uv run --env-file .env jean-code [--sandbox] [--agent orchestrator]
"""

import argparse
import queue
import signal

from jean_code.agents.coding import coding_agent
from jean_code.agents.experts import orchestrator_agent
from jean_code.config import USAGE_LOG
from jean_code.core.agent import DEFAULT_MODEL, Agent
from jean_code.core.driver import Approve, Driver, denied
from jean_code.core.events import (
    Approval,
    ApprovalRequested,
    Compaction,
    Delta,
    JobEnded,
    MessageAdded,
    TurnEnded,
    TurnStarted,
)
from jean_code.core.llm import AssistantMessage, Message, ToolMessage
from jean_code.core.node import AgentNode
from jean_code.core.usage import breakdown_lines, human, usage_line
from jean_code.core.view import AgentView, Subscription
from jean_code.logs import Recorder, process_logs

POLL_SECONDS = 0.1  # how often a wait for the next event lets a Ctrl+C through


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


class Printer:
    """Prints the main agent's work as it happens.

    Its reply is printed as it streams in, and each message as it lands. A streamed line stays open until anything
    else is printed, which ends it first.
    """

    def __init__(self):
        self.open: str | None = None  # the kind of the streamed line being printed, if one is open
        self.streamed = {"reasoning": "", "content": ""}

    def line(self, text: str) -> None:
        """Prints `text` on a line of its own."""
        self.end_line()
        print(text)

    def end_line(self) -> None:
        """Ends the streamed line, if one is open."""
        if self.open is not None:
            print()
            self.open = None

    def delta(self, kind: str, text: str) -> None:
        """Prints a piece of the reply, on the line for its kind: thinking or the answer."""
        if kind != self.open:
            self.end_line()
            print(dim("thinking> ") if kind == "reasoning" else "agent> ", end="")
            self.open = kind
        print(dim(text) if kind == "reasoning" else text, end="", flush=True)
        self.streamed[kind] += text

    def message(self, msg: Message) -> None:
        """Prints a message as it lands.

        For a reply, that's what of it didn't stream in, then its tool calls. User messages aren't printed.
        """
        match msg:
            case ToolMessage():
                self.line(f"result> {msg.content}")
            case AssistantMessage():
                for kind, text in (("reasoning", msg.reasoning or ""), ("content", msg.content)):
                    if rest := text[len(self.streamed[kind]) :]:  # all of it, for a reply that didn't stream
                        self.delta(kind, rest)
                self.end_line()
                self.streamed = {"reasoning": "", "content": ""}
                for call in msg.tool_calls:
                    print(f"tool> {call.function.name}({call.function.arguments})")
        # user messages: the user just typed them; the marker and summaries show as compaction lines


def compaction_line(event: Compaction) -> str:
    """Returns the line for a compaction, or for why it didn't happen.

    For example `[compacted 38 messages · ~210.0k → ~24.0k tokens]`.
    """
    if event.status == "compacted":
        return (
            f"[compacted {event.removed} messages · ~{human(event.tokens_before)} → "
            f"~{human(event.tokens_after or 0)} tokens]"
        )
    return f"[compaction {event.status}]"


def drain(events: Subscription) -> None:
    """Drops the events that queued up while the prompt waited.

    They are background runs' events, whose results reach the model with the next message anyway.
    """
    while True:
        try:
            events.get(timeout=0)
        except queue.Empty:
            return


def repl(driver: Driver, recorder: Recorder | None = None) -> None:
    """Chats until the user exits: reads a line, then runs it and prints it live.

    Calls that need approval are asked about on the terminal, and Ctrl+C mid-turn stops the turn. `/compact`
    compacts now, and `/cost` shows the session's spend and the API key's credits. With a `recorder`, its trace
    folder is shown at the start, and a log that stopped is reported once the turn ends.
    """
    print(f"Chatting with {driver.agent.model}. Type 'exit' or 'quit' to stop, '/cost' for spend.")
    if recorder and recorder.trace_dir:
        print(dim(f"trace: {recorder.trace_dir}"))
    reported = 0
    with driver.view.subscribe() as (_, events):
        while True:
            if pending := driver.agent.subagents.pending if driver.agent.subagents else 0:
                results, them = ("results", "them") if pending > 1 else ("result", "it")
                print(dim(f"[{pending} subagent {results} waiting; your next message delivers {them}]"))
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            drain(events)  # after the prompt: what arrived while it waited

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break
            if user_input == "/cost":
                print_cost(driver)
                continue
            run_and_print(driver, events, user_input)
            if recorder and len(recorder.errors) > reported:
                for error in recorder.errors[reported:]:
                    print(f"[log stopped] {error}")
                reported = len(recorder.errors)


def ask_approval(driver: Driver, approval: Approval) -> None:
    """Asks on the terminal whether a call may run, and answers it.

    y runs it; anything else refuses it, with the text as the reason. Ctrl+C or Ctrl+D stops the turn instead.
    """
    try:
        answer = input(f"approve {approval.tool}({approval.arguments})? [y/N, or a reason to deny] ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        driver.cancel()  # also answers this approval, so the turn ends
        return
    if answer.lower() in {"y", "yes"}:
        driver.answer(approval.id, None)
    else:
        driver.answer(approval.id, denied("" if answer.lower() in {"", "n", "no"} else answer))


def next_event(events: Subscription):
    """Returns the next event, waiting in short steps so a Ctrl+C's handler can run."""
    while True:
        try:
            return events.get(timeout=POLL_SECONDS)
        except queue.Empty:
            continue


def run_and_print(driver: Driver, events: Subscription, line: str) -> JobEnded:
    """Runs `line` as a job, and prints it as its events arrive.

    `line` is a turn, or `/compact`. Each call that waits for approval is asked about. The main agent's reply
    is printed as it streams in, and its other messages as they land; any other agent gets a line when its turn
    starts and one when it ends. Afterwards it prints the error, why the turn stopped if it didn't answer, and a
    usage line for the calls the whole tree made meanwhile.

    Returns:
        How the job ended.
    """
    start = len(driver.view.state().calls)
    if line == "/compact":  # a UI command: never sent to the model as a message
        driver.compact()
    else:
        driver.send(line)

    def on_sigint(signum, frame):
        driver.cancel()

    compactions = 0
    out = Printer()
    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        while True:
            event = next_event(events)
            match event:
                case Delta(path="main"):
                    out.delta(event.kind, event.text)
                case MessageAdded(path="main"):
                    out.message(event.message)
                case ApprovalRequested():
                    out.end_line()
                    signal.signal(signal.SIGINT, previous)  # at the prompt, Ctrl+C is a KeyboardInterrupt again
                    try:
                        ask_approval(driver, event.approval)
                    finally:
                        signal.signal(signal.SIGINT, on_sigint)
                case TurnStarted(path=path) if path != "main":
                    out.line(dim(f"[{path} {driver.view.state().agents[path].profile}: started]"))
                case TurnEnded(path=path) if path != "main":
                    out.line(dim(f"[{path}: {event.end or 'failed'}]"))
                case Compaction(path="main"):
                    compactions += 1
                    out.line(dim(compaction_line(event)))
                case JobEnded():
                    break
    finally:
        signal.signal(signal.SIGINT, previous)
        out.end_line()  # a reply cut off, or a call that failed, mid-stream

    if event.error:
        print(f"[error] {event.error}")
    elif event.kind == "compact" and not compactions:  # an outcome no compaction record holds
        print(dim(f"[compaction {event.result}]"))
    elif event.kind == "turn" and event.result != "answered":
        print(f"[{event.result}]")
    state = driver.view.state()
    if calls := state.calls[start:]:
        print(dim(f"{usage_line(calls)} · context ~{human(state.agents['main'].context_tokens)}"))
    return event


def print_cost(driver: Driver) -> None:
    """Prints the session's spend, where it went, and the API key's credits.

    The spend covers the agent and any subagents; where it went is shown once there are several agents.
    """
    state = driver.view.state()
    print(f"session: ${state.agents['main'].spent:.4f} over {len(state.calls)} model calls")
    if len(state.agents) > 1:
        for line in breakdown_lines(driver.agent.node.breakdown()):
            print(dim(line))
    credits = getattr(driver.agent.client, "credits", None)
    if credits is None:  # only OpenRouter reports credits
        return
    try:
        key = credits()
    except Exception as e:  # noqa: BLE001 - the credits line is optional
        print(f"credits unavailable: {type(e).__name__}: {e}")
        return
    line = f"API key: ${key.get('usage') or 0:.2f} used"
    if key.get("limit") is not None:
        line += f" of ${key['limit']:.2f} (${key.get('limit_remaining') or 0:.2f} left)"
    print(line)


def main() -> None:
    """Parses the command line, builds the agent, and chats until the user exits."""
    parser = argparse.ArgumentParser(description="Chat with a coding agent that has a shell, or with the orchestrator.")
    parser.add_argument(
        "--agent",
        choices=("coding", "orchestrator"),
        default="coding",
        help="coding: a shell, web tools and helpers; orchestrator: no shell, it runs expert subagents",
    )
    parser.add_argument("--sandbox", action="store_true", help="coding agent: run commands in a Docker container")
    parser.add_argument(
        "--no-network", action="store_true", help="no network in the containers (--sandbox, or the experts')"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--reasoning", default="medium", choices=("low", "medium", "high"), help="reasoning effort")
    parser.add_argument("--max-cost", type=float, default=None, help="stop once the session has spent this many USD")
    parser.add_argument(
        "--compact-at",
        type=int,
        default=256_000,
        help="compact the context after a turn that ends over this many tokens",
    )
    cli = parser.parse_args()
    if cli.agent == "orchestrator" and cli.sandbox:
        parser.error("--sandbox is for the coding agent: the orchestrator has no shell")
    if cli.max_cost is not None and cli.max_cost <= 0:
        parser.error(f"--max-cost must be positive, got {cli.max_cost}")

    def build(node: AgentNode, approve: Approve) -> Agent:
        common = {"reasoning": cli.reasoning, "compact_at": cli.compact_at, "node": node}
        if cli.agent == "orchestrator":  # nothing it calls needs approval
            return orchestrator_agent(cli.model, network=not cli.no_network, **common)
        return coding_agent(cli.model, sandbox=cli.sandbox, network=not cli.no_network, approve=approve, **common)

    view = AgentView()
    recorder = Recorder(view, process_logs.new_tree(), USAGE_LOG)  # before the agent: it gets every event
    try:
        try:
            driver = Driver(build, max_cost=cli.max_cost, view=view)
        except ValueError as e:  # a setting the agent refuses, e.g. --compact-at over half the context window
            parser.error(str(e))
        try:
            repl(driver, recorder)
        finally:
            running = driver.agent.subagents.running() if driver.agent.subagents else 0
            if running:
                print(dim(f"[stopping {running} subagent{'s' if running > 1 else ''}…]"))
            driver.close()
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
