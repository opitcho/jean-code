"""Terminal chat with the coding agent, or with the orchestrator, through a `Session`.

Run with: uv run --env-file .env jean-code [--sandbox] [--agent orchestrator]
"""

import argparse
import queue
import signal

from pydantic import ValidationError

from jean_code.session import (
    DEFAULT_MODEL,
    Approval,
    ApprovalRequested,
    AssistantMessage,
    Compaction,
    JobEnded,
    Message,
    MessageAdded,
    Session,
    Settings,
    Subscription,
    ToolMessage,
    TurnEnded,
    TurnStarted,
    breakdown_lines,
    denied,
    human,
    usage_line,
)

POLL_SECONDS = 0.1  # how often a wait for the next event lets a Ctrl+C through


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


def print_message(msg: Message) -> None:
    """A message of the main agent's, as it lands: results, thinking, answers and the calls it makes."""
    match msg:
        case ToolMessage():
            print(f"result> {msg.content}")
        case AssistantMessage():
            if msg.reasoning:
                print(dim(f"thinking> {msg.reasoning}"))
            if msg.content:
                print(f"agent> {msg.content}")
            for call in msg.tool_calls:
                print(f"tool> {call.function.name}({call.function.arguments})")
    # user messages: the user just typed them; the marker and summaries show as compaction lines


def compaction_line(event: Compaction) -> str:
    """`[compacted 38 messages · ~210.0k → ~24.0k tokens]`, or why a compaction didn't happen."""
    if event.status == "compacted":
        return (
            f"[compacted {event.removed} messages · ~{human(event.tokens_before)} → "
            f"~{human(event.tokens_after or 0)} tokens]"
        )
    return f"[compaction {event.status}]"


def drain(events: Subscription) -> None:
    """Drop what queued up while the prompt waited: background runs' events, whose results reach the model with the
    next message anyway."""
    while True:
        try:
            events.get(timeout=0)
        except queue.Empty:
            return


def repl(session: Session) -> None:
    """Read a line, then run it as a job while printing it live and asking about the calls that need approval.

    Ctrl+C mid-turn stops it. `/compact` compacts now, `/cost` shows the session's spend and the API key's credits.
    """
    print(f"Chatting with {session.state().agents['main'].model}. Type 'exit' or 'quit' to stop, '/cost' for spend.")
    with session.subscribe() as (_, events):
        while True:
            if pending := session.pending():
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
                print_cost(session)
                continue
            run_and_print(session, events, user_input)


def ask_approval(session: Session, approval: Approval) -> None:
    """Ask on the terminal whether a call may run, and answer it: y runs it, anything else refuses it, with the text
    as the reason. Ctrl+C or Ctrl+D stops the turn instead."""
    try:
        answer = input(f"approve {approval.tool}({approval.arguments})? [y/N, or a reason to deny] ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        session.cancel()  # also answers this approval, so the turn ends
        return
    if answer.lower() in {"y", "yes"}:
        session.answer(approval.id, None)
    else:
        session.answer(approval.id, denied("" if answer.lower() in {"", "n", "no"} else answer))


def next_event(events: Subscription):
    """The next event; waits in short steps, so a Ctrl+C's handler runs while nothing arrives."""
    while True:
        try:
            return events.get(timeout=POLL_SECONDS)
        except queue.Empty:
            continue


def run_and_print(session: Session, events: Subscription, line: str) -> JobEnded:
    """Start a job for `line` (a turn, or `/compact`) and print it as its events arrive, asking about each call that
    waits for approval. The main agent's messages are printed in full; any other agent gets a line when its turn
    starts and one when it ends. Afterwards prints the error, why the turn stopped if it didn't answer, and a usage
    line for the calls made meanwhile in the whole tree.

    Returns how the job ended.
    """
    start = len(session.state().calls)
    if line == "/compact":  # a UI command: never sent to the model as a message
        session.compact()
    else:
        session.send(line)

    def on_sigint(signum, frame):
        session.cancel()

    compactions = 0
    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        while True:
            event = next_event(events)
            match event:
                case MessageAdded(path="main"):
                    print_message(event.message)
                case ApprovalRequested():
                    signal.signal(signal.SIGINT, previous)  # at the prompt, Ctrl+C is a KeyboardInterrupt again
                    try:
                        ask_approval(session, event.approval)
                    finally:
                        signal.signal(signal.SIGINT, on_sigint)
                case TurnStarted(path=path) if path != "main":
                    print(dim(f"[{path} {session.state().agents[path].profile}: started]"))
                case TurnEnded(path=path) if path != "main":
                    print(dim(f"[{path}: {event.end or 'failed'}]"))
                case Compaction(path="main"):
                    compactions += 1
                    print(dim(compaction_line(event)))
                case JobEnded():
                    break
    finally:
        signal.signal(signal.SIGINT, previous)

    if event.error:
        print(f"[error] {event.error}")
    elif event.kind == "compact" and not compactions:  # an outcome no compaction record holds
        print(dim(f"[compaction {event.result}]"))
    elif event.kind == "turn" and event.result != "answered":
        print(f"[{event.result}]")
    state = session.state()
    if calls := state.calls[start:]:
        print(dim(f"{usage_line(calls)} · context ~{human(state.agents['main'].context_tokens)}"))
    return event


def print_cost(session: Session) -> None:
    """The session's spend (the agent and any subagents), where it went once there are several agents, and the
    key's credits."""
    state = session.state()
    print(f"session: ${state.agents['main'].spent:.4f} over {len(state.calls)} model calls")
    if len(state.agents) > 1:
        for line in breakdown_lines(session.breakdown()):
            print(dim(line))
    try:
        key = session.credits()
    except Exception as e:  # noqa: BLE001 - the credits line is optional
        print(f"credits unavailable: {type(e).__name__}: {e}")
        return
    if key is None:  # only OpenRouter reports credits
        return
    line = f"API key: ${key.get('usage') or 0:.2f} used"
    if key.get("limit") is not None:
        line += f" of ${key['limit']:.2f} (${key.get('limit_remaining') or 0:.2f} left)"
    print(line)


def main() -> None:
    """Parse the command line, start the session and chat with it until the user exits."""
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
    parser.add_argument("--reasoning", default="medium", help="reasoning effort: low, medium or high")
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
    try:
        settings = Settings(
            kind=cli.agent,
            sandbox=cli.sandbox,
            network=not cli.no_network,
            model=cli.model,
            reasoning=cli.reasoning,
            max_cost=cli.max_cost,
            compact_at=cli.compact_at,
        )
    except ValidationError as e:
        parser.error("; ".join(f"--{'-'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()))

    session = Session(settings)
    try:
        repl(session)
    finally:
        running = sum(a.running for path, a in session.state().agents.items() if path != "main")
        if running:
            print(dim(f"[stopping {running} subagent{'s' if running > 1 else ''}…]"))
        session.close()


if __name__ == "__main__":
    main()
