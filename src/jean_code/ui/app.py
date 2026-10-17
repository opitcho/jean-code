"""Streamlit UI: a live chat with an agent and its tool calls, plus an inspector.

Run with: uv run --env-file .env streamlit run src/jean_code/ui/app.py

The chat updates live while a job or any subagent runs, the reply streaming in as it's written. Every message an
agent has had stays in its chat: those a compaction took out of the context are dimmed, not lost. The sidebar picks
which agent the chat and inspector show: the main agent, or any agent in its tree.
"""

import html
import json
import re
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from jean_code.agents.coding import coding_agent
from jean_code.agents.experts import orchestrator_agent
from jean_code.config import USAGE_LOG
from jean_code.core import llm
from jean_code.core.agent import DEFAULT_MODEL, Agent
from jean_code.core.compaction import HEAD_MARKER_TAG, SUMMARY_TAG
from jean_code.core.driver import Approve, Busy, Driver, denied
from jean_code.core.events import Compaction
from jean_code.core.node import AgentNode
from jean_code.core.usage import human, usage_line
from jean_code.core.view import AgentState, AgentView, Item, TreeState
from jean_code.logs import Recorder, process_logs

MAIN = "main"  # the main agent's path
AGENT_KINDS = ("Coding", "Orchestrator")  # the sidebar's Agent select; the orchestrator has no shell
REASONING_LEVELS = ("off", "low", "medium", "high")  # "off" sends no reasoning setting
POLL_SECONDS = 0.5
CSS_PATH = Path(__file__).with_name("app.css")

# kind -> (label, icon, accent color)
KINDS = {
    "system": ("SYSTEM", "⚙", "#94A3B8"),
    "user": ("USER", "◉", "#38BDF8"),
    "thinking": ("THINKING", "✦", "#A78BFA"),
    "tool_call": ("TOOL CALL", "⚒", "#FBBF24"),
    "tool_result": ("TOOL RESULT", "↳", "#34D399"),
    "agent": ("AGENT", "◆", "#F472B6"),
    "approval": ("APPROVAL", "⚠", "#FB923C"),
    "marker": ("HEAD END", "┄", "#94A3B8"),
    "summary": ("SUMMARY", "≡", "#2DD4BF"),
}
REASONING_TAIL = 400  # characters of a streaming thought its closed row shows: its latest ones
# status -> (color in the agent picker, dot and pill classes in the header). A turn's other stops use "stopped".
STATUSES = {
    "running": ("green", "run", "status-run"),
    "awaiting approval": ("orange", "wait", "status-wait"),
    "idle": ("gray", "", ""),
    "answered": ("violet", "done", ""),
    "stopped": ("orange", "stop", "status-stop"),
    "failed": ("red", "fail", "status-fail"),
}
ERROR_PREFIXES = ("Error", "Denied by user", "Not run")  # results the agent writes when a call didn't succeed
EXIT_CODE = re.compile(r"exit code: (\d+)[^\n]*\n?")  # first line of a bash result, see shell.format_result
LONG_ARG = 80  # string arguments longer than this get their own code block in the inspector
STATE_CONFIG = ("model", "reasoning", "max_tool_rounds", "max_cost")
CONTEXT_CONFIG = ("compact_at", "keep_first", "keep_last", "context_window", "head_end")
USAGE_COLUMNS = {
    "time": st.column_config.TextColumn("time"),
    "provider": st.column_config.TextColumn("provider"),
    "prompt_tokens": st.column_config.NumberColumn("in"),
    "cached_tokens": st.column_config.NumberColumn("cached"),
    "completion_tokens": st.column_config.NumberColumn("out"),
    "reasoning_tokens": st.column_config.NumberColumn("reasoning"),
    "cost": st.column_config.NumberColumn("cost", format="$%.5f"),
}


def driver() -> Driver:
    return st.session_state.driver


def results_waiting() -> int:
    """Returns how many background subagent results the main agent hasn't received yet."""
    subagents = driver().agent.subagents
    return subagents.pending if subagents else 0


def to_jsonable(value):
    """Converts models, dataclasses and containers into something st.json renders.

    Best effort: anything else becomes its `str`.
    """
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    return str(value)


def mget(msg, key, default=None) -> Any:
    """Reads `key` from a message, whether it's a dict or an attribute-style object."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def turn_kind(msg) -> str:
    """Returns a message's kind: by its calls, then its thinking, then its role."""
    role = mget(msg, "role")
    if mget(msg, "tool_calls"):
        return "tool_call"
    if mget(msg, "reasoning") and not mget(msg, "content"):
        return "thinking"
    content = mget(msg, "content")
    if role == "user" and isinstance(content, str) and content.startswith(HEAD_MARKER_TAG):
        return "marker"
    if role == "user" and isinstance(content, str) and content.startswith(SUMMARY_TAG):
        return "summary"
    if role in ("tool", "user", "system"):
        return {"tool": "tool_result"}.get(role, role)
    return "agent"


def parse_args(call) -> dict:
    """Returns a tool call's arguments, parsed from the JSON string the model sent.

    Arguments that don't parse as a JSON object read as none.
    """
    try:
        value = json.loads(call.function.arguments or "{}")
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def is_error(msg) -> bool:
    """Returns whether a tool result reports a failure.

    A failure is a tool error, a denial, a skipped call or a non-zero exit code.
    """
    content = str(mget(msg, "content") or "")
    exit_code = EXIT_CODE.match(content)
    return content.startswith(ERROR_PREFIXES) or (exit_code is not None and exit_code.group(1) != "0")


def one_line(text) -> str:
    return " ".join(str(text).split())


def clip(text, limit: int) -> str:
    text = one_line(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def first_arg(args: dict) -> str:
    """Returns a call's first argument as one line; "" if it has none.

    For example `ls -la` for a bash call.
    """
    first = next(iter(args.values()), None)
    if first is None:
        return ""
    return one_line(first if isinstance(first, str) else json.dumps(first))


def result_line(msg) -> str:
    """Returns a result's first line of output.

    A failed command's exit code comes before the line.
    """
    content = str(mget(msg, "content") or "")
    exit_code = EXIT_CODE.match(content)
    if exit_code:
        content = content[exit_code.end() :]
    first = next((line for line in content.splitlines() if line.strip()), "(no output)")
    failed = f"exit {exit_code.group(1)} · " if exit_code and exit_code.group(1) != "0" else ""
    return failed + first


def call_names(entries: list[Item]) -> dict[str, str]:
    """Returns each call's tool name by call id, to match a result by `tool_call_id`."""
    return {call.id: call.function.name for e in entries for call in mget(e.message, "tool_calls") or []}


def compaction_of(agent: AgentState, msg) -> Compaction | None:
    """Returns the compaction whose summary is `msg`, if any."""
    return next((c for c in agent.compactions if c.summary is msg), None)


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


def money(cost: float | None) -> str:
    return "—" if cost is None else f"${cost:.5f}"


def own_cost(agent: AgentState) -> float:
    """Returns what the agent's own calls cost; unknown costs count as 0."""
    return sum(c.cost or 0 for c in agent.calls)


def context_size(agent: AgentState, compact_at: int | None) -> str:
    """Returns the context size against the compaction threshold: `~24.0k / 256.0k`."""
    size = f"~{human(agent.context_tokens)}"
    return f"{size} / {human(compact_at)}" if compact_at else size


# ---------------------------------------------------------------- what the state means for the screen


def is_running(state: TreeState, path: str) -> bool:
    """Returns whether the agent at `path` is working, not waiting for the user.

    The main agent works while its job runs and no call waits for approval; any other agent during its turn.
    """
    if path == MAIN:
        return state.busy and state.approval is None
    return state.agents[path].running


def status_of(state: TreeState, path: str) -> tuple[str, str | None]:
    """Returns the agent's status, and a failure's error as its detail.

    The status is "running", "awaiting approval", how its last turn ended ("answered", a stop reason, "failed"),
    or else "idle".
    """
    agent = state.agents[path]
    if path == MAIN and state.approval is not None:
        return "awaiting approval", None
    if is_running(state, path):
        return "running", None
    if agent.error:
        return "failed", agent.error
    return (agent.end or "idle"), None


def style(status: str) -> tuple[str, str, str]:
    return STATUSES.get(status, STATUSES["stopped"])


def signature(state: TreeState) -> tuple:
    """Returns what the sidebar and the input show, to tell when they're stale.

    That's each agent and whether it runs, whether the session is busy or waits for approval, and how many results
    wait. A change means a full rerun, so they catch up.
    """
    agents = tuple((path, a.running) for path, a in state.agents.items())
    return agents, state.busy, state.approval is not None, results_waiting()


def live(state: TreeState) -> bool:
    """Returns whether anything can change without the user; the view then polls."""
    return state.busy or any(a.running for a in state.agents.values())


# ---------------------------------------------------------------- controls (widget callbacks)


def reset_view() -> None:
    """Forgets what the UI showed of the previous chat."""
    st.session_state.open = {}  # path -> keys of the rows opened in its chat
    st.session_state.view = MAIN  # the agent picker's value
    st.session_state.compactions_seen = 0
    st.session_state.last_job_seen = None


def orchestrating() -> bool:
    """Returns whether the sidebar's Agent select is on the orchestrator."""
    return st.session_state.get("agent_kind", AGENT_KINDS[0]) == "Orchestrator"


def reasoning_effort() -> str | None:
    """Returns the sidebar's reasoning level as `Agent.reasoning` takes it.

    "off" becomes None.
    """
    level = st.session_state.get("reasoning", "medium")
    return None if level == "off" else level


def choice() -> tuple[bool, bool]:
    """Returns what the sidebar picks to build: (orchestrator, Docker sandbox)."""
    return orchestrating(), st.session_state.get("sandbox", False)


def build_agent(node: AgentNode, approve: Approve) -> Agent:
    """Builds the agent the sidebar picks, on `node`; it streams its replies."""
    common = {"reasoning": reasoning_effort(), "node": node}
    orchestrator, sandbox = choice()
    if orchestrator:  # nothing it calls needs approval
        return orchestrator_agent(DEFAULT_MODEL, **common)
    return coding_agent(DEFAULT_MODEL, sandbox=sandbox, approve=approve, **common)


def new_driver() -> Driver:
    """Builds a new chat's driver, with a recorder for its trace and the usage log."""
    st.session_state.built = choice()
    view = AgentView()
    recorder = Recorder(view, process_logs.new_tree(), USAGE_LOG)  # before the agent: it gets every event
    try:
        driver = Driver(build_agent, view=view)
    except BaseException:
        recorder.close()
        raise
    st.session_state.recorder = recorder
    return driver


def apply_settings() -> None:
    """Applies the sidebar's settings.

    A new agent kind or sandbox switch starts a new chat; reasoning applies from the next request.
    """
    if choice() != st.session_state.get("built", choice()):
        new_chat()
    else:
        driver().configure(reasoning=reasoning_effort())


def new_chat() -> None:
    driver().close()
    if recorder := st.session_state.get("recorder"):
        recorder.close()
    st.session_state.driver = new_driver()
    reset_view()


def opened() -> set[str]:
    """Returns the keys of the rows open in the chat of the agent on view."""
    return st.session_state.open.setdefault(st.session_state.view, set())


def toggle(key: str) -> None:
    """Opens the row `key` in the chat of the agent on view, or closes it."""
    opened().symmetric_difference_update({key})


def show(path: str) -> None:
    """Shows the agent at `path`.

    It's a button callback because the picker's value can only change before the picker is drawn.
    """
    st.session_state.view = path


# ---------------------------------------------------------------- html snippets (styles in app.css)


def kind_css() -> str:
    """Returns each kind's accent as `--c` CSS, for its blocks and `k-<kind>` tags.

    Errors override it.
    """
    rules = "".join(f'[class*="st-key-blk-{k}-"], .k-{k} {{ --c:{c}; }}\n' for k, (_, _, c) in KINDS.items())
    return f'<style>{rules}[class*="st-key-blk-"][class*="-err-"], .err {{ --c:var(--err); }}</style>'


def pill(text, *, bold=None, cls: str = "", dot: str | None = None) -> str:
    """Returns a rounded label: an optional status dot and bold value, then the text."""
    dot_html = f'<span class="dot {dot}"></span>' if dot is not None else ""
    bold_html = f"<b>{html.escape(str(bold))}</b> " if bold is not None else ""
    return f'<span class="pill {cls}">{dot_html}{bold_html}{html.escape(str(text))}</span>'


def chips(items, *, on=(), empty_text: str = "none") -> str:
    """Returns a wrapping row of pills; items in `on` are highlighted."""
    if not items:
        return f'<div class="row">{pill(empty_text)}</div>'
    return '<div class="row">' + "".join(pill(i, cls="mono on" if i in on else "mono") for i in items) + "</div>"


def row(
    kind: str, body: str, *, name: str = "", aside: str = "", is_open: bool | None = None, classes: str = ""
) -> str:
    """Returns a one-line chat row: an icon, a tool's name or a label, then `body`.

    The icon and label are the kind's. `aside` goes on the right, and a chevron after it if the row opens (`is_open`
    not None).
    """
    label, icon, _ = KINDS[kind]
    lead = f'<span class="tl-name">{html.escape(name)}</span>' if name else f'<span class="tl-label">{label}</span>'
    mono = " mono" if kind in ("tool_call", "tool_result") else ""
    aside_html = f'<span class="tl-aside">{aside}</span>' if aside else ""
    chevron = "" if is_open is None else f'<span class="tl-chev">{"▾" if is_open else "▸"}</span>'
    return (
        f'<div class="tl-row k-{kind}{" open" if is_open else ""} {classes}"><span class="tl-icon">{icon}</span>'
        f'{lead}<span class="tl-body{mono}">{html.escape(body)}</span>{aside_html}{chevron}</div>'
    )


def kv_grid(rows: dict, fmt=json.dumps) -> str:
    """Returns a two-column key/value table.

    Values go through `fmt`: JSON by default, so strings show quoted.
    """
    cells = (
        "".join(f'<div class="k">{html.escape(str(k))}</div><div>{html.escape(fmt(v))}</div>' for k, v in rows.items())
        or '<div class="k">—</div><div>none</div>'
    )
    return f'<div class="kv">{cells}</div>'


def eyebrow(text: str) -> str:
    """Returns a small uppercase section label."""
    return f'<div class="eyebrow">{html.escape(text)}</div>'


def empty(icon: str, text: str) -> None:
    st.html(f'<div class="empty"><div class="big">{icon}</div>{text}</div>')


# ---------------------------------------------------------------- rendering


def render_sidebar_top(state: TreeState) -> None:
    """Draws the brand, the Agent select and, for the coding agent, the sandbox toggle.

    Both controls start a new chat, so they're disabled while a job runs.
    """
    with st.sidebar:
        st.html(
            '<div class="brand"><div class="brand-mark">🧮</div>'
            '<div><div class="brand-name">jean-code</div><div class="brand-sub">agent inspector</div></div></div>'
        )
        st.html(eyebrow("ENVIRONMENT"))
        st.selectbox(
            "Agent",
            AGENT_KINDS,
            key="agent_kind",
            on_change=apply_settings,
            disabled=state.busy,
            help="Coding: a shell, web tools and helper subagents. Orchestrator: no shell; it runs expert subagents "
            "(planner, pm, architect, engineer, auditor), each in its own sandbox. Starts a new chat.",
        )
        if orchestrating():
            return  # no shell, so no sandbox question
        st.toggle(
            "Run bash in Docker sandbox",
            key="sandbox",
            on_change=apply_settings,
            disabled=state.busy,
            help="Off: commands run on this machine in the directory the app was started from, and need "
            "approval. On: they run in the jean-code-sandbox container without asking. Starts a new chat.",
        )


def render_sidebar(state: TreeState) -> None:
    """Draws the rest of the sidebar: shell, skills, model, reasoning, budget, agents."""
    if orchestrating():
        note = None  # no shell
    elif choice()[1]:
        note = "<b>Docker sandbox</b> — commands run without asking"
    else:
        note = "<b>Local shell</b> — each command needs approval"
    agent = driver().agent
    skills = sorted(agent.skills.catalog)
    main = state.agents[MAIN]
    with st.sidebar:
        if note:
            st.html(f'<div class="shell-note">{note}</div>')
        st.html(eyebrow(f"SKILLS · {len(skills)}"))
        st.html(chips(skills, on=agent.skills.loaded, empty_text="no skills found"))
        st.html(eyebrow("SESSION"))
        st.html('<div class="row">' + pill(main.model, cls="model") + "</div>")
        if recorder := st.session_state.get("recorder"):
            st.caption(
                f"Log stopped: {'; '.join(recorder.errors)}" if recorder.errors else f"Trace: {recorder.trace_dir}"
            )
        st.segmented_control(
            "Reasoning effort",
            REASONING_LEVELS,
            default="medium",
            required=True,
            key="reasoning",
            on_change=apply_settings,
            help="Sent with each of the main agent's requests, from the next one on, "
            "so it can change mid-chat. Subagents keep their own (medium).",
        )
        if limit := agent.max_cost:  # the agent and any subagents
            st.progress(min(main.spent / limit, 1.0), text=f"${main.spent:.4f} of ${limit:.2f} budget")
        render_agents(state)


def agent_label(state: TreeState, path: str) -> str:
    """Returns the picker's line for an agent: its name, subagents indented by depth.

    It must never change, status included: the browser sends the picked label back, and a label that no longer
    matches an option loses the pick.
    """
    if path == MAIN:
        return f"**{path}**"
    indent = " " * (path.count("/") - 1)  # em spaces: markdown collapses plain ones
    return f"{indent}↳ **#{path.rsplit('/', 1)[-1]}** {state.agents[path].profile}"


def agent_caption(state: TreeState, path: str) -> str:
    """Returns the picker's second line for an agent: status, calls, spend and task.

    That's a status dot and the status, the agent's own calls and spend, and a subagent's task.
    """
    status, detail = status_of(state, path)
    agent = state.agents[path]
    parts = [
        f":{style(status)[0]}[●] " + (f"{status} ({clip(detail, 40)})" if detail else status),
        f"{len(agent.calls)} calls · ${own_cost(agent):.4f}",
    ]
    if path != MAIN:
        parts.append(clip(re.sub(r"[*_`#>\[\]]+", "", agent.task), 48))  # plain text: captions are markdown
    return " · ".join(parts)


def render_agents(state: TreeState) -> None:
    """Draws the agent picker, and a notice for results waiting to be delivered.

    The picker lists the main agent and every agent in its tree. It's refreshed by full reruns, which a change of
    `signature` triggers.
    """
    paths = list(state.agents)
    st.html(eyebrow(f"AGENTS · {len(paths)}"))
    st.radio(
        "Agent to inspect",
        paths,
        key="view",
        format_func=lambda path: agent_label(state, path),
        captions=[agent_caption(state, p) for p in paths],
        label_visibility="collapsed",
    )
    if len(paths) == 1:
        st.caption("No subagents started yet.")
    if pending := results_waiting():
        st.info(
            f"{pending} subagent result{'s' if pending > 1 else ''} waiting; your next message delivers "
            f"{'them' if pending > 1 else 'it'}.",
            icon=":material/inbox:",
        )


def render_header(state: TreeState, path: str, entries: list[Item], ctx: list[Item]) -> None:
    """Draws the header of the agent on view: its status, counts, context and actions.

    The Inspector toggle opens the inspector beside the chat. The other action is Stop or New chat for the main
    agent, and Cancel run or Back to main for a subagent. Below the header goes a failure, or why the agent stopped
    before answering.
    """
    agent = state.agents[path]
    tool_calls = sum(turn_kind(e.message) == "tool_call" for e in entries)
    compacted = sum(not e.in_context for e in entries)
    status, detail = status_of(state, path)
    _, dot, cls = style(status)
    pills = [pill(path, bold=f"#{path.rsplit('/', 1)[-1]} {agent.profile}", cls="on")] if path != MAIN else []
    pills += [pill(agent.model, cls="model"), pill(status, dot=dot, cls=cls)]
    pills += [
        pill(label, bold=n)
        for n, label in (
            (len(entries), "turns"),
            (tool_calls, "tool calls"),
            (len(ctx), "in context"),
            (compacted, "compacted"),
        )
        if n
    ]
    pills.append(pill("context", bold=context_size(agent, driver().agent_at(path).compact_at), cls="mono"))
    if agent.calls:
        pills.append(pill(usage_line(agent.calls), cls="mono"))

    with st.container(horizontal=True, vertical_alignment="center", wrap=False):
        st.html(f'<div class="hdr"><span class="hdr-title">jean-code</span>{"".join(pills)}</div>')
        st.toggle("Inspector", key="inspector", help="The agent's last request, its context and its state.")
        if path == MAIN:
            if state.busy:  # also while a call waits for approval: Stop answers it
                st.button("Stop", icon=":material/stop_circle:", on_click=driver().cancel, type="primary")
            else:
                st.button("New chat", icon=":material/refresh:", on_click=new_chat)
        elif agent.running:
            st.button(
                "Cancel run",
                icon=":material/cancel:",
                on_click=driver().cancel,
                args=(path,),
                type="primary",
                help="Stops this subagent, and any it started, after the model call it is making.",
            )
        else:
            st.button("Back to main", icon=":material/arrow_back:", on_click=show, args=(MAIN,))

    if status == "failed":
        st.error(detail, icon=":material/error:")
    elif path == MAIN and state.last_job and state.last_job.error and not state.busy:
        st.error(state.last_job.error, icon=":material/error:")
    elif status not in ("running", "awaiting approval", "idle", "answered", *(("cancelled",) if path == MAIN else ())):
        st.warning(f"Stopped before answering: {status}", icon=":material/warning:")


def fold(key: str, row_html) -> bool:
    """Draws a row that opens and closes on click; returns whether it's open.

    `row_html` takes whether the row is open and returns its HTML. What it opens to is drawn by the caller, after.
    """
    is_open = key in opened()
    with st.container(key=f"head-{key}"):
        st.html(row_html(is_open))
        st.button(" ", key=f"hit-{key}", on_click=toggle, args=(key,), width="stretch")
    return is_open


def render_thought(key: str, text: str, out: str) -> None:
    """Draws a thought as a row, which opens to the thought in full."""
    with st.container(key=f"blk-thinking-{out}{key}"):
        if fold(key, lambda is_open: row("thinking", "" if is_open else one_line(text), is_open=is_open)):
            with st.container(key=f"detail-{key}"):
                st.markdown(text)


def render_call(key: str, name: str, args: dict, result: Item | None, waiting: str, out: str) -> None:
    """Draws a tool call and its result as one row, which opens to their content.

    That's the arguments, then the output. Without a result, `waiting` says why: "running…", "awaiting approval"
    or "no result".
    """
    err = result is not None and is_error(result.message)
    if result is None:
        aside, classes = html.escape(waiting), "run" if waiting == "running…" else ""
    else:
        aside, classes = ("✗ " if err else "✓ ") + html.escape(clip(result_line(result.message), 60)), ""
    with st.container(key=f"blk-tool_call-{'err-' if err else ''}{out}{key}"):
        if fold(
            key,
            lambda is_open: row("tool_call", first_arg(args), name=name, aside=aside, is_open=is_open, classes=classes),
        ):
            with st.container(key=f"detail-{key}"):
                render_args(args)
                if result is not None:
                    st.html(eyebrow("OUTPUT"))
                    render_output(result)


def render_item(agent: AgentState, e: Item, results: dict[str, Item], waiting: str) -> None:
    """Draws one message: its thought, its text, and each of its calls with its result.

    Users' messages are bubbles and an agent's text is prose; other messages are rows that open to their content.
    A call without a result shows `waiting`.
    """
    msg, kind = e.message, turn_kind(e.message)
    out = "" if e.in_context else "out-"
    if reasoning := mget(msg, "reasoning"):
        render_thought(f"r{e.uid}", reasoning, out)
    content = mget(msg, "content")
    if kind in ("user", "agent") or (kind == "tool_call" and content):
        with st.container(key=f"blk-{'user' if kind == 'user' else 'agent'}-{out}{e.uid}"):
            st.markdown(content or "*(no content)*")
    elif kind in ("system", "marker", "summary"):
        key = f"m{e.uid}"
        with st.container(key=f"blk-{kind}-{out}{key}"):
            if fold(key, lambda is_open: row(kind, "" if is_open else one_line(content or ""), is_open=is_open)):
                with st.container(key=f"detail-{key}"):
                    if kind == "summary" and (rec := compaction_of(agent, msg)):
                        st.html(
                            kv_grid(
                                {
                                    "messages removed": rec.removed,
                                    "tokens": f"~{human(rec.tokens_before)} → ~{human(rec.tokens_after or 0)}",
                                    "summary call": money(rec.cost),
                                },
                                fmt=str,
                            )
                        )
                    st.markdown(content or "*(no content)*")
    if kind == "tool_call":
        for i, call in enumerate(mget(msg, "tool_calls")):
            render_call(f"c{e.uid}-{i}", call.function.name, parse_args(call), results.get(call.id), waiting, out)
    elif kind == "tool_result":  # a result whose call isn't in the timeline
        render_call(f"c{e.uid}", "tool", {}, e, "", out)


def render_draft(agent: AgentState) -> None:
    """Draws the reply streaming in: its thought, then its text so far.

    The thought is a row showing its latest words, which opens to the thought in full as it grows. It stays open
    once the reply lands.
    """
    draft = agent.draft
    if not draft:
        with st.container(key="blk-thinking-live"):
            st.html(row("thinking", "thinking…", classes="ghost"))
        return
    if reasoning := draft.get("reasoning"):
        tail = reasoning if len(reasoning) <= REASONING_TAIL else "…" + reasoning[-REASONING_TAIL:]
        # the reply lands as the agent's next item, so this is the key its thought's row will have
        key = f"r{max((i.uid for i in agent.items), default=0) + 1}"
        with st.container(key="blk-thinking-live"):
            if fold(
                key, lambda is_open: row("thinking", "" if is_open else tail, is_open=is_open, classes="ghost stream")
            ):
                with st.container(key=f"detail-{key}"):
                    st.markdown(reasoning)
    if content := draft.get("content"):
        with st.container(key="blk-agent-live"):
            st.markdown(content)


def render_chat(agent: AgentState, entries: list[Item], running: bool, pending, queued: str | None) -> None:
    """Draws the agent's chat, every message it has had in order.

    A call shows on one row with its result. Compacted messages are dimmed, with a divider where the context
    starts again. After the last message comes the reply streaming in while the agent runs, or the call waiting
    for approval; then a message waiting for the agent to stop.
    """
    names = call_names(entries)
    results = {mget(e.message, "tool_call_id"): e for e in entries if turn_kind(e.message) == "tool_result"}
    last_calls = next((e for e in reversed(entries) if turn_kind(e.message) == "tool_call"), None)
    live = "running…" if running else "awaiting approval" if pending is not None else "no result"
    with st.container(height=640, autoscroll=True, border=False, key="timeline"):
        if not entries and not running:
            empty("✦", "Ask something below — the agent's thoughts, tool calls and answers<br>show up here.")
        with st.container(key="rail"):
            compacted_run = 0
            for e in entries:
                if e.in_context and compacted_run:
                    text = f"⟲ {compacted_run} messages compacted"
                    if rec := compaction_of(agent, e.message):
                        text += f" · ~{human(rec.tokens_before)} → ~{human(rec.tokens_after or 0)} tokens"
                    st.html(f'<div class="divider">{text}</div>')
                compacted_run = 0 if e.in_context else compacted_run + 1
                if turn_kind(e.message) == "tool_result" and mget(e.message, "tool_call_id") in names:
                    continue  # drawn on its call's row
                render_item(agent, e, results, live if e is last_calls else "no result")

            if running:
                render_draft(agent)
            elif pending is not None:
                render_approval(pending)
            if queued is not None:
                with st.container(key="blk-user-queued"):
                    st.markdown(f"*stopping… then:* {queued}")


def render_approval(approval) -> None:
    """Draws the call waiting for approval, with Approve and Deny.

    Sending a message instead denies it, with the message as the reason.
    """
    with st.container(key="blk-approval-pending"):
        st.html(row("approval", f"{approval.tool}() wants to run:"))
        command = approval.arguments.get("command") if approval.tool == "bash" else None
        code = command or json.dumps(approval.arguments, indent=2)
        st.code(code, language="bash" if command else "json", wrap_lines=True)
        reason = st.text_input(
            "Reason",
            key="deny_reason",
            placeholder="optional reason, sent to the model on deny",
            label_visibility="collapsed",
        )
        with st.container(horizontal=True):
            if st.button("Approve", icon=":material/check:", type="primary"):
                driver().answer(approval.id)
                st.rerun()
            if st.button("Deny", icon=":material/block:"):
                driver().answer(approval.id, denied(reason))
                st.rerun()


def render_args(args: dict) -> None:
    """Draws a call's arguments.

    Short ones go in a key/value table; commands and long or multi-line strings in code blocks.
    """

    def is_long(k, v):
        return isinstance(v, str) and (k == "command" or "\n" in v or len(v) > LONG_ARG)

    short = {k: v for k, v in args.items() if not is_long(k, v)}
    if short or not args:
        st.html(kv_grid(short))
    for k, v in args.items():
        if is_long(k, v):
            st.html(eyebrow(k))
            st.code(v, language="bash" if k == "command" else None, wrap_lines=True)


def render_output(result: Item) -> None:
    """Draws a tool result's content.

    It's pretty JSON when it parses as JSON, height-capped when long, and red when the call failed.
    """
    content, language = str(mget(result.message, "content")), None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, (dict, list)):
            content, language = json.dumps(parsed, indent=2), "json"
    except ValueError:
        pass
    height = 420 if content.count("\n") > 20 else "content"
    with st.container(key=f"errout-{result.uid}") if is_error(result.message) else nullcontext():
        st.code(content, language=language, wrap_lines=True, height=height)


def render_request(path: str, agent: AgentState) -> None:
    """Draws the agent's last request: model, tools, last call, usage and payload."""
    request = driver().agent_at(path).last_request
    req = llm.render_request(request) if request else None
    if not req:
        empty("⇅", "No request sent yet.")
        return
    tools = [(t.get("function") or {}).get("name", "?") for t in req.get("tools", [])]
    reasoning = req.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else reasoning
    st.html(
        '<div class="hdr">'
        + pill(req.get("model"), cls="model")
        + pill("reasoning", bold=effort or "none")
        + pill("messages", bold=len(req.get("messages", [])))
        + "</div>"
    )
    st.html(eyebrow(f"TOOLS · {len(tools)}") + chips(tools))

    if calls := agent.calls:
        last = calls[-1]
        st.html(
            eyebrow("LAST CALL")
            + kv_grid(
                {
                    "provider": last.provider or "—",
                    "tokens in": f"{human(last.prompt_tokens)} ({human(last.cached_tokens)} cached)",
                    "tokens out": f"{human(last.completion_tokens)} ({human(last.reasoning_tokens)} reasoning)",
                    "cost": money(last.cost),
                },
                fmt=str,
            )
        )
        st.html(eyebrow(f"USAGE · {len(calls)} CALLS · {usage_line(calls)}"))
        rows = [{**asdict(c), "time": c.time[11:19]} for c in reversed(calls)]  # ISO time -> HH:MM:SS
        st.dataframe(
            rows,
            hide_index=True,
            column_order=list(USAGE_COLUMNS),
            column_config=USAGE_COLUMNS,
            height=min(38 + 35 * len(calls), 280),
        )

    st.html(eyebrow("PAYLOAD"))
    with st.expander("Messages sent", icon=":material/forum:"):
        st.json(to_jsonable(req.get("messages", [])), expanded=1)
    with st.expander("Full request", icon=":material/data_object:"):
        st.json(to_jsonable(req), expanded=1)


def render_context(path: str, agent: AgentState) -> None:
    """Draws the context's size and settings, and every compaction attempt.

    The size is shown against the compaction threshold.
    """
    details = driver().agent_at(path)
    if details.compact_at:
        st.progress(
            min(agent.context_tokens / details.compact_at, 1.0),
            text=f"context {context_size(agent, details.compact_at)} tokens "
            "(compacts after a turn ending over the threshold)",
        )
    else:
        st.html(pill("context", bold=context_size(agent, None), cls="mono") + pill("automatic compaction off"))
    st.html(eyebrow("SETTINGS") + kv_grid({k: getattr(details, k) for k in CONTEXT_CONFIG}))

    st.html(eyebrow(f"COMPACTIONS · {len(agent.compactions)}"))
    if not agent.compactions:
        st.caption("None yet. Type /compact to compact now.")
        return
    rows = [
        {
            "#": i + 1,
            "status": c.status,
            "removed": c.removed,
            "before": human(c.tokens_before),
            "after": human(c.tokens_after) if c.tokens_after is not None else "—",
            "cost": money(c.cost),
        }
        for i, c in enumerate(agent.compactions)
    ]
    st.dataframe(rows, hide_index=True)
    for i, c in enumerate(agent.compactions):
        if c.summary is not None:
            with st.expander(f"Summary #{i + 1}", icon=":material/summarize:"):
                st.markdown(c.summary.content)


def render_state(path: str, agent: AgentState) -> None:
    """Draws the agent's config, the tools needing approval, its skills and its state."""
    details = driver().agent_at(path)
    st.html(eyebrow("CONFIG") + kv_grid({k: getattr(details, k) for k in STATE_CONFIG}))
    st.html(eyebrow("NEEDS APPROVAL") + chips(sorted(details.needs_approval)))
    st.html(eyebrow("LOADED SKILLS") + chips(details.skills.loaded))
    summary = {
        "path": agent.path,
        "profile": agent.profile,
        "task": agent.task,
        "running": agent.running,
        "end": agent.end,
        "error": agent.error,
        "messages": len(agent.items),
        "in context": len(agent.context),
        "calls": len(agent.calls),
        "spent (with its subagents)": round(agent.spent, 6),
        "context tokens": agent.context_tokens,
    }
    st.html(eyebrow("AGENT STATE"))
    st.json(to_jsonable(summary), expanded=1)


def announce(state: TreeState) -> None:
    """Toasts the main agent's compaction attempts since the last full rerun.

    It also toasts the outcome of a /compact that left no record ("nothing to compact", a refusal). It runs in
    the full script, which reruns when a job ends.
    """
    records = state.agents[MAIN].compactions
    new = records[st.session_state.compactions_seen :]
    for rec in new:
        st.toast(compaction_line(rec), icon=":material/compress:")
    st.session_state.compactions_seen = len(records)
    job = state.last_job
    if job is not st.session_state.last_job_seen:
        st.session_state.last_job_seen = job
        if job is not None and job.kind == "compact" and not any(rec.status == job.result for rec in new):
            st.toast(f"[compaction {job.result or job.error}]", icon=":material/compress:")


# ---------------------------------------------------------------- app

st.set_page_config(page_title="jean-code", page_icon="🧮", layout="wide")
st.html(CSS_PATH)
st.html(kind_css())

if "driver" not in st.session_state:
    st.session_state.driver = new_driver()
if "open" not in st.session_state:
    reset_view()
state = driver().view.state()
render_sidebar_top(state)
render_sidebar(state)
announce(state)
st.session_state.signature_seen = signature(state)


@st.fragment(run_every=POLL_SECONDS if live(state) else None)
def live_view() -> None:
    """Draws the header, chat and, if toggled on, inspector of the agent on view.

    It polls while anything runs, and reruns the whole script when `signature` changes.
    """
    state = driver().view.state()
    if signature(state) != st.session_state.signature_seen:
        st.rerun()  # an agent started or stopped, a job ended, or a result is waiting: the sidebar catches up

    path = st.session_state.view if st.session_state.view in state.agents else MAIN
    agent = state.agents[path]
    entries = agent.items
    running = is_running(state, path)
    approval = state.approval if path == MAIN else None  # only the main agent asks

    inspecting = st.session_state.get("inspector", False)
    page = st.container(key="page-wide" if inspecting else "page")  # without the inspector: one centred column
    with page:
        render_header(state, path, entries, [e for e in entries if e.in_context])
    if inspecting:
        chat_col, inspector_col = page.columns([3, 2], gap="large")
    else:
        chat_col, inspector_col = page.container(), None

    with chat_col:
        if path != MAIN:
            st.html(eyebrow(f"CHAT · {agent_label(state, path).strip().replace('**', '')}"))
        queued = state.queued if path == MAIN else None
        render_chat(agent, entries, running, approval, queued)
        if path != MAIN:
            st.chat_input("Subagents take no messages: pick the main agent to chat", disabled=True)
        elif prompt := st.chat_input(
            "Stopping…"
            if queued is not None
            else "Send to stop and redirect…"
            if running
            else "Approve or deny the call, or say why not…"
            if approval
            else "Message the agent…",
            disabled=queued is not None,
        ):
            try:
                if approval is not None:
                    driver().answer(approval.id, denied(prompt))
                elif prompt.strip() == "/compact":  # a UI command: never sent to the model as a message
                    driver().compact()
                else:
                    driver().send(prompt)  # while busy: stops the turn, and this message's turn runs next
            except Busy:
                st.toast("The agent is busy: stop it, or wait for it to finish.", icon=":material/hourglass_top:")
            st.rerun()

    if inspector_col is not None:
        with inspector_col:
            request_tab, context_tab, state_tab = st.tabs(["Request", "Context", "Agent state"])
            with request_tab:
                render_request(path, agent)
            with context_tab:
                render_context(path, agent)
            with state_tab:
                render_state(path, agent)


live_view()
