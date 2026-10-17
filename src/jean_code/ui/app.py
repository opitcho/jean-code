"""Streamlit UI for the agent: a live, clickable timeline of turns plus an inspector.

Run with: uv run --env-file .env streamlit run src/jean_code/ui/app.py

The app drives the agent through a `Session` kept in `st.session_state.session`, and draws `session.state()` on each
rerun, polling while a job or any subagent runs. Every message an agent has had stays in its timeline: those a
compaction took out of the context are dimmed, not lost.

The sidebar picks which agent the timeline and inspector show: the main agent, or any agent in its tree.
"""

import html
import json
import re
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from jean_code.session import (
    HEAD_MARKER_TAG,
    SUMMARY_TAG,
    AgentState,
    Compaction,
    Item,
    Session,
    SessionState,
    Settings,
    denied,
    human,
    render_message,
    usage_line,
)

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
MONO_KINDS = {"tool_call", "tool_result"}
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


def session() -> Session:
    return st.session_state.session


def to_jsonable(value):
    """Best-effort conversion of models, dataclasses and containers into something st.json can render."""
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
    """Read `key` from a message whether it's a plain dict or an attribute-style object."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def turn_kind(msg) -> str:
    """Classify a transcript message by its most important part: calls, then thinking, then role."""
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
    """A tool call's arguments: the agent stores them as the model sent them, a JSON string."""
    try:
        value = json.loads(call.function.arguments or "{}")
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def is_error(msg) -> bool:
    """A tool result that reports a failure: a tool error, a denial, a skipped call or a non-zero exit code."""
    content = str(mget(msg, "content") or "")
    exit_code = EXIT_CODE.match(content)
    return content.startswith(ERROR_PREFIXES) or (exit_code is not None and exit_code.group(1) != "0")


def one_line(text) -> str:
    return " ".join(str(text).split())


def clip(text, limit: int) -> str:
    text = one_line(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def call_summary(call) -> str:
    """`name · first argument`, e.g. `bash · ls -la`."""
    args = parse_args(call)
    first = next(iter(args.values()), None)
    if first is None:
        return call.function.name
    return f"{call.function.name} · {first if isinstance(first, str) else json.dumps(first)}"


def result_summary(msg, name: str) -> str:
    """`name → first line of output`, with the exit code when a command failed."""
    content = str(mget(msg, "content") or "")
    exit_code = EXIT_CODE.match(content)
    if exit_code:
        content = content[exit_code.end() :]
    first = next((line for line in content.splitlines() if line.strip()), "(no output)")
    failed = f"exit {exit_code.group(1)} · " if exit_code and exit_code.group(1) != "0" else ""
    return f"{name} → {failed}{first}"


def preview(msg, kind: str, names: dict, limit: int = 90) -> str:
    """One-line summary of a message for its timeline block."""
    if kind == "tool_call":
        text = "; ".join(call_summary(c) for c in mget(msg, "tool_calls"))
    elif kind == "tool_result":
        text = result_summary(msg, names.get(mget(msg, "tool_call_id"), "tool"))
    elif kind == "thinking":
        text = mget(msg, "reasoning")
    else:
        text = mget(msg, "content") or "(no content)"
    if kind in ("agent", "user"):
        text = re.sub(r"[*_`#>]+", "", str(text))  # drop markdown markers from the one-liner
    return clip(text, limit)


def results_of(entries: list[Item], pos: int) -> list[Item]:
    """The tool results following the call at `pos`, in call order (results match calls by position)."""
    results = []
    for e in entries[pos + 1 :]:
        if turn_kind(e.message) != "tool_result":
            break
        results.append(e)
    return results


def call_of(entries: list[Item], pos: int) -> Item | None:
    """The tool call a result at `pos` belongs to: the nearest call before its run of results."""
    for e in reversed(entries[:pos]):
        kind = turn_kind(e.message)
        if kind == "tool_call":
            return e
        if kind != "tool_result":
            return None
    return None


def call_names(entries: list[Item]) -> dict[str, str]:
    """Tool name of every call seen, by call id, so a result can be matched to its tool by `tool_call_id`."""
    return {call.id: call.function.name for e in entries for call in mget(e.message, "tool_calls") or []}


def compaction_of(agent: AgentState, msg) -> Compaction | None:
    """The compaction whose summary is `msg`, if any."""
    return next((c for c in agent.compactions if c.summary is msg), None)


def compaction_line(event: Compaction) -> str:
    """`[compacted 38 messages · ~210.0k → ~24.0k tokens]`, or why a compaction didn't happen."""
    if event.status == "compacted":
        return (
            f"[compacted {event.removed} messages · ~{human(event.tokens_before)} → "
            f"~{human(event.tokens_after or 0)} tokens]"
        )
    return f"[compaction {event.status}]"


def money(cost: float | None) -> str:
    return "—" if cost is None else f"${cost:.5f}"


def own_cost(agent: AgentState) -> float:
    """What the agent's own calls cost; unknown costs count as 0."""
    return sum(c.cost or 0 for c in agent.calls)


def context_size(agent: AgentState, compact_at: int | None) -> str:
    """`~24.0k / 256.0k`: the context's size against the compaction threshold."""
    size = f"~{human(agent.context_tokens)}"
    return f"{size} / {human(compact_at)}" if compact_at else size


# ---------------------------------------------------------------- what the state means for the screen


def is_running(state: SessionState, path: str) -> bool:
    """Working, not waiting for the user: the main agent while its job runs and nothing waits for approval; any
    other agent during its turn."""
    if path == MAIN:
        return state.busy and state.approval is None
    return state.agents[path].running


def status_of(state: SessionState, path: str) -> tuple[str, str | None]:
    """(status, detail): "running", "awaiting approval", or how the agent's last turn ended ("answered", a stop
    reason, "failed"), else "idle"."""
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


def signature(state: SessionState) -> tuple:
    """What the sidebar and the input show: each agent and whether it runs, whether the session is busy or waits
    for approval, results waiting. A change means a full rerun, so they catch up."""
    agents = tuple((path, a.running) for path, a in state.agents.items())
    return agents, state.busy, state.approval is not None, session().pending()


def live(state: SessionState) -> bool:
    """Whether anything can change without the user: then the view polls."""
    return state.busy or any(a.running for a in state.agents.values())


# ---------------------------------------------------------------- controls (widget callbacks)


def reset_view() -> None:
    """Forget what the UI showed of the previous chat."""
    st.session_state.selected = {}  # path -> selected uid; None or missing follows the latest
    st.session_state.view = MAIN  # the agent picker's value
    st.session_state.compactions_seen = 0
    st.session_state.last_job_seen = None


def orchestrating() -> bool:
    """Whether the sidebar's Agent select is on the orchestrator."""
    return st.session_state.get("agent_kind", AGENT_KINDS[0]) == "Orchestrator"


def reasoning_effort() -> str | None:
    """The sidebar's reasoning level as `Settings.reasoning` takes it: None for "off"."""
    level = st.session_state.get("reasoning", "medium")
    return None if level == "off" else level


def sidebar_settings() -> dict:
    return {
        "kind": "orchestrator" if orchestrating() else "coding",
        "sandbox": st.session_state.get("sandbox", False),
        "reasoning": reasoning_effort(),
    }


def apply_settings() -> None:
    """The sidebar changed: the agent kind or the sandbox switch start a new chat; reasoning applies to the next
    request."""
    before = session().state().settings
    session().configure(sidebar_settings())
    if session().state().settings.kind != before.kind or session().state().settings.sandbox != before.sandbox:
        reset_view()


def new_chat() -> None:
    session().reset()
    reset_view()


def select(uid: int | None) -> None:
    """Select a block in the timeline of the agent on view; None follows the latest."""
    st.session_state.selected[st.session_state.view] = uid


def show(path: str) -> None:
    """Show another agent. A button callback: the picker's value can only change before the picker is drawn."""
    st.session_state.view = path


def answer_approval(approval_id: int, refusal: str | None = None) -> None:
    session().answer(approval_id, refusal)
    select(None)


# ---------------------------------------------------------------- html snippets (styles in app.css)


def kind_css() -> str:
    """Each kind's accent as `--c`, on its timeline block and on anything tagged `k-<kind>`; errors override it."""
    rules = "".join(f'[class*="st-key-blk-{k}-"], .k-{k} {{ --c:{c}; }}\n' for k, (_, _, c) in KINDS.items())
    return f'<style>{rules}[class*="st-key-blk-"][class*="-err-"], .err {{ --c:var(--err); }}</style>'


def pill(text, *, bold=None, cls: str = "", dot: str | None = None) -> str:
    """A rounded label: an optional status dot, an optional bold value, then the text."""
    dot_html = f'<span class="dot {dot}"></span>' if dot is not None else ""
    bold_html = f"<b>{html.escape(str(bold))}</b> " if bold is not None else ""
    return f'<span class="pill {cls}">{dot_html}{bold_html}{html.escape(str(text))}</span>'


def chips(items, *, on=(), empty_text: str = "none") -> str:
    """A wrapping row of pills; items in `on` are highlighted."""
    if not items:
        return f'<div class="row">{pill(empty_text)}</div>'
    return '<div class="row">' + "".join(pill(i, cls="mono on" if i in on else "mono") for i in items) + "</div>"


def card(kind: str, text: str, *, num: int | None = None, classes: str = "", mono: bool = False) -> str:
    """A timeline card: kind label and icon, optional #number, and a one-line preview."""
    label, icon, _ = KINDS[kind]
    num_html = f'<span class="tl-num">#{num}</span>' if num is not None else ""
    return (
        f'<div class="tl-card k-{kind} {classes}"><div class="tl-head"><span>{icon}</span><span>{label}</span>'
        f'{num_html}</div><div class="tl-prev{" mono" if mono else ""}">{html.escape(text)}</div></div>'
    )


def kv_grid(rows: dict, fmt=json.dumps) -> str:
    """A two-column key/value table; values go through `fmt` (JSON by default, so strings show quoted)."""
    cells = (
        "".join(f'<div class="k">{html.escape(str(k))}</div><div>{html.escape(fmt(v))}</div>' for k, v in rows.items())
        or '<div class="k">—</div><div>none</div>'
    )
    return f'<div class="kv">{cells}</div>'


def eyebrow(text: str) -> str:
    """Small uppercase section label."""
    return f'<div class="eyebrow">{html.escape(text)}</div>'


def empty(icon: str, text: str) -> None:
    st.html(f'<div class="empty"><div class="big">{icon}</div>{text}</div>')


def show_reasoning(text: str) -> None:
    st.html(eyebrow("REASONING"))
    with st.container(key="thought"):
        st.markdown(text)


# ---------------------------------------------------------------- rendering


def render_sidebar_top(state: SessionState) -> None:
    """Brand, the Agent select and (for the coding agent) the sandbox toggle. Both start a new chat, so they're
    disabled while a job runs."""
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


def render_sidebar(state: SessionState) -> None:
    if orchestrating():
        note = None  # no shell
    elif state.settings.sandbox:
        note = "<b>Docker sandbox</b> — commands run without asking"
    else:
        note = "<b>Local shell</b> — each command needs approval"
    details = session().inspect(MAIN)
    main = state.agents[MAIN]
    with st.sidebar:
        if note:
            st.html(f'<div class="shell-note">{note}</div>')
        st.html(eyebrow(f"SKILLS · {len(details.skills)}"))
        st.html(chips(details.skills, on=details.loaded_skills, empty_text="no skills found"))
        st.html(eyebrow("SESSION"))
        st.html('<div class="row">' + pill(main.model, cls="model") + "</div>")
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
        if limit := state.settings.max_cost:  # the agent and any subagents
            st.progress(min(main.spent / limit, 1.0), text=f"${main.spent:.4f} of ${limit:.2f} budget")
        render_agents(state)


def agent_label(state: SessionState, path: str) -> str:
    """The picker's line for an agent: its name, subagents indented by depth. It must never change, status included:
    the browser sends the picked label back, and a label that no longer matches an option loses the pick."""
    if path == MAIN:
        return f"**{path}**"
    indent = " " * (path.count("/") - 1)  # em spaces: markdown collapses plain ones
    return f"{indent}↳ **#{path.rsplit('/', 1)[-1]}** {state.agents[path].profile}"


def agent_caption(state: SessionState, path: str) -> str:
    """The picker's second line: a status dot and status, the agent's own calls and spend, and a subagent's task."""
    status, detail = status_of(state, path)
    agent = state.agents[path]
    parts = [
        f":{style(status)[0]}[●] " + (f"{status} ({clip(detail, 40)})" if detail else status),
        f"{len(agent.calls)} calls · ${own_cost(agent):.4f}",
    ]
    if path != MAIN:
        parts.append(clip(re.sub(r"[*_`#>\[\]]+", "", agent.task), 48))  # plain text: captions are markdown
    return " · ".join(parts)


def render_agents(state: SessionState) -> None:
    """The agent picker: the main agent and every agent in its tree, and a notice for results waiting to be
    delivered. Refreshed by full reruns, which a change of `signature` triggers."""
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
    if pending := session().pending():
        st.info(
            f"{pending} subagent result{'s' if pending > 1 else ''} waiting; your next message delivers "
            f"{'them' if pending > 1 else 'it'}.",
            icon=":material/inbox:",
        )


def render_header(state: SessionState, path: str, entries: list[Item], ctx: list[Item]) -> None:
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
    pills.append(pill("context", bold=context_size(agent, session().inspect(path).compact_at), cls="mono"))
    if agent.calls:
        pills.append(pill(usage_line(agent.calls), cls="mono"))

    with st.container(horizontal=True, vertical_alignment="center", wrap=False):
        st.html(f'<div class="hdr"><span class="hdr-title">jean-code</span>{"".join(pills)}</div>')
        if path == MAIN:
            if state.busy:  # also while a call waits for approval: Stop answers it
                st.button("Stop", icon=":material/stop_circle:", on_click=session().cancel, type="primary")
            else:
                st.button("New chat", icon=":material/refresh:", on_click=new_chat)
        elif agent.running:
            st.button(
                "Cancel run",
                icon=":material/cancel:",
                on_click=session().cancel,
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


def render_timeline(agent: AgentState, entries: list[Item], selected: Item | None, running: bool, pending) -> None:
    names = call_names(entries)
    with st.container(height=640, autoscroll=True, border=False, key="timeline"):
        if not entries and not running:
            empty("✦", "Ask something below — every thought, tool call and answer<br>shows up here as a block.")
        with st.container(key="rail"):
            compacted_run = 0
            for e in entries:
                if e.in_context and compacted_run:
                    text = f"⟲ {compacted_run} messages compacted"
                    if rec := compaction_of(agent, e.message):
                        text += f" · ~{human(rec.tokens_before)} → ~{human(rec.tokens_after or 0)} tokens"
                    st.html(f'<div class="divider">{text}</div>')
                compacted_run = 0 if e.in_context else compacted_run + 1

                kind = turn_kind(e.message)
                err = kind == "tool_result" and is_error(e.message)
                flags = (("sel", selected is e), ("out", not e.in_context), ("err", err))
                classes = " ".join(c for c, on in flags if on)
                with st.container(key=f"blk-{kind}-{'err-' if err else ''}{e.uid}"):
                    st.html(
                        card(kind, preview(e.message, kind, names), num=e.uid, classes=classes, mono=kind in MONO_KINDS)
                    )
                    st.button(" ", key=f"hit-{e.uid}", on_click=select, args=(e.uid,), width="stretch")

            if running:
                with st.container(key="blk-thinking-live"):
                    st.html(card("thinking", "waiting for model…", classes="ghost"))
            elif pending is not None:
                render_approval(pending)


def render_approval(approval) -> None:
    """The call waiting for approval, with Approve / Deny. Sending a message instead denies it, as the reason."""
    with st.container(key="blk-approval-pending"):
        st.html(card("approval", f"{approval.tool}() wants to run:", mono=True))
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
                answer_approval(approval.id)
                st.rerun()
            if st.button("Deny", icon=":material/block:"):
                answer_approval(approval.id, denied(reason))
                st.rerun()


def render_args(args: dict) -> None:
    """Short arguments in a key/value table; commands and long or multi-line strings as code blocks."""

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
    """A tool result's content: pretty JSON when it parses as JSON, height-capped when long, red when it failed."""
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


def render_tool_call(entries: list[Item], pos: int, running: bool, pending: bool) -> None:
    msg = entries[pos].message
    if mget(msg, "reasoning"):
        show_reasoning(mget(msg, "reasoning"))
    if mget(msg, "content"):
        st.markdown(mget(msg, "content"))
    results = {mget(e.message, "tool_call_id"): e for e in results_of(entries, pos)}
    is_last = pos + len(results) == len(entries) - 1
    for call in mget(msg, "tool_calls"):
        st.html(f'<div class="fn-name">{html.escape(call.function.name)}()</div>')
        render_args(parse_args(call))
        result = results.get(call.id)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.html(eyebrow("↳ RESULT"))
            if result is not None:
                st.button(
                    "Jump to result",
                    icon=":material/arrow_downward:",
                    on_click=select,
                    args=(result.uid,),
                    key=f"jump-result-{result.uid}",
                    type="tertiary",
                )
        if result is not None:
            render_output(result)
        elif running and is_last:
            st.caption("running…")
        elif pending and is_last:
            st.caption("waiting for approval")
        else:
            st.caption("no result")


def render_tool_result(entries: list[Item], pos: int) -> None:
    result = entries[pos]
    name = call_names(entries).get(mget(result.message, "tool_call_id"), "tool")
    err = " err" if is_error(result.message) else ""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.html(f'{eyebrow("FROM")}<div class="fn-name{err}">{html.escape(name)}()</div>')
        if call := call_of(entries, pos):
            st.button(
                "Jump to call", icon=":material/arrow_upward:", on_click=select, args=(call.uid,), type="tertiary"
            )
    render_output(result)


def render_text(msg, kind: str) -> None:
    """User, system, agent and thinking messages: the reasoning, then the markdown content."""
    if mget(msg, "reasoning"):
        show_reasoning(mget(msg, "reasoning"))
    if kind != "thinking":
        st.markdown(mget(msg, "content") or "*(no content)*")


def render_turn(
    agent: AgentState, entries: list[Item], selected: Item | None, ctx: list[Item], running: bool, pending: bool
) -> None:
    if selected is None:
        empty("◇", "Click a block in the timeline to inspect it.")
        return

    pos = entries.index(selected)
    msg, kind = selected.message, turn_kind(selected.message)
    label, icon, _ = KINDS[kind]
    err = " err" if kind == "tool_result" and is_error(msg) else ""
    if selected.in_context:
        where = next(i for i, item in enumerate(ctx) if item.uid == selected.uid)
        meta = f'<span class="insp-meta">context message {where + 1} of {len(ctx)}</span>'
    else:
        meta = '<span class="insp-meta out">⟲ compacted — no longer in context</span>'

    with st.container(horizontal=True, vertical_alignment="center"):
        badge = f'<span class="badge">{icon} {label} #{selected.uid}</span>'
        st.html(f'<div class="insp-head k-{kind}{err}">{badge}{meta}</div>')
        if st.session_state.selected.get(st.session_state.view) is None:
            st.html(pill("⤓ following latest"), width="content")
        else:
            st.button(
                "Follow latest", icon=":material/vertical_align_bottom:", on_click=select, args=(None,), type="tertiary"
            )

    if kind == "tool_call":
        render_tool_call(entries, pos, running, pending)
    elif kind == "tool_result":
        render_tool_result(entries, pos)
    else:
        if kind == "summary" and (rec := compaction_of(agent, msg)):
            st.html(
                eyebrow("COMPACTION")
                + kv_grid(
                    {
                        "messages removed": rec.removed,
                        "tokens": f"~{human(rec.tokens_before)} → ~{human(rec.tokens_after or 0)}",
                        "summary call": money(rec.cost),
                    },
                    fmt=str,
                )
            )
        render_text(msg, kind)

    with st.expander("Raw message", icon=":material/data_object:"):
        st.json(render_message(msg))


def render_request(path: str, agent: AgentState) -> None:
    req = session().last_request(path)
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
    """The context's size against the compaction threshold, the settings, and every compaction attempt."""
    details = session().inspect(path)
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
    details = session().inspect(path)
    st.html(eyebrow("CONFIG") + kv_grid({k: getattr(details, k) for k in STATE_CONFIG}))
    st.html(eyebrow("NEEDS APPROVAL") + chips(details.needs_approval))
    st.html(eyebrow("LOADED SKILLS") + chips(details.loaded_skills))
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
        "tools": details.tools,
    }
    st.html(eyebrow("AGENT STATE"))
    st.json(to_jsonable(summary), expanded=1)


def announce(state: SessionState) -> None:
    """Toast each compaction attempt of the main agent's since the last full rerun, and the outcome of a /compact
    that left no record ("nothing to compact", a refusal). Runs in the full script, which reruns when a job ends."""
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

if "session" not in st.session_state:
    st.session_state.session = Session(Settings(**sidebar_settings()))
if "selected" not in st.session_state:
    reset_view()
state = session().state()
render_sidebar_top(state)
render_sidebar(state)
announce(state)
st.session_state.signature_seen = signature(state)


@st.fragment(run_every=POLL_SECONDS if live(state) else None)
def live_view() -> None:
    state = session().state()
    if signature(state) != st.session_state.signature_seen:
        st.rerun()  # an agent started or stopped, a job ended, or a result is waiting: the sidebar catches up

    path = st.session_state.view if st.session_state.view in state.agents else MAIN
    agent = state.agents[path]
    entries = agent.items
    ctx = [e for e in entries if e.in_context]
    running = is_running(state, path)
    approval = state.approval if path == MAIN else None  # only the main agent asks
    by_uid = {e.uid: e for e in entries}
    selected = by_uid.get(st.session_state.selected.get(path)) or (entries[-1] if entries else None)

    render_header(state, path, entries, ctx)
    timeline_col, inspector_col = st.columns([2, 3], gap="large")

    with timeline_col:
        st.html(eyebrow("TIMELINE" if path == MAIN else f"TIMELINE · {agent_label(state, path).strip()}"))
        render_timeline(agent, entries, selected, running, approval)
        if path != MAIN:
            st.chat_input("Subagents take no messages: pick the main agent to chat", disabled=True)
        elif prompt := st.chat_input(
            "Agent is working…"
            if running
            else "Approve or deny the call, or say why not…"
            if approval
            else "Message the agent…",
            disabled=running,
        ):
            if approval is not None:
                answer_approval(approval.id, denied(prompt))
            elif prompt.strip() == "/compact":  # a UI command: never sent to the model as a message
                session().compact()
            else:
                session().send(prompt)
            select(None)
            st.rerun()

    with inspector_col:
        turn_tab, request_tab, context_tab, state_tab = st.tabs(["Turn", "Request", "Context", "Agent state"])
        with turn_tab:
            render_turn(agent, entries, selected, ctx, running, approval is not None)
        with request_tab:
            render_request(path, agent)
        with context_tab:
            render_context(path, agent)
        with state_tab:
            render_state(path, agent)


live_view()
