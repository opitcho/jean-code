"""Streamlit UI for the agent: a live, clickable timeline of turns plus an inspector.

Run with: uv run --env-file .env streamlit run src/app.py

The agent stays UI-agnostic: `run_turn` runs in a background thread and this UI polls
`agent.messages`. The timeline is the UI's own history of every message it has seen,
matched by object identity, so it survives the context window being rewritten
(e.g. compaction): turns that leave `agent.messages` are dimmed, not lost.
"""

import html
import json
import re
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from agent import DEFAULT_MODEL
from coding import coding_agent
from config import USAGE_LOG
from prompts import HEAD_MARKER_TAG, SUMMARY_TAG
from repl import compaction_line
from subagents import run_status
from usage import COST_UNKNOWN, human, record, usage_line

MODEL = DEFAULT_MODEL
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
ERROR_PREFIXES = ("Error", "Denied by user", "Not run")  # results the agent writes when a call didn't succeed
EXIT_CODE = re.compile(r"exit code: (\d+)[^\n]*\n?")  # first line of a bash result, see shell.format_result
LONG_ARG = 80  # string arguments longer than this get their own code block in the inspector
STATE_CONFIG = ("model", "reasoning", "max_tool_rounds", "max_cost")
CONTEXT_CONFIG = ("compact_at", "keep_first", "keep_last", "context_window", "head_end")
STATE_HIDDEN = {"client", "messages", "last_request", "last_response", "usage", "compactions",  # shown elsewhere
                "needs_approval", "loaded_skills", *STATE_CONFIG, *CONTEXT_CONFIG,
                "budget", "subagents"}  # the sidebar; their trees link back up (child.parent), so vars() would recurse
USAGE_COLUMNS = {
    "time": st.column_config.TextColumn("time"),
    "provider": st.column_config.TextColumn("provider"),
    "prompt_tokens": st.column_config.NumberColumn("in"),
    "cached_tokens": st.column_config.NumberColumn("cached"),
    "completion_tokens": st.column_config.NumberColumn("out"),
    "reasoning_tokens": st.column_config.NumberColumn("reasoning"),
    "cost": st.column_config.NumberColumn("cost", format="$%.5f"),
}


@dataclass
class Entry:
    """A message the UI has seen. `uid` is stable even if the message leaves the context."""

    uid: int
    msg: dict
    in_context: bool = True


def to_jsonable(value):
    """Best-effort conversion of arbitrary agent state into something st.json can render."""
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if callable(value):
        return f"{getattr(value, '__name__', type(value).__name__)}()"
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def mget(msg, key, default=None):
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
        value = json.loads(call["function"].get("arguments") or "{}")
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
        return call["function"]["name"]
    return f"{call['function']['name']} · {first if isinstance(first, str) else json.dumps(first)}"


def result_summary(msg, name: str) -> str:
    """`name → first line of output`, with the exit code when a command failed."""
    content = str(mget(msg, "content") or "")
    exit_code = EXIT_CODE.match(content)
    if exit_code:
        content = content[exit_code.end():]
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


def sync_history(entries: list[Entry], ctx: list) -> None:
    """Mark which seen messages are still in the context, and add unseen ones after the message before them.

    A compaction's summary thus lands where it sits in the context, after the messages it replaced.
    Entries hold references to their messages, so `id()` can't be reused while they exist.
    """
    ctx_ids = {id(m) for m in ctx}
    position = {}
    for i, e in enumerate(entries):
        e.in_context = id(e.msg) in ctx_ids
        position[id(e.msg)] = i
    start = 0  # where the next unseen message may go: after the previous context message
    for m in ctx:
        if id(m) in position:
            start = position[id(m)] + 1
            continue
        # before the next entry still in context, so it follows the compacted entries it replaced
        at = next((j for j in range(start, len(entries)) if entries[j].in_context), len(entries))
        st.session_state.next_uid += 1
        entries.insert(at, Entry(st.session_state.next_uid, m))
        position = {id(e.msg): i for i, e in enumerate(entries)}
        start = at + 1


def results_of(entries: list[Entry], pos: int) -> list[Entry]:
    """The tool results following the call at `pos`, in call order (results match calls by position)."""
    results = []
    for e in entries[pos + 1 :]:
        if turn_kind(e.msg) != "tool_result":
            break
        results.append(e)
    return results


def call_of(entries: list[Entry], pos: int) -> Entry | None:
    """The tool call a result at `pos` belongs to: the nearest call before its run of results."""
    for e in reversed(entries[:pos]):
        kind = turn_kind(e.msg)
        if kind == "tool_call":
            return e
        if kind != "tool_result":
            return None
    return None


def call_names(entries: list[Entry]) -> dict[str, str]:
    """Tool name of every call seen, by call id, so a result can be matched to its tool by `tool_call_id`."""
    return {call.get("id"): call["function"]["name"] for e in entries for call in mget(e.msg, "tool_calls") or []}


def compaction_of(agent, msg) -> dict | None:
    """The compaction record whose summary is `msg`, if any."""
    return next((c for c in agent.compactions if c.get("summary") is msg), None)


def compaction_cost(rec: dict) -> str:
    cost = ((rec.get("response") or {}).get("usage") or {}).get("cost")
    return "—" if cost is None else f"${cost:.5f}"


def context_size(agent) -> str:
    """`~24.0k / 256.0k`: the context's size against the compaction threshold."""
    size = f"~{human(agent.context_tokens)}"
    return f"{size} / {human(agent.compact_at)}" if agent.compact_at else size


# ---------------------------------------------------------------- background run


def is_running() -> bool:
    thread = st.session_state.run["thread"]
    return thread is not None and thread.is_alive()


def subagent_state(agent) -> tuple[int, int]:
    """(running, pending) for the agent's subagents. While any run, the view keeps polling; when either changes,
    it reruns in full, so the sidebar and the notice catch up."""
    subagents = agent.subagents
    return (subagents.running(), subagents.pending) if subagents else (0, 0)


def start_run(fn, *args) -> None:
    """Run an agent method (`run_turn`, `approve`, `deny`) off the script thread; the UI polls the agent's state.

    The model calls it makes are logged as they're charged, through `agent.budget.log_path` (set in `new_chat`).
    """
    run = st.session_state.run

    def target():
        try:
            fn(*args)
        except Exception as e:  # surfaced in the header; never call st.* from here
            run["error"] = f"{type(e).__name__}: {e}"

    run["error"] = run["notice"] = None
    run["thread"] = threading.Thread(target=target, daemon=True)
    run["thread"].start()
    st.session_state.was_running = True
    st.session_state.selected_uid = None  # follow the live turn


def compact_now(agent, run: dict) -> None:
    """`/compact`: compact regardless of the threshold. An outcome that leaves no record in `agent.compactions`
    (refused, nothing to compact) is kept as the run's notice; recorded ones are toasted from the records.

    Runs on the background thread, so it gets the agent and run as arguments: no `st.session_state` there."""
    before = len(agent.compactions)
    status = agent.compact(force=True)
    if len(agent.compactions) == before:
        run["notice"] = f"[compaction {status}]"


def new_chat() -> None:
    """A fresh agent; the sidebar toggle picks where its shell runs. The old agent's jobs/container are closed."""
    if old := st.session_state.get("agent"):
        old.close()
    st.session_state.agent = coding_agent(MODEL, sandbox=st.session_state.get("sandbox", False))
    st.session_state.agent.budget.log_path = USAGE_LOG  # every call in its tree, subagents' between turns included
    st.session_state.entries = []
    st.session_state.selected_uid = None
    st.session_state.run = {"thread": None, "error": None, "notice": None}
    st.session_state.compactions_seen = 0


def select(uid: int | None) -> None:
    st.session_state.selected_uid = uid


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
    cells = "".join(
        f'<div class="k">{html.escape(str(k))}</div><div>{html.escape(fmt(v))}</div>' for k, v in rows.items()
    ) or '<div class="k">—</div><div>none</div>'
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


def render_sidebar_top() -> None:
    """Brand and the sandbox toggle; drawn before the agent exists, since the toggle decides its shell."""
    with st.sidebar:
        st.html(
            '<div class="brand"><div class="brand-mark">🧮</div>'
            '<div><div class="brand-name">jean-code</div><div class="brand-sub">agent inspector</div></div></div>'
        )
        st.html(eyebrow("ENVIRONMENT"))
        st.toggle("Run bash in Docker sandbox", key="sandbox", on_change=new_chat,
                  help="Off: commands run on this machine in the directory the app was started from, and need "
                       "approval. On: they run in the jean-code-sandbox container without asking. Starts a new chat.")


def render_sidebar(agent) -> None:
    if st.session_state.get("sandbox", False):
        note = "<b>Docker sandbox</b> — commands run without asking"
    else:
        note = "<b>Local shell</b> — each command needs approval"
    with st.sidebar:
        st.html(f'<div class="shell-note">{note}</div>')
        st.html(eyebrow(f"SKILLS · {len(agent.skills)}"))
        st.html(chips(sorted(agent.skills), on=agent.loaded_skills, empty_text="no skills found"))
        st.html(eyebrow("SESSION"))
        st.html('<div class="row">' + pill(agent.model, cls="model")
                + pill("reasoning", bold=agent.reasoning or "off") + "</div>")
        budget = agent.budget  # the agent and any subagents
        if budget.limit:
            st.progress(min(budget.spent / budget.limit, 1.0),
                        text=f"${budget.spent:.4f} of ${budget.limit:.2f} budget")
        if agent.subagents:
            render_subagents(agent.subagents)


def render_subagents(subagents) -> None:
    """The runs so far, one line each (status, spend, task), and a notice for results waiting to be delivered."""
    runs = subagents.snapshot()
    st.html(eyebrow(f"SUBAGENTS · {len(runs)}"))
    if not runs:
        st.caption("none started")
    for id, profile, run in runs:
        status, reason, _ = run_status(run)
        detail = f" ({reason})" if reason else ""
        st.caption(f"**#{id}** {profile} · {status}{detail} · ${run.agent.budget.spent:.4f} · {clip(one_line(run.task), 60)}")
    if pending := subagents.pending:
        st.info(f"{pending} subagent result{'s' if pending > 1 else ''} waiting; your next message delivers "
                f"{'them' if pending > 1 else 'it'}.", icon=":material/inbox:")


def render_header(agent, entries: list[Entry], ctx: list, running: bool) -> None:
    tool_calls = sum(turn_kind(e.msg) == "tool_call" for e in entries)
    compacted = sum(not e.in_context for e in entries)
    if running:
        status = pill("running", dot="run", cls="status-run")
    elif agent.pending_calls:
        status = pill("awaiting approval", dot="wait", cls="status-wait")
    else:
        status = pill("idle", dot="")
    pills = [pill(agent.model, cls="model"), status]
    pills += [pill(label, bold=n) for n, label in
              ((len(entries), "turns"), (tool_calls, "tool calls"), (len(ctx), "in context"), (compacted, "compacted"))
              if n]
    pills.append(pill("context", bold=context_size(agent), cls="mono"))
    if agent.usage:
        pills.append(pill(usage_line(agent.usage), cls="mono"))

    with st.container(horizontal=True, vertical_alignment="center", wrap=False):
        st.html(f'<div class="hdr"><span class="hdr-title">jean-code</span>{"".join(pills)}</div>')
        if running:
            st.button("Stop", icon=":material/stop_circle:", on_click=agent.interrupt, type="primary")
        else:
            st.button("New chat", icon=":material/refresh:", on_click=new_chat)

    if error := st.session_state.run["error"]:
        st.error(error, icon=":material/error:")
    if not running and agent.interrupted and agent.last_cost_unknown and not agent.over_budget:
        st.warning(COST_UNKNOWN, icon=":material/warning:")
    if not running and agent.interrupted and agent.stop_reason:
        st.warning(agent.stop_reason, icon=":material/warning:")


def render_timeline(entries: list[Entry], selected: Entry | None, running: bool, agent) -> None:
    names = call_names(entries)
    with st.container(height=640, autoscroll=True, border=False, key="timeline"):
        if not entries and not running:
            empty("✦", "Ask something below — every thought, tool call and answer<br>shows up here as a block.")
        with st.container(key="rail"):
            compacted_run = 0
            for e in entries:
                if e.in_context and compacted_run:
                    text = f"⟲ {compacted_run} messages compacted"
                    if rec := compaction_of(agent, e.msg):
                        text += f" · ~{human(rec['tokens_before'])} → ~{human(rec['tokens_after'])} tokens"
                    st.html(f'<div class="divider">{text}</div>')
                compacted_run = 0 if e.in_context else compacted_run + 1

                kind = turn_kind(e.msg)
                err = kind == "tool_result" and is_error(e.msg)
                classes = " ".join(c for c, on in (("sel", selected is e), ("out", not e.in_context), ("err", err)) if on)
                with st.container(key=f"blk-{kind}-{'err-' if err else ''}{e.uid}"):
                    st.html(card(kind, preview(e.msg, kind, names), num=e.uid, classes=classes,
                                 mono=kind in MONO_KINDS))
                    st.button(" ", key=f"hit-{e.uid}", on_click=select, args=(e.uid,), width="stretch")

            if running:
                with st.container(key="blk-thinking-live"):
                    st.html(card("thinking", "waiting for model…", classes="ghost"))
            elif agent.pending_calls:
                render_approval(agent)


def render_approval(agent) -> None:
    """The first call waiting for approval, with Approve / Deny. Sending a message instead also answers it."""
    name, arguments = agent.pending_calls[0]
    with st.container(key="blk-approval-pending"):
        st.html(card("approval", f"{name}() wants to run:", mono=True))
        command = arguments.get("command") if name == "bash" else None
        st.code(command or json.dumps(arguments, indent=2), language="bash" if command else "json", wrap_lines=True)
        reason = st.text_input("Reason", key="deny_reason", placeholder="optional reason, sent to the model on deny",
                               label_visibility="collapsed")
        with st.container(horizontal=True):
            if st.button("Approve", icon=":material/check:", type="primary"):
                start_run(agent.approve)
                st.rerun()
            if st.button("Deny", icon=":material/block:"):
                start_run(agent.deny, reason)
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


def render_output(result: Entry) -> None:
    """A tool result's content: pretty JSON when it parses as JSON, height-capped when long, red when it failed."""
    content, language = str(mget(result.msg, "content")), None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, (dict, list)):
            content, language = json.dumps(parsed, indent=2), "json"
    except ValueError:
        pass
    height = 420 if content.count("\n") > 20 else "content"
    with st.container(key=f"errout-{result.uid}") if is_error(result.msg) else nullcontext():
        st.code(content, language=language, wrap_lines=True, height=height)


def render_tool_call(entries: list[Entry], pos: int, running: bool, pending: bool) -> None:
    msg = entries[pos].msg
    if mget(msg, "reasoning"):
        show_reasoning(mget(msg, "reasoning"))
    if mget(msg, "content"):
        st.markdown(mget(msg, "content"))
    results = {mget(e.msg, "tool_call_id"): e for e in results_of(entries, pos)}
    is_last = pos + len(results) == len(entries) - 1
    for call in mget(msg, "tool_calls"):
        st.html(f'<div class="fn-name">{html.escape(call["function"]["name"])}()</div>')
        render_args(parse_args(call))
        result = results.get(call.get("id"))
        with st.container(horizontal=True, vertical_alignment="center"):
            st.html(eyebrow("↳ RESULT"))
            if result is not None:
                st.button("Jump to result", icon=":material/arrow_downward:", on_click=select, args=(result.uid,),
                          key=f"jump-result-{result.uid}", type="tertiary")
        if result is not None:
            render_output(result)
        elif running and is_last:
            st.caption("running…")
        elif pending and is_last:
            st.caption("waiting for approval")
        else:
            st.caption("no result")


def render_tool_result(entries: list[Entry], pos: int) -> None:
    result = entries[pos]
    name = call_names(entries).get(mget(result.msg, "tool_call_id"), "tool")
    err = " err" if is_error(result.msg) else ""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.html(f'{eyebrow("FROM")}<div class="fn-name{err}">{html.escape(name)}()</div>')
        if call := call_of(entries, pos):
            st.button("Jump to call", icon=":material/arrow_upward:", on_click=select, args=(call.uid,),
                      type="tertiary")
    render_output(result)


def render_text(msg, kind: str) -> None:
    """User, system, agent and thinking messages: the reasoning, then the markdown content."""
    if mget(msg, "reasoning"):
        show_reasoning(mget(msg, "reasoning"))
    if kind != "thinking":
        st.markdown(mget(msg, "content") or "*(no content)*")


def render_turn(agent, entries: list[Entry], selected: Entry | None, ctx: list, running: bool, pending: bool) -> None:
    if selected is None:
        empty("◇", "Click a block in the timeline to inspect it.")
        return

    pos = entries.index(selected)
    msg, kind = selected.msg, turn_kind(selected.msg)
    label, icon, _ = KINDS[kind]
    err = " err" if kind == "tool_result" and is_error(msg) else ""
    if selected.in_context:
        where = next(i for i, m in enumerate(ctx) if m is msg)
        meta = f'<span class="insp-meta">context message {where + 1} of {len(ctx)}</span>'
    else:
        meta = '<span class="insp-meta out">⟲ compacted — no longer in context</span>'

    with st.container(horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="insp-head k-{kind}{err}"><span class="badge">{icon} {label} #{selected.uid}</span>{meta}</div>')
        if st.session_state.selected_uid is None:
            st.html(pill("⤓ following latest"), width="content")
        else:
            st.button("Follow latest", icon=":material/vertical_align_bottom:", on_click=select, args=(None,),
                      type="tertiary")

    if kind == "tool_call":
        render_tool_call(entries, pos, running, pending)
    elif kind == "tool_result":
        render_tool_result(entries, pos)
    else:
        if kind == "summary" and (rec := compaction_of(agent, msg)):
            st.html(eyebrow("COMPACTION") + kv_grid({
                "messages removed": rec["removed"],
                "tokens": f"~{human(rec['tokens_before'])} → ~{human(rec['tokens_after'])}",
                "summary call": compaction_cost(rec),
            }, fmt=str))
        render_text(msg, kind)

    with st.expander("Raw message", icon=":material/data_object:"):
        st.json(to_jsonable(msg))


def render_request(agent) -> None:
    req = agent.last_request
    if not req:
        empty("⇅", "No request sent yet.")
        return
    tools = [
        (t.get("function") or {}).get("name", "?") if isinstance(t, dict) else getattr(t, "__name__", str(t))
        for t in req.get("tools", [])
    ]
    reasoning = req.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else reasoning
    st.html(
        '<div class="hdr">' + pill(req.get("model"), cls="model") + pill("reasoning", bold=effort or "none")
        + pill("messages", bold=len(req.get("messages", []))) + "</div>"
    )
    st.html(eyebrow(f"TOOLS · {len(tools)}") + chips(tools))

    if agent.usage:
        records = [record(raw) for raw in agent.usage]
        last = records[-1]
        cost = "—" if last["cost"] is None else f"${last['cost']:.5f}"
        st.html(eyebrow("LAST CALL") + kv_grid({
            "provider": last["provider"] or "—",
            "tokens in": f"{human(last['prompt_tokens'])} ({human(last['cached_tokens'])} cached)",
            "tokens out": f"{human(last['completion_tokens'])} ({human(last['reasoning_tokens'])} reasoning)",
            "cost": cost,
        }, fmt=str))
        st.html(eyebrow(f"USAGE · {len(agent.usage)} CALLS · {usage_line(agent.usage)}"))
        rows = [{**r, "time": (r["time"] or "")[11:19]} for r in reversed(records)]  # ISO time -> HH:MM:SS
        st.dataframe(rows, hide_index=True, column_order=list(USAGE_COLUMNS),
                     column_config=USAGE_COLUMNS, height=min(38 + 35 * len(agent.usage), 280))

    st.html(eyebrow("PAYLOAD"))
    with st.expander("Messages sent", icon=":material/forum:"):
        st.json(to_jsonable(req.get("messages", [])), expanded=1)
    with st.expander("Full request", icon=":material/data_object:"):
        st.json(to_jsonable(req), expanded=1)


def render_context(agent) -> None:
    """The context's size against the compaction threshold, the settings, and every compaction attempt."""
    if agent.compact_at:
        st.progress(min(agent.context_tokens / agent.compact_at, 1.0),
                    text=f"context {context_size(agent)} tokens (compacts after a turn ending over the threshold)")
    else:
        st.html(pill("context", bold=context_size(agent), cls="mono") + pill("automatic compaction off"))
    st.html(eyebrow("SETTINGS") + kv_grid({k: getattr(agent, k) for k in CONTEXT_CONFIG}))

    st.html(eyebrow(f"COMPACTIONS · {len(agent.compactions)}"))
    if not agent.compactions:
        st.caption("None yet. Type /compact to compact now.")
        return
    rows = [{
        "#": i + 1,
        "status": rec["status"] or "running…",
        "removed": rec["removed"],
        "before": human(rec["tokens_before"]),
        "after": human(rec["tokens_after"]) if "tokens_after" in rec else "—",
        "cost": compaction_cost(rec),
    } for i, rec in enumerate(agent.compactions)]
    st.dataframe(rows, hide_index=True)
    for i, rec in enumerate(agent.compactions):
        if rec.get("summary"):
            with st.expander(f"Summary #{i + 1}", icon=":material/summarize:"):
                st.markdown(rec["summary"]["content"])


def render_state(agent) -> None:
    config = {k: to_jsonable(getattr(agent, k)) for k in STATE_CONFIG if hasattr(agent, k)}
    st.html(eyebrow("CONFIG") + kv_grid(config))
    st.html(eyebrow("NEEDS APPROVAL") + chips(sorted(agent.needs_approval)))
    st.html(eyebrow("LOADED SKILLS") + chips(agent.loaded_skills))

    attrs = {k: to_jsonable(v) for k, v in vars(agent).items() if k not in STATE_HIDDEN}
    short = {k: v for k, v in attrs.items() if not isinstance(v, (dict, list)) and len(json.dumps(v)) <= LONG_ARG}
    st.html(eyebrow("OTHER") + kv_grid(short))
    for attr, value in attrs.items():
        if attr not in short:
            with st.expander(attr, icon=":material/data_object:"):
                if isinstance(value, str):
                    st.code(value, language=None, wrap_lines=True)
                else:
                    st.json(value, expanded=1)


def announce_compactions(agent) -> None:
    """Toast each compaction attempt that finished since the last full rerun, and a `/compact` notice.

    Runs in the full script, which reruns when a run ends, so a compaction at the end of a turn is noticed.
    """
    records = agent.compactions
    while st.session_state.compactions_seen < len(records) and records[st.session_state.compactions_seen]["status"]:
        st.toast(compaction_line(records[st.session_state.compactions_seen]), icon=":material/compress:")
        st.session_state.compactions_seen += 1
    if notice := st.session_state.run.pop("notice", None):
        st.toast(notice, icon=":material/compress:")


# ---------------------------------------------------------------- app

st.set_page_config(page_title="jean-code", page_icon="🧮", layout="wide")
st.html(CSS_PATH)
st.html(kind_css())

render_sidebar_top()
if "agent" not in st.session_state:
    new_chat()
    st.session_state.next_uid = 0
    st.session_state.was_running = False
render_sidebar(st.session_state.agent)
announce_compactions(st.session_state.agent)
st.session_state.subagents_seen = subagent_state(st.session_state.agent)


@st.fragment(run_every=POLL_SECONDS if is_running() or st.session_state.subagents_seen[0] else None)
def live_view() -> None:
    agent, entries = st.session_state.agent, st.session_state.entries
    running = is_running()
    if st.session_state.was_running and not running:
        st.session_state.was_running = False
        st.rerun()  # full rerun: stop polling, re-enable the input and refresh the sidebar
    if subagent_state(agent) != st.session_state.subagents_seen:
        st.rerun()  # a run finished or a result was delivered: refresh the sidebar, and stop polling once none runs

    ctx = list(agent.messages)  # snapshot: the run thread appends concurrently
    sync_history(entries, ctx)
    by_uid = {e.uid: e for e in entries}
    selected = by_uid.get(st.session_state.selected_uid) or (entries[-1] if entries else None)

    render_header(agent, entries, ctx, running)
    timeline_col, inspector_col = st.columns([2, 3], gap="large")

    with timeline_col:
        st.html(eyebrow("TIMELINE"))
        render_timeline(entries, selected, running, agent)
        prompt = st.chat_input("Agent is working…" if running else "Message the agent…", disabled=running)
        if prompt:
            if prompt.strip() == "/compact":  # a UI command: never sent to the model as a message
                start_run(compact_now, agent, st.session_state.run)
            else:
                start_run(agent.run_turn, prompt)
            st.rerun()

    with inspector_col:
        turn_tab, request_tab, context_tab, state_tab = st.tabs(["Turn", "Request", "Context", "Agent state"])
        with turn_tab:
            render_turn(agent, entries, selected, ctx, running, bool(agent.pending_calls))
        with request_tab:
            render_request(agent)
        with context_tab:
            render_context(agent)
        with state_tab:
            render_state(agent)


live_view()
