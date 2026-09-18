"""Streamlit UI for BaseAgent: a live, clickable timeline of turns plus an inspector.

Run with: uv run streamlit run src/app.py

The agent stays UI-agnostic: `run_turn` runs in a background thread and this UI polls
`agent.messages`. The timeline is the UI's own history of every message it has seen,
matched by object identity, so it survives the context window being rewritten
(e.g. compaction): turns that leave `agent.messages` are dimmed, not lost.
"""

import html
import json
import re
import threading
from dataclasses import dataclass

import streamlit as st

from base import CodingAgent
from shell import DockerShell, LocalShell

MODEL = "qwen3.6"
POLL_SECONDS = 0.5

# kind -> (label, icon, accent color)
KINDS = {
    "system": ("SYSTEM", "⚙", "#94A3B8"),
    "user": ("USER", "◉", "#38BDF8"),
    "thinking": ("THINKING", "✦", "#A78BFA"),
    "tool_call": ("TOOL CALL", "⚒", "#FBBF24"),
    "tool_result": ("TOOL RESULT", "↳", "#34D399"),
    "agent": ("AGENT", "◆", "#F472B6"),
    "approval": ("APPROVAL", "⚠", "#FB923C"),
}
MONO_KINDS = {"tool_call", "tool_result"}


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
    if isinstance(value, (list, tuple)):
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
    if mget(msg, "thinking") and not mget(msg, "content"):
        return "thinking"
    if role in ("tool", "user", "system"):
        return {"tool": "tool_result"}.get(role, role)
    return "agent"


def format_call(call) -> str:
    fn = call["function"]
    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in fn["arguments"].items())
    return f"{fn['name']}({args})"


def preview(msg, kind: str, limit: int = 90) -> str:
    """One-line summary of a message for its timeline block."""
    if kind == "tool_call":
        text = "; ".join(format_call(c) for c in mget(msg, "tool_calls"))
    elif kind == "tool_result":
        text = f"→ {mget(msg, 'content')}"
    elif kind == "thinking":
        text = mget(msg, "thinking")
    else:
        text = mget(msg, "content") or "(no content)"
    if kind in ("agent", "user"):
        text = re.sub(r"[*_`#>]+", "", str(text))  # drop markdown markers from the one-liner
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def sync_history(entries: list[Entry], ctx: list) -> None:
    """Mark which seen messages are still in the context, and append unseen ones in order.

    Entries hold references to their messages, so `id()` can't be reused while they exist.
    """
    ctx_ids = {id(m) for m in ctx}
    known = set()
    for e in entries:
        e.in_context = id(e.msg) in ctx_ids
        known.add(id(e.msg))
    for m in ctx:
        if id(m) not in known:
            st.session_state.next_uid += 1
            entries.append(Entry(st.session_state.next_uid, m))


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


# ---------------------------------------------------------------- background run


def is_running() -> bool:
    thread = st.session_state.run["thread"]
    return thread is not None and thread.is_alive()


def start_run(fn, *args) -> None:
    """Run an agent method (`run_turn`, `approve`, `deny`) off the script thread; the UI polls the agent's state."""
    run = st.session_state.run

    def target():
        try:
            fn(*args)
        except Exception as e:  # surfaced in the header; never call st.* from here
            run["error"] = f"{type(e).__name__}: {e}"

    run["error"] = None
    run["thread"] = threading.Thread(target=target, daemon=True)
    run["thread"].start()
    st.session_state.was_running = True
    st.session_state.selected_uid = None  # follow the live turn


def new_chat() -> None:
    """A fresh agent; the sidebar toggle picks where its shell runs. The old shell's jobs/container are closed."""
    if old := st.session_state.get("agent"):
        old.shell.close()
    shell = DockerShell() if st.session_state.get("sandbox", False) else LocalShell()
    st.session_state.agent = CodingAgent(model=MODEL, shell=shell)
    st.session_state.entries = []
    st.session_state.selected_uid = None
    st.session_state.run = {"thread": None, "error": None}


def select(uid: int | None) -> None:
    st.session_state.selected_uid = uid


def live_status(agent) -> tuple[str, str]:
    """(kind, text) for the live block; requests aren't streamed, so we only know one is running."""
    return "thinking", "waiting for model…"


# ---------------------------------------------------------------- styling

CSS = """
<style>
.block-container { padding-top: 3rem; padding-bottom: 1rem; max-width: 1500px; }
header[data-testid="stHeader"] { background: transparent; }

/* header */
.hdr { display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; }
.hdr-title { font-size:1.45rem; font-weight:700; letter-spacing:-.02em; margin-right:.3rem;
  background: linear-gradient(90deg,#E5E7EB,#A78BFA 60%,#F472B6);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.pill { display:inline-flex; align-items:center; gap:.4rem; padding:.18rem .65rem; border-radius:999px;
  border:1px solid #1F2430; background:#12151C; color:#8B93A7; font-size:.78rem; white-space:nowrap; }
.pill b { color:#E5E7EB; font-weight:600; }
.pill.model { font-family:'JetBrains Mono',monospace; color:#C4B5FD; border-color:#2E2A4A; }
.dot { width:8px; height:8px; border-radius:50%; background:#4B5563; }
.dot.run { background:#34D399; box-shadow:0 0 0 0 #34D39988; animation: ping 1.4s infinite; }
@keyframes ping { 0% { box-shadow:0 0 0 0 #34D39988; } 100% { box-shadow:0 0 0 8px #34D39900; } }
.section { font-size:.7rem; font-weight:600; letter-spacing:.14em; color:#6B7280; margin:.2rem 0 .1rem; }

/* timeline rail */
.st-key-rail { position:relative; padding-left:26px; gap:.55rem; }
.st-key-rail::before { content:""; position:absolute; left:9px; top:10px; bottom:10px; width:2px;
  background: linear-gradient(#1F2430, #2A3040 50%, #1F2430); border-radius:2px; }
[class*="st-key-blk-"] { position:relative; gap:0; }
[class*="st-key-blk-"]::before { content:""; position:absolute; left:-21px; top:15px; width:10px; height:10px;
  border-radius:50%; background:var(--c); box-shadow:0 0 0 3px #0B0D12; z-index:1; }
""" + "".join(
    f'[class*="st-key-blk-{k}-"], .k-{k} {{ --c:{c}; }}\n' for k, (_, _, c) in KINDS.items()
) + """
/* timeline card */
.tl-card { border:1px solid #1F2430; border-left:3px solid var(--c); border-radius:10px;
  padding:.5rem .8rem .55rem; background: color-mix(in srgb, var(--c) 7%, #12151C);
  transition: transform .14s ease, background .14s ease, box-shadow .14s ease; }
[class*="st-key-blk-"]:hover .tl-card { transform: translateX(3px);
  background: color-mix(in srgb, var(--c) 13%, #12151C); }
.tl-card.sel { border-color: color-mix(in srgb, var(--c) 70%, #1F2430);
  box-shadow: 0 0 0 1px var(--c), 0 0 26px -6px var(--c);
  background: color-mix(in srgb, var(--c) 15%, #12151C); }
.tl-card.out { opacity:.42; filter:saturate(.35); }
.tl-head { display:flex; align-items:center; gap:.45rem; font-size:.68rem; font-weight:700;
  letter-spacing:.12em; color:var(--c); }
.tl-num { margin-left:auto; color:#6B7280; font-weight:500; letter-spacing:0; font-family:'JetBrains Mono',monospace; }
.tl-prev { margin-top:.2rem; color:#D1D5DB; font-size:.86rem; line-height:1.35;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.k-thinking .tl-prev { color:#A1A1AA; font-style:italic; }
.tl-prev.mono { font-family:'JetBrains Mono',monospace; font-size:.8rem; }

/* invisible hit-area button stretched over each card */
[class*="st-key-hit-"] { position:absolute !important; inset:0; z-index:3; margin:0 !important; }
[class*="st-key-hit-"] div, [class*="st-key-hit-"] button { width:100% !important; height:100% !important; }
[class*="st-key-hit-"] button { opacity:0; cursor:pointer; }

/* live ghost block + compaction divider */
.tl-card.ghost { border-style:dashed; border-left-style:solid; position:relative; overflow:hidden; }
.tl-card.ghost::after { content:""; position:absolute; inset:0;
  background: linear-gradient(100deg, transparent 20%, color-mix(in srgb, var(--c) 22%, transparent) 50%, transparent 80%);
  background-size:200% 100%; animation: shimmer 1.3s linear infinite; }
@keyframes shimmer { 0% { background-position:150% 0; } 100% { background-position:-50% 0; } }
[class*="st-key-blk-"][class*="-live"]::before { animation: pulse 1.1s ease-in-out infinite; }
@keyframes pulse { 50% { opacity:.35; transform:scale(.75); } }
.divider { display:flex; align-items:center; gap:.6rem; color:#6B7280; font-size:.72rem;
  letter-spacing:.08em; margin:.15rem 0; }
.divider::before, .divider::after { content:""; flex:1; height:1px;
  background: repeating-linear-gradient(90deg,#2A3040 0 6px,transparent 6px 10px); }

/* inspector */
.insp-head { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .6rem; }
.badge { display:inline-flex; align-items:center; gap:.45rem; padding:.28rem .7rem; border-radius:8px;
  font-size:.74rem; font-weight:700; letter-spacing:.12em; color:var(--c);
  background: color-mix(in srgb, var(--c) 14%, #12151C); border:1px solid color-mix(in srgb, var(--c) 35%, #1F2430); }
.insp-meta { margin-left:auto; color:#6B7280; font-size:.78rem; }
.insp-meta.out { color:#FBBF24; }
.fn-name { font-family:'JetBrains Mono',monospace; font-size:1.25rem; color:#FBBF24; margin:.1rem 0 .6rem; }
.kv { display:grid; grid-template-columns: max-content 1fr; border:1px solid #1F2430; border-radius:10px; overflow:hidden; }
.kv div { padding:.4rem .8rem; border-top:1px solid #1F2430; font-family:'JetBrains Mono',monospace; font-size:.84rem; }
.kv div:nth-child(-n+2) { border-top:none; }
.kv .k { color:#8B93A7; background:#12151C; }
.sub { font-size:.7rem; font-weight:600; letter-spacing:.14em; color:#6B7280; margin:1rem 0 .35rem; }
.st-key-thought { border-left:2px solid #A78BFA; padding-left:1rem; color:#A1A1AA; font-style:italic; }
.empty { text-align:center; color:#6B7280; padding:4rem 1rem; border:1px dashed #1F2430; border-radius:12px; }
.empty .big { font-size:2rem; margin-bottom:.4rem; }
</style>
"""


# ---------------------------------------------------------------- rendering


def render_header(agent, entries: list[Entry], ctx: list, running: bool) -> None:
    tool_calls = sum(turn_kind(e.msg) == "tool_call" for e in entries)
    compacted = sum(not e.in_context for e in entries)
    if running:
        status = '<span class="dot run"></span><b>running</b>'
    elif agent.pending_calls:
        status = '<span class="dot" style="background:#FB923C"></span><b>awaiting approval</b>'
    else:
        status = '<span class="dot"></span>idle'
    pills = [
        f'<span class="pill model">{html.escape(agent.model)}</span>',
        f'<span class="pill">{status}</span>',
        f'<span class="pill"><b>{len(entries)}</b> turns</span>',
        f'<span class="pill"><b>{tool_calls}</b> tool calls</span>',
        f'<span class="pill"><b>{len(ctx)}</b> in context</span>',
    ]
    if compacted:
        pills.append(f'<span class="pill"><b>{compacted}</b> compacted</span>')

    with st.container(horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="hdr"><span class="hdr-title">jean-code</span>{"".join(pills)}</div>')
        if running:
            st.button("Stop", icon=":material/stop_circle:", on_click=agent.interrupt, type="primary")
        else:
            st.button("New chat", icon=":material/refresh:", on_click=new_chat)

    if error := st.session_state.run["error"]:
        st.error(error, icon=":material/error:")


def render_timeline(entries: list[Entry], selected: Entry | None, running: bool, agent) -> None:
    with st.container(height=640, autoscroll=True, border=False):
        if not entries and not running:
            st.html(
                '<div class="empty"><div class="big">✦</div>'
                "Ask something below — every thought, tool call and answer<br>shows up here as a block.</div>"
            )
        with st.container(key="rail"):
            compacted_run = 0
            for e in entries:
                if e.in_context and compacted_run:
                    st.html(f'<div class="divider">⟲ {compacted_run} turns compacted</div>')
                compacted_run = 0 if e.in_context else compacted_run + 1

                kind = turn_kind(e.msg)
                label, icon, _ = KINDS[kind]
                classes = f"tl-card k-{kind}"
                classes += " sel" if selected is e else ""
                classes += "" if e.in_context else " out"
                mono = " mono" if kind in MONO_KINDS else ""
                with st.container(key=f"blk-{kind}-{e.uid}"):
                    st.html(
                        f'<div class="{classes}"><div class="tl-head"><span>{icon}</span>'
                        f'<span>{label}</span><span class="tl-num">#{e.uid}</span></div>'
                        f'<div class="tl-prev{mono}">{html.escape(preview(e.msg, kind))}</div></div>'
                    )
                    st.button(" ", key=f"hit-{e.uid}", on_click=select, args=(e.uid,), width="stretch")

            if running:
                kind, status = live_status(agent)
                label, icon, _ = KINDS[kind]
                with st.container(key=f"blk-{kind}-live"):
                    st.html(
                        f'<div class="tl-card ghost k-{kind}"><div class="tl-head"><span>{icon}</span>'
                        f'<span>{label}</span></div><div class="tl-prev">{status}</div></div>'
                    )
            elif agent.pending_calls:
                render_approval(agent)


def render_approval(agent) -> None:
    """The first call waiting for approval, with Approve / Deny. Sending a message instead also answers it."""
    name, arguments = agent.pending_calls[0]
    label, icon, _ = KINDS["approval"]
    with st.container(key="blk-approval-pending"):
        st.html(
            f'<div class="tl-card k-approval"><div class="tl-head"><span>{icon}</span><span>{label}</span></div>'
            f'<div class="tl-prev mono">{html.escape(name)}() wants to run:</div></div>'
        )
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


def render_turn(entries: list[Entry], selected: Entry | None, ctx: list, running: bool, pending: bool) -> None:
    if selected is None:
        st.html('<div class="empty"><div class="big">◇</div>Click a block in the timeline to inspect it.</div>')
        return

    pos = entries.index(selected)
    msg, kind = selected.msg, turn_kind(selected.msg)
    label, icon, _ = KINDS[kind]
    if selected.in_context:
        where = next(i for i, m in enumerate(ctx) if m is msg)
        meta = f'<span class="insp-meta">context message {where + 1} of {len(ctx)}</span>'
    else:
        meta = '<span class="insp-meta out">⟲ compacted — no longer in context</span>'

    with st.container(horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="insp-head k-{kind}"><span class="badge">{icon} {label} #{selected.uid}</span>{meta}</div>')
        if st.session_state.selected_uid is None:
            st.html('<span class="pill">⤓ following latest</span>', width="content")
        else:
            st.button("Follow latest", icon=":material/vertical_align_bottom:", on_click=select, args=(None,),
                      type="tertiary")

    if kind == "tool_call":
        if mget(msg, "thinking"):
            with st.container(key="thought"):
                st.markdown(mget(msg, "thinking"))
        if mget(msg, "content"):
            st.markdown(mget(msg, "content"))
        results = results_of(entries, pos)
        for i, call in enumerate(mget(msg, "tool_calls")):
            fn = call["function"]
            st.html(f'<div class="fn-name">{html.escape(fn["name"])}()</div>')
            cells = "".join(
                f'<div class="k">{html.escape(str(k))}</div><div>{html.escape(json.dumps(v))}</div>'
                for k, v in fn["arguments"].items()
            ) or '<div class="k">—</div><div>no arguments</div>'
            st.html(f'<div class="kv">{cells}</div>')
            st.html('<div class="sub">↳ RESULT</div>')
            if i < len(results):
                result = results[i]
                st.code(str(mget(result.msg, "content")), language=None, wrap_lines=True)
                st.button("Jump to result", icon=":material/arrow_downward:", on_click=select, args=(result.uid,),
                          key=f"jump-result-{result.uid}")
            elif running and pos + len(results) == len(entries) - 1:
                st.caption("running…")
            elif pending and pos + len(results) == len(entries) - 1:
                st.caption("waiting for approval")
            else:
                st.caption("no result")
    elif kind == "tool_result":
        st.html(f'<div class="sub">FROM</div><div class="fn-name">{html.escape(str(mget(msg, "tool_name")))}()</div>')
        st.code(str(mget(msg, "content")), language=None, wrap_lines=True)
        if call := call_of(entries, pos):
            st.button("Jump to call", icon=":material/arrow_upward:", on_click=select, args=(call.uid,))
    elif kind == "thinking":
        with st.container(key="thought"):
            st.markdown(mget(msg, "thinking"))
    else:
        if mget(msg, "thinking"):
            with st.container(key="thought"):
                st.markdown(mget(msg, "thinking"))
        st.markdown(mget(msg, "content") or "*(no content)*")

    with st.expander("Raw message", icon=":material/data_object:"):
        st.json(to_jsonable(msg))


def render_request(agent) -> None:
    req = agent.last_request
    if not req:
        st.html('<div class="empty"><div class="big">⇅</div>No request sent yet.</div>')
        return
    tools = ", ".join(getattr(t, "__name__", str(t)) for t in req.get("tools", [])) or "none"
    st.html(
        '<div class="hdr">'
        f'<span class="pill model">{html.escape(str(req.get("model")))}</span>'
        f'<span class="pill">think <b>{html.escape(str(req.get("think")))}</b></span>'
        f'<span class="pill">stream <b>{html.escape(str(req.get("stream")))}</b></span>'
        f'<span class="pill">tools <b>{html.escape(tools)}</b></span>'
        f'<span class="pill"><b>{len(req.get("messages", []))}</b> messages</span></div>'
    )
    with st.expander("Messages sent", icon=":material/forum:"):
        st.json(to_jsonable(req.get("messages", [])), expanded=1)
    with st.expander("Full request", icon=":material/data_object:"):
        st.json(to_jsonable(req), expanded=1)


def render_state(agent) -> None:
    attrs = {k: to_jsonable(v) for k, v in vars(agent).items()
             if k not in ("client", "messages", "last_request")}  # shown elsewhere / not interesting
    scalars = {k: v for k, v in attrs.items() if not isinstance(v, (dict, list))}
    cells = "".join(
        f'<div class="k">{html.escape(k)}</div><div>{html.escape(json.dumps(v))}</div>' for k, v in scalars.items()
    )
    if cells:
        st.html(f'<div class="kv">{cells}</div>')
    for attr, value in attrs.items():
        if attr not in scalars:
            with st.expander(attr, icon=":material/data_object:"):
                st.json(value, expanded=1)


# ---------------------------------------------------------------- app

st.set_page_config(page_title="jean-code", page_icon="🧮", layout="wide")
st.html(CSS)

st.sidebar.toggle("Run bash in Docker sandbox", key="sandbox", on_change=new_chat,
                  help="Off: commands run on this machine in workspace/ and need approval. "
                       "On: they run in the jean-code-sandbox container without asking. Starts a new chat.")

if "agent" not in st.session_state:
    new_chat()
    st.session_state.next_uid = 0
    st.session_state.was_running = False


@st.fragment(run_every=POLL_SECONDS if is_running() else None)
def live_view() -> None:
    agent, entries = st.session_state.agent, st.session_state.entries
    running = is_running()
    if st.session_state.was_running and not running:
        st.session_state.was_running = False
        st.rerun()  # full rerun: stop polling and re-enable the input

    ctx = list(agent.messages)  # snapshot: the run thread appends concurrently
    sync_history(entries, ctx)
    by_uid = {e.uid: e for e in entries}
    selected = by_uid.get(st.session_state.selected_uid) or (entries[-1] if entries else None)

    render_header(agent, entries, ctx, running)
    timeline_col, inspector_col = st.columns([2, 3], gap="large")

    with timeline_col:
        st.html('<div class="section">TIMELINE</div>')
        render_timeline(entries, selected, running, agent)
        prompt = st.chat_input("Agent is working…" if running else "Message the agent…", disabled=running)
        if prompt:
            start_run(agent.run_turn, prompt)
            st.rerun()

    with inspector_col:
        turn_tab, request_tab, state_tab = st.tabs(["Turn", "Request", "Agent state"])
        with turn_tab:
            render_turn(entries, selected, ctx, running, bool(agent.pending_calls))
        with request_tab:
            render_request(agent)
        with state_tab:
            render_state(agent)


live_view()
