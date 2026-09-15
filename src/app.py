"""Streamlit chat UI with a live, generic view of BaseAgent's internal state.

Run with: uv run streamlit run src/app.py
"""

import streamlit as st

from base import BaseAgent

MODEL = "qwen3:8b"


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
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def mget(msg, key, default=None):
    """Read `key` from a message whether it's a plain dict or an attribute-style object."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


st.set_page_config(page_title="min-agent", layout="wide")

if "agent" not in st.session_state:
    st.session_state.agent = BaseAgent(model=MODEL)
agent = st.session_state.agent

chat_col, state_col = st.columns([3, 2])

with chat_col:
    st.subheader("Chat")
    for msg in agent.messages:
        role = mget(msg, "role")
        if role == "system":
            continue
        with st.chat_message(role):
            st.markdown(mget(msg, "content") or "*(no content)*")

    user_input = st.chat_input("Message the agent...")
    if user_input:
        with st.spinner("Thinking..."):
            agent.step(user_input)
        st.rerun()

with state_col:
    st.subheader("Agent state")
    for attr, value in vars(agent).items():
        if attr == "client":  # not interesting / not JSON-serializable
            continue

        if attr == "messages":
            st.markdown(f"**messages** ({len(value)})")
            if not value:
                st.caption("(empty)")
            for i, msg in enumerate(value):
                role = mget(msg, "role", "?")
                with st.expander(f"{i}: {role}", expanded=False):
                    st.json(to_jsonable(msg))
            continue

        with st.expander(attr, expanded=(attr == "last_request")):
            st.json(to_jsonable(value))
