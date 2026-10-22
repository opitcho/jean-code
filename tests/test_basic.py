"""Tests for the chat and web agents' wiring.

Nothing here calls a model. Each test names the bug it would catch.
"""

import pytest

from conftest import FakeClient, response
from jean_code.agents.basic import chat_agent, web_agent


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # builds a client; nothing here sends a request


def test_the_chat_agent_sends_no_tools():
    # Catches a tool, skill or subagent creeping into the agent that shows the model alone.
    client = FakeClient([response("hello")])
    agent = chat_agent("m", client=client)
    try:
        assert agent._tools == {}
        assert agent.run_turn("hi") == "answered"
        assert "tools" not in client.requests[0]
    finally:
        agent.close()


def test_the_web_agent_has_only_the_web_tools():
    # Catches a shell or subagent on the web agent, or a web tool that waits for an approval nobody can give.
    agent = web_agent("m")
    try:
        assert set(agent._tools) == {"web_search", "fetch_page"}
        assert agent.needs_approval == set()
    finally:
        agent.close()


def test_the_chat_and_web_agents_stream():
    # Catches a reply that lands whole: nothing shows while it's written, and a stop can't cut it off.
    for build in (chat_agent, web_agent):
        agent = build("m")
        assert agent.stream
        agent.close()
