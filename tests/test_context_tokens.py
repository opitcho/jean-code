"""Tests for `Agent.context_tokens`: the size of the next request's prompt, which decides when to compact.

Each test names the bug it would catch. The bookkeeping tests set the estimator to a flat 10 tokens per
estimated value, so the expected numbers are exact and don't depend on the estimator.
"""

import pytest

import agent as agent_module
from agent import Agent, estimate_tokens
from conftest import FakeClient, call, response, usage


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@pytest.fixture
def flat(monkeypatch):
    monkeypatch.setattr(agent_module, "estimate_tokens", lambda value: 10)


def make_agent(responses, **kwargs) -> Agent:
    return Agent(client=FakeClient(responses), tools={"add": add}, **kwargs)


def test_reply_counted_once(flat):
    # Catches the reply counted twice (measured and estimated), or the measurement stored before it was appended.
    agent = make_agent([response("hi", usage=usage(prompt=1000, completion=50))])
    agent.run_turn("hello")
    assert agent.context_tokens == 1050


def test_tool_results_after_the_call_are_estimated(flat):
    # Catches tool results not counted: an off-by-one on the index stored with the measurement.
    agent = make_agent([response(tool_calls=[call("add", a=1, b=2), call("add", a=3, b=4)],
                                 usage=usage(prompt=1000, completion=50))])
    agent.messages.append({"role": "user", "content": "add twice"})
    agent.step()
    assert [m["role"] for m in agent.messages[-2:]] == ["tool", "tool"]
    assert agent.context_tokens == 1050 + 20


def test_new_user_message_counted_when_the_next_call_is_made(flat):
    # Catches the loop seeing the old size, so the hard-limit guard and the trigger act one call late.
    seen = []

    class Watching(FakeClient):
        def complete(self, request):
            seen.append(agent.context_tokens)
            return super().complete(request)

    agent = Agent(client=Watching([response("a", usage=usage(prompt=1000, completion=50)), response("b")]),
                  tools={"add": add}, keep_first=2)  # no head marker before "two": only the user message is new
    agent.run_turn("one")
    agent.run_turn("two")
    assert seen[1] == 1050 + 10


def test_a_fresh_measurement_replaces_the_estimates(flat):
    # Catches stale estimates added on top of a new measurement.
    agent = make_agent([
        response(tool_calls=[call("add", a=1, b=2)], usage=usage(prompt=1000, completion=50)),
        response("3", usage=usage(prompt=2000, completion=80)),
    ])
    agent.run_turn("add")
    assert agent.context_tokens == 2080


def test_a_response_without_usage_keeps_the_last_measurement(flat):
    # Catches a crash on missing usage, or the count freezing while messages keep coming.
    agent = make_agent([
        response(tool_calls=[call("add", a=1, b=2)], usage=usage(prompt=1000, completion=50)),
        response("3", no_usage=True),
    ])
    agent.run_turn("add")
    # measured 1050 after the first reply, then the tool result and the second reply
    assert agent.context_tokens == 1050 + 10 + 10


def test_before_any_call_the_tool_schemas_count():
    # Catches the estimate leaving out the tool schemas, which are part of every prompt.
    without = Agent(client=FakeClient([]), system_prompt="sys")
    with_tools = Agent(client=FakeClient([]), system_prompt="sys", tools={"add": add})
    assert with_tools.context_tokens > without.context_tokens + 20


def test_estimator_counts_bytes_not_characters():
    # Catches counting characters, which is too low for Chinese (~0.6 tokens per character per DeepSeek).
    chinese = {"role": "user", "content": "请帮我修复这个测试" * 12 + "好"}  # 109 + 1 characters
    assert estimate_tokens(chinese) >= 0.6 * 110
    assert estimate_tokens("x" * 300) == 101  # 302 bytes with the quotes, rounded up
