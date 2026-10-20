"""One whole turn on a fixed trace: the model reasons and calls `add`, the tool runs, the model answers. Checks the
agent's state at the end, all of it, against what the trace says it must be.

The two replies are real ones from OpenRouter, in the shape a non-streamed response has them: one merged
`reasoning_details` entry, `index` on each tool call and `refusal: null`.
"""

import pytest

from conftest import FakeClient
from jean_code.core.agent import DEFAULT_MODEL, Agent
from jean_code.core.llm import render_message
from jean_code.core.tool_schema import tool
from jean_code.core.usage import ModelCall

SYSTEM = "You are a calculator. Use the add tool for sums."
QUESTION = "What is 1234 + 5678? Use the add tool, then tell me the result."
REASONING = "We need answer. Need use add tool. Call with a=1234 b=5678."
CALL_ID = "call_86f3eb8e58454d2fbde8d831"
ANSWER = "1234 + 5678 = **6912**"
META = {"model": "deepseek/deepseek-v4.1-flash", "object": "chat.completion"}

CALLING = {
    "role": "assistant",
    "content": "",
    "refusal": None,
    "reasoning": REASONING,
    "reasoning_details": [{"type": "reasoning.text", "text": REASONING, "format": "unknown", "index": 0}],
    "tool_calls": [
        {
            "index": 0,
            "id": CALL_ID,
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1234, "b": 5678}'},
        }
    ],
}
ANSWERING = {"role": "assistant", "content": ANSWER, "refusal": None}


def usage(prompt: int, completion: int, reasoning: int, cost: float) -> dict:
    """A usage payload in OpenRouter's shape, with the counts and cost of a real call."""
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost": cost,
        "is_byok": False,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0},
        "cost_details": {"upstream_inference_cost": cost},
        "completion_tokens_details": {"reasoning_tokens": reasoning, "image_tokens": 0, "audio_tokens": 0},
    }


TRACE = [
    {
        **META,
        "id": "gen-1789747504-RyoGTjA4HffARiEWPE0t",
        "provider": "Relace",
        "created": 1789747504,
        "choices": [{"index": 0, "message": CALLING, "finish_reason": "tool_calls"}],
        "usage": usage(317, 76, 15, 9.315e-05),
    },
    {
        **META,
        "id": "gen-1789747506-TZDb0XyTFfYczXFe7HA9",
        "provider": "Alibaba",
        "created": 1789747506,
        "choices": [{"index": 0, "message": ANSWERING, "finish_reason": "stop"}],
        "usage": usage(410, 12, 0, 6.87e-05),
    },
]

# The whole transcript after the turn, as wire dicts. Each reply is kept as the provider sent it, `refusal: null`
# aside: render_message leaves out None, so it is never sent back.
TRANSCRIPT = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": QUESTION},
    {key: value for key, value in CALLING.items() if value is not None},
    {"role": "tool", "tool_call_id": CALL_ID, "content": "6912"},
    {"role": "assistant", "content": ANSWER},
]


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def run_trace() -> tuple[Agent, FakeClient, str]:
    """An agent with `add`, after one turn on TRACE: the agent, its client and how the turn ended."""
    client = FakeClient(TRACE)
    agent = Agent(client=client, system_prompt=SYSTEM, tools={"add": add})
    end = agent.run_turn(QUESTION)
    return agent, client, end


def test_the_turn_ends_with_the_exact_transcript():
    # Catches a reply changed on the way into the transcript (reasoning dropped, a call's fields lost), a wrong or
    # missing tool result, or a message added or left out.
    agent, client, end = run_trace()
    assert end == "answered"
    assert [render_message(m) for m in agent.messages] == TRANSCRIPT
    assert client.responses == []  # both replies used, no more calls


def test_each_request_sends_the_transcript_so_far():
    # Catches what the model gets back differing from what the agent stored: reasoning stripped, the tool result
    # missing, or the settings and tools changing between calls.
    _, client, _ = run_trace()
    assert [request["messages"] for request in client.requests] == [TRANSCRIPT[:2], TRANSCRIPT[:4]]
    for request in client.requests:
        assert request["model"] == DEFAULT_MODEL
        assert request["reasoning"] == {"effort": "medium"}
        assert [schema["function"]["name"] for schema in request["tools"]] == ["add"]


def test_each_call_is_booked_with_its_receipt():
    # Catches a call left out of the usage or the budget, or one booked with another call's metadata or counts.
    agent, _, _ = run_trace()
    path = agent.node.path
    assert agent.node.entries == [
        ModelCall(
            "gen-1789747504-RyoGTjA4HffARiEWPE0t",
            "Relace",
            "deepseek/deepseek-v4.1-flash",
            1789747504,
            9.315e-05,
            path,
            prompt_tokens=317,
            completion_tokens=76,
            reasoning_tokens=15,
            finish_reason="tool_calls",
        ),
        ModelCall(
            "gen-1789747506-TZDb0XyTFfYczXFe7HA9",
            "Alibaba",
            "deepseek/deepseek-v4.1-flash",
            1789747506,
            6.87e-05,
            path,
            prompt_tokens=410,
            completion_tokens=12,
            finish_reason="stop",
        ),
    ]
    assert agent.node.spent == pytest.approx(9.315e-05 + 6.87e-05)
    assert agent.context_tokens == 410 + 12  # measured by the last call: its prompt plus the answer
