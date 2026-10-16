"""Tests for the completion codec in `llm.py`: parsing a provider's response into typed messages and usage, and
rendering messages and requests back to the wire format.

Each test names the bug it would catch.
"""

import json

import pytest
from pydantic import ValidationError

from conftest import REAL_ASSISTANT, response, usage
from jean_code.core.llm import (
    AssistantMessage,
    ChatRequest,
    MalformedResponse,
    SystemMessage,
    UserMessage,
    parse_completion,
    parse_receipt,
    render_message,
    render_request,
)


def test_real_message_round_trips_exactly():
    # Catches losing reasoning_details, a tool call's index or type, or any provider extra.
    assert render_message(AssistantMessage.model_validate(REAL_ASSISTANT)) == REAL_ASSISTANT


def test_plain_reply_renders_without_tool_calls():
    # Catches `"tool_calls": []` being added to every plain reply we send back, which some providers reject.
    raw = {"role": "assistant", "content": "hi"}
    assert render_message(AssistantMessage.model_validate(raw)) == raw


def test_null_content_and_arguments_read_as_empty():
    # Catches a codec stricter than today's `json.loads(arguments or "{}")`.
    msg = AssistantMessage.model_validate(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ping", "arguments": None}}],
        }
    )
    assert msg.content == "" and msg.tool_calls[0].function.arguments == ""


def test_arguments_sent_as_an_object_become_json_text():
    # Catches a provider that sends arguments as an object failing a call the tool could run.
    msg = AssistantMessage.model_validate(
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "add", "arguments": {"a": 1, "b": 2}}}]}
    )
    assert json.loads(msg.tool_calls[0].function.arguments) == {"a": 1, "b": 2}


def test_unknown_fields_are_kept():
    # Catches a fallback provider's extra fields being dropped from what we send back.
    raw = {
        "role": "assistant",
        "content": "x",
        "vendor_note": "n",
        "tool_calls": [
            {"id": "c1", "index": 0, "type": "function", "vendor": 1, "function": {"name": "ping", "arguments": "{}"}}
        ],
    }
    assert render_message(AssistantMessage.model_validate(raw)) == raw


def test_messages_are_frozen():
    # Catches a message changing after it's in a transcript that compaction keeps as history.
    with pytest.raises(ValidationError):
        AssistantMessage(content="a").content = "b"  # type: ignore[misc]


def test_completion_from_a_response():
    c = parse_completion(response("hi", id="gen-1"))
    assert (c.message.content, c.finish_reason, c.receipt.generation_id, c.receipt.usage.cost) == (
        "hi",
        "stop",
        "gen-1",
        0.001,
    )


def test_missing_usage_gives_a_receipt_without_usage():
    assert parse_completion(response("hi", no_usage=True)).receipt.usage is None


def test_unknown_finish_reason_passes_through():
    r = response("hi")
    r["choices"][0]["finish_reason"] = "vendor_stop"
    assert parse_completion(r).finish_reason == "vendor_stop"


def test_empty_choices_raises_with_the_receipt():
    # Catches a paid call whose reply can't be read losing what's needed to book it.
    r = response("x", usage=usage(cost=0.005))
    r["choices"] = []
    with pytest.raises(MalformedResponse, match="choices") as e:
        parse_completion(r)
    assert e.value.receipt is not None and e.value.receipt.usage.cost == 0.005


@pytest.mark.parametrize(
    "message, path",
    [
        (
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"arguments": "{}"}}]},
            "choices.0.message.tool_calls.0.function.name",
        ),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "ping"}}]}, "choices.0.message.tool_calls.0.id"),
        ({"role": "assistant", "content": [{"type": "text", "text": "hi"}]}, "choices.0.message.content"),
    ],
    ids=["tool call without a name", "tool call without an id", "content as a list of parts"],
)
def test_unusable_message_names_the_path(message, path):
    r = response("x")
    r["choices"][0]["message"] = message
    with pytest.raises(MalformedResponse, match=path.replace(".", r"\.")):
        parse_completion(r)


def test_malformed_usage_is_reported_and_dropped():
    # Catches a malformed usage object either crashing or being reported as "no cost".
    receipt, error = parse_receipt({"id": "gen-u", "usage": {"cost": 0.004, "prompt_tokens_details": []}})
    assert receipt.generation_id == "gen-u" and receipt.usage is None
    assert error is not None and "prompt_tokens_details" in error
    r = response("x", id="gen-u")
    r["usage"] = {"cost": 0.004, "prompt_tokens_details": []}
    with pytest.raises(MalformedResponse, match="malformed usage in gen-u"):
        parse_completion(r)


def test_odd_metadata_reads_as_none():
    # Catches an odd id or timestamp from a fallback provider costing us the record of a paid call.
    receipt, error = parse_receipt({"id": 7, "created": "1789748403", "provider": None, "usage": {"cost": 0.001}})
    assert (receipt.generation_id, receipt.created, receipt.provider, receipt.usage.cost, error) == (
        None,
        None,
        None,
        0.001,
        None,
    )


def test_not_an_object_has_no_receipt():
    with pytest.raises(MalformedResponse) as e:
        parse_completion([])
    assert e.value.receipt is None


def test_error_text_leaves_out_the_input():
    # Catches an error message that drags the whole response along.
    r = response("SECRET-CONTENT")
    r["choices"][0]["message"]["tool_calls"] = [{"id": "c1", "function": {}}]
    with pytest.raises(MalformedResponse) as e:
        parse_completion(r)
    assert "SECRET-CONTENT" not in str(e.value)


def test_render_request_matches_todays_body():
    # Catches a request body that differs from what Agent._build_request sends today.
    req = ChatRequest("m", (SystemMessage(content="sys"), UserMessage(content="hi")), ({"type": "function"},), "low")
    assert render_request(req) == {
        "model": "m",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "reasoning": {"effort": "low"},
        "tools": [{"type": "function"}],
    }
    assert render_request(ChatRequest("m", (), (), None)) == {"model": "m", "messages": []}


def test_null_tool_calls_read_as_none():
    # Catches `"tool_calls": null` from an OpenAI-compatible fallback failing a reply the agent always accepted.
    r = response("hi")
    r["choices"][0]["message"]["tool_calls"] = None
    message = parse_completion(r).message
    assert message.tool_calls == () and render_message(message) == {"role": "assistant", "content": "hi"}
