"""StreamingClient: a streamed reply yields its text as it arrives and is rebuilt into the reply `complete` returns."""

import itertools
import json
import random

import httpx
import pytest

from conftest import REAL_ASSISTANT, REAL_META, REAL_USAGE, usage
from jean_code.core import llm
from jean_code.core.llm import ChatRequest, Completion, MalformedResponse, StreamingClient, TextDelta, render_message

REQUEST = ChatRequest("m", (), (), None)
# The chunks of a real streamed reply, from `.experiments/output/openrouter_stream_roundtrip.txt` (round 1).
META = {
    "id": "gen-1789747504-RyoGTjA4HffARiEWPE0t",
    "object": "chat.completion.chunk",
    "created": 1789747504,
    "model": "deepseek/deepseek-v4.1-flash",
    "provider": "Relace",
}
THOUGHTS = ["We", " need answer.", " Need use add tool. Call", " with a", "=1234 b=567", "8."]
CALL_ID = "call_86f3eb8e58454d2fbde8d831"


def chunk(delta: dict, finish_reason=None, **extra) -> dict:
    choice = {"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": finish_reason}
    return {**META, "choices": [{**choice, "native_finish_reason": finish_reason}], **extra}


def thought(text: str) -> dict:
    detail = {"type": "reasoning.text", "text": text, "format": "unknown", "index": 0}
    return chunk({"content": "", "reasoning": text, "reasoning_details": [detail]})


def arguments(text: str) -> dict:
    return chunk({"content": None, "tool_calls": [{"index": 0, "function": {"arguments": text}}]})


TOOL_CALL_STREAM = [
    *map(thought, THOUGHTS),
    chunk(
        {
            "content": None,
            "tool_calls": [
                {"index": 0, "id": CALL_ID, "type": "function", "function": {"name": "add", "arguments": ""}}
            ],
        }
    ),
    arguments(""),
    arguments("{"),
    arguments('"a": 1234'),
    arguments(', "b": 5678}'),
    chunk({"content": "", "reasoning": None}, "tool_calls"),
    chunk({"content": ""}, "tool_calls", service_tier=None, usage=usage(317, 76, reasoning=15, cost=9.315e-05)),
]


def sse(*chunks: dict, keep_alives: int = 0) -> httpx.Response:
    """A streamed response: OpenRouter's keep-alive comments, each chunk as a `data:` event, then the end marker."""
    events = [": OPENROUTER PROCESSING"] * keep_alives + [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return httpx.Response(200, text="".join(f"{event}\n\n" for event in events))


def streaming_client(monkeypatch, *replies: httpx.Response, sent: list | None = None) -> StreamingClient:
    """A StreamingClient whose HTTP calls return `replies` in order, with no backoff sleeps. Request bodies go to
    `sent`."""
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    queue = list(replies)

    def reply(request: httpx.Request) -> httpx.Response:
        if sent is not None:
            sent.append(json.loads(request.content))
        return queue.pop(0)

    client = StreamingClient("https://example.test/api", "key", extra={"provider": {"order": ["deepseek"]}})
    client.http = httpx.Client(transport=httpx.MockTransport(reply))
    return client


def split(items) -> tuple[list[TextDelta], Completion]:
    """The deltas of a stream and its completion, checking the completion comes last and only there."""
    *deltas, completion = items
    assert all(isinstance(delta, TextDelta) for delta in deltas)
    assert isinstance(completion, Completion)
    return deltas, completion


def test_a_streamed_tool_call_is_rebuilt_into_the_reply_complete_would_return(monkeypatch):
    # Catches tool-call or reasoning fragments not being merged by index, or a reply shaped unlike a stored one.
    deltas, completion = split(streaming_client(monkeypatch, sse(*TOOL_CALL_STREAM)).stream(REQUEST))
    reasoning = "".join(THOUGHTS)
    assert deltas == [TextDelta("reasoning", text) for text in THOUGHTS]
    assert render_message(completion.message) == {
        "role": "assistant",
        "content": "",
        "reasoning": reasoning,
        "reasoning_details": [{"type": "reasoning.text", "text": reasoning, "format": "unknown", "index": 0}],
        "tool_calls": [
            {
                "index": 0,
                "id": CALL_ID,
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1234, "b": 5678}'},
            }
        ],
    }
    assert render_message(completion.message).keys() == REAL_ASSISTANT.keys()
    assert completion.finish_reason == "tool_calls"


def test_the_metadata_comes_from_the_chunks_and_the_usage_from_the_last(monkeypatch):
    # Catches a streamed call that can't be booked: metadata or the final usage chunk being dropped.
    _, completion = split(streaming_client(monkeypatch, sse(*TOOL_CALL_STREAM)).stream(REQUEST))
    assert (completion.generation_id, completion.provider, completion.created) == (META["id"], "Relace", 1789747504)
    assert completion.usage is not None
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.cost) == (
        317,
        76,
        9.315e-05,
    )


def test_reasoning_then_answer_stream_in_order_past_keep_alives(monkeypatch):
    # Catches keep-alive comments breaking the parse, or answer text arriving out of order.
    stream = sse(
        thought("The tool returned 6912."),
        chunk({"content": "123"}),
        chunk({"content": "4 + 5678 ="}),
        chunk({"content": " **6912**"}),
        chunk({"content": "", "reasoning": None}, "stop", usage=usage()),
        keep_alives=3,
    )
    deltas, completion = split(streaming_client(monkeypatch, stream).stream(REQUEST))
    assert [(d.kind, d.text) for d in deltas] == [
        ("reasoning", "The tool returned 6912."),
        ("content", "123"),
        ("content", "4 + 5678 ="),
        ("content", " **6912**"),
    ]
    assert completion.message.content == "1234 + 5678 = **6912**"
    assert completion.message.tool_calls == ()
    assert completion.finish_reason == "stop"


def test_the_request_asks_for_a_stream_with_the_clients_extra_fields(monkeypatch):
    # Catches streaming not being requested, or provider routing being dropped from streamed calls.
    sent: list = []
    split(streaming_client(monkeypatch, sse(*TOOL_CALL_STREAM), sent=sent).stream(REQUEST))
    assert sent == [{"provider": {"order": ["deepseek"]}, "model": "m", "messages": [], "stream": True}]


def test_nothing_is_sent_until_the_stream_is_read(monkeypatch):
    # Catches a request sent on creating the generator, which a caller that never reads it wouldn't expect.
    sent: list = []
    stream = streaming_client(monkeypatch, sse(*TOOL_CALL_STREAM), sent=sent).stream(REQUEST)
    assert sent == []
    next(stream)
    assert len(sent) == 1


def test_a_rate_limited_stream_is_retried(monkeypatch):
    # Catches the retry loop being lost when the request is held open for streaming.
    sent: list = []
    client = streaming_client(monkeypatch, httpx.Response(429, text="slow down"), sse(*TOOL_CALL_STREAM), sent=sent)
    _, completion = split(client.stream(REQUEST))
    assert completion.generation_id == META["id"]
    assert len(sent) == 2


def test_an_http_error_raises_with_its_body(monkeypatch):
    # Catches an unread streamed error body hiding the provider's message.
    client = streaming_client(monkeypatch, httpx.Response(400, text="bad model"))
    with pytest.raises(RuntimeError, match=r"HTTP 400 .*bad model"):
        list(client.stream(REQUEST))


def test_an_error_chunk_raises(monkeypatch):
    # Catches a provider error mid-stream ending as a silent, truncated reply.
    error = {**META, "error": {"message": "provider down"}, "choices": []}
    client = streaming_client(monkeypatch, sse(thought("We"), error))
    with pytest.raises(RuntimeError, match="provider down"):
        list(client.stream(REQUEST))


def test_a_stream_with_no_chunk_is_malformed(monkeypatch):
    # Catches an empty stream being parsed into an empty reply with no metadata.
    client = streaming_client(monkeypatch, sse())
    with pytest.raises(MalformedResponse, match="without a chunk"):
        list(client.stream(REQUEST))


def whole(message: dict, finish_reason: str) -> dict:
    """A non-streamed response carrying `message`, with real metadata and usage."""
    choice = {"index": 0, "message": message, "finish_reason": finish_reason}
    return {**REAL_META, "object": "chat.completion", "choices": [choice], "usage": REAL_USAGE}


ANSWER = "The tool returned 6912."
WHOLE_RESPONSES = {
    "parallel tool calls": whole(REAL_ASSISTANT, "tool_calls"),
    "answer": whole(
        {
            "role": "assistant",
            "content": "1234 + 5678 = **6912**",
            "refusal": None,
            "reasoning": ANSWER,
            "reasoning_details": [{"type": "reasoning.text", "text": ANSWER, "format": "unknown", "index": 0}],
        },
        "stop",
    ),
}


def pieces(text: str, rng: random.Random) -> list[str]:
    """`text` cut at random points; repeated cuts give the empty pieces providers also send."""
    cuts = sorted(rng.choices(range(len(text) + 1), k=rng.randint(0, 6)))
    bounds = [0, *cuts, len(text)]
    return [text[start:end] for start, end in itertools.pairwise(bounds)]


def as_chunks(response: dict, rng: random.Random) -> list[dict]:
    """The chunks a provider could stream for a whole `response`, in the order DeepSeek sends them
    (`.experiments/output/stream_wire.txt`): reasoning, answer, then each tool call's fragments in turn, with every
    text cut at random points and the usage on the last choice or in a chunk of its own."""
    meta = {key: value for key, value in response.items() if key not in ("object", "choices", "usage")}
    choice = response["choices"][0]
    message = choice["message"]

    def chunk(delta: dict, finish_reason=None) -> dict:
        choice = {"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": finish_reason}
        return {**meta, "object": "chat.completion.chunk", "choices": [choice]}

    chunks = []
    if message.get("reasoning"):
        [detail] = message["reasoning_details"]  # each reasoning chunk carries the same text as a detail fragment
        assert detail["text"] == message["reasoning"]
        for piece in pieces(message["reasoning"], rng):
            chunks.append(chunk({"content": "", "reasoning": piece, "reasoning_details": [{**detail, "text": piece}]}))
    for piece in pieces(message.get("content") or "", rng):
        chunks.append(chunk({"content": piece}))
    for call in message.get("tool_calls", []):
        head = {"index": call["index"], "id": call["id"], "type": call["type"]}
        function = {"name": call["function"]["name"], "arguments": ""}
        chunks.append(chunk({"content": None, "tool_calls": [{**head, "function": function}]}))
        for piece in pieces(call["function"]["arguments"], rng):
            fragment = {"index": call["index"], "function": {"arguments": piece}}
            chunks.append(chunk({"content": None, "tool_calls": [fragment]}))
    chunks.append(chunk({"content": ""}, choice["finish_reason"]))
    if rng.random() < 0.5:
        chunks[-1]["usage"] = response["usage"]
    else:
        chunks.append({**meta, "object": "chat.completion.chunk", "choices": [], "usage": response["usage"]})
    return chunks


def compared(completion: Completion) -> dict:
    """A completion's fields, with the message as it is sent back: a whole reply's `refusal: null` isn't sent, so it
    doesn't count."""
    return {**vars(completion), "message": render_message(completion.message)}


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("name", WHOLE_RESPONSES)
def test_a_streamed_reply_completes_like_the_same_reply_sent_whole(monkeypatch, name, seed):
    # Catches a stream rebuilt into a different reply than `complete` parses, wherever the provider cuts its text.
    rng = random.Random(seed)  # noqa: S311 - seeded cut points, not secrets
    response = WHOLE_RESPONSES[name]
    sent_whole = streaming_client(monkeypatch, httpx.Response(200, json=response)).complete(REQUEST)
    stream = sse(*as_chunks(response, rng), keep_alives=rng.randint(0, 2))
    deltas, streamed = split(streaming_client(monkeypatch, stream).stream(REQUEST))
    assert compared(streamed) == compared(sent_whole)
    message = response["choices"][0]["message"]
    for kind in ("reasoning", "content"):
        assert "".join(delta.text for delta in deltas if delta.kind == kind) == (message.get(kind) or "")
