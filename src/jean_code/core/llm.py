"""The chat-completions wire format: typed messages and usage, and OpenRouter clients, plain and streaming.

`parse_completion` reads a response and `_parse_chunk` a streamed chunk, and `render_request` and `render_message`
are the only code that writes one; a streamed reply is rebuilt from its chunks by `_join_chunks`. The message types
mirror the OpenAI chat format field for field and keep unknown fields as pydantic extras, so a message renders back
to exactly the dict it was parsed from.
"""

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# DeepSeek's own endpoint: unquantized, cheapest cache reads, and every request lands on the same cache.
# Pinning turns off OpenRouter's sticky routing, which only matters while DeepSeek is down.
OPENROUTER_ROUTING = {"provider": {"order": ["deepseek"], "allow_fallbacks": True, "require_parameters": True}}
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def _null_as(empty: object) -> BeforeValidator:
    """Null reads as the field's empty value, as the agent has always treated it; the field keeps its plain type."""
    return BeforeValidator(lambda value: empty if value is None else value)


def _json_text(value: object) -> object:
    """Arguments as the JSON text the tool boundary parses: an object from a provider that didn't encode it is
    encoded here."""
    return json.dumps(value) if isinstance(value, dict | list) else value


class FunctionCall(BaseModel):
    """The function a tool call names, with its arguments as the model wrote them: JSON text, parsed by the tool."""

    model_config = ConfigDict(frozen=True, extra="allow")
    name: str
    arguments: Annotated[str, _null_as(""), BeforeValidator(_json_text)] = ""


class ToolCall(BaseModel):
    """One tool call in an assistant message. Wire keys we don't use (`index`, `type`) are kept as extras."""

    model_config = ConfigDict(frozen=True, extra="allow")
    id: str
    function: FunctionCall


class AssistantMessage(BaseModel):
    """A model's reply, as the provider sent it. Unknown fields (`reasoning_details`, `refusal`, provider extras) are
    kept as extras, because the provider needs them back."""

    model_config = ConfigDict(frozen=True, extra="allow")
    role: Literal["assistant"] = "assistant"
    content: Annotated[str, _null_as("")] = ""
    reasoning: str | None = None
    tool_calls: Annotated[tuple[ToolCall, ...], _null_as(())] = Field(
        default=(), exclude_if=lambda calls: not calls
    )  # a plain reply has none


class UserMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["user"] = "user"
    content: str


class SystemMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["system"] = "system"
    content: str


class ToolMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


Message = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")]


class TokenDetails(BaseModel):
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


class Usage(BaseModel):
    """The provider's usage object as sent: None means "not reported", which is not the same as 0. Fields we don't
    read (`cost_details`, `is_byok`, …) are dropped: usage is never sent back."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    prompt_tokens_details: TokenDetails | None = None
    completion_tokens_details: TokenDetails | None = None


@dataclass(frozen=True)
class Completion:
    """A parsed response: the reply, why the model stopped (as sent), and what the provider says it served and
    charged, which is everything needed to book the call."""

    message: AssistantMessage
    finish_reason: str | None
    generation_id: str
    provider: str | None  # OpenRouter's field; other endpoints leave it out
    model: str
    created: int  # Unix seconds
    usage: Usage | None


@dataclass(frozen=True)
class TextDelta:
    """A piece of a streamed reply as it arrives: part of the answer (`content`) or of the model's `reasoning`."""

    kind: Literal["content", "reasoning"]
    text: str


@dataclass(frozen=True)
class ChatRequest:
    """One chat request. `tools` are JSON Schemas; `reasoning` is the effort ("low", "medium", "high") or None."""

    model: str
    messages: tuple[Message, ...]
    tools: tuple[dict, ...]
    reasoning: str | None


class MalformedResponse(RuntimeError):
    """The response isn't a chat completion we can use. The call isn't booked: nothing in it is trusted."""


class _Choice(BaseModel):
    message: AssistantMessage
    finish_reason: str | None = None


class _Response(BaseModel):
    """The parts of a response we read; the rest is ignored."""

    id: str
    provider: str | None = None
    model: str
    created: int
    usage: Usage | None = None
    choices: Annotated[list[_Choice], Field(min_length=1)]


def describe(e: ValidationError) -> str:
    """One `path: message` per problem, without the input, which can be the whole response."""
    return "; ".join(f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors())


def parse_completion(data: object) -> Completion:
    """The response as a `Completion`. Raises `MalformedResponse` naming every field that doesn't fit."""
    if not isinstance(data, dict):
        raise MalformedResponse(f"response is not a JSON object: {type(data).__name__}")
    try:
        r = _Response.model_validate(data)
    except ValidationError as e:
        raise MalformedResponse(f"malformed response {data.get('id')!r}: {describe(e)}") from e
    choice = r.choices[0]
    return Completion(choice.message, choice.finish_reason, r.id, r.provider, r.model, r.created, r.usage)


class _FunctionDelta(BaseModel):
    """The `function` of a tool-call fragment: the call's first fragment brings the `name`, and each fragment the
    next piece of the `arguments`."""

    name: str | None = None
    arguments: Annotated[str, _null_as("")] = ""


class _ToolCallDelta(BaseModel):
    """A fragment of one tool call. A call's fragments share its `index` and arrive in order: the first brings the
    call's `id`, `type` and `name`, the later ones the rest of its `arguments`."""

    index: int
    id: str | None = None
    type: str | None = None
    function: _FunctionDelta = Field(default_factory=_FunctionDelta)


class _Delta(BaseModel):
    """What one chunk adds to the reply; null reads as nothing added. `reasoning_details` fragments stay plain dicts,
    kept as sent: their fields differ by provider (`text`, `summary`, encrypted `data`)."""

    content: Annotated[str, _null_as("")] = ""
    reasoning: Annotated[str, _null_as("")] = ""
    tool_calls: Annotated[list[_ToolCallDelta], _null_as([])] = []
    reasoning_details: Annotated[list[dict], _null_as([])] = []


class _ChunkChoice(BaseModel):
    delta: _Delta = Field(default_factory=_Delta)
    finish_reason: str | None = None


class _Chunk(BaseModel):
    """One streamed chunk. `id`, `provider`, `model` and `created` repeat on every chunk; `usage` comes on the last."""

    id: str
    provider: str | None = None
    model: str
    created: int
    usage: Usage | None = None
    choices: list[_ChunkChoice] = []


def _parse_chunk(line: str) -> _Chunk:
    """One `data:` line of a stream as a chunk. Raises `RuntimeError` for a provider error reported mid-stream, and
    `MalformedResponse` for a line that isn't a chunk."""
    try:
        data = json.loads(line.removeprefix("data: "))
    except ValueError as e:
        raise MalformedResponse(f"stream chunk is not JSON: {e}") from e
    if not isinstance(data, dict):
        raise MalformedResponse(f"stream chunk is not a JSON object: {line}")
    if "error" in data:  # OpenRouter reports a provider error mid-stream as a chunk
        raise RuntimeError(f"model error: {data['error']}")
    try:
        return _Chunk.model_validate(data)
    except ValidationError as e:
        raise MalformedResponse(f"malformed stream chunk {data.get('id')!r}: {describe(e)}") from e


def _join_tool_calls(deltas: list[_Delta]) -> list[dict]:
    """The tool calls of a streamed reply, each joined from its fragments: a new `index` starts a new call, and a
    fragment with the last call's `index` adds to its `arguments`."""
    calls: list[dict] = []
    for delta in deltas:
        for fragment in delta.tool_calls:
            if calls and calls[-1]["index"] == fragment.index:
                calls[-1]["function"]["arguments"] += fragment.function.arguments
            else:
                calls.append(fragment.model_dump(exclude_none=True))
    return calls


def _join_reasoning_details(deltas: list[_Delta]) -> list[dict]:
    """The `reasoning_details` of a streamed reply. A fragment with the same `index` as the last entry adds to its
    `text` (DeepSeek streams its reasoning this way); any other fragment is a new entry, kept as sent."""
    details: list[dict] = []
    for delta in deltas:
        for fragment in delta.reasoning_details:
            index = fragment.get("index")
            if details and index is not None and details[-1].get("index") == index:
                details[-1]["text"] = details[-1].get("text", "") + fragment.get("text", "")
            else:
                details.append(dict(fragment))
    return details


def _join_chunks(chunks: list[_Chunk]) -> Completion:
    """The completion a streamed reply stands for, rebuilt from its chunks. Raises `MalformedResponse` for a stream
    with no chunk, or a reply that isn't a valid message (a tool call without an `id` or a `name`)."""
    if not chunks:
        raise MalformedResponse("stream ended without a chunk")
    deltas: list[_Delta] = []
    finish_reason = None
    usage = None
    for chunk in chunks:
        if chunk.usage is not None:
            usage = chunk.usage
        for choice in chunk.choices:
            deltas.append(choice.delta)
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    message: dict = {
        "content": "".join(delta.content for delta in deltas),
        "reasoning": "".join(delta.reasoning for delta in deltas) or None,
        "tool_calls": _join_tool_calls(deltas),
    }
    reasoning_details = _join_reasoning_details(deltas)
    if reasoning_details:
        message["reasoning_details"] = reasoning_details
    last = chunks[-1]
    try:
        reply = AssistantMessage.model_validate(message)
    except ValidationError as e:
        raise MalformedResponse(f"malformed streamed reply {last.id!r}: {describe(e)}") from e
    return Completion(reply, finish_reason, last.id, last.provider, last.model, last.created, usage)


def render_message(message: Message) -> dict:
    """The wire dict for a message: exactly what it was parsed from, with `None`s left out."""
    return message.model_dump(mode="json", exclude_none=True)


def render_request(request: ChatRequest) -> dict:
    """The request body: the model, the rendered messages, and the reasoning effort and tools when set."""
    body: dict = {"model": request.model, "messages": [render_message(m) for m in request.messages]}
    if request.reasoning:
        body["reasoning"] = {"effort": request.reasoning}
    if request.tools:
        body["tools"] = list(request.tools)
    return body


class ChatBackend(Protocol):
    """What an agent needs from a model endpoint: one chat completion per call. `ChatClient` is the real one;
    tests pass scripted ones."""

    def complete(self, request: ChatRequest) -> Completion: ...


class ChatClient:
    """Sends chat completions to an OpenAI-compatible endpoint and returns the parsed response.

    `extra` holds request fields sent with every call (such as provider routing).
    Failed requests are retried with backoff.
    """

    def __init__(self, base_url: str, api_key: str, extra: dict | None = None, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.extra = extra or {}
        self.retries = retries
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(600, connect=10),
        )

    def complete(self, request: ChatRequest) -> Completion:
        """Send `request` and return the parsed completion. Raises `MalformedResponse` for a response it can't use."""
        response = self._send({**self.extra, **render_request(request)})
        response.read()
        try:
            data = response.json()
        except ValueError as e:
            raise MalformedResponse(f"response body is not JSON: {e}") from e
        if isinstance(data, dict) and "error" in data:  # OpenRouter can report a provider error with status 200
            raise RuntimeError(f"model error: {data['error']}")
        return parse_completion(data)

    def _send(self, body: dict) -> httpx.Response:
        """POST a chat completion and return the response with its body unread: the caller reads or closes it.
        Retries the statuses in RETRY_STATUSES with backoff; raises on any other error."""
        request = self.http.build_request("POST", f"{self.base_url}/chat/completions", json=body)
        attempt = 0
        while True:
            response = self.http.send(request, stream=True)
            if response.status_code not in RETRY_STATUSES or attempt >= self.retries:
                break
            response.close()
            time.sleep(2**attempt)
            attempt += 1
        if response.is_error:
            response.read()
            raise RuntimeError(f"HTTP {response.status_code} from {self.base_url}: {response.text}")
        return response

    def credits(self) -> dict:
        """The API key's spend and limit, from OpenRouter's /key endpoint."""
        response = self.http.get(f"{self.base_url}/key")
        response.raise_for_status()
        return response.json()["data"]


class StreamingClient(ChatClient):
    """A `ChatClient` that can also stream a reply; `complete` and `credits` are unchanged."""

    def stream(self, request: ChatRequest) -> Iterator[TextDelta | Completion]:
        """Send `request` with streaming on: yield each piece of reasoning and answer as it arrives, then the parsed
        `Completion`. Nothing is sent until the first item is asked for, and stopping early closes the connection.
        Raises like `complete`."""
        chunks: list[_Chunk] = []
        response = self._send({**self.extra, **render_request(request), "stream": True})
        try:
            for line in response.iter_lines():
                # Blank separators, ": OPENROUTER PROCESSING" keep-alives and the end marker carry no chunk
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = _parse_chunk(line)
                chunks.append(chunk)
                for choice in chunk.choices:
                    if choice.delta.reasoning:
                        yield TextDelta("reasoning", choice.delta.reasoning)
                    if choice.delta.content:
                        yield TextDelta("content", choice.delta.content)
        finally:
            response.close()
        yield _join_chunks(chunks)


def openrouter(api_key: str | None = None) -> ChatClient:
    """A client for OpenRouter, pinned to DeepSeek's endpoint. Reads OPENROUTER_API_KEY by default."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; run with `uv run --env-file .env ...`")
    return ChatClient(OPENROUTER_URL, api_key, extra=OPENROUTER_ROUTING)
