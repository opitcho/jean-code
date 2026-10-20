"""The chat-completions wire format, and the OpenRouter clients that speak it.

The message types (`Message`), `Usage` and `ChatRequest` mirror the OpenAI chat format field for field, and keep
unknown fields as pydantic extras, so a message renders back to exactly the dict it was parsed from.
`parse_completion` reads a response; `render_request` and `render_message` write one. `ChatClient` sends
requests, and `StreamingClient` can also stream the reply; `ChatBackend` and `StreamingBackend` are what an agent
needs from each.
"""

import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# DeepSeek's own endpoint: unquantized, cheapest cache reads, and every request lands on the same cache.
# Pinning turns off OpenRouter's sticky routing, which only matters while DeepSeek is down.
OPENROUTER_ROUTING = {"provider": {"order": ["deepseek"], "allow_fallbacks": True, "require_parameters": True}}
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
# A connection that couldn't be opened (a failed DNS lookup, a Wi-Fi drop): nothing was sent, so resending is safe
NETWORK_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


def _null_as(empty: object) -> BeforeValidator:
    """Makes null read as the field's empty value, so the field keeps its plain type."""
    return BeforeValidator(lambda value: empty if value is None else value)


def _json_text(value: object) -> object:
    """Encodes arguments a provider sent as an object into the JSON text tools parse."""
    return json.dumps(value) if isinstance(value, dict | list) else value


class FunctionCall(BaseModel):
    """The function a tool call names, with its arguments as the model wrote them.

    Attributes:
        name: The tool's name.
        arguments: JSON text; the tool parses it.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    name: str
    arguments: Annotated[str, _null_as(""), BeforeValidator(_json_text)] = ""


class ToolCall(BaseModel):
    """One tool call in an assistant message.

    Wire keys we don't use (`index`, `type`) are kept as extras.

    Attributes:
        id: The call's id, which its result refers to.
        function: The tool and its arguments.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    id: str
    function: FunctionCall


class AssistantMessage(BaseModel):
    """A model's reply, as the provider sent it.

    Unknown fields (`reasoning_details`, `refusal`, provider extras) are kept as extras, because the provider needs
    them back.

    Attributes:
        content: The reply's text; "" if there's none.
        reasoning: The model's reasoning text, if the provider sent it.
        tool_calls: The tools it calls; empty for a plain reply.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    role: Literal["assistant"] = "assistant"
    content: Annotated[str, _null_as("")] = ""
    reasoning: str | None = None
    tool_calls: Annotated[tuple[ToolCall, ...], _null_as(())] = Field(default=(), exclude_if=lambda calls: not calls)


class UserMessage(BaseModel):
    """A message from the user, or one the agent writes in the user's place.

    Attributes:
        content: Its text.
    """

    model_config = ConfigDict(frozen=True)
    role: Literal["user"] = "user"
    content: str


class SystemMessage(BaseModel):
    """The system prompt, as a message.

    Attributes:
        content: Its text.
    """

    model_config = ConfigDict(frozen=True)
    role: Literal["system"] = "system"
    content: str


class ToolMessage(BaseModel):
    """A tool call's result.

    Attributes:
        tool_call_id: The id of the call it answers.
        content: The result's text.
    """

    model_config = ConfigDict(frozen=True)
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


Message = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")]


class TokenDetails(BaseModel):
    """A breakdown of a usage token count; None means not reported.

    Attributes:
        cached_tokens: Prompt tokens read from the cache.
        reasoning_tokens: Completion tokens spent reasoning.
    """

    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


class Usage(BaseModel):
    """The provider's usage object as sent; None means "not reported", not 0.

    Fields we don't read (`cost_details`, `is_byok`, …) are dropped: usage is never sent back.

    Attributes:
        prompt_tokens: Tokens in the prompt.
        completion_tokens: Tokens in the reply.
        cost: What the call cost, in USD.
        prompt_tokens_details: How many of the prompt tokens were cached.
        completion_tokens_details: How many of the reply tokens were reasoning.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    prompt_tokens_details: TokenDetails | None = None
    completion_tokens_details: TokenDetails | None = None


@dataclass(frozen=True)
class Completion:
    """A parsed response: everything needed to book the call.

    Attributes:
        message: The reply.
        finish_reason: Why the model stopped, as sent.
        generation_id: The response's id.
        provider: Who served it; OpenRouter's field, which other endpoints leave out.
        model: The model that answered.
        created: When it was created, in Unix seconds.
        usage: What the provider says it served and charged.
    """

    message: AssistantMessage
    finish_reason: str | None
    generation_id: str
    provider: str | None
    model: str
    created: int
    usage: Usage | None


@dataclass(frozen=True)
class TextDelta:
    """A piece of a streamed reply as it arrives.

    Attributes:
        kind: Whether it's part of the answer (`content`) or of the model's `reasoning`.
        text: The piece.
    """

    kind: Literal["content", "reasoning"]
    text: str


@dataclass(frozen=True)
class ChatRequest:
    """One chat request.

    Attributes:
        model: OpenRouter model id.
        messages: The transcript to send.
        tools: The tools' JSON Schemas.
        reasoning: The effort ("low", "medium" or "high"), or None to send no setting.
    """

    model: str
    messages: tuple[Message, ...]
    tools: tuple[dict, ...]
    reasoning: str | None


class MalformedResponse(RuntimeError):
    """The response isn't a chat completion we can use.

    Nothing in it is trusted, not even its usage.
    """


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
    """Formats `e` as one `path: message` per problem, without the input.

    The input is left out because it can be the whole response.

    Args:
        e: A pydantic validation error.
    """
    return "; ".join(f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors())


def parse_completion(data: object) -> Completion:
    """Parses a chat-completions response.

    Args:
        data: The response's JSON.

    Returns:
        The response as a `Completion`.

    Raises:
        MalformedResponse: The response doesn't fit; the message names every field that doesn't.
    """
    if not isinstance(data, dict):
        raise MalformedResponse(f"response is not a JSON object: {type(data).__name__}")
    try:
        r = _Response.model_validate(data)
    except ValidationError as e:
        raise MalformedResponse(f"malformed response {data.get('id')!r}: {describe(e)}") from e
    choice = r.choices[0]
    return Completion(choice.message, choice.finish_reason, r.id, r.provider, r.model, r.created, r.usage)


class _FunctionDelta(BaseModel):
    """The `function` of a tool-call fragment.

    The call's first fragment brings the `name`, and each fragment the next piece of the `arguments`.
    """

    name: str | None = None
    arguments: Annotated[str, _null_as("")] = ""


class _ToolCallDelta(BaseModel):
    """A fragment of one tool call.

    A call's fragments share its `index` and arrive in order: the first brings the call's `id`, `type` and `name`,
    the later ones the rest of its `arguments`.
    """

    index: int
    id: str | None = None
    type: str | None = None
    function: _FunctionDelta = Field(default_factory=_FunctionDelta)


class _Delta(BaseModel):
    """What one chunk adds to the reply; null reads as nothing added.

    `reasoning_details` fragments stay plain dicts, kept as sent: their fields differ by provider (`text`,
    `summary`, encrypted `data`).
    """

    content: Annotated[str, _null_as("")] = ""
    reasoning: Annotated[str, _null_as("")] = ""
    tool_calls: Annotated[list[_ToolCallDelta], _null_as([])] = []
    reasoning_details: Annotated[list[dict], _null_as([])] = []


class _ChunkChoice(BaseModel):
    delta: _Delta = Field(default_factory=_Delta)
    finish_reason: str | None = None


class _Chunk(BaseModel):
    """One streamed chunk.

    `id`, `provider`, `model` and `created` repeat on every chunk; `usage` comes on the last.
    """

    id: str
    provider: str | None = None
    model: str
    created: int
    usage: Usage | None = None
    choices: list[_ChunkChoice] = []


def _parse_chunk(line: str) -> _Chunk:
    """Parses one `data:` line of a stream.

    Raises:
        RuntimeError: The provider reported an error mid-stream.
        MalformedResponse: The line isn't a chunk.
    """
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
    """Joins a streamed reply's tool calls from their fragments.

    A new `index` starts a new call, and a fragment with the last call's `index` adds to its `arguments`.
    """
    calls: list[dict] = []
    for delta in deltas:
        for fragment in delta.tool_calls:
            if calls and calls[-1]["index"] == fragment.index:
                calls[-1]["function"]["arguments"] += fragment.function.arguments
            else:
                calls.append(fragment.model_dump(exclude_none=True))
    return calls


def _join_reasoning_details(deltas: list[_Delta]) -> list[dict]:
    """Joins a streamed reply's `reasoning_details` from their fragments.

    A fragment with the same `index` as the last entry adds to its `text` (DeepSeek streams its reasoning this
    way); any other fragment is a new entry, kept as sent.
    """
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
    """Rebuilds the completion a streamed reply stands for from its chunks.

    Raises:
        MalformedResponse: There's no chunk, or the reply isn't a valid message (a tool call without an `id` or a
            `name`).
    """
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
    """Returns the wire dict for `message`: exactly what it was parsed from.

    `None`s are left out.

    Args:
        message: The message to render.
    """
    return message.model_dump(mode="json", exclude_none=True)


def render_request(request: ChatRequest) -> dict:
    """Returns the wire body for `request`.

    The body holds the model and the rendered messages, plus the reasoning effort and the tools when set.

    Args:
        request: The request to render.
    """
    body: dict = {"model": request.model, "messages": [render_message(m) for m in request.messages]}
    if request.reasoning:
        body["reasoning"] = {"effort": request.reasoning}
    if request.tools:
        body["tools"] = list(request.tools)
    return body


class ChatBackend(Protocol):
    """What an agent needs from a model endpoint: one chat completion per call.

    `ChatClient` is the real one.
    """

    def complete(self, request: ChatRequest) -> Completion:
        """Returns the completion for `request`.

        Args:
            request: The request to send.
        """
        ...


@runtime_checkable
class StreamingBackend(ChatBackend, Protocol):
    """A `ChatBackend` that can also stream a reply.

    `StreamingClient` is the real one.
    """

    def stream(self, request: ChatRequest, stop: Callable[[], bool] | None = None) -> Iterator[TextDelta | Completion]:
        """Yields the reply to `request` as it arrives, as `StreamingClient.stream` does.

        Args:
            request: The request to send.
            stop: Asked as the reply arrives; True cuts it off.

        Yields:
            The reply's pieces, then its `Completion`, unless `stop` cut it off before its first chunk.
        """
        ...


class ChatClient:
    """A client for an OpenAI-compatible chat-completions endpoint.

    Failed requests are retried with backoff. The arguments it keeps are attributes of the same name.

    Args:
        base_url: The endpoint's base URL, e.g. "https://openrouter.ai/api/v1".
        api_key: Sent as the bearer token.
        extra: Request fields sent with every call, such as provider routing.
        retries: How many times a request is retried on a retryable status.

    Attributes:
        http: The HTTP client the requests go through.
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
        """Sends `request` and returns the parsed completion.

        Args:
            request: The request to send.

        Raises:
            MalformedResponse: The response can't be used.
            RuntimeError: The endpoint returned an HTTP error, or reported a model error.
        """
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
        """POSTs a chat completion and returns the response with its body unread.

        The caller reads or closes it. Retries the statuses in `RETRY_STATUSES` with backoff.

        Raises:
            RuntimeError: The final response is an HTTP error.
        """
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
        """Returns the API key's spend and limit, from OpenRouter's /key endpoint.

        Raises:
            httpx.HTTPStatusError: The endpoint returned an error.
        """
        response = self.http.get(f"{self.base_url}/key")
        response.raise_for_status()
        return response.json()["data"]


class StreamingClient(ChatClient):
    """A `ChatClient` that can also stream a reply."""

    def stream(self, request: ChatRequest, stop: Callable[[], bool] | None = None) -> Iterator[TextDelta | Completion]:
        """Sends `request` with streaming on, and yields the reply as it arrives.

        Nothing is sent until the first item is asked for, and stopping early closes the connection.

        Args:
            request: The request to send.
            stop: Asked before each line that arrives, keep-alives included; once it returns True, the connection is
                closed and the reply so far is yielded as it stands, without a `finish_reason` or usage. None to
                read the whole reply.

        Yields:
            Each piece of reasoning or answer as a `TextDelta`, then the parsed `Completion`; no `Completion` if
            `stop` cut the reply off before its first chunk.

        Raises:
            MalformedResponse: A chunk, or the reply joined from them, can't be used.
            RuntimeError: The endpoint returned an HTTP error, or reported a model error mid-stream.
        """
        chunks: list[_Chunk] = []
        stopped = False
        response = self._send({**self.extra, **render_request(request), "stream": True})
        try:
            for line in response.iter_lines():
                if stop is not None and stop():
                    stopped = True
                    break
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
        if chunks or not stopped:  # a stop before the first chunk leaves nothing to yield
            yield _join_chunks(chunks)


def openrouter(api_key: str | None = None) -> StreamingClient:
    """Returns a client for OpenRouter, pinned to DeepSeek's endpoint, that can stream.

    Args:
        api_key: The OpenRouter key; `OPENROUTER_API_KEY` by default.

    Raises:
        RuntimeError: No key was given, and `OPENROUTER_API_KEY` is not set.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; run with `uv run --env-file .env ...`")
    return StreamingClient(OPENROUTER_URL, api_key, extra=OPENROUTER_ROUTING)
