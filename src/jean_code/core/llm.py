"""The chat-completions wire format: typed messages and usage, and an OpenRouter client.

`parse_completion` is the only code that reads a response, and `render_request` and `render_message` the only code
that writes one. The message types mirror the OpenAI chat format field for field and keep unknown fields as pydantic
extras, so a message renders back to exactly the dict it was parsed from.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, runtime_checkable

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


@runtime_checkable
class HasCredits(Protocol):
    """An endpoint that reports the API key's spend and limit, like OpenRouter's."""

    def credits(self) -> dict: ...


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
        response = self._post({**self.extra, **render_request(request)})
        try:
            data = response.json()
        except ValueError as e:
            raise MalformedResponse(f"response body is not JSON: {e}") from e
        if isinstance(data, dict) and "error" in data:  # OpenRouter can report a provider error with status 200
            raise RuntimeError(f"model error: {data['error']}")
        return parse_completion(data)

    def _post(self, body: dict) -> httpx.Response:
        """POST a chat completion, retrying the statuses in RETRY_STATUSES with backoff. Raises on any other error."""
        attempt = 0
        while True:
            response = self.http.post(f"{self.base_url}/chat/completions", json=body)
            if response.status_code in RETRY_STATUSES and attempt < self.retries:
                time.sleep(2**attempt)
                attempt += 1
                continue
            if response.is_error:
                raise RuntimeError(f"HTTP {response.status_code} from {self.base_url}: {response.text}")
            return response

    def credits(self) -> dict:
        """The API key's spend and limit, from OpenRouter's /key endpoint."""
        response = self.http.get(f"{self.base_url}/key")
        response.raise_for_status()
        return response.json()["data"]


def openrouter(api_key: str | None = None) -> ChatClient:
    """A client for OpenRouter, pinned to DeepSeek's endpoint. Reads OPENROUTER_API_KEY by default."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; run with `uv run --env-file .env ...`")
    return ChatClient(OPENROUTER_URL, api_key, extra=OPENROUTER_ROUTING)
