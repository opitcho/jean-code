"""The chat-completions wire format: typed messages and usage, an OpenRouter client, and tool schemas.

`parse_completion` is the only code that reads a response, and `render_request` and `render_message` the only code
that writes one. The message types mirror the OpenAI chat format field for field and keep unknown fields as pydantic
extras, so a message renders back to exactly the dict it was parsed from.
"""

import inspect
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, get_type_hints, runtime_checkable

import httpx
from griffe import Docstring, DocstringSectionKind
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, create_model

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# DeepSeek's own endpoint: unquantized, cheapest cache reads, and every request lands on the same cache.
# Pinning turns off OpenRouter's sticky routing, which only matters while DeepSeek is down.
OPENROUTER_ROUTING = {"provider": {"order": ["deepseek"], "allow_fallbacks": True, "require_parameters": True}}
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def _empty_if_null(value: object) -> object:
    """Null `content` reads as empty, as the agent has always treated it; the field stays a plain `str`."""
    return "" if value is None else value


def _arguments_text(value: object) -> object:
    """Arguments as the JSON text the tool boundary parses: null reads as empty, and an object from a provider that
    didn't encode it is encoded here."""
    if value is None:
        return ""
    if isinstance(value, dict | list):
        return json.dumps(value)
    return value


def _lenient(kind: type) -> BeforeValidator:
    """Metadata of the wrong type reads as None instead of failing, so a receipt can always be read."""
    return BeforeValidator(lambda value: value if isinstance(value, kind) and not isinstance(value, bool) else None)


class FunctionCall(BaseModel):
    """The function a tool call names, with its arguments as the model wrote them: JSON text, parsed by the tool."""

    model_config = ConfigDict(frozen=True, extra="allow")
    name: str
    arguments: Annotated[str, BeforeValidator(_arguments_text)] = ""


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
    content: Annotated[str, BeforeValidator(_empty_if_null)] = ""
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = Field(default=(), exclude_if=lambda calls: not calls)  # a plain reply has none


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
    model_config = ConfigDict(extra="allow")
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


class Usage(BaseModel):
    """The provider's usage object as sent: None means "not reported", which is not the same as 0."""

    model_config = ConfigDict(extra="allow")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    prompt_tokens_details: TokenDetails | None = None
    completion_tokens_details: TokenDetails | None = None


class Receipt(BaseModel):
    """What the provider says it served and charged: everything needed to book the call. Read from the response's
    `id`, `provider`, `model`, `created` and `usage`; the rest of the response is ignored here."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)
    generation_id: Annotated[str | None, _lenient(str)] = Field(default=None, alias="id")
    provider: Annotated[str | None, _lenient(str)] = None
    model: Annotated[str | None, _lenient(str)] = None
    created: Annotated[int | None, _lenient(int)] = None
    usage: Usage | None = None


@dataclass(frozen=True)
class Completion:
    """A parsed response: the reply, why the model stopped (as sent), and the receipt."""

    message: AssistantMessage
    finish_reason: str | None
    receipt: Receipt


@dataclass(frozen=True)
class ChatRequest:
    """One chat request. `tools` are JSON Schemas; `reasoning` is the effort ("low", "medium", "high") or None."""

    model: str
    messages: tuple[Message, ...]
    tools: tuple[dict, ...]
    reasoning: str | None


class MalformedResponse(RuntimeError):
    """The response isn't a chat completion we can use. `receipt` is what could be read before the problem, so the
    call can still be booked; None when the body wasn't a JSON object at all."""

    def __init__(self, detail: str, receipt: Receipt | None):
        super().__init__(detail)
        self.receipt = receipt


class _Choice(BaseModel):
    message: AssistantMessage
    finish_reason: Annotated[str | None, _lenient(str)] = None


class _Choices(BaseModel):
    choices: Annotated[list[_Choice], Field(min_length=1)]


def describe(e: ValidationError) -> str:
    """One `path: message` per problem, without the input, which can be the whole response."""
    return "; ".join(f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors())


def parse_receipt(data: dict) -> tuple[Receipt, str | None]:
    """The response's receipt, and why its usage object couldn't be read (that receipt then has `usage=None`).
    Never raises: the metadata fields are lenient, so only `usage` can fail."""
    try:
        return Receipt.model_validate(data), None
    except ValidationError as e:
        return Receipt.model_validate({**data, "usage": None}), describe(e)


def parse_completion(data: object) -> Completion:
    """The response as a `Completion`. The receipt is read first, so a `MalformedResponse` raised for anything after
    it still carries what's needed to book the call."""
    if not isinstance(data, dict):
        raise MalformedResponse(f"response is not a JSON object: {type(data).__name__}", receipt=None)
    receipt, usage_error = parse_receipt(data)
    if usage_error:
        raise MalformedResponse(f"malformed usage in {receipt.generation_id}: {usage_error}", receipt)
    try:
        choice = _Choices.model_validate(data).choices[0]
    except ValidationError as e:
        detail = f"malformed response {receipt.generation_id} from {receipt.provider}: {describe(e)}"
        raise MalformedResponse(detail, receipt) from e
    return Completion(choice.message, choice.finish_reason, receipt)


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
            raise MalformedResponse(f"response body is not JSON: {e}", receipt=None) from e
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


ARG_SECTIONS = (DocstringSectionKind.parameters, DocstringSectionKind.other_parameters)  # Args, Keyword Args, …
DATA_KEYWORDS = {"default", "examples", "const", "enum"}  # schema keywords whose values are data, not schemas


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its description and its argument descriptions by name.

    Arguments come from `Args:` and `Keyword Args:`; other sections (Returns, Raises, Note, …) are left out.
    """
    sections = Docstring(inspect.cleandoc(doc or ""), lineno=1).parse("google", warnings=False)
    description = [s.value for s in sections if s.kind is DocstringSectionKind.text]
    args = {p.name: " ".join(p.description.split()) for s in sections if s.kind in ARG_SECTIONS for p in s.value}
    return "\n\n".join(description).strip(), args


def drop_titles(schema: Any) -> Any:
    """Remove the `title` strings pydantic adds; they only cost tokens. Properties named `title` are kept."""
    if isinstance(schema, dict):
        return {
            k: v if k in DATA_KEYWORDS else drop_titles(v)
            for k, v in schema.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(schema, list):
        return [drop_titles(v) for v in schema]
    return schema


def tool_schema(name: str, fn: Callable) -> dict:
    """The OpenAI tool definition for `fn`, from its type hints and Google-style docstring.

    Arguments with defaults are optional and show their default. Pass bound methods, so `self` is not an argument.
    `*args` and `**kwargs` are left out; positional-only parameters are refused, since tools are called by keyword.
    """
    description, arg_docs = parse_docstring(inspect.getdoc(fn))
    hints = get_type_hints(fn, include_extras=True)  # keeps Annotated[..., Field(...)] constraints
    fields = {}
    for i, param in enumerate(inspect.signature(fn).parameters.values()):
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.kind is param.POSITIONAL_ONLY:
            raise ValueError(
                f"tool {name!r}: parameter {param.name!r} is positional-only, but tools are called by keyword"
            )
        default = ... if param.default is param.empty else param.default
        # Passing description=None would erase one given in Annotated[..., Field(description=...)].
        docs: dict[str, Any] = {"description": arg_docs[param.name]} if arg_docs.get(param.name) else {}
        # A neutral field name with the real name as alias: pydantic refuses names like `_x` and warns on ones
        # like `json`.
        fields[f"arg{i}"] = (hints.get(param.name, Any), Field(default, alias=param.name, **docs))
    parameters = create_model(f"{name}_args", **fields).model_json_schema()
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": drop_titles(parameters)},
    }


def tool(fn: Callable) -> Callable:
    """Mark `fn` as a tool: a function a model calls. Its name, parameters and docstring are what the model sees, so
    changing any of them changes the model's API. `Agent` refuses tools without the mark."""
    fn.is_tool = True  # pyright: ignore[reportFunctionMemberAccess] - functions take attributes; the stubs don't say
    return fn


def is_tool(fn: Callable) -> bool:
    """Whether `fn` was marked with `@tool`. A bound method has its function's mark."""
    return getattr(fn, "is_tool", False) is True
