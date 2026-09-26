"""The chat-completions wire format: an OpenRouter client and tool schemas.

Everything here is plain dicts in the OpenAI chat format, so what goes over the wire is exactly
what the agent stores and shows.
"""

import inspect
import os
import time
from typing import Any, Callable, get_type_hints

import httpx
from griffe import Docstring, DocstringSectionKind
from pydantic import Field, create_model

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# DeepSeek's own endpoint: unquantized, cheapest cache reads, and every request lands on the same cache.
# Pinning turns off OpenRouter's sticky routing, which only matters while DeepSeek is down.
OPENROUTER_ROUTING = {"provider": {"order": ["deepseek"], "allow_fallbacks": True, "require_parameters": True}}
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


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

    def complete(self, request: dict) -> dict:
        """Send `request` and return the whole response: id, provider, choices and usage."""
        body = {**self.extra, **request}
        for attempt in range(self.retries + 1):
            response = self.http.post(f"{self.base_url}/chat/completions", json=body)
            if response.status_code in RETRY_STATUSES and attempt < self.retries:
                time.sleep(2**attempt)
                continue
            if response.is_error:
                raise RuntimeError(f"HTTP {response.status_code} from {self.base_url}: {response.text}")
            data = response.json()
            if "error" in data:  # OpenRouter can report a provider error with status 200
                raise RuntimeError(f"model error: {data['error']}")
            return data

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
        return {k: v if k in DATA_KEYWORDS else drop_titles(v)
                for k, v in schema.items() if not (k == "title" and isinstance(v, str))}
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
            raise ValueError(f"tool {name!r}: parameter {param.name!r} is positional-only, but tools are called by keyword")
        default = ... if param.default is param.empty else param.default
        # Passing description=None would erase one given in Annotated[..., Field(description=...)].
        docs = {"description": arg_docs[param.name]} if arg_docs.get(param.name) else {}
        # A neutral field name with the real name as alias: pydantic refuses names like `_x` and warns on ones like `json`.
        fields[f"arg{i}"] = (hints.get(param.name, Any), Field(default, alias=param.name, **docs))
    parameters = create_model(f"{name}_args", **fields).model_json_schema()
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": drop_titles(parameters)}}
