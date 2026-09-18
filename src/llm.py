"""The chat-completions wire format: an OpenRouter client and tool schemas.

Everything here is plain dicts in the OpenAI chat format, so what goes over the wire is exactly
what the agent stores and shows. Findings behind the details are in
`.claude/.archived_plans/openrouter-deepseek.md` and `.experiments/openrouter_stream_roundtrip.py`.
"""

import inspect
import os
import re
import time
from typing import Any, Callable, get_type_hints

import httpx
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


SECTION = re.compile(r"^(Args|Arguments|Returns|Raises|Yields|Examples?|Notes?):\s*$")
ARG_LINE = re.compile(r"^(\w+)(?:\s*\(.*?\))?:\s*(.*)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its description and its `Args:` descriptions by name."""
    description, args = [], {}
    section, current, arg_indent = None, None, None
    for line in inspect.cleandoc(doc or "").splitlines():
        if SECTION.match(line):
            section, current = SECTION.match(line).group(1), None
        elif section is None:
            description.append(line)
        elif section in ("Args", "Arguments") and line.strip():
            indent = len(line) - len(line.lstrip())
            match = ARG_LINE.match(line.strip())
            if match and (arg_indent is None or indent <= arg_indent):
                arg_indent, current = indent, match.group(1)
                args[current] = match.group(2)
            elif current:  # a continuation line of the current argument
                args[current] += " " + line.strip()
    return "\n".join(description).strip(), args


def drop_titles(schema: Any) -> Any:
    """Remove the `title` strings pydantic adds; they only cost tokens. Properties named `title` are kept."""
    if isinstance(schema, dict):
        return {k: drop_titles(v) for k, v in schema.items() if not (k == "title" and isinstance(v, str))}
    if isinstance(schema, list):
        return [drop_titles(v) for v in schema]
    return schema


def tool_schema(name: str, fn: Callable) -> dict:
    """The OpenAI tool definition for `fn`, from its type hints and Google-style docstring.

    Arguments with defaults are optional and show their default. Pass bound methods, so `self` is not an argument.
    """
    description, arg_docs = parse_docstring(inspect.getdoc(fn))
    hints = get_type_hints(fn)
    fields = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        default = ... if param.default is param.empty else param.default
        fields[param.name] = (hints.get(param.name, Any), Field(default, description=arg_docs.get(param.name)))
    parameters = create_model(f"{name}_args", **fields).model_json_schema()
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": drop_titles(parameters)}}
