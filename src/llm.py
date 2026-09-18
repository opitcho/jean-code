"""The chat-completions wire format: a streaming OpenRouter client, delta merging and tool schemas.

Everything here is plain dicts in the OpenAI chat format, so what goes over the wire is exactly
what the agent stores and shows. Findings behind the details are in
`.claude/.archived_plans/openrouter-deepseek.md` and `.experiments/openrouter_stream_roundtrip.py`.
"""

import inspect
import json
import os
import re
import time
from typing import Any, Callable, Iterator, get_type_hints

import httpx
from pydantic import Field, create_model

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# DeepSeek's own endpoint: unquantized, cheapest cache reads, and every request lands on the same cache.
# Pinning turns off OpenRouter's sticky routing, which only matters while DeepSeek is down.
OPENROUTER_ROUTING = {"provider": {"order": ["deepseek"], "allow_fallbacks": True, "require_parameters": True}}
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


class ChatClient:
    """Streams chat completions from an OpenAI-compatible endpoint as parsed chunks.

    `extra` holds request fields sent with every call (such as provider routing).
    Failed requests are retried with backoff only before the first chunk arrives: a stream
    that breaks halfway raises, since its partial reply can't be resumed.
    """

    def __init__(self, base_url: str, api_key: str, extra: dict | None = None, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.extra = extra or {}
        self.retries = retries
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(600, connect=10),
        )

    def stream(self, request: dict) -> Iterator[dict]:
        """Send `request` with stream=True and yield each chunk, skipping keep-alive comments and [DONE]."""
        body = {**self.extra, **request, "stream": True}
        for attempt in range(self.retries + 1):
            with self.http.stream("POST", f"{self.base_url}/chat/completions", json=body) as response:
                if response.status_code in RETRY_STATUSES and attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                if response.is_error:
                    response.read()
                    raise RuntimeError(f"HTTP {response.status_code} from {self.base_url}: {response.text}")
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue  # blank lines and ": OPENROUTER PROCESSING" keep-alives
                    chunk = json.loads(line[6:])
                    if "error" in chunk:  # an error after the stream started comes as a chunk
                        raise RuntimeError(f"stream error: {chunk['error']}")
                    yield chunk
                return

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


def merge_delta(reply: dict, delta: dict) -> None:
    """Fold one streamed delta into the assistant reply being built.

    Text is appended. `reasoning_details` arrive as one fragment per chunk sharing an `index`, and are
    merged into one block per index: the model must get back exactly the blocks it produced.
    Tool calls arrive by `index`, with the id and name first and the arguments split across chunks.
    """
    for key in ("content", "reasoning"):
        if delta.get(key):
            reply[key] = (reply.get(key) or "") + delta[key]

    for fragment in delta.get("reasoning_details") or []:
        details = reply.setdefault("reasoning_details", [])
        block = next((d for d in details if d.get("index") == fragment.get("index")
                      and d.get("type") == fragment.get("type")), None)
        if block is None:
            details.append(dict(fragment))
            continue
        for key, value in fragment.items():
            if key in ("text", "summary", "data") and isinstance(value, str):
                block[key] = block.get(key, "") + value
            elif value is not None:
                block[key] = value

    for piece in delta.get("tool_calls") or []:
        calls = reply.setdefault("tool_calls", [])
        while len(calls) <= piece["index"]:
            calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        call = calls[piece["index"]]
        call["id"] = piece.get("id") or call["id"]
        function = piece.get("function") or {}
        call["function"]["name"] += function.get("name") or ""
        call["function"]["arguments"] += function.get("arguments") or ""


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
