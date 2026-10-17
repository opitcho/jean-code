"""Tool schemas, built from Python functions.

What a model sees of a tool is its name, its typed parameters and its Google-style docstring. `@tool` marks a
function as a tool; `tool_schema` builds the OpenAI tool definition for it.
"""

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from griffe import Docstring, DocstringSectionKind
from pydantic import Field, create_model

ARG_SECTIONS = (DocstringSectionKind.parameters, DocstringSectionKind.other_parameters)  # Args, Keyword Args, …
DATA_KEYWORDS = {"default", "examples", "const", "enum"}  # schema keywords whose values are data, not schemas


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Splits a Google-style docstring into its description and its argument docs.

    Arguments come from `Args:` and `Keyword Args:`; other sections (Returns, Raises, Note, …) are left out.

    Args:
        doc: The docstring; None reads as empty.

    Returns:
        The description, and each argument's description by name.
    """
    sections = Docstring(inspect.cleandoc(doc or ""), lineno=1).parse("google", warnings=False)
    description = [s.value for s in sections if s.kind is DocstringSectionKind.text]
    args = {p.name: " ".join(p.description.split()) for s in sections if s.kind in ARG_SECTIONS for p in s.value}
    return "\n\n".join(description).strip(), args


def drop_titles(schema: Any) -> Any:
    """Removes the `title` strings pydantic adds to `schema`; they only cost tokens.

    Properties named `title` are kept.

    Args:
        schema: A JSON Schema, or any part of one.

    Returns:
        The schema without them.
    """
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
    """Returns the OpenAI tool definition for `fn`, from its type hints and docstring.

    Arguments with defaults are optional and show their default. `*args` and `**kwargs` are left out.

    Args:
        name: The name the model calls the tool by.
        fn: The function; pass a bound method, so `self` is not an argument.

    Raises:
        ValueError: A parameter is positional-only, but tools are called by keyword.
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
    """Marks `fn` as a tool: a function a model calls.

    Its name, parameters and docstring are what the model sees, so changing any of them changes the model's API.

    Args:
        fn: The function.

    Returns:
        `fn` itself.
    """
    fn.is_tool = True  # pyright: ignore[reportFunctionMemberAccess] - functions take attributes; the stubs don't say
    return fn


def is_tool(fn: Callable) -> bool:
    """Returns whether `fn` was marked with `@tool`.

    A bound method has its function's mark.

    Args:
        fn: Any callable.
    """
    return getattr(fn, "is_tool", False) is True
