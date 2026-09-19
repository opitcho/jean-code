"""Tests for the docstring parser and tool-schema builder in `llm.py`.

Expectations follow the Google docstring style as Sphinx napoleon reads it, and the JSON schema
an OpenAI-compatible endpoint needs to call the tool correctly.
"""

import copy
import functools
from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict

import pytest
from pydantic import BaseModel, Field

from llm import drop_titles, parse_docstring, tool_schema


# ---------------------------------------------------------------- parse_docstring


def test_none_and_empty_docstrings():
    assert parse_docstring(None) == ("", {})
    assert parse_docstring("") == ("", {})
    assert parse_docstring("   \n  ") == ("", {})


def test_description_only_keeps_paragraphs():
    doc = """Summary line.

    Longer explanation that
    spans lines.
    """
    assert parse_docstring(doc) == ("Summary line.\n\nLonger explanation that\nspans lines.", {})


def test_simple_args():
    doc = """Read a file.

    Args:
        path: The file to read.
        limit: How many lines.
    """
    assert parse_docstring(doc) == ("Read a file.", {"path": "The file to read.", "limit": "How many lines."})


@pytest.mark.parametrize("header", ["Args", "Arguments", "Parameters"])
def test_args_section_aliases(header):
    doc = f"""Do it.

    {header}:
        x: The x.
    """
    assert parse_docstring(doc)[1] == {"x": "The x."}


def test_typed_args():
    doc = """Do it.

    Args:
        a (int): Plain.
        b (dict[str, list[int]]): Nested generics.
        c (Callable[[int, str], bool] | None, optional): Callable with a nested list.
    """
    assert parse_docstring(doc)[1] == {
        "a": "Plain.",
        "b": "Nested generics.",
        "c": "Callable with a nested list.",
    }


def test_colons_inside_descriptions():
    doc = """Fetch.

    Args:
        url: The URL, e.g. https://example.com:8080/path.
        mode: One of: fast, slow.
    """
    assert parse_docstring(doc)[1] == {
        "url": "The URL, e.g. https://example.com:8080/path.",
        "mode": "One of: fast, slow.",
    }


def test_multiline_arg_descriptions():
    doc = """Search.

    Args:
        query: The text to search for. It can
            span several lines, and each continuation
            is joined with a space.
        limit: Max results.
    """
    assert parse_docstring(doc)[1] == {
        "query": "The text to search for. It can span several lines, and each continuation is joined with a space.",
        "limit": "Max results.",
    }


def test_continuation_that_looks_like_an_arg():
    doc = """Run.

    Args:
        command: The shell command.
            Note: it runs in the sandbox.
            timeout: is not an argument here, just text.
        cwd: Working directory.
    """
    assert parse_docstring(doc)[1] == {
        "command": "The shell command. Note: it runs in the sandbox. timeout: is not an argument here, just text.",
        "cwd": "Working directory.",
    }


def test_blank_line_inside_an_arg_description():
    doc = """Run.

    Args:
        a: First paragraph.

            Second paragraph.
        b: Next.
    """
    assert parse_docstring(doc)[1] == {"a": "First paragraph. Second paragraph.", "b": "Next."}


def test_description_starting_on_the_next_line():
    doc = """Run.

    Args:
        path:
            The file to read.
        mode (str):
            How to open it.
    """
    assert parse_docstring(doc)[1] == {"path": "The file to read.", "mode": "How to open it."}


def test_arg_with_empty_description():
    doc = """Run.

    Args:
        flag:
        other: Something.
    """
    assert parse_docstring(doc)[1] == {"flag": "", "other": "Something."}


@pytest.mark.parametrize("section", ["Returns", "Raises", "Example", "Warning", "See Also"])
def test_sections_after_args_do_not_leak_in(section):
    doc = f"""Run.

    Args:
        x: The x.

    {section}:
        y: not an argument.
    """
    assert parse_docstring(doc) == ("Run.", {"x": "The x."})


def test_sections_before_args():
    doc = """Run.

    Note:
        Something to note.

    Args:
        x: The x.
    """
    assert parse_docstring(doc) == ("Run.", {"x": "The x."})


def test_star_args_and_kwargs_keep_their_stars():
    doc = """Run.

    Args:
        x: The x.
        *args: Extra positionals.
        **kwargs: Extra options.
    """
    assert parse_docstring(doc)[1] == {
        "x": "The x.",
        "*args": "Extra positionals.",
        "**kwargs": "Extra options.",
    }


def test_keyword_args_section():
    doc = """Run.

    Args:
        x: The x.

    Keyword Args:
        verbose: Print more.
    """
    assert parse_docstring(doc)[1] == {"x": "The x.", "verbose": "Print more."}


def test_description_line_that_looks_like_a_section_inline():
    doc = """Returns: the thing, described inline on one line.

    Args:
        x: The x.
    """
    assert parse_docstring(doc) == ("Returns: the thing, described inline on one line.", {"x": "The x."})


def test_description_with_indented_block():
    doc = """Run a script.

    For example:

        run("ls -la")

    Args:
        script: The script.
    """
    description, args = parse_docstring(doc)
    assert description == 'Run a script.\n\nFor example:\n\n    run("ls -la")'
    assert args == {"script": "The script."}


# ---------------------------------------------------------------- drop_titles


def test_drop_titles_keeps_property_named_title():
    schema = {"title": "M", "properties": {"title": {"title": "Title", "type": "string"}}}
    assert drop_titles(schema) == {"properties": {"title": {"type": "string"}}}


def test_drop_titles_keeps_defaults_and_examples():
    schema = {
        "properties": {
            "meta": {"type": "object", "default": {"title": "Untitled"}},
            "tag": {"type": "string", "examples": [{"title": "x"}]},
        }
    }
    expected = copy.deepcopy(schema)
    assert drop_titles(schema) == expected


# ---------------------------------------------------------------- tool_schema


def params(fn, name="tool"):
    return tool_schema(name, fn)["function"]["parameters"]


def types(prop):
    """The JSON types a property accepts, flattening a top-level anyOf."""
    return {s.get("type") for s in prop.get("anyOf", [prop])}


def test_basic_function():
    def read(path: str, limit: int = 100) -> str:
        """Read a file.

        Args:
            path: The file to read.
            limit: Max lines.

        Returns:
            The contents.
        """

    assert tool_schema("read", read) == {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file to read."},
                    "limit": {"type": "integer", "default": 100, "description": "Max lines."},
                },
                "required": ["path"],
            },
        },
    }


def test_no_docstring():
    def f(x: int): ...

    schema = tool_schema("f", f)
    assert schema["function"]["description"] == ""
    assert schema["function"]["parameters"]["properties"] == {"x": {"type": "integer"}}


def test_no_parameters():
    def f():
        """Nothing."""

    assert params(f) == {"type": "object", "properties": {}}


def test_bound_method_drops_self_and_star_params():
    class Tools:
        def run(self, cmd: str, *args, timeout: float = 5.0, **kwargs):
            """Run.

            Args:
                cmd: The command.
                timeout: Seconds.
            """

    p = params(Tools().run)
    assert set(p["properties"]) == {"cmd", "timeout"}
    assert p["required"] == ["cmd"]
    assert p["properties"]["timeout"] == {"type": "number", "default": 5.0, "description": "Seconds."}


def test_optional_and_union_types():
    def f(a: Optional[int] = None, b: int | str = 1, c: list[str] | None = None): ...

    p = params(f)
    assert "required" not in p
    props = p["properties"]
    assert types(props["a"]) == {"integer", "null"}
    assert props["a"]["default"] is None
    assert types(props["b"]) == {"integer", "string"}
    assert props["b"]["default"] == 1
    assert types(props["c"]) == {"array", "null"}
    assert props["c"]["default"] is None


def test_nested_generics():
    def f(a: dict[str, list[int]], b: list[tuple[int, str]], c: dict[str, dict[str, float]]): ...

    p = params(f)["properties"]
    assert p["a"] == {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "integer"}}}
    # How a tuple is encoded (prefixItems or not) is left open: endpoint support varies.
    assert p["b"]["type"] == "array"
    assert p["b"]["items"]["type"] == "array"
    assert p["c"] == {"type": "object", "additionalProperties": {"type": "object", "additionalProperties": {"type": "number"}}}


def test_literal_and_enum():
    class Color(Enum):
        RED = "red"
        BLUE = "blue"

    def f(mode: Literal["fast", "slow"], color: Color = Color.RED): ...

    p = params(f)
    assert p["properties"]["mode"] == {"enum": ["fast", "slow"], "type": "string"}
    assert p["properties"]["color"] == {"$ref": "#/$defs/Color", "default": "red"}
    assert p["$defs"]["Color"] == {"enum": ["red", "blue"], "type": "string"}


class Point(BaseModel):
    x: int
    y: int = 0


class Shape(BaseModel):
    title: str
    points: list[Point]


class Options(TypedDict):
    verbose: bool
    depth: int


def test_nested_models_and_typeddict():
    def f(shape: Shape, opts: Options | None = None):
        """Draw.

        Args:
            shape: The shape.
            opts: Options.
        """

    p = params(f)
    assert p["properties"]["shape"] == {"$ref": "#/$defs/Shape", "description": "The shape."}
    assert p["$defs"]["Shape"] == {
        "type": "object",
        "properties": {"title": {"type": "string"}, "points": {"type": "array", "items": {"$ref": "#/$defs/Point"}}},
        "required": ["title", "points"],
    }
    assert p["$defs"]["Point"] == {
        "type": "object",
        "properties": {"x": {"type": "integer"}, "y": {"type": "integer", "default": 0}},
        "required": ["x"],
    }
    assert p["$defs"]["Options"]["properties"] == {"verbose": {"type": "boolean"}, "depth": {"type": "integer"}}
    assert set(p["$defs"]["Options"]["required"]) == {"verbose", "depth"}


def test_parameter_named_title():
    def f(title: str, name: str = "x"):
        """Set.

        Args:
            title: The new title.
        """

    p = params(f)
    assert p["properties"]["title"] == {"type": "string", "description": "The new title."}


def test_dict_default_containing_title_is_kept():
    def f(meta: dict = {"title": "Untitled"}): ...  # noqa: B006

    assert params(f)["properties"]["meta"]["default"] == {"title": "Untitled"}


def test_unannotated_parameter_accepts_anything():
    def f(x, y: int = 1): ...

    p = params(f)
    assert p["properties"]["x"] == {}
    assert p["required"] == ["x"]


def test_annotated_constraints_are_kept():
    def f(n: Annotated[int, Field(ge=1, le=10)] = 5):
        """Pick.

        Args:
            n: How many.
        """

    assert params(f)["properties"]["n"] == {
        "type": "integer", "minimum": 1, "maximum": 10, "default": 5, "description": "How many.",
    }


def test_annotated_description_kept_when_docstring_is_silent():
    def f(n: Annotated[int, Field(description="From Annotated.")] = 5):
        """Pick."""

    assert params(f)["properties"]["n"]["description"] == "From Annotated."


def test_string_annotations_resolve():
    ns: dict[str, Any] = {}
    exec(
        "from __future__ import annotations\n"
        "def f(a: list[int], b: str | None = None):\n"
        "    '''Do.\n\n    Args:\n        a: Numbers.\n    '''\n",
        ns,
    )
    p = params(ns["f"])
    assert p["properties"]["a"] == {"type": "array", "items": {"type": "integer"}, "description": "Numbers."}
    assert p["required"] == ["a"]


def test_wrapped_function_uses_the_original_signature():
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper

    @deco
    def f(path: str, n: int = 1):
        """Do.

        Args:
            path: A path.
        """

    p = params(f)
    assert set(p["properties"]) == {"path", "n"}
    assert p["properties"]["path"] == {"type": "string", "description": "A path."}


def test_keyword_only_params():
    def f(a: int, *, c: int = 3): ...

    p = params(f)
    assert set(p["properties"]) == {"a", "c"}
    assert p["required"] == ["a"]


def test_positional_only_params_are_rejected():
    # Tools are called with keyword arguments, so a positional-only parameter could never be passed.
    def f(a: int, /, b: int): ...

    with pytest.raises(ValueError):
        tool_schema("f", f)


def test_param_names_that_shadow_basemodel_attributes():
    def f(schema: str, json: str, copy: bool = False): ...

    p = params(f)
    assert set(p["properties"]) == {"schema", "json", "copy"}
    assert p["required"] == ["schema", "json"]


def test_param_starting_with_underscore():
    def f(_hidden: int, visible: int): ...

    assert set(params(f)["properties"]) == {"_hidden", "visible"}
