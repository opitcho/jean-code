"""Checks the docstring rule ruff can't: a summary is one line of at most 80 chars.

The rule is in CLAUDE.md. `@tool` docstrings are prompts, so they're left out.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "jean_code"


def is_tool(node: ast.AST) -> bool:
    return any(isinstance(d, ast.Name) and d.id == "tool" for d in getattr(node, "decorator_list", []))


def bad_summaries(path: Path) -> list[str]:
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or is_tool(node):
            continue
        doc = ast.get_docstring(node)
        summary = doc.split("\n\n")[0] if doc else ""
        if doc and ("\n" in summary or len(summary) > 80 or not summary.endswith((".", "?", "!"))):
            bad.append(f"{path.name}:{getattr(node, 'lineno', 1)}  {summary.splitlines()[0]}")
    return bad


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_summary_is_one_short_line(path):
    assert bad_summaries(path) == []
