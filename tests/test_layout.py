"""The package's layers: core <- tools <- agents <- ui, each importing only from the layers to its left."""

import ast
from pathlib import Path

import pytest

import jean_code

PACKAGE = Path(jean_code.__file__).parent
FORBIDDEN = {  # layer -> the layers it must not import
    "core": ("tools", "agents", "ui"),
    "tools": ("agents", "ui"),
    "agents": ("ui",),
}


def imports(path: Path) -> list[ast.Import | ast.ImportFrom]:
    return [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Import | ast.ImportFrom)]


@pytest.mark.parametrize("layer", FORBIDDEN)
def test_layer_imports_nothing_above_it(layer):
    """Each layer imports only from the layers below it, so the folders say what depends on what."""
    files = list((PACKAGE / layer).glob("*.py"))
    assert files
    banned = tuple(f"jean_code.{name}" for name in FORBIDDEN[layer])
    for path in files:
        for node in imports(path):  # includes TYPE_CHECKING blocks
            assert getattr(node, "level", 0) == 0, (path.name, "relative import")
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not [n for n in names if n.startswith(banned)], (path.name, names)
