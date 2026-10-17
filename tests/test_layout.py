"""The package's layers: core <- tools <- agents <- ui, each importing only from the layers to its left.
Clients (`ui/`) go further: they drive agents only through a `Driver`."""

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
DRIVING = {"run_turn", "step", "interrupt"}  # Agent methods only a Driver calls; ui/ uses the Driver's instead


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
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:  # the module, and each name as a submodule: `from jean_code import core` is caught too
                names = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)]
            assert not [n for n in names if n.startswith(banned)], (path.name, names)


def test_clients_drive_agents_only_through_a_driver():
    """A client that called an agent directly would bypass the driver's job thread, busy flag and approvals."""
    for path in (PACKAGE / "ui").glob("*.py"):
        calls = [
            n.func.attr
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in DRIVING
        ]
        assert calls == [], (path.name, calls)
