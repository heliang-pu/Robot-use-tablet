"""The package's layering, checked from the import statements: inner layers know nothing of outer ones."""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "tablet_tap"

# layer -> the layers it may import from (besides itself and the standard / third-party libraries)
ALLOWED = {
    "errors": set(),
    "core": {"errors"},
    "config": {"core", "errors"},
    "robot": {"core", "config", "errors"},
    "vision": {"core", "errors"},
    "calibration": {"core", "robot", "vision", "errors"},
    "agent": {"core", "robot", "vision", "errors"},
    "speech": {"errors"},
    "android": {"errors"},  # setting the tablet up over adb
    "app": {"core", "config", "robot", "vision", "calibration", "agent", "errors"},  # the composition root
    "cli": {"android", "app", "core", "config", "robot", "vision", "calibration", "agent", "speech", "errors"},
    "web": {"app", "core", "config", "robot", "vision", "calibration", "agent", "speech", "errors"},  # browser console
}


def layer_of(path: pathlib.Path) -> str:
    first = path.relative_to(PACKAGE).parts[0]
    return first.removesuffix(".py")


def imported_layers(path: pathlib.Path) -> set[str]:
    layers = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tablet_tap"):
            names = [node.module] if node.module != "tablet_tap" else [f"tablet_tap.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names if a.name.startswith("tablet_tap.")]
        layers.update(name.split(".")[1] for name in names if "." in name)
    return layers


MODULES = sorted(p for p in PACKAGE.rglob("*.py") if p.name not in ("__init__.py", "__main__.py"))


def test_every_module_belongs_to_a_layer():
    assert {layer_of(p) for p in MODULES} == set(ALLOWED)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_imports_point_inward(path):
    layer = layer_of(path)
    outside = imported_layers(path) - {layer} - ALLOWED[layer]
    assert not outside, f"{path.relative_to(PACKAGE)} ({layer}) must not import from {sorted(outside)}"
