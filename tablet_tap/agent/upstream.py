"""Import shim for the unmodified Mobile-Agent-v3.5 files in ``third_party/mobile_agent_v35/mobile_use``.

Upstream's files import each other by generic top-level names (``from utils import ...``, ``from packages import ...``).
Putting their directory on ``sys.path`` would leave a module called ``utils`` in ``sys.modules`` for the whole process
and could collide with any other top-level ``packages`` on the path. So each file is
loaded by path under a private name; the generic names exist in ``sys.modules`` only while the run script is executed
and whatever was registered under them before is put back afterwards.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

UPSTREAM_DIR = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "mobile_agent_v35" / "mobile_use"
_PREFIX = "_mobile_agent_v35_"
_MISSING = object()


def _load(name: str) -> types.ModuleType:
    """Execute ``UPSTREAM_DIR/<name>.py`` as the module ``_mobile_agent_v35_<name>``."""
    private = _PREFIX + name
    if private in sys.modules:
        return sys.modules[private]
    spec = importlib.util.spec_from_file_location(private, UPSTREAM_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream module {name} from {UPSTREAM_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[private] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(private, None)
        raise
    return module


def _load_all() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Load utils, packages and the run script; the generic names are aliased only during the last import."""
    stub = None
    if importlib.util.find_spec("qwen_vl_utils") is None:
        # utils.py does `from qwen_vl_utils import smart_resize` and then defines its own smart_resize further down,
        # which is the one it uses. The package itself pulls in torch and torchvision, which the robot-side
        # environment does not carry, so a placeholder stands in for the import.
        stub = types.ModuleType("qwen_vl_utils")
        stub.smart_resize = None
        sys.modules["qwen_vl_utils"] = stub
    try:
        utils_mod = _load("utils")
    finally:
        if stub is not None:
            sys.modules.pop("qwen_vl_utils", None)
    if getattr(utils_mod.smart_resize, "__module__", None) != utils_mod.__name__:
        raise ImportError("upstream utils.py no longer defines its own smart_resize: install qwen-vl-utils")
    packages_mod = _load("packages")
    aliases = {"utils": utils_mod, "packages": packages_mod}
    saved = {name: sys.modules.get(name, _MISSING) for name in aliases}
    sys.modules.update(aliases)
    try:
        run_mod = _load("run_gui_owl_1_5_for_mobile")
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return utils_mod, packages_mod, run_mod


utils, packages, run_mobile = _load_all()

# utils.py
SYSTEM_PROMPT = utils.SYSTEM_PROMPT
ERROR_CALLING_LLM = utils.ERROR_CALLING_LLM
GUIOwlWrapper = utils.GUIOwlWrapper
annotate_screenshot = utils.annotate_screenshot
build_messages = utils.build_messages
smart_resize = utils.smart_resize

# run_gui_owl_1_5_for_mobile.py
FORMAT_RETRY_HINT = run_mobile.FORMAT_RETRY_HINT
TOOL_CALL_OPEN = run_mobile.TOOL_CALL_OPEN
parse_action = run_mobile.parse_action
rescale_coordinates = run_mobile.rescale_coordinates

__all__ = [
    "ERROR_CALLING_LLM",
    "FORMAT_RETRY_HINT",
    "GUIOwlWrapper",
    "SYSTEM_PROMPT",
    "TOOL_CALL_OPEN",
    "UPSTREAM_DIR",
    "annotate_screenshot",
    "build_messages",
    "packages",
    "parse_action",
    "rescale_coordinates",
    "run_mobile",
    "smart_resize",
    "utils",
]
