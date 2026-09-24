"""Discover an application's user-defined hook modules.

An application supplies two files in its working directory:
- `modeling_user_defined.py` — the modeling hooks
- `control_user_defined.py`  — the control hooks

This module is the single place that imports them, so library code never
does `from modeling_user_defined import ...` directly. Resolution order:

1. If `cfg.user_hooks_path` is set, treat it as a path or dotted module name
   and import that.
2. Otherwise prepend CWD to sys.path and import by the canonical module name.
3. If still missing, raise RuntimeError with a clear hint.

Either way the module is imported once per process and reused: a run looks
its hooks up many times, and a hooks file's top level must not re-run on
every lookup.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_MODELING_MODULE = "modeling_user_defined"
_CONTROL_MODULE = "control_user_defined"


def _import_from_file(path: Path) -> ModuleType:
    """Import a hooks file as a module named after its own stem.

    Named after the file, not the canonical hooks name, so a hooks file that
    itself imports `modeling_user_defined` gets the real one rather than a
    half-initialised copy of itself. Its directory goes on sys.path, as it
    would for a script, so it can import its siblings.
    """
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"Hooks file not found at {path}")
    name = path.stem
    loaded = sys.modules.get(name)
    if loaded is not None and getattr(loaded, "__file__", None) == str(path):
        return loaded

    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before running it, as `import` does, so the file's own
    # dataclasses and pickled objects can resolve their module.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _import_from_path_or_dotted(path_or_module: str) -> ModuleType:
    p = Path(path_or_module)
    if p.suffix == ".py" or p.exists():
        return _import_from_file(p)
    return importlib.import_module(path_or_module)


def _import_from_cwd(module_name: str) -> ModuleType:
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise  # the hooks file exists; something it imports does not
        raise RuntimeError(
            f"Could not find {module_name!r} in CWD ({cwd}). "
            f"Either place a {module_name}.py file in the working directory, "
            f"or set cfg.user_hooks_path to a file path or dotted module name."
        ) from exc


def load_modeling_hooks(cfg) -> ModuleType:
    """Return the modeling hooks module for this run."""
    if cfg.user_hooks_path:
        return _import_from_path_or_dotted(cfg.user_hooks_path)
    return _import_from_cwd(_MODELING_MODULE)


def load_control_hooks(cfg) -> ModuleType:
    """Return the control hooks module for this run."""
    if cfg.user_hooks_path:
        return _import_from_path_or_dotted(cfg.user_hooks_path)
    return _import_from_cwd(_CONTROL_MODULE)
