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
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_MODELING_MODULE = "modeling_user_defined"
_CONTROL_MODULE = "control_user_defined"


def _import_from_path_or_dotted(path_or_module: str, fallback_name: str) -> ModuleType:
    p = Path(path_or_module)
    if p.suffix == ".py" or p.exists():
        path = p if p.is_absolute() else Path.cwd() / p
        if not path.exists():
            raise RuntimeError(f"Hooks file not found at {path}")
        spec = importlib.util.spec_from_file_location(fallback_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fallback_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(path_or_module)


def _import_from_cwd(module_name: str) -> ModuleType:
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not find {module_name!r} in CWD ({cwd}). "
            f"Either place a {module_name}.py file in the working directory, "
            f"or set cfg.user_hooks_path to a file path or dotted module name."
        ) from exc


def load_modeling_hooks(cfg) -> ModuleType:
    """Return the modeling hooks module for this run."""
    if cfg.user_hooks_path:
        return _import_from_path_or_dotted(cfg.user_hooks_path, _MODELING_MODULE)
    return _import_from_cwd(_MODELING_MODULE)


def load_control_hooks(cfg) -> ModuleType:
    """Return the control hooks module for this run."""
    if cfg.user_hooks_path:
        return _import_from_path_or_dotted(cfg.user_hooks_path, _CONTROL_MODULE)
    return _import_from_cwd(_CONTROL_MODULE)
