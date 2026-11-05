"""Discover an application's user-defined hook modules.

An application supplies two files in its working directory:
- `nt_model.py` — the modeling hooks
- `nt_control.py` — the control hooks

This module is the single place that imports them, so library code never
does `from nt_model import ...` directly. Resolution order:

1. If `cfg.user_hooks_path` is set, treat it as a path or dotted module name
   and import that.
2. Otherwise prepend CWD to sys.path and import by the canonical module name.
3. If still missing, raise RuntimeError with a clear hint.

Either way the module is imported once per process and reused: a run looks
its hooks up many times, and a hooks file's top level must not re-run on
every lookup.

A trained run carries the hooks file it was trained with (`snapshot_hooks`),
and every reload points `cfg.user_hooks_path` at that copy
(`run_hooks_path`), never at the working directory: a rebuild reads hooks
that no checkpoint records, `power_metric` for one, so editing the file after training would otherwise silently change
what a trained run means.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

_MODELING_MODULE = "nt_model"
_CONTROL_MODULE = "nt_control"

#: What each stage's hooks file is logged as in its run.
MODELING_SNAPSHOT = f"{_MODELING_MODULE}.py"
CONTROL_SNAPSHOT = f"{_CONTROL_MODULE}.py"


def _import_from_file(path: Path) -> ModuleType:
    """Import a hooks file as a module named after its own path.

    Not the canonical hooks name, so a hooks file that itself imports
    `nt_model` gets the real one rather than a half-initialised
    copy of itself; and unique to the path, so a run's snapshot never
    replaces the working directory's file of the same name, or the reverse.
    Its directory goes on sys.path, as it would for a script, so it can
    import its siblings.
    """
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"Hooks file not found at {path}")
    name = f"{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    loaded = sys.modules.get(name)
    if loaded is not None:
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


def find_cwd_modeling_hooks() -> Optional[ModuleType]:
    """The working directory's modeling hooks, or None if it has no
    `nt_model.py` — for config-free tools that only want the column roles."""
    if not (Path.cwd() / f"{_MODELING_MODULE}.py").exists():
        return None
    return _import_from_cwd(_MODELING_MODULE)


def load_control_hooks(cfg) -> ModuleType:
    """Return the control hooks module for this run."""
    if cfg.user_hooks_path:
        return _import_from_path_or_dotted(cfg.user_hooks_path)
    return _import_from_cwd(_CONTROL_MODULE)


# ── A run's own copy ─────────────────────────────────────────────


def snapshot_hooks(hooks: ModuleType, name: str, logger) -> None:
    """Log the file `hooks` was imported from to the run as `name`.

    Called as a run opens, so the copy is the file training reads. It is one
    file: a hooks file importing a module beside it is refused, since the
    run would not carry that module.
    """
    path = Path(hooks.__file__)
    siblings = _sibling_imports(path)
    if siblings:
        raise RuntimeError(
            f"{path.name} imports {siblings} from beside it. A run keeps a "
            "copy of its hooks file to reload with, and that copy is one "
            f"file. Move that code into {path.name}, or install it as a "
            "package."
        )
    shutil.copyfile(path, Path(logger.savedir) / name)
    logger.log_artifacts({})


def run_hooks_path(artifacts_dir: Path, name: str) -> str:
    """The hooks file a run was trained with, for `cfg.user_hooks_path`."""
    path = Path(artifacts_dir) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{name} is missing from the run's artifacts ({artifacts_dir}); "
            "every run trains with a copy of its hooks file. A run without "
            "one was trained by an older version: retrain it."
        )
    return str(path)


def _sibling_imports(path: Path) -> list[str]:
    """Modules `path` imports that live beside it (or relatively)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            found.add("." * node.level + (node.module or "").split(".")[0])
    return sorted(
        name for name in found
        if name.startswith(".") or (path.parent / f"{name}.py").exists()
        or (path.parent / name / "__init__.py").exists()
    )
