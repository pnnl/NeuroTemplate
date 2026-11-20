"""Load, once, what the `nt` commands load, so a fresh install pays for it in `nt init`.

The first time a new environment opens a file, it is slow: Python writes its bytecode,
and on a managed machine a malware scanner reads every new library before it may load.
Across the 1-2 GB that torch, Neuromancer and MLflow install, that is a minute or more,
and without this it lands on the first few runs: a column listing that should take a
second takes a minute. Here it lands in setup, where it is expected.

It imports every command's module, then does once what a run does beyond its imports:
opens an MLflow store (which builds its schema) and logs a run to it, and saves a figure
as a PDF. All of that happens in a temporary directory, which is removed. Nothing here
is required: a step that fails is skipped, and the run that needs it loads it then.

The first import of a module writes its bytecode, a new file, which the scanner reads
when the next process opens it. So `nt init` runs this twice: in full, then with
`--imports-only`, which opens that bytecode; otherwise the first run would.

    python -m neurotemplate.warmup [--imports-only]
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings
from importlib import import_module

# What `mlflow ui` serves from, beside the commands' own modules.
UI_MODULES = ("mlflow.server.fastapi_app", "uvicorn")


def _quietly(step) -> None:
    try:
        step()
    except Exception:  # the run that needs it loads it then
        pass


def _imports() -> None:
    from neurotemplate.cli import COMMANDS
    names = dict.fromkeys(target.split(":")[0] for target in COMMANDS.values())
    for name in [*names, *UI_MODULES]:
        _quietly(lambda name=name: import_module(name))


def _tracked_run() -> None:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("warmup")
    with mlflow.start_run():
        mlflow.log_metric("x", 1.0)
        mlflow.log_dict({"x": 1}, "x.json")


def _figure() -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("warmup")
    fig.savefig("warmup.pdf")
    plt.close(fig)


def main() -> None:
    warnings.simplefilter("ignore")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    _imports()
    if "--imports-only" in sys.argv[1:]:
        return
    # An open SQLite handle can keep Windows from removing the store; leave it then.
    with tempfile.TemporaryDirectory(prefix="nt-warmup-", ignore_cleanup_errors=True) as tmp:
        os.chdir(tmp)
        _quietly(_tracked_run)
        _quietly(_figure)
        os.chdir(os.path.dirname(tmp))


if __name__ == "__main__":
    main()
