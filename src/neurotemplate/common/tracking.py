"""The MLflow run behind one `nt ... train` invocation, opened and closed in one place.

`RunLogger` is Neuromancer's `MLFlowLogger` with the single-item calls the
pipelines need made public, so nothing reaches into its private `_mlflow`.
Functions that log take a `RunLog` — the surface they call — so a test can
hand them a double that records instead of opening an MLflow run.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Optional, Protocol, Union, runtime_checkable

from neuromancer.loggers import MLFlowLogger


@runtime_checkable
class RunLog(Protocol):
    """What the pipelines log through: a `RunLogger`, or a test double."""

    def log_metrics(self, output: dict, step: int = 0) -> None: ...
    def log_dict(self, data: dict, name: str) -> None: ...
    def log_artifacts(self, artifacts: dict = ...) -> None: ...
    def log_artifact(self, path: Union[str, Path],
                     artifact_path: Optional[str] = None) -> None: ...
    def log_text(self, text: str, name: str) -> None: ...


class RunLogger(MLFlowLogger):
    """`MLFlowLogger`, plus public calls for one file, one text and one param.

    Neuromancer's logger has no public way to log a single artifact; it keeps
    the `mlflow` module as `_mlflow`, and these wrap it in one place.
    """

    def log_artifact(self, path: Union[str, Path],
                     artifact_path: Optional[str] = None) -> None:
        """Log one file or directory to the run, under `artifact_path`."""
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_text(self, text: str, name: str) -> None:
        """Log `text` to the run as the file `name`."""
        self._mlflow.log_text(text, name)

    def log_param(self, key: str, value) -> None:
        self._mlflow.log_param(key, value)

    def log_parameters(self) -> None:
        """Log every config field as a run param. Neuromancer's version also
        prints each field's type to stdout, a leftover debug line."""
        self._mlflow.log_params({k: getattr(self.args, k) for k in vars(self.args)})

    def fail(self) -> None:
        """Close the run as FAILED."""
        self._mlflow.end_run(status="FAILED")


@contextmanager
def tracked_run(
    cfg,
    *,
    stdout: list[str],
    logout: list[str],
    run_name: Optional[str] = None,
) -> Iterator[RunLogger]:
    """Open the top-level MLflow run for `cfg`, and always close it.

    A run that raises is marked FAILED and its scratch `savedir` removed, so
    a crash leaves neither a run stuck at RUNNING in the MLflow UI nor a stray
    temp directory in the working directory. A run that finishes is closed by
    the logger's own `clean_up`, which also removes `savedir` — anything to
    keep must be logged to MLflow before the block ends.

    `run_name` overrides `cfg.run` for this run's MLflow name only; `cfg`
    itself is not modified.

    `save_weights=False` stops Neuromancer's Trainer from writing its own
    checkpoints at the end of `train()` — among them a pickled `best_model.pth`
    that nothing reads. Each pipeline writes `best_model_state_dict.pth`
    itself (`save_checkpoint`).
    """
    args = cfg if run_name is None else replace(cfg, run=run_name)
    logger = RunLogger(
        args=args,
        savedir=cfg.savedir,
        stdout=stdout,
        verbosity=cfg.epoch_verbose if cfg.verbose else 0,
        logout=logout,
        nested=False,
        save_weights=False,
    )
    try:
        yield logger
    except BaseException:
        logger.fail()
        shutil.rmtree(logger.savedir, ignore_errors=True)
        raise
    logger.clean_up()


#: The one checkpoint a run carries: the trained Problem's state_dict.
CHECKPOINT = "best_model_state_dict.pth"


def save_checkpoint(logger: RunLog, state_dict: dict) -> None:
    """Log `state_dict` to the run as its checkpoint, `CHECKPOINT`."""
    logger.log_artifacts({CHECKPOINT: state_dict})
