"""The MLflow run behind one `nt ... train` invocation, opened and closed in one place.

`RunLogger` is Neuromancer's `MLFlowLogger` with the single-item calls the
pipelines need made public, so nothing reaches into its private `_mlflow`.
Functions that log take a `RunLog` — the surface they call — so a test can
hand them a double that records instead of opening an MLflow run.
"""

from __future__ import annotations

import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Protocol, Union, runtime_checkable

import mlflow

from neurotemplate.nm.loggers import MLFlowLogger

# MLflow narrates its own bookkeeping at INFO: forty lines of database migration
# on a project's first run, and a line for each experiment it creates. None of it
# is the user's to act on; its warnings still show. After `import mlflow`, which
# sets these levels itself.
for _name in ("mlflow", "alembic"):
    logging.getLogger(_name).setLevel(logging.WARNING)


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

    def log_parameters(self) -> None:
        """Log every config field as a run param. Neuromancer's version also
        prints each field's type to stdout, a leftover debug line."""
        self._mlflow.log_params({k: getattr(self.args, k) for k in vars(self.args)})

    def log_weights(self, model) -> None:
        """Log the trainable parameter count, without Neuromancer's print."""
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._mlflow.log_metric("nparams", float(n))

    def fail(self) -> None:
        """Close the run as FAILED."""
        self._mlflow.end_run(status="FAILED")


@contextmanager
def tracked_run(
    cfg,
    *,
    stdout: list[str],
    logout: list[str],
) -> Iterator[RunLogger]:
    """Open the top-level MLflow run for `cfg`, and always close it.

    A run that raises is marked FAILED and its scratch `savedir` removed, so
    a crash leaves neither a run stuck at RUNNING in the MLflow UI nor a stray
    temp directory in the working directory. A run that finishes is closed by
    the logger's own `clean_up`, which also removes `savedir` — anything to
    keep must be logged to MLflow before the block ends.

    `save_weights=False` stops Neuromancer's Trainer from writing its own
    checkpoints at the end of `train()` — among them a pickled `best_model.pth`
    that nothing reads. Each pipeline writes `best_model_state_dict.pth`
    itself (`save_checkpoint`).
    """
    logger = RunLogger(
        args=cfg,
        savedir=cfg.savedir,
        stdout=stdout,
        verbosity=0,  # `common.progress` prints instead
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


def artifact_file(run_id: str, name: str) -> Path:
    """Where a finished run's artifact `name` is on disk, for a closing summary:
    relative to the working directory when it is under it, as it is in a project.

    Training writes the reportcard in a scratch `savedir` that the run removes
    once it is logged, so the artifact directory is the one place it stays.
    """
    uri = mlflow.get_run(run_id).info.artifact_uri
    path = Path(uri.removeprefix("file://")) / name
    return path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path


#: The one checkpoint a run carries: the trained Problem's state_dict.
CHECKPOINT = "best_model_state_dict.pth"


def save_checkpoint(logger: RunLog, state_dict: dict) -> None:
    """Log `state_dict` to the run as its checkpoint, `CHECKPOINT`."""
    logger.log_artifacts({CHECKPOINT: state_dict})
