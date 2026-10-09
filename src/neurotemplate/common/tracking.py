"""The MLflow run behind one `nt ... train` invocation, opened and closed in one place."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator, Optional

from neuromancer.loggers import MLFlowLogger


@contextmanager
def tracked_run(
    cfg,
    *,
    stdout: list[str],
    logout: list[str],
    run_name: Optional[str] = None,
) -> Iterator[MLFlowLogger]:
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
    logger = MLFlowLogger(
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
        logger._mlflow.end_run(status="FAILED")
        shutil.rmtree(logger.savedir, ignore_errors=True)
        raise
    logger.clean_up()


#: The one checkpoint a run carries: the trained Problem's state_dict.
CHECKPOINT = "best_model_state_dict.pth"


def save_checkpoint(logger: MLFlowLogger, state_dict: dict) -> None:
    """Log `state_dict` to the run as its checkpoint, `CHECKPOINT`."""
    logger.log_artifacts({CHECKPOINT: state_dict})
