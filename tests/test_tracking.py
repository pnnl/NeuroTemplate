"""A training run that crashes is closed as FAILED and leaves no scratch dir."""

from dataclasses import dataclass
from pathlib import Path

import mlflow
import pytest

from neurotemplate.common.tracking import tracked_run


@dataclass
class _Cfg:
    location: str
    savedir: str
    exp: str = "tracking-test"
    run: str = "crash"
    verbose: bool = False
    epoch_verbose: int = 1


@pytest.fixture(autouse=True)
def _restore_tracking_uri():
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)


def test_a_crashed_run_is_marked_failed_and_cleaned_up(tmp_path):
    cfg = _Cfg(location=f"sqlite:///{tmp_path / 'mlflow.db'}",
               savedir=str(tmp_path / "scratch"))
    with pytest.raises(RuntimeError, match="boom"):
        with tracked_run(cfg, stdout=[], logout=[]) as logger:
            run_id = logger.run.info.run_id
            raise RuntimeError("boom")

    assert mlflow.get_run(run_id).info.status == "FAILED"
    assert not Path(cfg.savedir).exists()
    assert mlflow.active_run() is None
