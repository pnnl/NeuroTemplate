"""The MLflow run around training: a crash is closed as FAILED and leaves no
scratch dir; a finished run carries exactly one checkpoint, the state dict."""

from dataclasses import dataclass
from pathlib import Path

import mlflow
import pytest

from neurotemplate.common.tracking import CHECKPOINT, tracked_run


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


def test_a_finished_run_carries_its_state_dict_and_no_pickled_model(
        tmp_path, monkeypatch):
    """Neuromancer's Trainer pickles the whole model as `best_model.pth`
    unless told not to; the pipeline writes the state dict itself. Forgetting
    that write on one path makes that path's runs unloadable."""
    import torch
    from control_fixtures import full_setup
    from neurotemplate.control.closed_loop import (
        build_closed_loop, build_control_loss, build_problem,
    )
    from neurotemplate.control.pipeline import train_policy
    from neurotemplate.control.policy import build_policy
    from neurotemplate.control.scenarios import build_loaders

    monkeypatch.chdir(tmp_path)   # MLflow's default artifact root is ./mlruns
    s = full_setup(location=f"sqlite:///{tmp_path / 'mlflow.db'}",
                   savedir=str(tmp_path / "scratch"), verbose=False)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    problem = build_problem(build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg),
                            build_control_loss(cfg, sysid_cfg, nm, ns))
    loaders = build_loaders(cfg, sysid_cfg, nm, [s["norm_df"]], [s["norm_df"]])
    with tracked_run(cfg, stdout=[], logout=[]) as logger:
        run_id = logger.run.info.run_id
        train_policy(cfg, problem, *loaders, logger=logger)

    logged = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
    assert logged == [CHECKPOINT]
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=CHECKPOINT)
    problem.load_state_dict(torch.load(path))
