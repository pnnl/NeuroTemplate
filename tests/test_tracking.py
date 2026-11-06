"""The MLflow run around training: a crash is closed as FAILED and leaves no
scratch dir; a finished run carries exactly one checkpoint, the state dict."""


import mlflow
import pytest

from neurotemplate.common.tracking import CHECKPOINT, tracked_run


@pytest.fixture(autouse=True)
def _restore_tracking_uri():
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)


def test_training_that_cannot_return_a_trained_model_stops_with_a_reason():
    """Silent runs too: this is not a matter of what gets printed."""
    from neurotemplate.common.progress import Progress

    progress = Progress("silent", 1)
    progress.start(9)
    with pytest.raises(RuntimeError, match="step 1.*every weight is lost"):
        progress.update(1, float("nan"), 1.0)

    # A non-finite dev loss may recover, so only one that never is finite is refused.
    progress = Progress("silent", 1)
    progress.start(9)
    progress.update(1, 1.0, float("inf"))
    with pytest.raises(RuntimeError, match="No finite dev loss"):
        progress.finish()

    progress.update(2, 1.0, 0.5)
    progress.finish()


def test_a_run_of_the_other_kind_or_outside_the_store_says_so(tmp_path, monkeypatch):
    from neurotemplate.common.config import ConfigError
    from neurotemplate.control.config import ControlConfig
    from neurotemplate.modeling import load
    from neurotemplate.modeling.config import ModelingConfig

    monkeypatch.setattr(load, "_load_run_params", lambda run_id: {"epochs": 3})
    with pytest.raises(ConfigError, match="is a model run; this takes a control run"):
        load.config_from_run_params("abc", ControlConfig)
    monkeypatch.setattr(load, "_load_run_params", lambda run_id: {"sysid_run_id": "x"})
    with pytest.raises(ConfigError, match="is a control run; this takes a model run"):
        load.config_from_run_params("abc", ModelingConfig)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    with pytest.raises(ConfigError, match="run this from the project"):
        load._get_run("0" * 32)


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
    from neurotemplate.control.windows import build_loaders

    monkeypatch.chdir(tmp_path)   # MLflow's default artifact root is ./mlruns
    s = full_setup(location=f"sqlite:///{tmp_path / 'mlflow.db'}",
                   savedir=str(tmp_path / "scratch"), progress="silent")
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    problem = build_problem(build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg),
                            build_control_loss(cfg, sysid_cfg, nm))
    loaders = build_loaders(cfg, sysid_cfg, nm, [s["frame"]], [s["frame"]])
    with tracked_run(cfg, stdout=[], logout=[]) as logger:
        run_id = logger.run.info.run_id
        train_policy(cfg, problem, *loaders, logger=logger)

    logged = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
    assert logged == [CHECKPOINT]
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=CHECKPOINT)
    problem.load_state_dict(torch.load(path))
