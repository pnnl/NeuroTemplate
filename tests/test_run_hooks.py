"""A trained run reloads with the hooks file it trained with, not today's.

A rebuild reads hooks no checkpoint records — the reference rule a residual
is anchored to has no weights — so a reload that read the working directory's
file would silently turn a trained run into a different controller.
"""

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
import torch

from neurotemplate.common.tracking import tracked_run
from neurotemplate.control import load as control_load
from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem,
)
from neurotemplate.control.pipeline import train_policy
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import build_loaders, build_window_dict
from neurotemplate.hooks import CONTROL_SNAPSHOT, load_control_hooks, snapshot_hooks
from control_fixtures import SETPT, full_setup, make_norm_stats

FIXTURE_HOOKS = Path(__file__).parent / "fixtures" / "control_user_defined.py"


@pytest.fixture(autouse=True)
def _restore_tracking_uri():
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)
    for name in list(sys.modules):
        if name.startswith(("site_control", "control_user_defined_")):
            del sys.modules[name]


def test_editing_the_hooks_file_after_training_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # MLflow's default artifact root is ./mlruns
    hooks_file = tmp_path / "site_control.py"
    shutil.copyfile(FIXTURE_HOOKS, hooks_file)
    s = full_setup(residual_delta={SETPT: [1.0, 1.0]}, user_hooks_path=str(hooks_file),
                   location=f"sqlite:///{tmp_path / 'mlflow.db'}",
                   savedir=str(tmp_path / "scratch"), verbose=False)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    closed_loop = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    problem = build_problem(closed_loop, build_control_loss(cfg, sysid_cfg, nm, ns))
    loaders = build_loaders(cfg, sysid_cfg, nm, [s["norm_df"]], [s["norm_df"]])
    with tracked_run(cfg, stdout=[], logout=[]) as logger:
        run_id = logger.run.info.run_id
        snapshot_hooks(load_control_hooks(cfg), CONTROL_SNAPSHOT, logger)
        train_policy(cfg, problem, *loaders, logger=logger)

    windows = build_window_dict(s["norm_df"], cfg, sysid_cfg, nm,
                                P=cfg.train_pred_steps, stride=cfg.history_steps)
    with torch.no_grad():
        trained = closed_loop(dict(windows))["U"]

    # A new rule in the working directory's file, and no cached copy of it.
    with hooks_file.open("a") as f:
        f.write("\n_rule = reference_control\n"
                "def reference_control(*args):\n"
                "    return {k: v + 1.0 for k, v in _rule(*args).items()}\n")
    for name in list(sys.modules):
        if name.startswith("site_control"):
            del sys.modules[name]

    monkeypatch.setattr(control_load, "load_frozen_dynamics", lambda *args: (
        s["dyn_sys"], s["dyn_node"], sysid_cfg, nm, make_norm_stats(s["plant"])))
    _, reloaded, *_ = control_load.load_trained_policy(run_id)
    with torch.no_grad():
        assert torch.equal(reloaded(dict(windows))["U"], trained)


def test_a_hooks_file_importing_a_sibling_is_refused_at_snapshot(tmp_path):
    """The copy is one file; a sibling it imports would be read from wherever
    the reload happens to run, silently."""
    (tmp_path / "site_helpers.py").write_text("BAND = 'Comfort Low [°C]'\n")
    (tmp_path / "site_control.py").write_text("from site_helpers import BAND\n")
    hooks = SimpleNamespace(__file__=str(tmp_path / "site_control.py"))
    logger = SimpleNamespace(savedir=str(tmp_path))
    with pytest.raises(RuntimeError, match=r"\['site_helpers'\]"):
        snapshot_hooks(hooks, CONTROL_SNAPSHOT, logger)
