"""What does not train says so: `lr: 0` keeps the model as built, with no
dev loss or train time to report; a parameter that could never train is
refused before training starts; and a surrogate a controller cannot
differentiate through is refused before any control run starts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from neuromancer.trainer import Trainer

from neurotemplate.common.data_utils import compute_norm_stats
from neurotemplate.control.sensitivity import command_sensitivity
from neurotemplate.modeling import experiment
from neurotemplate.modeling.contract import Windowed
from neurotemplate.modeling.pipeline import train_and_evaluate
from fake_logger import FakeLogger
from model_fixtures import trainable_cfg

NM_MAPPING = {"Y": ["y0"], "U": ["u0", "u1"], "D": ["d0"], "t": ["sec_elapsed"]}


def _frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "y0": rng.normal(20.0, 2.0, n), "u0": rng.uniform(0.0, 1.0, n),
        "u1": rng.uniform(0.0, 1.0, n), "d0": rng.normal(15.0, 5.0, n),
        "sec_elapsed": np.arange(n) * 300.0,
    })


def test_lr_zero_trains_nothing_and_reports_no_training(tmp_path, monkeypatch):
    def no_training(self):
        raise AssertionError("lr: 0 ran the training loop")
    monkeypatch.setattr(Trainer, "train", no_training)
    cfg = trainable_cfg(tmp_path, NM_MAPPING, lr=0.0, progress="silent")
    df = _frame()
    run = train_and_evaluate(
        cfg=cfg, train_dfs=[df], dev_dfs=[df], test_dfs=[df], nm_mapping=NM_MAPPING,
        norm_stats=compute_norm_stats(df, norm_type=cfg.norm_type), grad_masks={},
        synth_cfg=None, freq_s=300.0, logger=FakeLogger())

    assert run.best_dev_loss is None and run.train_time_s is None
    card = next(tmp_path.glob("*reportcard*.md")).read_text()
    assert "**Training:** none (`lr: 0`)" in card
    assert "Best dev loss" not in card and "Train time" not in card


def test_a_parameter_with_no_gradient_path_is_refused_before_training(tmp_path, monkeypatch):
    def with_an_unused_parameter(*args, **kwargs):
        system = get_dynamics(*args, **kwargs)
        system.register_parameter("orphan", nn.Parameter(torch.zeros(1)))
        return system
    get_dynamics = experiment.get_dynamics
    monkeypatch.setattr(experiment, "get_dynamics", with_an_unused_parameter)
    cfg = trainable_cfg(tmp_path, NM_MAPPING, progress="silent")
    df = _frame()
    with pytest.raises(ValueError, match=r"no gradient path.*orphan"):
        experiment.build_trainer(
            cfg=cfg, train_dfs=[df], dev_dfs=[df], nm_mapping=NM_MAPPING, grad_masks={},
            norm_stats=compute_norm_stats(df, norm_type=cfg.norm_type))


class _Through(nn.Module):
    """y_next from u0 only; `leave_graph` computes it outside autograd."""

    def __init__(self, leave_graph: bool):
        super().__init__()
        self.leave_graph = leave_graph

    def forward(self, y, u, d):
        u0 = u[:, -1, :1]
        if self.leave_graph:
            u0 = torch.from_numpy(u0.detach().numpy())
        return y[:, -1] + 0.1 * u0


def _system(model: nn.Module) -> SimpleNamespace:
    return SimpleNamespace(nodes=[SimpleNamespace(callable=Windowed(model, (1, 2, 1), 3))])


def test_a_command_the_model_ignores_reads_zero_and_no_gradient_path_raises():
    frames = [_frame()]
    cpu = torch.device("cpu")

    shares = command_sensitivity(_system(_Through(False)), frames, NM_MAPPING, cpu)
    assert [(c.column, c.zero_share) for c in shares] == [("u0", 0.0), ("u1", 1.0)]

    with pytest.raises(ValueError, match="no gradient path"):
        command_sensitivity(_system(_Through(True)), frames, NM_MAPPING, cpu)
