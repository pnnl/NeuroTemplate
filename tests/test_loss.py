"""
Tracking-loss alignment.

`yn` carries the seeded history plus one extra timestep, so the ref loss must
score only the prediction span — the same slice
pipeline.evaluate_and_plot reports. If the seeded prefix leaked in it would
contribute zero error and shrink the loss as history_steps grows.
"""
import pytest
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.loss import build_loss
from model_fixtures import model_cfg, steps, uniform_past, unit_stats

NY, NU, ND = 2, 1, 1
P = 6


def make_cfg(history_steps: int) -> Config:
    return model_cfg(model="mlp", past=uniform_past(nm_mapping(), history_steps),
                     horizon=steps(P),
                     ref_gain=1.0, grad_gain=0.0)


def nm_mapping() -> dict[str, list[str]]:
    return {"Y": ["y0", "y1"], "U": ["u0"], "D": ["d0"], "t": ["sec_elapsed"]}


def build(history_steps: int):
    cfg = make_cfg(history_steps)
    system = get_dynamics(cfg, nm_mapping(), unit_stats(nm_mapping()))
    loss = build_loss(
        cfg=cfg, system=system,
        grad_masks={}, nm_mapping=nm_mapping(), norm_stats=unit_stats(nm_mapping()),
    )
    return cfg, loss


def rollout_dict(history_steps: int, offset: float, batch: int = 3) -> dict:
    """A finished rollout where every prediction is `offset` off the truth and
    the seeded prefix matches exactly, as the System guarantees."""
    total = history_steps + P
    torch.manual_seed(0)
    Y = torch.randn(batch, total, NY)
    yn = torch.cat([Y, Y[:, -1:, :]], dim=1).clone()
    yn[:, history_steps:, :] += offset
    return {"Y": Y, "yn": yn}


@pytest.mark.parametrize("history_steps", [12])
def test_ref_loss_scores_only_the_prediction_span(history_steps):
    """A constant error over every prediction must give exactly that squared
    error, independent of how long the seeded history is."""
    _, loss = build(history_steps)

    value = float(loss(rollout_dict(history_steps, offset=0.5))["loss"])

    assert value == pytest.approx(0.25, abs=1e-6)


def test_ref_loss_ignores_the_seeded_prefix():
    """Perturbing the seed changes nothing — those positions are not scored."""
    history_steps = 4
    _, loss = build(history_steps)

    clean = rollout_dict(history_steps, offset=0.5)
    dirty = {k: v.clone() for k, v in clean.items()}
    dirty["yn"][:, :history_steps, :] += 10.0

    assert float(loss(clean)["loss"]) == pytest.approx(
        float(loss(dirty)["loss"]), abs=1e-6
    )
