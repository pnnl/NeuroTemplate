import pytest
import torch

from neurotemplate.control.scenarios import horizons
from neurotemplate.control.closed_loop import build_control_loss
from control_fixtures import full_setup


def _data_dict(s, y_ctrl_value):
    """Synthetic rolled-out dict matching the loss's slicing."""
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    P = cfg.train_pred_steps
    b = 2
    ny, nu, n_ctrl = len(nm["Y"]), len(nm["U"]), 5
    W = hmax + P
    return {
        "yn": torch.full((b, W, ny), float(y_ctrl_value)),
        "U": torch.zeros(b, W, nu),
        "ymin": torch.zeros(b, W, n_ctrl),
        "ymax": torch.ones(b, W, n_ctrl),
    }


def test_comfort_penalty_zero_inside_positive_outside():
    s = full_setup(comfort_weight=1.0, du_weight=0.0, effort_weight=0.0)
    loss = build_control_loss(
        s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"],
    )

    inside = loss(_data_dict(s, 0.5))["loss"]
    outside = loss(_data_dict(s, 5.0))["loss"]

    assert float(inside) == pytest.approx(0.0, abs=1e-6)
    assert float(outside) > 0.0


def test_effort_objective_builds_and_runs():
    s = full_setup(comfort_weight=1.0, du_weight=0.0, effort_weight=1.0)
    loss = build_control_loss(
        s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"],
    )
    names = [o.name for o in loss.objectives]
    assert "effort_loss" in names
    out = loss(_data_dict(s, 0.5))
    assert torch.isfinite(out["loss"])


def test_no_active_objective_raises():
    s = full_setup(comfort_weight=0.0, du_weight=0.0, effort_weight=0.0)
    with pytest.raises(ValueError, match="No active control objective"):
        build_control_loss(
            s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"],
        )
