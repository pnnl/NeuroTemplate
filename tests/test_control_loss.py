"""The declared objective's arithmetic: operand, map, norm, units, gain.

Each test hands `build_control_loss` a rolled-out dict it can compute by hand,
so what is pinned is the term a gain multiplies — a wrong slice, a wrong unit
or a wrong reduction trains silently on the wrong objective.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import build_control_loss
from neurotemplate.control.losses import power_mean
from neurotemplate.control.scenarios import horizons
from control_fixtures import HIGH, LOW, SETPT, ZONE, full_setup


def _data(s, zone, low=24.0, high=25.0, u=None):
    """A rolled-out dict with the zone held at `zone` °C (a scalar, or one
    value per scored step) against the band [low, high]."""
    cfg, nm = s["cfg"], s["nm"]
    _, _, hmax = horizons(cfg, s["sysid_cfg"])
    P, b = cfg.train_pred_steps, 2
    W = hmax + P

    yn = torch.zeros(b, W, len(nm["Y"]))
    yn[:, hmax:, nm["Y"].index(ZONE)] = torch.as_tensor(zone, dtype=torch.float32)
    R = torch.empty(b, W, 2)
    R[..., 0], R[..., 1] = high, low                          # sorted: High, Low
    U = torch.zeros(b, W - 1, len(nm["U"])) if u is None else u
    return {"yn": yn, "U": U, "D": torch.zeros(b, W, len(nm["D"])), "R": R}


def _gains(**gains):
    return {"comfort_lower": 0.0, "comfort_upper": 0.0, "du": 0.0, **gains}


def test_the_bound_is_the_rms_violation_in_physical_units():
    """norm 2 is sqrt(mean(h²)) per window, not mean(h²): a 1 °C overshoot on
    one step of P costs 1/sqrt(P) °C, whatever the zone's stretch."""
    s = full_setup(gains=_gains(comfort_upper=3.0))
    P = s["cfg"].train_pred_steps
    zone = torch.full((P,), 24.5)
    zone[2] = 26.0                                  # 1 °C above the 25 °C ceiling
    out = build_control_loss(s["cfg"], s["sysid_cfg"], s["nm"])(_data(s, zone))
    assert float(out["comfort_upper_loss"]) == pytest.approx(3.0 / P ** 0.5, rel=1e-4)
    assert float(out["comfort_lower_loss"]) == 0.0


def test_relu_bounds_from_below_and_nrelu_from_above():
    """The direction lives in the map, not in a name."""
    s = full_setup(gains=_gains(comfort_lower=1.0, comfort_upper=1.0))
    loss = build_control_loss(s["cfg"], s["sysid_cfg"], s["nm"])
    cold, hot = loss(_data(s, 23.0)), loss(_data(s, 26.0))
    assert float(cold["comfort_lower_loss"]) == pytest.approx(1.0, rel=1e-4)
    assert float(cold["comfort_upper_loss"]) == 0.0
    assert float(hot["comfort_upper_loss"]) == pytest.approx(1.0, rel=1e-4)
    assert float(hot["comfort_lower_loss"]) == 0.0


def test_a_first_difference_is_scored_per_unit_of_movement():
    """A gain per °C of setpoint movement."""
    s = full_setup(gains=_gains(du=1.0))
    cfg, nm = s["cfg"], s["nm"]
    _, _, hmax = horizons(cfg, s["sysid_cfg"])
    u = torch.randn(2, hmax - 1 + cfg.train_pred_steps, 1)
    got = float(build_control_loss(cfg, s["sysid_cfg"], nm)(
        _data(s, 24.5, u=u))["du_loss"])

    commands = u[:, hmax - 1:, 0]
    expected = (commands[:, 1:] - commands[:, :-1]).abs().mean()
    assert got == pytest.approx(float(expected), rel=1e-5)


# ── power_mean ───────────────────────────────────────────────────


def test_power_mean_is_monotone_in_q_and_bounded_by_the_max():
    p = torch.tensor([[1.0, 2.0, 3.0, 10.0]])
    values = [float(power_mean(p, q)) for q in (1.0, 2.0, 4.0, 8.0, 32.0)]
    assert values == sorted(values)
    assert values[-1] <= float(p.max()) + 1e-6
    assert values[-1] > values[0]


def test_power_mean_reduces_per_window_then_averages_over_the_batch():
    """A pooled power mean would give most of its gradient to the worst
    window; check the two-window arithmetic by hand."""
    p = torch.tensor([[0.0, 2.0], [6.0, 6.0]])
    w0 = ((0.0 ** 4 + 2.0 ** 4) / 2) ** (1 / 4)
    assert float(power_mean(p, 4.0)) == pytest.approx((w0 + 6.0) / 2, rel=1e-6)


def test_an_all_zero_window_does_not_poison_the_gradient():
    """The root has an infinite derivative at 0, and a zone inside its band
    all window is the common case. Pin that it stays finite."""
    h = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    power_mean(h, 2.0).backward()
    assert torch.isfinite(h.grad).all()
    assert h.grad[1, 1] > 0
