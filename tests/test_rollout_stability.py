"""Rollout-shape diagnostics: does a ramping model get flagged?

The motivating failure: a surrogate posted 0.67 °C RMSE on its own short test
window, then diverged 10 °C in closed loop because it had stopped responding to
its inputs and was emitting a monotone ramp. RMSE cannot see that. These tests
pin the two properties that can:

  - a ramp is flagged even when its RMSE is small,
  - a genuinely monotone *truth* is NOT flagged (the diagnostic is relative),
  - a model that merely *jitters* on a held signal is NOT flagged as ramping.

That last one is the false positive the first version of this diagnostic had.
Staged equipment (fan power, DX power, air flow) holds one value for long
stretches — measured on real test data, 57-96% of its consecutive diffs are
exactly zero — so its true rise fraction is ~0.03. Any continuous model jitters
instead of holding and lands near 0.50, which is a rise excess of ~0.47 while
tracking the signal fine. Rise excess alone therefore flagged three healthy
channels out of eight on a real run.
"""

import pytest
import torch

from neurotemplate.common.metrics import (
    rollout_stability,
    stability_to_dict,
    stability_to_markdown,
)


def _cycle(n: int = 100, amp: float = 2.0) -> torch.Tensor:
    """A cycling signal, like a thermostat-controlled zone temperature."""
    t = torch.linspace(0, 6 * torch.pi, n)
    return (24.0 + amp * torch.sin(t)).reshape(1, n, 1)


def test_perfect_tracking_has_no_excess():
    """Predicting the truth exactly → zero rise excess, drift ratio 1."""
    y = _cycle()
    s = rollout_stability(y, y, ["zone"])[0]
    assert s.name == "zone"
    assert s.pred_rise_fraction == pytest.approx(s.true_rise_fraction)
    assert s.rise_excess == pytest.approx(0.0)
    assert s.drift_ratio == pytest.approx(1.0)


def test_monotone_ramp_is_flagged_despite_small_rmse():
    """The motivating failure: a ramp with a *smaller* RMSE than a good model.

    The ramp is constructed to stay close to the truth in absolute terms, so
    RMSE alone would rank it well. Rise fraction still separates them.
    """
    true = _cycle(n=100, amp=2.0)
    # A gentle ramp through the middle of the band the truth cycles in: it
    # never strays far, so RMSE stays below the truth's own amplitude.
    ramp = torch.linspace(23.5, 24.5, 100).reshape(1, 100, 1)

    ramp_rmse = (ramp - true).pow(2).mean().sqrt()
    assert ramp_rmse < 2.0, "ramp should look acceptable to RMSE"

    s = rollout_stability(ramp, true, ["zone"])[0]
    assert s.pred_rise_fraction == pytest.approx(1.0)
    assert s.true_rise_fraction < 0.6
    assert s.rise_excess > 0.25, "a ramp must exceed the reportcard warn line"
    assert s.drift_ratio > 2.0


def test_monotone_truth_is_not_flagged():
    """A model that correctly tracks a monotone signal must stay unflagged.

    This is why the diagnostic is relative rather than an absolute threshold on
    rise fraction — some channels really do only go up (cumulative energy).
    """
    true = torch.linspace(0.0, 1000.0, 80).reshape(1, 80, 1)
    pred = true + 3.0  # biased but the same shape
    s = rollout_stability(pred, true, ["energy"])[0]
    assert s.pred_rise_fraction == pytest.approx(1.0)
    assert s.true_rise_fraction == pytest.approx(1.0)
    assert s.rise_excess == pytest.approx(0.0)
    assert s.drift_ratio == pytest.approx(1.0)


def test_a_prediction_that_stops_moving_is_flagged():
    """The other degenerate mode, and the one that reads as rise fraction 0
    rather than 1: a model that has stopped responding at all."""
    s = rollout_stability(torch.full((1, 100, 1), 24.0), _cycle(), ["zone"])[0]
    assert s.pred_rise_fraction == pytest.approx(0.0)
    assert s.rise_excess > 0.25
    assert s.looks_like_ramp()


def _staged(n: int = 96) -> torch.Tensor:
    """Staged equipment: holds one value for long stretches, then steps.

    Mirrors the real signature — most consecutive diffs are exactly zero, so the
    true rise fraction is small even though nothing is monotone.
    """
    x = torch.zeros(1, n, 1)
    x[:, n // 3: 2 * n // 3, 0] = 800.0
    x[:, 2 * n // 3:, 0] = 200.0
    return x


def test_jitter_on_a_held_signal_is_not_called_a_ramp():
    """The false positive that shipped first: high excess, but not monotone.

    A continuous model cannot hold a value exactly, so on staged equipment it
    jitters and its rise fraction lands near 0.5 against a truth near 0.03.
    That is a big rise excess and not a ramp, so `looks_like_ramp` must say no.
    """
    true = _staged()
    jitter = true + 40.0 * torch.sin(torch.linspace(0, 60, 96)).reshape(1, 96, 1)

    s = rollout_stability(jitter, true, ["fan power"])[0]
    assert s.true_rise_fraction < 0.10, "truth should mostly hold"
    assert s.rise_excess > 0.25, "excess alone would flag this"
    assert 0.3 < s.pred_rise_fraction < 0.7, "jitter sits mid-range"
    assert not s.looks_like_ramp(), "jitter is not a ramp"


def test_a_real_ramp_on_a_held_signal_is_still_flagged():
    """The fix must not blunt the true positive it was built for."""
    true = _staged()
    ramp = torch.linspace(0.0, 1500.0, 96).reshape(1, 96, 1)

    s = rollout_stability(ramp, true, ["fan power"])[0]
    assert s.pred_rise_fraction == pytest.approx(1.0)
    assert s.looks_like_ramp()


def test_looks_like_ramp_requires_both_conditions():
    """Neither condition alone is sufficient, in both directions."""
    # Monotone prediction AND monotone truth: extreme, but excess is 0.
    true = torch.linspace(0.0, 100.0, 50).reshape(1, 50, 1)
    s = rollout_stability(true + 1.0, true, ["energy"])[0]
    assert s.pred_rise_fraction == pytest.approx(1.0)
    assert not s.looks_like_ramp(), "matching a monotone truth is correct"

    # Thresholds are tunable, and tightening `extreme` un-flags a borderline case.
    ramp = torch.linspace(20.0, 30.0, 96).reshape(1, 96, 1)
    s = rollout_stability(ramp, _cycle(96), ["zone"])[0]
    assert s.looks_like_ramp(excess_warn=0.25)
    assert not s.looks_like_ramp(excess_warn=0.99), "excess gate must apply"


def test_markdown_does_not_flag_staged_equipment():
    """End-to-end: the reportcard table must stay quiet on the jitter case."""
    true = _staged()
    jitter = true + 40.0 * torch.sin(torch.linspace(0, 60, 96)).reshape(1, 96, 1)
    md = stability_to_markdown(rollout_stability(jitter, true, ["fan power"]))
    assert "ramp?" not in md


def test_drift_ratio_is_inf_when_only_prediction_drifts():
    """A drifting prediction against a non-drifting truth is unbounded, not 0/0."""
    true = torch.cat([torch.zeros(1, 50, 1), torch.zeros(1, 50, 1)], dim=1)
    pred = torch.linspace(0.0, 5.0, 100).reshape(1, 100, 1)
    s = rollout_stability(pred, true, ["y"])[0]
    assert s.true_drift == pytest.approx(0.0)
    assert s.drift_ratio == float("inf")

    # ...and the 0/0 case beside it resolves to 1.0 (healthy), not NaN.
    still = torch.zeros(1, 20, 1)
    assert rollout_stability(still, still, ["y"])[0].drift_ratio == pytest.approx(1.0)


def test_averaged_over_windows_not_pooled():
    """One diverging window is not diluted by the step counts of the others.

    Two windows, one perfect and one ramping: the reported rise fraction must
    be the mean of the per-window fractions, so a single bad window moves it.
    """
    good = _cycle(n=60)
    ramp = torch.linspace(20.0, 30.0, 60).reshape(1, 60, 1)
    true = torch.cat([good, good], dim=0)
    pred = torch.cat([good, ramp], dim=0)

    s = rollout_stability(pred, true, ["zone"])[0]
    good_rise = float((good.diff(dim=1) > 0).float().mean())
    assert s.pred_rise_fraction == pytest.approx((good_rise + 1.0) / 2, abs=1e-5)


def test_channels_are_independent():
    """A ramp on one channel does not contaminate a healthy neighbour."""
    true = torch.cat([_cycle(), _cycle()], dim=-1)
    pred = true.clone()
    pred[..., 1] = torch.linspace(20.0, 30.0, 100)

    healthy, ramping = rollout_stability(pred, true, ["ok", "bad"])
    assert healthy.rise_excess == pytest.approx(0.0)
    assert ramping.rise_excess > 0.25


def test_markdown_flags_only_the_ramping_channel():
    true = torch.cat([_cycle(), _cycle()], dim=-1)
    pred = true.clone()
    pred[..., 1] = torch.linspace(20.0, 30.0, 100)

    md = stability_to_markdown(rollout_stability(pred, true, ["ok", "bad"]))
    lines = {row.split("|")[1].strip(): row for row in md.splitlines()[2:]}
    assert "ramp?" not in lines["ok"]
    assert "ramp?" in lines["bad"]


def test_dict_is_json_serializable_and_includes_derived_fields():
    import json

    y = _cycle()
    d = stability_to_dict(rollout_stability(y, y, ["zone"]))
    assert set(d["zone"]) == {
        "pred_rise_fraction", "true_rise_fraction", "rise_excess",
        "pred_drift", "true_drift", "drift_ratio", "looks_like_ramp",
    }
    assert d["zone"]["looks_like_ramp"] is False
    json.dumps(d)  # must not raise


@pytest.mark.parametrize("bad_call", [
    lambda: rollout_stability(torch.zeros(1, 5, 1), torch.zeros(1, 5, 2), ["a"]),
    lambda: rollout_stability(torch.zeros(5, 1), torch.zeros(5, 1), ["a"]),
    lambda: rollout_stability(torch.zeros(1, 5, 2), torch.zeros(1, 5, 2), ["a"]),
    lambda: rollout_stability(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), ["a"]),
])
def test_input_validation(bad_call):
    """Shape mismatch, wrong ndim, name-count mismatch, too-short rollout."""
    with pytest.raises(ValueError):
        bad_call()
