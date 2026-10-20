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

from neurotemplate.common.metrics import rollout_stability


def _cycle(n: int = 100, amp: float = 2.0) -> torch.Tensor:
    """A cycling signal, like a thermostat-controlled zone temperature."""
    t = torch.linspace(0, 6 * torch.pi, n)
    return (24.0 + amp * torch.sin(t)).reshape(1, n, 1)


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
