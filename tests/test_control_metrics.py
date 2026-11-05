"""
Tests for the closed-loop metrics: comfort, energy, and the MLflow keys
built on them.

The comfort cases are a table of scenarios, because that is what they are —
one band, one trajectory through it, and the numbers that should fall out.
"""
import pytest
import torch

from neurotemplate.metrics.control import (
    comfort_metrics,
    energy_metrics,
    hours_per_step,
)


# ── hours_per_step ───────────────────────────────────────────────


def test_hours_per_step_from_posix_seconds():
    # 300 s sampling -> 5 minutes -> 1/12 h
    t = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1) * 300.0
    assert hours_per_step(t) == pytest.approx(1.0 / 12.0)


# ── comfort_metrics ──────────────────────────────────────────────

BAND_LO, BAND_HI = 20.0, 25.0


def _band(shape):
    return torch.full(shape, BAND_LO), torch.full(shape, BAND_HI)


@pytest.mark.parametrize("y,step_hours,expected", [
    # One step 2.0 below and one 1.0 above: mean is over violating steps only.
    pytest.param(torch.tensor([[[18.0], [26.0], [22.0], [22.0]]]), 1.0,
                 dict(violation_rate=0.5, frac_above=0.25, frac_below=0.25,
                      mean_violation=1.5, max_violation=2.0, degree_hours=3.0),
                 id="below and above are counted apart"),
    # Degree-hours is a per-window integral averaged over windows, so one bad
    # window out of two halves the reported figure.
    pytest.param(torch.cat([torch.full((1, 4, 1), 26.0),
                            torch.full((1, 4, 1), 22.0)]), 0.25,
                 dict(violation_rate=0.5, degree_hours=0.5),
                 id="one bad window out of two"),
])
def test_comfort_reads_a_trajectory_through_the_band(y, step_hours, expected):
    ymin, ymax = _band(y.shape)
    (m,) = comfort_metrics(y, ymin, ymax, ["zone"], step_hours)
    assert m.name == "zone"
    for field, value in expected.items():
        assert getattr(m, field) == pytest.approx(value), field


# ── energy_metrics ───────────────────────────────────────────────


def test_energy_mean_total_and_peaks():
    # Two windows of 4 steps at 0.5 h each (2 h per window).
    power = torch.tensor([
        [100.0, 100.0, 100.0, 100.0],
        [200.0, 200.0, 200.0, 200.0],
    ])
    e = energy_metrics(power, step_hours=0.5)

    assert e.mean_power == pytest.approx(150.0)
    # per-window energy: 200 and 400 -> mean 300
    assert e.total_energy == pytest.approx(300.0)
    assert e.peak_power == pytest.approx(200.0)
    assert e.mean_window_peak == pytest.approx(150.0)


def test_a_weight_integrates_against_power():
    # One window, 4 steps of 0.5 h: 100 kW at $0.10 then 200 kW at $0.40.
    power = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    price = torch.tensor([[0.10, 0.10, 0.40, 0.40]])
    e = energy_metrics(power, step_hours=0.5, weights={"tariff": price})
    expected = (100 * 0.10 + 100 * 0.10 + 200 * 0.40 + 200 * 0.40) * 0.5
    assert e.weighted_energy["tariff"] == pytest.approx(expected)
