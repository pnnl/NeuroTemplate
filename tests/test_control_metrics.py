"""
Tests for the closed-loop metrics: comfort, energy, and the MLflow keys
built on them.

The comfort cases are a table of scenarios, because that is what they are —
one band, one trajectory through it, and the numbers that should fall out.
"""
import pytest
import torch

from neurotemplate.control.metrics import (
    actuation_metrics,
    comfort_metrics,
    energy_metrics,
    hours_per_step,
)
from fake_logger import FakeLogger


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
    # 1.0 above the ceiling for exactly half of a 2 h horizon.
    pytest.param(torch.tensor([[22.0] * 4 + [26.0] * 4]).reshape(1, 8, 1), 0.25,
                 dict(violation_rate=0.5, frac_above=0.5, frac_below=0.0,
                      mean_violation=1.0, max_violation=1.0, degree_hours=1.0),
                 id="half the horizon, one degree above"),
    # One step 2.0 below and one 1.0 above: mean is over violating steps only.
    pytest.param(torch.tensor([[[18.0], [26.0], [22.0], [22.0]]]), 1.0,
                 dict(violation_rate=0.5, frac_above=0.25, frac_below=0.25,
                      mean_violation=1.5, max_violation=2.0, degree_hours=3.0),
                 id="below and above are counted apart"),
    # No violating step must give 0, not a NaN from an empty mean.
    pytest.param(torch.full((2, 5, 1), 22.0), 1.0,
                 dict(violation_rate=0.0, mean_violation=0.0,
                      max_violation=0.0, degree_hours=0.0),
                 id="all in band"),
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


def test_comfort_is_reported_per_channel():
    y = torch.stack([torch.full((4,), 26.0),      # always above
                     torch.full((4,), 22.0)],     # always in band
                    dim=-1).unsqueeze(0)
    ymin, ymax = _band(y.shape)
    hot, ok = comfort_metrics(y, ymin, ymax, ["hot", "ok"], step_hours=1.0)
    assert hot.violation_rate == pytest.approx(1.0)
    assert ok.violation_rate == pytest.approx(0.0)


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


def test_energy_single_spike_moves_peak_but_barely_moves_window_peak():
    """A one-off spike in one window sets peak_power outright, but shifts
    mean_window_peak by only 1/n_windows of the gap — which is the point of
    reporting both."""
    n_windows = 4
    power = torch.full((n_windows, 6), 100.0)
    flat = energy_metrics(power, step_hours=1.0)

    power[0, 3] = 500.0
    spiked = energy_metrics(power, step_hours=1.0)

    assert flat.peak_power == pytest.approx(100.0)
    assert spiked.peak_power == pytest.approx(500.0)
    assert spiked.mean_window_peak == pytest.approx(
        100.0 + (500.0 - 100.0) / n_windows
    )


def test_a_weight_integrates_against_power():
    # One window, 4 steps of 0.5 h: 100 kW at $0.10 then 200 kW at $0.40.
    power = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    price = torch.tensor([[0.10, 0.10, 0.40, 0.40]])
    e = energy_metrics(power, step_hours=0.5, weights={"tariff": price})
    expected = (100 * 0.10 + 100 * 0.10 + 200 * 0.40 + 200 * 0.40) * 0.5
    assert e.weighted_energy["tariff"] == pytest.approx(expected)


def test_a_weight_prefers_the_cheap_hours_at_equal_energy():
    """The number the whole feature exists to move: same kWh, different bill."""
    price = {"tariff": torch.tensor([[0.10, 0.10, 0.40, 0.40]])}
    early = energy_metrics(torch.tensor([[200.0, 200.0, 100.0, 100.0]]), 0.5, price)
    late = energy_metrics(torch.tensor([[100.0, 100.0, 200.0, 200.0]]), 0.5, price)
    assert early.total_energy == pytest.approx(late.total_energy)
    assert early.weighted_energy["tariff"] < late.weighted_energy["tariff"]


# ── MLflow logging ───────────────────────────────────────────────


def _results(violation_rate=0.5, power=1000.0):
    from neurotemplate.common.metrics import per_channel_metrics
    from neurotemplate.control.pipeline import ControlResults

    y = torch.full((2, 4, 1), 26.0)
    ymin = torch.full((2, 4, 1), 20.0)
    ymax = torch.full((2, 4, 1), 25.0)
    u = torch.full((2, 4, 1), 23.0)
    return ControlResults(
        comfort_violation_rate=violation_rate,
        comfort_mean_exceedance=1.0,  # y is constant 26.0, ymax 25.0 -> excursion 1.0 everywhere
        comfort=comfort_metrics(y, ymin, ymax, ["zone [°C]"], 0.25),
        mean_abs_du=0.25,
        actuation=actuation_metrics(u, ["setpoint [°C]"],
                                    torch.tensor([22.0]), torch.tensor([28.0])),
        # A constant band never jumps, so there is nothing to audit.
        response=[],
        tracking=per_channel_metrics(y, 0.5 * (ymin + ymax), ["zone [°C]"]),
        energy=energy_metrics(torch.full((2, 4), power), 0.25),
        eval_time_s=0.1,
    )


def test_log_control_results_flattens_metrics():
    from neurotemplate.control.pipeline import log_control_results

    logger = FakeLogger()
    log_control_results(logger, _results())

    assert logger.metrics["control_violation_rate"] == pytest.approx(0.5)
    assert logger.metrics["control_peak_power"] == pytest.approx(1000.0)
    # Channel names are sanitized into MLflow-safe keys (no brackets/degree sign).
    assert "control_zone_C_degree_hours" in logger.metrics
    assert logger.dicts["control_metrics.json"]["comfort"]
