import pytest
import torch

from neurotemplate.control.metrics import (
    actuation_metrics,
    comfort_metrics,
    comfort_to_dict,
    comfort_to_markdown,
    energy_metrics,
    hours_per_step,
)


# ── hours_per_step ───────────────────────────────────────────────


def test_hours_per_step_from_posix_seconds():
    # 300 s sampling -> 5 minutes -> 1/12 h
    t = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1) * 300.0
    assert hours_per_step(t) == pytest.approx(1.0 / 12.0)


def test_hours_per_step_needs_two_steps():
    with pytest.raises(ValueError, match="at least 2 timesteps"):
        hours_per_step(torch.zeros(1, 1, 1))


def test_hours_per_step_rejects_non_increasing_time():
    t = torch.tensor([[[10.0], [10.0]]])
    with pytest.raises(ValueError, match="Non-increasing"):
        hours_per_step(t)


# ── comfort_metrics ──────────────────────────────────────────────


def test_comfort_half_the_horizon_one_degree_above():
    """One channel sitting 1.0 above the band for exactly half of a 2-hour
    horizon: rate 0.5, all of it 'above', mean excursion 1.0, 1.0 degree-hours."""
    n_steps = 8
    step_hours = 2.0 / n_steps  # 2 h total
    y = torch.full((1, n_steps, 1), 22.0)
    y[:, n_steps // 2:, :] = 26.0        # 1.0 above the 25.0 ceiling
    ymin = torch.full((1, n_steps, 1), 20.0)
    ymax = torch.full((1, n_steps, 1), 25.0)

    (m,) = comfort_metrics(y, ymin, ymax, ["zone"], step_hours)

    assert m.name == "zone"
    assert m.violation_rate == pytest.approx(0.5)
    assert m.frac_above == pytest.approx(0.5)
    assert m.frac_below == pytest.approx(0.0)
    assert m.mean_violation == pytest.approx(1.0)
    assert m.max_violation == pytest.approx(1.0)
    # 1.0 excursion held for 1.0 h
    assert m.degree_hours == pytest.approx(1.0)


def test_comfort_separates_below_from_above():
    y = torch.tensor([[[18.0], [26.0], [22.0], [22.0]]])  # 1 below, 1 above
    ymin = torch.full((1, 4, 1), 20.0)
    ymax = torch.full((1, 4, 1), 25.0)

    (m,) = comfort_metrics(y, ymin, ymax, ["zone"], step_hours=1.0)

    assert m.frac_below == pytest.approx(0.25)
    assert m.frac_above == pytest.approx(0.25)
    assert m.violation_rate == pytest.approx(0.5)
    # excursions are 2.0 (below) and 1.0 (above) -> mean 1.5 over violating steps
    assert m.mean_violation == pytest.approx(1.5)
    assert m.max_violation == pytest.approx(2.0)
    assert m.degree_hours == pytest.approx(3.0)


def test_comfort_all_in_band_is_all_zeros():
    y = torch.full((2, 5, 1), 22.0)
    ymin = torch.full((2, 5, 1), 20.0)
    ymax = torch.full((2, 5, 1), 25.0)

    (m,) = comfort_metrics(y, ymin, ymax, ["zone"], step_hours=1.0)

    assert m.violation_rate == 0.0
    assert m.mean_violation == 0.0   # no violating steps: 0, not NaN
    assert m.max_violation == 0.0
    assert m.degree_hours == 0.0


def test_comfort_degree_hours_averages_across_windows():
    """Degree-hours is a per-window integral averaged over windows, so one bad
    window out of two halves the reported figure."""
    y = torch.full((2, 4, 1), 22.0)
    y[0] = 26.0                          # window 0 is 1.0 above throughout
    ymin = torch.full((2, 4, 1), 20.0)
    ymax = torch.full((2, 4, 1), 25.0)

    (m,) = comfort_metrics(y, ymin, ymax, ["zone"], step_hours=0.25)

    # window 0: 4 steps x 1.0 x 0.25 h = 1.0; window 1: 0.0 -> mean 0.5
    assert m.degree_hours == pytest.approx(0.5)
    assert m.violation_rate == pytest.approx(0.5)


def test_comfort_is_per_channel():
    y = torch.stack([
        torch.full((4,), 26.0),   # channel 0 always above
        torch.full((4,), 22.0),   # channel 1 always in band
    ], dim=-1).unsqueeze(0)
    ymin = torch.full((1, 4, 2), 20.0)
    ymax = torch.full((1, 4, 2), 25.0)

    hot, ok = comfort_metrics(y, ymin, ymax, ["hot", "ok"], step_hours=1.0)

    assert hot.violation_rate == pytest.approx(1.0)
    assert ok.violation_rate == pytest.approx(0.0)


def test_comfort_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        comfort_metrics(
            torch.zeros(1, 4, 1), torch.zeros(1, 3, 1), torch.zeros(1, 3, 1),
            ["zone"], 1.0,
        )


def test_comfort_rejects_name_count_mismatch():
    z = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError, match="channel_names"):
        comfort_metrics(z, z, z, ["only_one"], 1.0)


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


def test_energy_rejects_wrong_rank():
    with pytest.raises(ValueError, match="2D"):
        energy_metrics(torch.zeros(2, 3, 1), step_hours=1.0)


def test_energy_cost_is_unset_without_a_tariff():
    assert energy_metrics(torch.ones(2, 4), step_hours=1.0).energy_cost is None


def test_energy_cost_integrates_price_against_power():
    # One window, 4 steps of 0.5 h: 100 kW at $0.10 then 200 kW at $0.40.
    power = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    price = torch.tensor([[0.10, 0.10, 0.40, 0.40]])
    e = energy_metrics(power, step_hours=0.5, price=price)
    expected = (100 * 0.10 + 100 * 0.10 + 200 * 0.40 + 200 * 0.40) * 0.5
    assert e.energy_cost == pytest.approx(expected)


def test_energy_cost_prefers_the_cheap_hours_at_equal_energy():
    """The number the whole feature exists to move: same kWh, different bill."""
    price = torch.tensor([[0.10, 0.10, 0.40, 0.40]])
    early = torch.tensor([[200.0, 200.0, 100.0, 100.0]])
    late = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    assert (energy_metrics(early, 0.5, price).total_energy
            == pytest.approx(energy_metrics(late, 0.5, price).total_energy))
    assert (energy_metrics(early, 0.5, price).energy_cost
            < energy_metrics(late, 0.5, price).energy_cost)


def test_energy_rejects_a_price_that_is_not_co_indexed():
    with pytest.raises(ValueError, match="co-indexed"):
        energy_metrics(torch.ones(2, 4), 1.0, price=torch.ones(2, 3))


# ── renderers ────────────────────────────────────────────────────


def test_comfort_markdown_has_a_row_per_channel():
    y = torch.full((1, 4, 2), 22.0)
    ymin = torch.full((1, 4, 2), 20.0)
    ymax = torch.full((1, 4, 2), 25.0)
    md = comfort_to_markdown(comfort_metrics(y, ymin, ymax, ["a", "b"], 1.0))

    lines = md.splitlines()
    assert len(lines) == 4          # header + separator + 2 channels
    assert lines[2].startswith("| a |")
    assert lines[3].startswith("| b |")


def test_comfort_dict_is_keyed_by_channel_without_name_field():
    y = torch.full((1, 4, 1), 22.0)
    ymin = torch.full((1, 4, 1), 20.0)
    ymax = torch.full((1, 4, 1), 25.0)
    d = comfort_to_dict(comfort_metrics(y, ymin, ymax, ["zone"], 1.0))

    assert set(d) == {"zone"}
    assert "name" not in d["zone"]
    assert "degree_hours" in d["zone"]


# ── MLflow logging ───────────────────────────────────────────────


class _FakeLogger:
    def __init__(self):
        self.metrics = {}
        self.dicts = {}

    def log_metrics(self, flat):
        self.metrics.update(flat)

    def log_dict(self, payload, name):
        self.dicts[name] = payload


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

    logger = _FakeLogger()
    log_control_results(logger, _results())

    assert logger.metrics["control_violation_rate"] == pytest.approx(0.5)
    assert logger.metrics["control_peak_power"] == pytest.approx(1000.0)
    # Channel names are sanitized into MLflow-safe keys (no brackets/degree sign).
    assert "control_zone_C_degree_hours" in logger.metrics
    assert logger.dicts["control_metrics.json"]["learned"]["comfort"]


def test_log_control_results_prefixes_baseline():
    from neurotemplate.control.pipeline import log_control_results

    logger = _FakeLogger()
    log_control_results(logger, _results(0.5, 1000.0),
                        baseline=_results(0.9, 1500.0))

    assert logger.metrics["control_violation_rate"] == pytest.approx(0.5)
    assert logger.metrics["baseline_violation_rate"] == pytest.approx(0.9)
    assert logger.metrics["baseline_peak_power"] == pytest.approx(1500.0)
    assert set(logger.dicts["control_metrics.json"]) == {"learned", "baseline"}


def test_log_control_results_handles_missing_energy_and_logger():
    from neurotemplate.control.pipeline import log_control_results

    r = _results()
    r.energy = None
    logger = _FakeLogger()
    log_control_results(logger, r)
    assert "control_peak_power" not in logger.metrics
    assert logger.dicts["control_metrics.json"]["learned"]["energy"] is None

    log_control_results(None, r)  # must be a no-op, not a crash
