"""
Tests for the `nt data heads` / `nt data decay` statistics.

`diagnostics.py` is deliberately pure (numpy in, dataclasses/strings out) so
these tests build synthetic arrays instead of a dataset and a config file.

The head-recommendation tests are built around one claim in particular: the
statistic must key off the SHAPE of the distribution (modes with unoccupied
gaps between them), not the number of distinct values. `test_ordinal_is_
not_driven_by_distinct_value_counting` is the test that would fail if
`recommend_head` were secretly doing `len(np.unique(x))` instead.
"""

import math

import numpy as np
import pytest

from neurotemplate.modeling.diagnostics import (
    actuator_off_mask,
    estimate_decay,
    recommend_head,
    summarize_taus,
)


# ── Synthetic column generators ───────────────────────────────────

def _staged(seed=0, n=3000, centers=(0.0, 4210.0, 8070.0), noise=150.0):
    """A staged signal: noise on every sample, so cardinality alone looks
    continuous, but the values cluster around `centers` with empty gaps
    between them -- the DX-compressor motivating case from §4.2."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(centers), n)
    return np.asarray(centers)[labels] + rng.normal(0, noise, n)


def _binary(seed=0, n=500):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n).astype(float)


def _floor_pinned(seed=0, n_zero=400, n_tail=1600):
    """Mass piled at zero, continuous positive tail -- the BoundedHead case."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(n_zero), rng.exponential(5.0, n_tail) + 0.01])
    rng.shuffle(x)
    return x


# ── recommend_head: one shape, one kind ───────────────────────────

@pytest.mark.parametrize("make_x,expected_kind", [
    pytest.param(_staged, "ordinal", id="staged-three-level"),
    pytest.param(_binary, "binary", id="zero-one-flag"),
    pytest.param(_floor_pinned, "bounded", id="floor-pinned-continuous-tail"),
])
def test_recommend_head_kind_from_shape(make_x, expected_kind):
    rec = recommend_head(make_x(), "col")
    assert rec.kind == expected_kind


def test_staged_signal_recommends_the_right_number_of_levels():
    rec = recommend_head(_staged(), "Total DX Power [W]")
    assert rec.kind == "ordinal"
    assert "n_levels=3" in rec.snippet
    assert "project_physical_to_norm" in rec.snippet


# ── estimate_decay: recover a known tau, and say so when there is none ──

def _exponential_decay_series(
    tau_h=8.0, steps_per_hour=12.0, n=2000,
    off_windows=((200, 900), (1200, 1900)), gap0=6.0,
):
    """A slowly-moving reference plus two clean free-decay windows where the
    state's gap to the reference decays exactly like exp(-t/tau_h)."""
    t_hours = np.arange(n) / steps_per_hour
    ref = 20.0 + 5.0 * np.sin(2 * np.pi * t_hours / 24.0)
    state = ref.copy()
    off = np.zeros(n, dtype=bool)
    for start, stop in off_windows:
        t0 = t_hours[start:stop] - t_hours[start]
        state[start:stop] = ref[start:stop] + gap0 * np.exp(-t0 / tau_h)
        off[start:stop] = True
    return t_hours, state, ref, off


def test_estimate_decay_recovers_known_tau():
    tau_true = 8.0
    t_hours, state, ref, off = _exponential_decay_series(tau_h=tau_true)
    windows = estimate_decay(t_hours, state, ref, off, min_run_steps=48)

    assert len(windows) == 2
    for w in windows:
        assert w.tau_h == pytest.approx(tau_true, rel=1e-6)
        assert w.half_life_h == pytest.approx(tau_true * math.log(2), rel=1e-6)
        assert w.r_squared == pytest.approx(1.0, abs=1e-6)

    median, lo, hi = summarize_taus(windows)
    assert median == pytest.approx(tau_true, rel=1e-6)
    assert lo == pytest.approx(tau_true, rel=1e-6)
    assert hi == pytest.approx(tau_true, rel=1e-6)


def test_estimate_decay_fits_only_the_decay_before_the_reference_is_crossed():
    """Overnight decay toward recorded ambient, then the morning warms past
    the zone. The crossing used to be fitted with the decay and pull tau off;
    the decay before it is the measurement."""
    tau_true, n_decay = 8.0, 96                    # 8 h of clean decay at 12/h
    t_hours = np.arange(150) / 12.0
    gap = np.empty(150)
    gap[:n_decay] = 3.0 * np.exp(-t_hours[:n_decay] / tau_true)
    gap[n_decay:] = -0.1 * np.arange(1, 150 - n_decay + 1)   # ambient now above
    ref = np.full(150, 15.0)
    off = np.ones(150, dtype=bool)
    (window,) = estimate_decay(t_hours, ref + gap, ref, off, min_run_steps=48)
    assert window.n_steps == n_decay
    assert window.tau_h == pytest.approx(tau_true, rel=1e-6)


# ── Small pure helpers used to build the decay windows ───────────


def test_actuator_off_mask_true_only_when_every_actuator_is_at_rest():
    u1 = np.array([0.0, 0.01, 5.0, 0.0])
    u2 = np.array([10.0, 10.0, 10.0, 9.0])
    mask = actuator_off_mask({"u1": u1, "u2": u2}, {"u1": 0.0, "u2": 10.0}, tol_frac=0.02)
    # row 0: both at rest. row 1: u1 within tol. row 2: u2 at rest but u1 isn't.
    # row 3: u1 at rest but u2 (9.0) is 1.0 away, tol=0.02*(10.0-9.0)=0.02.
    assert mask.tolist() == [True, True, False, False]


def test_a_latch_shows_up_as_two_columns_and_its_thresholds_and_capacity_are_read_back():
    """A cooling stage that starts 0.3 past its setpoint and stops 0.2 back,
    removing 3 °C/h against a 1.5 °C/h gain: at an error of zero it runs if
    and only if it ran the step before, and the estimates find the three
    numbers. A switch at zero runs on the error alone."""
    import numpy as np
    from neurotemplate.modeling.diagnostics import latch_estimates, latch_table
    rng = np.random.default_rng(0)
    step_h, n = 1 / 12, 20_000
    setpoint = 24.0 + np.repeat(rng.uniform(-1, 1, n // 200), 200)
    zone, running = np.empty(n), np.zeros(n, dtype=bool)
    zone[0] = 24.0
    for k in range(n - 1):
        error = zone[k] - setpoint[k]
        running[k] = error > 0.3 or (k > 0 and running[k - 1] and error > -0.2)
        zone[k + 1] = zone[k] + (1.5 - 3.0 * running[k]) * step_h + rng.normal(0, 0.03)
    error = zone - setpoint
    middle = [r for r in latch_table(running, error) if r.low == 0.0][0]
    assert middle.p_after_running > 0.9 and middle.p_after_idle < 0.1
    estimate = latch_estimates([(running, error, zone)], sign=-1, step_h=step_h)
    assert estimate.start_error[1] == pytest.approx(0.3, abs=0.1)
    assert estimate.stop_error[1] == pytest.approx(-0.2, abs=0.1)
    assert estimate.capacity[1] == pytest.approx(3.0, rel=0.15)
    memoryless = [r for r in latch_table(error > 0, error) if r.low == 0.0][0]
    assert memoryless.p_after_running == memoryless.p_after_idle == 1.0
