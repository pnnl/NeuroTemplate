"""
Tests for the `nt data heads` / `nt data decay` statistics.

`diagnostics.py` is deliberately pure (numpy in, dataclasses/strings out) so
these tests build synthetic arrays instead of a dataset and a config file —
see generalization_report.md §4.2 for what each shape is supposed to mean and
improvement_plans/rc_identification.md §3 for the decay estimator's.

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


def _unimodal(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    return rng.normal(5.0, 2.0, n)


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
    pytest.param(_unimodal, "regression", id="continuous-unimodal"),
    pytest.param(_binary, "binary", id="zero-one-flag"),
    pytest.param(_floor_pinned, "bounded", id="floor-pinned-continuous-tail"),
])
def test_recommend_head_kind_from_shape(make_x, expected_kind):
    rec = recommend_head(make_x(), "col")
    assert rec.kind == expected_kind


def test_ordinal_is_not_driven_by_distinct_value_counting():
    """The motivating failure from §4.2: a staged compressor carries sensor
    noise on every sample, so it has thousands of distinct floats and looks
    continuous by cardinality -- it must still be recognized as staged by
    its shape (modes with empty gaps between them)."""
    x = _staged(n=3000)
    assert len(np.unique(x)) > 2000  # cardinality alone would say "continuous"
    rec = recommend_head(x, "Total DX Power [W]")
    assert rec.kind == "ordinal"


def test_staged_signal_recommends_the_right_number_of_levels():
    rec = recommend_head(_staged(), "Total DX Power [W]")
    assert rec.kind == "ordinal"
    assert "n_levels=3" in rec.snippet
    assert "project_physical_to_norm" in rec.snippet


def test_a_rare_binary_flag_is_flagged_too():
    x = np.array([0.0] * 990 + [1.0] * 10)
    rec = recommend_head(x, "Compressor On/Off")
    assert rec.kind == "binary"
    assert "RARE LEVEL: 1 holds 1.0%" in rec.evidence


def test_bounded_head_bounds_only_the_pinned_side():
    rec = recommend_head(_floor_pinned(), "Solar Irradiance [W/m2]")
    assert rec.kind == "bounded"
    assert "float('inf')" in rec.snippet    # open (continuous) side
    assert "float('-inf')" not in rec.snippet  # the floor is a real bound


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


def test_estimate_decay_reports_no_window_when_runs_are_too_short():
    t_hours, state, ref, _off = _exponential_decay_series()
    off = np.zeros_like(_off)
    off[500:520] = True  # 20 steps, far short of the 48-step minimum
    windows = estimate_decay(t_hours, state, ref, off, min_run_steps=48)
    assert windows == ()
    assert summarize_taus(windows) == (None, None, None)


def test_estimate_decay_skips_a_run_that_is_not_decaying():
    """A run where the gap grows instead of shrinking contributes no window
    rather than a bogus (positive-slope) tau."""
    n = 300
    t_hours = np.arange(n) / 12.0
    ref = np.zeros(n)
    state = 1.0 + 0.01 * np.arange(n)  # gap growing linearly, never decaying
    off = np.ones(n, dtype=bool)
    windows = estimate_decay(t_hours, state, ref, off, min_run_steps=48)
    assert windows == ()


def test_estimate_decay_rejects_a_tau_the_window_could_not_resolve():
    """The `data/spring` failure: a 5-hour window fitting a 127-hour
    constant. The fit is excellent -- over a span that short an exponential
    IS a straight line -- so r² cannot catch it, and the span has to."""
    steps_per_hour = 12.0
    t_hours = np.arange(60) / steps_per_hour        # 5 hours
    ref = np.zeros(60)
    state = 6.0 * np.exp(-t_hours / 127.0)
    off = np.ones(60, dtype=bool)
    assert estimate_decay(t_hours, state, ref, off, min_run_steps=48) == ()


# ── Small pure helpers used to build the decay windows ───────────


def test_actuator_off_mask_true_only_when_every_actuator_is_at_rest():
    u1 = np.array([0.0, 0.01, 5.0, 0.0])
    u2 = np.array([10.0, 10.0, 10.0, 9.0])
    mask = actuator_off_mask({"u1": u1, "u2": u2}, {"u1": 0.0, "u2": 10.0}, tol_frac=0.02)
    # row 0: both at rest. row 1: u1 within tol. row 2: u2 at rest but u1 isn't.
    # row 3: u1 at rest but u2 (9.0) is 1.0 away, tol=0.02*(10.0-9.0)=0.02.
    assert mask.tolist() == [True, True, False, False]
