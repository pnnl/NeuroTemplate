"""Whiteness of one-step residuals: white noise passes, a carried-over error
and a missing daily driver do not, and the daily one is named as daily."""

import numpy as np

from neurotemplate.modeling.whiteness import whiteness

STEP_S = 300.0  # 5-minute data
N = 288 * 14    # two weeks


def test_white_noise_is_white():
    rng = np.random.default_rng(0)
    w = whiteness(rng.normal(size=N), "noise", STEP_S)
    assert w.white
    assert abs(w.lag1) < 0.05


def test_an_error_carried_to_the_next_step_is_not_white():
    rng = np.random.default_rng(1)
    e = np.zeros(N)
    for t in range(1, N):
        e[t] = 0.8 * e[t - 1] + rng.normal()
    w = whiteness(e, "ar1", STEP_S)
    assert not w.white
    assert 0.75 < w.lag1 < 0.85


def test_a_missing_daily_driver_peaks_at_a_day():
    rng = np.random.default_rng(2)
    t = np.arange(N)
    e = rng.normal(size=N) + 2.0 * np.sin(2 * np.pi * t / 288)
    w = whiteness(e, "gains", STEP_S)
    assert not w.white
    assert w.period_band == "daily (18–30 h)"
    assert abs(w.peak_period_h - 24.0) < 0.5
