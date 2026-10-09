"""The data-preparation tools, each on a synthetic series with a known answer.

The library never calls these; they are what a user runs on a record before
the loader sees it.
"""

import numpy as np
import pandas as pd
import pytest

from neurotemplate.prep import average, expand, fill_gaps, smooth


def _series(values, freq="5min", start="2025-07-01 06:35"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq),
                     dtype=float)


def test_average_coarsens_and_shows_a_missing_timestamp_as_nan():
    s = _series(np.arange(12.0))                   # 06:35 .. 07:30, 5-minute
    ragged = s.drop(s.index[8:11])                 # 07:15 .. 07:25 never recorded
    coarse = average(ragged, "15min")
    assert coarse.index.freq == "15min"
    at = lambda hhmm: coarse[pd.Timestamp(f"2025-07-01 {hhmm}")]  # noqa: E731
    assert at("06:30") == pytest.approx(np.mean([0, 1]))
    assert at("06:45") == pytest.approx(np.mean([2, 3, 4]))
    assert at("07:00") == pytest.approx(np.mean([5, 6, 7]))
    assert np.isnan(at("07:15"))
    assert at("07:30") == pytest.approx(11.0)


def test_expand_holds_a_tariff_over_its_whole_interval_and_interpolates_a_ramp():
    hourly = _series([0.1, 0.4, 0.2], freq="1h", start="2025-07-01 00:00")
    held = expand(hourly, "15min", "hold")
    assert held.tolist() == [0.1] * 4 + [0.4] * 4 + [0.2] * 4

    ramp = _series([0.0, 4.0, 8.0], freq="1h", start="2025-07-01 00:00")
    for method in ("linear", "quadratic"):
        fine = expand(ramp, "15min", method)
        assert fine.tolist() == pytest.approx(list(np.arange(9.0)))

    with pytest.raises(ValueError, match="whole number"):
        expand(hourly, "25min", "hold")


def test_fill_gaps_fills_short_runs_only_and_never_past_the_ends():
    s = _series(np.arange(20.0))
    s.iloc[[3]] = np.nan                # 5 minutes: filled
    s.iloc[[7, 8]] = np.nan             # 10 minutes: filled
    s.iloc[[12, 13, 14]] = np.nan       # 15 minutes: too long, stays
    s.iloc[[0, 19]] = np.nan            # at the ends: nothing on one side
    filled = fill_gaps(s, "10min")
    assert filled.iloc[[3, 7, 8]].tolist() == pytest.approx([3.0, 7.0, 8.0])
    assert filled.iloc[[12, 13, 14, 0, 19]].isna().all()


def test_smooth_recovers_a_noisy_daily_cycle_and_fills_its_gap():
    rng = np.random.default_rng(0)
    hours = np.arange(24 * 14)
    truth = 20.0 + 0.02 * hours + 5.0 * np.sin(2 * np.pi * hours / 24)
    s = _series(truth + rng.normal(0, 0.5, len(hours)), freq="1h")
    s.iloc[200:206] = np.nan

    smoothed = smooth(s, period="1D")
    assert smoothed.index.equals(s.index)
    assert np.sqrt(np.mean((smoothed - truth) ** 2)) < 0.3   # below the noise
    assert np.abs(smoothed.iloc[200:206] - truth[200:206]).max() < 1.0
    patched = s.fillna(smoothed)
    assert patched.notna().all() and patched.iloc[10] == s.iloc[10]


def test_smooth_without_a_period_is_a_trend_smoother():
    rng = np.random.default_rng(1)
    truth = np.linspace(10.0, 14.0, 200)
    smoothed = smooth(_series(truth + rng.normal(0, 0.3, 200)))
    assert np.sqrt(np.mean((smoothed - truth) ** 2)) < 0.15
