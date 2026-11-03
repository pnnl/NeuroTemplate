"""
Whiteness of one-step-ahead residuals: what a model has left unexplained.

A model that has captured a channel's dynamics leaves one-step-ahead
prediction errors that are white noise — uncorrelated from one step to the
next, with no excess power at any period. Bacher & Madsen (2011) select
building models with two tests of that, both here:

  - the autocorrelation function (ACF) of the residuals at lags 1..L, each
    against ±1.96/√n, the band a white series stays inside 95% of the time;
  - the cumulated periodogram: the share of the residuals' power below each
    frequency. White noise has a flat spectrum, so its cumulated periodogram
    is a straight line, and the Kolmogorov–Smirnov band ±1.36/√m holds it 95%
    of the time.

The period where the residuals hold the most power says what is missing:
structure under an hour is a fast state or a lag the model lacks, a daily
peak is a daily driver no input carries (occupancy, internal gains), and
power at the longest periods is a slow drift.

Only one-step-ahead residuals can be white. A rollout's error at one step
carries the error of the step before, so a free-run residual is correlated
however good the model is; the rollout is scored by RMSE against persistence
instead.

Pure numpy: residual series in, verdicts out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from neurotemplate._typing import beartype

#: A white series has 5% of its ACF lags outside ±1.96/√n by chance; more
#: than this share is structure.
ACF_OUTSIDE_LIMIT = 0.10
#: Kolmogorov–Smirnov 95% coefficient for the cumulated periodogram.
KS_95 = 1.36

#: Bands the period of largest residual power is grouped into, in hours.
PERIOD_BANDS = (
    ("under 1 h", 0.0, 1.0),
    ("1–6 h", 1.0, 6.0),
    ("6–18 h", 6.0, 18.0),
    ("daily (18–30 h)", 18.0, 30.0),
    ("over 30 h", 30.0, float("inf")),
)


@dataclass(frozen=True)
class Whiteness:
    """Both tests on one channel's one-step-ahead residuals."""
    name: str
    n: int                 # residuals tested
    lag1: float            # ACF at lag 1: how much of an error carries to the next step
    lags: int              # L, the lags tested
    acf_outside: float     # share of lags 1..L outside ±1.96/√n
    cp_deviation: float    # largest |cumulated periodogram − straight line|
    cp_bound: float        # its 95% Kolmogorov–Smirnov bound
    peak_period_h: float   # period of the largest periodogram ordinate

    @property
    def white(self) -> bool:
        return (self.acf_outside <= ACF_OUTSIDE_LIMIT
                and self.cp_deviation <= self.cp_bound)

    @property
    def period_band(self) -> str:
        for label, low, high in PERIOD_BANDS:
            if low <= self.peak_period_h < high:
                return label
        raise ValueError(f"period {self.peak_period_h} h fits no band")


def autocorrelation(e: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample ACF of `e` at lags 1..max_lag, the mean removed."""
    e = e - e.mean()
    denom = float(np.sum(e * e))
    return np.array([float(np.sum(e[:-k] * e[k:])) / denom
                     for k in range(1, max_lag + 1)])


def cumulated_periodogram(e: np.ndarray) -> np.ndarray:
    """Share of the residuals' power at or below each Fourier frequency
    k/n, k = 1..m with m = (n − 1) // 2; the zero frequency (the mean) is
    left out."""
    e = e - e.mean()
    m = (len(e) - 1) // 2
    power = np.abs(np.fft.rfft(e))[1:m + 1] ** 2
    return np.cumsum(power) / power.sum()


@beartype
def whiteness(e: np.ndarray, name: str, step_s: float) -> Whiteness:
    """Both tests on one residual series `e`, sampled every `step_s` seconds.

    The ACF is tested to a day of lags, or a quarter of the series if that is
    shorter: a daily driver is the commonest thing a building model misses.
    """
    n = len(e)
    if n < 16 or not np.std(e) > 0:
        raise ValueError(f"`{name}`: whiteness needs a varying residual series "
                         f"of at least 16 steps, got {n}.")
    lags = max(1, min(round(86400 / step_s), n // 4))
    acf = autocorrelation(e, lags)
    cp = cumulated_periodogram(e)
    m = len(cp)
    line = np.arange(1, m + 1) / m
    power = np.diff(np.concatenate([[0.0], cp]))
    peak = int(np.argmax(power)) + 1          # Fourier index k: period n / k steps
    return Whiteness(
        name=name,
        n=n,
        lag1=float(acf[0]),
        lags=lags,
        acf_outside=float(np.mean(np.abs(acf) > 1.96 / np.sqrt(n))),
        cp_deviation=float(np.max(np.abs(cp - line))),
        cp_bound=KS_95 / np.sqrt(m),
        peak_period_h=n / peak * step_s / 3600,
    )


@beartype
def whiteness_to_dict(results: list[Whiteness]) -> dict:
    return {w.name: {**{k: v for k, v in asdict(w).items() if k != "name"},
                     "white": w.white, "period_band": w.period_band}
            for w in results}
