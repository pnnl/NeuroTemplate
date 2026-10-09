"""
Tools for preparing a record before training, run from a notebook or a script.

Plain functions over a pandas Series or DataFrame with a DatetimeIndex:

  - `average(x, step)`              onto an even grid, coarser or ragged
  - `expand(x, step, method)`       onto a finer grid: hold, linear, quadratic
  - `fill_gaps(x, max_gap)`         interpolate NaN runs up to a duration
  - `smooth(s, period=None)`        Kalman smoother, local linear trend
                                    (+ seasonal); needs `neurotemplate[prep]`

The library never calls them. `nt model train` and `nt control train` still
refuse an unevenly spaced timestamp or a NaN — training never invents data —
and see only the finished CSV. Cleaning a record is the user's initiative,
and nothing here reports or checks how it was done.

Durations (`step`, `max_gap`, `period`) are anything `pd.Timedelta` reads:
"15min", "1h", "1D". A duration does not change meaning with the grid.
"""

from __future__ import annotations

from typing import Union

import pandas as pd

Signal = Union[pd.Series, pd.DataFrame]


def _grid_step(index: pd.Index) -> pd.Timedelta:
    """The step of an evenly spaced DatetimeIndex; refuse any other."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("needs a DatetimeIndex with at least two rows.")
    steps = index.to_series().diff().dropna().unique()
    if len(steps) != 1:
        raise ValueError(
            f"needs an evenly spaced index; found {len(steps)} different "
            "steps. Put it on a grid first with `average`.")
    return pd.Timedelta(steps[0])


def _steps_in(duration, step: pd.Timedelta, what: str) -> int:
    """How many grid steps `duration` is; it must be a whole number of them."""
    duration = pd.Timedelta(duration)
    count = duration / step
    if count != int(count) or count < 1:
        raise ValueError(f"{what} ({duration}) must be a whole number of the "
                         f"grid's steps ({step}).")
    return int(count)


def average(x: Signal, step) -> Signal:
    """The mean of each `step`-long bin, on an even grid.

    Coarsens a record (5-minute data to 15 or 60) and puts a ragged one on
    an even grid. A bin with no sample becomes a visible NaN rather than
    disappearing.
    """
    return x.resample(pd.Timedelta(step)).mean()


def expand(x: Signal, step, method: str) -> Signal:
    """Expand an evenly spaced record onto a finer grid of `step`.

    `method`:
      - "hold"       a held quantity (a setpoint, a tariff): each value holds
                     for its whole interval, including the last one.
      - "linear"     a smooth physical quantity, straight between samples.
      - "quadratic"  the same, curved (the cleaning notebook's choice for
                     outdoor temperature). Needs scipy.

    The interpolating methods end at the last sample: there is nothing to
    interpolate toward past it.
    """
    coarse = _grid_step(x.index)
    step = pd.Timedelta(step)
    _steps_in(coarse, step, "the record's step")
    if method == "hold":
        grid = pd.date_range(x.index[0], x.index[-1] + coarse - step, freq=step)
        return x.reindex(grid, method="ffill")
    if method in ("linear", "quadratic"):
        grid = pd.date_range(x.index[0], x.index[-1], freq=step)
        return x.reindex(grid).interpolate(method=method)
    raise ValueError(f"method must be hold, linear or quadratic; got {method!r}.")


def _fill_series(s: pd.Series, max_steps: int) -> pd.Series:
    missing = s.isna()
    run = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(run).transform("sum")
    filled = s.interpolate(method="time", limit_area="inside")
    return filled.mask(missing & (run_length > max_steps))


def fill_gaps(x: Signal, max_gap) -> Signal:
    """Interpolate (linearly in time) every NaN run no longer than `max_gap`.

    A run of k missing samples on a grid of step Δ lasts k·Δ. Longer runs,
    and runs at either end of the record (no sample on one side), stay NaN.
    """
    max_steps = _steps_in(max_gap, _grid_step(x.index), "max_gap")
    if isinstance(x, pd.DataFrame):
        return x.apply(_fill_series, max_steps=max_steps)
    return _fill_series(x, max_steps)


def smooth(s: pd.Series, period=None, pad: int = 5) -> pd.Series:
    """A Kalman-smoothed copy of one evenly spaced signal.

    A local linear trend (statsmodels `UnobservedComponents`, noise
    variances fitted to the data), plus a seasonal component when given a
    `period` (outdoor temperature: "1D"). With no period it is a plain trend
    smoother.

    The seasonal state has one entry per step of the period, so fit on a
    coarse grid (hourly, for a daily cycle) and `expand` afterwards.

    With a period, the fit is warmed up on `pad` periods of padding: the
    first `pad` periods, moved back in time and shifted to meet the first
    sample, so the seasonal component has settled before the record starts.

    It fills gaps as a side effect, and it replaces every value. To patch
    only the gaps:

        s.fillna(smooth(s, period="1D"))
    """
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
    except ImportError as error:
        raise ImportError("smooth needs statsmodels: install "
                          "neurotemplate[prep].") from error

    step = _grid_step(s.index)
    seasonal = None if period is None else _steps_in(period, step, "period")
    padded = s if seasonal is None else _pad(s, seasonal * pad, step)

    model = UnobservedComponents(padded.to_numpy(dtype=float),
                                 level="local linear trend", seasonal=seasonal)
    model.initialize_approximate_diffuse()
    result = model.fit(disp=False)
    values = result.level["smoothed"]
    if seasonal is not None:
        values = values + result.seasonal["smoothed"]
    return pd.Series(values[len(padded) - len(s):], index=s.index, name=s.name)


def _pad(s: pd.Series, length: int, step: pd.Timedelta) -> pd.Series:
    """`s` with its first `length` samples copied in front of it, shifted to
    meet its first sample."""
    if len(s) < length:
        raise ValueError(f"the record has {len(s)} samples; padding needs "
                         f"{length} (pad periods of the season).")
    head = s.iloc[:length]
    offset = s.dropna().iloc[0] - head.dropna().iloc[-1]
    index = head.index - length * step
    return pd.concat([pd.Series(head.to_numpy() + offset, index=index), s])
