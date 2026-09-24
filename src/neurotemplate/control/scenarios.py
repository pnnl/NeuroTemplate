"""
Closed-loop training/eval data.

A lean control-specific windower. Per window:

    H_sys  = sysid_cfg.in_steps   (frozen thermal model's history window)
    H_ctrl = cfg.in_steps         (policy's history window)
    Hmax   = max(H_sys, H_ctrl)
    P      = pure prediction length (train_pred_steps for train, test_window-Hmax for eval)
    W      = Hmax + P

    yn   : seed = first Hmax real Y steps   [n, Hmax, ny]  (grows in rollout)
    U    : seed = first Hmax real U steps   [n, Hmax, nu]  (policy appends)
    D    : full real disturbances           [n, W,   nd]
    t    : full sec_elapsed (for plots)     [n, W,   1]
    ymin : comfort lower-band signal        [n, W,   n_ctrl]  (normalized)
    ymax : comfort upper-band signal        [n, W,   n_ctrl]
    price: sampled energy tariff, scaled    [n, W,   1]   (only when
           `cfg.train_price_path` is set; see `build_price`)

Both the band and the price are *sampled* per window rather than sliced out of
the dataframe: they are exogenous signals the dynamics never sees, and their
windows are deliberately independent of the weather windows.

The closed-loop System rolls out with start_iter = Hmax - 1 and an explicit
nsteps = P, so predicted yn occupies indices [Hmax : Hmax+P].
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
from pandas import DataFrame, Timestamp

from neuromancer.dataset import DictDataset

from neurotemplate.common.data_utils import get_data, sliding_windows
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


@dataclass(frozen=True)
class BandSpec:
    """All knobs `comfort_band_signal` needs for one rollout phase."""
    lo: float
    hi: float
    width_min: float
    width_max: float
    sub_samples: int = 1


@beartype
def horizons(cfg, sysid_cfg) -> tuple[int, int, int]:
    """(H_sys, H_ctrl, Hmax)."""
    h_sys = sysid_cfg.in_steps
    h_ctrl = cfg.in_steps
    return h_sys, h_ctrl, max(h_sys, h_ctrl)


@beartype
def price_scale(cfg) -> Optional[float]:
    """The price unit the `price` signal is divided by, or None with no tariff.

    The price CSVs carry physical units ($/kWh); everything else in the
    window dict is normalized, so the dict carries `price / price_scale` and
    the objective multiplies the scale back in when it wants dollars.
    Deliberately a pure scaling with no shift, so a zero price stays a zero
    cost and the cost term stays a per-timestep reweighting of the energy
    gradient rather than a different quantity.
    """
    if cfg.train_price_path is None:
        return None
    scale = cfg.price_scale
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
        raise ValueError(
            "train_price_path is set but price_scale is missing or not a "
            f"positive number (got {scale!r}). Price CSVs carry physical "
            "$/kWh; the dict carries price / price_scale so the policy sees "
            "an O(1) input."
        )
    return float(scale)


@lru_cache(maxsize=None)
def _load_price_series(path: str, column: Optional[str],
                       repeat: int) -> tuple[torch.Tensor, float]:
    """([T, 1] physical price series, seconds past midnight at row 0).

    Cached by path. Reuses `get_data`'s CSV/timestamp convention, so a price
    file follows the same rules as the plant data. `repeat` upsamples a file
    coarser than the plant's timestep (e.g. hourly price over 5-minute plant
    data, repeat=12) onto the plant's per-step grid; the file must already be
    at the plant's resolution or a whole-number multiple coarser than it.

    The second return value is what `price_align: day` needs: a file need not
    begin at midnight, so the offset from the preceding one is what says where
    the day boundaries sit in the series.
    """
    df, _ = get_data(path)
    cols = [c for c in df.columns if c != "sec_elapsed"]
    if column is not None:
        if column not in cols:
            raise ValueError(
                f"price_column {column!r} not found in {path}. Columns: {cols}"
            )
        col = column
    elif len(cols) == 1:
        col = cols[0]
    else:
        raise ValueError(
            f"{path} has more than one non-timestamp column {cols}; set "
            "price_column to pick one."
        )
    series = torch.tensor(df[col].to_numpy(), dtype=torch.float32)
    if not bool(torch.isfinite(series).all()):
        raise ValueError(
            f"{path} contains a non-finite price value. Check for gaps in "
            "the source series — a NaN here propagates silently into the "
            "cost term."
        )
    first = df.index[0]
    midnight_offset = float((first - first.normalize()).total_seconds())
    return series.repeat_interleave(repeat).unsqueeze(-1), midnight_offset


@beartype
def _steps_per_day(t: torch.Tensor) -> Optional[int]:
    """Plant timesteps in 24 h, read off the window's own `sec_elapsed`.

    None when a window is too short to show a timestep, or when the timestep
    does not divide a day — `price_align: day` is the only caller and it says
    so itself rather than silently rounding.
    """
    if t.shape[1] < 2:
        return None
    dt = float(t[0, 1, 0] - t[0, 0, 0])
    if dt <= 0 or 86400.0 % dt != 0:
        return None
    return int(86400.0 / dt)


@beartype
def _window_time_of_day(df: DataFrame, t: torch.Tensor,
                        steps_per_day: int) -> torch.Tensor:
    """Each window's first step, as a step index into the plant's own day.

    `t` is elapsed seconds from the file's first row, so the wall clock comes
    from the dataframe's own DatetimeIndex — the one `get_data` sets.
    """
    origin = df.index[0]
    if not isinstance(origin, Timestamp):
        raise ValueError(
            "price_align: day matches the tariff to the plant's clock, so the "
            f"plant frame needs a DatetimeIndex; this one has {type(origin)}. "
            "Frames from `get_data` carry one. Set price_align: random for a "
            "frame that has no wall clock."
        )
    step_seconds = 86400.0 / steps_per_day
    origin_tod = float((origin - origin.normalize()).total_seconds())
    starts = (origin_tod + t[:, 0, 0].double()) / step_seconds
    return starts.round().long() % steps_per_day


@beartype
def _day_aligned_starts(series_len: int, nsteps: int, steps_per_day: int,
                        phase: int, window_tod: torch.Tensor,
                        generator=None) -> torch.Tensor:
    """Price-series offsets whose time of day matches each window's.

    `phase` is where the price file's row 0 sits within its own day, so price
    index i has time of day `(phase + i) mod steps_per_day`. Solving that
    against each window's own time of day gives the first legal offset, and
    legal offsets then step by a whole day. The *day* is still drawn at
    random; only the time of day is pinned.
    """
    first = (window_tod - phase) % steps_per_day
    n_days = (series_len - nsteps - first) // steps_per_day + 1
    if bool((n_days < 1).any()):
        raise ValueError(
            f"price_align: day needs a price record long enough to hold a "
            f"{nsteps}-step window starting at each plant window's time of "
            f"day; this one has {series_len} steps, which is not enough for "
            f"the window starting at step {int(first[(n_days < 1).nonzero()[0, 0]])} "
            "of its day. Use a longer price record, shorten the window, or "
            "set price_align: random."
        )
    # rand() rather than randint() because the number of legal days differs
    # per window (a later time of day has one fewer whole day behind it).
    draw = torch.rand(window_tod.shape, generator=generator,
                      dtype=torch.float32)
    days = torch.minimum((draw * n_days).long(), n_days - 1)
    return first + days * steps_per_day


@beartype
def build_price(cfg, n_samples: int, nsteps: int, price_path: Optional[str],
                generator=None, steps_per_day: Optional[int] = None,
                window_tod: Optional[torch.Tensor] = None,
                ) -> Optional[torch.Tensor]:
    """A sampled tariff of shape [n_samples, nsteps, 1], scaled, or None.

    Price is exogenous data the policy and the objective read and the dynamics
    never sees, so it travels in the window dict rather than as a D column: a D
    column would be baked into the frozen surrogate's input width, and would
    tell the dynamics model that electricity price drives zone temperature.

    Sampled per window from a real historical series (`price_path` — one CSV
    per split, so the test split can carry price swings the eval should be
    judged on), exactly like the comfort band — not derived from `t`. Two
    reasons, and the second is the one that matters:

      - `t` is `sec_elapsed`, which counts from the first row of the file
        rather than from midnight, so a t-derived hour-of-day is an unverifiable
        per-file assumption.
      - A tariff that is a deterministic function of the clock is a tariff the
        policy can learn *as* the clock. D is strongly hour-of-day correlated,
        so "read the price" and "read the time" become the same task and the
        policy posts a convincing load-shifting trace with no price
        responsiveness in it at all. Real day-to-day price variation is what
        breaks that degeneracy.

    The library draws its own window offsets, so price windows are independent
    of the weather windows they are scored beside. That is deliberate: it
    removes the residual price↔weather correlation as a second route to the
    same shortcut, and it decouples the amount of price data needed from the
    length of the plant record.

    `cfg.price_align` chooses how those offsets are drawn:

      random  (default) — a uniformly random step offset. The tariff's evening
        peak lands at a random hour of the plant's day, so nothing about the
        window's position can be read as a clock.
      day — the offset is chosen so the tariff's time of day matches the
        plant window's own, with the *day* still drawn at random. This is what
        makes pre-cooling legible: an 8 pm peak is drawn at the plant's 8 pm,
        in every window and in every plot. It also partially re-opens the
        shortcut above — a series that peaks at hour 19 on most days is that
        much more learnable as "hour 19" — so a price response measured under
        `day` should be confirmed by re-scoring the trained policy against a
        rewritten tariff (hold the weather, move the peak, watch the load) and
        not by the trajectory alone.

    Reproducibility comes from the same `generator` discipline the band uses:
    `evaluate_closed_loop` reseeds from EVAL_BAND_SEED, so the learned policy
    and the baseline are scored against a byte-identical tariff. Price is drawn
    *after* the band so that enabling it leaves band realizations unchanged.
    """
    if price_path is None:
        return None
    scale = price_scale(cfg)
    assert scale is not None  # price_path set => cfg.train_price_path set => scale checked above
    series, midnight_offset = _load_price_series(
        price_path, cfg.price_column, cfg.price_repeat,
    )
    if series.shape[0] < nsteps:
        raise ValueError(
            f"{price_path} has {series.shape[0]} steps but a window needs "
            f"{nsteps}. Use a longer price record or shorten train_pred_steps."
        )
    if cfg.price_align == "day":
        if steps_per_day is None or window_tod is None:
            raise ValueError(
                "price_align: day needs a plant timestep that divides 24 h "
                "evenly, so the tariff's clock can be matched to the plant's. "
                "This run's timestep does not (or the window is too short to "
                "show one). Set price_align: random."
            )
        if window_tod.shape[0] != n_samples:
            raise ValueError(
                f"window_tod has {window_tod.shape[0]} entries but "
                f"{n_samples} windows were asked for; it carries one time of "
                "day per window."
            )
        phase = round(midnight_offset * steps_per_day / 86400.0)
        starts = _day_aligned_starts(series.shape[0], nsteps, steps_per_day,
                                     phase, window_tod, generator)
    else:
        starts = torch.randint(0, series.shape[0] - nsteps + 1, (n_samples,),
                               generator=generator)
    idx = starts[:, None] + torch.arange(nsteps)[None, :]
    # Negative prices are allowed and mean what they say: a market paying to
    # absorb load. The cost term then correctly wants more of it, which is why
    # a non-zero energy or comfort weight alongside it is not optional.
    return series[idx] / scale


@beartype
def build_window_dict(
    df: DataFrame,
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    P: int,
    stride: int,
    band: Optional[BandSpec] = None,
    price_path: Optional[str] = None,
    generator=None,
) -> dict:
    """Lean closed-loop window dict for one normalized split.

    `band` defaults to `cfg.train_band()` and `price_path` to
    `cfg.train_price_path` if not provided — a caller building dev/test
    windows passes its own split's path explicitly (see `build_loaders` and
    `pipeline.evaluate_closed_loop`).
    """
    _, _, hmax = horizons(cfg, sysid_cfg)
    W = hmax + P
    device = sysid_cfg.device

    if band is None:
        band = cfg.train_band()
    if price_path is None:
        price_path = cfg.train_price_path

    Y = sliding_windows(df, nm_mapping["Y"], W, stride, device)
    U = sliding_windows(df, nm_mapping["U"], W, stride, device)
    D = sliding_windows(df, nm_mapping["D"], W, stride, device)
    t = sliding_windows(df, nm_mapping["t"], W, stride, device)
    n = Y.shape[0]

    ymin, ymax = load_control_hooks(cfg).comfort_band_signal(
        nm_mapping, norm_stats, n_samples=n, nsteps=W, band=band,
        generator=generator,
    )

    out = {
        "yn": Y[:, :hmax, :].contiguous(),
        "U": U[:, :hmax, :].contiguous(),
        "D": D,
        "t": t,
        "ymin": ymin.to(device),
        "ymax": ymax.to(device),
    }

    # An extra key the rollout does not write passes straight through
    # SystemPreview.forward, so the tariff is readable in the loss as
    # variable("price") and in `out["price"]` at evaluation time. Drawn after
    # the band, so turning price on does not shift the band realization.
    steps_per_day = _steps_per_day(t)
    tod = (None if steps_per_day is None or cfg.price_align != "day"
           else _window_time_of_day(df, t, steps_per_day))
    price = build_price(cfg, n, W, price_path, generator,
                        steps_per_day=steps_per_day, window_tod=tod)
    if price is not None:
        out["price"] = price.to(device)
    return out


@beartype
def build_window_dict_multi(
    dfs: list[DataFrame],
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    P: int,
    stride: int,
    band: Optional[BandSpec] = None,
    price_path: Optional[str] = None,
    generator=None,
) -> dict:
    """Per-file windows pooled into one dict, with a `source_idx` lookup.

    No window crosses a file boundary because each df is windowed in
    isolation. Single-file inputs (a length-1 list) produce identical
    tensors to `build_window_dict`, plus an all-zeros `source_idx`.
    """
    if not dfs:
        raise ValueError("build_window_dict_multi requires at least one dataframe")
    device = sysid_cfg.device

    parts = [
        build_window_dict(
            df, cfg, sysid_cfg, nm_mapping, norm_stats, P, stride,
            band=band, price_path=price_path, generator=generator,
        )
        for df in dfs
    ]
    out = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    out["source_idx"] = torch.cat([
        torch.full((p["yn"].shape[0],), i, dtype=torch.long, device=device)
        for i, p in enumerate(parts)
    ])
    return out


@beartype
def build_loaders(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    train_dfs: list[DataFrame],
    dev_dfs: list[DataFrame],
):
    """(train_loader, dev_loader). Stride = H_ctrl for unique starts.

    `train_dfs` / `dev_dfs` are always lists (length >= 1). With multiple
    files per split, windows are pooled but no window spans two files.
    """
    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    gen = torch.Generator(device=sysid_cfg.device)
    gen.manual_seed(cfg.seed)

    def loader(dfs, name, shuffle, price_path):
        data = build_window_dict_multi(
            dfs, cfg, sysid_cfg, nm_mapping, norm_stats,
            P=cfg.train_pred_steps, stride=h_ctrl,
            band=cfg.train_band(), price_path=price_path, generator=gen,
        )
        ds = DictDataset(data, name=name)
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, collate_fn=ds.collate_fn,
            shuffle=shuffle, generator=gen, drop_last=False,
        )

    return (loader(train_dfs, "train", True, cfg.train_price_path),
            loader(dev_dfs, "dev", False, cfg.dev_price_path))
