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
    price: sampled energy tariff, scaled    [n, W,   1]   (only when the
           `price_signal` hook exists; see `build_price`)

Both the band and the price are *sampled* per window rather than sliced out of
the dataframe: they are exogenous signals the dynamics never sees, and their
windows are deliberately independent of the weather windows.

The closed-loop System rolls out with start_iter = Hmax - 1 and an explicit
nsteps = P, so predicted yn occupies indices [Hmax : Hmax+P].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from pandas import DataFrame

from neuromancer.dataset import DictDataset

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

    The hook returns the tariff in physical units ($/kWh); everything else in
    the window dict is normalized, so the dict carries `price / PRICE_SCALE`
    and the objective multiplies the scale back in when it wants dollars.
    Deliberately a pure scaling with no shift, so a zero price stays a zero
    cost and the cost term stays a per-timestep reweighting of the energy
    gradient rather than a different quantity.
    """
    hooks = load_control_hooks(cfg)
    if getattr(hooks, "price_signal", None) is None:
        return None
    scale = getattr(hooks, "PRICE_SCALE", None)
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
        raise ValueError(
            "control_user_defined.price_signal is defined but PRICE_SCALE is "
            f"missing or not a positive number (got {scale!r}). The hook "
            "returns physical prices; the dict carries price / PRICE_SCALE so "
            "the policy sees an O(1) input."
        )
    return float(scale)


@beartype
def build_price(cfg, n_samples: int, nsteps: int,
                generator=None) -> Optional[torch.Tensor]:
    """A sampled tariff of shape [n_samples, nsteps, 1], scaled, or None.

    Price is exogenous data the policy and the objective read and the dynamics
    never sees, so it travels in the window dict rather than as a D column: a D
    column would be baked into the frozen surrogate's input width, and would
    tell the dynamics model that electricity price drives zone temperature.

    Sampled per window from a real historical series, exactly like the comfort
    band — not derived from `t`. Two reasons, and the second is the one that
    matters:

      - `t` is `sec_elapsed`, which counts from the first row of the file
        rather than from midnight, so a t-derived hour-of-day is an unverifiable
        per-file assumption.
      - A tariff that is a deterministic function of the clock is a tariff the
        policy can learn *as* the clock. D is strongly hour-of-day correlated,
        so "read the price" and "read the time" become the same task and the
        policy posts a convincing load-shifting trace with no price
        responsiveness in it at all. Real day-to-day price variation is what
        breaks that degeneracy.

    The use case draws its own window offsets, so price windows are independent
    of the weather windows they are scored beside. That is deliberate: it
    removes the residual price↔weather correlation as a second route to the
    same shortcut, and it decouples the amount of price data needed from the
    length of the plant record.

    Reproducibility comes from the same `generator` discipline the band uses:
    `evaluate_closed_loop` reseeds from EVAL_BAND_SEED, so the learned policy
    and the baseline are scored against a byte-identical tariff. Price is drawn
    *after* the band so that enabling it leaves band realizations unchanged.
    """
    hooks = load_control_hooks(cfg)
    fn = getattr(hooks, "price_signal", None)
    if fn is None:
        return None
    scale = price_scale(cfg)

    price = fn(n_samples, nsteps, generator)
    if tuple(price.shape) != (n_samples, nsteps, 1):
        raise ValueError(
            f"price_signal must return [{n_samples}, {nsteps}, 1], got "
            f"{tuple(price.shape)}. One price per timestep, per window."
        )
    if not bool(torch.isfinite(price).all()):
        raise ValueError(
            "price_signal returned a non-finite price. Check for gaps in the "
            "source series — a NaN here propagates silently into the cost term."
        )
    # Negative prices are allowed and mean what they say: a market paying to
    # absorb load. The cost term then correctly wants more of it, which is why
    # a non-zero energy or comfort weight alongside it is not optional.
    return price / scale


def _sliding(df: DataFrame, cols: list[str], steps: int, stride: int,
             device) -> torch.Tensor:
    """[n_windows, steps, len(cols)] sliding windows over df rows."""
    x = torch.tensor(df[cols].to_numpy(), dtype=torch.float32, device=device)
    if x.shape[0] < steps:
        raise ValueError(
            f"Split has {x.shape[0]} rows but a control window needs {steps}. "
            "Reduce train_pred_steps/test_window/in_steps or use a longer split."
        )
    starts = torch.arange(0, x.shape[0] - steps + 1, stride, device=device)
    idx = starts[:, None] + torch.arange(steps, device=device)[None, :]
    return x.index_select(0, idx.flatten()).reshape(
        starts.shape[0], steps, x.shape[1]
    )


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
    generator=None,
) -> dict:
    """Lean closed-loop window dict for one normalized split.

    `band` defaults to `cfg.train_band()` if not provided.
    """
    _, _, hmax = horizons(cfg, sysid_cfg)
    W = hmax + P
    device = sysid_cfg.device

    if band is None:
        band = cfg.train_band()

    Y = _sliding(df, nm_mapping["Y"], W, stride, device)
    U = _sliding(df, nm_mapping["U"], W, stride, device)
    D = _sliding(df, nm_mapping["D"], W, stride, device)
    t = _sliding(df, nm_mapping["t"], W, stride, device)
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
    price = build_price(cfg, n, W, generator)
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
            band=band, generator=generator,
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

    def loader(dfs, name, shuffle):
        data = build_window_dict_multi(
            dfs, cfg, sysid_cfg, nm_mapping, norm_stats,
            P=cfg.train_pred_steps, stride=h_ctrl,
            band=cfg.train_band(), generator=gen,
        )
        ds = DictDataset(data, name=name)
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, collate_fn=ds.collate_fn,
            shuffle=shuffle, generator=gen, drop_last=False,
        )

    return loader(train_dfs, "train", True), loader(dev_dfs, "dev", False)
