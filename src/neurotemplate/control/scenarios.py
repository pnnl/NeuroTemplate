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

    return {
        "yn": Y[:, :hmax, :].contiguous(),
        "U": U[:, :hmax, :].contiguous(),
        "D": D,
        "t": t,
        "ymin": ymin.to(device),
        "ymax": ymax.to(device),
    }


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
