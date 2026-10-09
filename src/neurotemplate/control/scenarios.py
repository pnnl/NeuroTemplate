"""
Closed-loop training/eval data.

A lean control-specific windower. Per window:

    H_sys  = sysid_cfg.in_steps   (frozen thermal model's history window)
    H_ctrl = cfg.in_steps         (policy's history window)
    Hmax   = max(H_sys, H_ctrl)
    P      = pure prediction length (train_pred_steps for train, test_window-Hmax for eval)
    W      = Hmax + P

    yn   : seed = first Hmax real Y steps   [n, Hmax,   ny]  (grows in rollout)
    U    : seed = first Hmax-1 real U steps [n, Hmax-1, nu]  (policy appends)
    D    : full real disturbances           [n, W,   nd]
    t    : full sec_elapsed (for plots)     [n, W,   1]
    R    : the reference columns            [n, W,   n_ref]  (normalized; only
           when the objective names one, see control.losses)

R is read from each split's reference files, joined row for row onto the plant
frame by `load_control_splits`, and windowed by the same `sliding_windows` call
as the rest. It is a key the rollout never writes, so it passes through
SystemPreview untouched and never reaches the dynamics, whose Node reads only
yn, U and D.

The closed-loop System rolls out with start_iter = Hmax - 1 and an explicit
nsteps = P, so predicted yn occupies indices [Hmax : Hmax+P].

U is seeded one step shorter than yn because of where each is written within a
rollout step. SystemPreview reads every key at index i but appends each output
at the end of its buffer. yn is written last and read at the start of the next
step, so it needs Hmax seeded steps. U is written by the policy and read by the
dynamics within the same step, so it needs Hmax - 1: the command the policy
chooses at step i then lands at U[i], and the dynamics applies it immediately
rather than one step late. See `scored_steps` for the resulting alignment.
"""

from __future__ import annotations

import torch
from pandas import DataFrame

from neuromancer.dataset import DictDataset

from neurotemplate.common.config import paths_as_list
from neurotemplate.common.data_utils import (
    NormStats, get_data_splits, load_split, normalize_df, sliding_windows,
)
from neurotemplate.control.losses import (
    ROLE_KEYS, Objective, check_weights, resolve_objective,
)

from neurotemplate._typing import beartype


@beartype
def horizons(cfg, sysid_cfg) -> tuple[int, int, int]:
    """(H_sys, H_ctrl, Hmax)."""
    h_sys = sysid_cfg.in_steps
    h_ctrl = cfg.in_steps
    return h_sys, h_ctrl, max(h_sys, h_ctrl)


@beartype
def eval_pred_steps(cfg, hmax: int) -> int:
    """P for the evaluation rollout. `test_window` counts the hmax warmup, as
    on the sys-id side, so a window no longer than hmax scores nothing."""
    steps = cfg.test_window - hmax
    if steps <= 0:
        raise ValueError(
            f"test_window ({cfg.test_window}) must be > hmax ({hmax}), the "
            "longer of the surrogate's and the policy's in_steps; otherwise "
            "there is no rollout after the warmup."
        )
    return steps


@beartype
def scored_steps(key: str, hmax: int) -> slice:
    """Time slice of `key` covering the P scored steps of a closed-loop rollout.

    Scored step k is the transition into index hmax + k. Every key is read at
    that index except U, which is read one index earlier: U[hmax - 1 + k] is
    the command the policy chose from the signals at hmax - 1 + k, and the
    dynamics applied it in that same rollout step to produce yn[hmax + k].
    Works on tensors and on neuromancer variables alike.
    """
    start = hmax - 1 if key == "U" else hmax
    return slice(start, None)


@beartype
def scored_series(objective: Objective, name: str, data: dict, hmax: int,
                  nm_mapping: dict[str, list[str]], norm_stats: NormStats,
                  physical: bool = True) -> torch.Tensor:
    """One named Y, U, D or reference column over the scored horizon, [b, P].

    Physical units unless `physical` is False. `norm_stats` must carry the
    reference columns (`losses.extend_norm_stats`).
    """
    role = objective.roles[name]
    key = ROLE_KEYS[role]
    j = objective.columns_of(role, nm_mapping).index(name)
    x = data[key][:, scored_steps(key, hmax), j]
    if not physical:
        return x
    stats = norm_stats[name]
    return x * stats["stretch"] + stats["shift"]


def _span(df: DataFrame) -> str:
    return f"{df.index[0]} to {df.index[-1]} ({len(df)} rows)"


def _attach_references(plants: list[DataFrame], split: str, cfg,
                       r_columns: list[str]) -> list[DataFrame]:
    """Each plant frame with its reference file's `r_columns` joined on.

    The pairing is positional and the timestamps must match exactly: a
    reference covering other dates is re-stamped when it is prepared, so
    whether it is aligned to the wall clock or used as a shape is a choice made
    in the data, not one the loader guesses at.
    """
    plant_paths = paths_as_list(getattr(cfg, f"{split}_data_path"))
    ref_paths = paths_as_list(getattr(cfg, f"{split}_reference_path"))
    if len(ref_paths) != len(plant_paths):
        raise ValueError(
            f"{split}_reference_path lists {len(ref_paths)} file(s) but "
            f"{split}_data_path lists {len(plant_paths)}; they pair up one to "
            "one. To pair one plant file with several reference draws, list "
            "the plant file once per draw."
        )
    refs, _ = load_split(ref_paths, f"{split} reference")
    joined = []
    for plant, ref, plant_path, ref_path in zip(plants, refs, plant_paths,
                                                ref_paths):
        if not plant.index.equals(ref.index):
            raise ValueError(
                f"{ref_path} must carry exactly the timestamps of its plant "
                f"file {plant_path}: the reference covers {_span(ref)}, the "
                f"plant file {_span(plant)}."
            )
        missing = [c for c in r_columns if c not in ref.columns]
        if missing:
            raise ValueError(
                f"{ref_path} has no column(s) {missing}, which the objective "
                f"names. Its columns: {[c for c in ref.columns if c != 'sec_elapsed']}"
            )
        clash = [c for c in r_columns if c in plant.columns]
        if clash:
            raise ValueError(
                f"{clash} are columns of both {plant_path} and {ref_path}. A "
                "reference column must come from one place; rename it in one "
                "of them."
            )
        joined.append(plant.join(ref[r_columns]))
    return joined


@beartype
def load_control_splits(
    cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> tuple[list[DataFrame], list[DataFrame], list[DataFrame]]:
    """(train, dev, test) plant frames with their reference columns, normalized.

    One list per split, one frame per plant file. `norm_stats` must carry the
    reference columns (`losses.extend_norm_stats`).
    """
    objective = resolve_objective(cfg, nm_mapping)
    train, dev, test, _ = get_data_splits(cfg)
    splits = []
    for split, plants in (("train", train), ("dev", dev), ("test", test)):
        if objective.r_columns:
            plants = _attach_references(plants, split, cfg, objective.r_columns)
        check_weights(objective, plants, split)
        splits.append([normalize_df(df, norm_stats) for df in plants])
    return splits[0], splits[1], splits[2]


@beartype
def build_window_dict(
    df: DataFrame,
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    P: int,
    stride: int,
) -> dict:
    """Lean closed-loop window dict for one normalized frame (see the module
    docstring for the keys)."""
    _, _, hmax = horizons(cfg, sysid_cfg)
    W = hmax + P
    device = sysid_cfg.device

    Y = sliding_windows(df, nm_mapping["Y"], W, stride, device)
    U = sliding_windows(df, nm_mapping["U"], W, stride, device)
    out = {
        "yn": Y[:, :hmax, :].contiguous(),
        "U": U[:, :hmax - 1, :].contiguous(),   # one shorter: see module docstring
        "D": sliding_windows(df, nm_mapping["D"], W, stride, device),
        "t": sliding_windows(df, nm_mapping["t"], W, stride, device),
    }
    r_columns = resolve_objective(cfg, nm_mapping).r_columns
    if r_columns:
        out["R"] = sliding_windows(df, r_columns, W, stride, device)
    return out


@beartype
def build_window_dict_multi(
    dfs: list[DataFrame],
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    P: int,
    stride: int,
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
        build_window_dict(df, cfg, sysid_cfg, nm_mapping, P, stride)
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
            dfs, cfg, sysid_cfg, nm_mapping, P=cfg.train_pred_steps, stride=h_ctrl,
        )
        ds = DictDataset(data, name=name)
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, collate_fn=ds.collate_fn,
            shuffle=shuffle, generator=gen, drop_last=False,
        )

    return loader(train_dfs, "train", True), loader(dev_dfs, "dev", False)
