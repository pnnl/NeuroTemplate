"""
Closed-loop training/eval data.

A lean control-specific windower. Per window:

    H_sys  = sysid_cfg.history_steps  (frozen thermal model's history window)
    H_ctrl = cfg.history_steps        (policy's deepest `past`, plus the current step)
    Hmax   = max(H_sys, H_ctrl)
    P      = pure prediction length (`horizon` for train, `test_horizon` for eval)
    F      = the deepest `future` in policy_window (`forecast_tail`): every step's
             preview reads real rows of the record, to the rollout's last step
    W      = Hmax + P + F

    yn   : seed = first Hmax real Y steps   [n, Hmax,   ny]  (grows in rollout)
    U    : seed = first Hmax-1 real U steps [n, Hmax-1, nu]  (policy appends)
    D    : full real disturbances           [n, W,   nd]
    t    : full sec_elapsed (for plots)     [n, W,   1]
    R    : the reference columns            [n, W,   n_ref]  (only when the
           objective names one, see control.losses)

Every key is in physical units: the policy normalises its own inputs, and the
surrogate is behind the model contract (`modeling.contract`).

D, t and R run F rows past the rollout so the policy's preview at the last
steps reads real rows, not SystemPreview's `nearest` copy of the final one.
The closed loop trims them back to Hmax + P after the rollout
(`closed_loop.ForecastSystem`), so everything downstream sees P scored steps.

R is the record's columns no role claims that the run names, windowed by the same
`sliding_windows` call as the rest. It is a key the rollout never writes, so it passes
through SystemPreview untouched and never reaches the dynamics, whose Node reads only
yn, U and D. A draw (`cfg.draws`) is the record again with some columns replaced, one
more pass over the training windows.

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

from neurotemplate.nm.dataset import DictDataset

from neurotemplate.common.data_utils import (
    check_complete, get_data, get_data_splits, sliding_windows,
)
from neurotemplate.control.losses import (
    POWER, ROLE_KEYS, Objective, check_weights, resolve_objective,
)

from neurotemplate._typing import beartype


@beartype
def horizons(cfg, sysid_cfg) -> tuple[int, int, int]:
    """(H_sys, H_ctrl, Hmax). Seeding Hmax - 1 recorded commands is what lets
    a policy read `past` of them."""
    h_sys = sysid_cfg.history_steps
    h_ctrl = cfg.history_steps
    return h_sys, h_ctrl, max(h_sys, h_ctrl)


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
                  nm_mapping: dict[str, list[str]]) -> torch.Tensor:
    """One named Y, U, D or reference column over the scored horizon, [b, P]."""
    role = objective.roles[name]
    key = ROLE_KEYS[role]
    j = objective.columns_of(role, nm_mapping).index(name)
    return data[key][:, scored_steps(key, hmax), j]


def _with_draw(path: str, records: list[DataFrame]) -> DataFrame:
    """The record `path`'s timestamps match, with every column they share replaced: the
    same weeks against another band or tariff. Matching no record is a refusal, and so
    is a column the record does not have, which could only be a typo."""
    draw, _ = get_data(path)
    for record in records:
        if record.index.equals(draw.index):
            stray = sorted(set(draw.columns) - set(record.columns))
            if stray:
                raise ValueError(f"draw {path} has column(s) {stray} that its record does not.")
            return record.assign(**{c: draw[c] for c in draw.columns if c != "sec_elapsed"})
    spans = [f"{r.index[0]} to {r.index[-1]} ({len(r)} rows)" for r in records]
    raise ValueError(f"draw {path} covers {draw.index[0]} to {draw.index[-1]} ({len(draw)} rows), "
                     f"which matches no training record: {spans}. A draw carries exactly its "
                     "record's timestamps.")


@beartype
def load_control_splits(
    cfg,
    nm_mapping: dict[str, list[str]],
) -> tuple[list[DataFrame], list[DataFrame], list[DataFrame]]:
    """(train, dev, test) plant frames with their reference columns joined on.

    One list per split, one frame per plant file, physical units.
    """
    objective = resolve_objective(cfg, nm_mapping)
    train, dev, test, _ = get_data_splits(cfg)
    train = train + [_with_draw(path, train) for path in cfg.draws]
    read = [c for role in ("Y", "U", "D") for c in nm_mapping[role]] + objective.r_columns
    for split, frames in (("train", train), ("dev", dev), ("test", test)):
        missing = [c for c in objective.r_columns if c not in frames[0].columns]
        if missing:
            raise ValueError(
                f"{missing} are not columns of the surrogate (Y {nm_mapping['Y']}, U "
                f"{nm_mapping['U']}, D {nm_mapping['D']}) or `{POWER}`, and the {split} "
                f"record has no such column. It has: "
                f"{[c for c in frames[0].columns if c != 'sec_elapsed']}")
        check_complete(frames, read, split)
        check_weights(objective, frames, split)
    return train, dev, test


def shortest_test_rollout(cfg, sysid_cfg, frames: list[DataFrame]) -> int:
    """How many steps the shortest test file holds after a rollout's history and the
    forecast its last step reads."""
    _, _, hmax = horizons(cfg, sysid_cfg)
    return min(len(frame) for frame in frames) - hmax - forecast_tail(cfg)


def forecast_tail(cfg) -> int:
    """F: how far past the rollout's last step the controller reads a forecast. MPC
    previews every D and reference column over its whole horizon."""
    tail = max((entry.get("future", 0) for entry in cfg.window_steps.values()), default=0)
    return max(tail, cfg.horizon_steps) if cfg.policy == "mpc" else tail


@beartype
def build_window_dict(
    df: DataFrame,
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    P: int,
    stride: int,
) -> dict:
    """Lean closed-loop window dict for one frame (see the module docstring
    for the keys)."""
    _, _, hmax = horizons(cfg, sysid_cfg)
    W = hmax + P + forecast_tail(cfg)
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
    """(train_loader, dev_loader), windows starting `cfg.train_stride` and
    `cfg.dev_stride` apart.

    `train_dfs` / `dev_dfs` are always lists (length >= 1). With multiple
    files per split, windows are pooled but no window spans two files.
    """
    gen = torch.Generator(device=sysid_cfg.device)
    gen.manual_seed(cfg.seed)

    def loader(dfs, name, shuffle, stride):
        data = build_window_dict_multi(
            dfs, cfg, sysid_cfg, nm_mapping, P=cfg.horizon_steps, stride=stride,
        )
        ds = DictDataset(data, name=name)
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, collate_fn=ds.collate_fn,
            shuffle=shuffle, generator=gen, drop_last=False,
        )

    return (loader(train_dfs, "train", True, cfg.train_stride_steps),
            loader(dev_dfs, "dev", False, cfg.dev_stride_steps))
