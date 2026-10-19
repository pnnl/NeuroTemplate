import json
import re
import warnings
from typing import Optional, TypedDict

import pandas as pd
import torch
from neuromancer.dataset import DictDataset

from neurotemplate.common.config import paths_as_list

from neurotemplate._typing import beartype


class ColumnStats(TypedDict):
    """One column's entry in `norm_stats`; see `compute_norm_stats`."""
    norm_type: str          # "min_max" | "abs_max" | "z_score" | "none"
    shift: float
    stretch: float
    min: float
    max: float
    mean: float
    stddev: float
    median: float
    unit: Optional[str]     # `parse_column_unit(name)`


#: Per-column normalisation stats, keyed by column name. Computed once from
#: the training split, persisted with the run as `norm_stats.json`.
NormStats = dict[str, ColumnStats]


# ---------------------- Data utilities ---------------------- #

@beartype
def get_data(path: str) -> tuple[pd.DataFrame, float]:
    """Load one CSV onto a regular DatetimeIndex, with a `sec_elapsed` column.

    Returns `(df, timestep_seconds)`. The time column is auto-detected by name.
    The timestep is the most common spacing between rows, and every row must
    keep it: the loader neither fills nor splits at a gap, so a missing or
    out-of-order timestamp is refused with where it happens.
    """
    df = pd.read_csv(path)

    time_candidates = [
        "timestamp", "time", "date", "datetime"
    ]

    t_col = next((c for c in df.columns if c.lower() in time_candidates), None)

    if t_col is None:
        raise ValueError("Could not auto-detect time column in dataset.")

    parsed_ts = pd.to_datetime(df[t_col], errors="coerce")
    if parsed_ts.isna().any():
        bad_examples = df.loc[parsed_ts.isna(), t_col].astype(str).head(5).tolist()
        raise ValueError(
            f"Timestamps in column '{t_col}' are not in a compatible datetime format. "
            f"Example invalid values: {bad_examples}"
        )

    df["timestamp"] = parsed_ts

    # Drop time column if it doesn't have name "timestamp", this doesn't
    # change the base dataset just standardizes the data in this flow
    if t_col != "timestamp":
        df = df.drop(columns=t_col, axis=1)

    spacing = df["timestamp"].diff().iloc[1:]
    if spacing.empty:
        raise ValueError(f"{path}: need at least two rows to find the timestep.")
    backwards = spacing[spacing <= pd.Timedelta(0)]
    if not backwards.empty:
        row = backwards.index[0]
        raise ValueError(
            f"{path}: timestamps must be strictly increasing; "
            f"{df['timestamp'][row]} does not come after the row before it "
            f"({len(backwards)} such row(s))."
        )
    step = spacing.mode().iloc[0]
    off_step = spacing[spacing != step]
    if not off_step.empty:
        row = off_step.index[0]
        raise ValueError(
            f"{path}: timestamps are not evenly spaced. The timestep is {step}, "
            f"but {len(off_step)} row(s) break it, the first at "
            f"{df['timestamp'][row]} ({spacing[row]} after the row before). "
            "Gaps are not filled or split automatically: repair the record, or "
            "split the file at each gap and pass the pieces as a list of paths."
        )
    freq_seconds = step.total_seconds()

    df = df.set_index("timestamp")

    # get datetime in seconds from epoch
    # to undo this, run pd.to_datetime(1e9*df['time_sec'])+start_time
    df['sec_elapsed'] = [date.timestamp() for date in df.index]
    df['sec_elapsed'] -= df['sec_elapsed'].iloc[0]

    return df, freq_seconds


def _align_columns_to_train(
    train_cols: set,
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Reject missing columns, warn-and-drop extras. Returns the aligned df."""
    cols = set(df.columns)
    if cols == train_cols:
        return df
    missing = sorted(train_cols - cols)
    extra = sorted(cols - train_cols)
    if missing:
        raise ValueError(
            f"{label} columns do not match train columns. "
            f"Missing: {missing or 'none'}, extra: {extra or 'none'}"
        )
    warnings.warn(
        f"{label} columns do not match train. "
        f"Extra columns being dropped: {extra}"
    )
    return df.drop(columns=extra, axis=1)


def _validate_split_columns(
    train_dfs: list[pd.DataFrame],
    dev_dfs: list[pd.DataFrame],
    test_dfs: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Enforce column compatibility across every train/dev/test file.

    Train files must all match `train_dfs[0]` exactly. Dev and test files
    may carry extra columns (dropped with a warning) but must not be
    missing any. Returns the column-aligned dev and test lists.
    """
    train_cols = set(train_dfs[0].columns)
    for i, df in enumerate(train_dfs[1:], start=1):
        if set(df.columns) != train_cols:
            missing = sorted(train_cols - set(df.columns))
            extra = sorted(set(df.columns) - train_cols)
            raise ValueError(
                f"train[{i}] columns do not match train[0]. "
                f"Missing: {missing or 'none'}, extra: {extra or 'none'}"
            )
    dev_dfs = [
        _align_columns_to_train(train_cols, df, f"dev[{i}]")
        for i, df in enumerate(dev_dfs)
    ]
    test_dfs = [
        _align_columns_to_train(train_cols, df, f"test[{i}]")
        for i, df in enumerate(test_dfs)
    ]
    return dev_dfs, test_dfs


def _warn_timestamp_overlap(
    train_idx: pd.Index,
    other_idx: pd.Index,
    label: str,
) -> None:
    overlap = train_idx.intersection(other_idx)
    if len(overlap) > 0:
        warnings.warn(
            f"{len(overlap)} timestamps from training found in {label} data.",
            stacklevel=2,
        )


def load_split(paths: list[str], name: str) -> tuple[list[pd.DataFrame], float]:
    """Read every CSV in `paths` and return (dfs, shared_freq_seconds).

    Each file must have the same sampling frequency and no NaNs. Returns
    a list of one df for the single-path case so callers always iterate.
    """
    if not paths:
        raise ValueError(f"{name} split has no data paths.")
    dfs: list[pd.DataFrame] = []
    freq: Optional[float] = None
    for i, path in enumerate(paths):
        df, f = get_data(path)
        if df.isna().any().any():
            raise ValueError(f"NaNs found in {name}[{i}] ({path})")
        if freq is None:
            freq = f
        elif f != freq:
            raise ValueError(
                f"{name}[{i}] freq ({f}) != {name}[0] freq ({freq}) "
                f"(path: {path})"
            )
        dfs.append(df)
    return dfs, freq


@beartype
def get_data_splits(cfg) -> tuple[
    list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], float
]:
    """Load train/dev/test CSVs (one or many per split) and validate.

    `cfg.{train,dev,test}_data_path` may each be a single path string or
    a list of path strings. Returns three lists of dataframes (length
    >= 1) plus the shared sampling frequency. Downstream code iterates
    the lists so the single-file case requires no special handling.
    """
    train_dfs, train_freq = load_split(paths_as_list(cfg.train_data_path), "train")
    dev_dfs, dev_freq = load_split(paths_as_list(cfg.dev_data_path), "dev")
    test_dfs, test_freq = load_split(paths_as_list(cfg.test_data_path), "test")

    if train_freq != dev_freq:
        raise ValueError(f"Train freq != dev freq: {train_freq} vs {dev_freq}")
    if train_freq != test_freq:
        raise ValueError(f"Train freq != test freq: {train_freq} vs {test_freq}")

    dev_dfs, test_dfs = _validate_split_columns(train_dfs, dev_dfs, test_dfs)

    # Pairwise overlap warnings so a multi-file split flags every offending
    # combination rather than silently picking the first one.
    for ti, t_df in enumerate(train_dfs):
        for di, d_df in enumerate(dev_dfs):
            _warn_timestamp_overlap(t_df.index, d_df.index, f"dev[{di}] vs train[{ti}]")
        for tei, te_df in enumerate(test_dfs):
            _warn_timestamp_overlap(t_df.index, te_df.index, f"test[{tei}] vs train[{ti}]")

    return train_dfs, dev_dfs, test_dfs, train_freq


# Build dense gradient sign masks from the provided mapping file.
def get_grad_masks(
    grad_path: Optional[str],
    nm_mapping: dict[str, list[str]],
) -> dict[str, torch.Tensor]:
    """
    Args:
        grad_path: JSON mapping path that enables/defines gradient constraints (None disables).
        nm_mapping: Column mapping that defines Y/U/D indices for mask construction.

    Returns dense sign masks for gradient constraints,
    entries are all {-1,0,+1}

    Output:
        grad_mask["Y"]: (1, 1, ny, ny)  with entries in {-1,0,+1}
        grad_mask["U"]: (1, 1, ny, nu)
        grad_mask["D"]: (1, 1, ny, nd)

    Keys naming a Y channel this mapping does not have are skipped with a
    warning, so one grad file can serve several mappings; an unknown
    dependency column under a known key is an error.
    """
    if grad_path is None:
        return {}

    with open(grad_path, "r", encoding="utf-8") as f:
        key_mapping = json.load(f)

    Y_names = nm_mapping["Y"]
    U_names = nm_mapping["U"]
    D_names = nm_mapping["D"]

    ny, nu, nd = len(Y_names), len(U_names), len(D_names)

    grad_masks = {}

    skipped = [k for k in key_mapping if k not in Y_names]
    if skipped:
        warnings.warn(
            f"{grad_path} constrains columns that are not Y channels of this "
            f"mapping; ignored: {skipped}",
            stacklevel=2,
        )

    for key_i, deps in key_mapping.items():
        if key_i not in Y_names:
            continue

        i = Y_names.index(key_i)

        # deps is expected to be a dict: {key_j: sign, ...}
        for key_j, sign in deps.items():

            if key_j in Y_names:
                yud = "Y"
                j = Y_names.index(key_j)
                nx = ny  # In case the default tensor needs to be built
            elif key_j in U_names:
                yud = "U"
                j = U_names.index(key_j)
                nx = nu  # In case the default tensor needs to be built
            elif key_j in D_names:
                yud = "D"
                j = D_names.index(key_j)
                nx = nd  # In case the default tensor needs to be built
            else:
                raise ValueError(f"Grad key {key_j} not in Y/U/D.")

            if sign not in (-1, 1):
                raise ValueError(
                    f"grad constraint for ({key_i}, {key_j}) must be -1/1, got {sign}"
                )

            if yud not in grad_masks:
                grad_masks[yud] = torch.zeros((1, 1, ny, nx), dtype=torch.int8)

            grad_masks[yud][0, 0, i, j] = sign

    return grad_masks


def sliding_windows(
    df: pd.DataFrame, cols: list[str], steps: int, stride: int, device,
) -> torch.Tensor:
    """[n_windows, steps, len(cols)] windows over `df[cols]`, one every `stride` rows.

    Raises when `df` is shorter than one window rather than returning an empty
    tensor that fails somewhere less obvious later.
    """
    x = torch.tensor(df[cols].to_numpy(), dtype=torch.float32, device=device)
    if x.shape[0] < steps:
        raise ValueError(
            f"Data has {x.shape[0]} rows but a window needs {steps}. Use a "
            "longer file or shorten the window (train_pred_steps/test_window/"
            "history_steps)."
        )
    # Row w of `idx` holds the row numbers of window w: its start plus 0..steps-1.
    starts = torch.arange(0, x.shape[0] - steps + 1, stride, device=device)
    idx = starts[:, None] + torch.arange(steps, device=device)[None, :]
    return x.index_select(0, idx.flatten()).reshape(starts.shape[0], steps, x.shape[1])


@beartype
def get_sw_dict(
    df: pd.DataFrame,
    cfg,
    nm_mapping: dict[str, list[str]],
    steps: Optional[int] = None,
    stride: Optional[int] = None,
) -> dict:
    """Sliding-window Y/U/D/t tensors for one normalized dataframe.

    Args:
        df: Full normalized dataframe used to build sliding windows.
        cfg: provides history_steps/train_pred_steps and device placement.
        nm_mapping: Column mapping that defines Y/U/D/t extraction order.
        steps: window length. None uses history_steps + train_pred_steps; a
            negative value means one window spanning the whole frame.
        stride: rows between window starts. None uses history_steps.
    """
    if steps is None:
        steps = cfg.history_steps + cfg.train_pred_steps
    elif steps < 0:
        steps = df.shape[0]
    stride = stride or cfg.history_steps

    Y, U, D, t = [sliding_windows(df, nm_mapping[key], steps, stride, cfg.device)
                  for key in ("Y", "U", "D", "t")]
    return {"t": t, "Y": Y, "U": U, "D": D, "yn": Y[:, :cfg.history_steps, :]}


@beartype
def get_sw_dict_multi(
    dfs: list[pd.DataFrame],
    cfg,
    nm_mapping: dict[str, list[str]],
    steps: Optional[int] = None,
    stride: Optional[int] = None,
) -> dict:
    """Build per-file sliding windows and concatenate along the batch dim.

    No window ever spans two files: each df is windowed in isolation via
    `get_sw_dict`, then the resulting [n_windows, steps, features]
    tensors are stacked on dim 0. Also returns a `source_idx`
    [n_windows] tensor giving the originating df index for each window,
    used downstream for per-file plot routing.

    Single-file inputs (a length-1 list) produce tensors identical to
    `get_sw_dict(dfs[0], ...)`, plus an all-zeros `source_idx`. Multi-file
    inputs require every df to be long enough to support `steps` —
    heterogeneous window lengths would prevent concatenation, so fail
    fast at the first too-short file.
    """
    if not dfs:
        raise ValueError("get_sw_dict_multi requires at least one dataframe")

    # Resolve `steps` once so the per-file check sees the actual window length.
    if steps is None:
        resolved_steps = cfg.history_steps + cfg.train_pred_steps
    elif steps < 0:
        # Per-file "full data window" only makes sense for one df.
        if len(dfs) > 1:
            raise ValueError(
                "steps<0 (full-window) is ambiguous with multiple files; "
                "pass an explicit window length."
            )
        resolved_steps = dfs[0].shape[0]
    else:
        resolved_steps = steps

    for i, df in enumerate(dfs):
        if df.shape[0] < resolved_steps:
            raise ValueError(
                f"File {i} has {df.shape[0]} rows but window length is "
                f"{resolved_steps}. Use a longer file or shorten the "
                "window (train_pred_steps/test_window/history_steps)."
            )

    parts = [get_sw_dict(df, cfg, nm_mapping, resolved_steps, stride) for df in dfs]
    out = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    out["source_idx"] = torch.cat([
        torch.full(
            (p["Y"].shape[0],), i, dtype=torch.long, device=cfg.device,
        )
        for i, p in enumerate(parts)
    ])
    return out


# Build a DictDataset DataLoader for a split.
@beartype
def get_loader(
    dfs: list[pd.DataFrame],
    cfg,
    nm_mapping: dict[str, list[str]],
    name: str,
    steps: Optional[int] = None,
):
    """
    Args:
        dfs: One or more normalized dataframes for this split. Each is
            windowed in isolation and the resulting tensors are stacked,
            so no window spans two source files.
        cfg providing batch_size, device placement, and RNG seed.
        nm_mapping: Column mapping that defines Y/U/D/t extraction order.
        name: Split name stored on the DictDataset for logging/debugging.
        steps: Optional override for window length (None uses history_steps + train_pred_steps).
    """

    data = get_sw_dict_multi(dfs, cfg, nm_mapping, steps)

    dataset = DictDataset(data, name=name)

    # On cfg.device, which resolve_device also made torch's default device:
    # the sampler's randperm runs there, and needs a generator that matches.
    generator = torch.Generator(device=cfg.device)
    generator.manual_seed(cfg.seed)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size,
        collate_fn=dataset.collate_fn, shuffle=True,
        generator=generator,
        drop_last=False,
    )

    return loader


# Convert windowed tensors back to a DataFrame (optionally unnormalize).
@beartype
def to_df(
    data: dict,
    norm_stats: NormStats,
    nm_mapping: dict[str, list[str]],
    unnorm: Optional[bool] = False,
):
    """
    Args:
        data: Windowed tensor dict (Y/U/D/t) to flatten into columns.
        norm_stats: Shift/stretch stats used to undo normalization if requested.
        nm_mapping: Column mapping that defines output column ordering.
        unnorm: If True, applies unnormalize_df() before returning the DataFrame.
    """

    df_dict = {}
    for key in data:
        assert key in nm_mapping, f"{key} not present in neuromancer mapping!"

        for (idx, col) in enumerate(nm_mapping[key]):
            # Data is assumed to either be: (batch, timesteps, features)
            #                           or: (timesteps, features)
            if data[key].ndim == 3:
                df_dict[col] = data[key][:, :, idx].flatten()
            elif data[key].ndim == 2:
                df_dict[col] = data[key][:, idx].flatten()
            else:
                raise ValueError(f"Unexpected data shape: {data[key].shape}")

    df = pd.DataFrame.from_dict(df_dict)

    if unnorm:
        return unnormalize_df(df, norm_stats)
    else:
        return df


# Concatenate a list of per-file DataFrames into one, skipping pd.concat's
# copy when there's only one file (the common case).
@beartype
def pool_dfs(dfs: list) -> pd.DataFrame:
    return dfs[0] if len(dfs) == 1 else pd.concat(dfs)


# Trailing bracketed token on a column name, e.g. the "[°C]" in
# "Zone Air Temperature [°C]". Anchored to the end of the string so a
# bracket that isn't the last thing in the name doesn't count.
_TRAILING_UNIT_RE = re.compile(r"\[([^\[\]]*)\]\s*$")


@beartype
def parse_column_unit(col: str) -> Optional[str]:
    """Extract a column's unit from its own name: the trailing "[...]".

    "Zone Air Temperature [°C]" -> "°C"; "Ambient Temperature [degC]" ->
    "degC"; "Total DX Power [W]" -> "W". Purely positional — the last
    bracket pair, only if it is also the end of the string — and purely
    extractive: whatever is between the brackets is returned verbatim, with
    no attempt to normalize or validate it as a real unit string. That's the
    dataset owner's business, not this function's.

    Returns None when there is no trailing bracket, when a bracket exists
    but isn't at the end (e.g. "Foo [note] Bar"), or when it's empty
    ("Foo []"). A column with no unit is not an error: `unit_of` and
    `axis_label` both degrade to the bare column/role name.
    """
    match = _TRAILING_UNIT_RE.search(col)
    if match is None:
        return None
    unit = match.group(1).strip()
    return unit or None


@beartype
def unit_of(norm_stats: NormStats, col: str) -> Optional[str]:
    """The unit for `col`, or None if there isn't one to report.

    Covers three cases the same way: `col` absent from `norm_stats`, its
    name carrying no parseable unit, and `norm_stats` reloaded from a run
    trained before per-column units existed (no "unit" key at all). None of
    these is an error — callers fall back to the bare column/role name.
    """
    return norm_stats.get(col, {}).get("unit")


@beartype
def axis_label(
    norm_stats: NormStats,
    cols: list[str],
    fallback: str,
) -> str:
    """Axis label for a group of columns plotted on one axis.

    Renders "<fallback> [unit]" when every column in `cols` shares one known
    unit; otherwise (units disagree, or none is known) renders bare
    `fallback`. Mixed units are real here, not hypothetical: a comfort band
    can span zones, a setpoint axis can span actuators, a power row can span
    meters, and none of those is a single physical channel.
    """
    units = {unit_of(norm_stats, c) for c in cols}
    if len(units) == 1:
        unit = units.pop()
        if unit is not None:
            return f"{fallback} [{unit}]"
    return fallback


# Compute per-column statistics and normalization parameters.
# The returned dict contains all descriptive stats for each column,
# plus "shift" and "stretch" determined by the chosen norm_type.
# This lets downstream code (e.g. synthetic scenarios) access the
# real physical min/max regardless of which normalization was used.
_VALID_NORM_TYPES = {"min_max", "abs_max", "z_score"}


@beartype
def compute_norm_stats(
    df: pd.DataFrame,
    norm_type: Optional[str] = "z_score",
    column_overrides: Optional[dict[str, Optional[str]]] = None,
) -> NormStats:
    """
    Args:
        df: DataFrame used to compute per-column normalization stats.
        norm_type: Global normalization strategy:
            - "min_max": (x - min) / (max - min)  →  [0, 1]
            - "abs_max": x / max(|min|, |max|)     →  [-1, 1]
            - "z_score": (x - mean) / std          →  (-inf, +inf)
            - None:      no normalization (identity)
        column_overrides: Per-column norm_type overrides from
            user_defined.column_norm_config(). Keys are column names;
            values follow the same convention as norm_type, with None
            meaning "skip normalization for this column". Columns not
            present in the dict use the global norm_type.
    Returns:
        norm_stats: {col_name: {"shift", "stretch", "min", "max",
                                 "mean", "stddev", "median", "unit"}}
            "unit" is `parse_column_unit(col_name)` — None unless the column
            name ends in a bracketed token.
    """
    # Validate overrides up-front so bad values fail at startup.
    for col, nt in (column_overrides or {}).items():
        if nt is not None and nt not in _VALID_NORM_TYPES:
            raise ValueError(
                f"column_overrides['{col}'] = '{nt}' is not a recognised "
                f"norm_type. Valid options: {sorted(_VALID_NORM_TYPES)} or None."
            )

    # sec_elapsed stays in raw seconds regardless of any override.
    _ALWAYS_IDENTITY = {"sec_elapsed"}

    def _effective_norm(col: str) -> Optional[str]:
        if col in _ALWAYS_IDENTITY:
            return None
        if column_overrides and col in column_overrides:
            return column_overrides[col]
        return norm_type

    def col_stats(col):
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_mean = float(df[col].mean())
        col_std = float(df[col].std())
        col_median = float(df[col].median())

        nt = _effective_norm(col)
        if nt is None:
            shift, stretch = 0.0, 1.0
        elif nt == "min_max":
            shift, stretch = col_min, col_max - col_min
        elif nt == "abs_max":
            shift, stretch = 0.0, max(abs(col_min), abs(col_max))
        elif nt == "z_score":
            shift, stretch = col_mean, col_std
        else:
            raise AssertionError(f"unhandled norm_type: {nt}")  # guarded above

        # A constant column has no spread to divide by, so its stretch falls
        # back to 1 and it stays in raw physical units. Any column that varies
        # at all is normalised by its own spread, however small: an absolute
        # floor would leave a µA current or a kg/m³ concentration raw and
        # unflagged. The predicate is the one `degenerate_columns` uses, so
        # every fallback is reported.
        #
        # The fallback is silent and has a sharp edge: a column constant in
        # *training* is not necessarily constant later. A furnace that never
        # fires in the training excitation contributes a constant `Gas Energy
        # [J]`, and when the evaluator's trajectory does fire it, 1.6e9 J
        # reaches the model as an input of 1.5e9 rather than something near
        # zero. Two of four measured surrogates diverged to 1e17 within 48
        # steps for exactly this reason.
        if col_min == col_max:
            stretch = 1.0

        return {
            "norm_type": nt if nt is not None else "none",
            "shift": shift,
            "stretch": stretch,
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "stddev": col_std,
            "median": col_median,
            "unit": parse_column_unit(col),
        }

    return {col: col_stats(col) for col in df.columns}


@beartype
def degenerate_columns(norm_stats: NormStats) -> list[str]:
    """Columns that were constant in the data `norm_stats` was fitted on.

    Their `stretch` fell back to 1 (see `compute_norm_stats`), so
    normalization is the identity and they carry raw physical units into the
    model. Any later data in which the column *does* vary is then wildly out of
    range — which is how a surrogate that scores 1.0 °C RMSE on its own test
    split diverges to 1e17 on the evaluator's trajectory.

    Detected from `min == max` rather than from `stretch == 1`, because a column
    can legitimately have unit spread.
    """
    return [col for col, s in norm_stats.items() if s["min"] == s["max"]]


@beartype
def degenerate_mapped_columns(
    norm_stats: NormStats,
    nm_mapping: dict[str, list[str]],
) -> list[str]:
    """`degenerate_columns` restricted to columns the model actually consumes.

    A constant column the mapping never references is harmless — it is dropped
    before the model ever sees it.
    """
    mapped = {c for key in ("Y", "U", "D") for c in nm_mapping.get(key, [])}
    return [c for c in degenerate_columns(norm_stats) if c in mapped]


@beartype
def warn_degenerate_columns(
    norm_stats: NormStats,
    nm_mapping: dict[str, list[str]],
) -> list[str]:
    """Print a startup warning for constant columns; return the ones found.

    The reportcard says this too, but only once the run has finished. The
    condition is knowable the moment `norm_stats` is frozen — before a single
    epoch — and someone watching a loss curve scroll past should not have to
    open a markdown file to learn that a channel is carrying raw physical units.
    """
    frozen = degenerate_mapped_columns(norm_stats, nm_mapping)
    if frozen:
        print(
            "WARNING: constant in the training data, so normalization is the "
            "identity and these columns reach the model in raw physical "
            "units: " + ", ".join(frozen)
        )
        print(
            "  If any of them varies at evaluation time, its value arrives "
            "orders of magnitude out of range and the rollout diverges. "
            "Exclude the column from the mapping or widen the excitation."
        )
    return frozen


# Apply precomputed normalization stats to a dataframe.
@beartype
def normalize_df(
    df: pd.DataFrame,
    norm_stats: NormStats,
):
    """
    Args:
        df: DataFrame to normalize in-place (copied before modification).
        norm_stats: Per-column stats dict applied to columns present in df.
    """
    df_norm = df.copy()
    for col, stats in norm_stats.items():
        if col in df_norm.columns:
            df_norm[col] = (df_norm[col] - stats["shift"]) / stats["stretch"]
    return df_norm


@beartype
def to_physical(
    x: torch.Tensor,
    cols: list[str],
    norm_stats: NormStats,
) -> torch.Tensor:
    """Undo normalization on a tensor whose last axis is `cols`, any leading shape."""
    shift = torch.tensor([norm_stats[c]["shift"] for c in cols],
                         dtype=x.dtype, device=x.device)
    stretch = torch.tensor([norm_stats[c]["stretch"] for c in cols],
                           dtype=x.dtype, device=x.device)
    return x * stretch + shift


# Undo normalization using stored shift/stretch stats.
@beartype
def unnormalize_df(
    df_norm: pd.DataFrame,
    norm_stats: NormStats,
):
    """
    Args:
        df_norm: Normalized DataFrame to convert back to original scale.
        norm_stats: Per-column stats dict used to invert normalization.
    """
    df = df_norm.copy()
    for col, stats in norm_stats.items():
        if col in df.columns:
            df[col] = df_norm[col] * stats["stretch"] + stats["shift"]
    return df
