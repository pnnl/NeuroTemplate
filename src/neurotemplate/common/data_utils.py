import pandas as pd
import torch
import json
import warnings
from typing import Optional

from neuromancer.dataset import DictDataset

from neurotemplate.common.config import paths_as_list

from neurotemplate._typing import beartype
    

# ---------------------- Data utilities ---------------------- #

# Load CSV data, enforce timestamp frequency, and add elapsed time column.
@beartype
def get_data(
    path: str, 
    verbose: Optional[bool] = False,
):
    """
    Args:
        path: CSV file path used to load the raw dataset.
        verbose: Reserved for caller-level logging control (currently unused here).
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

    df_freq = pd.infer_freq(df["timestamp"])
    if df_freq is None:
        raise ValueError("Could not infer frequency from 'timestamp' column.")

    idx = pd.date_range(
        df["timestamp"].min(), 
        df["timestamp"].max(), 
        freq=df_freq
    )

    df = df.set_index("timestamp").reindex(idx)

    # Convert frequency to total seconds. `pd.infer_freq` returns bare
    # abbreviations for unit-multiple frequencies ("h", "D", "min"), and
    # `pd.to_timedelta` rejects those — go through the offset, which resolves
    # both those and the multiplied forms ("5min", "15min", "2h").
    freq_seconds = pd.tseries.frequencies.to_offset(df_freq).nanos / 1e9

    assert freq_seconds > 0, f"Non-positive frequency inferred: {freq_seconds}"

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


def _load_split(paths: list[str], name: str, verbose: bool):
    """Read every CSV in `paths` and return (dfs, shared_freq_seconds).

    Each file must have the same sampling frequency and no NaNs. Returns
    a list of one df for the single-path case so callers always iterate.
    """
    dfs: list[pd.DataFrame] = []
    freq: Optional[float] = None
    for i, path in enumerate(paths):
        df, f = get_data(path, verbose)
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
    assert freq is not None  # _check_paths_field rejects empty lists upstream
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
    train_dfs, train_freq = _load_split(
        paths_as_list(cfg.train_data_path), "train", cfg.verbose,
    )
    dev_dfs, dev_freq = _load_split(
        paths_as_list(cfg.dev_data_path), "dev", cfg.verbose,
    )
    test_dfs, test_freq = _load_split(
        paths_as_list(cfg.test_data_path), "test", cfg.verbose,
    )

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
) -> dict[str, torch.tensor]:
    """
    Args:
        grad_path: JSON mapping path that enables/defines gradient constraints (None disables).
        nm_mapping: Column mapping that defines Y/U/D indices for mask construction.

    Returns dense sign masks for gradient constraints,
    entries are all {-1,0,+1}

    Output:
        grad_mask["Y"]: (ny, ny)  with entries in {-1,0,+1}
        grad_mask["U"]: (ny, nu)
        grad_mask["D"]: (ny, nd)
    """
    if grad_path is None:
        return {}

    with open(grad_path, "r") as f:
        key_mapping = json.load(f)

    Y_names = nm_mapping["Y"]
    U_names = nm_mapping["U"]
    D_names = nm_mapping["D"]

    ny, nu, nd = len(Y_names), len(U_names), len(D_names)

    grad_masks = {}

    for key_i, deps in key_mapping.items():
        if key_i not in Y_names:
            continue
            # raise ValueError(f"Trying to constrain {key_i}, \
            #     which is not a predicted value (not in Y).")

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


# Create sliding window tensors for Y/U/D/t based on split indices.
@beartype
def get_sw_dict(
    df: pd.DataFrame,
    cfg,
    nm_mapping: dict[str, list[str]],
    steps: Optional[int] = None,
    stride: Optional[int] = None
):
    """
    Args:
        df: Full normalized dataframe used to build sliding windows.
        cfg providing in_steps/train_pred_steps, device placement, and verbosity.
        nm_mapping: Column mapping that defines Y/U/D/t extraction order.
        steps: Optional override for window length (None uses in_steps + train_pred_steps).
    """

    # Resolve window length based on args or full-length rollouts.
    if steps is None:
        # Default: history window + training rollout length.
        steps = cfg.in_steps + cfg.train_pred_steps
    elif steps < 0:
        # if the time steps is negative, we give 
        # the full data window. This is used for
        # test sets where we want to do long
        # autoregressive rollouts to test a model
        steps = df.shape[0]
    elif steps > df.shape[0]:
        warnmsg = f"Warning: Sub-Dataset size of {df.shape[0]}"
        warnmsg += f" cannot support data window size {steps}"
        warnmsg += f"\nReducing to maximum window size: {df.shape[0]}"
        if cfg.verbose:
            print(warnmsg)
        steps = df.shape[0]

    if not stride:
        stride = cfg.in_steps

    # Construct windowed tensors for a given key with stride = in_steps.
    def build_sliding_windows(df, key):

        x = torch.tensor(
            df[nm_mapping[key]].to_numpy(), 
            dtype=torch.float32,
            device=cfg.device
        )  # shape (steps, features)

        starts = torch.arange(
            0,
            x.shape[0] - steps + 1, 
            stride, 
            device=cfg.device
        )  # these are the beginning of each sliding window

        nw = starts.shape[0]  # number of windows
        feat = x.shape[1]  # feature dimension

        # [:,None] turns starts into a column vector of shape [num_windows,1]
        # [None, :] turns torch.arange(steps) into a vector of shape [1, steps]
        # Adding them together row by row increments the number of steps by 
        # the start index, and you get a tensor of shape [num_windows, steps]
        # where each window row has the indices which belong in that window
        idx = starts[:, None]+torch.arange(steps, device=cfg.device)[None, :]

        # flatten out those indices, select them all, and reshape the result
        xw = x.index_select(0, idx.flatten()).reshape(nw, steps, feat)
        return xw

    # --- Extract Y,U,D,t ---
    Y, U, D, t = [build_sliding_windows(df, key) for key in ['Y', 'U', 'D', 't']]
    
    sliding_window_dataset = {
        't': t, 'Y': Y,
        'U': U, 'D': D,
        'yn': Y[:, 0:cfg.in_steps, :]
    }

    return sliding_window_dataset


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
        resolved_steps = cfg.in_steps + cfg.train_pred_steps
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
                "window (train_pred_steps/test_window/in_steps)."
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
        steps: Optional override for window length (None uses in_steps + train_pred_steps).
    """

    data = get_sw_dict_multi(dfs, cfg, nm_mapping, steps)

    dataset = DictDataset(data, name=name)

    # Keep DataLoader RNG on CPU for portability across devices.
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
    norm_stats: dict[str, dict[str, float]],
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
) -> dict[str, dict[str, float]]:
    """
    Args:
        df: DataFrame used to compute per-column normalization stats.
        norm_type: Global normalization strategy:
            - "min_max": (x - min) / (max - min)  →  [0, 1]
            - "abs_max": x / abs(max)              →  [-1, 1]
            - "z_score": (x - mean) / std          →  (-inf, +inf)
            - None:      no normalization (identity)
        column_overrides: Per-column norm_type overrides from
            user_defined.column_norm_config(). Keys are column names;
            values follow the same convention as norm_type, with None
            meaning "skip normalization for this column". Columns not
            present in the dict use the global norm_type.
    Returns:
        norm_stats: {col_name: {"shift", "stretch", "min", "max",
                                 "mean", "stddev", "median"}}
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

    # Safety wrapper: avoids dividing by zero for constant columns.
    #
    # The rescue is silent and has a sharp edge. `stretch = 1` makes
    # normalization the identity for that column, so it stays in raw physical
    # units. That is harmless while the column really is constant, but a column
    # constant in *training* is not necessarily constant later: a furnace that
    # never fires in the training excitation contributes a constant `Gas Energy
    # [J]`, and when the evaluator's trajectory does fire it, 1.6e9 J reaches the
    # model as an input of 1.5e9 rather than something near zero. Two of four
    # measured surrogates diverged to 1e17 within 48 steps for exactly this
    # reason. `degenerate_columns` below makes the condition visible instead of
    # leaving it to be rediscovered from the wreckage.
    def pos(x):
        return x if x > 1e-6 else 1

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
            shift, stretch = col_min, pos(col_max - col_min)
        elif nt == "abs_max":
            shift, stretch = 0.0, pos(abs(col_max))
        elif nt == "z_score":
            shift, stretch = col_mean, pos(col_std)
        else:
            raise AssertionError(f"unhandled norm_type: {nt}")  # guarded above

        # return {
        #     "shift": 0.0,
        #     "stretch": 1.0,
        #     "min": col_min,
        #     "max": col_max,
        #     "mean": col_mean,
        #     "stddev": col_std,
        #     "median": col_median,
        # }

        return {
            "norm_type": nt if nt is not None else "none",
            "shift": shift,
            "stretch": stretch,
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "stddev": col_std,
            "median": col_median,
        }

    return {col: col_stats(col) for col in df.columns}


@beartype
def degenerate_columns(norm_stats: dict[str, dict[str, float]]) -> list[str]:
    """Columns that were constant in the data `norm_stats` was fitted on.

    Their `stretch` fell back to 1 (see `pos` in `compute_norm_stats`), so
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
    norm_stats: dict[str, dict[str, float]],
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
    norm_stats: dict[str, dict[str, float]],
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
    norm_stats: dict[str, dict[str, float]],
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


# Undo normalization using stored shift/stretch stats.
@beartype
def unnormalize_df(
    df_norm: pd.DataFrame,
    norm_stats: dict[str, dict[str, float]],
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
