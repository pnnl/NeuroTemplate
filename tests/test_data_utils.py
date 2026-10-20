"""Sliding-window construction and norm round-trip identity."""

import numpy as np
import pandas as pd
import pytest
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.common.config import dict_to_dataclass
from neurotemplate.common.data_utils import (
    normalize_df,
    compute_norm_stats,
    get_sw_dict,
    get_sw_dict_multi,
    unnormalize_df,
)


@pytest.fixture
def synthetic_df():
    """A 100-row DataFrame with Y/U/D-like columns and sec_elapsed."""
    n = 100
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y0": rng.normal(20.0, 2.0, n),
        "y1": rng.normal(22.0, 1.5, n),
        "u0": rng.uniform(0.0, 1.0, n),
        "d0": rng.normal(15.0, 5.0, n),
        "sec_elapsed": np.arange(n) * 300.0,
    })
    return df


@pytest.fixture
def nm_mapping():
    return {"Y": ["y0", "y1"], "U": ["u0"], "D": ["d0"], "t": ["sec_elapsed"]}


@pytest.fixture
def cfg():
    c = dict_to_dataclass({
        "history": {"y0": 3, "y1": 3, "u0": 3, "d0": 3},
        "train_pred_steps": 6,
        "device": "cpu",
        "verbose": False,
    }, Config)
    c.device = torch.device("cpu")
    return c


def test_norm_z_score_round_trip(synthetic_df):
    stats = compute_norm_stats(synthetic_df, norm_type="z_score")
    normed = normalize_df(synthetic_df, stats)
    back = unnormalize_df(normed, stats)
    for col in synthetic_df.columns:
        assert np.allclose(back[col].to_numpy(), synthetic_df[col].to_numpy())


def test_column_overrides_apply_per_column_norm(synthetic_df):
    """Override one column to min_max while the global is z_score."""
    stats = compute_norm_stats(
        synthetic_df, norm_type="z_score",
        column_overrides={"y0": "min_max"},
    )
    # y0 overridden to min_max: shift == min, stretch == max - min
    assert np.isclose(stats["y0"]["shift"], synthetic_df["y0"].min())
    assert np.isclose(stats["y0"]["stretch"], synthetic_df["y0"].max() - synthetic_df["y0"].min())
    # y1 uses the global z_score: shift == mean, stretch == std
    assert np.isclose(stats["y1"]["shift"], synthetic_df["y1"].mean())
    assert np.isclose(stats["y1"]["stretch"], synthetic_df["y1"].std())


def test_sec_elapsed_always_identity_regardless_of_override(synthetic_df):
    """sec_elapsed must never be normalized, even if explicitly overridden."""
    stats = compute_norm_stats(
        synthetic_df, norm_type="z_score",
        column_overrides={"sec_elapsed": "min_max"},
    )
    assert stats["sec_elapsed"]["shift"] == 0.0
    assert stats["sec_elapsed"]["stretch"] == 1.0


def test_sliding_window_shapes(synthetic_df, nm_mapping, cfg):
    """Default windowing: steps=history_steps+train_pred_steps=10, stride=history_steps=4."""
    data = get_sw_dict(synthetic_df, cfg, nm_mapping)
    n_rows = len(synthetic_df)
    steps = cfg.history_steps + cfg.train_pred_steps
    expected_windows = (n_rows - steps) // cfg.history_steps + 1

    assert data["Y"].shape == (expected_windows, steps, 2)
    assert data["U"].shape == (expected_windows, steps, 1)
    assert data["D"].shape == (expected_windows, steps, 1)
    assert data["t"].shape == (expected_windows, steps, 1)
    # Initial conditions: yn is the first history_steps of Y
    assert data["yn"].shape == (expected_windows, cfg.history_steps, 2)
    assert torch.equal(data["yn"], data["Y"][:, :cfg.history_steps, :])


def test_sliding_window_stride(synthetic_df, nm_mapping, cfg):
    """Each window starts cfg.history_steps after the previous (default stride)."""
    data = get_sw_dict(synthetic_df, cfg, nm_mapping)
    # First window starts at row 0; second at row history_steps
    if data["Y"].shape[0] >= 2:
        first = data["Y"][0, 0, 0].item()
        second = data["Y"][1, 0, 0].item()
        expected_second = synthetic_df["y0"].iloc[cfg.history_steps]
        assert np.isclose(second, expected_second)
        expected_first = synthetic_df["y0"].iloc[0]
        assert np.isclose(first, expected_first)


# ── Multi-file windowing ─────────────────────────────────────────


def _two_synthetic_dfs(n=100):
    """Two dfs with disjoint, deterministic value ranges so we can tell
    by inspection which file each window came from. File A rows live in
    [0, 99]; file B rows in [1000, 1099] — any window mixing the two
    would have a value gap > 900 inside it."""
    def mk(offset):
        return pd.DataFrame({
            "y0": np.arange(n, dtype=float) + offset,
            "y1": np.arange(n, dtype=float) + offset + 0.5,
            "u0": np.zeros(n),
            "d0": np.zeros(n),
            "sec_elapsed": np.arange(n) * 300.0,
        })
    return [mk(0.0), mk(1000.0)]


def test_multi_file_windows_never_cross_boundary(nm_mapping, cfg):
    """A cross-file window would contain a > 900-wide value gap in y0;
    every per-file window has y0 spanning < 100."""
    dfs = _two_synthetic_dfs(n=100)
    data = get_sw_dict_multi(dfs, cfg, nm_mapping)
    Y = data["Y"]  # [n_windows, steps, 2]
    spans = Y[:, :, 0].max(dim=1).values - Y[:, :, 0].min(dim=1).values
    assert (spans < 200).all(), f"some windows cross the file boundary: {spans}"
    # source_idx is a contiguous run per file (no interleaving).
    src = data["source_idx"].tolist()
    flips = sum(1 for a, b in zip(src[:-1], src[1:]) if a != b)
    assert flips == 1, f"source_idx should be contiguous runs per file: {src}"
