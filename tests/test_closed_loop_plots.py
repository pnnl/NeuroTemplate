"""Closed-loop figures and trajectory CSVs carry the plant's real clock.

`t` is `sec_elapsed`, which counts from its own file's first row. Reading it
as POSIX seconds stamps every artifact 1970-01-01 and hides the error whenever
a split happens to start at midnight, which is exactly when hour-of-day still
looks right. These tests use non-midnight, non-1970 origins so that mistake
cannot pass.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from neurotemplate.control.pipeline import evaluate_closed_loop, plot_closed_loop
from neurotemplate.control.policy import build_policy
from neurotemplate.control.closed_loop import build_closed_loop
from control_fixtures import full_setup

# Deliberately not midnight and not the epoch.
ORIGIN_A = pd.Timestamp("2024-08-02 06:35")
ORIGIN_B = pd.Timestamp("2023-11-19 13:05")


def rollout(n_files=1):
    """Evaluate an untrained closed loop over `n_files` identical test files."""
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    cl = build_closed_loop(
        s["dyn_node"], build_policy(cfg, sysid_cfg, nm, ns), cfg, sysid_cfg,
    )
    _, out = evaluate_closed_loop(
        cfg, sysid_cfg, cl, [s["norm_df"]] * n_files, nm, ns,
    )
    return s, out


def read_csvs(plot_dir):
    """Every trajectory CSV under `plot_dir`, keyed by its day folder name."""
    return {
        path.parent.name: pd.read_csv(path, index_col="timestamp",
                                      parse_dates=["timestamp"])
        for path in sorted(plot_dir.rglob("closed_loop_*.csv"))
    }


def test_each_window_gets_its_own_files_origin(tmp_path):
    """Windows are pooled across files; `source_idx` has to route them back."""
    s, out = rollout(n_files=2)
    plot_dir = tmp_path / "plots"
    plot_closed_loop(
        s["cfg"], s["sysid_cfg"], out, s["nm"], s["norm_stats"], plot_dir,
        origins=[ORIGIN_A, ORIGIN_B],
    )

    csvs = read_csvs(plot_dir)
    years = {df.index[0].year for df in csvs.values()}
    assert years == {2024, 2023}, f"expected both files' clocks, got {years}"
    assert any(name.startswith("Aug_02") for name in csvs)
    assert any(name.startswith("Nov_19") for name in csvs)
