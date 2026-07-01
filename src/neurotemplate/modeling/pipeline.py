"""
Shared orchestration for the sysid pipeline.

Public surface (functions that earn their keep by composing multiple
non-trivial steps):

  - evaluate_and_plot(...)  — rollout + per-channel metrics + plotting
  - render_reportcard(...)  — describe_dynamics + render
  - write_reportcard(...)   — disk + MLflow
  - log_test_results(...)   — flatten metrics into MLflow log_metrics
  - train_and_evaluate(...) — one full training run (used by main + optuna Phase 2)

`synth_cfg` is computed once at the entry point (via user_defined.synth_cfg)
and passed through everywhere. None means "synth subsystem disabled for
this run" — the single fact, not a re-check at every callsite.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter as tic
from typing import Optional

import torch
from pandas import DataFrame

from neurotemplate._typing import beartype

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.common.data_utils import get_sw_dict_multi, to_df, unnormalize_df
from neurotemplate.modeling.experiment import train_with_curriculum
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.common.metrics import (
    ChannelMetrics,
    metrics_to_dict,
    per_channel_metrics,
)
from neurotemplate.modeling.dynamics import describe_dynamics
from neurotemplate.modeling.plotting import plot_predictions, plot_synthetic
from neurotemplate.modeling.reportcard import render_reportcard as _render_reportcard
from neurotemplate.modeling.scenarios import _unnormalize_y, build_scenarios


# ── Result type ──────────────────────────────────────────────────


@dataclass
class TestResults:
    """Bundle of test-time outputs. Tensors stay internal to evaluate_and_plot."""
    metrics: list[ChannelMetrics]
    aggregate_rmse: float  # legacy single-number RMSE in normalized space
    test_time_s: float


# ── Test eval + plotting (one operation) ────────────────────────


@beartype
def evaluate_and_plot(
    cfg: ModelingConfig,
    dyn_sys,
    test_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
    norm_stats: dict,
    synth_cfg: Optional[dict],
    freq_s: float,
    plot_run_dir: Path,
    logger=None,
) -> TestResults:
    """
    Roll the model out on the test split, compute per-channel metrics,
    and (if cfg.plot) drop prediction + synth plots in plot_run_dir.

    The raw rollout tensors do not escape this function — anything a
    caller needs ends up in TestResults or in the on-disk plots.

    `test_dfs` is always a list (length >= 1). Windows from every file
    are pooled into one rollout and the resulting per-channel metrics
    are pooled too; `_plot_predictions` routes each batch's plot to its
    originating df via `source_idx`.

    `synth_cfg=None` disables the synth subsystem (no plots, no eval).
    """
    test_data = get_sw_dict_multi(
        dfs=test_dfs, cfg=cfg, nm_mapping=nm_mapping,
        steps=cfg.test_window, stride=cfg.test_stride,
    )
    # source_idx is bookkeeping for the per-batch plot router only — the
    # surrogate Node doesn't expect it among its inputs.
    source_idx = test_data.pop("source_idx")

    t0 = tic()
    with torch.no_grad():
        test_out = dyn_sys(test_data)
    test_time_s = tic() - t0

    # System output prepends the initial condition; slice past the in_steps
    # history when comparing to ground truth.
    un_pred = _unnormalize_y(
        test_out["yn"][:, cfg.in_steps:-1, :], nm_mapping, norm_stats
    )
    un_true = _unnormalize_y(
        test_out["Y"][:, cfg.in_steps:, :], nm_mapping, norm_stats
    )
    metrics = per_channel_metrics(un_pred, un_true, nm_mapping["Y"])

    # Legacy aggregate (averaged across channels in normalized space). Kept
    # for back-compat with the pre-refactor `test_rmse` MLflow key.
    err_norm = (
        test_out["yn"][:, cfg.in_steps:-1, :]
        - test_out["Y"][:, cfg.in_steps:, :]
    )
    aggregate_rmse = err_norm.pow(2).mean().sqrt().item()

    if cfg.plot:
        _plot_predictions(
            cfg, test_out, test_dfs, source_idx, nm_mapping, norm_stats,
            plot_run_dir, logger,
        )
        if synth_cfg is not None:
            _plot_synth(
                cfg, dyn_sys, nm_mapping, norm_stats, synth_cfg,
                freq_s, plot_run_dir, logger,
            )
        elif cfg.verbose:
            print("Synthetic scenarios disabled (synth_cfg=None); skipping.")

    return TestResults(
        metrics=metrics,
        aggregate_rmse=aggregate_rmse,
        test_time_s=test_time_s,
    )


def _plot_predictions(
    cfg, test_out, test_dfs, source_idx, nm_mapping, norm_stats,
    plot_run_dir, logger,
):
    """Per-batch prediction plots, routed back to each window's source df.

    `source_idx[batch]` is the originating df index in `test_dfs`. Each
    prediction window's `sec_elapsed` is anchored to its source df's
    start timestamp inside `plot_predictions`, so cross-file pooling
    would misalign the time axis if we used a single concatenated df.
    """
    df_trues = [unnormalize_df(df, norm_stats) for df in test_dfs]
    src = source_idx.detach().cpu().tolist()
    for batch in range(test_out["yn"].shape[0]):
        pred_dict = {
            "Y": test_out["yn"][batch, :-1, :].detach().cpu().numpy(),
            "t": test_out["t"][batch].detach().cpu().numpy(),
        }
        df_pred = to_df(
            data=pred_dict, norm_stats=norm_stats,
            nm_mapping=nm_mapping, unnorm=True,
        )
        plot_predictions(
            cfg=cfg,
            df_true=df_trues[src[batch]], df_pred=df_pred,
            run_dir=str(plot_run_dir),
            verbose=cfg.verbose,
            logger=logger,
        )


def _plot_synth(
    cfg, dyn_sys, nm_mapping, norm_stats, synth_cfg, freq_s,
    plot_run_dir, logger,
):
    """Synthetic-scenario figures."""
    dyn_sys.eval()
    scenarios = build_scenarios(
        system=dyn_sys,
        nm_mapping=nm_mapping,
        norm_stats=norm_stats,
        synth_cfg=synth_cfg,
        in_steps=cfg.in_steps,
        n_scenarios=cfg.synth_n_plot,
        n_steps=cfg.synth_n_steps,
        device=cfg.device,
    )
    if scenarios is None:
        return  # already handled at caller via verbose print

    if cfg.verbose:
        print("Running synthetic scenario evaluation...")
    with torch.no_grad():
        results = [s.evaluate() for s in scenarios]

    if cfg.verbose:
        for r in results:
            if "modes" in r:
                for mode_name, mode_data in r["modes"].items():
                    losses = mode_data["losses"]
                    mean_loss = sum(losses) / len(losses) if losses else 0.0
                    print(f"  {r['name']} ({mode_name}): mean loss = {mean_loss:.4f}")
            else:
                losses = r["losses"]
                mean_loss = sum(losses) / len(losses) if losses else 0.0
                print(f"  {r['name']}: mean loss = {mean_loss:.4f}")

    synth_dir = plot_run_dir / "synthetic"
    y_cols = nm_mapping["Y"]
    temp_range = (
        min(norm_stats[c]["min"] for c in y_cols),
        max(norm_stats[c]["max"] for c in y_cols),
    )
    plot_synthetic(
        results, run_dir=str(synth_dir),
        freq_s=freq_s, n_plot=cfg.synth_n_plot,
        verbose=cfg.verbose, temp_range=temp_range,
    )
    if logger is not None and hasattr(logger, "_mlflow"):
        logger._mlflow.log_artifact(str(synth_dir), artifact_path="plots")


# ── Reportcard ──────────────────────────────────────────────────


@beartype
def render_reportcard(
    cfg: ModelingConfig,
    dyn_sys,
    test_results: TestResults,
    norm_stats: dict,
    nm_mapping: dict[str, list[str]],
    *,
    head_stack: Optional[HeadStack] = None,
    test_dfs: Optional[list[DataFrame]] = None,
    run_name: Optional[str] = None,
    best_dev_loss: Optional[float] = None,
    train_time_s: Optional[float] = None,
    split_sizes: Optional[dict[str, object]] = None,
) -> str:
    """
    Render the markdown reportcard for one run.

    `norm_stats` feeds the per-column normalization labels in the Data
    section. `test_dfs` (already normalized) feeds the bounded-channel
    coverage check — head bounds live in normalized space, so the
    normalized split is what we want to compare against. Multi-file
    splits are concatenated for the coverage check only.
    """
    arch = describe_dynamics(dyn_sys, cfg)
    if test_dfs is None:
        test_df_for_coverage = None
    elif len(test_dfs) == 1:
        test_df_for_coverage = test_dfs[0]
    else:
        import pandas as _pd
        test_df_for_coverage = _pd.concat(test_dfs)
    return _render_reportcard(
        cfg=cfg,
        arch=arch,
        test_metrics=test_results.metrics,
        nm_mapping=nm_mapping,
        head_stack=head_stack,
        norm_stats=norm_stats,
        test_df=test_df_for_coverage,
        run_name=run_name,
        best_dev_loss=best_dev_loss,
        train_time_s=train_time_s,
        test_time_s=test_results.test_time_s,
        split_sizes=split_sizes,
    )


def write_reportcard(
    markdown: str,
    out_dir: Path,
    logger=None,
) -> Path:
    """
    Write the reportcard to <out_dir>/reportcard.md and to MLflow.

    The on-disk copy is convenient for local inspection but may be removed
    by MLFlowLogger.clean_up() (which rmtrees savedir at end of a run);
    the MLflow copy via log_text is the durable one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reportcard.md"
    path.write_text(markdown)
    if logger is not None and hasattr(logger, "_mlflow"):
        logger._mlflow.log_text(markdown, "reportcard.md")
    return path


# ── MLflow metric logging ───────────────────────────────────────


def log_test_results(
    logger,
    test_results: TestResults,
    nm_mapping: dict[str, list[str]],
) -> None:
    """Flatten per-channel metrics into MLflow-safe metric keys + log dict."""
    if logger is None:
        return

    flat: dict[str, float] = {"test_rmse": test_results.aggregate_rmse}
    for m in test_results.metrics:
        safe = _safe_metric_key(m.name)
        flat[f"test_{safe}_rmse"] = m.rmse
        flat[f"test_{safe}_mae"] = m.mae
        flat[f"test_{safe}_r2"] = m.r2
        if m.nrmse == m.nrmse:  # not NaN
            flat[f"test_{safe}_nrmse"] = m.nrmse

    logger.log_metrics(flat)
    if hasattr(logger, "log_dict"):
        logger.log_dict(
            metrics_to_dict(test_results.metrics), name="test_metrics.json"
        )


_METRIC_KEY_SANITIZER = re.compile(r"[^A-Za-z0-9_.\-]+")


def _safe_metric_key(name: str) -> str:
    """Sanitize a column name into something MLflow accepts as a metric key."""
    return _METRIC_KEY_SANITIZER.sub("_", name).strip("_")


# ── Single full training run (consolidates main + optuna Phase 2) ──


@beartype
def train_and_evaluate(
    cfg: ModelingConfig,
    *,
    train_dfs: list[DataFrame],
    dev_dfs: list[DataFrame],
    test_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
    norm_stats: dict,
    grad_masks: dict,
    synth_cfg: Optional[dict],
    freq_s: float,
    logger,
    head_stack: Optional[HeadStack] = None,
    extra_metrics: Optional[dict] = None,
) -> dict:
    """
    Train one model, evaluate it, plot, and write the reportcard.

    Used by main.py end-to-end and by optuna_main.py for the Phase 2 retrain
    with best params. The lighter per-trial loop in optuna_main.objective
    calls train_with_curriculum directly — it deliberately skips
    eval/plot/reportcard for speed.

    `extra_metrics` lets the caller log run-level numbers that aren't part
    of the standard set (e.g. optuna's `search_time`).

    Returns a dict with dyn_sys, best_dev_loss, train_time_s, and
    test_results so the entry point can print a final summary.
    """
    train_t0 = tic()
    dyn_sys, best_dev_loss = train_with_curriculum(
        cfg=cfg,
        train_dfs=train_dfs,
        dev_dfs=dev_dfs,
        nm_mapping=nm_mapping,
        grad_masks=grad_masks,
        norm_stats=norm_stats,
        synth_cfg=synth_cfg,
        head_stack=head_stack,
        global_logger=logger,
    )
    train_time_s = tic() - train_t0

    metrics_to_log = {
        "best_dev_loss": best_dev_loss,
        "train_time": train_time_s,
    }
    if extra_metrics:
        metrics_to_log.update(extra_metrics)
    logger.log_metrics(metrics_to_log)
    logger.log_dict(norm_stats, name="norm_stats.json")
    logger.log_dict(nm_mapping, name="nm_mapping.json")
    if grad_masks:
        logger.log_dict(grad_masks, name="grad_mask.json")
    # Persist head config so load_model can rebuild the exact head topology
    # used at training time, independent of user_defined.py drift.
    if head_stack is not None:
        logger.log_dict(head_stack.to_config(), name="head_config.json")

    plot_run_dir = Path(cfg.savedir) / "plots"
    test_results = evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys,
        test_dfs=test_dfs,
        nm_mapping=nm_mapping, norm_stats=norm_stats,
        synth_cfg=synth_cfg, freq_s=freq_s,
        plot_run_dir=plot_run_dir, logger=logger,
    )
    log_test_results(logger, test_results, nm_mapping)

    if cfg.verbose:
        print(f"Test inference time: {test_results.test_time_s:.2f}s")

    md = render_reportcard(
        cfg=cfg, dyn_sys=dyn_sys, test_results=test_results,
        norm_stats=norm_stats, nm_mapping=nm_mapping,
        head_stack=head_stack,
        test_dfs=test_dfs,
        run_name=cfg.run,
        best_dev_loss=best_dev_loss,
        train_time_s=train_time_s,
        split_sizes={
            name: {"rows": sum(len(d) for d in dfs), "files": len(dfs)}
            for name, dfs in (("train", train_dfs), ("dev", dev_dfs), ("test", test_dfs))
        },
    )
    write_reportcard(md, Path(cfg.savedir), logger=logger)

    return {
        "dyn_sys": dyn_sys,
        "best_dev_loss": best_dev_loss,
        "train_time_s": train_time_s,
        "test_results": test_results,
    }
