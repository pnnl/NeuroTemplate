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
and passed to the evaluation path. None means "physics checks disabled for
this run" — the single fact, not a re-check at every callsite. Training never
sees it: the scenarios are assertions on the trained surrogate, not a loss.

`train_and_evaluate` also owns the seed-ensemble loop (`cfg.ensemble_seeds`):
every number this module produces is otherwise a single seed's draw, and this
is the one place that both trains a final model and produces the reportcard,
so it is the DRY home for turning several draws into a reported spread. See
ModelingConfig.ensemble_seeds and reportcard.EnsembleSpread.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter as tic
from typing import Optional

import torch
from pandas import DataFrame

from neurotemplate._typing import beartype

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.common.data_utils import get_sw_dict_multi, to_df, unnormalize_df
from neurotemplate.modeling.experiment import set_seeds, train_with_curriculum
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.common.metrics import (
    ChannelMetrics,
    RolloutStability,
    _safe_metric_key,
    metrics_to_dict,
    per_channel_metrics,
    rollout_stability,
    stability_to_dict,
)
from neurotemplate.common.reportcard import write_reportcard as _write_reportcard
from neurotemplate.modeling.dynamics import describe_dynamics
from neurotemplate.modeling.plotting import plot_predictions, plot_synthetic
from neurotemplate.modeling.reportcard import (
    EnsembleSpread,
    render_reportcard as _render_reportcard,
    spread_stats,
)
from neurotemplate.modeling.scenarios import (
    ReadoutSummary,
    ScenarioOutcome,
    ScenarioResult,
    SynthSpec,
    _unnormalize_y,
    build_scenarios,
    checks_to_dict,
    format_checks,
    readouts_to_dict,
    run_scenarios,
)


# ── Block spec resolution ────────────────────────────────────────


def resolve_block_spec(hooks, nm_mapping: dict[str, list[str]], cfg: ModelingConfig):
    """Call the hooks module's `block_mapping`, falling back to the
    deprecated `zone_mapping`, or None if neither is defined.

    One fallback, not a chain: `block_mapping` wins whenever a hooks module
    defines both, so an application mid-migration only ever has one spec in
    effect. Shared by cli/train.py and cli/optuna.py, which both need this
    exact lookup before building the trainer.
    """
    if hasattr(hooks, "block_mapping"):
        return hooks.block_mapping(nm_mapping, cfg)
    if hasattr(hooks, "zone_mapping"):
        return hooks.zone_mapping(nm_mapping, cfg)
    return None


# ── Result type ──────────────────────────────────────────────────


@dataclass
class TestResults:
    """Bundle of test-time outputs. Tensors stay internal to evaluate_and_plot."""
    metrics: list[ChannelMetrics]
    stability: list[RolloutStability]  # rollout shape, to catch ramping models
    aggregate_rmse: float  # legacy single-number RMSE in normalized space
    test_time_s: float
    # Physics assertions on the thermal states. None when the use case opted
    # out of synth_cfg.
    physics: Optional[list[ScenarioResult]] = None
    # Rollout length the scenarios ran for, so the reportcard can state it.
    physics_steps: Optional[int] = None
    # Order statistics for the Y channels that are not thermal states. These
    # carry no verdict — see scenarios.ReadoutSummary.
    readouts: Optional[list[ReadoutSummary]] = None


# ── Test eval + plotting (one operation) ────────────────────────


@beartype
def evaluate_and_plot(
    cfg: ModelingConfig,
    dyn_sys,
    test_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
    norm_stats: dict,
    synth_cfg: Optional[SynthSpec],
    freq_s: float,
    plot_run_dir: Path,
    logger=None,
    grad_masks: Optional[dict] = None,
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
    # RMSE alone cannot see a model that has stopped responding to its inputs
    # and is emitting a ramp; compare the rollout's shape to the truth's too.
    stability = rollout_stability(un_pred, un_true, nm_mapping["Y"])

    # Legacy aggregate (averaged across channels in normalized space). Kept
    # for back-compat with the pre-refactor `test_rmse` MLflow key.
    err_norm = (
        test_out["yn"][:, cfg.in_steps:-1, :]
        - test_out["Y"][:, cfg.in_steps:, :]
    )
    aggregate_rmse = err_norm.pow(2).mean().sqrt().item()

    # Physics assertions on synthetic states the recorded data never visits.
    # The verdicts are the point; the figures are for when one surprises you.
    outcome = _run_physics_checks(
        cfg, dyn_sys, nm_mapping, norm_stats, synth_cfg, freq_s,
        plot_run_dir, logger, grad_masks,
    )

    if cfg.plot:
        _plot_predictions(
            cfg, test_out, test_dfs, source_idx, nm_mapping, norm_stats,
            plot_run_dir, logger,
        )

    return TestResults(
        metrics=metrics,
        stability=stability,
        aggregate_rmse=aggregate_rmse,
        test_time_s=test_time_s,
        physics=None if outcome is None else outcome.verdicts,
        physics_steps=None if outcome is None else synth_steps(cfg, freq_s),
        readouts=None if outcome is None else outcome.readouts,
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


@beartype
def synth_steps(cfg: ModelingConfig, freq_s: float) -> int:
    """Scenario horizon in steps, from `cfg.synth_hours` and the data's own dt.

    The sampling frequency is already on hand wherever the scenarios run, so
    the horizon can be specified in hours instead of an unlabelled step count.
    Floored at 2 steps, which is the least a rollout can measure anything over.
    """
    if freq_s <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {freq_s}")
    return max(2, round(cfg.synth_hours * 3600.0 / freq_s))


def _run_physics_checks(
    cfg, dyn_sys, nm_mapping, norm_stats, synth_cfg, freq_s,
    plot_run_dir, logger, grad_masks=None,
) -> Optional[ScenarioOutcome]:
    """Assert the three physics scenarios; plot their traces if plotting is on.

    Returns None when the use case opted out (`synth_cfg` hook returns None).
    """
    dyn_sys.eval()
    scenarios = build_scenarios(
        system=dyn_sys,
        nm_mapping=nm_mapping,
        norm_stats=norm_stats,
        synth_cfg=synth_cfg,
        in_steps=cfg.in_steps,
        n_scenarios=cfg.synth_n_scenarios,
        n_steps=synth_steps(cfg, freq_s),
        device=cfg.device,
        grad_masks=grad_masks,
    )
    if scenarios is None:
        if cfg.verbose:
            print("Physics checks disabled (synth_cfg=None); skipping.")
        return None

    outcome = run_scenarios(scenarios)
    if cfg.verbose:
        print(format_checks(outcome.verdicts))

    if cfg.plot:
        synth_dir = plot_run_dir / "synthetic"
        plot_synthetic(
            [s.trace(cfg.synth_n_plot) for s in scenarios],
            run_dir=str(synth_dir),
            freq_s=freq_s, n_plot=cfg.synth_n_plot, verbose=cfg.verbose,
            y_unit=cfg.synth_y_unit,
        )
        if logger is not None and hasattr(logger, "_mlflow"):
            logger._mlflow.log_artifact(str(synth_dir), artifact_path="plots")

    return outcome


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
    ensemble: Optional[EnsembleSpread] = None,
) -> str:
    """
    Render the markdown reportcard for one run.

    `norm_stats` feeds the per-column normalization labels in the Data
    section. `test_dfs` (already normalized) feeds the bounded-channel
    coverage check — head bounds live in normalized space, so the
    normalized split is what we want to compare against. Multi-file
    splits are concatenated for the coverage check only.

    `ensemble` is the optional cfg.ensemble_seeds > 1 spread (see
    train_and_evaluate); None renders no extra section.
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
        stability=test_results.stability,
        physics=test_results.physics,
        physics_steps=test_results.physics_steps,
        readouts=test_results.readouts,
        ensemble=ensemble,
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
    """Write the sysid reportcard to <out_dir>/reportcard.md and to MLflow."""
    return _write_reportcard(markdown, out_dir, "reportcard.md", logger=logger)


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
    for st in test_results.stability:
        safe = _safe_metric_key(st.name)
        flat[f"test_{safe}_rise_excess"] = st.rise_excess
        flat[f"test_{safe}_drift_ratio"] = st.drift_ratio
    if test_results.physics is not None:
        # One gate number to sort a leaderboard by, plus per-(scenario, channel)
        # detail — so a channel someone does not care about can be excluded from
        # a comparison rather than quietly moving the aggregate.
        flat["physics_checks_passed"] = float(
            sum(r.passed for r in test_results.physics)
        )
        flat["physics_checks_total"] = float(len(test_results.physics))
        for r in test_results.physics:
            key = _safe_metric_key(r.key)
            flat[f"physics_{key}"] = r.value
            flat[f"physics_{key}_passed"] = float(r.passed)

    logger.log_metrics(flat)
    if hasattr(logger, "log_dict"):
        logger.log_dict(
            metrics_to_dict(test_results.metrics), name="test_metrics.json"
        )
        logger.log_dict(
            stability_to_dict(test_results.stability),
            name="rollout_stability.json",
        )
        if test_results.physics is not None:
            logger.log_dict(
                checks_to_dict(test_results.physics),
                name="physics_checks.json",
            )
        if test_results.readouts:
            logger.log_dict(
                readouts_to_dict(test_results.readouts),
                name="readout_channels.json",
            )


# ── Seed ensemble ─────────────────────────────────────────────────
#
# `cfg.ensemble_seeds > 1` reruns train + evaluate at cfg.seed + 1 ..
# cfg.seed + ensemble_seeds - 1 (the primary, seed + 0, is trained by
# train_and_evaluate itself) and reduces the per-seed results to an
# EnsembleSpread. No block, family, or hook ever sees a seed — this loop
# is the only place one is threaded, via the global RNG set_seeds() reseeds.


def _tau_slow_by_channel(arch: dict) -> dict[str, float]:
    """Flatten describe_dynamics's nested block info to {channel: tau_slow}.

    Walks the same `blocks` nesting dynamics._describe_block builds (a
    BlockStack's per-block descriptors, each optionally nested again). Only
    2r2c blocks carry `time_constants_h`, so every other family is simply
    absent from the result.
    """
    out: dict[str, float] = {}

    def walk(info: Optional[dict]) -> None:
        if info is None:
            return
        for channel, taus in info.get("time_constants_h", {}).items():
            out[channel] = taus["slow"]
        for sub in info.get("blocks", {}).values():
            walk(sub)

    for key in ("fy", "fu", "fd"):
        walk(arch.get("blocks", {}).get(key))
    return out


def _build_ensemble_spread(
    n_seeds: int,
    per_seed_metrics: list[list[ChannelMetrics]],
    per_seed_physics: list[Optional[list[ScenarioResult]]],
    per_seed_tau_slow: list[dict[str, float]],
) -> EnsembleSpread:
    """Group each metric's per-seed values under its channel/key and reduce
    with `spread_stats`. Every list argument has one entry per seed, in the
    same order (primary first)."""
    rmse: dict[str, list[float]] = {}
    for metrics in per_seed_metrics:
        for m in metrics:
            rmse.setdefault(m.name, []).append(m.rmse)

    physics: dict[str, list[float]] = {}
    for results in per_seed_physics:
        for r in (results or []):
            physics.setdefault(r.key, []).append(r.value)

    tau_slow: dict[str, list[float]] = {}
    for taus in per_seed_tau_slow:
        for channel, value in taus.items():
            tau_slow.setdefault(channel, []).append(value)

    return EnsembleSpread(
        n_seeds=n_seeds,
        rmse={k: spread_stats(v) for k, v in rmse.items()},
        physics={k: spread_stats(v) for k, v in physics.items()},
        tau_slow={k: spread_stats(v) for k, v in tau_slow.items()},
    )


def _run_ensemble_members(
    cfg: ModelingConfig,
    *,
    train_dfs: list[DataFrame],
    dev_dfs: list[DataFrame],
    test_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
    norm_stats: dict,
    grad_masks: dict,
    synth_cfg: Optional[SynthSpec],
    freq_s: float,
    head_stack: Optional[HeadStack],
    block_spec: Optional[dict[str, dict]],
    plot_run_dir: Path,
    primary_dyn_sys,
    primary_test_results: TestResults,
) -> EnsembleSpread:
    """Train `cfg.ensemble_seeds - 1` more models (seeds cfg.seed + 1 ..
    cfg.seed + ensemble_seeds - 1), score each through the same evaluation
    path as the primary, and return the spread across the whole ensemble
    (primary included).

    Quiet and unplotted: only the primary model's weights, plots, and MLflow
    artifacts are kept (see ModelingConfig.ensemble_seeds) — these extra
    members exist purely to widen the spread this function returns.
    """
    quiet_cfg = replace(cfg, plot=False, verbose=False)

    per_seed_metrics = [primary_test_results.metrics]
    per_seed_physics = [primary_test_results.physics]
    per_seed_tau_slow = [_tau_slow_by_channel(describe_dynamics(primary_dyn_sys, cfg))]

    for i in range(1, cfg.ensemble_seeds):
        if cfg.verbose:
            print(f"Training ensemble member {i + 1}/{cfg.ensemble_seeds} "
                  f"(seed={cfg.seed + i})...")
        set_seeds(cfg.seed + i)
        dyn_sys, _ = train_with_curriculum(
            cfg=quiet_cfg,
            train_dfs=train_dfs, dev_dfs=dev_dfs,
            nm_mapping=nm_mapping, grad_masks=grad_masks, norm_stats=norm_stats,
            head_stack=head_stack, block_spec=block_spec,
            global_logger=None,
        )
        test_results = evaluate_and_plot(
            cfg=quiet_cfg, dyn_sys=dyn_sys, test_dfs=test_dfs,
            nm_mapping=nm_mapping, norm_stats=norm_stats,
            synth_cfg=synth_cfg, freq_s=freq_s,
            plot_run_dir=plot_run_dir, logger=None, grad_masks=grad_masks,
        )
        per_seed_metrics.append(test_results.metrics)
        per_seed_physics.append(test_results.physics)
        per_seed_tau_slow.append(_tau_slow_by_channel(describe_dynamics(dyn_sys, cfg)))

    return _build_ensemble_spread(
        cfg.ensemble_seeds, per_seed_metrics, per_seed_physics, per_seed_tau_slow,
    )


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
    synth_cfg: Optional[SynthSpec],
    freq_s: float,
    logger,
    head_stack: Optional[HeadStack] = None,
    block_spec: Optional[dict[str, dict]] = None,
    extra_metrics: Optional[dict] = None,
) -> dict:
    """
    Train one model, evaluate it, plot, and write the reportcard.

    Used by main.py end-to-end and by optuna_main.py for the Phase 2 retrain
    with best params. The lighter per-trial loop in optuna_main.objective
    calls train_with_curriculum directly — it deliberately skips
    eval/plot/reportcard for speed.

    When `cfg.ensemble_seeds > 1`, this also trains that many more models at
    cfg.seed + 1 .. cfg.seed + ensemble_seeds - 1 (see `_run_ensemble_members`)
    and reports the spread across all of them. The primary run (this one, at
    cfg.seed + 0) stays the one whose weights are logged, whose plots are
    written, and whose reportcard is rendered — the ensemble adds error bars
    around that one artifact, it does not change what artifact you get.

    `extra_metrics` lets the caller log run-level numbers that aren't part
    of the standard set (e.g. optuna's `search_time`).

    Returns a dict with dyn_sys, best_dev_loss, train_time_s, test_results,
    and ensemble (None unless cfg.ensemble_seeds > 1) so the entry point can
    print a final summary.
    """
    train_t0 = tic()
    # Reseed here rather than trust the caller to have left the RNG at
    # cfg.seed: the ensemble loop below reseeds at cfg.seed + i for i > 0,
    # and the primary (i = 0) needs the same explicit contract rather than
    # an implicit "whatever the caller last seeded".
    set_seeds(cfg.seed)
    dyn_sys, best_dev_loss = train_with_curriculum(
        cfg=cfg,
        train_dfs=train_dfs,
        dev_dfs=dev_dfs,
        nm_mapping=nm_mapping,
        grad_masks=grad_masks,
        norm_stats=norm_stats,
        head_stack=head_stack,
        block_spec=block_spec,
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
    # Persist the block split so `nt model load` rebuilds the same topology
    # even if the hooks file has moved on.
    if block_spec:
        logger.log_dict(block_spec, name="block_spec.json")
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
        grad_masks=grad_masks,
    )
    log_test_results(logger, test_results, nm_mapping)

    if cfg.verbose:
        print(f"Test inference time: {test_results.test_time_s:.2f}s")

    ensemble = None
    if cfg.ensemble_seeds > 1:
        ensemble = _run_ensemble_members(
            cfg,
            train_dfs=train_dfs, dev_dfs=dev_dfs, test_dfs=test_dfs,
            nm_mapping=nm_mapping, norm_stats=norm_stats, grad_masks=grad_masks,
            synth_cfg=synth_cfg, freq_s=freq_s, head_stack=head_stack,
            block_spec=block_spec, plot_run_dir=plot_run_dir,
            primary_dyn_sys=dyn_sys, primary_test_results=test_results,
        )

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
        ensemble=ensemble,
    )
    write_reportcard(md, Path(cfg.savedir), logger=logger)

    return {
        "dyn_sys": dyn_sys,
        "best_dev_loss": best_dev_loss,
        "train_time_s": train_time_s,
        "test_results": test_results,
        "ensemble": ensemble,
    }
