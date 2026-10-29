"""
Shared orchestration for the sysid pipeline.

Public surface (functions that earn their keep by composing multiple
non-trivial steps):

  - evaluate_and_plot(...)  — rollout + per-channel metrics + plotting
  - prepare_data(...)       — everything computed from the data before a model
  - render_reportcard(...)  — describe_dynamics + render
  - write_reportcard(...)   — disk + MLflow
  - log_test_results(...)   — flatten metrics into MLflow log_metrics
  - train_and_evaluate(...) — one full training run (`nt model train`, and
                              `nt model optuna`'s final retrain)

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

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter as tic
from typing import Optional

import torch
from pandas import DataFrame

from neurotemplate._typing import beartype

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.common.data_utils import (
    compute_norm_stats,
    get_data_splits,
    get_grad_masks,
    get_sw_dict_multi,
    pool_dfs,
    to_df,
    warn_degenerate_columns,
)
from neurotemplate.modeling.calibration import (
    Calibration,
    balance_level_weights,
    calibrate_decision_thresholds,
)
from neurotemplate.modeling.experiment import set_seeds, train_with_curriculum
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.common.metrics import (
    ChannelMetrics,
    LevelMetrics,
    RolloutStability,
    metric_key,
    level_metrics,
    level_metrics_to_dict,
    metrics_to_dict,
    per_channel_metrics,
    rollout_stability,
    stability_to_dict,
)
from neurotemplate.common.reportcard import write_reportcard as _write_reportcard
from neurotemplate.modeling.contract import contract_model, normalized_core
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
    build_scenarios,
    checks_to_dict,
    format_checks,
    readouts_to_dict,
    run_scenarios,
)


# ── Block spec resolution ────────────────────────────────────────


ROLES = ("Y", "U", "D", "t")


def resolve_nm_mapping(hooks, df: DataFrame, cfg: ModelingConfig) -> dict[str, list[str]]:
    """Call the hooks module's `data_mapping` and check what it returns.

    Checked here rather than in the hook, so a rewritten hook gets the same
    checks: the four roles and nothing else, every name a column of `df`,
    no column in two roles or twice in one, a non-empty Y, and
    `cfg.exclude_columns` both matching some column (a pattern that matches
    nothing is a typo) and matching no mapped column. Every problem is listed
    at once.
    """
    nm_mapping = hooks.data_mapping(df=df, cfg=cfg)
    problems = []
    if set(nm_mapping) != set(ROLES):
        problems.append(f"keys must be exactly {list(ROLES)}, got {sorted(nm_mapping)}")
    named = [c for role in ROLES for c in nm_mapping.get(role, [])]
    unknown = [c for c in named if c not in df.columns]
    if unknown:
        problems.append(f"columns not in the dataframe: {unknown}")
    repeated = sorted({c for c in named if named.count(c) > 1})
    if repeated:
        problems.append(f"columns in more than one role (or twice in one): {repeated}")
    if not nm_mapping.get("Y"):
        problems.append("Y is empty, so the model has nothing to predict")
    unmatched = [p for p in cfg.exclude_columns if not any(p in c for c in df.columns)]
    if unmatched:
        problems.append(f"cfg.exclude_columns patterns match no column: {unmatched}")
    excluded = [c for c in named if any(p in c for p in cfg.exclude_columns)]
    if excluded:
        problems.append(f"columns matching cfg.exclude_columns are still mapped: {excluded}")
    if problems:
        raise ValueError("data_mapping returned an unusable mapping:\n  - "
                         + "\n  - ".join(problems))
    return nm_mapping


def resolve_block_spec(hooks, nm_mapping: dict[str, list[str]], cfg: ModelingConfig):
    """Call the hooks module's optional `block_mapping`, or None if it has none."""
    if hasattr(hooks, "block_mapping"):
        return hooks.block_mapping(nm_mapping, cfg)
    return None


def build_head_stack(
    hooks, cfg: ModelingConfig, nm_mapping: dict[str, list[str]],
    norm_stats: dict, train_dfs: list[DataFrame],
) -> HeadStack:
    """The hook's head stack, in physical units, with level weights balanced
    from the training data when `cfg.balance_levels`."""
    head_stack = hooks.data_head_config(nm_mapping, norm_stats)
    if cfg.balance_levels:
        balance_level_weights(head_stack, train_dfs)
    return head_stack


@dataclass
class PreparedData:
    """What a training run computes from the data once, before any model.

    Field names are `train_and_evaluate`'s keyword arguments, so a prepared
    run is handed over as `train_and_evaluate(cfg, logger=..., **vars(data))`.
    """
    train_dfs: list[DataFrame]
    dev_dfs: list[DataFrame]
    test_dfs: list[DataFrame]
    freq_s: float
    nm_mapping: dict[str, list[str]]
    block_spec: Optional[dict[str, dict]]
    norm_stats: dict
    grad_masks: dict
    synth_cfg: Optional[SynthSpec]
    head_stack: HeadStack


def prepare_data(cfg: ModelingConfig, hooks) -> PreparedData:
    """Load the three splits, map them, fit the norm stats, build the heads.

    The splits stay in physical units: a model normalises for itself (see
    `contract`). Norm stats are fitted on all training files pooled, so each
    column's range is the union of the training data; windowing stays per
    file.
    Sets `cfg.steps_per_hour` from the data's own sampling frequency — before
    the hooks see `cfg`, since `2r2c` and `block_mapping` may read it.
    """
    train_raws, dev_raws, test_raws, freq_s = get_data_splits(cfg)
    cfg.steps_per_hour = 3600.0 / freq_s
    # Column roles only need the schema, and get_data_splits has already
    # checked every file carries the same columns.
    nm_mapping = resolve_nm_mapping(hooks, train_raws[0], cfg)
    block_spec = resolve_block_spec(hooks, nm_mapping, cfg)

    norm_stats = compute_norm_stats(
        df=pool_dfs(train_raws), norm_type=cfg.norm_type,
        column_overrides=hooks.column_norm_config(nm_mapping),
    )
    warn_degenerate_columns(norm_stats, nm_mapping)
    synth_cfg = hooks.synth_cfg(nm_mapping, norm_stats)
    if synth_cfg is not None:
        # Before the first epoch, not after the last: the checks run once
        # training is over, and `build_scenarios` re-validates on reload.
        synth_cfg.validate(nm_mapping)
    return PreparedData(
        train_dfs=train_raws,
        dev_dfs=dev_raws,
        test_dfs=test_raws,
        freq_s=freq_s,
        nm_mapping=nm_mapping,
        block_spec=block_spec,
        norm_stats=norm_stats,
        grad_masks=get_grad_masks(grad_path=cfg.grad_path, nm_mapping=nm_mapping),
        synth_cfg=synth_cfg,
        head_stack=build_head_stack(hooks, cfg, nm_mapping, norm_stats, train_raws),
    )


def _train(cfg, *, train_dfs, dev_dfs, nm_mapping, grad_masks, norm_stats,
           head_stack, block_spec, logger):
    """train_with_curriculum, then the post-training threshold calibration
    when `cfg.calibrate_thresholds` — one step, so the primary and every
    ensemble member get the same treatment. Returns (dyn_sys, best_dev_loss,
    calibration); calibration is {} without ordinal heads or with the flag
    off."""
    dyn_sys, best_dev_loss = train_with_curriculum(
        cfg=cfg, train_dfs=train_dfs, dev_dfs=dev_dfs,
        nm_mapping=nm_mapping, grad_masks=grad_masks, norm_stats=norm_stats,
        head_stack=head_stack, block_spec=block_spec, global_logger=logger,
    )
    calibration: dict[str, Calibration] = {}
    if cfg.calibrate_thresholds:
        calibration = calibrate_decision_thresholds(dyn_sys, cfg, dev_dfs, nm_mapping)
        if cfg.progress in ("live", "log") and calibration:
            for name, c in calibration.items():
                print(f"Calibrated decision thresholds for {name!r}: "
                      + ", ".join(f"{t:.4f}" for t in c.thresholds)
                      + f"  ({len(c.trace)} pass(es), kept #{c.kept + 1})")
    return dyn_sys, best_dev_loss, calibration


# ── Result type ──────────────────────────────────────────────────


@dataclass
class TestResults:
    """Bundle of test-time outputs. Tensors stay internal to evaluate_and_plot."""
    metrics: list[ChannelMetrics]
    stability: list[RolloutStability]  # rollout shape, to catch ramping models
    normalized_rmse: float  # RMSE over every channel, each normalised by its stats
    test_time_s: float
    # Physics assertions on the thermal states. None when the use case opted
    # out of synth_cfg.
    physics: Optional[list[ScenarioResult]] = None
    # Rollout length the scenarios ran for, so the reportcard can state it.
    physics_steps: Optional[int] = None
    # Order statistics for the Y channels that are not thermal states. These
    # carry no verdict — see scenarios.ReadoutSummary.
    readouts: Optional[list[ReadoutSummary]] = None
    # Classification view of the binary / ordinal channels: NRMSE cannot tell
    # a head that works from one that never fires, these can. Empty when the
    # stack has no levelled head.
    levels: list[LevelMetrics] = field(default_factory=list)
    # The same metrics for persistence — hold the last seeded state for the
    # whole window — as the bar a surrogate has to clear. A regression head on
    # a staged channel learns exactly this and then collapses in the rollout.
    persistence: list[ChannelMetrics] = field(default_factory=list)
    persistence_levels: list[LevelMetrics] = field(default_factory=list)


@dataclass
class TrainedModel:
    """One finished training run: its MLflow id and what it produced."""
    run_id: str
    dyn_sys: object
    best_dev_loss: float
    train_time_s: float
    test_results: TestResults
    # None unless cfg.ensemble_seeds > 1.
    ensemble: Optional[EnsembleSpread]


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
    plot_run_dir: Optional[Path],
    logger=None,
    grad_masks: Optional[dict] = None,
) -> TestResults:
    """
    Roll the model out on the test split, compute per-channel metrics,
    and (if cfg.plot) drop prediction + synth plots in plot_run_dir, which
    may be None only when cfg.plot is off.

    The raw rollout tensors do not escape this function — anything a
    caller needs ends up in TestResults or in the on-disk plots.

    `test_dfs` is always a list (length >= 1). Windows from every file
    are pooled into one rollout and the resulting per-channel metrics
    are pooled too; `_plot_predictions` routes each batch's plot to its
    originating df via `source_idx`.

    `synth_cfg=None` disables the synth subsystem (no plots, no eval).
    """
    if cfg.plot and plot_run_dir is None:
        raise ValueError("cfg.plot is on but no plot_run_dir was given.")
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

    # System output prepends the initial condition; slice past the history_steps
    # history when comparing to ground truth.
    pred = test_out["yn"][:, cfg.history_steps:-1, :]
    true = test_out["Y"][:, cfg.history_steps:, :]
    metrics = per_channel_metrics(pred, true, nm_mapping["Y"])
    # RMSE alone cannot see a model that has stopped responding to its inputs
    # and is emitting a ramp; compare the rollout's shape to the truth's too.
    stability = rollout_stability(pred, true, nm_mapping["Y"])

    # Persistence sees exactly what the rollout sees at its start — the
    # seeded history — and holds the last of it. Same span, same units.
    persist = test_out["Y"][:, cfg.history_steps - 1:cfg.history_steps, :].expand_as(true)
    persistence = per_channel_metrics(persist, true, nm_mapping["Y"])
    head_stack = trained_heads(dyn_sys, norm_stats)
    levels = _level_metrics(head_stack, pred, true)
    persistence_levels = _level_metrics(head_stack, persist, true)

    # One unitless headline number: pooling channels is only meaningful
    # with each normalised by its stats.
    model = contract_model(dyn_sys)
    normalized_rmse = (model.normalize("Y", pred) - model.normalize("Y", true)
                       ).pow(2).mean().sqrt().item()

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
        normalized_rmse=normalized_rmse,
        test_time_s=test_time_s,
        physics=None if outcome is None else outcome.verdicts,
        physics_steps=None if outcome is None else synth_steps(cfg, freq_s),
        readouts=None if outcome is None else outcome.readouts,
        levels=levels,
        persistence=persistence,
        persistence_levels=persistence_levels,
    )


def trained_heads(dyn_sys, norm_stats: dict) -> Optional[HeadStack]:
    """The trained model's head stack in physical units, calibrated decision
    thresholds included, or None without heads."""
    stack = normalized_core(dyn_sys).head_stack
    return None if stack is None else stack.physical(norm_stats)


def _level_metrics(
    head_stack: Optional[HeadStack],
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> list[LevelMetrics]:
    """Classification metrics for every levelled channel of the (physical) stack."""
    if head_stack is None:
        return []
    out = []
    for i, (name, head) in enumerate(zip(head_stack.channel_names, head_stack.heads)):
        head_levels = head.levels()
        if head_levels is None:
            continue
        out.append(level_metrics(
            head.level_index(y_pred[..., i]), head.level_index(y_true[..., i]),
            head_levels.tolist(), name,
        ))
    return out


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
    src = source_idx.detach().cpu().tolist()
    for batch in range(test_out["yn"].shape[0]):
        pred_dict = {
            "Y": test_out["yn"][batch, :-1, :].detach().cpu().numpy(),
            "t": test_out["t"][batch].detach().cpu().numpy(),
        }
        df_pred = to_df(data=pred_dict, nm_mapping=nm_mapping)
        plot_predictions(
            cfg=cfg,
            df_true=test_dfs[src[batch]], df_pred=df_pred,
            run_dir=str(plot_run_dir),
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
        history_steps=cfg.history_steps,
        n_scenarios=cfg.synth_n_scenarios,
        n_steps=synth_steps(cfg, freq_s),
        device=cfg.device,
        grad_masks=grad_masks,
    )
    if scenarios is None:
        return None

    outcome = run_scenarios(scenarios)
    if cfg.progress in ("live", "log"):
        print(format_checks(outcome.verdicts))

    if cfg.plot:
        synth_dir = plot_run_dir / "synthetic"
        plot_synthetic(
            [s.trace(cfg.synth_n_plot) for s in scenarios],
            run_dir=str(synth_dir), norm_stats=norm_stats,
            freq_s=freq_s, n_plot=cfg.synth_n_plot,
        )
        if logger is not None:
            logger.log_artifact(synth_dir, artifact_path="plots")

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
    split_sizes: Optional[dict[str, dict[str, int]]] = None,
    calibration: Optional[dict[str, Calibration]] = None,
    ensemble: Optional[EnsembleSpread] = None,
) -> str:
    """
    Render the markdown reportcard for one run.

    `norm_stats` feeds the per-column normalization labels in the Data
    section. `test_dfs` feeds the bounded-channel coverage check, against
    `head_stack`'s bounds; both are in physical units. Multi-file splits are
    concatenated for the coverage check only.

    `ensemble` is the optional cfg.ensemble_seeds > 1 spread (see
    train_and_evaluate); None renders no extra section.
    """
    arch = describe_dynamics(dyn_sys, cfg)
    test_df_for_coverage = None if test_dfs is None else pool_dfs(test_dfs)
    return _render_reportcard(
        cfg=cfg,
        arch=arch,
        test_metrics=test_results.metrics,
        stability=test_results.stability,
        physics=test_results.physics,
        physics_steps=test_results.physics_steps,
        readouts=test_results.readouts,
        levels=test_results.levels,
        persistence=test_results.persistence,
        persistence_levels=test_results.persistence_levels,
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
        calibration=calibration,
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

    flat: dict[str, float] = {"test_rmse": test_results.normalized_rmse}
    for m in test_results.metrics:
        safe = metric_key(m.name)
        flat[f"test_{safe}_rmse"] = m.rmse
        flat[f"test_{safe}_mae"] = m.mae
        flat[f"test_{safe}_r2"] = m.r2
        if not math.isnan(m.nrmse):
            flat[f"test_{safe}_nrmse"] = m.nrmse
    for st in test_results.stability:
        safe = metric_key(st.name)
        flat[f"test_{safe}_rise_excess"] = st.rise_excess
        flat[f"test_{safe}_drift_ratio"] = st.drift_ratio
    # Balanced accuracy is the number to sort a staged channel by — NRMSE
    # ranks a head that never fires above one that does.
    for lm in test_results.levels:
        flat[f"test_{metric_key(lm.name)}_balanced_acc"] = lm.balanced_accuracy
    for m in test_results.persistence:
        flat[f"persistence_{metric_key(m.name)}_rmse"] = m.rmse
    for lm in test_results.persistence_levels:
        flat[f"persistence_{metric_key(lm.name)}_balanced_acc"] = lm.balanced_accuracy
    if test_results.physics is not None:
        # One gate number to sort a leaderboard by, plus per-(scenario, channel)
        # detail — so a channel someone does not care about can be excluded from
        # a comparison rather than quietly moving the aggregate.
        flat["physics_checks_passed"] = float(
            sum(r.passed for r in test_results.physics)
        )
        flat["physics_checks_total"] = float(len(test_results.physics))
        for r in test_results.physics:
            key = metric_key(r.key)
            flat[f"physics_{key}"] = r.value
            flat[f"physics_{key}_passed"] = float(r.passed)

    logger.log_metrics(flat)
    logger.log_dict(
        metrics_to_dict(test_results.metrics), name="test_metrics.json"
    )
    logger.log_dict(
        stability_to_dict(test_results.stability),
        name="rollout_stability.json",
    )
    logger.log_dict(
        metrics_to_dict(test_results.persistence), name="persistence_metrics.json"
    )
    if test_results.levels:
        logger.log_dict(
            {
                "model": level_metrics_to_dict(test_results.levels),
                "persistence": level_metrics_to_dict(test_results.persistence_levels),
            },
            name="level_metrics.json",
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


@beartype
def run_summary_metrics(test_results: TestResults) -> dict[str, float]:
    """The two headline numbers a submission audit checks, for run_summary.json.

    `worst_channel_r2` is the min of the per-channel R² that
    `log_test_results` already logs, so it is the weakest channel rather
    than an average that can hide one. `worst_level_balanced_acc` is the
    same idea for the staged channels, present only when there are any.
    """
    out = {
        "test_rmse": test_results.normalized_rmse,
        "worst_channel_r2": min(m.r2 for m in test_results.metrics),
    }
    if test_results.levels:
        out["worst_level_balanced_acc"] = min(
            lm.balanced_accuracy for lm in test_results.levels
        )
    return out


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
    per_seed_levels: list[list[LevelMetrics]],
) -> EnsembleSpread:
    """Group each metric's per-seed values under its channel/key and reduce
    with `spread_stats`. Every list argument has one entry per seed, in the
    same order (primary first)."""
    rmse: dict[str, list[float]] = {}
    for metrics in per_seed_metrics:
        for m in metrics:
            rmse.setdefault(m.name, []).append(m.rmse)

    balanced_acc: dict[str, list[float]] = {}
    for levels in per_seed_levels:
        for lm in levels:
            balanced_acc.setdefault(lm.name, []).append(lm.balanced_accuracy)

    physics: dict[str, list[float]] = {}
    physics_passed: dict[str, int] = {}
    for results in per_seed_physics:
        for r in (results or []):
            physics.setdefault(r.key, []).append(r.value)
            physics_passed[r.key] = physics_passed.get(r.key, 0) + int(r.passed)

    tau_slow: dict[str, list[float]] = {}
    for taus in per_seed_tau_slow:
        for channel, value in taus.items():
            tau_slow.setdefault(channel, []).append(value)

    return EnsembleSpread(
        n_seeds=n_seeds,
        rmse={k: spread_stats(v) for k, v in rmse.items()},
        physics={k: spread_stats(v) for k, v in physics.items()},
        tau_slow={k: spread_stats(v) for k, v in tau_slow.items()},
        balanced_acc={k: spread_stats(v) for k, v in balanced_acc.items()},
        physics_passed=physics_passed,
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
    quiet_cfg = replace(cfg, plot=False, progress="silent")

    per_seed_metrics = [primary_test_results.metrics]
    per_seed_physics = [primary_test_results.physics]
    per_seed_tau_slow = [_tau_slow_by_channel(describe_dynamics(primary_dyn_sys, cfg))]
    per_seed_levels = [primary_test_results.levels]

    for i in range(1, cfg.ensemble_seeds):
        if cfg.progress != "silent":
            print(f"Training ensemble member {i + 1}/{cfg.ensemble_seeds} "
                  f"(seed={cfg.seed + i})...")
        set_seeds(cfg.seed + i)
        dyn_sys, _, _ = _train(
            quiet_cfg,
            train_dfs=train_dfs, dev_dfs=dev_dfs,
            nm_mapping=nm_mapping, grad_masks=grad_masks, norm_stats=norm_stats,
            head_stack=head_stack, block_spec=block_spec, logger=None,
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
        per_seed_levels.append(test_results.levels)

    return _build_ensemble_spread(
        cfg.ensemble_seeds, per_seed_metrics, per_seed_physics, per_seed_tau_slow,
        per_seed_levels,
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
) -> TrainedModel:
    """
    Train one model, evaluate it, plot, and write the reportcard.

    Used by `nt model train` end-to-end and by `nt model optuna` for the
    Phase 2 retrain with best params. The lighter per-trial loop in
    cli/optuna.objective calls train_with_curriculum directly — it
    deliberately skips eval/plot/reportcard for speed.

    When `cfg.ensemble_seeds > 1`, this also trains that many more models at
    cfg.seed + 1 .. cfg.seed + ensemble_seeds - 1 (see `_run_ensemble_members`)
    and reports the spread across all of them. The primary run (this one, at
    cfg.seed + 0) stays the one whose weights are logged, whose plots are
    written, and whose reportcard is rendered — the ensemble adds error bars
    around that one artifact, it does not change what artifact you get.

    `extra_metrics` lets the caller log run-level numbers that aren't part
    of the standard set (e.g. optuna's `search_time`).

    Returns the run as a `TrainedModel`, its id read off `logger`.
    """
    train_t0 = tic()
    # Reseed here rather than trust the caller to have left the RNG at
    # cfg.seed: the ensemble loop below reseeds at cfg.seed + i for i > 0,
    # and the primary (i = 0) needs the same explicit contract rather than
    # an implicit "whatever the caller last seeded".
    set_seeds(cfg.seed)
    dyn_sys, best_dev_loss, calibration = _train(
        cfg,
        train_dfs=train_dfs, dev_dfs=dev_dfs,
        nm_mapping=nm_mapping, grad_masks=grad_masks, norm_stats=norm_stats,
        head_stack=head_stack, block_spec=block_spec, logger=logger,
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
    # used at training time, independent of user_defined.py drift. Read off
    # the System's own stack: that is the one carrying the calibrated
    # decision thresholds, which are not in the saved weights.
    trained_stack = trained_heads(dyn_sys, norm_stats)
    if trained_stack is not None:
        logger.log_dict(trained_stack.to_config(), name="head_config.json")

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
        head_stack=trained_stack,
        test_dfs=test_dfs,
        run_name=cfg.run,
        best_dev_loss=best_dev_loss,
        train_time_s=train_time_s,
        split_sizes={
            name: {"rows": sum(len(d) for d in dfs), "files": len(dfs)}
            for name, dfs in (("train", train_dfs), ("dev", dev_dfs), ("test", test_dfs))
        },
        calibration=calibration,
        ensemble=ensemble,
    )
    write_reportcard(md, Path(cfg.savedir), logger=logger)

    return TrainedModel(
        run_id=logger.run.info.run_id,
        dyn_sys=dyn_sys,
        best_dev_loss=best_dev_loss,
        train_time_s=train_time_s,
        test_results=test_results,
        ensemble=ensemble,
    )
