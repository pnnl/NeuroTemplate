"""
Per-run reportcard.

A single markdown summary dropped in each run's artifact directory and
logged to MLflow. Sections (in order):

  1. Summary                — run name, timestamp, dev_loss, train_time
  2. Architecture           — factory, separable/residual, param count
  3. Data                   — split sizes, Y/U/D classification,
                              per-column normalization, head kinds, and any
                              column left in raw units because it was constant
  4. Test metrics           — per-channel RMSE/MAE/R²/NRMSE in physical units
  5. Rollout stability      — per-channel rise fraction and drift vs. the
                              truth's own, flagging models that have stopped
                              responding to their inputs and are ramping
  6. Physics checks         — equilibrium / step response / free decay, each a
                              pass-fail assertion per thermal state channel, on
                              synthetic states the recorded data never contains
  6b. Readout channels      — order statistics for the Y channels that are not
                              thermal states; reported without a verdict
  6c. Seed ensemble spread  — min/median/max of test RMSE, physics-check
                              values, and tau_slow across cfg.ensemble_seeds
                              independently-trained models. Present only when
                              ensemble_seeds > 1; every other number on this
                              page is a single seed's draw.
  8. Bounded channel coverage — for each bounded head, whether any test
                              samples fall outside the head's [lo, hi]
  9. Hyperparameters        — lr, batch_size, train_pred_steps, curriculum, gains

The "auto-detected anomalies" section listed in updates.md is left for
a future, more creative pass and intentionally not implemented here.

Single entry point: render_reportcard(...) returns the markdown string.
The caller (pipeline.py) is responsible for writing it to disk and
logging it to MLflow.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from neurotemplate.common.data_utils import degenerate_mapped_columns
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.heads import BoundedHead, HeadStack
from neurotemplate.common.metrics import (
    ChannelMetrics,
    RolloutStability,
    metrics_to_markdown,
    stability_to_markdown,
)
from neurotemplate.modeling.scenarios import (
    ReadoutSummary,
    ScenarioResult,
    checks_to_markdown,
    readouts_to_markdown,
)
from neurotemplate.common.reportcard import _fmt_float, _section

from neurotemplate._typing import beartype


# ── Seed ensemble spread ──────────────────────────────────────────
#
# Every number elsewhere on the reportcard comes from one trained model —
# one draw from whatever the seed happened to give. `pipeline.train_and_
# evaluate` optionally trains `cfg.ensemble_seeds` models (see that field's
# comment in ModelingConfig) and hands the per-seed results here to be
# turned into the spread these types describe. `spread_stats` is also used
# directly by cli/optuna.py to report the spread of a trial's per-seed dev
# losses — the same order-statistics question, asked of a different metric.


@dataclass(frozen=True)
class SeedSpread:
    """min/median/max of one metric across an ensemble of seeds.

    Order statistics, not mean +/- std: a handful of seeds is nowhere near
    enough to trust a Gaussian assumption, and printing a stray min or max
    is honest about that instead of implying more precision than a handful
    of draws support.
    """
    min: float
    median: float
    max: float


def spread_stats(values: list[float]) -> SeedSpread:
    """Build a SeedSpread from one metric's per-seed values."""
    return SeedSpread(min=min(values), median=statistics.median(values), max=max(values))


@dataclass(frozen=True)
class EnsembleSpread:
    """Spread across `cfg.ensemble_seeds` independently-trained models.

    Built by pipeline.train_and_evaluate from the same TestResults and
    describe_dynamics structures a single-seed run already produces — see
    ModelingConfig.ensemble_seeds. Each dict is keyed by whatever
    distinguishes one of its rows from another; empty dicts render nothing.
    """
    n_seeds: int
    rmse: dict[str, SeedSpread]        # channel -> spread
    physics: dict[str, SeedSpread]     # "scenario | channel" -> spread
    tau_slow: dict[str, SeedSpread]    # channel -> spread (2r2c models only)


def _spread_table(rows: list[tuple[str, SeedSpread]]) -> list[str]:
    """A 4-column min/median/max markdown table, one row per named spread."""
    lines = ["| Channel | Min | Median | Max |", "|---|---:|---:|---:|"]
    for name, s in rows:
        lines.append(f"| {name} | {s.min:.4g} | {s.median:.4g} | {s.max:.4g} |")
    return lines


def _ensemble_section(ensemble: Optional[EnsembleSpread]) -> Optional[str]:
    """Min/median/max across the seed ensemble, one table per metric family.

    Returns None when there is no ensemble (cfg.ensemble_seeds == 1), which
    is what keeps that case's rendering byte-identical to a run built before
    this feature existed.
    """
    if ensemble is None:
        return None

    body = [
        f"Across **{ensemble.n_seeds}** independently-trained seeds "
        f"(`cfg.seed` .. `cfg.seed + {ensemble.n_seeds - 1}`). Only the "
        "first seed's weights are logged, plotted, and behind every other "
        "number on this page — this section is the error bar around them.\n"
    ]

    if ensemble.rmse:
        body.append("**Test RMSE (physical units):**\n")
        body += _spread_table(sorted(ensemble.rmse.items()))
        body.append("")

    if ensemble.tau_slow:
        body.append("**`tau_slow` (hours):**\n")
        body += _spread_table(sorted(ensemble.tau_slow.items()))
        body.append("")

    if ensemble.physics:
        body.append("**Physics-check values:**\n")
        body.append("| Scenario | Channel | Min | Median | Max |")
        body.append("|---|---|---:|---:|---:|")
        for key, s in sorted(ensemble.physics.items()):
            scenario, channel = key.split(" | ", 1)
            body.append(f"| {scenario} | `{channel}` | {s.min:.4g} | "
                        f"{s.median:.4g} | {s.max:.4g} |")

    return _section("Seed ensemble spread", "\n".join(body))


def _head_label(head) -> str:
    """Short label for a Head instance, e.g. 'bounded [0, +∞]' or 'ordinal K=4'."""
    label = head.kind or type(head).__name__
    if hasattr(head, "lo") and hasattr(head, "hi"):
        lo = "−∞" if head.lo == float("-inf") else f"{head.lo:g}"
        hi = "+∞" if head.hi == float("inf") else f"{head.hi:g}"
        label += f" [{lo}, {hi}]"
    if hasattr(head, "n_levels"):
        label += f" K={head.n_levels}"
    return label


def _summary_section(
    run_name: str,
    timestamp: str,
    best_dev_loss: Optional[float],
    train_time_s: Optional[float],
    test_time_s: Optional[float],
) -> str:
    rows = [
        f"- **Run:** `{run_name}`",
        f"- **Timestamp:** {timestamp}",
        f"- **Best dev loss:** {_fmt_float(best_dev_loss)}",
    ]
    if train_time_s is not None:
        rows.append(f"- **Train time:** {train_time_s:.1f} s")
    if test_time_s is not None:
        rows.append(f"- **Test inference time:** {test_time_s:.2f} s")
    return _section("Summary", "\n".join(rows))


def _block_lines(name: str, info: Optional[dict], indent: str = "") -> list[str]:
    """One bullet for a block, plus nested bullets for what it reports itself.

    A per-block stack (BlockStack) nests one bullet per block, labelled with
    its family — the interesting fact once a spec can mix families in one
    graph. A 2R2C block adds the time constants it learned. `tau_slow` is
    the storage number — how long a temperature difference survives — so it
    is worth reading against what you know about the building.
    """
    if info is None:
        return []
    parts = [f"`{info['class']}`", f"params={info['param_count']:,}"]
    if "hsizes" in info and info["hsizes"]:
        parts.append(f"hsizes={info['hsizes']}")
    if "in_features" in info:
        parts.append(f"in={info['in_features']}")
    if "out_features" in info:
        parts.append(f"out={info['out_features']}")
    lines = [f"{indent}- **{name}:** " + ", ".join(parts)]

    for channel, taus in info.get("time_constants_h", {}).items():
        lines.append(f"{indent}  - {channel}: tau_fast={taus['fast']:.2f} h, "
                     f"tau_slow={taus['slow']:.2f} h")
    for block_name, sub in info.get("blocks", {}).items():
        label = f"block `{block_name}`"
        if sub is not None and sub.get("model"):
            label += f" ({sub['model']})"
        lines += _block_lines(label, sub, indent + "  ")
    return lines


def _architecture_section(arch: dict) -> str:
    lines = [
        f"- **Factory:** `{arch['factory']}`",
        f"- **Separable:** {arch['separable']}",
        f"- **Residual:** {arch['residual']}",
        f"- **Shapes:** in_steps={arch['in_steps']}, "
        f"ny={arch['ny']}, nu={arch['nu']}, nd={arch['nd']}",
        f"- **Total trainable params:** {arch['param_count']:,}",
    ]
    blocks = arch.get("blocks", {})
    block_lines = [
        line
        for key in ("fy", "fu", "fd")
        for line in _block_lines(key, blocks.get(key))
    ]
    if block_lines:
        lines.append("")
        lines.append("Blocks:")
        lines.extend(block_lines)
    return _section("Architecture", "\n".join(lines))


def _norm_label(stats: Optional[dict]) -> str:
    """Short normalization tag for a column, e.g. 'min_max' or 'none'."""
    if stats is None:
        return "?"
    return stats.get("norm_type", "?")


def _data_section(
    nm_mapping: dict,
    split_sizes: Optional[dict[str, object]],
    head_stack: Optional[HeadStack] = None,
    norm_stats: Optional[dict] = None,
) -> str:
    """`split_sizes` entries may be plain ints (legacy single-file shape)
    or dicts of the form {"rows": int, "files": int} (multi-file). When
    `files == 1` only the row count is rendered, so single-file reports
    are byte-identical to the pre-change output."""
    def cols(key: str) -> str:
        names = nm_mapping.get(key, [])
        if not names:
            return "*(none)*"
        lines = []
        for name in names:
            entry = f"`{name}`"
            tags = []
            if norm_stats is not None:
                tags.append(f"norm: {_norm_label(norm_stats.get(name))}")
            if key == "Y" and head_stack is not None:
                idx = head_stack.channel_names.index(name)
                label = _head_label(head_stack.heads[idx])
                if label != "regression":
                    tags.append(label)
            if tags:
                entry += " *(" + "; ".join(tags) + ")*"
            lines.append(entry)
        return "\n  - " + "\n  - ".join(lines)

    def _fmt_size(v):
        if isinstance(v, dict):
            rows = v.get("rows", 0)
            files = v.get("files", 1)
            return f"{rows:,}" if files == 1 else f"{rows:,} across {files} files"
        return f"{int(v):,}"

    parts = []
    if split_sizes:
        size_rows = ", ".join(
            f"{k}={_fmt_size(v)}" for k, v in split_sizes.items()
        )
        parts.append(f"- **Split sizes (rows):** {size_rows}")
    parts.append(f"- **Y ({len(nm_mapping.get('Y', []))} states):**" + cols("Y"))
    parts.append(f"- **U ({len(nm_mapping.get('U', []))} controls):**" + cols("U"))
    parts.append(f"- **D ({len(nm_mapping.get('D', []))} disturbances):**" + cols("D"))
    if norm_stats is not None:
        frozen = degenerate_mapped_columns(norm_stats, nm_mapping)
        if frozen:
            parts.append(
                "\n**⚠️ Constant in this data:** "
                + ", ".join(f"`{c}`" for c in frozen)
                + ". Normalization had nothing to scale, so these columns fall "
                "back to raw physical units. If any of them varies at evaluation "
                "time — a furnace that never fired here but does there — its "
                "value arrives orders of magnitude out of range and the rollout "
                "diverges. Either exclude the column from the mapping or widen "
                "the excitation so training covers the regime."
            )
    return _section("Data", "\n".join(parts))


def _bounded_coverage_section(
    head_stack: Optional[HeadStack],
    test_df: Optional[pd.DataFrame],
) -> Optional[str]:
    """
    For each bounded Y channel, check whether the test set contains any
    values outside the head's [lo, hi]. A non-zero count means the channel
    was set up to fail — the model literally cannot reach those samples
    because the head clamps/squashes them away.

    `test_df` must already be in the same space the head consumes (the
    normalized test split). Bounds and test values are compared directly;
    no extra projection.

    Returns None if any input is missing (section is just skipped).
    """
    if head_stack is None or test_df is None:
        return None

    rows = []
    for i, head in enumerate(head_stack.heads):
        if not isinstance(head, BoundedHead):
            continue
        name = head_stack.channel_names[i]
        if name not in test_df.columns:
            continue
        vals = test_df[name].to_numpy()
        n_below = int((vals < head.lo).sum())
        n_above = int((vals > head.hi).sum())
        n_total = vals.size
        n_out = n_below + n_above
        pct = 100.0 * n_out / n_total if n_total else 0.0
        lo_s = "−∞" if head.lo == float("-inf") else f"{head.lo:g}"
        hi_s = "+∞" if head.hi == float("inf") else f"{head.hi:g}"
        rows.append((
            name,
            f"{vals.min():.4g}",
            f"{vals.max():.4g}",
            f"[{lo_s}, {hi_s}]",
            f"{n_below}",
            f"{n_above}",
            f"{pct:.2f}%",
        ))

    if not rows:
        return None

    header = (
        "| Channel | Test min (norm) | Test max (norm) | Bounds (norm) | "
        "# below lo | # above hi | % outside |"
    )
    sep = "|" + "|".join(["---"] * 7) + "|"
    body_lines = [
        "Bounds and test values are both in normalized space (head bounds "
        "are interpreted in the space the head consumes). A non-zero "
        "outside count means the channel was set up to fail — the model "
        "literally cannot reach those samples.\n",
        header,
        sep,
    ]
    for r in rows:
        body_lines.append("| " + " | ".join(r) + " |")
    return _section("Bounded channel coverage", "\n".join(body_lines))


def _metrics_section(metrics: list[ChannelMetrics]) -> str:
    if not metrics:
        return _section("Test Metrics (per channel)", "*(no metrics computed)*")
    body = (
        "All values are in the physical units of each channel. "
        "NRMSE is normalized by the channel range (max − min) of the test set.\n\n"
        + metrics_to_markdown(metrics)
    )
    return _section("Test Metrics (per channel)", body)


def _stability_section(
    stability: list[RolloutStability], test_window: Optional[int] = None
) -> str:
    """Rollout shape vs. the truth's own shape — catches ramping models.

    A model that has stopped responding to its inputs can still post a
    tolerable RMSE on a short test window and then diverge by many degrees in
    closed loop. Comparing rise fraction and drift against the ground truth
    surfaces that before a policy is ever trained on the model.

    Everything here is measured over `test_window` steps, so the horizon is
    stated alongside the table. Two of four measured submissions passed this
    check at a 47-step window and diverged past ~300 steps on the same data;
    the diagnostic was right, it was just asked about a horizon far shorter
    than the one the controller would be evaluated over.
    """
    if not stability:
        return _section("Rollout stability", "*(no stability computed)*")
    flagged = [s.name for s in stability if s.looks_like_ramp()]
    body = stability_to_markdown(stability)
    if test_window is not None:
        body = (
            f"Measured over a **{test_window}-step** rollout. A model can be "
            "stable here and diverge over a longer one, so treat a short window "
            "as a weak check: closed-loop evaluation runs for thousands of "
            "steps.\n\n"
        ) + body
    if flagged:
        body += (
            "\n\n**⚠️ Likely ramping:** "
            + ", ".join(f"`{n}`" for n in flagged)
            + ". These channels move nearly monotonically, unlike the truth. A "
            "monotone rollout means the model is ignoring its inputs; RMSE on a "
            "short window will not show it. Free-run the model over a long "
            "horizon before training a controller on it."
        )
    # A high excess with a mid-range rise fraction is the staged-equipment case,
    # not a ramp: the truth holds a constant value most of the time and the model
    # jitters instead of holding. Say so, or the table reads as eight failures.
    noisy = [s.name for s in stability
             if s.rise_excess > 0.25 and not s.looks_like_ramp()]
    if noisy:
        body += (
            "\n\nHigh rise excess but *not* monotone: "
            + ", ".join(f"`{n}`" for n in noisy)
            + ". This is the expected signature for staged equipment (fans, "
            "compressors, flow), whose truth holds one value for long stretches "
            "while a continuous model jitters around it. Judge these channels on "
            "RMSE and R², not on rise excess."
        )
    return _section("Rollout stability", body)


def _physics_section(results: list[ScenarioResult],
                     hours: Optional[float] = None,
                     n_steps: Optional[int] = None) -> str:
    """Assertions on synthetic states the recorded data never contains.

    The test split cannot ask whether the model has learned any physics,
    because it never contains the HVAC fully off or the weather stepping to a
    new value. These do, and each answer is a number in physical units and a
    verdict rather than a loss.
    """
    if not results:
        return _section("Physics checks", "*(no scenarios ran)*")
    failed = [r for r in results if not r.passed]
    horizon = (
        f" Each rollout runs **{hours:g} h** ({n_steps} steps)."
        if hours is not None and n_steps is not None else ""
    )
    body = [
        f"{len(results) - len(failed)} of {len(results)} passed.{horizon} Each "
        "scenario puts the surrogate in a state the training data never "
        "visits and asserts what has to happen next, once per state channel — "
        "so a channel you are not relying on cannot hide one you are.\n",
        checks_to_markdown(results),
    ]
    if failed:
        body += [
            "",
            "**⚠️ Failed:** " + ", ".join(f"`{r.key}`" for r in failed)
            + ". A surrogate that fails these is not physically coherent, "
            "whatever its NRMSE says, and a policy trained against it will "
            "be optimising against the incoherence. These scenarios are "
            "deliberately outside the training distribution — which is also "
            "where a controller will take the model — so widening the "
            "excitation is as valid a fix as changing the architecture.",
        ]
    return _section("Physics checks", "\n".join(body))


def _readouts_section(readouts: list[ReadoutSummary]) -> Optional[str]:
    """What the non-state channels did, with no verdict attached.

    Readouts are seeded at their resting value and then run free — the model's
    own predictions feed back every step. Nothing asserts on them, because what
    a readout *should* do depends on the scenario (hold at off during
    equilibrium, move under a step response). The spread is still worth seeing:
    a readout that goes somewhere absurd is also an input to the next state
    prediction, which is otherwise an unattributable drift in the table above.
    """
    if not readouts:
        return None
    body = [
        "Y channels not named in `states`, so nothing above asserts on them. "
        "Statistics are over the free-running steps of each scenario — the "
        "seeded prefix is the channel's own resting value by construction, and "
        "is shown separately as the reference point.\n",
        readouts_to_markdown(readouts),
    ]
    return _section("Readout channels (reported, not asserted)",
                    "\n".join(body))


def _hyperparameters_section(cfg: ModelingConfig) -> str:
    rows = [
        f"- **lr:** {cfg.lr}",
        f"- **batch_size:** {cfg.batch_size}",
        f"- **epochs:** {cfg.epochs}",
        f"- **train_pred_steps:** {cfg.train_pred_steps}",
        f"- **dev_pred_steps:** {cfg.dev_pred_steps}",
        f"- **test_window:** {cfg.test_window}",
        f"- **curriculum_steps:** {cfg.curriculum_steps}, "
        f"ratio={cfg.curriculum_ratio}",
        f"- **norm_type (default):** {cfg.norm_type} "
        "*(per-column overrides reported in the Data section)*",
        f"- **seed:** {cfg.seed}",
    ]
    gains = []
    if cfg.ref_gain != 0:
        gains.append(f"ref={cfg.ref_gain}")
    if cfg.vel_gain != 0:
        gains.append(f"vel={cfg.vel_gain}")
    if cfg.sgn_gain != 0:
        gains.append(f"sgn={cfg.sgn_gain}")
    if cfg.grad_gain != 0:
        gains.append(f"grad={cfg.grad_gain}")
    rows.append(f"- **Loss gains:** {', '.join(gains) if gains else 'ref-only'}")
    return _section("Hyperparameters", "\n".join(rows))


@beartype
def render_reportcard(
    cfg: ModelingConfig,
    arch: dict,
    test_metrics: list[ChannelMetrics],
    nm_mapping: dict[str, list[str]],
    *,
    stability: Optional[list[RolloutStability]] = None,
    physics: Optional[list[ScenarioResult]] = None,
    physics_steps: Optional[int] = None,
    readouts: Optional[list[ReadoutSummary]] = None,
    ensemble: Optional[EnsembleSpread] = None,
    head_stack: Optional[HeadStack] = None,
    norm_stats: Optional[dict] = None,
    test_df: Optional[pd.DataFrame] = None,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    best_dev_loss: Optional[float] = None,
    train_time_s: Optional[float] = None,
    test_time_s: Optional[float] = None,
    split_sizes: Optional[dict[str, object]] = None,
) -> str:
    """
    Render a markdown reportcard for one training run.

    Args:
        cfg:           Resolved ModelingConfig used for the run.
        arch:          Output of models.describe_dynamics(dyn_sys, cfg).
        test_metrics:  Output of metrics.per_channel_metrics(...).
        stability:     Output of metrics.rollout_stability(...). Omitted for
                       callers that only have per-channel metrics; the section
                       is then skipped.
        nm_mapping:    {"Y": [...], "U": [...], "D": [...]} from data_mapping.
        ensemble:      Optional EnsembleSpread from cfg.ensemble_seeds > 1
                       independently-trained models. None (the default)
                       renders no extra section, which is what keeps
                       ensemble_seeds=1 byte-identical to a run built before
                       this feature existed.
        head_stack:    Optional HeadStack; when provided, non-regression Y
                       columns are annotated with their head kind in the data
                       section (e.g. "bounded [0, +∞]").
        run_name:      MLflow run name (falls back to cfg.run).
        timestamp:     ISO-ish timestamp string (defaults to now()).
        best_dev_loss: Best dev loss across training.
        train_time_s:  Total training wall-clock seconds.
        test_time_s:   Test rollout wall-clock seconds.
        split_sizes:   Optional {"train": N, "dev": N, "test": N} row counts.

    Returns:
        Markdown string. The caller writes it to disk and logs it.
    """
    name = run_name if run_name is not None else cfg.run
    ts = timestamp if timestamp is not None else datetime.now().isoformat(
        timespec="seconds"
    )

    parts = [
        f"# Reportcard: {name}\n",
        _summary_section(name, ts, best_dev_loss, train_time_s, test_time_s),
        _architecture_section(arch),
        _data_section(nm_mapping, split_sizes, head_stack, norm_stats),
        _metrics_section(test_metrics),
    ]
    if stability is not None:
        parts.append(_stability_section(stability, cfg.test_window))
    if physics is not None:
        parts.append(_physics_section(physics, cfg.synth_hours, physics_steps))
    if readouts is not None:
        readout_md = _readouts_section(readouts)
        if readout_md is not None:
            parts.append(readout_md)
    ensemble_md = _ensemble_section(ensemble)
    if ensemble_md is not None:
        parts.append(ensemble_md)
    coverage = _bounded_coverage_section(head_stack, test_df)
    if coverage is not None:
        parts.append(coverage)
    parts.append(_hyperparameters_section(cfg))
    return "\n".join(parts)
