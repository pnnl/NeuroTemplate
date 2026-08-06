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
  6. Bounded channel coverage — for each bounded head, whether any test
                              samples fall outside the head's [lo, hi]
  7. Hyperparameters        — lr, batch_size, train_pred_steps, curriculum, gains

The "auto-detected anomalies" section listed in updates.md is left for
a future, more creative pass and intentionally not implemented here.

Single entry point: render_reportcard(...) returns the markdown string.
The caller (pipeline.py) is responsible for writing it to disk and
logging it to MLflow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from neurotemplate.common.data_utils import degenerate_columns
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.heads import BoundedHead, HeadStack
from neurotemplate.common.metrics import (
    ChannelMetrics,
    RolloutStability,
    metrics_to_markdown,
    stability_to_markdown,
)
from neurotemplate.common.reportcard import _fmt_float, _section

from neurotemplate._typing import beartype


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


def _block_line(name: str, info: Optional[dict]) -> Optional[str]:
    if info is None:
        return None
    parts = [f"`{info['class']}`", f"params={info['param_count']:,}"]
    if "hsizes" in info and info["hsizes"]:
        parts.append(f"hsizes={info['hsizes']}")
    if "in_features" in info:
        parts.append(f"in={info['in_features']}")
    if "out_features" in info:
        parts.append(f"out={info['out_features']}")
    return f"- **{name}:** " + ", ".join(parts)


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
        line for line in (
            _block_line("fy", blocks.get("fy")),
            _block_line("fu", blocks.get("fu")),
            _block_line("fd", blocks.get("fd")),
        ) if line is not None
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
        frozen = degenerate_columns(norm_stats)
        mapped = {c for key in ("Y", "U", "D") for c in nm_mapping.get(key, [])}
        frozen = [c for c in frozen if c in mapped]
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
    if cfg.synth_gain != 0:
        gains.append(f"synth={cfg.synth_gain}")
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
    coverage = _bounded_coverage_section(head_stack, test_df)
    if coverage is not None:
        parts.append(coverage)
    parts.append(_hyperparameters_section(cfg))
    return "\n".join(parts)
