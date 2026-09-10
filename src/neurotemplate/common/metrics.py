"""
Per-channel evaluation metrics in physical units.

Computes RMSE, MAE, R², and NRMSE per Y channel rather than a single
aggregate, so each row of the report has units that make sense
(averaging across channels with different units is uninterpretable).

The single entry point is `per_channel_metrics(...)`, which takes
unnormalized [batch, time, ny] tensors plus the column-name list.

NRMSE is normalized by the per-channel range (max - min) of the true
values in this evaluation set. This is the most common convention in
system-ID literature and gives a bounded interpretation: 0.1 ≈ 10% of
channel range. If a channel is constant (max == min), NRMSE is set to
NaN to signal it's undefined.

None of those can see a rare event. A staged channel that is off 97% of
the time is scored *better* by NRMSE when the model never turns it on —
the best-NRMSE furnace surrogate in the head bake-off never fired once.
`level_metrics` is the classification view for such channels: per-level
recall and balanced accuracy (mean recall, so a constant predictor scores
1/K whatever the imbalance), plus the predicted-vs-true share of each
level, which is what distinguishes a working head from a constant.
"""

import math
import re
from dataclasses import dataclass
from typing import Optional

import torch

from neurotemplate._typing import beartype


_METRIC_KEY_SANITIZER = re.compile(r"[^A-Za-z0-9_.\-]+")


def _safe_metric_key(name: str) -> str:
    """Sanitize a column name into something MLflow accepts as a metric key."""
    return _METRIC_KEY_SANITIZER.sub("_", name).strip("_")


@dataclass
class ChannelMetrics:
    """Metrics for one Y channel, all in the channel's physical units."""
    name: str
    rmse: float
    mae: float
    r2: float
    nrmse: float  # NaN if channel is constant
    n_samples: int


@dataclass
class LevelMetrics:
    """Classification view of one levelled (binary / ordinal) Y channel.

    Every list is indexed by level, ascending, with `levels` in the channel's
    physical units. `recall[k]` is NaN when the truth never visits level k in
    this evaluation set — there is nothing to recall — and such levels are
    left out of the balanced accuracy rather than counted as zero.
    """
    name: str
    levels: list[float]
    true_share: list[float]
    pred_share: list[float]
    recall: list[float]
    n_samples: int

    @property
    def balanced_accuracy(self) -> float:
        """Mean recall over the levels the truth visits. A constant predictor
        scores 1/K (0.5 for binary) regardless of imbalance — that is the
        floor to read gains against, not 0."""
        present = [r for r in self.recall if not math.isnan(r)]
        return sum(present) / len(present) if present else float("nan")


@dataclass
class RolloutStability:
    """Shape of a free-run rollout for one channel, against the truth it tracks.

    RMSE says how far a prediction is from the truth; it does not say whether
    the prediction still *behaves* like the system. A model that has stopped
    responding to its inputs emits a monotone ramp, and a short rollout can
    hide that behind a tolerable RMSE while a long one diverges by 10 °C.

    Two numbers catch it, both compared against the truth rather than an
    absolute threshold (a genuinely monotone signal must not be flagged):

      - `rise_fraction`: share of consecutive steps that increase. A model
        tracking a cycling signal sits near the truth's own value; a ramp sits
        near 0 or 1.
      - `drift`: mean |last - first| per window, in physical units. A stable
        rollout returns near where it started; a ramp does not.

    `rise_excess` alone is not sufficient, which is worth spelling out because
    the obvious reading of it is wrong. Staged equipment (fan power, DX power,
    air flow) *holds* a constant value for most of a window — measured on real
    test data, 57-96% of its consecutive diffs are exactly zero — so its true
    rise fraction is near 0.03. Any continuous model jitters instead of holding
    and lands near 0.50, producing a rise excess of ~0.47 while tracking the
    signal perfectly well. Use `looks_like_ramp`, which additionally requires
    the prediction's own rise fraction to be *extreme*; jitter sits near 0.5, a
    ramp does not.
    """
    name: str
    pred_rise_fraction: float
    true_rise_fraction: float
    pred_drift: float
    true_drift: float

    @property
    def rise_excess(self) -> float:
        """|pred - true| rise fraction. 0 means the shape matches the truth.

        Large values mean "does not move like the truth", which includes both
        ramping and jittering on a held signal — see `looks_like_ramp` before
        concluding a model is ramping.
        """
        return abs(self.pred_rise_fraction - self.true_rise_fraction)

    def looks_like_ramp(self, excess_warn: float = 0.25,
                        extreme: float = 0.15) -> bool:
        """Whether this channel's rollout is monotone rather than merely noisy.

        Two conditions, because either alone gives false positives:
        the shape must differ from the truth's (`rise_excess > excess_warn`),
        *and* the prediction must be nearly monotone in one direction
        (rise fraction within `extreme` of 0 or 1). Measured on synthetic cases
        against a truth that holds 57% of the time: jitter scores a rise
        fraction of 0.484 and is not flagged, a genuine ramp scores 1.000 and
        is.
        """
        if self.rise_excess <= excess_warn:
            return False
        return (self.pred_rise_fraction < extreme
                or self.pred_rise_fraction > 1.0 - extreme)

    @property
    def drift_ratio(self) -> float:
        """Predicted drift over true drift. ~1 is healthy; large means a ramp.

        inf when the truth does not drift at all but the prediction does.
        """
        if self.true_drift > 0:
            return self.pred_drift / self.true_drift
        return float("inf") if self.pred_drift > 0 else 1.0


@beartype
def per_channel_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    channel_names: list[str],
) -> list[ChannelMetrics]:
    """
    Args:
        y_pred: [batch, time, ny] predictions in physical units.
        y_true: [batch, time, ny] ground truth in physical units.
        channel_names: column names for the ny axis (length must match).

    Returns:
        One ChannelMetrics per channel, in the order of channel_names.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred shape {tuple(y_pred.shape)} != y_true shape {tuple(y_true.shape)}"
        )
    if y_pred.ndim != 3:
        raise ValueError(
            f"Expected 3D [batch, time, ny] tensors; got {y_pred.ndim}D"
        )
    ny = y_pred.shape[-1]
    if len(channel_names) != ny:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries but tensors have ny={ny}"
        )

    # Detach + move to CPU once for stable scalar extraction.
    pred = y_pred.detach().cpu()
    true = y_true.detach().cpu()

    n_per_channel = pred.shape[0] * pred.shape[1]
    results: list[ChannelMetrics] = []

    for i, name in enumerate(channel_names):
        p = pred[:, :, i]
        t = true[:, :, i]
        err = p - t

        rmse = err.pow(2).mean().sqrt().item()
        mae = err.abs().mean().item()

        # R²: 1 - SS_res / SS_tot. Undefined when SS_tot == 0
        # (constant truth) — fall back to 0.0 to keep it a real number,
        # the NaN NRMSE flags the constant-channel case anyway.
        ss_res = err.pow(2).sum().item()
        t_mean = t.mean().item()
        ss_tot = (t - t_mean).pow(2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        t_range = t.max().item() - t.min().item()
        nrmse = rmse / t_range if t_range > 0 else float("nan")

        results.append(ChannelMetrics(
            name=name,
            rmse=rmse,
            mae=mae,
            r2=r2,
            nrmse=nrmse,
            n_samples=int(n_per_channel),
        ))

    return results


@beartype
def level_metrics(
    pred_idx: torch.Tensor,
    true_idx: torch.Tensor,
    levels: list[float],
    name: str,
) -> LevelMetrics:
    """Per-level recall and occupancy from level *indices* (any shape, same
    shape both sides) — see `Head.level_index` for how a value becomes one.
    `levels` only labels the rows, in physical units."""
    if pred_idx.shape != true_idx.shape:
        raise ValueError(
            f"pred_idx shape {tuple(pred_idx.shape)} != true_idx shape "
            f"{tuple(true_idx.shape)}"
        )
    pred = pred_idx.detach().cpu().reshape(-1)
    true = true_idx.detach().cpu().reshape(-1)
    n = pred.numel()
    true_share, pred_share, recall = [], [], []
    for k in range(len(levels)):
        at_k = true == k
        n_true = int(at_k.sum())
        true_share.append(n_true / n)
        pred_share.append(int((pred == k).sum()) / n)
        recall.append(
            int((pred[at_k] == k).sum()) / n_true if n_true else float("nan")
        )
    return LevelMetrics(
        name=name, levels=list(levels), true_share=true_share,
        pred_share=pred_share, recall=recall, n_samples=n,
    )


@beartype
def rollout_stability(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    channel_names: list[str],
) -> list[RolloutStability]:
    """Per-channel rollout shape diagnostics. Same tensor contract as
    `per_channel_metrics`: [batch, time, ny] in physical units.

    Statistics are computed per window and then averaged across windows, so a
    single diverging window cannot be hidden by the others' step counts.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred shape {tuple(y_pred.shape)} != y_true shape "
            f"{tuple(y_true.shape)}"
        )
    if y_pred.ndim != 3:
        raise ValueError(
            f"Expected 3D [batch, time, ny] tensors; got {y_pred.ndim}D"
        )
    if len(channel_names) != y_pred.shape[-1]:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries but tensors have "
            f"ny={y_pred.shape[-1]}"
        )
    if y_pred.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 timesteps to measure rollout shape; got "
            f"{y_pred.shape[1]}"
        )

    pred = y_pred.detach().cpu()
    true = y_true.detach().cpu()

    def rise(x: torch.Tensor) -> float:
        """Mean over windows of the fraction of increasing steps."""
        return float((x.diff(dim=1) > 0).float().mean(dim=1).mean())

    def drift(x: torch.Tensor) -> float:
        """Mean over windows of |last - first|, in physical units."""
        return float((x[:, -1] - x[:, 0]).abs().mean())

    return [
        RolloutStability(
            name=name,
            pred_rise_fraction=rise(pred[:, :, i]),
            true_rise_fraction=rise(true[:, :, i]),
            pred_drift=drift(pred[:, :, i]),
            true_drift=drift(true[:, :, i]),
        )
        for i, name in enumerate(channel_names)
    ]


@beartype
def stability_to_markdown(
    stability: list[RolloutStability], rise_excess_warn: float = 0.25
) -> str:
    """Render rollout stability as a markdown table, flagging likely ramps."""
    header = (
        "| Channel | Rise frac (pred / true) | Excess | "
        "Drift (pred / true) | Ratio | |"
    )
    sep = "|---|---:|---:|---:|---:|---|"
    rows = [header, sep]
    for s in stability:
        flag = "⚠️ ramp?" if s.looks_like_ramp(rise_excess_warn) else ""
        rows.append(
            f"| {s.name} | {s.pred_rise_fraction:.3f} / "
            f"{s.true_rise_fraction:.3f} | {s.rise_excess:.3f} | "
            f"{s.pred_drift:.4g} / {s.true_drift:.4g} | "
            f"{s.drift_ratio:.2f} | {flag} |"
        )
    return "\n".join(rows)


@beartype
def stability_to_dict(stability: list[RolloutStability]) -> dict:
    """JSON-serializable rollout stability (for MLflow / disk)."""
    return {
        s.name: {
            "pred_rise_fraction": s.pred_rise_fraction,
            "true_rise_fraction": s.true_rise_fraction,
            "rise_excess": s.rise_excess,
            "pred_drift": s.pred_drift,
            "true_drift": s.true_drift,
            "drift_ratio": s.drift_ratio,
            "looks_like_ramp": s.looks_like_ramp(),
        }
        for s in stability
    }


@beartype
def metrics_to_dict(metrics: list[ChannelMetrics]) -> dict:
    """Convert metrics list to a JSON-serializable dict (for MLflow / disk)."""
    return {
        m.name: {
            "rmse": m.rmse,
            "mae": m.mae,
            "r2": m.r2,
            "nrmse": m.nrmse,
            "n_samples": m.n_samples,
        }
        for m in metrics
    }


@beartype
def level_metrics_to_dict(metrics: list[LevelMetrics]) -> dict:
    """JSON-serializable level metrics (for MLflow / disk)."""
    return {
        m.name: {
            "levels": m.levels,
            "true_share": m.true_share,
            "pred_share": m.pred_share,
            "recall": m.recall,
            "balanced_accuracy": m.balanced_accuracy,
            "n_samples": m.n_samples,
        }
        for m in metrics
    }


def _fmt_recall(r: float) -> str:
    return "n/a" if math.isnan(r) else f"{r:.3f}"


@beartype
def level_metrics_to_markdown(
    metrics: list[LevelMetrics],
    reference: Optional[list[LevelMetrics]] = None,
) -> str:
    """Two tables: balanced accuracy per channel (beside `reference`'s, when
    given — persistence, on the reportcard), then one row per level."""
    ref_by_name = {m.name: m for m in (reference or [])}
    summary = ["| Channel | Balanced acc" + (" | Persistence |" if reference else " |"),
               "|---|---:" + ("|---:|" if reference else "|")]
    for m in metrics:
        row = f"| {m.name} | {_fmt_recall(m.balanced_accuracy)} |"
        if reference:
            ref = ref_by_name.get(m.name)
            row += f" {_fmt_recall(ref.balanced_accuracy) if ref else 'n/a'} |"
        summary.append(row)

    detail = ["| Channel | Level | True share | Pred share | Recall |",
              "|---|---:|---:|---:|---:|"]
    for m in metrics:
        for level, t, p, r in zip(m.levels, m.true_share, m.pred_share, m.recall):
            detail.append(
                f"| {m.name} | {level:g} | {t * 100:.1f}% | {p * 100:.1f}% | "
                f"{_fmt_recall(r)} |"
            )
    return "\n".join(summary) + "\n\n" + "\n".join(detail)


@beartype
def metrics_to_markdown(metrics: list[ChannelMetrics]) -> str:
    """Render metrics as a markdown table for the reportcard."""
    header = "| Channel | RMSE | MAE | R² | NRMSE |"
    sep = "|---|---:|---:|---:|---:|"
    rows = [header, sep]
    for m in metrics:
        nrmse_str = "n/a" if math.isnan(m.nrmse) else f"{m.nrmse:.4f}"
        rows.append(
            f"| {m.name} | {m.rmse:.4f} | {m.mae:.4f} | "
            f"{m.r2:.4f} | {nrmse_str} |"
        )
    return "\n".join(rows)
