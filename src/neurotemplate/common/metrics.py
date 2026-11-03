"""
Per-channel evaluation metrics in physical units.

Computes RMSE per Y channel rather than a single aggregate, so each number
has units that make sense (averaging across channels with different units
is uninterpretable), and `skill` against persistence: 1 − RMSE / the RMSE
of holding the last seeded value. Skill is the unitless comparison; a
persistence reference makes a range normalisation unnecessary.

The single entry point is `per_channel_metrics(...)`, which takes
unnormalized [batch, time, ny] tensors plus the column-name list.

Neither can see a rare event. A staged channel that is off 97% of
the time is scored *better* by RMSE when the model never turns it on —
the best-NRMSE furnace surrogate in the head bake-off never fired once.
`level_metrics` is the classification view for such channels: per-level
recall and balanced accuracy (mean recall, so a constant predictor scores
1/K whatever the imbalance), plus the predicted-vs-true share of each
level, which is what distinguishes a working head from a constant.
"""

import math
import re
from dataclasses import dataclass

import torch

from neurotemplate._typing import beartype


_METRIC_KEY_SANITIZER = re.compile(r"[^A-Za-z0-9_.\-]+")


def metric_key(name: str) -> str:
    """Sanitize a column name into something MLflow accepts as a metric key."""
    return _METRIC_KEY_SANITIZER.sub("_", name).strip("_")


@dataclass
class ChannelMetrics:
    """RMSE for one Y channel, in the channel's physical units."""
    name: str
    rmse: float
    n_samples: int


def skill(rmse: float, persistence_rmse: float) -> float:
    """1 − RMSE / persistence RMSE: 0 is no better than holding the last seeded
    value, 1 is perfect, below 0 is worse than holding it. NaN when
    persistence is already exact (a channel that never moves)."""
    if persistence_rmse == 0:
        return float("nan")
    return 1.0 - rmse / persistence_rmse


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


def _check_rollout_shapes(y_pred: torch.Tensor, y_true: torch.Tensor,
                          channel_names: list[str]) -> None:
    """Both [batch, time, ny], matching, with one name per channel."""
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred shape {tuple(y_pred.shape)} != y_true shape {tuple(y_true.shape)}"
        )
    if y_pred.ndim != 3:
        raise ValueError(f"Expected 3D [batch, time, ny] tensors; got {y_pred.ndim}D")
    if len(channel_names) != y_pred.shape[-1]:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries but tensors have "
            f"ny={y_pred.shape[-1]}"
        )


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
    _check_rollout_shapes(y_pred, y_true, channel_names)

    # Detach + move to CPU once for stable scalar extraction.
    pred = y_pred.detach().cpu()
    true = y_true.detach().cpu()

    n_per_channel = pred.shape[0] * pred.shape[1]
    results: list[ChannelMetrics] = []

    for i, name in enumerate(channel_names):
        p = pred[:, :, i]
        t = true[:, :, i]
        err = p - t
        results.append(ChannelMetrics(
            name=name,
            rmse=err.pow(2).mean().sqrt().item(),
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
    _check_rollout_shapes(y_pred, y_true, channel_names)
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
        m.name: {"rmse": m.rmse, "n_samples": m.n_samples}
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


