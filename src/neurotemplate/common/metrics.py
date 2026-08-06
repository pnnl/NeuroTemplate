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
"""

import re
from dataclasses import dataclass

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
    """
    name: str
    pred_rise_fraction: float
    true_rise_fraction: float
    pred_drift: float
    true_drift: float

    @property
    def rise_excess(self) -> float:
        """|pred - true| rise fraction. 0 is perfect; >0.25 means a ramp."""
        return abs(self.pred_rise_fraction - self.true_rise_fraction)

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
        flag = "⚠️ ramp?" if s.rise_excess > rise_excess_warn else ""
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
def metrics_to_markdown(metrics: list[ChannelMetrics]) -> str:
    """Render metrics as a markdown table for the reportcard."""
    header = "| Channel | RMSE | MAE | R² | NRMSE |"
    sep = "|---|---:|---:|---:|---:|"
    rows = [header, sep]
    for m in metrics:
        nrmse_str = f"{m.nrmse:.4f}" if m.nrmse == m.nrmse else "n/a"  # NaN check
        rows.append(
            f"| {m.name} | {m.rmse:.4f} | {m.mae:.4f} | "
            f"{m.r2:.4f} | {nrmse_str} |"
        )
    return "\n".join(rows)
