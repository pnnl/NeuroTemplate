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

from dataclasses import dataclass

import torch

from neurotemplate._typing import beartype


@dataclass
class ChannelMetrics:
    """Metrics for one Y channel, all in the channel's physical units."""
    name: str
    rmse: float
    mae: float
    r2: float
    nrmse: float  # NaN if channel is constant
    n_samples: int


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
