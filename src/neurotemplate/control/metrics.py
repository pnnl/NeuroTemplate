"""
Closed-loop control metrics in physical units.

Two groups, kept separate because they answer different questions and have
different units:

  - ComfortMetrics — per controlled channel: how often, how badly, and in
    which direction the comfort band was violated.
  - EnergyMetrics  — how much power the closed loop drew, and what the peaks
    looked like.

Everything here is a pure function over already-denormalized tensors. The
pipeline does the denormalization and the hook calls; this module only does
arithmetic, so the numbers are cheap to test against hand-computed cases.

Energy carries no unit conversion. `control_user_defined.power_metric`
returns whatever the use case meters, so power is reported in those native
units and energy in `power·h`. An app metering Watts converts to kWh itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from neurotemplate._typing import beartype


@dataclass
class ComfortMetrics:
    """Comfort-band performance for one controlled channel, in physical units."""
    name: str
    violation_rate: float   # fraction of (window, step) pairs outside the band
    frac_below: float       # too cold and too hot are different failures, so
    frac_above: float       # they are reported separately, not just summed
    mean_violation: float   # mean excursion over violating steps only (0 if none)
    max_violation: float    # worst single excursion
    degree_hours: float     # mean over windows of ∫|excursion| dt, in unit·h


@dataclass
class EnergyMetrics:
    """Closed-loop power/energy, in the native units of `power_metric`."""
    mean_power: float        # mean instantaneous power
    total_energy: float      # mean over windows of ∫power dt, in power·h
    peak_power: float        # worst instantaneous draw anywhere
    mean_window_peak: float  # mean of per-window maxima — demand-charge proxy


@beartype
def hours_per_step(t: torch.Tensor) -> float:
    """Timestep length in hours, from the `t` channel (POSIX seconds).

    `t` is [batch, time, 1]; the grid is uniform, so one difference is enough.
    """
    if t.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 timesteps to infer dt; got {t.shape[1]}"
        )
    dt_s = float(t[0, 1, 0] - t[0, 0, 0])
    if dt_s <= 0:
        raise ValueError(f"Non-increasing time channel: dt = {dt_s} s")
    return dt_s / 3600.0


@beartype
def comfort_metrics(
    y: torch.Tensor,
    ymin: torch.Tensor,
    ymax: torch.Tensor,
    channel_names: list[str],
    step_hours: float,
) -> list[ComfortMetrics]:
    """
    Args:
        y:     [batch, time, n_ctrl] controlled channels in physical units.
        ymin:  [batch, time, n_ctrl] comfort lower band, same units.
        ymax:  [batch, time, n_ctrl] comfort upper band, same units.
        channel_names: names for the n_ctrl axis.
        step_hours: rollout timestep in hours (see `hours_per_step`).

    Returns:
        One ComfortMetrics per channel, in the order of channel_names.
    """
    if not (y.shape == ymin.shape == ymax.shape):
        raise ValueError(
            f"Shape mismatch: y {tuple(y.shape)}, ymin {tuple(ymin.shape)}, "
            f"ymax {tuple(ymax.shape)}"
        )
    if y.ndim != 3:
        raise ValueError(f"Expected 3D [batch, time, n_ctrl]; got {y.ndim}D")
    if len(channel_names) != y.shape[-1]:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries but tensors have "
            f"n_ctrl={y.shape[-1]}"
        )

    y = y.detach().cpu()
    below = (ymin.detach().cpu() - y).clamp_min(0.0)   # >0 when too cold
    above = (y - ymax.detach().cpu()).clamp_min(0.0)   # >0 when too hot

    results = []
    for i, name in enumerate(channel_names):
        lo, hi = below[:, :, i], above[:, :, i]
        excursion = lo + hi           # only one side can be non-zero at a time
        violating = excursion > 0

        n_violating = int(violating.sum())
        mean_violation = (
            float(excursion[violating].mean()) if n_violating else 0.0
        )

        results.append(ComfortMetrics(
            name=name,
            violation_rate=float(violating.float().mean()),
            frac_below=float((lo > 0).float().mean()),
            frac_above=float((hi > 0).float().mean()),
            mean_violation=mean_violation,
            max_violation=float(excursion.max()),
            # Sum over time for each window, then average across windows: the
            # per-window integral is what a building operator recognizes.
            degree_hours=float(excursion.sum(dim=1).mean()) * step_hours,
        ))

    return results


@beartype
def energy_metrics(power: torch.Tensor, step_hours: float) -> EnergyMetrics:
    """
    Args:
        power: [batch, time] instantaneous power per rollout window.
        step_hours: rollout timestep in hours.

    Peak conventions, both reported because they answer different questions:
    `peak_power` is the worst instant anywhere in the evaluation, while
    `mean_window_peak` is the typical per-window peak — closer to what a
    demand charge bills against. Neither applies rolling-average smoothing;
    that is a tariff detail and belongs in the use case.
    """
    if power.ndim != 2:
        raise ValueError(f"Expected 2D [batch, time] power; got {power.ndim}D")

    power = power.detach().cpu()
    window_peaks = power.max(dim=1).values

    return EnergyMetrics(
        mean_power=float(power.mean()),
        total_energy=float(power.sum(dim=1).mean()) * step_hours,
        peak_power=float(power.max()),
        mean_window_peak=float(window_peaks.mean()),
    )


@beartype
def comfort_to_markdown(metrics: list[ComfortMetrics]) -> str:
    """Render comfort metrics as a markdown table for the reportcard."""
    header = (
        "| Channel | Violation rate | Below | Above | "
        "Mean excursion | Max excursion | Degree-hours |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for m in metrics:
        rows.append(
            f"| {m.name} | {m.violation_rate * 100:.2f}% | "
            f"{m.frac_below * 100:.2f}% | {m.frac_above * 100:.2f}% | "
            f"{m.mean_violation:.4f} | {m.max_violation:.4f} | "
            f"{m.degree_hours:.4f} |"
        )
    return "\n".join(rows)


@beartype
def comfort_to_dict(metrics: list[ComfortMetrics]) -> dict:
    """JSON-serializable comfort metrics (for MLflow / disk)."""
    return {m.name: {k: v for k, v in asdict(m).items() if k != "name"}
            for m in metrics}
