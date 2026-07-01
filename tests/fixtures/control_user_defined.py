"""
Domain bridge for the DPC controller (mirrors src/user_defined.py).

These are the user-owned hooks that connect the generic control machinery to
*this* building: which Y channels are comfort-controlled, what the comfort
band is, what the actuator limits are, and how electrical power is computed.

Everything returned in "normalized space" uses the same per-column
shift/stretch as the loaded sys-id run (norm_stats), so it is directly
comparable to the surrogate's normalized state/inputs.

Target dataset: single-zone Modelica RTU strip-mall (summer month).
  Y = [Zone Air Temperature, RTU Supply Air Temperature, RTU Supply Air Flow,
       Cooling Stage Signal, RTU Fan Power, RTU DX Power]
  U = [Zone Cooling Setpoint]
  D = [Outdoor Dry-Bulb, Global Horizontal Irradiance, Occupancy]
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from neurotemplate.control.scenarios import BandSpec

if os.environ.get("RUNTIME_TYPING", 0):
    from beartype import beartype
else:
    def beartype(fn): return fn


# ---------------------------------------------------------------------
# Required: which Y channels are comfort-controlled
# ---------------------------------------------------------------------
@beartype
def controlled_channels(
    nm_mapping: dict[str, list[str]],
) -> tuple[list[str], list[int]]:
    """
    Zone air-temperature column(s) (the signals the comfort band applies to)
    and their integer indices into nm_mapping["Y"].

    For the single-zone RTU dataset this is "Zone Air Temperature [°C]".
    """
    y_cols = nm_mapping["Y"]
    controlled = [
        c for c in y_cols
        if ("zone" in c.lower()) and ("air temperature" in c.lower())
    ]
    if not controlled:
        raise ValueError(
            "No comfort-controlled channels found in Y. Expected zone air "
            f"temperature columns; Y = {y_cols}"
        )
    idx = [y_cols.index(c) for c in controlled]
    return controlled, idx


# ---------------------------------------------------------------------
# Required: comfort band as a sampled time-varying signal
# ---------------------------------------------------------------------
@beartype
def comfort_band_signal(
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    n_samples: int,
    nsteps: int,
    band: BandSpec,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (ymin, ymax) comfort-band signals of shape
    [n_samples, nsteps, n_ctrl] in NORMALIZED space.

    `band.sub_samples` splits the window into that many equal-ish contiguous
    segments. Each segment draws its own (width, lower-bound) the same way
    a single-segment window does, so the result has `sub_samples - 1` step
    jumps in both `ymin` and `ymax`. `sub_samples=1` reproduces the legacy
    single-piece band.
    """
    k = band.sub_samples
    if k < 1:
        raise ValueError(f"band.sub_samples must be >= 1, got {k}")
    if k > nsteps:
        raise ValueError(
            f"band.sub_samples ({k}) cannot exceed nsteps ({nsteps})"
        )

    controlled, _ = controlled_channels(nm_mapping)
    n_ctrl = len(controlled)

    # One (width, low) per (sample, segment); channels share the band.
    widths = band.width_min + (band.width_max - band.width_min) * torch.rand(
        n_samples, k, 1, generator=generator,
    )
    lows = band.lo + ((band.hi - widths) - band.lo) * torch.rand(
        n_samples, k, 1, generator=generator,
    )

    # Spread the k segments across nsteps. First (nsteps % k) segments are
    # one step longer so the counts sum to nsteps.
    base, rem = divmod(nsteps, k)
    counts = torch.full((k,), base, dtype=torch.long)
    counts[:rem] += 1
    seg_widths = torch.repeat_interleave(widths, counts, dim=1)
    seg_lows = torch.repeat_interleave(lows, counts, dim=1)

    ymin_phys = seg_lows.expand(n_samples, nsteps, n_ctrl)
    ymax_phys = (seg_lows + seg_widths).expand(n_samples, nsteps, n_ctrl)

    shift = torch.tensor([norm_stats[c]["shift"] for c in controlled])
    stretch = torch.tensor([norm_stats[c]["stretch"] for c in controlled])

    ymin = (ymin_phys - shift) / stretch
    ymax = (ymax_phys - shift) / stretch
    return ymin, ymax


# ---------------------------------------------------------------------
# Required: actuator limits for the policy output (normalized U space)
# ---------------------------------------------------------------------
@beartype
def actuation_bounds(
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-U-channel (umin, umax) in NORMALIZED space for blocks.MLP_bounds.

    Uses each control column's observed physical min/max from the training
    data (norm_stats), mapped through that column's shift/stretch.
    """
    u_cols = nm_mapping["U"]
    if not u_cols:
        raise ValueError("No control (U) columns in nm_mapping.")

    umin = torch.tensor([
        (norm_stats[c]["min"] - norm_stats[c]["shift"]) / norm_stats[c]["stretch"]
        for c in u_cols
    ])
    umax = torch.tensor([
        (norm_stats[c]["max"] - norm_stats[c]["shift"]) / norm_stats[c]["stretch"]
        for c in u_cols
    ])
    return umin, umax


# ---------------------------------------------------------------------
# REQUIRED user hook: cooling power for the effort/energy objective
# ---------------------------------------------------------------------
# This dataset metres HVAC power directly: RTU Fan Power [W] and RTU DX Power
# [W] are Y channels produced by the surrogate. Total electrical power is just
# their sum:
#
#     P = RTU Fan Power + RTU DX Power           [W]
#
# Minimising this minimises HVAC electrical energy use.
POWER_METRIC_IMPLEMENTED = True


def _power_channel_indices(
    nm_mapping: dict[str, list[str]],
) -> tuple[int, int]:
    """Indices of (fan_power, dx_power) within nm_mapping['Y']."""
    y = nm_mapping["Y"]
    fan_idx = dx_idx = None
    for i, c in enumerate(y):
        cl = c.lower()
        if "fan power" in cl:
            fan_idx = i
        elif "dx power" in cl:
            dx_idx = i
    if fan_idx is None or dx_idx is None:
        raise ValueError(
            "Cannot compute HVAC power: need 'RTU Fan Power' and 'RTU DX "
            f"Power' channels in Y. Got Y = {y}"
        )
    return fan_idx, dx_idx


def has_power_channels(nm_mapping: dict[str, list[str]]) -> bool:
    """Whether the surrogate's Y exposes both power channels we need."""
    try:
        _power_channel_indices(nm_mapping)
        return True
    except ValueError:
        return False


def _unnorm(col_idx: int, col_name: str, yn: torch.Tensor,
            norm_stats: dict) -> torch.Tensor:
    s = norm_stats[col_name]
    return yn[:, :, col_idx] * s["stretch"] + s["shift"]


@beartype
def power_metric(
    yn: torch.Tensor,
    U: torch.Tensor,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> torch.Tensor:
    """
    Total HVAC electrical power per timestep.

    Args:
        yn: predicted state trajectory [batch, time, ny] (normalized)
        U:  control trajectory         [batch, time, nu] (unused here;
            the policy influences power through the dynamics)
    Returns:
        P [batch, time] in Watts (>= 0). The effort objective drives the
        closed-loop mean of this toward zero.
    """
    y = nm_mapping["Y"]
    fan_idx, dx_idx = _power_channel_indices(nm_mapping)
    fan = _unnorm(fan_idx, y[fan_idx], yn, norm_stats)
    dx = _unnorm(dx_idx, y[dx_idx], yn, norm_stats)

    return (fan + dx).clamp_min(0.0)
