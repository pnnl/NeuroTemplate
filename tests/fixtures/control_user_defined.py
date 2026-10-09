"""
Domain bridge for the DPC controller (mirrors src/user_defined.py).

These are the user-owned hooks that connect the generic control machinery to
*this* building: what the actuator limits are, how electrical power is
computed, and the rule a learned policy is measured against. The band itself
is data — reference columns the objective names (see control_fixtures).

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

import torch

from neurotemplate.common.data_utils import NormStats

if os.environ.get("RUNTIME_TYPING", 0):
    from beartype import beartype
else:
    def beartype(fn): return fn


# The comfort band's reference columns, as control_fixtures writes them.
LOW = "Comfort Low [°C]"
HIGH = "Comfort High [°C]"


# ---------------------------------------------------------------------
# Required: actuator limits for the policy output (normalized U space)
# ---------------------------------------------------------------------
@beartype
def actuation_bounds(
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
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
# REQUIRED user hook: cooling power for the energy/cost/peak objectives
# ---------------------------------------------------------------------
# This dataset metres HVAC power directly: RTU Fan Power [W] and RTU DX Power
# [W] are Y channels produced by the surrogate. Total electrical power is just
# their sum:
#
#     P = RTU Fan Power + RTU DX Power           [W]
#
# Minimising this minimises HVAC electrical energy use.
POWER_METRIC_IMPLEMENTED = True


# ---------------------------------------------------------------------
# Optional: the reference rule — a baseline, and the residual's anchor
# ---------------------------------------------------------------------
@beartype
def reference_control(
    y_now: torch.Tensor,
    r_now: dict[str, torch.Tensor],
    d_now: torch.Tensor,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> torch.Tensor:
    """Park the setpoint at the middle of the comfort band.

    The simplest rule that is actually trying: it ignores the disturbances and
    the zone state entirely and just aims at the centre of whatever band it is
    handed. Against a surrogate whose zone tracks the setpoint this is the
    exactly-optimal comfort controller, which makes it a sharp check on
    whether a learned policy is earning its complexity.

    Memoryless, current step only, normalized in and normalized U out — the
    same space the learned policy emits.
    """
    def physical(name):
        return r_now[name] * norm_stats[name]["stretch"] + norm_stats[name]["shift"]

    mid = 0.5 * (physical(LOW) + physical(HIGH))            # [b], °C
    u_cols = nm_mapping["U"]
    shift = torch.tensor([norm_stats[c]["shift"] for c in u_cols], device=mid.device)
    stretch = torch.tensor([norm_stats[c]["stretch"] for c in u_cols], device=mid.device)
    umin, umax = actuation_bounds(nm_mapping, norm_stats)
    u = (mid[:, None] - shift) / stretch                    # [b, nu]
    return torch.clamp(u, umin.to(u.device), umax.to(u.device))


@beartype
def power_channels(nm_mapping: dict[str, list[str]]) -> list[str]:
    """Y columns that meter electrical power, in plot order.

    For this dataset: RTU Fan Power and RTU DX Power. Returns [] when the
    surrogate's Y does not expose them, which turns off the power row on
    closed-loop plots.
    """
    y = nm_mapping["Y"]
    return [c for c in y if "fan power" in c.lower() or "dx power" in c.lower()]


@beartype
def plot_context_channels(nm_mapping: dict[str, list[str]]) -> list[str]:
    """D columns to overlay on the temperature axis of closed-loop plots.

    Outdoor dry-bulb is the driver that makes a zone-temperature trace
    legible. Returns [] to plot zone temperatures with no context trace.
    """
    return [
        c for c in nm_mapping["D"]
        if "outdoor" in c.lower() and "dry-bulb" in c.lower()
    ]


def _unnorm(col_idx: int, col_name: str, yn: torch.Tensor,
            norm_stats: dict) -> torch.Tensor:
    s = norm_stats[col_name]
    return yn[:, :, col_idx] * s["stretch"] + s["shift"]


@beartype
def power_metric(
    yn: torch.Tensor,
    U: torch.Tensor,
    D: torch.Tensor,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> torch.Tensor:
    """
    Total HVAC electrical power per timestep.

    Args:
        yn: predicted state trajectory [batch, time, ny] (normalized)
        U:  control trajectory         [batch, time, nu] (unused here;
            the policy influences power through the dynamics)
        D:  disturbance trajectory     [batch, time, nd] (unused here;
            both power channels are metered directly in Y)
    Returns:
        P [batch, time] in Watts (>= 0). The energy objective drives the
        closed-loop mean of this toward zero.
    """
    y = nm_mapping["Y"]
    metered = power_channels(nm_mapping)
    if not metered:
        raise ValueError(
            "Cannot compute HVAC power: need 'RTU Fan Power' and 'RTU DX "
            f"Power' channels in Y. Got Y = {y}"
        )

    total = sum(
        _unnorm(y.index(name), name, yn, norm_stats) for name in metered
    )
    return total.clamp_min(0.0)
