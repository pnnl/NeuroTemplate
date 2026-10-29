"""
Domain bridge for the DPC controller (mirrors src/user_defined.py).

These are the user-owned hooks that connect the generic control machinery to
*this* building: what the actuator limits are, how electrical power is
computed, and the rule a residual policy is anchored to. The band itself
is data — reference columns the objective names (see control_fixtures).

Every hook works in physical units.

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
# Required: actuator limits for the policy output
# ---------------------------------------------------------------------
@beartype
def actuation_bounds(
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-U-channel (umin, umax): each control column's observed min/max in
    the training data.
    """
    u_cols = nm_mapping["U"]
    if not u_cols:
        raise ValueError("No control (U) columns in nm_mapping.")
    umin = torch.tensor([norm_stats[c]["min"] for c in u_cols])
    umax = torch.tensor([norm_stats[c]["max"] for c in u_cols])
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
# Optional: the reference rule — the residual's anchor
# ---------------------------------------------------------------------
@beartype
def reference_control(
    y_now: torch.Tensor,
    r_now: dict[str, torch.Tensor],
    d_now: torch.Tensor,
    nm_mapping: dict[str, list[str]],
) -> dict[str, torch.Tensor]:
    """Park the setpoint at the middle of the comfort band.

    The simplest rule that is actually trying: it ignores the disturbances and
    the zone state entirely and just aims at the centre of whatever band it is
    handed. Against a surrogate whose zone tracks the setpoint this is the
    exactly-optimal comfort controller, which makes it a sharp check on
    whether a learned policy is earning its complexity.

    Memoryless, current step only, one entry per anchored U column.
    """
    (setpoint,) = nm_mapping["U"]
    return {setpoint: 0.5 * (r_now[LOW] + r_now[HIGH])}     # [b], °C


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


@beartype
def power_metric(
    yn: torch.Tensor,
    U: torch.Tensor,
    D: torch.Tensor,
    nm_mapping: dict[str, list[str]],
) -> torch.Tensor:
    """
    Total HVAC electrical power per timestep.

    Args:
        yn: predicted state trajectory [batch, time, ny]
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

    total = sum(yn[:, :, y.index(name)] for name in metered)
    return total.clamp_min(0.0)
