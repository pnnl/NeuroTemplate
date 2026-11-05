"""
Domain bridge for the DPC controller.

These are the user-owned hooks that connect the generic control machinery to
*this* building: how electrical power is computed, and what the plots show. The band
itself is data — reference columns the objective names (see control_fixtures).

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
