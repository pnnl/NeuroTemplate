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
# Optional: a rule-based controller to score the learned policy against
# ---------------------------------------------------------------------
class _BandMidpointPolicy(torch.nn.Module):
    """Park the setpoint at the middle of the comfort band.

    The simplest rule that is actually trying: it ignores the disturbances
    and the zone state entirely and just aims at the centre of whatever band
    it is handed. Against the `dummy` surrogate (next zone temp = setpoint)
    this is the exactly-optimal comfort controller, which makes it a sharp
    check on whether a learned policy is earning its complexity.

    Inputs are the *current* ymin/ymax (no preview window), both in
    normalized space, and the output is normalized U — the same space the
    learned policy emits.
    """

    def __init__(self, umin: torch.Tensor, umax: torch.Tensor, nu: int):
        super().__init__()
        self.nu = nu
        self.register_buffer("umin", umin)
        self.register_buffer("umax", umax)

    def forward(self, ymin: torch.Tensor, ymax: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (ymin + ymax)
        # One controlled channel per actuator for this dataset; broadcast the
        # single midpoint across U columns if a use case has more.
        u = midpoint.expand(*midpoint.shape[:-1], self.nu)
        return torch.clamp(u, self.umin, self.umax)


def build_baseline_policy(cfg, sysid_cfg, nm_mapping, norm_stats):
    """Rule-based controller the framework scores alongside the learned one.

    Returns a Neuromancer Node writing ["U"], or None to skip the
    comparison. Has no trainable parameters, so it is only ever evaluated.
    """
    from neuromancer.system import Node

    controlled, _ = controlled_channels(nm_mapping)
    nu = len(nm_mapping["U"])
    if len(controlled) != nu:
        # One setpoint per comfort-controlled zone is the assumption this
        # simple rule makes; say so rather than silently broadcasting wrong.
        raise ValueError(
            f"Band-midpoint baseline needs one U per controlled channel: "
            f"{len(controlled)} controlled channels vs {nu} U columns."
        )

    umin, umax = actuation_bounds(nm_mapping, norm_stats)
    device = sysid_cfg.device
    net = _BandMidpointPolicy(umin.to(device), umax.to(device), nu).to(device)
    return Node(net, ["ymin", "ymax"], ["U"], name="Band-midpoint baseline")


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
    norm_stats: dict[str, dict[str, float]],
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
