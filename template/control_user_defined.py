"""
Control hooks — the glue between the DPC machinery and *your* plant.

These answer: what the actuator limits are, and optionally what power costs
and what rule a residual policy is anchored to. The objective — which
outputs are held inside which band, what power is priced at — is not a hook:
it is the `losses` table in control.yaml, over columns of the surrogate and
of your reference files (see `make_references.py`). The Y/U/D roles come
from the frozen sys-id run.

Every hook works in physical units, the data's own.

The file must be named `control_user_defined.py` and live in the directory
you run `nt` from, or be pointed at by `user_hooks_path` in your YAML. Fill
the hooks in here, in place: each docstring is that hook's specification
for the library version you have installed. A control run keeps a copy of
this file, and reloading the run uses the copy.

Every required hook runs as written. The examples in comments are syntax:
the column names, and any number in them, belong to some other plant.
"""

from __future__ import annotations

import torch

from neurotemplate._typing import beartype
from neurotemplate.common.data_utils import NormStats


# ---------------------------------------------------------------------
# Required: actuator limits for the policy output
# ---------------------------------------------------------------------
@beartype
def actuation_bounds(
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-U-channel (umin, umax) in physical units; the policy output is
    squashed into this box.

    The default below is each U column's observed range in the training
    data — the region the surrogate was identified on. So the box is only as
    wide as the data's excitation: if the data never drove an actuator where
    a target needs it, no policy can command it there, and no residual
    either. Nothing on the reportcard says so; the reportcard's time pinned
    at a bound looks the same whether the bound is clipping or nothing was
    asked. Before training a controller, check this box against the band or
    targets it will be asked to hold. If it does not reach, the fix is data
    that goes there, not a wider box: widening it past the data asks the
    surrogate about inputs it has never seen.

    Narrowing it is often right: to keep one actuator out of another's way
    (two actuators that oppose each other can be boxed so they can never
    cross), or to park one that has no job in this season.
    """
    u_cols = nm_mapping["U"]
    if not u_cols:
        raise ValueError("No control (U) columns in nm_mapping.")
    umin = torch.tensor([norm_stats[c]["min"] for c in u_cols])
    umax = torch.tensor([norm_stats[c]["max"] for c in u_cols])
    return umin, umax


# ---------------------------------------------------------------------
# Required: which Y channels meter power
# ---------------------------------------------------------------------
@beartype
def power_channels(nm_mapping: dict[str, list[str]]) -> list[str]:
    """
    Y columns that meter power, in the order you want them plotted.

    Return [] when nothing is metered — that turns off the power row on the
    closed-loop plots. List them by name rather than by pattern: a name
    filter such as `"power" in c.lower()` also sweeps in channels nobody is
    billed for (a second fuel under an electricity tariff), and those then
    look like a cost to the objective.

        return ["Compressor Power [W]", "Fan Power [W]"]
    """
    return []


# ---------------------------------------------------------------------
# Required: context traces for the closed-loop plots
# ---------------------------------------------------------------------
@beartype
def plot_context_channels(nm_mapping: dict[str, list[str]]) -> list[str]:
    """
    D columns to overlay on the controlled-variable axis of the closed-loop
    plots — the driver that makes the trace legible (an ambient temperature,
    a load, irradiance). Return [] for none.
    """
    return []


# ---------------------------------------------------------------------
# The signal every loss term naming `power` reduces
# ---------------------------------------------------------------------
# By default the sum of `power_channels`: the meters the model predicts.
POWER_METRIC_IMPLEMENTED = True


@beartype
def power_metric(
    yn: torch.Tensor,
    U: torch.Tensor,
    D: torch.Tensor,
    nm_mapping: dict[str, list[str]],
) -> torch.Tensor:
    """
    Instantaneous power per timestep, shape [batch, time]: what a loss term
    naming `power` reduces (a mean for energy, a tariff-weighted mean for
    cost, a power mean for a peak), and what the reportcard integrates.

    Inputs are the trajectories in physical units: yn [batch, time, ny],
    U [batch, time, nu], D [batch, time, nd]. At each step, U is the command
    that drove the plant into that step's yn. The library converts no units:
    return Watts and the integral is in Wh.

    The default sums the meters `power_channels` names, as the model predicts
    them. What the policy learns from is this signal's gradient, so the
    meters must respond to the commands: model them in a block that sees the
    actuation (a declared `demand` the meter block can see; `nt guide
    thermal-zone`, step 4). Whether they do is the surrogate's question,
    answered on its own reportcard.

    Replace the body when nothing meters the power: a proxy from the physics
    of when the equipment runs, which does not decrease as the demand on it
    grows and is zero when it would not be called. A `relu` there stops the
    gradient on the inactive side; `soft_clamp_min`
    (`from neurotemplate.modeling.prestack import soft_clamp_min`) keeps the
    zero forward and a gradient back toward the active side.
    """
    channels = power_channels(nm_mapping)
    if not channels:
        raise NotImplementedError(
            "power_metric sums the meters power_channels names, and it names none. "
            "List the metered Y columns there, or write a proxy in power_metric.")
    return sum(yn[:, :, nm_mapping["Y"].index(c)] for c in channels)


# ---------------------------------------------------------------------
# Optional: the reference rule — the residual policy's anchor
# ---------------------------------------------------------------------
# With `residual_delta` set in the control YAML, each U column it names
# commands `reference_control(...)[column] + δ`, δ inside that column's
# [below, above] box: the rule does the routine work and the policy only
# adjusts it. The box is a trust region on surrogate error — it caps what a
# surrogate that is wrong somewhere can cost you. A U column it does not
# name is a free policy output over its actuation bounds. Return exactly
# the columns `residual_delta` names (the library refuses anything else).
#
# Current step only, memoryless, physical units in and out. `r_now` maps each
# reference column the objective names to its current value, [b]. Return
# continuous values: nothing is rounded to an actuator's resolution.
#
# What makes a good anchor:
#   - It is the floor the learned part is measured from, so make it the rule
#     you would otherwise run, and no worse. An anchor worse than the rule
#     you compare against makes every timid residual look like a loss.
#   - Keep it price-blind. r_now carries every reference column the
#     objective names, a tariff included, but an anchor that reads the price
#     leaves nothing to measure the learned price response against.
#   - Two anchored actuators that must not cross (heating above cooling)
#     should be unable to for any δ the boxes allow. Build that in — via
#     the anchor and the boxes, or `actuation_bounds` — rather than hoping
#     the objective forbids it.
#
# Sizing the boxes decides what is possible. Too narrow and there is
# nothing to learn. Room that takes an output past its band edge is room
# the optimizer will spend wherever the surrogate underprices being out of
# band. An actuator with no job in this data (heating in a cooling season)
# is best locked: box it [0, 0], or park it with `actuation_bounds`. A box
# edge the policy sits on all the time is a finding about the rule or the
# surrogate, not a tuning target.
#
# To score the rule itself, train with every box [0, 0]: the policy is then
# the rule, on your surrogate, on your test windows.
#
# Leave it commented out if you have no rule. The example brackets one band
# with a heating/cooling setpoint pair (`bracket_band` below).
#
# BAND_LOW = "Comfort Low [°C]"      # the band's columns in your reference
# BAND_HIGH = "Comfort High [°C]"    # files, as the losses table names them
# HEAT = "Heating Setpoint [°C]"     # the U columns the rule anchors, as
# COOL = "Cooling Setpoint [°C]"     # residual_delta names them
#
# @beartype
# def reference_control(
#     y_now: torch.Tensor,               # [b, ny] current state
#     r_now: dict[str, torch.Tensor],    # {reference column: [b]}
#     d_now: torch.Tensor,               # [b, nd] current disturbances
#     nm_mapping: dict[str, list[str]],
# ) -> dict[str, torch.Tensor]:          # {anchored U column: [b]}
#     """Heating at the band floor, cooling at its ceiling."""
#     cool, heat = bracket_band(r_now[BAND_LOW], r_now[BAND_HIGH])
#     return {HEAT: heat, COOL: cool}


@beartype
def bracket_band(
    lo: torch.Tensor,
    hi: torch.Tensor,
    cool_inset: float = 0.0,
    heat_inset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straddle a band with a (cooling, heating) setpoint pair, in physical units.

    Example code for `reference_control`, not library behaviour. Heating sits
    at the band floor and cooling at the ceiling, so the two bracket the band
    instead of both chasing one target.

    The insets pull each setpoint further inside its edge. A thermostat
    usually lets the zone float somewhat past its setpoint, and an inset
    corrects for that. It is a cost knob as much as a comfort one — a cooling
    setpoint pulled further in conditions more — so measure it on the plant
    rather than inheriting it. When the insets cross, both collapse to the
    band's midpoint, so heating never commands above cooling.

    Args:
        lo, hi: band edges, any broadcast-compatible shape.
        cool_inset: pull the cooling setpoint below the ceiling by this much.
        heat_inset: pull the heating setpoint above the floor by this much.

    Returns:
        (cooling, heating), each the shape of `lo`/`hi` broadcast together.
    """
    if cool_inset < 0 or heat_inset < 0:
        raise ValueError(
            f"insets must be >= 0, got cool={cool_inset}, heat={heat_inset}"
        )

    heat = lo + heat_inset
    cool = hi - cool_inset
    mid = 0.5 * (lo + hi)
    collapse = heat > cool
    return torch.where(collapse, mid, cool), torch.where(collapse, mid, heat)
