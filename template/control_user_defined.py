"""
Control hooks — the glue between the DPC machinery and *your* plant.

These answer: what the actuator limits are, and optionally what power costs
and what rule a residual policy is anchored to. The objective —
which outputs are held inside which band, what power is priced at — is not a
hook: it is the `losses` table in control.yaml, over columns of the surrogate
and of your reference files (see `make_references.py`). The Y/U/D roles and
the normalization come from the frozen sys-id run.

Everything returned "in normalized space" uses the surrogate's own
per-column shift/stretch (norm_stats), which also carries an entry for every
reference column the objective names.

The file must be named `control_user_defined.py` and live in the directory
you run `nt` from, or be pointed at by `user_hooks_path` in your YAML.

Every required hook runs as written. Then consider `reference_control`: a
policy anchored to a rule you trust, inside a box you chose, caps what the
surrogate's error can cost you.
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
    Per-U-channel (umin, umax) in NORMALIZED space; the policy output is
    squashed into this box.

    The default below is each U column's observed range in the training
    data — the region the surrogate was identified on. That makes the box
    only as wide as the data's excitation: if the data never drove an
    actuator where you need the policy to go, the policy cannot go there
    (the reportcard's Actuation section shows time pinned at each bound).
    Hard-code physical
    limits here to narrow the box, e.g. to keep one actuator out of
    another's way; widening it past the data asks the surrogate about
    inputs it has never seen.
    """
    u_cols = nm_mapping["U"]
    if not u_cols:
        raise ValueError("No control (U) columns in nm_mapping.")

    def norm(col: str, key: str) -> float:
        s = norm_stats[col]
        return (s[key] - s["shift"]) / s["stretch"]

    umin = torch.tensor([norm(c, "min") for c in u_cols])
    umax = torch.tensor([norm(c, "max") for c in u_cols])
    return umin, umax


# ---------------------------------------------------------------------
# Required: which Y channels meter power
# ---------------------------------------------------------------------
@beartype
def power_channels(nm_mapping: dict[str, list[str]]) -> list[str]:
    """
    Y columns that meter power, in the order you want them plotted.

    Return [] when nothing is metered — that turns off the power row on the
    closed-loop plots and the energy section of the reportcard.

        return [c for c in nm_mapping["Y"] if "power" in c.lower()]
    """
    return []


# ---------------------------------------------------------------------
# Required: context traces for the closed-loop plots
# ---------------------------------------------------------------------
@beartype
def plot_context_channels(nm_mapping: dict[str, list[str]]) -> list[str]:
    """
    D columns to overlay on the controlled-variable axis of the closed-loop
    plots — the driver that makes the trace legible (ambient temperature,
    load, solar). Return [] for none.
    """
    return []


# ---------------------------------------------------------------------
# Required when a power term is weighted: the signal they all reduce
# ---------------------------------------------------------------------
# Set to True once `power_metric` below is implemented. Until then a loss
# term naming `power` fails at problem build, rather than silently optimizing
# something you did not write.
POWER_METRIC_IMPLEMENTED = False


@beartype
def power_metric(
    yn: torch.Tensor,
    U: torch.Tensor,
    D: torch.Tensor,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> torch.Tensor:
    """
    Instantaneous power per timestep, shape [batch, time]: what a loss term
    naming `power` reduces (a mean for energy, a tariff-weighted mean for
    cost, a power mean for a peak), and what the reportcard integrates.

    Inputs are the normalized trajectories: yn [batch, time, ny],
    U [batch, time, nu], D [batch, time, nd]. Denormalize what you use
    (`x * stretch + shift`, or `_denorm` below). The library converts no
    units: return Watts and the integral is in Wh. At each step, U is the
    command that drove the plant into that step's yn.

    Two shapes work. If the surrogate predicts metered power, sum it — real
    units, so the energy numbers mean something outside this run:

        total = sum(_denorm(yn, c, nm_mapping, norm_stats)
                    for c in power_channels(nm_mapping))
        return total.clamp_min(0.0)

    Otherwise write a proxy from the states and inputs, from the physics of
    when the equipment draws power (e.g. a compressor that runs only while
    the zone is above its cooling setpoint). A proxy is also worth
    considering when a predicted power channel is staged or on/off: a
    regression head averages it, and the policy then cannot tell *when*
    power is drawn. Whatever you return, the gradient is what the policy
    learns from — check its shape in the pre-training sweep across the
    actuation box, and its scale in the objective-share printout.
    """
    raise NotImplementedError(
        "power_metric is not implemented. Name no `power` in the control "
        "config's losses, or implement this and set "
        "POWER_METRIC_IMPLEMENTED = True."
    )


def _denorm(x: torch.Tensor, col: str, nm_mapping: dict[str, list[str]],
            norm_stats: NormStats, role: str = "Y"
            ) -> torch.Tensor:
    """Physical units for one channel of a [batch, time, n] trajectory."""
    s = norm_stats[col]
    return x[:, :, nm_mapping[role].index(col)] * s["stretch"] + s["shift"]


# ---------------------------------------------------------------------
# Optional: the reference rule — the residual policy's anchor
# ---------------------------------------------------------------------
# With `residual_delta` set in the control YAML, each U column it names
# commands `reference_control(...)[column] + δ`, δ inside that column's
# [below, above] box: the rule does the routine work and the policy only
# adjusts it, inside a bound you chose. A U column it does not name is a free
# policy output. Return exactly the columns `residual_delta` names (the
# library refuses anything else).
#
# Current step only, memoryless, normalized in and normalized out — the same
# space the learned policy emits. `r_now` maps each reference column the
# objective names to its current value, [b], normalized; `norm_stats` has an
# entry for each, so `x * stretch + shift` gives physical units. Return
# continuous values: nothing is rounded to an actuator's resolution, which is
# the building's job.
#
# To score the rule itself, train with every box [0, 0]: the policy is then
# the rule.
#
# Leave it commented out if you have no rule. The example brackets one band
# with a heating/cooling setpoint pair (`bracket_band` below), with boxes
# that hold each inward: [0, 0.5] on heating, [0.5, 0] on cooling.
#
# BAND_LOW = "Comfort Low [°C]"      # the band's columns in your reference
# BAND_HIGH = "Comfort High [°C]"    # files, as the losses table names them
# HEAT = "Heating Setpoint [°C]"     # the U columns the rule anchors, as
# COOL = "Cooling Setpoint [°C]"     # residual_delta names them
#
# @beartype
# def reference_control(
#     y_now: torch.Tensor,               # [b, ny] current state, normalized
#     r_now: dict[str, torch.Tensor],    # {reference column: [b]}, normalized
#     d_now: torch.Tensor,               # [b, nd] current disturbances, normalized
#     nm_mapping: dict[str, list[str]],
#     norm_stats: NormStats,
# ) -> dict[str, torch.Tensor]:          # {anchored U column: [b]}, normalized
#     """Heating at the band floor, cooling at its ceiling."""
#     def physical(col):
#         return r_now[col] * norm_stats[col]["stretch"] + norm_stats[col]["shift"]
#
#     def normalized(col, value):
#         return (value - norm_stats[col]["shift"]) / norm_stats[col]["stretch"]
#
#     cool, heat = bracket_band(physical(BAND_LOW), physical(BAND_HIGH),
#                               cool_inset=0.3)
#     return {HEAT: normalized(HEAT, heat), COOL: normalized(COOL, cool)}


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

    The insets pull each setpoint further inside its edge. A zone usually
    floats somewhat above its cooling setpoint, so `cool_inset` corrects for
    that — measure it on the plant rather than treating it as a comfort
    margin: too large and the two setpoints collide. When the insets cross,
    both collapse to the band's midpoint, so heating never commands above
    cooling.

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


# ---------------------------------------------------------------------
# Optional: per-column disturbance preview
# ---------------------------------------------------------------------
# Leave commented out unless you need it: if this function EXISTS it fully
# replaces `d_preview` from the YAML, so even an empty stub disables preview.
#
# Keys are D column names; values are step counts — how far ahead the policy
# sees each forecast. A property of the plant, shared by every run against
# it. A reference column's preview (the band ahead, the tariff ahead) is per
# run, and is `reference_preview` in the YAML.
#
# @beartype
# def preview_config(nm_mapping: dict[str, list[str]]) -> dict[str, int]:
#     return {"Outdoor Temperature [degC]": 12}
