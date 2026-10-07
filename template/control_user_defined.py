"""
Control hooks — the glue between the DPC machinery and *your* plant.

These answer: which states are held inside a reference band, what that band
looks like, what the actuator limits are, and optionally what power costs
and what rule a learned policy should be measured against. The Y/U/D roles
and the normalization come from the frozen sys-id run.

Everything returned "in normalized space" uses the surrogate's own
per-column shift/stretch (norm_stats).

The file must be named `control_user_defined.py` and live in the directory
you run `nt` from, or be pointed at by `user_hooks_path` in your YAML.

Fill in REGULATED_COLUMNS first; every other required hook runs as written.
Then consider `reference_control`: a learned policy is only worth shipping
if it beats a simple rule, and this is how the reportcard knows the rule.
"""

from __future__ import annotations

from typing import Optional

import torch

from neurotemplate._typing import beartype
from neurotemplate.control.scenarios import BandSpec


# ---------------------------------------------------------------------
# Required: which Y channels the comfort band applies to
# ---------------------------------------------------------------------
# The states the policy is judged on — the OCP *regulated output*. Everything
# else in Y is along for the ride. Names must match the surrogate's Y columns
# exactly.
REGULATED_COLUMNS: list[str] = [
    # "Tank Temperature [degC]",
]


@beartype
def regulated(
    nm_mapping: dict[str, list[str]],
) -> tuple[list[str], list[int]]:
    """Return (column names, their integer indices into nm_mapping["Y"])."""
    y_cols = nm_mapping["Y"]
    missing = [c for c in REGULATED_COLUMNS if c not in y_cols]
    if missing or not REGULATED_COLUMNS:
        raise ValueError(
            "REGULATED_COLUMNS must name columns present in the surrogate's "
            f"Y. Missing: {missing or REGULATED_COLUMNS}; Y = {y_cols}"
        )
    return list(REGULATED_COLUMNS), [y_cols.index(c) for c in REGULATED_COLUMNS]


# ---------------------------------------------------------------------
# Required: the comfort band as a sampled time-varying signal
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
    Build (ymin, ymax) of shape [n_samples, nsteps, n_regulated] in
    NORMALIZED space. The band knobs come from YAML via `band`.

    The implementation below is plant-agnostic and needs no edits: per
    window segment it draws a width ~ U[width_min, width_max] and a lower
    bound ~ U[lo, hi - width], all in physical units, then normalizes.
    `band.sub_samples` splits the window into that many contiguous segments,
    producing `sub_samples - 1` step jumps; 1 gives a constant band.

    Rewrite only if your reference is not a randomly sampled interval — e.g.
    a fixed occupancy schedule, or a band read from a column of your data.
    """
    k = band.sub_samples
    if k < 1:
        raise ValueError(f"band.sub_samples must be >= 1, got {k}")
    if k > nsteps:
        raise ValueError(f"band.sub_samples ({k}) cannot exceed nsteps ({nsteps})")

    regulated_cols, _ = regulated(nm_mapping)
    n_ctrl = len(regulated_cols)

    # One (width, low) per (sample, segment); channels share the band.
    widths = band.width_min + (band.width_max - band.width_min) * torch.rand(
        n_samples, k, 1, generator=generator,
    )
    lows = band.lo + ((band.hi - widths) - band.lo) * torch.rand(
        n_samples, k, 1, generator=generator,
    )

    # Spread the k segments across nsteps; the first (nsteps % k) segments
    # are one step longer so the counts sum to nsteps.
    base, rem = divmod(nsteps, k)
    counts = torch.full((k,), base, dtype=torch.long)
    counts[:rem] += 1
    seg_widths = torch.repeat_interleave(widths, counts, dim=1)
    seg_lows = torch.repeat_interleave(lows, counts, dim=1)

    ymin_phys = seg_lows.expand(n_samples, nsteps, n_ctrl)
    ymax_phys = (seg_lows + seg_widths).expand(n_samples, nsteps, n_ctrl)

    shift = torch.tensor([norm_stats[c]["shift"] for c in regulated_cols])
    stretch = torch.tensor([norm_stats[c]["stretch"] for c in regulated_cols])

    return (ymin_phys - shift) / stretch, (ymax_phys - shift) / stretch


# ---------------------------------------------------------------------
# Required: actuator limits for the policy output
# ---------------------------------------------------------------------
@beartype
def actuation_bounds(
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
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
# Set to True once `power_metric` below is implemented. Until then any
# energy_weight / cost_weight / peak_weight > 0 fails at problem build,
# rather than silently optimizing something you did not write.
POWER_METRIC_IMPLEMENTED = False


@beartype
def power_metric(
    yn: torch.Tensor,
    U: torch.Tensor,
    D: torch.Tensor,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> torch.Tensor:
    """
    Instantaneous power per timestep, shape [batch, time]: the signal that
    energy_weight, cost_weight and peak_weight each reduce, and that the
    reportcard integrates.

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
        "power_metric is not implemented. Keep energy_weight: 0.0 in the "
        "control config, or implement this and set "
        "POWER_METRIC_IMPLEMENTED = True."
    )


def _denorm(x: torch.Tensor, col: str, nm_mapping: dict[str, list[str]],
            norm_stats: dict[str, dict[str, float]], role: str = "Y"
            ) -> torch.Tensor:
    """Physical units for one channel of a [batch, time, n] trajectory."""
    s = norm_stats[col]
    return x[:, :, nm_mapping[role].index(col)] * s["stretch"] + s["shift"]


# ---------------------------------------------------------------------
# Optional: the energy tariff, for the cost objective
# ---------------------------------------------------------------------
# No hook: the tariff is config (control.yaml's TARIFF section). Prefer a
# real historical price series to a fixed time-of-use table: a tariff that
# is a function of the clock can be learned as the clock, and the policy
# never has to read the price. Price windows are sampled independently of
# the plant windows, so put the price behaviour you want evaluated in the
# test file. (No price data? PG&E's Hourly Flex Pricing history is public.)


# ---------------------------------------------------------------------
# Optional: the reference rule — baseline, and the residual policy's anchor
# ---------------------------------------------------------------------
# Defined, it does two jobs:
#
#   1. Baseline. `nt control train` rolls the rule out on the same test
#      windows and band as the learned policy and adds a learned-vs-rule
#      table to the reportcard — the only direct answer to "is the learned
#      policy earning its keep?"
#   2. Anchor. With `residual_delta` set, the policy commands
#      `reference_control(...) + δ`, |δ| <= residual_delta: the rule does the
#      routine work and the policy only adjusts it, inside a bound you chose.
#
# Current step only, memoryless, normalized in and normalized U out — the
# same space the learned policy emits. A rule that needs its own state goes
# in `build_baseline_policy` below instead; define one or the other, never
# both (the library raises).
#
# Leave it commented out if you have no rule. The example below is for one
# setpoint per regulated channel; `bracket_band` after it is a rule for the
# common case of a heating/cooling setpoint pair holding one band.
#
# @beartype
# def reference_control(
#     yn_now: torch.Tensor,      # [b, ny]     current state, normalized
#     ymin_now: torch.Tensor,    # [b, n_ctrl] current band, normalized
#     ymax_now: torch.Tensor,    # [b, n_ctrl]
#     d_now: torch.Tensor,       # [b, nd]     current disturbances, normalized
#     nm_mapping: dict[str, list[str]],
#     norm_stats: dict[str, dict[str, float]],
# ) -> torch.Tensor:             # [b, nu]     normalized U
#     """Aim each setpoint at the middle of its channel's band."""
#     regulated_cols, _ = regulated(nm_mapping)
#     y_shift = torch.tensor([norm_stats[c]["shift"] for c in regulated_cols])
#     y_stretch = torch.tensor([norm_stats[c]["stretch"] for c in regulated_cols])
#     target = 0.5 * (ymin_now + ymax_now) * y_stretch + y_shift   # physical
#
#     u_cols = nm_mapping["U"]                       # one per regulated channel
#     u_shift = torch.tensor([norm_stats[c]["shift"] for c in u_cols])
#     u_stretch = torch.tensor([norm_stats[c]["stretch"] for c in u_cols])
#     umin, umax = actuation_bounds(nm_mapping, norm_stats)
#     return torch.clamp((target - u_shift) / u_stretch, umin, umax)


@beartype
def bracket_band(
    lo: torch.Tensor,
    hi: torch.Tensor,
    step: float,
    cool_inset: float = 0.0,
    heat_inset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straddle a band with a (cooling, heating) setpoint pair, in physical units.

    Example code for `reference_control`, not library behaviour. Heating sits
    at the band floor and cooling at the ceiling, so the two bracket the band
    instead of both chasing one target. Both snap to the `step` grid *toward
    the band interior* (heating up, cooling down), never outside it.

    The insets pull each setpoint further inside its edge. A zone usually
    floats somewhat above its cooling setpoint, so `cool_inset` corrects for
    that — measure it on the plant rather than treating it as a comfort
    margin: too large and the two setpoints collide. When the band is too
    narrow to seat both on the grid, both collapse to the grid point nearest
    the midpoint, so heating never commands above cooling.

    Args:
        lo, hi: band edges, any broadcast-compatible shape.
        step: setpoint grid (e.g. 0.5). Must be > 0.
        cool_inset: pull the cooling setpoint below the ceiling by this much.
        heat_inset: pull the heating setpoint above the floor by this much.

    Returns:
        (cooling, heating), each the shape of `lo`/`hi` broadcast together.

    Wired into `reference_control`, after denormalizing the band as above:

        cool, heat = bracket_band(lo, hi, step=0.5, cool_inset=0.3)
        phys = torch.cat([cool, heat], dim=-1)    # U order: [cooling, heating]
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if cool_inset < 0 or heat_inset < 0:
        raise ValueError(
            f"insets must be >= 0, got cool={cool_inset}, heat={heat_inset}"
        )

    heat = torch.ceil((lo + heat_inset) / step) * step
    cool = torch.floor((hi - cool_inset) / step) * step

    mid = torch.round(0.5 * (lo + hi) / step) * step
    collapse = heat > cool
    return torch.where(collapse, mid, cool), torch.where(collapse, mid, heat)


# ---------------------------------------------------------------------
# Optional: a baseline a memoryless rule cannot express
# ---------------------------------------------------------------------
# For a rule that needs its own state (an integrator, an equipment timer).
# Return a Neuromancer Node writing ["U"] with no trainable parameters,
# reading any of "yn", "ymin", "ymax", "D". It is scored, never trained, and
# cannot anchor a residual policy.
#
# def build_baseline_policy(cfg, sysid_cfg, nm_mapping, norm_stats):
#     from neuromancer.system import Node
#     umin, umax = actuation_bounds(nm_mapping, norm_stats)
#     net = MyStatefulRule(umin.to(sysid_cfg.device), umax.to(sysid_cfg.device))
#     return Node(net.to(sysid_cfg.device), ["ymin", "ymax"], ["U"],
#                 name="Stateful baseline")


# ---------------------------------------------------------------------
# Optional: per-signal preview
# ---------------------------------------------------------------------
# Leave commented out unless you need it: if this function EXISTS it fully
# replaces `d_preview` from the YAML, so even an empty stub disables preview.
#
# Keys are D column names or the literal "band" (future ymin/ymax); values
# are step counts. Band preview helps when the dynamics lag, so the policy
# can act before a band change rather than after it. The price preview is
# `price_preview` in the YAML, not here.
#
# @beartype
# def preview_config(nm_mapping: dict[str, list[str]]) -> dict[str, int]:
#     return {"band": 2, "Outdoor Temperature [degC]": 12}
