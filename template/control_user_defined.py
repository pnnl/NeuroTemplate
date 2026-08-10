"""
Control hooks — the glue between the DPC machinery and *your* plant.

These answer: which states are held inside a reference band, what that band
looks like, what the actuator limits are, and (optionally) what "effort"
costs. The Y/U/D roles and the normalization come from the frozen sys-id run,
so nothing here re-derives them.

Everything returned "in normalized space" uses the same per-column
shift/stretch as the loaded surrogate (norm_stats), so it lines up directly
with the surrogate's states and inputs.

The file must be named `control_user_defined.py` and live in the directory
you run `nt` from, or be pointed at by `user_hooks_path` in your YAML.

Fill in `controlled_channels` first. `comfort_band_signal` and
`actuation_bounds` below are generic and work unedited.
"""

from __future__ import annotations

from typing import Optional

import torch

from neurotemplate._typing import beartype
from neurotemplate.control.scenarios import BandSpec


# ---------------------------------------------------------------------
# Required: which Y channels the comfort band applies to
# ---------------------------------------------------------------------
# The states the policy is judged on. Everything else in Y is along for the
# ride. Names must match the surrogate's Y columns exactly.
CONTROLLED_COLUMNS: list[str] = [
    # "Tank Temperature [degC]",
]


@beartype
def controlled_channels(
    nm_mapping: dict[str, list[str]],
) -> tuple[list[str], list[int]]:
    """Return (column names, their integer indices into nm_mapping["Y"])."""
    y_cols = nm_mapping["Y"]
    missing = [c for c in CONTROLLED_COLUMNS if c not in y_cols]
    if missing or not CONTROLLED_COLUMNS:
        raise ValueError(
            "CONTROLLED_COLUMNS must name columns present in the surrogate's "
            f"Y. Missing: {missing or CONTROLLED_COLUMNS}; Y = {y_cols}"
        )
    return list(CONTROLLED_COLUMNS), [y_cols.index(c) for c in CONTROLLED_COLUMNS]


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
    Build (ymin, ymax) of shape [n_samples, nsteps, n_controlled] in
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

    controlled, _ = controlled_channels(nm_mapping)
    n_ctrl = len(controlled)

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

    shift = torch.tensor([norm_stats[c]["shift"] for c in controlled])
    stretch = torch.tensor([norm_stats[c]["stretch"] for c in controlled])

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

    The default below uses each U column's observed physical range in the
    training data, which keeps the policy inside the region the surrogate
    was actually identified on. Replace with hard-coded physical limits if
    the real actuator range is wider or narrower than what the data covers.
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
# Required when `effort_weight > 0`: the quantity the effort term minimizes
# ---------------------------------------------------------------------
# Set to True once `power_metric` below is implemented. The library checks
# this flag at problem-build time and fails fast if effort_weight > 0 while
# it is False, so `effort_weight` must stay 0.0 in your YAML until then.
#
# Failing fast is the point. The tempting alternative — quietly falling back
# to an actuation proxy when the surrogate meters no power — substitutes a
# different objective in different units, and it is where three of this
# library's four recorded campaigns lost iterations. See `power_metric`.
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
    Instantaneous power per timestep, shape [batch, time]. The training
    objective drives its closed-loop mean toward zero and the reportcard
    integrates it over time.

    Inputs are the normalized trajectories: yn [batch, time, ny],
    U [batch, time, nu], D [batch, time, nd]. Denormalize whatever you use
    (`x * stretch + shift`) — the library never converts units, and labels
    the integrated result `power·h`, so returning Watts gives Wh.

    The shape this should take: sum the power channels the surrogate already
    predicts.

        total = sum(_denorm(yn, c, nm_mapping, norm_stats)
                    for c in power_channels(nm_mapping))
        return total.clamp_min(0.0)

    Metered power channels are the best thing to return when the surrogate
    predicts them, because the units are real and the reportcard's energy
    numbers then mean something outside this run.

    An actuation proxy such as `(setpoint - ambient) ** 2` is a legitimate
    alternative when nothing meters power. It is a different quantity in
    different units, so read the energy section as relative rather than
    physical — but a term that drives good closed-loop behaviour is a good
    term, whatever its analytic minimum looks like.

    The one thing that has actually cost campaigns here is **scale**, not
    shape: a proxy weighted to a previous dataset came out at 1.7x the entire
    comfort objective and every pooled metric looked fine for ten consecutive
    runs. Before training, the library prints each term's value and share of
    the objective, and sweeps this term across each actuation box so its
    magnitude and shape are both on screen. Compare them and set
    `effort_weight` from what you see.

    If you would rather not weight an effort term at all, leave this
    unimplemented and keep `effort_weight: 0.0`.
    """
    raise NotImplementedError(
        "power_metric is not implemented. Keep effort_weight: 0.0 in the "
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
# Optional: a rule-based controller to score the learned policy against
# ---------------------------------------------------------------------
def build_baseline_policy(cfg, sysid_cfg, nm_mapping, norm_stats):
    """
    Return a Neuromancer Node writing ["U"], or None to skip the comparison
    (the reportcard then says the section is empty).

    Worth the twenty lines: it is the only thing that tells you whether the
    learned policy is beating the obvious rule. The Node must have no
    trainable parameters — it is evaluated, never trained — and emit
    normalized U, the same space the learned policy emits.

        from neuromancer.system import Node
        umin, umax = actuation_bounds(nm_mapping, norm_stats)
        net = MyRule(umin.to(sysid_cfg.device), umax.to(sysid_cfg.device))
        return Node(net.to(sysid_cfg.device), ["ymin", "ymax"], ["U"],
                    name="Bracket baseline")

    Inputs can be any of the closed-loop signals ("yn", "ymin", "ymax", "D");
    see tests/fixtures/control_user_defined.py for a complete worked example.

    On what rule to write, for a two-actuator heating/cooling setpoint pair:
    **bracket the band** — put the heating setpoint at the band floor and the
    cooling setpoint at (or just inside) the band ceiling, so the two
    setpoints straddle the band rather than both aiming at one target. Two
    details decide whether it works, and both are easy to get backwards:

      - **Round each setpoint toward the band interior** (heating up, cooling
        down) when snapping to a quantization grid. Rounding outward puts a
        setpoint outside the band it is supposed to hold.
      - **A small inset on the cooling side is a correction, not a safety
        margin.** A zone typically floats *above* its cooling setpoint,
        because the plant's setpoint→zone gain is well under 1 °C/°C. Aiming
        exactly at the ceiling therefore violates constantly. But the useful
        inset range can be narrower than one grid step: too large an inset
        makes the two setpoints collide, and the rule stops bracketing at all.
        Measure the gain and sweep the inset rather than guessing.

    A plain band-midpoint rule (`0.5 * (ymin + ymax)` on both actuators) is
    simpler and worth having as a sanity check, but it does not bracket, so it
    gives up the deadband the two actuators could be holding.
    """
    return None


# ---------------------------------------------------------------------
# Optional: how many future steps of each signal the policy previews
# ---------------------------------------------------------------------
# LEAVE THIS COMMENTED OUT unless you want per-signal preview. If the
# function merely EXISTS, it fully replaces `cfg.d_preview` — so a stub
# returning {} would silently disable a preview set in YAML.
#
# Keys are D column names (disturbance forecast) or the literal "band"
# (future ymin/ymax). Values are non-negative step counts; omitted signals
# get no preview. The disturbance window is one block, so it is sized to the
# largest per-column value requested.
#
# Band preview earns its keep when the dynamics lag: the state at step i
# responds to the input chosen at i-1, so the policy needs to see the band
# it is steering into or it overshoots at every band change.
#
# @beartype
# def preview_config(nm_mapping: dict[str, list[str]]) -> dict[str, int]:
#     return {"band": 2, "Ambient Temperature [degC]": 2}
