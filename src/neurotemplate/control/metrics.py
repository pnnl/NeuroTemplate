"""
Closed-loop control metrics in physical units.

Four groups, kept separate because they answer different questions and have
different units:

  - ComfortMetrics — per controlled channel: how often, how badly, and in
    which direction the comfort band was violated.
  - EnergyMetrics  — how much power the closed loop drew, and what the peaks
    looked like.
  - ActuationMetrics — per control input: how hard it worked and how much of
    the evaluation it spent pinned at a bound.
  - JumpResponse   — whether the loop actually reacted when the reference band
    moved, which the pooled comfort numbers cannot see.

The last two exist because pooled comfort scores are silent about behaviour.
A controller that pinned one of its two actuators at a bound for 92% of the
evaluation scored *better* on violation rate than the fixed version of itself,
and only a hand-written trajectory audit found it.

Neither carries a verdict on its own. Time on a bound is not evidence of a
fault — a setpoint held at its least-aggressive limit while nothing is demanded
of it produces exactly the same number as one that is stuck. What separates
those two is whether the loop moved when the reference band asked it to, which
is what JumpResponse measures. Read them together.

Everything here is a pure function over already-denormalized tensors. The
pipeline does the denormalization and the hook calls; this module only does
arithmetic, so the numbers are cheap to test against hand-computed cases.

Energy carries no unit conversion. `control_user_defined.power_metric`
returns whatever the use case meters, so power is reported in those native
units and energy in `power·h`. An app metering Watts converts to kWh itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch

from neurotemplate._typing import beartype


@dataclass
class ComfortMetrics:
    """Comfort-band performance for one controlled channel, in physical units.

    On which number to *tune*: prefer `mean_exceedance` or `rms_exceedance`.
    Both average over all steps, so they keep shrinking as a controller moves
    further inside the band, whereas `violation_rate` is a threshold count that
    saturates. When these metrics are computed against a surrogate rather than
    the real plant, that difference decides the outcome: measured over nine
    controllers spanning 2.5–52.2% true violation, argmin of surrogate
    `violation_rate` selected one 30 points worse than the best available,
    while argmin of either exceedance metric selected the true optimum.

    `violation_rate` remains the right *acceptance* criterion against the plant
    — it is what a building operator asks about — and `mean_violation` is the
    worst of the four to tune on, because averaging over violating steps only
    makes a permanently-but-barely-out-of-band controller look good.
    """
    name: str
    violation_rate: float   # fraction of (window, step) pairs outside the band
    frac_below: float       # too cold and too hot are different failures, so
    frac_above: float       # they are reported separately, not just summed
    mean_violation: float   # mean excursion over violating steps only (0 if none)
    mean_exceedance: float  # mean excursion over ALL steps — tune on this
    rms_exceedance: float   # root-mean-square excursion, the training loss's own
    max_violation: float    # worst single excursion
    degree_hours: float     # mean over windows of ∫|excursion| dt, in unit·h


@dataclass
class EnergyMetrics:
    """Closed-loop power/energy, in the native units of `power_metric`."""
    mean_power: float        # mean instantaneous power
    total_energy: float      # mean over windows of ∫power dt, in power·h
    peak_power: float        # worst instantaneous draw anywhere
    mean_window_peak: float  # mean of per-window maxima — demand-charge proxy
    # Mean over windows of ∫price·power dt. None when no tariff is defined.
    # Units are price·power·h, and the library converts nothing: a
    # `power_metric` in Watts against a price_scale in $/kWh is off by 1000.
    energy_cost: Optional[float] = None


# A command within this share of the actuation range of a bound counts as
# sitting on that bound, and a step of at least this share counts as the
# controller having moved. Bound methods like `sigmoid_scale` never reach a
# limit exactly, so an exact comparison would report no saturation ever.
SATURATION_TOL = 0.01


@dataclass
class ActuationMetrics:
    """What one control input did over the evaluation, in physical units.

    `frac_at_min` / `frac_at_max` are the cheapest diagnostic in this module.
    They say how much of the evaluation each control spent against a limit,
    which no comfort statistic reports — but they do not say whether that was
    wrong. Pair them with `JumpResponse` before concluding anything.
    """
    name: str
    mean_value: float      # mean commanded value
    mean_abs_du: float     # mean |Δ| between consecutive commands
    frac_at_min: float     # share of steps sitting on the lower bound
    frac_at_max: float     # share of steps sitting on the upper bound
    frac_saturated: float  # share sitting on *either* bound
    u_min: float           # the actuation bounds themselves, for context
    u_max: float


@dataclass
class ResidualMetrics:
    """How far a residual policy's commands sat from their reference rule.

    `frac_at_cap` is the line that earns its place: if the ±Δ box is binding
    most of the time, either the reference is wrong for this plant or the
    optimizer is riding a surrogate artifact into the wall. Both are things to
    know before reading the comfort number.

    It is None at Δ = 0, where the question does not exist: every deviation is
    trivially at the bound of a zero-width box, so the fraction is 1.0 and
    means nothing. Δ = 0 is the configuration the library recommends running
    first (it scores the bare rule), so this is a case worth stating rather
    than a corner. `mean_abs_delta` still earns its place there — it is the
    assertion that the residual path really is inert.
    """
    delta_cap: float        # Δ, the configured half-width, in physical units
    mean_abs_delta: float   # mean |commanded − reference|
    max_abs_delta: float
    frac_at_cap: Optional[float]  # share pinned at ±Δ; None when Δ = 0


@dataclass
class JumpResponse:
    """The loop's answer to reference-band step changes, one demand direction.

    A band jump that leaves the controlled channel outside the new band is a
    request the controller either answers or ignores. Pooled violation rates
    cannot tell those apart — a policy can score well by sitting still
    somewhere lucky — so each jump is scored on three separate things:

      actuated — some control moved on the same step as the jump
      closed   — the channel then moved in the demanded direction
      resolved — the channel was inside the band by the end of the segment

    `mean_du` breaks the reaction down per control input, which is what names
    the actuator that is not pulling its weight.
    """
    name: str                   # controlled channel
    direction: str              # "rise" (channel below band) or "fall" (above)
    n_jumps: int
    frac_actuated: float
    frac_closed: float
    frac_resolved: float
    mean_move: float            # mean signed channel movement over the segment
    mean_span: float            # mean steps scored per jump, i.e. the horizon
    mean_du: dict[str, float]   # mean signed Δcommand per control input


@beartype
def hours_per_step(t: torch.Tensor) -> float:
    """Timestep length in hours, from the `t` channel (seconds elapsed since
    the file's first row).

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
            mean_exceedance=float(excursion.mean()),
            rms_exceedance=float(excursion.pow(2).mean().sqrt()),
            max_violation=float(excursion.max()),
            # Sum over time for each window, then average across windows: the
            # per-window integral is what a building operator recognizes.
            degree_hours=float(excursion.sum(dim=1).mean()) * step_hours,
        ))

    return results


@beartype
def energy_metrics(power: torch.Tensor, step_hours: float,
                   price: Optional[torch.Tensor] = None) -> EnergyMetrics:
    """
    Args:
        power: [batch, time] instantaneous power per rollout window.
        step_hours: rollout timestep in hours.
        price: optional [batch, time] tariff in physical units, co-indexed
            with `power`. None leaves `energy_cost` unset.

    Peak conventions, both reported because they answer different questions:
    `peak_power` is the worst instant anywhere in the evaluation, while
    `mean_window_peak` is the typical per-window peak — closer to what a
    demand charge bills against. Neither applies rolling-average smoothing;
    that is a tariff detail and belongs in the use case.
    """
    if power.ndim != 2:
        raise ValueError(f"Expected 2D [batch, time] power; got {power.ndim}D")
    if price is not None and price.shape != power.shape:
        raise ValueError(
            f"price {tuple(price.shape)} must be co-indexed with power "
            f"{tuple(power.shape)} — same windows, same steps."
        )

    power = power.detach().cpu()
    window_peaks = power.max(dim=1).values
    cost = (None if price is None else
            float((power * price.detach().cpu()).sum(dim=1).mean()) * step_hours)

    return EnergyMetrics(
        mean_power=float(power.mean()),
        total_energy=float(power.sum(dim=1).mean()) * step_hours,
        peak_power=float(power.max()),
        mean_window_peak=float(window_peaks.mean()),
        energy_cost=cost,
    )


@beartype
def actuation_metrics(
    u: torch.Tensor,
    u_names: list[str],
    u_min: torch.Tensor,
    u_max: torch.Tensor,
) -> list[ActuationMetrics]:
    """
    Args:
        u:      [batch, time, nu] commanded controls in physical units.
        u_names: names for the nu axis.
        u_min/u_max: actuation bounds in physical units, one per control.

    Returns:
        One ActuationMetrics per control, in the order of u_names.
    """
    if u.ndim != 3:
        raise ValueError(f"Expected 3D [batch, time, nu] controls; got {u.ndim}D")
    if len(u_names) != u.shape[-1]:
        raise ValueError(
            f"u_names has {len(u_names)} entries but the control tensor has "
            f"nu={u.shape[-1]}"
        )
    if u_min.shape != (len(u_names),) or u_max.shape != (len(u_names),):
        raise ValueError(
            f"u_min/u_max must have one entry per control ({len(u_names)}); "
            f"got {tuple(u_min.shape)} and {tuple(u_max.shape)}"
        )

    u = u.detach().cpu()
    u_min, u_max = u_min.detach().cpu(), u_max.detach().cpu()
    tol = SATURATION_TOL * (u_max - u_min)
    du = (u[:, 1:, :] - u[:, :-1, :]).abs()

    results = []
    for j, name in enumerate(u_names):
        at_min = u[:, :, j] <= u_min[j] + tol[j]
        at_max = u[:, :, j] >= u_max[j] - tol[j]
        results.append(ActuationMetrics(
            name=name,
            mean_value=float(u[:, :, j].mean()),
            mean_abs_du=float(du[:, :, j].mean()),
            frac_at_min=float(at_min.float().mean()),
            frac_at_max=float(at_max.float().mean()),
            # Counted as an OR, not a sum: a degenerate box (u_min == u_max)
            # puts every command on both bounds at once, and a "200% pinned"
            # line reads as a bug rather than as the config error it is.
            frac_saturated=float((at_min | at_max).float().mean()),
            u_min=float(u_min[j]),
            u_max=float(u_max[j]),
        ))
    return results


@beartype
def residual_metrics(
    u: torch.Tensor,
    u_ref: torch.Tensor,
    delta_cap: float,
    tol: float,
) -> ResidualMetrics:
    """
    Args:
        u:     [batch, time, nu] commanded controls in physical units.
        u_ref: [batch, time, nu] the reference rule's commands, same units.
        delta_cap: the configured Δ, in physical units.
        tol:   how close to Δ counts as pinned. Half a setpoint grid step is
            the natural choice when quantization is on, because the commanded
            deviation can only land on grid points.

    This is the *realized* residual — measured after quantization and after
    the actuator clamp — so it is the deviation that was actually deployed,
    not the network's pre-bound output.

    One consequence worth knowing before reading `mean_abs_delta`: when
    quantization is on and the reference rule is not itself snapped to the
    same grid, `u` cannot equal `u_ref` even at δ = 0, so mean |δ| has a floor
    of up to half a grid step which is snapping rather than steering. A rule
    built over `bracket_band` is grid-aligned and has no such floor.
    """
    if u.shape != u_ref.shape:
        raise ValueError(
            f"u {tuple(u.shape)} and u_ref {tuple(u_ref.shape)} must match"
        )
    if delta_cap < 0:
        raise ValueError(f"delta_cap must be >= 0, got {delta_cap}")

    delta = (u - u_ref).detach().cpu().abs()
    return ResidualMetrics(
        delta_cap=delta_cap,
        mean_abs_delta=float(delta.mean()),
        max_abs_delta=float(delta.max()),
        # A zero-width box has no interior, so "pinned at the bound" is true of
        # every step and says nothing. Reported as absent rather than as 100%.
        frac_at_cap=(None if delta_cap == 0 else
                     float((delta >= delta_cap - tol).float().mean())),
    )


def _band_jumps(lo: torch.Tensor, hi: torch.Tensor) -> list[int]:
    """Steps at which the band moved, for one (window, channel) pair.

    Read off the band signal itself rather than reconstructed from
    `BandSpec.sub_samples`, so a use case that overrides `comfort_band_signal`
    with an occupancy schedule or a recorded column is audited just the same.

    A band that changes at *every* step — a continuously varying reference
    rather than a piecewise-constant one — leaves no segment to score, and
    `response_audit` then reports nothing rather than reporting noise.
    """
    changed = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    return (changed.nonzero().flatten() + 1).tolist()


@beartype
def response_audit(
    y: torch.Tensor,
    ymin: torch.Tensor,
    ymax: torch.Tensor,
    u: torch.Tensor,
    channel_names: list[str],
    u_names: list[str],
    u_min: torch.Tensor,
    u_max: torch.Tensor,
) -> list[JumpResponse]:
    """Score every reference-band step change that left the channel out of band.

    Args:
        y/ymin/ymax: [batch, time, n_ctrl] controlled channel and its band, in
            physical units.
        u:      [batch, time, nu] commanded controls in physical units.
        u_min/u_max: actuation bounds in physical units, used to size the step
            that counts as the controller having reacted.

    Each jump is scored over the segment that follows it — up to the next jump,
    or the end of the window. That keeps the horizon parameter-free: the
    question is whether the loop answered the request before the request
    changed again.

    Returns one JumpResponse per (channel, direction) that had at least one
    such jump, so an empty list means the band never moved the channel out of
    its own band and there was nothing to audit.
    """
    if not (y.shape == ymin.shape == ymax.shape):
        raise ValueError(
            f"Shape mismatch: y {tuple(y.shape)}, ymin {tuple(ymin.shape)}, "
            f"ymax {tuple(ymax.shape)}"
        )
    if y.ndim != 3 or u.ndim != 3:
        raise ValueError("Expected 3D [batch, time, n] tensors")
    if y.shape[:2] != u.shape[:2]:
        raise ValueError(
            f"y and u must share batch/time; got {tuple(y.shape[:2])} vs "
            f"{tuple(u.shape[:2])}"
        )
    if len(channel_names) != y.shape[-1] or len(u_names) != u.shape[-1]:
        raise ValueError(
            f"Name lists must match tensor widths: {len(channel_names)} vs "
            f"n_ctrl={y.shape[-1]}, {len(u_names)} vs nu={u.shape[-1]}"
        )

    y, ymin, ymax, u = (t.detach().cpu() for t in (y, ymin, ymax, u))
    du_tol = SATURATION_TOL * (u_max.detach().cpu() - u_min.detach().cpu())
    n_windows, n_steps, _ = y.shape

    # (channel, direction) -> list of per-jump records.
    hits: dict[tuple[str, str], list[dict]] = {}
    for w in range(n_windows):
        for c, name in enumerate(channel_names):
            lo, hi = ymin[w, :, c], ymax[w, :, c]
            jumps = _band_jumps(lo, hi)
            for n, i in enumerate(jumps):
                end = jumps[n + 1] - 1 if n + 1 < len(jumps) else n_steps - 1
                if end <= i:
                    continue
                y_at, y_end = float(y[w, i, c]), float(y[w, end, c])
                lo_i, hi_i = float(lo[i]), float(hi[i])
                if lo_i <= y_at <= hi_i:
                    continue  # the jump asked for nothing
                direction = "rise" if y_at < lo_i else "fall"
                sign = 1.0 if direction == "rise" else -1.0
                du = u[w, i, :] - u[w, i - 1, :]
                hits.setdefault((name, direction), []).append({
                    "actuated": bool((du.abs() > du_tol).any()),
                    "closed": sign * (y_end - y_at) > SATURATION_TOL * (hi_i - lo_i),
                    "resolved": (float(ymin[w, end, c]) <= y_end
                                 <= float(ymax[w, end, c])),
                    "move": y_end - y_at,
                    "span": float(end - i),
                    "du": du,
                })

    def mean_of(rows: list[dict], key: str) -> float:
        return sum(float(r[key]) for r in rows) / len(rows)

    results = []
    for name in channel_names:                 # stable, name-then-direction order
        for direction in ("rise", "fall"):
            rows = hits.get((name, direction))
            if not rows:
                continue
            mean_du = torch.stack([r["du"] for r in rows]).mean(dim=0)
            results.append(JumpResponse(
                name=name,
                direction=direction,
                n_jumps=len(rows),
                frac_actuated=mean_of(rows, "actuated"),
                frac_closed=mean_of(rows, "closed"),
                frac_resolved=mean_of(rows, "resolved"),
                mean_move=mean_of(rows, "move"),
                mean_span=mean_of(rows, "span"),
                mean_du={n: float(mean_du[j]) for j, n in enumerate(u_names)},
            ))
    return results


@beartype
def comfort_to_markdown(metrics: list[ComfortMetrics]) -> str:
    """Render comfort metrics as a markdown table for the reportcard."""
    header = (
        "| Channel | Mean exceedance | RMS exceedance | Violation rate | "
        "Below | Above | Mean excursion (violating only) | Max excursion | "
        "Degree-hours |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for m in metrics:
        rows.append(
            f"| {m.name} | {m.mean_exceedance:.4f} | {m.rms_exceedance:.4f} | "
            f"{m.violation_rate * 100:.2f}% | "
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


@beartype
def actuation_to_markdown(metrics: list[ActuationMetrics]) -> str:
    """Render per-control actuation as a markdown table.

    No flag column: time spent on a bound is reported, not judged. A setpoint
    parked at its least-aggressive limit through a period that demands nothing
    of it is correct control and produces the same number as a stuck one.
    """
    rows = [
        "| Control | Bounds | Mean | Mean \\|Δu\\| | At lower bound | "
        "At upper bound |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m in metrics:
        rows.append(
            f"| {m.name} | [{m.u_min:.4g}, {m.u_max:.4g}] | "
            f"{m.mean_value:.4g} | {m.mean_abs_du:.4g} | "
            f"{m.frac_at_min * 100:.1f}% | {m.frac_at_max * 100:.1f}% |"
        )
    return "\n".join(rows)


@beartype
def response_to_markdown(responses: list[JumpResponse],
                         u_names: list[str]) -> str:
    """Render the band-jump audit: one row per (channel, demand direction)."""
    du_cols = "".join(f" Δ`{n}` |" for n in u_names)
    rows = [
        "| Channel | Demand | Jumps | Steps scored | Actuated | Moved right "
        f"way | Back in band | Mean move |{du_cols}",
        "|---|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(u_names),
    ]
    for r in responses:
        du = "".join(f" {r.mean_du[n]:+.3g} |" for n in u_names)
        rows.append(
            f"| {r.name} | {r.direction} | {r.n_jumps} | {r.mean_span:.0f} | "
            f"{r.frac_actuated * 100:.0f}% | {r.frac_closed * 100:.0f}% | "
            f"{r.frac_resolved * 100:.0f}% | {r.mean_move:+.3g} |{du}"
        )
    return "\n".join(rows)


@beartype
def actuation_to_dict(metrics: list[ActuationMetrics]) -> dict:
    """JSON-serializable actuation metrics (for MLflow / disk)."""
    return {m.name: {k: v for k, v in asdict(m).items() if k != "name"}
            for m in metrics}


@beartype
def response_to_dict(responses: list[JumpResponse]) -> dict:
    """JSON-serializable band-jump audit (for MLflow / disk)."""
    return {f"{r.name} | {r.direction}": {k: v for k, v in asdict(r).items()
                                          if k not in ("name", "direction")}
            for r in responses}
