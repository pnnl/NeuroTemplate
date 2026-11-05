"""
The thermostat: from setpoints to delivered heat, as a few physical parameters.

    setpoint ─[thermostat]→ Q delivered (kW) ─[RC network]→ zone temperature

Each side — cooling, heating — is a tracker on its error `e`, positive while
it is called for (`zone − setpoint` cooling, `setpoint − zone` heating):

    Q = sign · clip(K · (e − offset), 0, capacity)

with the sides adding. `K` is a gain in kW/K, `capacity` in kW, `offset` in
K. The ladder of forms, each one more parameter per side:

    tracker      K and capacity
    +offset      where the tracker starts, against the setpoint
    +outdoor     capacity linear in outdoor temperature, about a reference
    +stage2      a second stage of its own capacity, `gap` K further on
    latched      not a tracker: a stage that switches on at `on_at` past the
                 setpoint and holds until `off_at` back on the satisfied side
                 (in kW, one carried stage per side)

The clip is soft on the way back (`soft_clamp_min`'s trick): the value is the
hard clip, the gradient a softplus's, so a controller parked on the satisfied
side still feels the way back. The fit itself uses the smooth form, whose
value and gradient agree, as a line search needs.

Fitted one step at a time: the heat over the step from k to k+1, which the
meter reports at row k+1, against the measured zone, setpoint and outdoor
temperature at row k, under a Gaussian likelihood, so nested forms get the
ladder's likelihood-ratio test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F

from neurotemplate.modeling.network import (
    COLLAPSE_MARGIN, DTYPE, LOG_LIMIT, affine_scan, standard_errors, suggested_box)
from neurotemplate.modeling.hinge import straight_through

SIDES = {"cooling": -1, "heating": +1}
#: The smallest smoothing width: well under any capacity that does real work, and large
#: enough that 1 / width² stays a float32.
MIN_WIDTH = 1e-3


@dataclass(frozen=True)
class Form:
    offset: bool = False
    outdoor: bool = False
    stage2: bool = False
    latched: bool = False

    @property
    def name(self) -> str:
        return ("latched" if self.latched else "tracker") + "+offset" * self.offset \
            + "+outdoor" * self.outdoor + "+stage2" * self.stage2

    def extensions(self) -> list["Form"]:
        return [replace(self, **{k: True}) for k in ("offset", "outdoor", "stage2")
                if not getattr(self, k)]

    def side_parameters(self) -> list[tuple[str, bool]]:
        """`(name, positive)` per side: positive ones are fitted in log space."""
        names = [("gain", True), ("capacity", True)]
        if self.latched:
            names += [("on_at", False), ("off_at", False)]
        elif self.offset:
            names.append(("offset", False))
        if self.outdoor:
            names.append(("capacity_per_K", False))
        if self.stage2:
            names += [("capacity2", True), ("gap", True)]
        return names


def parameter_names(form: Form, sides: tuple[str, ...]) -> list[tuple[str, bool]]:
    return [(f"{side}.{name}", positive) for side in sides
            for name, positive in form.side_parameters()] + [("noise", True)]


def unpack(form: Form, sides: tuple[str, ...], theta: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: torch.exp(t) if positive else t
            for (name, positive), t in zip(parameter_names(form, sides), theta)}


def _floored_width(width: torch.Tensor) -> torch.Tensor:
    """A smoothing width no smaller than `MIN_WIDTH`, and exactly `width` above it. A capacity
    fitted with an outdoor slope can fall to 1e-25 at some hour: the forward pass survives
    that, but the gradient's 1 / width² overflows and takes every weight to NaN. As a `relu`,
    because onnx2c cannot generate a Max."""
    return width + torch.relu(MIN_WIDTH - width)


def _smooth_clip(u: torch.Tensor, cap: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    width = _floored_width(width)
    return width * (F.softplus(u / width) - F.softplus((u - cap) / width))


def soft_clip(u: torch.Tensor, cap: torch.Tensor, smooth: bool) -> torch.Tensor:
    """`clip(u, 0, cap)`; its gradient a softplus's, which never dies. The
    smoothing is a twentieth of the capacity. `smooth` returns the smooth
    form's value too, for a fit."""
    soft = _smooth_clip(u, cap, cap / 20)
    if smooth:
        return soft
    # min(a, cap) as a − relu(a − cap): onnx2c cannot generate a Min.
    above = torch.relu(u)
    return straight_through(soft, above - torch.relu(above - cap))


def side_heat(p: dict, side: str, form: Form, error: torch.Tensor, outdoor: torch.Tensor,
              reference: float, smooth: bool, stage: Optional[torch.Tensor] = None
              ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """`(heat, stage)` for one side and one step: `error` is positive while
    the side is called for. `stage` is a latched side's carried state, None
    on a cold call, which starts it where a memoryless thermostat would be."""
    capacity = _capacity(p, side, form, outdoor, reference)
    if form.latched:
        on, hold = _switches(p, side, error, capacity)
        stage = on if stage is None else on + (1 - on) * hold * stage
        heat = capacity * stage
    else:
        heat = soft_clip(p[f"{side}.gain"] * (error - _start_at(p, side, form)), capacity,
                         smooth)
    return SIDES[side] * (heat + _stage2(p, side, form, error, smooth)), stage


def _capacity(p, side, form, outdoor, reference):
    capacity = p[f"{side}.capacity"]
    if form.outdoor:
        capacity = F.softplus(capacity + p[f"{side}.capacity_per_K"] * (outdoor - reference))
    return capacity


def _start_at(p, side, form):
    """The error at which the first stage starts."""
    if form.latched:
        return p[f"{side}.on_at"]
    return p[f"{side}.offset"] if form.offset else 0.0


def _switches(p, side, error, capacity):
    """A latched stage's `(on, hold)`: smooth steps whose slope, times the
    capacity, is the side's gain."""
    width = _floored_width(capacity / (2 * p[f"{side}.gain"]))
    return (torch.sigmoid((error - p[f"{side}.on_at"]) / width),
            torch.sigmoid((error + p[f"{side}.off_at"]) / width))


def _stage2(p, side, form, error, smooth):
    if not form.stage2:
        return 0.0
    start = _start_at(p, side, form) + p[f"{side}.gap"]
    return soft_clip(p[f"{side}.gain"] * (error - start), p[f"{side}.capacity2"], smooth)


def errors(setpoints: dict[str, torch.Tensor], zone: torch.Tensor) -> dict[str, torch.Tensor]:
    """Each side's error, positive while it is called for."""
    return {side: SIDES[side] * (sp - zone) for side, sp in setpoints.items()}


def one_step_heat(p: dict, form: Form, setpoints: dict[str, torch.Tensor], zone: torch.Tensor,
                  outdoor: torch.Tensor, reference: float) -> torch.Tensor:
    """Predicted heat over each step from the measured row at its start. A
    latched stage runs through the record: `s ← on + (1 − on)·hold·s` is
    affine in `s`, so it runs as a scan."""
    total = torch.zeros_like(zone)
    for side, error in errors(setpoints, zone).items():
        if form.latched:
            capacity = _capacity(p, side, form, outdoor, reference)
            on, hold = _switches(p, side, error, capacity)
            carry, rest = affine_scan(((1 - on) * hold)[1:, None, None], on[1:, None])
            stage = torch.cat([on[:1], carry[:, 0, 0] * on[0] + rest[:, 0]])
            heat = capacity * stage + _stage2(p, side, form, error, smooth=True)
            total = total + SIDES[side] * heat
        else:
            total = total + side_heat(p, side, form, error, outdoor, reference, smooth=True)[0]
    return total


@dataclass
class Record:
    """One file: the zone, each declared side's setpoint and the outdoor
    temperature at each row, and the heat meter's reading."""

    zone: torch.Tensor
    setpoints: dict[str, torch.Tensor]
    outdoor: torch.Tensor
    heat: torch.Tensor


def negative_log_likelihood(form: Form, sides: tuple[str, ...], theta: torch.Tensor,
                            records: list[Record], reference: float) -> torch.Tensor:
    p = unpack(form, sides, theta)
    total = torch.zeros((), dtype=DTYPE)
    for r in records:
        predicted = one_step_heat(p, form, {s: sp[:-1] for s, sp in r.setpoints.items()},
                                  r.zone[:-1], r.outdoor[:-1], reference)
        residual = r.heat[1:] - predicted
        total = total + 0.5 * (residual.shape[0] * torch.log(2 * math.pi * p["noise"] ** 2)
                               + ((residual / p["noise"]) ** 2).sum())
    return total


@dataclass
class ThermostatFit:
    form: Form
    sides: tuple[str, ...]
    values: dict[str, float]
    se: dict[str, float]             # of the log for a positive parameter, else of the value
    log_likelihood: float
    n_obs: int
    reference: float

    @property
    def n_params(self) -> int:
        return len(self.values)

    @property
    def bic(self) -> float:
        return -2 * self.log_likelihood + self.n_params * math.log(self.n_obs)


def _start(form: Form, sides: tuple[str, ...], records: list[Record],
           seed: Optional[dict[str, float]]) -> torch.Tensor:
    """A gentle tracker, every capacity at the largest heat measured on its
    side; a second stage starts nearly empty, so an extension starts as its
    parent and can only gain likelihood. `seed` (a parent's values) wins."""
    heat = torch.cat([r.heat for r in records])
    start = {"noise": float(heat.std())}
    for side in sides:
        peak = float((SIDES[side] * heat).clamp(min=0).max()) or 1.0
        start.update({f"{side}.gain": 10.0, f"{side}.capacity": peak,
                      f"{side}.offset": 0.0, f"{side}.capacity_per_K": 0.0,
                      f"{side}.on_at": 0.0, f"{side}.off_at": 0.5,
                      f"{side}.capacity2": peak / 100, f"{side}.gap": 0.5})
    start.update(seed or {})
    return torch.tensor([math.log(start[name]) if positive else start[name]
                         for name, positive in parameter_names(form, sides)], dtype=DTYPE)


def fit(form: Form, sides: tuple[str, ...], records: list[Record],
        seed: Optional[dict[str, float]] = None, max_iter: int = 300) -> ThermostatFit:
    """Maximum likelihood by L-BFGS; the reference outdoor temperature is
    the record's mean."""
    reference = float(torch.cat([r.outdoor for r in records]).mean())
    theta = _start(form, sides, records, seed).requires_grad_(True)
    positive = torch.tensor([pos for _, pos in parameter_names(form, sides)])
    optimizer = torch.optim.LBFGS([theta], lr=1, max_iter=max_iter, tolerance_grad=1e-6,
                                  tolerance_change=1e-9, history_size=20,
                                  line_search_fn="strong_wolfe")

    def bounded(t):
        return torch.where(positive, LOG_LIMIT * torch.tanh(t / LOG_LIMIT), t)

    def closure():
        optimizer.zero_grad()
        loss = negative_log_likelihood(form, sides, bounded(theta), records, reference)
        if not torch.isfinite(loss):
            return torch.tensor(1e12, dtype=DTYPE)
        loss.backward()
        return loss

    optimizer.step(closure)
    theta = bounded(theta).detach()
    hessian = torch.autograd.functional.hessian(
        lambda t: negative_log_likelihood(form, sides, t, records, reference), theta)
    collapsed = positive & (theta.abs() > LOG_LIMIT - COLLAPSE_MARGIN)
    _, se, _ = standard_errors(hessian, collapsed)
    names = [name for name, _ in parameter_names(form, sides)]
    return ThermostatFit(
        form, sides,
        {k: float(v) for k, v in unpack(form, sides, theta).items()},
        dict(zip(names, map(float, se))),
        -float(negative_log_likelihood(form, sides, theta, records, reference)),
        sum(r.heat.shape[0] - 1 for r in records), reference)


@dataclass
class Selection:
    fits: dict[str, ThermostatFit]
    parents: dict[str, str]
    chosen: str
    latched: Optional[str]       # the latched form of the chosen one, by BIC against it


def forward_select(sides: tuple[str, ...], records: list[Record], latched: bool = True,
                   p_stop: float = 0.05) -> Selection:
    """The ladder's forward selection over tracker forms, then the chosen
    form's latched twin, which nests nothing, against it by BIC."""
    from neurotemplate.modeling.ladder import lr_p_value

    current = Form()
    fits = {current.name: fit(current, sides, records)}
    parents = {}
    while True:
        candidates = current.extensions()
        if not candidates:
            break
        for form in candidates:
            fits[form.name] = fit(form, sides, records, seed=fits[current.name].values)
            parents[form.name] = current.name
        best = max(candidates, key=lambda f: fits[f.name].log_likelihood)
        parent, child = fits[current.name], fits[best.name]
        if lr_p_value(parent.log_likelihood, parent.n_params,
                      child.log_likelihood, child.n_params) > p_stop:
            break
        current = best
    twin = None
    if latched:
        twin_form = replace(current, latched=True, offset=False)
        fits[twin_form.name] = fit(twin_form, sides, records, seed=fits[current.name].values)
        twin = twin_form.name
    return Selection(fits, parents, current.name, twin)


def declaration(fit: ThermostatFit, setpoints: dict[str, str], gain_limit: float,
                z: float = 3.0) -> dict:
    """The thermostat as `block_mapping` declares it, per side: its setpoint
    column and each parameter as a box from its error bar (log space for a
    scale, linear for an offset); and the gain past which the loop it closes
    on the zone rings."""
    out = {"gain_limit": float(f"{gain_limit:.4g}")}
    for side in fit.sides:
        entry = {"setpoint": setpoints[side]}
        for name, positive in fit.form.side_parameters():
            key = f"{side}.{name}"
            value, se = fit.values[key], fit.se[key]
            if positive:
                entry[name] = suggested_box(value, se, z)
            elif math.isinf(se):
                entry[name] = None      # the data does not place it: declare it yourself
            else:
                entry[name] = [float(f"{value - z * se:.4g}"), float(f"{value + z * se:.4g}")]
        out[side] = entry
    if fit.form.outdoor:
        out["outdoor_reference"] = round(fit.reference, 1)
    out["form"] = fit.form.name
    return out


# ── The end-to-end check ────────────────────────────────────────

REGIMES = ("satisfied", "tracking", "saturated")


def composed_rollout(fit: ThermostatFit, network, values: dict[str, float], x0: torch.Tensor,
                     setpoints: dict[str, torch.Tensor], outdoor: torch.Tensor,
                     drive: torch.Tensor, heat_column: int, dt_h: float
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """`(zone, heat)` [H, W]: the thermostat closing the loop on the frozen
    network, from states `x0` [W, n]. Row j of every input [H, W] is the
    step's start; the network's heat column in `drive` [H, W, 1 + m] is
    replaced by the thermostat's. A latched stage starts cold."""
    from neurotemplate.modeling.network import as_tensors, discrete_model

    a, b, _, _ = discrete_model(network, as_tensors(values), dt_h)
    p = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in fit.values.items()}
    x, stages = x0, {}
    zones, heats = [], []
    for j in range(drive.shape[0]):
        zone = x[:, 0]
        heat = torch.zeros_like(zone)
        for side, error in errors({s: sp[j] for s, sp in setpoints.items()}, zone).items():
            q, stages[side] = side_heat(p, side, fit.form, error, outdoor[j], fit.reference,
                                        smooth=False, stage=stages.get(side))
            heat = heat + q
        step = drive[j].clone()
        step[:, heat_column] = heat
        zones.append(zone)
        heats.append(heat)
        x = x @ a.T + step @ b.T
    return torch.stack(zones), torch.stack(heats)


def regimes(fit: ThermostatFit, heat: torch.Tensor, outdoor: torch.Tensor) -> torch.Tensor:
    """Each measured step's regime, as an index into `REGIMES`: satisfied
    under a tenth of the active side's capacity, saturated over nine tenths."""
    p = {k: torch.tensor(v, dtype=DTYPE) for k, v in fit.values.items()}
    side = torch.where(heat < 0, -1, 1)
    capacity = torch.zeros_like(heat)
    for name, sign in SIDES.items():
        if name in fit.sides:
            full = _capacity(p, name, fit.form, outdoor, fit.reference)
            if fit.form.stage2:
                full = full + p[f"{name}.capacity2"]
            capacity = torch.where(side == sign, full.expand_as(heat), capacity)
    share = heat.abs() / capacity.clamp(min=1e-9)
    return (share >= 0.1).long() + (share >= 0.9).long()
