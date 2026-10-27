"""
2R2C — a grey-box thermal zone: two resistances, two capacitances.

The building the black-box families cannot see
==============================================

`mlp`, `nssm`, `lssm` and `gru` all predict the next zone temperature from a
window of the last few observations. That is enough to be accurate
one step ahead, and it is why they can still be wrong about the thing that
matters for load shifting: **how long a temperature difference survives.**
A model scored over a four-hour rollout can re-derive the zone from the last
few setpoints and never learn to integrate; coolth stored at midday is then
gone by the evening peak, and pre-cooling cannot pay inside the model no
matter what the real building does.

2R2C fixes the storage by building it in. The zone air is one capacitance,
the building's thermal mass — walls, slab, furniture — is a second, hidden
one. Heat moves zone <-> ambient through one resistance and zone <-> mass
through the other:

        ambient ──R_o──┬── zone air (C_z) ──R_m── mass (C_m)
                       │
                    heat input

That topology has two time constants: a fast one (the air responding to the
HVAC, well under an hour) and a slow one (the mass charging and discharging,
many hours). The slow one is the storage, and no amount of window is a
substitute for it, because the mass temperature is *not in the data*.

What is grey and what is black
==============================

The state update is fixed physics::

    z_{k+1} = z_k + a_m (m_k - z_k) + a_o (T_oa,k - z_k)
                  + Σ_linear      k_j · x_j
                  + Σ_saturating  sign_j · cap_j · tanh(x_j / w_j)
                  + g( undeclared inputs )
    m_{k+1} = m_k + a_c (z_k - m_k)

with `z` the observed zone temperature, `m` the hidden mass temperature, and
three positive per-step exchange rates `a_o`, `a_m`, `a_c`. Which term an
input column feeds is declared per column in `block_mapping` (see
`blocks`); every declared column feeds exactly one branch, and the branches
add. A block with no declarations is the classical form with `g` alone —
one MLP over the current inputs, where the plug loads, internal gains,
solar gain and anything else unmetered all go — and conduction to a *zero*
ambient, `-a_o z_k`, which is only correct because `g` is affine in D and
D carries the outdoor temperature. The library learns the forcing and
asserts the storage.

The declared branches are the grey-box form (Bacher & Madsen 2011): every
known term explicit and linear in its input, a residual `g` for what is
genuinely unknown.

  ambient      one D column, the outdoor temperature. Conduction becomes
               `a_o (T_oa - z)` with `T_oa` read from the column rather than
               left for `g` to re-form, and the column leaves `g`'s input:
               anything else `g` could learn from it would be a
               weather-correlated bias, not conduction. A `tau_ambient` box
               wide enough to hold the truth matters most here: with the
               column declared, a wrong `a_o` cannot be absorbed by `g` any
               more — whatever free term is left would learn a constant heat
               gain of several °C/h to cancel it, which holds on the
               training distribution and nowhere else.
  linear       a U or D column entering as one sign-fixed coefficient times
               its physical value: measured power removes heat (`sign = -1`),
               solar irradiance adds it (`+1`). No bias — a constant heat gain
               belongs in `g`'s intercept, and a block that declares every
               column has no `g` and no free constant: unmetered gains then
               have to enter through a declared column (occupancy, say) or
               go unmodelled, because a free constant cannot be told apart
               from the ambient rate on data that never sits at zero
               outdoor gap.
  saturating   what a `demand`-declared setpoint becomes here. The
               pre-stack has already replaced the column with the
               thermostat's clamped error (see `prestack.Demand`); this branch
               turns it into a staged coil's response, `cap · tanh(x / w)`:
               `cap` is the declared `capacity`, °C per hour at full stage,
               fitted inside its box, and `w` the °C of demand over which it
               ramps from off to full. `tanh` is bounded in
               value and in slope, and its derivative never reaches zero —
               a hard clamp past the knee would be the dead-zone problem
               again.

Every non-MLP branch works in physical units and scales its result by the
zone's `stretch`, never shifting it, so each holds the norm stats it needs
as persistent buffers.

Two properties fall out of that split, and they are the reason to reach for
this family over `lssm`:

**Unconditional stability.** `g` reads inputs only — never `z`, never `m` —
so the state matrix is fixed by three positive rates whatever the MLP does.
The rates come from the two modes and the ambient time constant (see
`RCBlock.rates`), each held inside a declared box with `tau_fast` at least
`MIN_TAU_STEPS` steps, so every rate is below `1/MIN_TAU_STEPS = 0.5` and
the update's eigenvalues, `1 - 1/tau_fast` and `1 - 1/tau_slow`, sit inside
the unit circle. A 2R2C rollout cannot diverge, at any horizon, under any
input — including the adversarial inputs a DPC policy will find.

The declared branches keep that guarantee. `ambient` and `linear` are
linear in *inputs*, not state, and do not touch the state matrix. The
`saturating` branch does read `z`, through the demand feature, and its
local slope `s(z)` is nonnegative, zero inside the deadband and at most
`cap / w`; the zone's own coefficient becomes `1 - a_m - a_o - s(z)`. The
parameterization holds `cap / w` below the headroom `1 - a_m - a_o`, so
that diagonal stays nonnegative, the state matrix stays entrywise
nonnegative and entrywise below the undeclared block's — whose spectral
radius `mode_taus` already places inside the unit circle — and a product of
such matrices along a rollout is bounded by powers of it. Extra damping
that only acts when the zone is warm can only lower the spectral radius.

**Readable time constants.** The eigenvalues are reported in hours on the
reportcard. `tau_slow` is the number pre-cooling lives on, and you can check
it against the building instead of taking the surrogate's word for it.

The storage is a prior, held in a box
=====================================

Each block declares three time constants in hours, in its `block_mapping`
entry's `params` — a scalar pins one, `[min, max]` lets training move it
inside that box:

  tau_fast     the zone air settling after a push, against structure and
               outdoors at once — the quick swing right after an HVAC step
  tau_ambient  the air leaking to the outdoors while the structure sits at
               the air's temperature — the initial slope of an HVAC-off drift
  tau_slow     the building, air and structure together, relaxing to the
               outdoors with nothing running — the drift after its first
               hour, what `nt data decay` fits

These three rather than the component rates because they are what one
measures, so they are where priors are actually held. All three rates are
positive exactly when `tau_fast < tau_ambient < tau_slow`, so the boxes must
be disjoint in that order; every point inside them is then a valid building,
and training cannot leave the feasible set. There is no default: a prior
about a building is the user's to state.

A box, not a free fit, because the slow mode is not identifiable from a
training rollout. The mass temperature is unobserved, so every rollout seeds
it from the zone's history window — an hour at most, against a mass that lags by
many. The seed is therefore wrong, and the optimizer has two ways to reduce
the damage: fix the forcing, or shorten the slow mode so the wrong seed
matters less. It picks the second. Measured on synthetic data from a known
31.6 h building, seeded with the true mass, training moves tau_slow the
right way (13.4 h -> 16.5 h); seeded the way a real rollout must seed it,
the same training drives it to 8.7 h. A box does not cure that — it stops
it, and the reportcard says when a time constant is pressed against a bound:
the data (or the seed) disagreeing with the prior is the finding.

What it gives up
================

`g` cannot see `z`, so on its own it cannot represent a thermostat deadband
— "the compressor is off whenever the setpoint is above the zone" is a
function of the *difference*, and an undeclared block renders a setpoint as
a monotone response with no cut-off. The `demand` declaration is the
answer: the difference is computed outside `g`, clamped, and handed to the
bounded `saturating` branch, which is the one shape of state-dependence the
stability argument admits. What the block still cannot do is let a
*learned* map see state — a `2r2c` block refuses `reads` for that reason.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromancer.modules import blocks

from neurotemplate.modeling import windows
from neurotemplate.modeling.bounded import BoundedParameter, bounded_parameters, parse_box
from neurotemplate.modeling.prestack import demand_scale

#: Fastest time constant the parameterization allows, in timesteps. Its
#: reciprocal bounds every rate, and that is what keeps the discrete update
#: inside the unit circle — so this is the stability guarantee's one knob,
#: not a tuning parameter. A two-step time constant is already a zone with
#: essentially no thermal inertia.
MIN_TAU_STEPS = 2.0

#: A 2r2c block's `params`: its three time constants in hours, in the order
#: they must be — `tau_fast < tau_ambient < tau_slow`.
TAU_KEYS = ("tau_fast_h", "tau_ambient_h", "tau_slow_h")

#: Raw parameter behind a `linear` branch's coefficient at construction:
#: `softplus(-4) ≈ 0.018`, so a fresh block starts as near-pure free decay
#: for the same reason `g` starts at zero — an O(1) forcing would swamp the
#: RC dynamics for the first epochs. Far enough from softplus's flat side
#: that Adam moves it on the first step.
_NEAR_ZERO_THETA = -4.0


def time_constant_boxes(params: dict, steps_per_hour: float,
                        block: str) -> dict[str, tuple[float, float]]:
    """A 2r2c block's `params`, validated: `{tau key: (min_h, max_h)}`.

    Refuses, naming every problem at once: a missing or unknown key; boxes
    that overlap or are out of order (the rates are positive exactly when
    `tau_fast < tau_ambient < tau_slow`); and a `tau_fast` floor under
    `MIN_TAU_STEPS` steps of this data, which the stability guarantee needs.
    """
    problems = []
    missing = [k for k in TAU_KEYS if k not in params]
    unknown = sorted(set(params) - set(TAU_KEYS))
    if missing:
        problems.append(f"missing {missing}; a 2r2c block declares all three time "
                        "constants, in hours, as a scalar or [min, max]")
    if unknown:
        problems.append(f"unknown key(s) {unknown}; a 2r2c block reads {list(TAU_KEYS)}")
    boxes = {}
    for key in TAU_KEYS:
        if key in params:
            try:
                boxes[key] = parse_box(key, params[key])
            except ValueError as error:
                problems.append(str(error))
    if len(boxes) == len(TAU_KEYS):
        for lower, upper in zip(TAU_KEYS, TAU_KEYS[1:]):
            if not boxes[lower][1] < boxes[upper][0]:
                problems.append(
                    f"{lower} {list(boxes[lower])} must lie wholly below {upper} "
                    f"{list(boxes[upper])}: the rates are positive only when "
                    "tau_fast < tau_ambient < tau_slow")
        floor_h = MIN_TAU_STEPS / steps_per_hour
        if boxes["tau_fast_h"][0] < floor_h:
            problems.append(
                f"tau_fast_h starts at {boxes['tau_fast_h'][0]:g} h, under the "
                f"{MIN_TAU_STEPS:g}-step floor of this data's timestep "
                f"({floor_h:g} h) that keeps the rollout stable")
    if problems:
        raise ValueError(f"2r2c block {block!r} has invalid `params`:\n  - "
                         + "\n  - ".join(problems))
    return boxes


def demand_boxes(inputs: dict[str, dict], block: str) -> dict[str, dict]:
    """Each saturating `demand` entry's boxes, validated:
    `{column: {"capacity": (lo, hi)[, "on_at": (lo, hi), "off_at": (lo, hi)]}}`.

    `capacity` (°C per hour at full stage) is required: like the time
    constants, it is a prior about the building with no library default. A
    `hysteresis` entry is `{"on_at": box, "off_at": box}` in °C of error.
    Every problem is named at once.
    """
    boxes, problems = {}, []
    for column, entry in inputs.items():
        if entry["kind"] != "demand" or entry.get("branch", "saturating") != "saturating":
            continue
        box = {}
        if "capacity" not in entry:
            problems.append(f"{column!r} declares no `capacity`: °C per hour the zone "
                            "moves at full stage, a scalar or [min, max] (`nt data latch` "
                            "estimates it)")
        else:
            try:
                box["capacity"] = parse_box(f"{column!r} capacity", entry["capacity"])
            except ValueError as error:
                problems.append(str(error))
        latch = entry.get("hysteresis")
        if latch is not None:
            if not isinstance(latch, dict) or set(latch) != {"on_at", "off_at"}:
                problems.append(f"{column!r} hysteresis must be {{on_at: box, off_at: box}} "
                                f"in °C of error, got {latch!r}")
            else:
                for key in ("on_at", "off_at"):
                    try:
                        box[key] = parse_box(f"{column!r} hysteresis {key}", latch[key],
                                             log=False)
                    except ValueError as error:
                        problems.append(str(error))
        boxes[column] = box
    if problems:
        raise ValueError(f"2r2c block {block!r} has an invalid `demand` declaration:\n  - "
                         + "\n  - ".join(problems))
    return boxes


def mode_taus(a_ambient, a_zone_mass, a_mass_zone):
    """The two time constants of the RC network, in timesteps.

    These are what the component rates *mean*: the state matrix is
    ``I - L`` with::

        L = [[a_m + a_o, -a_m],
             [-a_c,       a_c]]

    `L` has real positive eigenvalues (trace and determinant are both
    positive, and the discriminant below is a sum of squares), so the
    update's eigenvalues `1 - λ` are real and, as long as the rates sum
    below 2, inside the unit circle. `tau = 1 / λ`.

    Args:
        a_ambient / a_zone_mass / a_mass_zone: per-step rates, any shape,
            broadcast together elementwise.

    Returns:
        (tau_fast, tau_slow), the same shape as the inputs.
    """
    trace = a_zone_mass + a_ambient + a_mass_zone
    # Algebraically `sqrt(trace^2 - 4 a_o a_c)`, rearranged into a sum of
    # squares so round-off cannot drive it negative near a repeated root.
    gap = ((a_zone_mass + a_ambient - a_mass_zone) ** 2
           + 4 * a_zone_mass * a_mass_zone).sqrt()
    return 2.0 / (trace + gap), 2.0 / (trace - gap)


# ── Declared forcing branches ──────────────────────────────────
#
# Each takes its input column normalised, works in physical units, and
# returns an increment in the zone's normalised units — `zone_shift` and
# `zone_stretch` are per channel, `[ny]`, so a term broadcasts across a
# multi-channel block. The stats are persistent buffers, so a reloaded
# checkpoint keeps its own numbers.


def _buffer(module: nn.Module, name: str, value) -> None:
    module.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))


class AmbientTerm(nn.Module):
    """`a_o (T_oa - z)`: conduction to the declared outdoor temperature.

    The rate `a_o` belongs to the block (it is one of its time constants) and
    is handed in each step, the way `SaturatingTerm` is handed its headroom.
    """

    def __init__(self, column: dict, zone_shift: list[float], zone_stretch: list[float]):
        super().__init__()
        _buffer(self, "shift", column["shift"])
        _buffer(self, "stretch", column["stretch"])
        _buffer(self, "zone_shift", zone_shift)
        _buffer(self, "zone_stretch", zone_stretch)

    def forward(self, zone: torch.Tensor, outdoor: torch.Tensor,
                rate: torch.Tensor) -> torch.Tensor:
        zone_phys = zone * self.zone_stretch + self.zone_shift
        outdoor_phys = outdoor * self.stretch + self.shift
        return rate * (outdoor_phys - zone_phys) / self.zone_stretch


class LinearTerm(nn.Module):
    """`k · x`, with `k = sign · softplus(θ) / stretch_x` in °C per step per
    physical unit of the column. Dividing by the column's stretch makes the
    near-zero start the same size whether power arrives in W or kW."""

    def __init__(self, sign: int, column: dict, zone_stretch: list[float]):
        super().__init__()
        self.sign = int(sign)
        self.theta = nn.Parameter(torch.tensor(_NEAR_ZERO_THETA))
        _buffer(self, "shift", column["shift"])
        _buffer(self, "stretch", column["stretch"])
        _buffer(self, "zone_stretch", zone_stretch)

    def coefficient(self) -> torch.Tensor:
        """`k`, in zone units per step per physical unit of the column."""
        return self.sign * F.softplus(self.theta) / self.stretch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_phys = x * self.stretch + self.shift
        return self.coefficient() * x_phys / self.zone_stretch


class SaturatingTerm(nn.Module):
    """`sign · cap · tanh(x / w)`: a staged coil's response to demand.

    `cap` is the declared `capacity` (°C per hour at full stage), fitted
    inside its box and used per step; the slope at the origin is `headroom ·
    sigmoid(θ_s)`, so `w = cap / slope` and the bound `cap / w ≤ headroom`
    holds by construction — `headroom` is the block's `1 - a_m - a_o`, handed
    in each step so the bound follows the rates if they train. `θ_s = 0`
    starts at half the budget.

    The input is the pre-stack's demand feature, scaled to about [0, 1] by
    `demand_scale`; this branch holds the same scale and undoes it, so `w`
    reads in °C of demand.
    """

    def __init__(self, sign: int, control: dict, state: dict, column: str,
                 boxes: dict, steps_per_hour: float):
        super().__init__()
        self.sign = int(sign)
        self.steps_per_hour = float(steps_per_hour)
        self.capacity = BoundedParameter(f"{column}: capacity_C_per_h",
                                         *boxes["capacity"], (column,))
        self.theta_slope = nn.Parameter(torch.tensor(0.0))
        _buffer(self, "scale", demand_scale(sign, control, state))
        _buffer(self, "zone_stretch", state["stretch"])

    def cap(self) -> torch.Tensor:
        """°C per step at full stage."""
        return self.capacity.value() / self.steps_per_hour

    def slope_share(self) -> torch.Tensor:
        """Fraction of the stability headroom the slope at the origin uses."""
        return torch.sigmoid(self.theta_slope)

    def width(self, headroom: torch.Tensor) -> torch.Tensor:
        """`w`: °C of demand over which the response ramps from off to full."""
        return self.cap() / (headroom * self.slope_share())

    def forward(self, x: torch.Tensor, headroom: torch.Tensor) -> torch.Tensor:
        demand_c = x * self.scale
        response = self.sign * self.cap() * torch.tanh(demand_c / self.width(headroom))
        return response / self.zone_stretch


class LatchedTerm(SaturatingTerm):
    """A thermostat stage with hysteresis: `sign · cap · s`, `s` carried between steps.

    The stage switches on once the error is `on_at` past the setpoint and, once
    on, stays on until the error is `off_at` back on the satisfied side:

        s = on + (1 - on) · s_prev · hold,
        on = sigmoid((e - on_at) / τ),   hold = sigmoid((e + off_at) / τ)

    with `e` the signed error in °C (positive while the stage is called for;
    the pre-stack does not clamp a hysteresis column) and both thresholds
    fitted inside their declared linear boxes. `s` stays in [0, 1], and
    `ds/de ≤ 1 / (2τ)`, so `τ = cap / (2 · headroom · share)` keeps the slope
    bound `SaturatingTerm` keeps, and the rollout stays bounded: the stage is
    a bounded state driving a stable zone. A cold call has no carried stage
    and starts from `s_prev = on`, the stage a memoryless thermostat would be in.
    """

    def __init__(self, sign: int, control: dict, state: dict, column: str,
                 boxes: dict, steps_per_hour: float):
        super().__init__(sign, control, state, column, boxes, steps_per_hour)
        self.on_at = BoundedParameter(f"{column}: on_at_C", *boxes["on_at"], (column,),
                                      log=False)
        self.off_at = BoundedParameter(f"{column}: off_at_C", *boxes["off_at"], (column,),
                                       log=False)

    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`(on_at, off_at)` in °C of error."""
        return self.on_at.value(), self.off_at.value()

    def width(self, headroom: torch.Tensor) -> torch.Tensor:
        """`τ`: °C of error over which each switch ramps."""
        return super().width(headroom) / 2

    def forward(self, x: torch.Tensor, headroom: torch.Tensor,
                previous: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """`(increment, stage)`: the branch's increment and the stage it leaves."""
        error = x * self.scale
        tau = self.width(headroom)
        on_at, off_at = self.thresholds()
        on = torch.sigmoid((error - on_at) / tau)
        hold = torch.sigmoid((error + off_at) / tau)
        previous = on if previous is None else previous
        stage = on + (1 - on) * previous * hold
        response = self.sign * self.cap() * stage
        return response / self.zone_stretch, stage


def physical_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Every grey-box physical scalar under `module`: the bounded time
    constants and the declared branches' coefficients.

    A grey-box fit steps these at their own learning rate
    (`cfg.rc_branch_lr`): Adam moves a raw parameter by about `lr` per
    update, and a coil capacity or a time constant has units of travel to
    cover where a network weight has thousandths. A pinned time constant is
    included too; its gradient is zero, so Adam leaves it where it is.
    """
    terms = (LinearTerm, SaturatingTerm)
    params = [p for m in module.modules() if isinstance(m, terms) for p in m.parameters()]
    params += [b.theta for b in bounded_parameters(module)]
    return list({id(p): p for p in params}.values())


class RCBlock(nn.Module):
    """One 2R2C network per Y channel, sharing a learned heat-input map.

    Reads a key-major window (see `windows`) of `y_steps` rows of its own
    zones, which seed the mass, and the current step of U and D, which is
    all the heat input reads. Returns the **increment** `z_{k+1} - z_k`.
    `SSMGrad` adds `z_k` back, which is why `2r2c` requires `residual=True`.

    Hidden state
    ------------
    The mass temperature is not in the data, so this block carries it, the
    way `GRUBlock` carries a hidden state:

      - Cold (no state): seed the mass at the window mean of the observed
        zone temperature and integrate it across the window, so it enters
        the first prediction already charged by the history.
      - Warm: advance it one step from the newest zone temperature.

    Call `reset()` at a sequence boundary; the next call re-seeds from its
    own window. `make_reset_init` wires that into the System automatically.

    Under vmap (jacrev, for the sampled gradient penalty) the input is 1-D
    and the block runs cold without storing state, matching `GRUBlock`.

    Args:
        ny / nu / nd: channel counts for this block's own slice of Y, U, D.
        y_steps: rows of the zone window: the zones' lag, plus the current
            step.
        steps_per_hour: timesteps per hour. Only ever used to convert the
            time constants from hours to steps.
        hidden_size / num_layers: size of the learned heat-input MLP.
        taus: the storage prior, `{tau key: (min_h, max_h)}` for every key in
            `TAU_KEYS`, as `time_constant_boxes` validates it. Every channel
            of the block fits its own values inside the same boxes.
        branch_boxes: each saturating demand column's boxes, as
            `demand_boxes` validates them.
        channel_names: Y column names, for labelling `describe`.
        u_names / d_names: this block's U and D column names, in the order
            its window carries them; what `inputs` and `ambient` refer to.
        inputs: per-column declarations, `{column: {"kind", "sign", ...}}`,
            as validated by `blocks.build_blocks`. A `demand` column feeds a
            `SaturatingTerm` — or the MLP, with `"branch": "mlp"`, at the
            cost of the stability bound — a `linear` one a `LinearTerm`;
            the rest feed the MLP.
        ambient: the D column read by the explicit conduction term, or None
            for conduction to a zero ambient.
        norm_stats: per-column norm stats, needed for any declared branch.
    """

    def __init__(self, ny: int, nu: int, nd: int, y_steps: int,
                 steps_per_hour: float, hidden_size: int, num_layers: int,
                 taus: dict[str, tuple[float, float]],
                 channel_names: tuple[str, ...] = (),
                 u_names: tuple[str, ...] = (),
                 d_names: tuple[str, ...] = (),
                 inputs: Optional[dict[str, dict]] = None,
                 ambient: Optional[str] = None,
                 norm_stats: Optional[dict[str, dict]] = None,
                 branch_boxes: Optional[dict[str, dict]] = None):
        super().__init__()
        self.widths = (ny, nu, nd)
        self.ny = ny
        self.channel_names = channel_names or tuple(f"y{i}" for i in range(ny))
        steps = (y_steps, 1, 1)
        self.in_features = y_steps * ny + nu + nd
        self.out_features = ny

        self.steps_per_hour = steps_per_hour

        y_block = windows.key_blocks(self.widths, steps)[0]
        self._y_steps = windows.per_step(y_block, ny, y_steps)
        _, self._u_last, self._d_last = windows.last_step(self.widths, steps)

        self.tau_fast, self.tau_ambient, self.tau_slow = (
            BoundedParameter(key[:-2], *taus[key], self.channel_names)
            for key in TAU_KEYS)

        # Each declared column is a position into the drive vector
        # `cat([u_last, d_last])`: U columns first, then D.
        inputs = inputs or {}
        u_names = u_names or tuple(f"u{i}" for i in range(nu))
        d_names = d_names or tuple(f"d{i}" for i in range(nd))
        drive_names = list(u_names) + list(d_names)
        if (inputs or ambient) and norm_stats is None:
            raise ValueError("RCBlock needs norm_stats for a declared branch")
        if any(e["kind"] == "demand" for e in inputs.values()) and ny != 1:
            raise ValueError(f"a block with a `demand` column has one Y channel, not {ny}")
        zone_shift = [norm_stats[c]["shift"] for c in self.channel_names] if norm_stats else []
        zone_stretch = [norm_stats[c]["stretch"] for c in self.channel_names] if norm_stats else []

        self.ambient_name = ambient
        self._ambient_pos = drive_names.index(ambient) if ambient else None
        self.ambient = (AmbientTerm(norm_stats[ambient], zone_shift, zone_stretch)
                        if ambient else None)

        self.linear_names = [c for c, e in inputs.items() if e["kind"] == "linear"]
        self._linear_pos = [drive_names.index(c) for c in self.linear_names]
        self.linear = nn.ModuleList(
            LinearTerm(inputs[c]["sign"], norm_stats[c], zone_stretch)
            for c in self.linear_names)

        demands = [c for c, e in inputs.items() if e["kind"] == "demand"]
        self.saturating_names = [
            c for c in demands if inputs[c].get("branch", "saturating") == "saturating"]
        self._saturating_pos = [drive_names.index(c) for c in self.saturating_names]
        self.saturating = nn.ModuleList(
            (LatchedTerm if "on_at" in branch_boxes[c] else SaturatingTerm)(
                inputs[c]["sign"], norm_stats[c], norm_stats[self.channel_names[0]],
                c, branch_boxes[c], steps_per_hour)
            for c in self.saturating_names)
        self.n_latched = sum(isinstance(t, LatchedTerm) for t in self.saturating)
        # Demand features routed into `g` by declaration: a ReLU MLP has no
        # slope bound, so the stability guarantee is off for this block.
        self.demand_in_mlp = [c for c in demands if c not in self.saturating_names]

        # Heat input `g`, from every column without a structured branch —
        # never from the state, except through a `demand_in_mlp` column.
        # That restriction is the whole stability argument; see the module
        # docstring before relaxing it further.
        structured = (set(self.linear_names) | set(self.saturating_names)
                      | ({ambient} if ambient else set()))
        self.mlp_names = [c for c in drive_names if c not in structured]
        self.register_buffer(
            "_mlp_pos",
            torch.tensor([drive_names.index(c) for c in self.mlp_names], dtype=torch.long),
            persistent=False,
        )
        if self.mlp_names:
            self.gain = blocks.MLP(
                insize=len(self.mlp_names), outsize=ny, bias=True,
                linear_map=nn.Linear, nonlin=nn.ReLU,
                hsizes=num_layers * [hidden_size],
            )
            # Start with no forcing at all. A freshly initialized MLP emits
            # O(1), and one step of a normalized temperature is O(0.01), so
            # an untouched gain would swamp the RC dynamics for the first
            # epochs and the optimizer would spend them undoing its own
            # initialization.
            final = self.gain.linear[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            # Every column is declared: no residual term at all. A free
            # constant here would be a heat gain that is unidentifiable
            # against the ambient rate — the two trade off along a ridge
            # that only the equilibrium scenario, at zero outdoor gap,
            # can see.
            self.gain = None

        self._mass: torch.Tensor | None = None
        # One stage per latched branch, [batch, n_latched]; carried like the mass.
        self._stage: torch.Tensor | None = None

    # ── parameters ──────────────────────────────────────────────

    def rates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(a_ambient, a_zone_mass, a_mass_zone)` per step, from the time
        constants.

        With `λ = 1/tau` in steps: `a_o = λ_ambient`, `a_c = λ_fast·λ_slow/a_o`
        and `a_m = λ_fast + λ_slow − a_o − a_c` — the rates whose state matrix
        has exactly the modes `tau_fast` and `tau_slow` (`mode_taus` inverts
        this). All three are positive because the boxes are disjoint in the
        order `tau_fast < tau_ambient < tau_slow`, and each is below
        `λ_fast <= 1/MIN_TAU_STEPS`.
        """
        fast, ambient, slow = (1 / (tau.value() * self.steps_per_hour)
                               for tau in (self.tau_fast, self.tau_ambient, self.tau_slow))
        a_mass_zone = fast * slow / ambient
        return ambient, fast + slow - ambient - a_mass_zone, a_mass_zone

    # The stateful-block protocol (see `neurotemplate.export.state`): the
    # hidden state as one [batch, width] tensor, so an exporter can lift it
    # into explicit graph I/O without knowing what this class calls it.

    def rollout_state(self) -> torch.Tensor | None:
        """The mass temperature, then any latched stages: [batch, ny + n_latched],
        or None if cold."""
        if self._mass is None or not self.n_latched:
            return self._mass
        return torch.cat([self._mass, self._stage], dim=-1)

    def set_rollout_state(self, state: torch.Tensor | None) -> None:
        """Restore a mass temperature and stages, or None to force a cold start."""
        if state is None:
            self._mass, self._stage = None, None
            return
        self._mass = state[..., :self.ny]
        self._stage = state[..., self.ny:] if self.n_latched else None

    # ── forward ─────────────────────────────────────────────────

    def _seed_mass(self, x: torch.Tensor,
                   a_mass_zone: torch.Tensor) -> torch.Tensor:
        """Charge a fresh mass state from the observed history window.

        The mass is unobserved, so its value at the start of a rollout is a
        modeling choice. Starting at the window's mean zone temperature and
        then integrating the mass equation across the window is the choice
        that needs no extra data and is right in the two limits that matter:
        a zone sitting at steady state hands the mass that steady value, and
        a zone that has been driven hard hands it something lagging behind.
        """
        steps = [x[..., s] for s in self._y_steps]
        mass = torch.stack(steps, dim=0).mean(dim=0)
        # The final step is the one the first prediction consumes, so it
        # drives the update rather than being folded in ahead of it.
        for zone in steps[:-1]:
            mass = mass + a_mass_zone * (zone - mass)
        return mass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Under vmap the batch dimension is gone and state cannot be stored,
        # so every call is a cold one. Same convention as GRUBlock.
        if x.ndim == 1:
            return self.step(x, None, None)[0]
        increment, self._mass, self._stage = self.step(x, self._mass, self._stage)
        return increment

    def layout(self) -> tuple:
        """Everything `step` reads that is not a tensor: two blocks with the
        same layout and parameter shapes compute the same function of their
        own parameters, so `BlockStack` can run them as one batched call."""
        return (self.in_features, self.ny, self.steps_per_hour, self.n_latched,
                tuple((s.start, s.stop) for s in self._y_steps),
                (self._u_last.start, self._u_last.stop),
                (self._d_last.start, self._d_last.stop),
                self._ambient_pos, tuple(self._linear_pos),
                tuple(self._saturating_pos),
                tuple(t.sign for t in self.linear),
                tuple((type(t), t.sign) for t in self.saturating),
                self.gain is None)

    def step(self, x: torch.Tensor, mass: Optional[torch.Tensor],
             stage: Optional[torch.Tensor]
             ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """`(increment, mass, stage)` after one step, from the state carried
        in; `mass` None is a cold call that seeds it from the window. Pure:
        `forward` is what stores the state."""
        a_ambient, a_zone_mass, a_mass_zone = self.rates()

        zone = x[..., self._y_steps[-1]]
        drive = torch.cat([x[..., self._u_last], x[..., self._d_last]], dim=-1)

        if mass is None:
            mass = self._seed_mass(x, a_mass_zone)
            stage = None

        if self.ambient is None:
            conduction = -a_ambient * zone
        else:
            conduction = self.ambient(zone, self._column(drive, self._ambient_pos),
                                      a_ambient)
        heat, stage = self._heat_input(drive, headroom=1 - a_zone_mass - a_ambient,
                                       stage=stage)
        increment = a_zone_mass * (mass - zone) + conduction + heat
        return increment, mass + a_mass_zone * (zone - mass), stage

    @staticmethod
    def _column(drive: torch.Tensor, pos: int) -> torch.Tensor:
        """One column of the drive vector, keeping its trailing axis."""
        return drive[..., pos:pos + 1]

    def _heat_input(self, drive: torch.Tensor, headroom: torch.Tensor,
                    stage: Optional[torch.Tensor] = None
                    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """`(heat, stage)`: the declared branches plus `g`, summed, and the
        stages the latched branches leave (None without one). `headroom` is
        the zone's remaining self-coefficient, which bounds every saturating
        slope; `stage` is the stages carried in, None on a cold call."""
        if self.gain is None:
            total = drive.new_zeros(*drive.shape[:-1], self.ny)
        else:
            total = self.gain(drive.index_select(-1, self._mlp_pos))
        for pos, term in zip(self._linear_pos, self.linear):
            total = total + term(self._column(drive, pos))
        stages = []
        for pos, term in zip(self._saturating_pos, self.saturating):
            if isinstance(term, LatchedTerm):
                previous = None if stage is None else stage[..., len(stages):len(stages) + 1]
                heat, new_stage = term(self._column(drive, pos), headroom, previous)
                stages.append(new_stage)
            else:
                heat = term(self._column(drive, pos), headroom)
            total = total + heat
        return total, (torch.cat(stages, dim=-1) if stages else None)

    # ── description ─────────────────────────────────────────────

    def describe(self) -> dict:
        """Per-channel time constants in hours, and the forcing branches.

        `tau_slow` is the storage number: how long a temperature difference
        put into this zone survives. It is the one figure worth checking
        against the real building; the three time constants' boxes and where
        each fit landed in its box are reported with every bounded parameter
        (see `bounded`). `forcing` lists which column feeds which branch and
        what each learned: `k` in zone units per step per physical unit for
        `linear`; `w` (°C of demand) and `slope_share` (the fraction of the
        stability budget the slope uses) for `saturating`, whose capacity and
        thresholds are bounded parameters.
        `demand_in_mlp` names the demand columns routed into `g` by
        declaration, for which the stability bound does not hold.
        """
        with torch.no_grad():
            a_ambient, a_zone_mass, _ = self.rates()
            headroom = 1 - a_zone_mass - a_ambient
            fast, slow = self.tau_fast.value(), self.tau_slow.value()
            forcing = {
                "ambient": None if self.ambient is None else {"column": self.ambient_name},
                "linear": {
                    name: {"k": round(float(term.coefficient()), 6)}
                    for name, term in zip(self.linear_names, self.linear)
                },
                "saturating": {
                    name: {"w": round(float(term.width(headroom)), 4),
                           "slope_share": round(float(term.slope_share()), 3),
                           "latched": isinstance(term, LatchedTerm)}
                    for name, term in zip(self.saturating_names, self.saturating)
                },
                "mlp": list(self.mlp_names),
                # hsizes of the learned heat-input map `g` — the one hidden
                # mapping this otherwise-transparent block contains. `gain`
                # is a neuromancer MLP, which exposes per-layer widths only
                # via `linear` (a ModuleList including the output layer).
                "mlp_hsizes": (
                    [int(lin.out_features) for lin in self.gain.linear[:-1]]
                    if self.gain is not None else []
                ),
                "demand_in_mlp": list(self.demand_in_mlp),
            }
        return {
            "time_constants_h": {
                name: {"fast": round(float(f), 3), "slow": round(float(sl), 3)}
                for name, f, sl in zip(self.channel_names, fast, slow)
            },
            "forcing": forcing,
        }


# ── Running same-shaped blocks as one call ───────────────────────

def rc_groups(modules: list[nn.Module]) -> list[list[int]]:
    """Positions of the `RCBlock`s that can run as one batched call: two or
    more with the same `layout` and the same tensor shapes. A building's
    per-zone blocks declared from one template land in one group."""
    by_key: dict[tuple, list[int]] = {}
    for i, module in enumerate(modules):
        if isinstance(module, RCBlock):
            shapes = tuple((name, tuple(t.shape)) for name, t in _tensors(module).items())
            by_key.setdefault((module.layout(), shapes), []).append(i)
    return [group for group in by_key.values() if len(group) > 1]


def _tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter and buffer, persistent or not, by name."""
    return {**dict(module.named_parameters()), **dict(module.named_buffers())}


class _Step(nn.Module):
    """`block.step` as a module's forward, so `functional_call` can run it
    against another block's tensors."""

    def __init__(self, block: RCBlock):
        super().__init__()
        self.block = block

    def forward(self, x, mass, stage):
        return self.block.step(x, mass, stage)


class RCGroup:
    """One step of several same-layout `RCBlock`s as a single call.

    Their parameters and buffers are stacked on a new leading axis and
    `vmap` runs `RCBlock.step` once over it, so a building of N zones costs
    the ops of one zone rather than N times them. The blocks still own
    their parameters and their state: each block is handed its own mass
    back, so `describe`, the checkpoint, the reset hook and the exporter's
    state lifting see N ordinary blocks.

    The stack is built on a rollout's cold first call and reused for its
    warm steps, where the parameters cannot change, so gradients from every
    step flow back through one stack. A warm call also rebuilds it if the
    grad mode or any parameter's version (an optimizer step, a
    `load_state_dict`) differs from when it was built, so it can never read
    stale weights or a stack without a graph. `Parameter` objects survive
    `.to` and `load_state_dict`, so the list checked is fixed; buffers are
    constant within a rollout.

    A plain object, not a module: it holds the blocks without registering
    them a second time. It checks itself on construction (`_check`), so a
    `layout` that misses something `step` reads fails at build time rather
    than training every zone on block 0's value of it.
    """

    def __init__(self, blocks: list[RCBlock]):
        self.blocks = blocks
        self._step = _Step(blocks[0])
        self._names = list(_tensors(blocks[0]))
        self._params = [p for b in blocks for p in b.parameters()]
        self._key: Optional[tuple] = None
        self._stacked: dict[str, torch.Tensor] = {}
        self._check()

    def _check(self) -> None:
        """One cold step, grouped and block by block, on a random window."""
        width = self.blocks[0].in_features
        device = next(iter(_tensors(self.blocks[0]).values())).device
        x = torch.randn(2, len(self.blocks), width, device=device,
                        generator=torch.Generator(device).manual_seed(0))
        def cold():
            for block in self.blocks:
                block.set_rollout_state(None)

        with torch.no_grad():
            cold()
            grouped = self(x).reshape(2, len(self.blocks), -1)
            cold()
            alone = torch.stack([b(x[:, i]) for i, b in enumerate(self.blocks)], dim=1)
            cold()
        if not torch.allclose(grouped, alone, rtol=1e-4, atol=1e-6):
            raise RuntimeError(
                "RCGroup: these blocks share a `layout()` but compute different "
                "functions, so `RCBlock.step` reads something `layout()` leaves "
                "out. Add it there.")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """`x` [batch, G, width] -> increments [batch, G * ny]."""
        mass = self._stacked_state("_mass")
        latched = self.blocks[0].n_latched > 0
        stage = self._stacked_state("_stage") if latched else None
        stacked = self._tensors_stacked(cold=mass is None)
        state_dims = None if mass is None else 0

        def call(tensors, x, mass, stage):
            out = torch.func.functional_call(self._step, tensors, (x, mass, stage))
            return out if latched else out[:2]

        out = torch.func.vmap(
            call, in_dims=(0, 1, state_dims, state_dims if latched else None),
            out_dims=(1, 0, 0) if latched else (1, 0))(stacked, x, mass, stage)
        for block, block_mass in zip(self.blocks, out[1].unbind(0)):
            block._mass = block_mass
        if latched:
            for block, block_stage in zip(self.blocks, out[2].unbind(0)):
                block._stage = block_stage
        increment = out[0]
        return increment.reshape(*increment.shape[:-2], -1)

    def _tensors_stacked(self, cold: bool) -> dict[str, torch.Tensor]:
        key = (torch.is_grad_enabled(), tuple(p._version for p in self._params))
        if cold or key != self._key:
            tensors = [_tensors(b) for b in self.blocks]
            self._stacked = {f"block.{n}": torch.stack([ts[n] for ts in tensors])
                             for n in self._names}
            self._key = key
        return self._stacked

    def _stacked_state(self, name: str) -> Optional[torch.Tensor]:
        """The blocks' carried `name`, stacked, or None when all are cold.
        They are reset together, so a mix is a broken reset."""
        states = [getattr(b, name) for b in self.blocks]
        if all(s is None for s in states):
            return None
        if any(s is None for s in states):
            raise RuntimeError(f"RCGroup: some blocks carry {name} and some are cold")
        return torch.stack(states)
