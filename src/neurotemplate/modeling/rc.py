"""
2R2C — a grey-box thermal zone: two resistances, two capacitances.

The building the black-box families cannot see
==============================================

`mlp`, `nssm`, `lssm` and `gru` all predict the next zone temperature from a
window of the last `in_steps` observations. That is enough to be accurate
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
               weather-correlated bias, not conduction.
  linear       a U or D column entering as one sign-fixed coefficient times
               its physical value: measured power removes heat (`sign = -1`),
               solar irradiance adds it (`+1`). No bias — a constant heat gain
               belongs in `g`'s intercept.
  saturating   what a `demand`-declared setpoint becomes here. The
               pre-stack has already replaced the column with the
               thermostat's clamped error (see `heads.Demand`); this branch
               turns it into a staged coil's response, `cap · tanh(x / w)`:
               `cap` is °C per step at full stage and `w` the °C of demand
               over which it ramps from off to full. `tanh` is bounded in
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
Rates are parameterized as `1 / (MIN_TAU_STEPS + softplus(θ))`, so each is
below `1/MIN_TAU_STEPS = 0.5` and their sum below 2, which is exactly the
condition for the discrete update's eigenvalues to sit inside the unit
circle (see `mode_taus`). A 2R2C rollout cannot diverge, at any horizon,
under any input — including the adversarial inputs a DPC policy will find.

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

The storage is a prior, not a fit
================================

The three rates default to **frozen** (`cfg.rc_train_taus`). That is not
timidity, it is what the data supports. The mass temperature is unobserved,
so every rollout has to seed it from the `in_steps` window — an hour at
most, against a mass that lags by many. The seed is therefore wrong, and
the optimizer has two ways to reduce the damage: fix the forcing, or
shorten the slow mode so the wrong seed matters less. It picks the second.
Measured on synthetic data from a known 31.6 h building, seeded with the
true mass, training moves tau_slow the right way (13.4 h -> 16.5 h); seeded
the way a real rollout must seed it, the same training drives it to 8.7 h.
Letting the window's own history choose the seed does not help, because an
hour of zone temperature carries no information about a ten-hour state.

So `tau_slow` is something you assert and the reportcard reads back, not
something the fit discovers. Set it from what you know — a free-decay
episode gives it directly — and let training learn the heat input around
it. Unfreeze only when the rollout is long relative to the slow mode and
the data actually excites it.

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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromancer.modules import blocks

from neurotemplate.modeling import windows
from neurotemplate.modeling.heads import demand_scale

#: Fastest exchange the parameterization allows, in timesteps. Its reciprocal
#: bounds every rate, and rates summing below 2 is what keeps the discrete
#: update inside the unit circle — so this is the stability guarantee's one
#: knob, not a tuning parameter. A two-step time constant is already a zone
#: with essentially no thermal inertia.
MIN_TAU_STEPS = 2.0

#: Default *component* time constants in hours: zone<->ambient, zone<->mass
#: from the zone's side, and mass<->zone from the mass's side. The last two
#: differ by the capacitance ratio C_m / C_z = a_m / a_c, so 1 h against 4 h
#: makes the mass four times heavier than the air. Override per run with
#: `cfg.rc_tau_ambient_h` and friends.
#:
#: These three are not the numbers you read off the model. The *modes* are,
#: and they are neither of the three: at a 5-minute timestep this triple
#: gives tau_fast = 0.6 h and tau_slow = 13.4 h — an ordinary commercial
#: building. `mode_taus` is the conversion, and the reportcard prints it, so
#: set these, train, and read back what you actually built.
DEFAULT_TAU_AMBIENT_H = 2.0
DEFAULT_TAU_ZONE_MASS_H = 1.0
DEFAULT_TAU_MASS_ZONE_H = 4.0

#: Raw parameter behind every learned branch coefficient at construction:
#: `softplus(-4) ≈ 0.018`, so a fresh block starts as near-pure free decay
#: for the same reason `g` starts at zero — an O(1) forcing would swamp the
#: RC dynamics for the first epochs. Far enough from softplus's flat side
#: that Adam moves it on the first step.
_NEAR_ZERO_THETA = -4.0

#: Where `softplus(theta)` becomes `theta` to double precision. The exact
#: inverse `log(expm1(x))` differs from `x` by `log1p(-exp(-x))`, which at
#: x = 36 is ~2e-16 — far below the ~7e-15 gap between adjacent float64s of
#: that size — so past this point the two return the same float. Purely a
#: numerical threshold; nothing about the model changes at it.
_SOFTPLUS_LINEAR = 36.0


def _init_theta(tau_hours: float, steps_per_hour: float) -> float:
    """Raw parameter whose `rates()` reading is `tau_hours`.

    Inverts `tau = MIN_TAU_STEPS + softplus(theta)`. Coarse data can make a
    requested time constant unrepresentable — at an hourly timestep the
    1 h zone<->mass default is below the two-step floor — so the request is
    clamped rather than allowed to produce a NaN. The reportcard prints what
    the clamp settled on.

    The far end is only an arithmetic hazard. `softplus` saturates to the
    identity, so the exact inverse converges to its own argument — but
    `math.expm1` overflows float64 past ~709.78 on the way there, which made
    every `tau_hours` above `(709.78 + MIN_TAU_STEPS) / steps_per_hour` raise
    a bare `OverflowError`. That ceiling tightens as the data gets finer:
    59 h at 5-minute samples, 11.9 h at 1-minute — below the `tau_slow` the
    shipped defaults themselves produce, so ordinary priors hit it. Past
    `_SOFTPLUS_LINEAR` the identity *is* the exact inverse in float64, so it
    is returned instead.

    Nothing else changes at that switch, and nothing needs to: softplus
    saturating to the identity means its derivative is 1, which is the
    best-conditioned corner of this parameterization rather than a degenerate
    one. A long time constant that comes back out of `rates()` is exactly the
    one that went in; whether it is a *good* prior is what the reportcard's
    `tau_slow` is for.
    """
    tau_steps = max(tau_hours * steps_per_hour, MIN_TAU_STEPS + 1.0)
    excess = tau_steps - MIN_TAU_STEPS
    if excess > _SOFTPLUS_LINEAR:
        return excess
    return math.log(math.expm1(excess))


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
    """`a_o (T_oa - z)`: conduction to the declared outdoor temperature."""

    def __init__(self, column: dict, zone_shift: list[float], zone_stretch: list[float]):
        super().__init__()
        _buffer(self, "shift", column["shift"])
        _buffer(self, "stretch", column["stretch"])
        _buffer(self, "zone_shift", zone_shift)
        _buffer(self, "zone_stretch", zone_stretch)

    def forward(self, a_ambient: torch.Tensor, zone: torch.Tensor,
                outdoor: torch.Tensor) -> torch.Tensor:
        zone_phys = zone * self.zone_stretch + self.zone_shift
        outdoor_phys = outdoor * self.stretch + self.shift
        return a_ambient * (outdoor_phys - zone_phys) / self.zone_stretch


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

    `cap = softplus(θ_c)` is °C per step at full stage; the slope at the
    origin is `headroom · sigmoid(θ_s)`, so `w = cap / slope` and the bound
    `cap / w ≤ headroom` holds by construction — `headroom` is the block's
    `1 - a_m - a_o`, handed in each step so the bound follows the rates if
    they train. `θ_s = 0` starts at half the budget; `cap` starts near zero.

    The input is the pre-stack's demand feature, scaled to about [0, 1] by
    `demand_scale`; this branch holds the same scale and undoes it, so `w`
    reads in °C of demand.
    """

    def __init__(self, sign: int, control: dict, state: dict):
        super().__init__()
        self.sign = int(sign)
        self.theta_cap = nn.Parameter(torch.tensor(_NEAR_ZERO_THETA))
        self.theta_slope = nn.Parameter(torch.tensor(0.0))
        _buffer(self, "scale", demand_scale(sign, control, state))
        _buffer(self, "zone_stretch", state["stretch"])

    def cap(self) -> torch.Tensor:
        return F.softplus(self.theta_cap)

    def slope_share(self) -> torch.Tensor:
        """Fraction of the stability headroom the slope at the origin uses."""
        return torch.sigmoid(self.theta_slope)

    def width(self, headroom: torch.Tensor) -> torch.Tensor:
        """`w`: °C of demand over which the response ramps from off to full."""
        return self.cap() / (headroom * self.slope_share())

    def forward(self, x: torch.Tensor, headroom: torch.Tensor) -> torch.Tensor:
        demand_c = x * self.scale
        return self.sign * self.cap() * torch.tanh(demand_c / self.width(headroom)) / self.zone_stretch


class RCBlock(nn.Module):
    """One 2R2C network per Y channel, sharing a learned heat-input map.

    Reads the same flattened `in_steps` history window every other block
    reads (see `windows`), and returns the **increment** `z_{k+1} - z_k`.
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
        in_steps: history window length.
        steps_per_hour: timesteps per hour. Only ever used to convert time
            constants between hours and steps — on the way in for
            `tau_hours`, on the way out for `describe`.
        hidden_size / num_layers: size of the learned heat-input MLP.
        tau_hours: the storage prior, `(ambient, zone_mass, mass_zone)`.
        train_taus: let the optimizer move that prior. Off by default; the
            module docstring explains why.
        channel_names: Y column names, for labelling `describe`.
        u_names / d_names: this block's U and D column names, in the order
            its window carries them; what `inputs` and `ambient` refer to.
        inputs: per-column declarations, `{column: {"kind", "sign", ...}}`,
            as validated by `blocks.build_blocks`. A `demand` column feeds a
            `SaturatingTerm`, a `linear` one a `LinearTerm`; the rest feed
            the MLP.
        ambient: the D column read by the explicit conduction term, or None
            for conduction to a zero ambient.
        norm_stats: per-column norm stats, needed for any declared branch.
    """

    def __init__(self, ny: int, nu: int, nd: int, in_steps: int,
                 steps_per_hour: float, hidden_size: int, num_layers: int,
                 tau_hours: tuple[float, float, float] = (
                     DEFAULT_TAU_AMBIENT_H, DEFAULT_TAU_ZONE_MASS_H,
                     DEFAULT_TAU_MASS_ZONE_H),
                 train_taus: bool = False,
                 channel_names: tuple[str, ...] = (),
                 u_names: tuple[str, ...] = (),
                 d_names: tuple[str, ...] = (),
                 inputs: Optional[dict[str, dict]] = None,
                 ambient: Optional[str] = None,
                 norm_stats: Optional[dict[str, dict]] = None):
        super().__init__()
        self.widths = (ny, nu, nd)
        self.ny = ny
        self.channel_names = channel_names or tuple(f"y{i}" for i in range(ny))
        self.in_steps = in_steps
        self.in_features = in_steps * (ny + nu + nd)
        self.out_features = ny

        self.steps_per_hour = steps_per_hour

        y_block = windows.key_blocks(self.widths, in_steps)[0]
        self._y_steps = windows.per_step(y_block, ny, in_steps)
        _, self._u_last, self._d_last = windows.last_step(self.widths, in_steps)

        self._theta = nn.ParameterDict({
            name: nn.Parameter(
                torch.full((ny,), _init_theta(hours, steps_per_hour)),
                requires_grad=train_taus,
            )
            for name, hours in zip(("ambient", "zone_mass", "mass_zone"), tau_hours)
        })

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

        self.saturating_names = [c for c, e in inputs.items() if e["kind"] == "demand"]
        self._saturating_pos = [drive_names.index(c) for c in self.saturating_names]
        self.saturating = nn.ModuleList(
            SaturatingTerm(inputs[c]["sign"], norm_stats[c], norm_stats[self.channel_names[0]])
            for c in self.saturating_names)

        # Heat input `g`, from the undeclared inputs only — never from the
        # state. That restriction is the whole stability argument; see the
        # module docstring before relaxing it.
        declared = set(inputs) | ({ambient} if ambient else set())
        self.mlp_names = [c for c in drive_names if c not in declared]
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
            self.intercept = None
        else:
            # Every column is declared: what is left of `g` is a constant
            # heat gain.
            self.gain = None
            self.intercept = nn.Parameter(torch.zeros(ny))

        self._mass: torch.Tensor | None = None

    # ── parameters ──────────────────────────────────────────────

    def rates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three per-step exchange rates, each in (0, 1/MIN_TAU_STEPS)."""
        return tuple(
            1.0 / (MIN_TAU_STEPS + nn.functional.softplus(self._theta[name]))
            for name in ("ambient", "zone_mass", "mass_zone")
        )

    def reg_error(self) -> torch.Tensor:
        if self.gain is None:
            return torch.zeros(())
        weights = [p.pow(2).mean() for p in self.gain.parameters() if p.dim() > 1]
        return torch.mean(torch.stack(weights))

    def reset(self) -> None:
        """Clear the mass temperature. Call before each new sequence."""
        self._mass = None

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
        a_ambient, a_zone_mass, a_mass_zone = self.rates()

        zone = x[..., self._y_steps[-1]]
        drive = torch.cat([x[..., self._u_last], x[..., self._d_last]], dim=-1)

        # Under vmap the batch dimension is gone and state cannot be stored,
        # so every call is a cold one. Same convention as GRUBlock.
        batched = x.ndim > 1
        mass = self._mass if batched else None
        if mass is None:
            mass = self._seed_mass(x, a_mass_zone)

        if self.ambient is None:
            conduction = -a_ambient * zone
        else:
            conduction = self.ambient(a_ambient, zone, self._column(drive, self._ambient_pos))
        increment = (a_zone_mass * (mass - zone) + conduction
                     + self._heat_input(drive, headroom=1 - a_zone_mass - a_ambient))
        if batched:
            self._mass = mass + a_mass_zone * (zone - mass)
        return increment

    @staticmethod
    def _column(drive: torch.Tensor, pos: int) -> torch.Tensor:
        """One column of the drive vector, keeping its trailing axis."""
        return drive[..., pos:pos + 1]

    def _heat_input(self, drive: torch.Tensor, headroom: torch.Tensor) -> torch.Tensor:
        """The declared branches plus `g`, summed. `headroom` is the zone's
        remaining self-coefficient, which bounds every saturating slope."""
        if self.gain is None:
            total = self.intercept.expand(*drive.shape[:-1], self.ny)
        else:
            total = self.gain(drive.index_select(-1, self._mlp_pos))
        for pos, term in zip(self._linear_pos, self.linear):
            total = total + term(self._column(drive, pos))
        for pos, term in zip(self._saturating_pos, self.saturating):
            total = total + term(self._column(drive, pos), headroom)
        return total

    # ── description ─────────────────────────────────────────────

    def describe(self) -> dict:
        """Per-channel time constants in hours, and the forcing branches.

        `tau_slow` is the storage number: how long a temperature difference
        put into this zone survives. It is the one figure worth checking
        against the real building. `forcing` lists which column feeds which
        branch and what each learned: `k` in zone units per step per
        physical unit for `linear`; `cap` (°C/step), `w` (°C of demand) and
        `slope_share` (the fraction of the stability budget the slope uses)
        for `saturating`.
        """
        with torch.no_grad():
            a_ambient, a_zone_mass, _ = self.rates()
            tau_fast, tau_slow = mode_taus(*self.rates())
            headroom = 1 - a_zone_mass - a_ambient
            forcing = {
                "ambient": self.ambient_name,
                "linear": {
                    name: {"k": round(float(term.coefficient()), 6)}
                    for name, term in zip(self.linear_names, self.linear)
                },
                "saturating": {
                    name: {"cap": round(float(term.cap()), 4),
                           "w": round(float(term.width(headroom)), 4),
                           "slope_share": round(float(term.slope_share()), 3)}
                    for name, term in zip(self.saturating_names, self.saturating)
                },
                "mlp": list(self.mlp_names),
            }
        per_hour = self.steps_per_hour
        return {
            "time_constants_h": {
                name: {"fast": round(float(fast) / per_hour, 3),
                       "slow": round(float(slow) / per_hour, 3)}
                for name, fast, slow in zip(self.channel_names, tau_fast, tau_slow)
            },
            "forcing": forcing,
        }
