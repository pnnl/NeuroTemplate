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

    z_{k+1} = z_k + a_m (m_k - z_k) - a_o z_k + g(u_k, d_k)
    m_{k+1} = m_k + a_c (z_k - m_k)

with `z` the observed zone temperature, `m` the hidden mass temperature, and
three positive per-step exchange rates `a_o`, `a_m`, `a_c`. The heat input
`g` is learned and black: an MLP over the current inputs. We do not have a
plug-load channel, an internal-gains channel, or a metered zone-level heat
extraction, so `g` is where all of that goes, along with the solar gain and
the ambient conduction term (see below). The library learns the forcing and
asserts the storage.

The `-a_o z_k` term is conduction to a *zero* ambient, which is only correct
because `g` is affine in D and D carries the outdoor temperature: the real
`+a_o·T_ambient` is one of the terms `g` can form. Nothing has to tell this
block which D column is the weather.

Two properties fall out of that split, and they are the reason to reach for
this family over `lssm`:

**Unconditional stability.** `g` reads inputs only — never `z`, never `m` —
so the state matrix is fixed by three positive rates whatever the MLP does.
Rates are parameterized as `1 / (MIN_TAU_STEPS + softplus(θ))`, so each is
below `1/MIN_TAU_STEPS = 0.5` and their sum below 2, which is exactly the
condition for the discrete update's eigenvalues to sit inside the unit
circle (see `mode_taus`). A 2R2C rollout cannot diverge, at any horizon,
under any input — including the adversarial inputs a DPC policy will find.

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

`g` cannot see `z`, so it cannot represent a thermostat deadband — "the
compressor is off whenever the setpoint is above the zone" is a function of
the *difference*, and this block will render it as a monotone response to
setpoint with no cut-off. Cooling delivered is therefore smooth where the
real plant's is staged. That is the price of the stability guarantee, and
it is why this is one family among several rather than the only one.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from neuromancer.modules import blocks

from neurotemplate.modeling import windows

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


def _init_theta(tau_hours: float, steps_per_hour: float) -> float:
    """Raw parameter whose `rates()` reading is `tau_hours`.

    Inverts `tau = MIN_TAU_STEPS + softplus(theta)`. Coarse data can make a
    requested time constant unrepresentable — at an hourly timestep the
    1 h zone<->mass default is below the two-step floor — so the request is
    clamped rather than allowed to produce a NaN. The floor is a starting
    point either way, and training moves it.
    """
    tau_steps = max(tau_hours * steps_per_hour, MIN_TAU_STEPS + 1.0)
    return math.log(math.expm1(tau_steps - MIN_TAU_STEPS))


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
        steps_per_hour: timesteps per hour, used only to turn the initial
            time constants from hours into steps.
        hidden_size / num_layers: size of the learned heat-input MLP.
    """

    def __init__(self, ny: int, nu: int, nd: int, in_steps: int,
                 steps_per_hour: float, hidden_size: int, num_layers: int,
                 tau_hours: tuple[float, float, float] = (
                     DEFAULT_TAU_AMBIENT_H, DEFAULT_TAU_ZONE_MASS_H,
                     DEFAULT_TAU_MASS_ZONE_H),
                 train_taus: bool = False,
                 channel_names: tuple[str, ...] = ()):
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

        # Heat input, from the current inputs only — never from the state.
        # That restriction is the whole stability argument; see the module
        # docstring before relaxing it.
        self.gain = blocks.MLP(
            insize=nu + nd, outsize=ny, bias=True,
            linear_map=nn.Linear, nonlin=nn.ReLU,
            hsizes=num_layers * [hidden_size],
        )
        # Start with no forcing at all, so the first epochs see a pure
        # free-decay building and the optimizer shapes the storage before it
        # shapes the heat input. Without this the MLP's O(1) initial output
        # swamps a normalized temperature step, which is O(0.01).
        final = self.gain.linear[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        self._mass: torch.Tensor | None = None

    # ── parameters ──────────────────────────────────────────────

    def rates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three per-step exchange rates, each in (0, 1/MIN_TAU_STEPS)."""
        return tuple(
            1.0 / (MIN_TAU_STEPS + nn.functional.softplus(self._theta[name]))
            for name in ("ambient", "zone_mass", "mass_zone")
        )

    def reg_error(self) -> torch.Tensor:
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

        increment = a_zone_mass * (mass - zone) - a_ambient * zone + self.gain(drive)
        if batched:
            self._mass = mass + a_mass_zone * (zone - mass)
        return increment

    # ── description ─────────────────────────────────────────────

    def describe(self) -> dict:
        """Per-channel time constants in hours, for the reportcard.

        `tau_slow` is the storage number: how long a temperature difference
        put into this zone survives. It is the one figure worth checking
        against the real building.
        """
        with torch.no_grad():
            tau_fast, tau_slow = mode_taus(*self.rates())
        per_hour = self.steps_per_hour
        return {
            "time_constants_h": {
                name: {"fast": round(float(fast) / per_hour, 3),
                       "slow": round(float(slow) / per_hour, 3)}
                for name, fast, slow in zip(self.channel_names, tau_fast, tau_slow)
            }
        }
