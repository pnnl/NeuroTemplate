"""
Behavioural tests for the four physics scenarios.

The subsystem used to have no test of scenario behaviour at all, and a latent
bug that only showed up on a plant whose Y was not all temperatures. Both are
addressed here: hand-written surrogates with known physics make every verdict
exactly predictable, and one test pins the bug (an equilibrium temperature must
never be written into a power channel).
"""

import pytest
import torch
from torch import nn

from neuromancer.system import Node, SystemPreview

from neurotemplate.modeling.contract import Windowed, contract_model
from neurotemplate.modeling.stepper import Stepper
from neurotemplate.modeling.hinge import Hinges
from neurotemplate.metrics.physics import (
    SynthSpec,
    build_scenarios,
)

ZONE = "zone [°C]"
POWER = "dx power [W]"
SETPT = "cooling setpoint [°C]"
AMBIENT = "outdoor [°C]"

NM = {"Y": [ZONE, POWER], "U": [SETPT], "D": [AMBIENT],
      "t": ["sec_elapsed"]}
NORM_STATS = {
    ZONE: {"shift": 0.0, "stretch": 1.0, "min": 18.0, "max": 30.0},
    POWER: {"shift": 0.0, "stretch": 1.0, "min": 0.0, "max": 6000.0},
    SETPT: {"shift": 0.0, "stretch": 1.0, "min": 20.0, "max": 26.0},
    AMBIENT: {"shift": 0.0, "stretch": 1.0, "min": -5.0, "max": 35.0},
}
HISTORY_STEPS, N_STEPS = 1, 12


def _spec(**overrides) -> SynthSpec:
    # The setpoint's `off` is its recorded maximum (26 °C), so the generated
    # sweep drives it down to 20 °C. `responds` is signed against the control
    # *increasing*: a higher cooling setpoint means less cooling, so +1.
    base = dict(
        states=[ZONE],
        ambient=AMBIENT,
        off={SETPT: 26.0, POWER: 0.0},
        responds={ZONE: {SETPT: +1}},
    )
    base.update(overrides)
    return SynthSpec(**base)


class _Plant(nn.Module):
    """A hand-written plant, as a core behind a `Stepper`. `step` returns the next [zone, power] row from the current
    y, u and d rows (one-step windows; the stats are the identity, so these
    are physical units).
    """

    def __init__(self, step):
        super().__init__()
        self.step = step
        # A zero parameter added to the output, so the training penalty has a
        # gradient path to check — a real dynamics block has parameters here.
        self.zero = nn.Parameter(torch.zeros(1))

    def forward(self, y, u, d):
        return self.step(y[:, -1], u[:, -1], d[:, -1]) + self.zero


def _system(step, nm=NM, stats=NORM_STATS) -> SystemPreview:
    widths = tuple(len(nm[key]) for key in ("Y", "U", "D"))
    model = Stepper(_Plant(step), nm, stats, HISTORY_STEPS, residual=False)
    node = Node(Windowed(model, widths, HISTORY_STEPS),
                ["yn", "U", "D"], ["yn"], name="plant")
    return SystemPreview([node], name="plant system", nstep_key="Y",
                         start_iter=HISTORY_STEPS - 1)


def _scenarios(step, spec=None, n=6, n_steps=N_STEPS):
    return build_scenarios(
        system=_system(step), nm_mapping=NM, norm_stats=NORM_STATS,
        synth_cfg=spec or _spec(), history_steps=HISTORY_STEPS, n_scenarios=n,
        n_steps=n_steps, device="cpu",
    )


# ── Reference plants ─────────────────────────────────────────────


OFF_SETPOINT = 26.0     # the `off` value in `_spec`; below it, cooling is on


def _cooling(u):
    return (u[:, 0] < OFF_SETPOINT).float()


def _perfect(y, u, d):
    """Relaxes toward ambient at 25% a step and loses 1 °C a step when cooling.

    Holds exactly at equilibrium (ambient equal to the zone, cooling off), so
    it should pass all three checks.
    """
    zone = y[:, 0]
    next_zone = zone + 0.25 * (d[:, 0] - zone) - 1.0 * _cooling(u)
    return torch.stack([next_zone, torch.zeros_like(zone)], dim=1)


def _inert(y, u, d):
    """Ignores everything: holds forever, whatever you do to it."""
    return torch.stack([y[:, 0], torch.zeros_like(y[:, 0])], dim=1)


def _rampng(y, u, d):
    """Drifts upward at rest — the failure equilibrium exists to catch."""
    return torch.stack([y[:, 0] + 0.5, torch.zeros_like(y[:, 0])], dim=1)


def _backwards(y, u, d):
    """Holds at rest, but responds to a call for cooling by heating up."""
    zone = y[:, 0]
    return torch.stack([zone + 0.4 * _cooling(u), torch.zeros_like(zone)],
                       dim=1)


# ── Equilibrium ──────────────────────────────────────────────────


def test_a_plant_that_holds_passes_equilibrium():
    (r,) = _scenarios(_perfect)[0].check()
    assert r.passed
    assert r.channel == ZONE
    assert r.value == pytest.approx(0.0, abs=1e-4)
    assert "Held to" in r.headline


def test_a_drifting_plant_fails_equilibrium():
    (r,) = _scenarios(_rampng)[0].check()
    assert not r.passed
    # 0.5 °C per step, mean over 12 steps of |k * 0.5| = 3.25 °C.
    assert r.value == pytest.approx(3.25, rel=1e-3)
    assert "integrating something it should not" in r.headline
    assert r.tolerance == pytest.approx(0.02 * (30.0 - 18.0))


# ── Step response ────────────────────────────────────────────────


def test_a_responsive_plant_passes_the_step():
    (r,) = _scenarios(_perfect)[1].check()
    assert r.passed
    assert r.name == f"Step response: {SETPT} to 20"
    assert r.value < 0                      # the zone came down
    assert "Moved" in r.headline


def test_a_backwards_response_is_distinguished_from_no_response():
    (r,) = _scenarios(_backwards)[1].check()
    assert not r.passed
    assert r.value > 0                      # moved, but upward
    assert "the wrong way" in r.headline


def test_the_step_is_measured_net_of_the_drift_at_rest():
    """A model that drifts at rest must not have its drift read as a response
    (field report: one −0.23 °C drift failed one sweep and passed another)."""
    def drifting(y, u, d):
        return _perfect(y, u, d) + torch.tensor([0.3, 0.0])

    (r,) = _scenarios(drifting)[1].check()
    (clean,) = _scenarios(_perfect)[1].check()
    assert r.value == pytest.approx(clean.value, abs=1e-4)


HEAT = "heating setpoint [°C]"


def test_a_hinge_target_inactive_everywhere_is_skipped_not_failed():
    """A heating setpoint swept to its recorded minimum (16 °C) calls for no
    heat in an 18-30 °C zone: a declared hinge makes that zero by
    construction, so the sweep asserts nothing. Its other end still does."""
    nm = {"Y": [ZONE, POWER], "U": [HEAT], "D": [AMBIENT], "t": ["sec_elapsed"]}
    stats = dict(NORM_STATS, **{HEAT: {"shift": 0.0, "stretch": 1.0,
                                       "min": 16.0, "max": 24.0}})
    spec = SynthSpec(states=[ZONE], ambient=AMBIENT, off={HEAT: 10.0, POWER: 0.0},
                     responds={ZONE: {HEAT: +1}})

    def thermostat(y, u, d):
        zone = y[:, 0]
        heat = (u[:, 0] > zone).float()
        return torch.stack([zone + 0.25 * (d[:, 0] - zone) + heat,
                            torch.zeros_like(zone)], dim=1)

    def on_hinge(y, u, d):
        """The same plant, handed the hinge a declaration derives from the setpoint."""
        zone = y[:, 0]
        return torch.stack([zone + 0.25 * (d[:, 0] - zone) + (u[:, 0] > 0).float(),
                            torch.zeros_like(zone)], dim=1)

    def step_results(declared: bool):
        system = _system(on_hinge if declared else thermostat, nm, stats)
        if declared:
            contract_model(system).hinges = Hinges(
                {HEAT: {"target": ZONE, "map": "relu"}}, nm, stats)
        return build_scenarios(
            system=system, nm_mapping=nm, norm_stats=stats, synth_cfg=spec,
            history_steps=HISTORY_STEPS, n_scenarios=16, n_steps=N_STEPS, device="cpu",
        )[1].check()

    undeclared = {r.name: r for r in step_results(declared=False)}
    assert not undeclared[f"Step response: {HEAT} to 16"].passed
    (declared,) = step_results(declared=True)
    assert declared.name == f"Step response: {HEAT} to 24"
    assert declared.passed


# ── Free decay ───────────────────────────────────────────────────


def test_a_plant_that_relaxes_toward_ambient_passes():
    (r,) = _scenarios(_perfect)[3].check()
    assert r.passed
    assert 0.0 < r.value <= 1.0
    assert "Closed" in r.headline and "gap to ambient" in r.headline


def test_a_plant_with_no_envelope_coupling_fails():
    (r,) = _scenarios(_inert)[3].check()
    assert not r.passed
    assert r.value == pytest.approx(0.0, abs=1e-6)
    assert "no envelope coupling" in r.headline


# ── Reachability ─────────────────────────────────────────────────
#
# The honest version of what the actuator probe was reaching for: instead of
# holding co-actuators at an arbitrary range midpoint over whatever weather
# happened to be in one test file, everything starts at a declared equilibrium
# and is pushed the same way on purpose.


def _reach(step, grad_masks=None, n_steps=N_STEPS):
    return build_scenarios(
        system=_system(step), nm_mapping=NM, norm_stats=NORM_STATS,
        synth_cfg=_spec(), history_steps=HISTORY_STEPS, n_scenarios=8,
        n_steps=n_steps, device="cpu", grad_masks=grad_masks,
    )[2]


def test_reachability_distinguishes_a_backwards_surrogate_from_a_narrow_one():
    """Crossed endpoints are a different finding from a small envelope.

    Caught on a real run: an undertrained surrogate came back with "Reaches
    only 19.22 to 12.47 — a span of -6.75", which reads as a range and is not
    one. Pushing everything toward a higher value landed *lower*, meaning the
    model contradicts every declared sign at once.
    """
    def backwards(y, u, d):
        # Cooling harder (lower setpoint) warms the zone.
        zone = y[:, 0]
        return torch.stack([zone + 0.5 * _cooling(u), torch.zeros_like(zone)],
                           dim=1)

    (r,) = _reach(backwards).check()

    assert not r.passed
    assert r.value < 0
    assert "Responds backwards" in r.headline
    assert "pushed down it settles at" in r.headline
    assert "cannot reach" not in r.headline


# ── The bug this subsystem used to have ──────────────────────────


def test_the_equilibrium_state_never_lands_in_a_readout_channel():
    """`states` exists so a Watt channel is not seeded as a temperature.

    Before the partition, the equilibrium temperature was written into every Y
    column and sampled from the min/max of all of them — on this plant that
    means "equilibrium temperatures" drawn from 0-6000 and 5000 written into
    the zone.
    """
    seen = {}

    def record(y, u, d):
        seen.setdefault("y0", y.clone())
        return torch.stack([y[:, 0], y[:, 1]], dim=1)

    _scenarios(record)[0].check()

    zone, power = seen["y0"][:, 0], seen["y0"][:, 1]
    assert torch.all((zone >= 18.0) & (zone <= 30.0)), "zone seeded out of range"
    # The readout defaults to its recorded minimum — 0 W, not a temperature.
    assert torch.all(power == 0.0)
