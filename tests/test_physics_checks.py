"""
Behavioural tests for the three physics scenarios.

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

from neurotemplate.modeling.scenarios import (
    Actuation,
    EquilibriumScenario,
    FreeDecayScenario,
    ScenarioResult,
    StepResponseScenario,
    SynthSpec,
    build_scenarios,
    checks_to_dict,
    checks_to_markdown,
    format_checks,
    run_checks,
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
IN_STEPS, N_STEPS = 1, 12


def _spec(**overrides) -> SynthSpec:
    base = dict(
        states=[ZONE],
        ambient=AMBIENT,
        off={SETPT: 26.0},
        actuations=[Actuation("cooling", {SETPT: 20.0}, direction=-1)],
    )
    base.update(overrides)
    return SynthSpec(**base)


class _Plant(nn.Module):
    """A hand-written plant. `step` returns the next [zone, power] row.

    Signature matches what the dynamics Node feeds: current y, u and d rows,
    all in physical units (normalization here is the identity).
    """

    def __init__(self, step):
        super().__init__()
        self.step = step
        # A zero parameter added to the output, so the training penalty has a
        # gradient path to check — a real dynamics block has parameters here.
        self.zero = nn.Parameter(torch.zeros(1))

    def forward(self, y, u, d):
        return self.step(y, u, d) + self.zero


def _system(step) -> SystemPreview:
    node = Node(_Plant(step), ["yn", "U", "D"], ["yn"], name="plant")
    return SystemPreview([node], name="plant system", nstep_key="Y",
                         start_iter=IN_STEPS - 1)


def _scenarios(step, spec=None, n=6, n_steps=N_STEPS):
    return build_scenarios(
        system=_system(step), nm_mapping=NM, norm_stats=NORM_STATS,
        synth_cfg=spec or _spec(), in_steps=IN_STEPS, n_scenarios=n,
        n_steps=n_steps, device="cpu",
    )


def _by_name(results: list[ScenarioResult], prefix: str) -> ScenarioResult:
    return next(r for r in results if r.name.startswith(prefix))


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
    assert "held to" in r.headline


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
    assert r.name == "Step response: cooling"
    assert r.value < 0                      # the zone came down
    assert "moved" in r.headline


def test_an_inert_actuator_is_named_as_such():
    (r,) = _scenarios(_inert)[1].check()
    assert not r.passed
    assert r.value == pytest.approx(0.0, abs=1e-6)
    assert "did not respond" in r.headline
    assert "no lever" in r.headline


def test_a_backwards_response_is_distinguished_from_no_response():
    (r,) = _scenarios(_backwards)[1].check()
    assert not r.passed
    assert r.value > 0                      # moved, but upward
    assert "the wrong way" in r.headline


def test_no_actuations_means_no_step_checks():
    results = _scenarios(_perfect, spec=_spec(actuations=[]))[1].check()
    assert results == []


# ── Free decay ───────────────────────────────────────────────────


def test_a_plant_that_relaxes_toward_ambient_passes():
    (r,) = _scenarios(_perfect)[2].check()
    assert r.passed
    assert 0.0 < r.value <= 1.0
    assert "closed" in r.headline and "gap to ambient" in r.headline


def test_a_plant_with_no_envelope_coupling_fails():
    (r,) = _scenarios(_inert)[2].check()
    assert not r.passed
    assert r.value == pytest.approx(0.0, abs=1e-6)
    assert "no envelope coupling" in r.headline


def test_overshooting_ambient_fails():
    def overshoot(y, u, d):
        zone = y[:, 0]
        return torch.stack([zone + 1.5 * (d[:, 0] - zone),
                            torch.zeros_like(zone)], dim=1)

    (r,) = _scenarios(overshoot, n_steps=1)[2].check()
    assert not r.passed
    assert r.value > 1.0
    assert "overshot ambient" in r.headline


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


# ── Spec validation ──────────────────────────────────────────────


def test_validate_names_every_problem_at_once():
    bad = SynthSpec(states=["nope"], ambient="also nope", off={},
                    actuations=[Actuation("x", {"nope": 1.0}, direction=0)])
    with pytest.raises(ValueError) as exc:
        bad.validate(NM)
    message = str(exc.value)
    assert "`states` names columns that are not in Y" in message
    assert "`ambient`" in message
    assert "missing: ['cooling setpoint [°C]']" in message
    assert "direction 0" in message


def test_validate_rejects_an_empty_state_list():
    with pytest.raises(ValueError, match="`states` is empty"):
        SynthSpec(states=[], ambient=AMBIENT, off={SETPT: 26.0}).validate(NM)


def test_validate_rejects_stray_off_columns():
    spec = _spec(off={SETPT: 26.0, "not a column": 1.0})
    with pytest.raises(ValueError, match="not in the mapping"):
        spec.validate(NM)


def test_build_scenarios_validates_before_constructing():
    with pytest.raises(ValueError, match="not usable with this column mapping"):
        build_scenarios(
            system=None, nm_mapping=NM, norm_stats=NORM_STATS,
            synth_cfg=_spec(states=["nope"]), in_steps=1, n_scenarios=2,
            n_steps=4, device="cpu",
        )


def test_thresholds_scale_with_the_channel_rather_than_assuming_degrees():
    """A plant measured in a 0-1 range must get a 0-1-sized tolerance."""
    stats = dict(NORM_STATS, **{ZONE: {"shift": 0.0, "stretch": 1.0,
                                       "min": 0.0, "max": 1.0}})
    scenarios = build_scenarios(
        system=_system(_perfect), nm_mapping=NM, norm_stats=stats,
        synth_cfg=_spec(), in_steps=IN_STEPS, n_scenarios=4, n_steps=N_STEPS,
        device="cpu",
    )
    (r,) = scenarios[0].check()
    assert r.tolerance == pytest.approx(0.02 * 1.0)


# ── Equilibrium seeding range ────────────────────────────────────


SUPPLY = "supply air [°C]"


def _two_state_stats(supply_lo=8.6, supply_hi=54.5):
    """Zone plus a supply-air channel with a far wider span, as measured."""
    return dict(
        NORM_STATS,
        **{ZONE: {"shift": 0.0, "stretch": 1.0, "min": 18.9, "max": 27.3},
           SUPPLY: {"shift": 0.0, "stretch": 1.0,
                    "min": supply_lo, "max": supply_hi}},
    )


def _two_state_scenarios(stats, step=None):
    nm = {**NM, "Y": [ZONE, POWER, SUPPLY]}

    def passthrough(y, u, d):
        return y.clone()

    return build_scenarios(
        system=_system(step or passthrough), nm_mapping=nm, norm_stats=stats,
        synth_cfg=_spec(states=[ZONE, SUPPLY]), in_steps=IN_STEPS,
        n_scenarios=8, n_steps=4, device="cpu",
    )


def test_equilibrium_is_seeded_from_the_intersection_not_the_union():
    """The union would seed a 51 °C building; every state has seen 18.9-27.3."""
    scenario = _two_state_scenarios(_two_state_stats())[0]

    assert (scenario.t_min, scenario.t_max) == pytest.approx((18.9, 27.3))

    sampled = scenario._sample_equilibrium(256)
    assert float(sampled.min()) >= 18.9
    assert float(sampled.max()) <= 27.3


def test_a_single_state_is_unaffected_by_the_intersection_rule():
    """Backward compatibility: with one state, intersection == union."""
    scenario = _scenarios(_perfect)[0]
    assert (scenario.t_min, scenario.t_max) == pytest.approx((18.0, 30.0))


def test_states_with_no_shared_temperature_fail_loudly():
    """A chiller loop and a zone never overlap in the data — say why."""
    stats = _two_state_stats(supply_lo=2.0, supply_hi=12.0)

    with pytest.raises(ValueError) as exc:
        _two_state_scenarios(stats)

    message = str(exc.value)
    assert "No temperature is shared by every channel named in `states`" in message
    assert "18.9..27.3" in message and "2..12" in message
    # The data is what is missing, not the physics.
    assert "not that they physically cannot reach one" in message


def test_the_free_decay_gap_scales_with_the_intersection():
    """MIN_GAP_FRAC is 10% of the seeding window, so it narrows with it."""
    wide = _scenarios(_perfect)[2]                      # zone alone: 18-30
    narrow = _two_state_scenarios(_two_state_stats())[2]  # intersection: 18.9-27.3

    assert wide.MIN_GAP_FRAC * (wide.t_max - wide.t_min) == pytest.approx(1.2)
    assert (narrow.MIN_GAP_FRAC * (narrow.t_max - narrow.t_min)
            == pytest.approx(0.84))


def test_a_constant_state_channel_does_not_divide_by_zero():
    """The reportcard warns about constant columns; this must not crash on one."""
    stats = dict(NORM_STATS, **{ZONE: {"shift": 0.0, "stretch": 1.0,
                                       "min": 22.0, "max": 22.0}})
    scenarios = build_scenarios(
        system=_system(_inert), nm_mapping=NM, norm_stats=stats,
        synth_cfg=_spec(), in_steps=IN_STEPS, n_scenarios=2, n_steps=4,
        device="cpu",
    )
    results = run_checks(scenarios)
    assert len(results) == 3
    assert all(r.tolerance > 0 for r in results[:2])


def test_off_is_required_for_controls_but_defaulted_elsewhere():
    """Controls must be declared; a disturbance may rest at its minimum."""
    spec = SynthSpec(states=[ZONE], ambient=AMBIENT, off={SETPT: 26.0})
    spec.validate(NM)   # no entry for the power readout, and that is fine

    with pytest.raises(ValueError, match="missing:"):
        SynthSpec(states=[ZONE], ambient=AMBIENT, off={}).validate(NM)


# ── Reporting ────────────────────────────────────────────────────


def test_run_checks_flattens_every_scenario():
    results = run_checks(_scenarios(_perfect))
    assert [r.name for r in results] == [
        "Equilibrium", "Step response: cooling", "Free decay",
    ]


def test_renderers_show_verdicts_not_losses():
    results = run_checks(_scenarios(_inert))

    table = checks_to_markdown(results)
    assert "**FAIL**" in table
    assert table.count("\n") == 1 + len(results)

    text = format_checks(results)
    assert "[FAIL]" in text and "[ok  ]" in text

    payload = checks_to_dict(results)
    assert set(payload) == {r.name for r in results}
    assert payload["Free decay"]["passed"] is False


def test_the_scenarios_are_assertions_and_carry_no_training_term():
    """The differentiable penalty is gone; nothing here feeds the loss.

    Physicality during training is the gradient sign penalty's job
    (`modeling/gradient_sampling.py`), which is still wired to `grad_gain`.
    """
    for s in _scenarios(_rampng):
        assert not hasattr(s, "compute")
    assert not any(hasattr(r, "penalty") for r in run_checks(_scenarios(_perfect)))


def test_traces_carry_physical_trajectories_for_the_plots():
    for scenario in _scenarios(_perfect):
        trace = scenario.trace(3)
        assert trace["predictions"].shape == (3, IN_STEPS + N_STEPS, len(NM["Y"]))
        assert trace["ambient_traj"].shape[0] == 3
        assert trace["state_names"] == [ZONE]
        assert trace["seed_steps"] == IN_STEPS
