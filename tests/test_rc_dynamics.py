"""
Tests for the 2R2C grey-box family and the per-zone split it rides on.

The family's whole claim is that it asserts things the black-box families
only hope for, so the tests are that list of claims:

  - the update is the RC network's, arithmetic and all, not something
    approximately like it;
  - the reported time constants are the update's real eigenvalues;
  - a rollout cannot diverge, whatever the inputs and wherever the rates
    sit inside their allowed range;
  - the hidden mass remembers past `in_steps`, which is the point;
  - the time constants are a frozen prior unless asked otherwise, so the
    number on the reportcard is the number you set;
  - each zone is genuinely independent of the others.

Anything that would silently produce a wrong model instead fails loudly:
2r2c without residual, without a timestep, or under a head that will not
add the state back.
"""
import math

import pytest
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import PerZoneBlock, get_dynamics
from neurotemplate.modeling.heads import BinaryHead, HeadStack, RegressionHead
from neurotemplate.modeling.rc import MIN_TAU_STEPS, RCBlock, mode_taus
from neurotemplate.modeling.zones import build_zones

SPH = 12.0            # 5-minute data
IN_STEPS = 6

MAPPING = {
    "Y": ["north T", "south T"],
    "U": ["north sp", "south sp", "ahu fan"],
    "D": ["oat", "ghi"],
    "t": ["sec_elapsed"],
}
SPLIT = {
    "north": {"Y": ["north T"], "U": ["north sp"]},
    "south": {"Y": ["south T"], "U": ["south sp"]},
}


def make_cfg(**over) -> Config:
    cfg = Config()
    cfg.model = "2r2c"
    cfg.in_steps = IN_STEPS
    cfg.log2_hidden_size = 4
    cfg.num_layers = 1
    cfg.steps_per_hour = SPH
    cfg.device = torch.device("cpu")
    cfg.verbose = False
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def make_block(ny=1, nu=1, nd=1, **over) -> RCBlock:
    torch.manual_seed(0)
    return RCBlock(ny, nu, nd, IN_STEPS, SPH, hidden_size=8, num_layers=1, **over)


def window(block: RCBlock, zone, drive_u, drive_d) -> torch.Tensor:
    """A key-major window holding a constant history and one input row."""
    ny, nu, nd = block.widths
    return torch.cat([
        torch.tensor(zone, dtype=torch.float32).repeat(1, IN_STEPS * ny),
        torch.tensor(drive_u, dtype=torch.float32).repeat(1, IN_STEPS * nu),
        torch.tensor(drive_d, dtype=torch.float32).repeat(1, IN_STEPS * nd),
    ], dim=-1)


def rollout(system, y0, steps, u=None, d=None, ny=2, nu=3, nd=2):
    data = {
        "yn": torch.full((1, IN_STEPS, ny), y0),
        "U": u if u is not None else torch.zeros(1, steps, nu),
        "D": d if d is not None else torch.zeros(1, steps, nd),
        "Y": torch.zeros(1, steps, ny),
    }
    return system(data)["yn"]


# ── The update really is the RC network ──────────────────────────

def test_the_step_is_the_rc_equation_to_the_last_digit():
    """Hand-compute one step of `z + a_m(m - z) - a_o z + g` and match it.

    The mass seeds at the window mean, which for a flat history is the zone
    itself, so the mass term drops out and what is left is the conduction
    loss plus the learned forcing.
    """
    block = make_block()
    a_o, a_m, _ = block.rates()
    x = window(block, 3.0, 0.5, -0.25)
    with torch.no_grad():
        forcing = block.gain(torch.tensor([[0.5, -0.25]]))
        expected = a_m * (3.0 - 3.0) - a_o * 3.0 + forcing
        assert torch.allclose(block(x), expected, atol=1e-6)


def test_the_reported_time_constants_are_the_updates_eigenvalues():
    """`mode_taus` is an algebraic shortcut; check it against the matrix."""
    block = make_block()
    a_o, a_m, a_c = (r[0] for r in block.rates())
    update = torch.tensor([[1 - a_m - a_o, a_m], [a_c, 1 - a_c]])
    eig = torch.linalg.eigvals(update).real.sort(descending=True).values
    fast, slow = mode_taus(a_o, a_m, a_c)
    # tau = 1/lambda for the continuous rate lambda = 1 - eigenvalue.
    assert torch.allclose(1 / (1 - eig[0]), slow, rtol=1e-4)
    assert torch.allclose(1 / (1 - eig[1]), fast, rtol=1e-4)


def test_the_default_prior_is_an_ordinary_building():
    """0.6 h of air and 13.4 h of mass, at a 5-minute timestep. If this
    moves, every default 2r2c run's storage moved with it."""
    taus = make_block().describe()["time_constants_h"]["y0"]
    assert taus["fast"] == pytest.approx(0.60, abs=0.02)
    assert taus["slow"] == pytest.approx(13.40, abs=0.05)


def test_the_prior_is_what_the_config_asked_for():
    slow_building = make_block(tau_hours=(8.0, 1.0, 4.0))
    fast_building = make_block(tau_hours=(0.5, 1.0, 4.0))
    assert (slow_building.describe()["time_constants_h"]["y0"]["slow"]
            > 4 * fast_building.describe()["time_constants_h"]["y0"]["slow"])


# ── Stability is structural, not empirical ───────────────────────

@pytest.mark.parametrize("theta", [-20.0, 0.0, 20.0])
def test_no_rate_can_leave_the_stable_range(theta):
    """Rates are capped at 1/MIN_TAU_STEPS, so the three of them sum below 2
    — the condition for the update's eigenvalues to sit inside the unit
    circle. The cap is a supremum the parameterization approaches but never
    passes; float32 rounds onto it, which is why this is `<=`."""
    block = make_block()
    with torch.no_grad():
        for name in block._theta:
            block._theta[name].fill_(theta)
    rates = torch.stack([r[0] for r in block.rates()])
    assert (rates > 0).all() and (rates <= 1 / MIN_TAU_STEPS).all()
    assert rates.sum() < 2.0


def test_a_week_of_adversarial_input_does_not_diverge():
    """The forcing never reads the state, so no input can make the loop
    run away — which is what a DPC policy searching for exploits needs."""
    torch.manual_seed(0)
    system = get_dynamics(make_cfg(), MAPPING)
    for block in system.nodes[0].callable.fy.blocks:
        with torch.no_grad():
            for name in block._theta:            # fastest allowed rates
                block._theta[name].fill_(-20.0)
            for p in block.gain.parameters():
                p.normal_(0, 1.0)
    steps = int(SPH) * 24 * 7
    out = rollout(system, 3.0, steps,
                  u=torch.randn(1, steps, 3) * 20,
                  d=torch.randn(1, steps, 2) * 20)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1e4


# ── The hidden mass is the whole point ───────────────────────────

def test_a_head_start_outlives_the_history_window():
    """A difference in initial temperature has to survive far longer than
    `in_steps`, or the family has bought nothing over a windowed MLP."""
    system = get_dynamics(make_cfg(), MAPPING)
    steps = int(SPH) * 24
    warm = rollout(system, 2.0, steps)[0, :, 0]
    cold = rollout(system, 0.0, steps)[0, :, 0]
    surviving = ((warm - cold) / (warm - cold)[0]).detach()
    assert surviving[IN_STEPS] > 0.9              # the window is not the memory
    assert surviving[int(SPH) * 6] > 0.35         # still there six hours later
    assert surviving[-1] < surviving[int(SPH) * 6]  # and it is decaying


def test_the_mass_carries_state_across_calls_until_reset():
    """Warm calls advance the mass; reset puts the block back to cold.

    The history has to be moving for this to show: a zone that has sat still
    hands the mass its own value, and a mass already at equilibrium with the
    zone stays there however many times it is stepped.
    """
    block = make_block()
    ramp = torch.arange(IN_STEPS, dtype=torch.float32).unsqueeze(0)
    x = torch.cat([ramp, torch.zeros(1, IN_STEPS * 2)], dim=-1)
    block.reset()
    first = block(x.clone())
    second = block(x.clone())
    assert not torch.allclose(first, second)      # the mass moved
    block.reset()
    assert torch.allclose(block(x.clone()), first)


def test_the_gradient_penalty_can_still_differentiate_the_block():
    """The sampled gradient penalty runs jacrev under vmap, where there is
    no batch dimension and no state to keep."""
    system = get_dynamics(make_cfg(), MAPPING)
    raw = system.nodes[0].callable
    x = torch.randn(4, IN_STEPS * 7)
    jac = torch.vmap(torch.func.jacrev(raw.compute_output, argnums=(0, 1, 2)))(
        x[:, :IN_STEPS * 2], x[:, IN_STEPS * 2:IN_STEPS * 5], x[:, IN_STEPS * 5:])
    assert all(torch.isfinite(j).all() for j in jac)
    assert jac[0].shape == (4, 2, IN_STEPS * 2)


# ── The prior stays put unless asked ─────────────────────────────

def test_the_time_constants_are_frozen_by_default():
    """A training rollout cannot identify them (see the module docstring),
    so they must not quietly drift away from what the config asserted."""
    block = make_block()
    assert all(not p.requires_grad for p in block._theta.parameters())
    assert any(p.requires_grad for p in block.gain.parameters())


def test_the_time_constants_train_when_asked():
    block = make_block(train_taus=True)
    assert all(p.requires_grad for p in block._theta.parameters())


def test_the_forcing_starts_at_zero():
    """Training opens on a pure free-decay building: an MLP's O(1) initial
    output would otherwise swamp a normalized temperature step."""
    block = make_block()
    assert torch.equal(block.gain(torch.randn(4, 2)), torch.zeros(4, 1))


# ── Zones are independent ────────────────────────────────────────

def test_one_zones_input_cannot_move_another_zones_state():
    torch.manual_seed(0)
    system = get_dynamics(make_cfg(), MAPPING, zone_spec=SPLIT)
    fy = system.nodes[0].callable.fy
    with torch.no_grad():
        for block in fy.blocks:
            for p in block.gain.parameters():
                p.normal_(0, 1.0)

    x = torch.randn(2, IN_STEPS * 7)
    fy.reset()
    before = fy(x.clone())
    nudged = x.clone()
    nudged[:, IN_STEPS * 2 + 0::3] += 5.0        # the north setpoint alone
    fy.reset()
    moved = (fy(nudged) - before).abs().max(0).values
    assert moved[0] > 0.1 and moved[1] == 0.0


def test_outputs_come_back_in_the_mappings_channel_order():
    """Zones may be declared in any order and hold their states in any
    order; the loss and the feedback loop both assume nm_mapping's."""
    reordered = {"south": {"Y": ["south T"], "U": ["south sp"]},
                 "north": {"Y": ["north T"], "U": ["north sp"]}}
    system = get_dynamics(make_cfg(), MAPPING, zone_spec=reordered)
    fy = system.nodes[0].callable.fy
    assert fy._to_y_order.tolist() == [1, 0]
    assert fy(torch.randn(2, IN_STEPS * 7)).shape == (2, 2)


def test_each_zone_holds_its_own_parameters_and_its_own_names():
    system = get_dynamics(make_cfg(), MAPPING, zone_spec=SPLIT)
    fy = system.nodes[0].callable.fy
    assert fy.zone_names == ["north", "south"]
    assert [b.channel_names for b in fy.blocks] == [("north T",), ("south T",)]
    north, south = (dict(b.named_parameters()) for b in fy.blocks)
    assert not torch.equal(north["gain.linear.0.weight"],
                           south["gain.linear.0.weight"])


def test_reset_reaches_every_zone():
    system = get_dynamics(make_cfg(), MAPPING, zone_spec=SPLIT)
    fy = system.nodes[0].callable.fy
    fy(torch.randn(2, IN_STEPS * 7))
    assert all(b._mass is not None for b in fy.blocks)
    fy.reset()
    assert all(b._mass is None for b in fy.blocks)


def test_per_zone_wraps_any_family_the_flag_allows():
    """`per_zone` is a split, not a family: an MLP gets the same treatment."""
    system = get_dynamics(make_cfg(model="mlp", per_zone=True), MAPPING,
                          zone_spec=SPLIT)
    fy = system.nodes[0].callable.fy
    assert isinstance(fy, PerZoneBlock)
    assert fy(torch.randn(2, IN_STEPS * 7)).shape == (2, 2)


# ── The ways a wrong model is refused ────────────────────────────

@pytest.mark.parametrize("over,message", [
    ({"residual": False}, "requires residual"),
    ({"steps_per_hour": None}, "steps_per_hour"),
])
def test_a_2r2c_that_would_be_wrong_is_refused(over, message):
    with pytest.raises(ValueError, match=message):
        get_dynamics(make_cfg(**over), MAPPING)


def test_a_head_that_will_not_add_the_state_back_is_refused():
    """The block emits an increment; a head that does not restore the state
    would read that increment as an absolute temperature."""
    stack = HeadStack([RegressionHead(), BinaryHead()], MAPPING["Y"])
    with pytest.raises(ValueError, match="residual-compatible"):
        get_dynamics(make_cfg(), MAPPING, head_stack=stack)


def test_per_zone_is_refused_where_it_would_do_nothing():
    with pytest.raises(ValueError, match="already per-zone"):
        make_cfg(per_zone=True, train_data_path="x.csv", dev_data_path="x.csv",
                 test_data_path="x.csv").validate()


def test_a_coarse_timestep_cannot_ask_for_an_impossible_prior():
    """At an hourly timestep the 1 h zone<->mass default is below the
    two-step floor; it clamps rather than producing a NaN."""
    hourly = RCBlock(1, 1, 1, IN_STEPS, steps_per_hour=1.0,
                     hidden_size=8, num_layers=1)
    taus = hourly.describe()["time_constants_h"]["y0"]
    assert math.isfinite(taus["fast"]) and math.isfinite(taus["slow"])
    assert 0 < taus["fast"] < taus["slow"]
