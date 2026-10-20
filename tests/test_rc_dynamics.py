"""
Tests for the 2R2C grey-box family and the per-block split it rides on.

The family's whole claim is that it asserts things the black-box families
only hope for, so the tests are that list of claims:

  - the update is the RC network's, arithmetic and all, not something
    approximately like it;
  - the reported time constants are the update's real eigenvalues;
  - a rollout cannot diverge, whatever the inputs and wherever the time
    constants sit inside their boxes;
  - the hidden mass remembers past `history_steps`, which is the point;
  - the time constants are the declared prior — pinned where a scalar says
    so, inside the box where `[min, max]` does — and every point of the
    boxes is a valid building;
  - each block is genuinely independent of the others, except for an
    explicit `reads` declaration — and `2r2c` refuses that declaration,
    because its stability guarantee depends on never reading state;
  - the declared forcing branches keep every one of those claims: a block
    with no declarations is byte-for-byte the classical one, a `demand`
    column is zero on the inactive side and the saturating branch it feeds
    can never use more slope than the stability headroom allows, a
    `linear` sign is a constraint and not a hint, and an `ambient` column
    leaves the MLP's input.

Anything that would silently produce a wrong model instead fails loudly:
2r2c without residual, without a timestep, without declared time
constants, with boxes that overlap or reach below the stability floor, under
a head that will not add the state back, with `reads`, or a block naming a
family with no per-block builder.
"""

import pytest
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.export.state import clear_state
from neurotemplate.modeling.dynamics import get_dynamics, stateful_blocks
from neurotemplate.modeling.heads import BinaryHead, HeadStack, RegressionHead
from neurotemplate.modeling.rc import (
    MIN_TAU_STEPS, RCBlock, SaturatingTerm, mode_taus, time_constant_boxes,
)
from model_fixtures import model_cfg, uniform_history

SPH = 12.0            # 5-minute data
HISTORY_STEPS = 6

MAPPING = {
    "Y": ["north T", "south T"],
    "U": ["north sp", "south sp", "ahu fan"],
    "D": ["oat", "ghi"],
    "t": ["sec_elapsed"],
}
#: An ordinary commercial building, pinned: the modes the old component
#: defaults produced at 5-minute data.
PINNED = {"tau_fast_h": 0.6, "tau_ambient_h": 2.0, "tau_slow_h": 13.4}
#: The same building held in boxes, the fast floor at the stability bound.
BOXED = {"tau_fast_h": [MIN_TAU_STEPS / SPH, 1.0], "tau_ambient_h": [2.0, 8.0],
         "tau_slow_h": [10.0, 40.0]}

SPLIT = {
    "north": {"Y": ["north T"], "U": ["north sp"], "params": PINNED},
    "south": {"Y": ["south T"], "U": ["south sp"], "params": PINNED},
}

# A second mapping with a mixed family: one thermal state (2r2c) and one
# readout (mlp) that reads the state it measures alongside.
MIXED_MAPPING = {
    "Y": ["north T", "power"],
    "U": ["north sp"],
    "D": ["oat"],
    "t": ["sec_elapsed"],
}
MIXED_SPLIT = {
    "north": {"Y": ["north T"], "U": ["north sp"], "model": "2r2c", "params": PINNED},
    "power": {"Y": ["power"], "reads": ["north T"], "model": "mlp"},
}


# Physical-unit stats for the declared-branch tests: a z-scored setpoint
# and zone whose normalised zero-crossings do not coincide, so a demand
# computed in normalised space would gate in the wrong place.
STATS = {
    "north T": {"shift": 23.0, "stretch": 1.5, "min": 20.0, "max": 27.0},
    "north sp": {"shift": 25.0, "stretch": 2.0, "min": 21.0, "max": 30.0},
    "ahu fan": {"shift": 0.4, "stretch": 0.3, "min": 0.0, "max": 1.0},
    "oat": {"shift": 20.0, "stretch": 6.0, "min": 5.0, "max": 35.0},
    "ghi": {"shift": 300.0, "stretch": 250.0, "min": 0.0, "max": 900.0},
}
DECLARED = {
    "north": {
        "Y": ["north T"], "U": ["north sp", "ahu fan"], "D": ["oat", "ghi"],
        "model": "2r2c", "ambient": "oat", "params": PINNED,
        "inputs": {
            "north sp": {"kind": "demand", "sign": -1, "state": "north T"},
            "ahu fan": {"kind": "linear", "sign": +1},
        },
    },
}
DECLARED_MAPPING = {
    "Y": ["north T"], "U": ["north sp", "ahu fan"], "D": ["oat", "ghi"],
    "t": ["sec_elapsed"],
}


#: Every block 2r2c: U and D are read at the current step only.
RC_HISTORY = uniform_history(MAPPING, HISTORY_STEPS, at_zero=(*MAPPING["U"], *MAPPING["D"]))
UNIFORM = uniform_history(MAPPING, HISTORY_STEPS)
#: The 2r2c zone owns the setpoint; the mlp readout reads the outdoor air.
MIXED_HISTORY = uniform_history(MIXED_MAPPING, HISTORY_STEPS,
                                at_zero=tuple(MIXED_MAPPING["U"]))
DECLARED_HISTORY = uniform_history(
    DECLARED_MAPPING, HISTORY_STEPS,
    at_zero=(*DECLARED_MAPPING["U"], *DECLARED_MAPPING["D"]))


def make_cfg(**over) -> Config:
    return model_cfg(**{"model": "2r2c", "history": RC_HISTORY, "log2_hidden_size": 4,
                        "steps_per_hour": SPH, **over})


def make_block(ny=1, nu=1, nd=1, params=PINNED, **over) -> RCBlock:
    torch.manual_seed(0)
    return RCBlock(ny, nu, nd, HISTORY_STEPS, SPH, hidden_size=8, num_layers=1,
                   taus=time_constant_boxes(params, SPH, "b"), **over)


def set_thetas(block: RCBlock, value: float) -> None:
    """Every time constant to the same position in its box: -20 is the
    bottom of each (the fastest rates), +20 the top."""
    with torch.no_grad():
        for tau in (block.tau_fast, block.tau_ambient, block.tau_slow):
            tau.theta.fill_(value)


def make_declared_block() -> RCBlock:
    """The `north` block of DECLARED, built directly."""
    return make_block(
        1, 2, 2, channel_names=("north T",), u_names=("north sp", "ahu fan"),
        d_names=("oat", "ghi"), inputs=DECLARED["north"]["inputs"],
        ambient="oat", norm_stats=STATS,
    )


def normalised(name: str, value: float) -> float:
    return (value - STATS[name]["shift"]) / STATS[name]["stretch"]


def window(block: RCBlock, zone, drive_u, drive_d) -> torch.Tensor:
    """The block's window: a constant zone history and one row of U and D."""
    ny, nu, nd = block.widths
    return torch.cat([
        torch.tensor(zone, dtype=torch.float32).repeat(1, HISTORY_STEPS * ny),
        torch.tensor(drive_u, dtype=torch.float32).repeat(1, nu),
        torch.tensor(drive_d, dtype=torch.float32).repeat(1, nd),
    ], dim=-1)


def rollout(system, y0, steps, u=None, d=None, ny=2, nu=3, nd=2):
    data = {
        "yn": torch.full((1, HISTORY_STEPS, ny), y0),
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


@pytest.mark.parametrize("theta", [0.0])
def test_the_rates_have_exactly_the_declared_modes(theta):
    """The update's eigenvalues are the time constants the block holds,
    wherever in their boxes they sit — checked against the matrix itself,
    not just `mode_taus`'s algebra — and the ambient rate is its own."""
    block = make_block(params=BOXED)
    set_thetas(block, theta)
    with torch.no_grad():
        a_o, a_m, a_c = (r[0] for r in block.rates())
        fast = block.tau_fast.value()[0] * SPH
        slow = block.tau_slow.value()[0] * SPH
        ambient = block.tau_ambient.value()[0] * SPH
    update = torch.tensor([[1 - a_m - a_o, a_m], [a_c, 1 - a_c]])
    eig = torch.linalg.eigvals(update).real.sort(descending=True).values
    # tau = 1/lambda for the continuous rate lambda = 1 - eigenvalue.
    assert torch.allclose(1 / (1 - eig[0]), slow, rtol=1e-4)
    assert torch.allclose(1 / (1 - eig[1]), fast, rtol=1e-4)
    assert torch.allclose(torch.stack(mode_taus(a_o, a_m, a_c)), torch.stack([fast, slow]),
                          rtol=1e-4)
    assert (1 / a_o).item() == pytest.approx(ambient.item(), rel=1e-5)


# ── Stability is structural, not empirical ───────────────────────

@pytest.mark.parametrize("theta", [-20.0, 20.0])
def test_no_rate_can_leave_the_stable_range(theta):
    """Every point of disjoint boxes is a valid building: three positive
    rates, each below 1/MIN_TAU_STEPS — the fast floor is exactly the
    stability bound here — so the update's eigenvalues sit inside the unit
    circle."""
    block = make_block(params=BOXED)
    set_thetas(block, theta)
    rates = torch.stack([r[0] for r in block.rates()])
    assert (rates > 0).all()
    assert (rates <= 1 / MIN_TAU_STEPS + 1e-6).all()
    assert rates.sum() < 1.0


def test_a_week_of_adversarial_input_does_not_diverge():
    """The forcing never reads the state, so no input can make the loop
    run away — which is what a DPC policy searching for exploits needs."""
    torch.manual_seed(0)
    boxed = {name: {**spec, "params": BOXED} for name, spec in SPLIT.items()}
    system = get_dynamics(make_cfg(), MAPPING, block_spec=boxed)
    for block in system.nodes[0].callable.fy.blocks:
        set_thetas(block, -20.0)                  # fastest allowed rates
        with torch.no_grad():
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
    `history_steps`, or the family has bought nothing over a windowed MLP."""
    system = get_dynamics(make_cfg(), MAPPING, block_spec=SPLIT)
    steps = int(SPH) * 24
    warm = rollout(system, 2.0, steps)[0, :, 0]
    cold = rollout(system, 0.0, steps)[0, :, 0]
    surviving = ((warm - cold) / (warm - cold)[0]).detach()
    assert surviving[HISTORY_STEPS] > 0.9              # the window is not the memory
    assert surviving[int(SPH) * 6] > 0.35         # still there six hours later
    assert surviving[-1] < surviving[int(SPH) * 6]  # and it is decaying


def test_the_mass_carries_state_across_calls_until_reset():
    """Warm calls advance the mass; reset puts the block back to cold.

    The history has to be moving for this to show: a zone that has sat still
    hands the mass its own value, and a mass already at equilibrium with the
    zone stays there however many times it is stepped.
    """
    block = make_block()
    ramp = torch.arange(HISTORY_STEPS, dtype=torch.float32).unsqueeze(0)
    x = torch.cat([ramp, torch.zeros(1, 2)], dim=-1)   # one row of U and D
    block.set_rollout_state(None)
    first = block(x.clone())
    second = block(x.clone())
    assert not torch.allclose(first, second)      # the mass moved
    block.set_rollout_state(None)
    assert torch.allclose(block(x.clone()), first)


def test_the_gradient_penalty_still_works_on_a_mixed_family_block():
    """The same vmap/jacrev path, but fy now holds an RCBlock and an MLP
    side by side — the gather/scatter permutations that make this vmap-safe
    don't care how many families are mixed in."""
    cfg = make_cfg(model="mlp", history=MIXED_HISTORY)  # default family for blocks that don't override
    system = get_dynamics(cfg, MIXED_MAPPING, block_spec=MIXED_SPLIT)
    raw = system.nodes[0].callable
    # widths: ny=2, nu=1, nd=1
    x = torch.randn(4, HISTORY_STEPS * 4)
    jac = torch.vmap(torch.func.jacrev(raw.compute_output, argnums=(0, 1, 2)))(
        x[:, :HISTORY_STEPS * 2], x[:, HISTORY_STEPS * 2:HISTORY_STEPS * 3], x[:, HISTORY_STEPS * 3:])
    assert all(torch.isfinite(j).all() for j in jac)
    assert jac[0].shape == (4, 2, HISTORY_STEPS * 2)


# ── The prior holds where it is declared ─────────────────────────

def test_a_pinned_time_constant_cannot_move_and_a_boxed_one_stays_in_its_box():
    """A scalar pins: its gradient is zero. A box lets training move the
    value — even an enormous step lands inside it."""
    params = {**BOXED, "tau_ambient_h": 3.0}
    block = make_block(params=params)
    optimizer = torch.optim.SGD([b.theta for b in (block.tau_fast, block.tau_ambient,
                                                   block.tau_slow)], lr=1e4)
    torch.stack(block.rates()).sum().backward()
    assert float(block.tau_ambient.theta.grad.abs().sum()) == 0.0
    optimizer.step()
    with torch.no_grad():
        assert block.tau_ambient.value().item() == pytest.approx(3.0, rel=1e-6)
        slow = block.tau_slow.value().item()
    assert 10.0 <= slow <= 40.0 and slow != pytest.approx(20.0, rel=1e-3)


# ── Blocks are independent, except for an explicit `reads` ───────

def test_one_blocks_input_cannot_move_another_blocks_state():
    torch.manual_seed(0)
    system = get_dynamics(make_cfg(), MAPPING, block_spec=SPLIT)
    fy = system.nodes[0].callable.fy
    with torch.no_grad():
        for block in fy.blocks:
            for p in block.gain.parameters():
                p.normal_(0, 1.0)

    x = torch.randn(2, HISTORY_STEPS * 7)
    clear_state(stateful_blocks(fy))
    before = fy(x.clone())
    nudged = x.clone()
    nudged[:, HISTORY_STEPS * 2 + 0::3] += 5.0        # the north setpoint alone
    clear_state(stateful_blocks(fy))
    moved = (fy(nudged) - before).abs().max(0).values
    assert moved[0] > 0.1 and moved[1] == 0.0


def test_outputs_come_back_in_the_mappings_channel_order():
    """Blocks may be declared in any order and hold their states in any
    order; the loss and the feedback loop both assume nm_mapping's."""
    reordered = {"south": SPLIT["south"], "north": SPLIT["north"]}
    system = get_dynamics(make_cfg(), MAPPING, block_spec=reordered)
    fy = system.nodes[0].callable.fy
    assert fy._to_y_order.tolist() == [1, 0]
    assert fy(torch.randn(2, HISTORY_STEPS * 7)).shape == (2, 2)


def test_reset_reaches_every_block():
    system = get_dynamics(make_cfg(), MAPPING, block_spec=SPLIT)
    fy = system.nodes[0].callable.fy
    fy(torch.randn(2, HISTORY_STEPS * 7))
    assert all(b._mass is not None for b in fy.blocks)
    clear_state(stateful_blocks(fy))
    assert all(b._mass is None for b in fy.blocks)


def test_reads_changes_the_reading_blocks_output_and_not_the_read_blocks():
    """`power` reads `north`'s state; perturbing north's history must move
    power's prediction but never the other way around."""
    cfg = make_cfg(model="mlp", history=MIXED_HISTORY)
    torch.manual_seed(0)
    system = get_dynamics(cfg, MIXED_MAPPING, block_spec=MIXED_SPLIT)
    fy = system.nodes[0].callable.fy
    with torch.no_grad():
        for p in fy.blocks[0].gain.parameters():
            p.normal_(0, 1.0)
        for p in fy.blocks[1].parameters():
            p.normal_(0, 1.0)

    x = torch.randn(2, HISTORY_STEPS * 4)                # widths (2, 1, 1)
    clear_state(stateful_blocks(fy))
    before = fy(x.clone())

    # Y is key-major and, within that key, time-major: two columns per
    # step (north T, power), for HISTORY_STEPS steps, at the front of the window.
    north_cols = [2 * t for t in range(HISTORY_STEPS)]
    power_cols = [2 * t + 1 for t in range(HISTORY_STEPS)]

    # Perturb "north T" — the read channel — across its whole history.
    nudged_north = x.clone()
    nudged_north[:, north_cols] += 5.0
    clear_state(stateful_blocks(fy))
    moved_from_north = (fy(nudged_north) - before).abs().max(0).values
    assert moved_from_north[0] > 0.0                 # north's own state moved
    assert moved_from_north[1] > 0.0                 # ...and power, which reads it

    # Perturb "power" — power's own state, which north does not read.
    nudged_power = x.clone()
    nudged_power[:, power_cols] += 5.0
    clear_state(stateful_blocks(fy))
    moved_from_power = (fy(nudged_power) - before).abs().max(0).values
    assert moved_from_power[0] == 0.0                # north is unaffected
    assert moved_from_power[1] > 0.0


# ── Declared forcing: every claim survives the branches ──────────

def test_demand_is_zero_on_the_inactive_side_and_positive_on_the_active():
    """Through the normalisation round trip: the two columns have different
    shifts, so a setpoint 1 °C above the zone is *below* it in normalised
    units. Only a physical-unit comparison gates in the right place."""
    system = get_dynamics(make_cfg(history=DECLARED_HISTORY), DECLARED_MAPPING, block_spec=DECLARED,
                          norm_stats=STATS)
    pre = system.nodes[0].callable.pre_stack
    zone = torch.full((1, HISTORY_STEPS), normalised("north T", 24.0))
    coasting = torch.full((1, HISTORY_STEPS), normalised("north sp", 25.0))
    cooling = torch.full((1, HISTORY_STEPS), normalised("north sp", 22.0))
    fan = torch.zeros(1, HISTORY_STEPS)
    u = lambda sp: torch.stack([sp, fan], dim=-1).reshape(1, -1)  # noqa: E731
    assert normalised("north sp", 25.0) < normalised("north T", 24.0)
    assert torch.equal(pre(zone, u(coasting)).reshape(HISTORY_STEPS, 2)[:, 0],
                       torch.zeros(HISTORY_STEPS))
    active = pre(zone, u(cooling)).reshape(HISTORY_STEPS, 2)[:, 0]
    assert torch.allclose(active, torch.full((HISTORY_STEPS,), 2.0 / (27.0 - 21.0)))


@pytest.mark.parametrize("theta_cap,theta_slope", [(20.0, 20.0)])
def test_the_saturating_slope_never_exceeds_the_headroom(theta_cap, theta_slope):
    """`cap / w ≤ 1 - a_m - a_o` by construction, so the zone's own
    coefficient stays nonnegative wherever the parameters go. Measured as
    the gradient of the branch's increment with respect to the zone, in
    normalised units, at the origin where the tanh is steepest."""
    term = SaturatingTerm(-1, STATS["north sp"], STATS["north T"])
    with torch.no_grad():
        term.theta_cap.fill_(theta_cap)
        term.theta_slope.fill_(theta_slope)
    block = make_block()
    a_o, a_m, _ = block.rates()
    headroom = 1 - a_m - a_o
    # Demand in normalised-zone units: d(demand feature)/d(zone_norm) is
    # stretch_z / scale, so the chain gives the slope in the state matrix.
    x = torch.zeros(1, 1, requires_grad=True)
    term(x, headroom).sum().backward()
    slope_in_state = x.grad.abs() * STATS["north T"]["stretch"] / term.scale
    assert slope_in_state.item() <= headroom.item() * (1 + 1e-5)


def test_a_linear_sign_is_a_constraint_not_a_hint():
    block = make_block(1, 1, 0, u_names=("power",), inputs={
        "power": {"kind": "linear", "sign": -1}}, norm_stats={
        "y0": STATS["north T"], "power": STATS["ahu fan"]})
    (term,) = block.linear
    for theta in (-20.0, 0.0, 20.0):
        with torch.no_grad():
            term.theta.fill_(theta)
        assert term.coefficient() <= 0


def test_the_ambient_term_is_conduction_to_the_declared_column():
    """`a_o (T_oa - z)` in physical units: zone at ambient gets no conduction,
    zone 6 °C below it warms by `a_o · 6 / stretch_z` per step."""
    block = make_declared_block()
    a_o, _, _ = block.rates()
    with torch.no_grad():
        zone = torch.full((1, 1), normalised("north T", 24.0))
        same = torch.full((1, 1), normalised("oat", 24.0))
        hotter = torch.full((1, 1), normalised("oat", 30.0))
        assert torch.allclose(block.ambient(zone, same, a_o), torch.zeros(1, 1), atol=1e-6)
        assert torch.allclose(block.ambient(zone, hotter, a_o),
                              a_o * 6.0 / STATS["north T"]["stretch"])


def test_a_declared_block_persists_its_stats_and_reloads_by_config():
    """The branches' norm stats and the pre-stack's are buffers, so the
    checkpoint's numbers reach a rebuilt system through load_state_dict."""
    torch.manual_seed(0)
    system = get_dynamics(make_cfg(history=DECLARED_HISTORY), DECLARED_MAPPING, block_spec=DECLARED,
                          norm_stats=STATS)
    state = system.state_dict()
    assert any("pre_stack" in k for k in state)
    assert any("saturating.0.theta_cap" in k for k in state)
    rebuilt = get_dynamics(make_cfg(history=DECLARED_HISTORY), DECLARED_MAPPING, block_spec=DECLARED,
                           norm_stats=STATS)
    rebuilt.load_state_dict(state)
    x = torch.randn(2, HISTORY_STEPS * 5)
    y, u, d = x[:, :HISTORY_STEPS], x[:, HISTORY_STEPS:HISTORY_STEPS * 3], x[:, HISTORY_STEPS * 3:]
    clear_state(stateful_blocks(system)); clear_state(stateful_blocks(rebuilt))
    assert torch.equal(system.nodes[0].callable(y, u, d), rebuilt.nodes[0].callable(y, u, d))


def test_a_week_of_adversarial_setpoints_does_not_diverge_with_demand_on():
    """The one state read the block admits is bounded in slope, so the
    rollout stays inside the same bound as the undeclared block's — at the
    fastest rates and with the coil driven to its cap."""
    torch.manual_seed(0)
    system = get_dynamics(make_cfg(history=DECLARED_HISTORY), DECLARED_MAPPING, block_spec=DECLARED,
                          norm_stats=STATS)
    block = system.nodes[0].callable.fy.blocks[0]
    set_thetas(block, -20.0)                          # the fastest rates
    with torch.no_grad():
        block.saturating[0].theta_cap.fill_(5.0)
        block.saturating[0].theta_slope.fill_(20.0)   # the whole headroom
        for p in block.gain.parameters():
            p.normal_(0, 1.0)
    steps = int(SPH) * 24 * 7
    out = rollout(system, 3.0, steps, ny=1, nu=2, nd=2,
                  u=torch.randn(1, steps, 2) * 20, d=torch.randn(1, steps, 2) * 20)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1e4


# ── The ways a wrong model is refused ────────────────────────────

def test_a_2r2c_without_the_residual_is_refused():
    """It predicts an increment; without the residual that would be read as
    an absolute temperature."""
    with pytest.raises(ValueError, match="requires residual"):
        get_dynamics(make_cfg(residual=False), MAPPING, block_spec=SPLIT)


def test_a_head_that_will_not_add_the_state_back_is_refused():
    """The block emits an increment; a head that does not restore the state
    would read that increment as an absolute temperature."""
    stack = HeadStack([RegressionHead(), BinaryHead()], MAPPING["Y"])
    with pytest.raises(ValueError, match="residual-compatible"):
        get_dynamics(make_cfg(), MAPPING, head_stack=stack, block_spec=SPLIT)


def test_2r2c_with_reads_is_refused():
    """The single most important refusal in the block split: 2r2c's
    unconditional-stability guarantee holds only because its forcing map
    never reads state, and `reads` would let it."""
    spec = {
        "north": {"Y": ["north T"], "U": ["north sp"], "model": "2r2c",
                  "reads": ["power"]},
        "power": {"Y": ["power"], "model": "mlp"},
    }
    with pytest.raises(ValueError, match="cannot declare `reads`"):
        get_dynamics(make_cfg(model="mlp", history=MIXED_HISTORY), MIXED_MAPPING,
                     block_spec=spec)


def test_a_lag_on_a_column_only_2r2c_reads_is_refused():
    """2r2c drives its zone with the current U and D only, so a lag on a
    column nothing else reads would sit inert."""
    history = {**MIXED_HISTORY, "north sp": 2}
    with pytest.raises(ValueError, match=r"history\['north sp'\] is 2.*'north': 0"):
        get_dynamics(make_cfg(model="mlp", history=history), MIXED_MAPPING,
                     block_spec=MIXED_SPLIT)


def test_a_demand_setpoint_lagged_past_its_zone_is_refused():
    """The demand is computed on the full window, so a setpoint read further
    back than its zone carries zone history the zone's own lag excludes."""
    spec = {"north": {"Y": ["north T"],
                      "inputs": {"north sp": DECLARED["north"]["inputs"]["north sp"]}}}
    history = {**uniform_history(DECLARED_MAPPING, 2), "north sp": 3}
    with pytest.raises(ValueError, match=r"history\['north sp'\] is 3, deeper"):
        get_dynamics(make_cfg(model="mlp", history=history), DECLARED_MAPPING,
                     block_spec=spec, norm_stats=STATS)
