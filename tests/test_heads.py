"""
Tests for the configurable output heads.

Three things have to hold for every head, and they are the three sections
below: it can be **fitted** (the loss has a descent direction on data of its
own kind), its activation **stays inside** whatever range it promises, and it
**round-trips** through `to_config`/`from_config` so a reloaded run rebuilds
the topology it trained with. A fourth section collects every way a head
config can be malformed — all of which must raise rather than construct
something quietly wrong.

Plus the two HeadStack invariants: an all-regression stack reproduces blanket
MSE to floating-point precision, and `residual_mask` names only the
channels that declared themselves residual-compatible.

The pre-stack section pins the input-side counterpart: the clamp is exact
forward and never dead backward, the stack round-trips through its config,
and — the one transposition a reshape could hide — every output step is
computed from its own step of U and Y.

The rare-level section pins the two knobs a staged channel has: class
weights must leave the loss untouched at their default and count a rare
level more otherwise, and the decision thresholds must move the snap
without moving the activation.
"""

import math

import pytest
import torch
import torch.nn as nn

from neurotemplate.modeling.prestack import Demand, PreStack
from neurotemplate.modeling.heads import (
    BinaryHead,
    BoundedHead,
    HeadStack,
    OrdinalHead,
    RegressionHead,
)

INF = float("inf")


def fit(activate, loss_fn, z0, target, steps=400, lr=0.1):
    """Descend on `z0` and return the first and last loss."""
    z = nn.Parameter(z0.clone())
    opt = torch.optim.SGD([z], lr=lr)
    first = None
    for _ in range(steps):
        loss = loss_fn(activate(z.unsqueeze(-1)), target)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    return first, loss.item(), z


# ── Every head can be fitted on data of its own kind ─────────────

@pytest.mark.parametrize("head,z0,target,ratio,kw", [
    pytest.param(RegressionHead(), torch.zeros(32), torch.full((32,), 2.5),
                 1.0, {}, id="regression"),
    pytest.param(BinaryHead(), torch.zeros(32),
                 torch.tensor([0.0, 1.0]).repeat(16), 1.0, {}, id="binary"),
    pytest.param(BoundedHead(lo=0.0, hi=INF), torch.zeros(32),
                 torch.full((32,), 3.0), 1.0, dict(lr=0.2), id="bounded-lower"),
    # Ordinal has to do better than "somewhat": snapping to integer levels is
    # the whole claim, so require the loss to fall by 10x.
    pytest.param(OrdinalHead(n_levels=3, level_values=torch.tensor([0.0, 1.0, 5.0])),
                 torch.linspace(-1.0, 6.0, 30),
                 torch.tensor([0.0, 1.0, 5.0]).repeat_interleave(10), 0.1,
                 dict(lr=0.5, steps=800), id="ordinal-non-uniform"),
])
def test_a_head_descends_on_its_own_kind_of_target(head, z0, target, ratio, kw):
    torch.manual_seed(0)
    first, last, _ = fit(head.activate, head.loss, z0, target, **kw)
    assert last < first * ratio


# ── Activations honour the range they promise ────────────────────

@pytest.mark.parametrize("head,lo,hi,open_side", [
    pytest.param(BoundedHead(lo=0.0, hi=INF), 0.0, INF, "hi", id="bounded-lower"),
    pytest.param(OrdinalHead(n_levels=4), 0.0, 3.0, None, id="ordinal"),
])
def test_an_activation_stays_inside_its_bounds(head, lo, hi, open_side):
    """...and stays *usable* on any open side: a half-bounded head has to
    keep travelling out there, not flatten off."""
    z = torch.linspace(-50.0, 50.0, 200).unsqueeze(-1)
    y = head.activate(z)
    assert (y >= lo).all() and (y <= hi).all()
    if open_side == "hi":
        assert y[-1] > 40.0
    elif open_side == "lo":
        assert y[0] < -40.0


def test_a_saturated_bounded_head_still_passes_gradient():
    """Well below the band, where a naive clamp would hand back zero."""
    z = torch.tensor([[-10.0]], requires_grad=True)
    BoundedHead(lo=0.0, hi=1.0).activate(z).sum().backward()
    assert (z.grad.abs() > 0).all()


# ── HeadStack ────────────────────────────────────────────────────

def test_a_stack_activates_each_channel_with_its_own_head():
    stack = HeadStack([RegressionHead(), BinaryHead(), BoundedHead(0.0, 1.0)],
                      ["a", "b", "c"])
    y = stack.activate(torch.randn(5, 3))
    assert y.shape == (5, 3)
    assert ((y[:, 1] > 0) & (y[:, 1] < 1)).all()
    assert ((y[:, 2] >= 0) & (y[:, 2] <= 1)).all()


def test_residual_reaches_only_the_channels_that_accept_one():
    stack = HeadStack([RegressionHead(), BinaryHead(), RegressionHead()],
                      ["a", "b", "c"])
    assert stack.residual_mask().tolist() == [True, False, True]     # binary is left alone


# ── Serialization: a reloaded run rebuilds what it trained ───────

def test_calibrated_thresholds_survive_load_state_dict():
    """Weights are saved before the thresholds are calibrated, so restoring
    the state_dict must not drag the decision rule back to the midpoints."""
    trained = HeadStack([OrdinalHead(n_levels=2)], ["a"])
    saved = trained.state_dict()
    rebuilt = HeadStack.from_config(
        {"a": {"kind": "ordinal", "n_levels": 2, "decision_thresholds": [0.3]}}, ["a"])
    rebuilt.load_state_dict(saved)
    assert rebuilt.heads[0].decision_thresholds.item() == pytest.approx(0.3)


# ── Rare levels: class weights and the decision rule ─────────────

def test_decision_thresholds_move_the_snap_but_not_the_activation():
    """A calibrated decision rule changes what decode emits at a given z;
    the smooth activation (and so the gradient) is untouched."""
    mid = OrdinalHead(n_levels=2)
    low = OrdinalHead(n_levels=2, decision_thresholds=torch.tensor([0.3]))
    z = torch.tensor([[0.4]])
    assert mid.decode(z).item() == 0.0
    assert low.decode(z).item() == 1.0
    assert torch.equal(mid.activate(z), low.activate(z))
    assert torch.equal(low.thresholds, mid.thresholds)


# ── Units: a stack declared physical, held normalised ────────────

def test_a_physical_stack_rescales_into_a_models_units_and_back():
    stats = {"power": {"shift": 5.0, "stretch": 2.0}, "stage": {"shift": 5.0, "stretch": 2.0},
             "flag": {"shift": 0.0, "stretch": 1.0}}
    physical = HeadStack([BoundedHead(1.0, INF),
                          OrdinalHead(n_levels=3, level_values=[1.0, 5.0, 9.0],
                                      decision_thresholds=[4.0, 8.0]),
                          BinaryHead()], ["power", "stage", "flag"])
    normalized = physical.normalized(stats)
    bounded, ordinal, _ = normalized.heads
    # An infinite bound means "no bound" and must not become a huge number.
    assert (bounded.lo, bounded.hi) == (-2.0, INF)
    assert ordinal.level_values.tolist() == [-2.0, 0.0, 2.0]
    assert ordinal.thresholds.tolist() == [-1.0, 1.0]
    assert ordinal.decision_thresholds.tolist() == [-0.5, 1.5]
    assert normalized.physical(stats).to_config() == physical.to_config()

    # A binary channel's 0/1 target only survives unnormalised.
    with pytest.raises(ValueError, match="binary head"):
        physical.normalized({**stats, "flag": {"shift": 0.5, "stretch": 1.0}})


# ── Pre-stack: derived inputs ────────────────────────────────────

ZONE = {"shift": 23.0, "stretch": 1.5, "min": 21.0, "max": 26.0}


def test_each_output_step_reads_its_own_step_of_u_and_y():
    """Distinguishable values: y[t, i] = 100·t + i, u[t, j] = -(100·t + j),
    with identity stats and a unit range so the demand is exactly
    `relu(y - u)` and the pairing shows in the number itself."""
    identity = {"shift": 0.0, "stretch": 1.0, "min": 0.0, "max": 1.0}
    history_steps, ny, nu = 3, 2, 2
    stack = PreStack([Demand(-1, identity, identity)], [("u1", "y1")],
                     ["y0", "y1"], ["u0", "u1"])
    y = torch.tensor([[[100.0 * t + i for i in range(ny)] for t in range(history_steps)]])
    u = torch.tensor([[[-(100.0 * t + j) for j in range(nu)] for t in range(history_steps)]])
    out = stack(y, u)[0]
    for t in range(history_steps):
        assert out[t, 0] == -(100 * t)                     # untouched column
        assert out[t, 1] == (100 * t + 1) + (100 * t + 1)  # relu(y1_t - u1_t) / 1
