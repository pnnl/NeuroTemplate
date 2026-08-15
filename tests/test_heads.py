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
MSE to floating-point precision, and `apply_residual` touches only the
channels that declared themselves residual-compatible.
"""

import math

import pytest
import torch
import torch.nn as nn

from neurotemplate.modeling.heads import (
    BinaryHead,
    BoundedHead,
    HeadStack,
    OrdinalHead,
    RegressionHead,
    bounds_from_norm,
    project_physical_to_norm,
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
    pytest.param(BoundedHead(lo=-1.0, hi=2.0), torch.zeros(32),
                 torch.full((32,), 1.5), 1.0, dict(lr=0.01, steps=600),
                 id="bounded-both"),
    pytest.param(BoundedHead(lo=0.0, hi=INF), torch.zeros(32),
                 torch.full((32,), 3.0), 1.0, dict(lr=0.2), id="bounded-lower"),
    pytest.param(BoundedHead(lo=-INF, hi=0.0), torch.zeros(32),
                 torch.full((32,), -2.5), 1.0, dict(lr=0.2), id="bounded-upper"),
    # Ordinal has to do better than "somewhat": snapping to integer levels is
    # the whole claim, so require the loss to fall by 10x.
    pytest.param(OrdinalHead(n_levels=4), torch.zeros(32),
                 torch.tensor([0.0, 1.0, 2.0, 3.0]).repeat(8), 0.1,
                 dict(lr=0.5, steps=800), id="ordinal-uniform"),
    pytest.param(OrdinalHead(n_levels=3, level_values=torch.tensor([0.0, 1.0, 5.0])),
                 torch.linspace(-1.0, 6.0, 30),
                 torch.tensor([0.0, 1.0, 5.0]).repeat_interleave(10), 0.1,
                 dict(lr=0.5, steps=800), id="ordinal-non-uniform"),
])
def test_a_head_descends_on_its_own_kind_of_target(head, z0, target, ratio, kw):
    torch.manual_seed(0)
    first, last, _ = fit(head.activate, head.loss, z0, target, **kw)
    assert last < first * ratio


def test_a_mixed_stack_still_has_a_descent_direction():
    torch.manual_seed(1)
    stack = HeadStack([RegressionHead(), BinaryHead()], ["a", "b"])
    target = torch.stack([torch.full((16,), 0.7),
                          torch.tensor([1.0, 0.0] * 8)], dim=-1)
    z = nn.Parameter(torch.zeros(16, 2))
    opt = torch.optim.SGD([z], lr=0.2)
    first = None
    for _ in range(300):
        loss = stack.loss(stack.activate(z), target)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


# ── Activations honour the range they promise ────────────────────

@pytest.mark.parametrize("head,lo,hi,open_side", [
    pytest.param(BoundedHead(lo=0.0, hi=1.0), 0.0, 1.0, None, id="bounded-both"),
    pytest.param(BoundedHead(lo=0.0, hi=INF), 0.0, INF, "hi", id="bounded-lower"),
    pytest.param(BoundedHead(lo=-INF, hi=5.0), -INF, 5.0, "lo", id="bounded-upper"),
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

def test_an_all_regression_stack_is_blanket_mse():
    """Identity heads must reproduce plain MSE to floating-point precision,
    or every existing run's loss quietly changed meaning."""
    torch.manual_seed(0)
    names = [f"ch_{i}" for i in range(4)]
    stack = HeadStack([RegressionHead() for _ in names], names)
    yhat, y = torch.randn(8, 12, 4), torch.randn(8, 12, 4)
    assert torch.allclose(((yhat - y) ** 2).mean(), stack.loss(yhat, y), atol=1e-7)


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
    y_prev = torch.tensor([[1.0, 99.0, 2.0], [3.0, 99.0, 4.0]])
    out = stack.apply_residual(torch.zeros(2, 3), y_prev)
    assert torch.equal(out[:, 0], y_prev[:, 0])
    assert torch.equal(out[:, 1], torch.zeros(2))     # binary is left alone
    assert torch.equal(out[:, 2], y_prev[:, 2])


# ── Serialization: a reloaded run rebuilds what it trained ───────

def test_a_mixed_stack_survives_a_config_round_trip():
    levels = torch.tensor([0.0, 0.5, 1.5])
    original = HeadStack(
        [RegressionHead(), BinaryHead(), BoundedHead(lo=-1.0, hi=2.0, beta=3.0),
         BoundedHead(lo=0.0, hi=INF),
         OrdinalHead(n_levels=3, level_values=levels, beta=10.0)],
        ["a", "b", "c", "d", "e"],
    )
    rebuilt = HeadStack.from_config(original.to_config(), original.channel_names)
    assert [h.kind for h in rebuilt.heads] == [h.kind for h in original.heads]
    assert (rebuilt.heads[2].lo, rebuilt.heads[2].hi, rebuilt.heads[2].beta) == (-1.0, 2.0, 3.0)
    assert rebuilt.heads[3].hi == INF                 # infinity is not a string
    assert rebuilt.heads[4].n_levels == 3
    assert torch.equal(rebuilt.heads[4].level_values, levels)


def test_a_half_bounded_head_is_still_half_bounded_after_a_reload():
    """`"inf"` on disk has to come back as a working floor, not a number."""
    stack = HeadStack.from_config(
        {"flow": {"kind": "bounded", "lo": 0.0, "hi": "inf", "beta": 5.0}}, ["flow"])
    head = stack.heads[0]
    assert isinstance(head, BoundedHead) and head.lo == 0.0 and head.hi == INF
    assert head.activate(torch.tensor([[-50.0]])).item() < 1e-10


def test_channels_the_config_does_not_mention_default_to_regression():
    stack = HeadStack.from_config({}, ["a", "b"])
    assert all(isinstance(h, RegressionHead) for h in stack.heads)


# ── Every way a head config can be malformed ─────────────────────

@pytest.mark.parametrize("build,message", [
    pytest.param(lambda: BoundedHead(lo=1.0, hi=0.0),
                 "requires hi > lo", id="hi below lo"),
    pytest.param(lambda: BoundedHead(lo=-INF, hi=INF),
                 "at least one finite bound", id="unbounded both ways"),
    pytest.param(lambda: OrdinalHead(n_levels=1),
                 "n_levels >= 2", id="one level"),
    pytest.param(lambda: OrdinalHead(n_levels=3,
                                     level_values=torch.tensor([0.0, 1.0, 1.0])),
                 "strictly increasing", id="non-increasing levels"),
    pytest.param(lambda: OrdinalHead(n_levels=4,
                                     init_thresholds=torch.tensor([0.5, 1.5])),
                 r"init_thresholds must have shape \(3,\)", id="wrong threshold count"),
    pytest.param(lambda: HeadStack([RegressionHead()] * 2, ["a", "a"]),
                 "must be unique", id="duplicate channel names"),
    pytest.param(lambda: HeadStack.from_config({"x": {"kind": "not_a_thing"}}, ["x"]),
                 "Unknown head kind", id="unknown kind"),
    pytest.param(lambda: HeadStack.from_config({"typo": {"kind": "binary"}}, ["x", "y"]),
                 "typo", id="unknown channel"),
])
def test_a_malformed_head_refuses_to_be_built(build, message):
    with pytest.raises(ValueError, match=message):
        build()


# ── Projection helpers ───────────────────────────────────────────

def test_bounds_project_from_physical_units_into_normalized_ones():
    stats = {"shift": 5.0, "stretch": 2.0, "min": 1.0, "max": 9.0}
    assert bounds_from_norm(stats) == (-2.0, 2.0)
    assert math.isclose(project_physical_to_norm(7.0, stats), 1.0)
    # An infinite bound means "no bound" and must not become a huge number.
    assert project_physical_to_norm(INF, stats) == INF
    assert project_physical_to_norm(-INF, stats) == -INF
