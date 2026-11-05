"""
Tests for the output heads.

A head can be fitted (the loss has a descent direction), its activation stays inside
the range it promises and keeps a gradient, the stack respects the residual, and a
stack rescales into a model's units and back. The hinge section pins the input-side
counterpart: the clamp is exact forward and never dead backward, and every output step
is computed from its own step of U and Y.
"""

import pytest
import torch
import torch.nn as nn

from neurotemplate.modeling.heads import BoundedHead, HeadStack, RegressionHead
from neurotemplate.modeling.hinge import Hinges

INF = float("inf")


@pytest.mark.parametrize("head", [RegressionHead(), BoundedHead(lo=0.0, hi=INF),
                                  BoundedHead(lo=-INF, hi=10.0), BoundedHead(lo=0.0, hi=10.0)],
                         ids=["regression", "lower", "upper", "both"])
def test_a_head_descends_on_a_target_in_its_range(head):
    z = nn.Parameter(torch.zeros(32))
    target = torch.full((32,), 3.0)
    opt = torch.optim.SGD([z], lr=0.2)
    first = None
    for _ in range(400):
        loss = head.loss(head.activate(z), target)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_an_activation_stays_inside_its_bounds_and_travels_on_an_open_side():
    z = torch.linspace(-50.0, 50.0, 200)
    y = BoundedHead(lo=0.0, hi=INF).activate(z)
    assert (y >= 0).all() and y[-1] > 40.0
    y = BoundedHead(lo=0.0, hi=1.0).activate(z)
    assert (y >= 0).all() and (y <= 1).all()


def test_a_saturated_bounded_head_still_passes_gradient():
    z = torch.tensor([-10.0], requires_grad=True)
    BoundedHead(lo=0.0, hi=1.0).activate(z).sum().backward()
    assert (z.grad.abs() > 0).all()


def test_a_stack_activates_each_channel_with_its_own_head_and_marks_the_residual():
    stack = HeadStack([RegressionHead(), BoundedHead(0.0, 1.0)], ["a", "b"])
    z = torch.randn(5, 2)
    y = stack(z)
    assert y.shape == (5, 2)
    assert torch.equal(y[:, 0], z[:, 0])
    assert ((y[:, 1] >= 0) & (y[:, 1] <= 1)).all()
    assert stack.residual_mask().tolist() == [True, False]


def test_a_physical_stack_rescales_into_a_models_units_and_back():
    stats = {"power": {"shift": 5.0, "stretch": 2.0}, "zone": {"shift": 0.0, "stretch": 1.0}}
    physical = HeadStack([BoundedHead(1.0, INF), RegressionHead()], ["power", "zone"])
    normalized = physical.normalized(stats)
    bounded = normalized.heads[0]
    # An infinite bound means "no bound" and must not become a huge number.
    assert (bounded.lo, bounded.hi) == (-2.0, INF)
    assert normalized.physical(stats).to_config() == physical.to_config()


def test_each_hinge_step_reads_its_own_step_of_u_and_y():
    """y[t, i] = 100·t + i, u[t, j] = -(100·t + j), with identity stats and a unit range,
    so the hinge is exactly `relu(y - u)` and the pairing shows in the number itself."""
    identity = {"shift": 0.0, "stretch": 1.0, "min": 0.0, "max": 1.0}
    mapping = {"Y": ["y0", "y1"], "U": ["u0", "u1"]}
    hinges = Hinges({"u1": {"target": "y1", "map": "nrelu"}}, mapping,
                    {c: identity for c in ("y0", "y1", "u0", "u1")})
    steps = 3
    y = torch.tensor([[[100.0 * t + i for i in range(2)] for t in range(steps)]])
    u = torch.tensor([[[-(100.0 * t + j) for j in range(2)] for t in range(steps)]])
    out = hinges(y, u)[0]
    for t in range(steps):
        assert out[t, 0] == -(100 * t)                     # untouched column
        assert out[t, 1] == (100 * t + 1) + (100 * t + 1)  # relu(y1_t - u1_t) / 1


def test_a_hinge_that_could_never_fire_is_refused():
    stats = {"zone": {"shift": 0.0, "stretch": 1.0, "min": 20.0, "max": 25.0},
             "sp": {"shift": 0.0, "stretch": 1.0, "min": 30.0, "max": 35.0}}
    with pytest.raises(ValueError, match="never active"):
        Hinges({"sp": {"target": "zone", "map": "nrelu"}}, {"Y": ["zone"], "U": ["sp"]}, stats)
