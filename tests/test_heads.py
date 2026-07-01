"""
Unit tests for the configurable-output-heads module.

Each head type passes its toy-data unit test: loss is non-negative and
decreases on its target signal, gradients flow through `activate`.

Plus two HeadStack invariants:
  - all-regression HeadStack.loss == mean blanket MSE (to fp precision)
  - apply_residual only touches residual_compatible channels
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


def _fit_single_head(head, make_z, target, steps=400, lr=0.1):
    """Optimize a scalar parameter so head.activate matches target."""
    p = nn.Parameter(torch.zeros(()))
    opt = torch.optim.SGD([p], lr=lr)
    first_loss = None
    for _ in range(steps):
        z = make_z(p)
        y = head.activate(z)
        loss = head.loss(y, target)
        if first_loss is None:
            first_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return first_loss, loss.item(), p


# ── Regression head ──────────────────────────────────────────────


def test_regression_head_fits_constant():
    head = RegressionHead()
    target = torch.full((32,), 2.5)
    first, last, p = _fit_single_head(
        head, make_z=lambda p: p.expand(32).unsqueeze(-1), target=target
    )
    assert last < first
    assert abs(p.item() - 2.5) < 1e-2


# ── Binary head ──────────────────────────────────────────────────


def test_binary_head_fits_zero_one():
    head = BinaryHead()
    target = torch.tensor([0.0, 1.0, 0.0, 1.0]).repeat(8)
    bias = nn.Parameter(torch.zeros(()))
    pattern = torch.tensor([-1.0, 1.0, -1.0, 1.0]).repeat(8)
    opt = torch.optim.SGD([bias], lr=0.5)
    first = None
    for _ in range(400):
        z = (bias * pattern).unsqueeze(-1)
        y = head.activate(z)
        loss = head.loss(y, target)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
    assert bias.item() > 0.5


# ── Bounded head ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lo, hi, target_val, lr, steps",
    [
        (-1.0, 2.0, 1.5, 0.01, 600),                  # fully bounded
        (0.0, float("inf"), 3.0, 0.2, 400),           # lower-bounded only
        (float("-inf"), 0.0, -2.5, 0.2, 400),         # upper-bounded only
    ],
    ids=["full", "lower-only", "upper-only"],
)
def test_bounded_head_fits_target(lo, hi, target_val, lr, steps):
    head = BoundedHead(lo=lo, hi=hi)
    target = torch.full((32,), target_val)
    first, last, p = _fit_single_head(
        head, make_z=lambda p: p.expand(32).unsqueeze(-1),
        target=target, steps=steps, lr=lr,
    )
    y_final = head.activate(p.expand(32).unsqueeze(-1))
    assert last < first
    assert (y_final >= lo).all() and (y_final <= hi).all()
    assert abs(y_final.mean().item() - target_val) < 0.05


def test_bounded_head_gradient_below_band():
    head = BoundedHead(lo=0.0, hi=1.0)
    z = torch.tensor([[-10.0]], requires_grad=True)
    y = head.activate(z)
    y.sum().backward()
    assert z.grad is not None
    assert (z.grad.abs() > 0).all()


def test_bounded_head_rejects_bad_range():
    try:
        BoundedHead(lo=1.0, hi=0.0)
    except ValueError:
        return
    raise AssertionError("BoundedHead must reject hi <= lo")


def test_lower_bounded_head_activation_respects_floor():
    head = BoundedHead(lo=0.0, hi=float("inf"))
    z = torch.linspace(-50.0, 50.0, 200).unsqueeze(-1)
    y = head.activate(z)
    assert (y >= 0.0).all()
    assert (y[-1] > 40.0).item()


def test_upper_bounded_head_activation_respects_ceiling():
    head = BoundedHead(lo=float("-inf"), hi=5.0)
    z = torch.linspace(-50.0, 50.0, 200).unsqueeze(-1)
    y = head.activate(z)
    assert (y <= 5.0).all()
    assert (y[0] < -40.0).item()


def test_bounded_head_rejects_both_infinite():
    try:
        BoundedHead(lo=float("-inf"), hi=float("inf"))
    except ValueError:
        return
    raise AssertionError("BoundedHead must reject the both-infinite case")


# ── Ordinal head ─────────────────────────────────────────────────


def test_ordinal_head_activation_bounded():
    head = OrdinalHead(n_levels=4)
    z = torch.linspace(-100.0, 100.0, 200).unsqueeze(-1)
    y = head.activate(z)
    assert (y >= 0.0).all() and (y <= 3.0).all()


def test_ordinal_head_fits_integer_targets():
    torch.manual_seed(0)
    head = OrdinalHead(n_levels=4)
    target = torch.tensor([0.0, 1.0, 2.0, 3.0]).repeat(8)
    z_param = nn.Parameter(torch.zeros(32))
    opt = torch.optim.SGD([z_param], lr=0.5)
    first = None
    for _ in range(800):
        y = head.activate(z_param.unsqueeze(-1))
        loss = head.loss(y, target)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.1


def test_ordinal_head_non_uniform_level_values_fit_and_monotone():
    """Non-uniform level_values=[0, 1, 5] (Δv=[1,4]) — verify loss decreases
    by 10× and that activate is monotone in z (Σ of monotone sigmoids)."""
    torch.manual_seed(0)
    levels = torch.tensor([0.0, 1.0, 5.0])
    head = OrdinalHead(n_levels=3, level_values=levels)
    target = levels.repeat_interleave(10)
    z_param = nn.Parameter(torch.linspace(-1.0, 6.0, 30))
    opt = torch.optim.SGD([z_param], lr=0.5)
    first = None
    for _ in range(800):
        y = head.activate(z_param.unsqueeze(-1))
        loss = head.loss(y, target)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.1

    z_grid = torch.linspace(-5.0, 10.0, 400).unsqueeze(-1)
    y_grid = head.activate(z_grid)
    assert (y_grid[1:] - y_grid[:-1] >= -1e-6).all()


def test_ordinal_head_rejects_non_increasing_level_values():
    try:
        OrdinalHead(n_levels=3, level_values=torch.tensor([0.0, 1.0, 1.0]))
    except ValueError:
        return
    raise AssertionError("OrdinalHead must reject non-strictly-increasing level_values")


def test_ordinal_head_requires_at_least_two_levels():
    try:
        OrdinalHead(n_levels=1)
    except ValueError:
        return
    raise AssertionError("OrdinalHead must reject n_levels < 2")


def test_ordinal_head_rejects_wrong_threshold_shape():
    try:
        OrdinalHead(n_levels=4, init_thresholds=torch.tensor([0.5, 1.5]))
    except ValueError:
        return
    raise AssertionError("OrdinalHead must reject wrong-sized init_thresholds")


# ── HeadStack ────────────────────────────────────────────────────


def test_head_stack_all_regression_matches_blanket_mse():
    """Identity-regression heads reproduce blanket MSE to fp precision."""
    torch.manual_seed(0)
    n_channels = 4
    names = [f"ch_{i}" for i in range(n_channels)]
    stack = HeadStack([RegressionHead() for _ in range(n_channels)], names)

    yhat = torch.randn(8, 12, n_channels)
    y = torch.randn(8, 12, n_channels)

    expected = ((yhat - y) ** 2).mean()
    actual = stack.loss(yhat, y)
    assert torch.allclose(expected, actual, atol=1e-7)


def test_head_stack_activate_shape():
    stack = HeadStack(
        [RegressionHead(), BinaryHead(), BoundedHead(0.0, 1.0)],
        ["a", "b", "c"],
    )
    z_raw = torch.randn(5, 3)
    y = stack.activate(z_raw)
    assert y.shape == (5, 3)
    assert ((y[:, 1] > 0) & (y[:, 1] < 1)).all()
    assert ((y[:, 2] >= 0) & (y[:, 2] <= 1)).all()


def test_head_stack_residual_only_on_compatible():
    """Residual must add y_prev to regression channels and skip binary."""
    stack = HeadStack(
        [RegressionHead(), BinaryHead(), RegressionHead()],
        ["a", "b", "c"],
    )
    z_raw = torch.zeros(2, 3)
    y_prev = torch.tensor([[1.0, 99.0, 2.0], [3.0, 99.0, 4.0]])
    out = stack.apply_residual(z_raw, y_prev)
    assert torch.equal(out[:, 0], y_prev[:, 0])
    assert torch.equal(out[:, 1], torch.zeros(2))
    assert torch.equal(out[:, 2], y_prev[:, 2])


def test_head_stack_loss_decreases_mixed_heads():
    """Mixed regression + binary head stack still has a descent direction."""
    torch.manual_seed(1)
    stack = HeadStack([RegressionHead(), BinaryHead()], ["a", "b"])
    target = torch.stack(
        [torch.full((16,), 0.7), torch.tensor([1.0, 0.0] * 8)], dim=-1,
    )
    z = nn.Parameter(torch.zeros(16, 2))
    opt = torch.optim.SGD([z], lr=0.2)
    first = None
    for _ in range(300):
        y = stack.activate(z)
        loss = stack.loss(y, target)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


# ── Projection helpers ──────────────────────────────────────────


def test_bounds_from_norm_projects_min_max():
    stats = {"shift": 5.0, "stretch": 2.0, "min": 1.0, "max": 9.0}
    lo, hi = bounds_from_norm(stats)
    assert math.isclose(lo, -2.0)
    assert math.isclose(hi, 2.0)


def test_project_physical_to_norm_passes_infinity_through():
    stats = {"shift": 5.0, "stretch": 2.0}
    assert project_physical_to_norm(float("inf"), stats) == float("inf")
    assert project_physical_to_norm(float("-inf"), stats) == float("-inf")
    assert math.isclose(project_physical_to_norm(7.0, stats), 1.0)


# ── HeadStack serialization ─────────────────────────────────────


def test_head_stack_to_config_roundtrip_preserves_kinds_and_bounds():
    """Roundtrip a mixed stack through to_config/from_config and verify
    every head reconstructs to the same kind and parameters."""
    levels = torch.tensor([0.0, 0.5, 1.5])
    original = HeadStack(
        [
            RegressionHead(),
            BinaryHead(),
            BoundedHead(lo=-1.0, hi=2.0, beta=3.0),
            BoundedHead(lo=0.0, hi=float("inf")),
            OrdinalHead(n_levels=3, level_values=levels, beta=10.0),
        ],
        ["a", "b", "c", "d", "e"],
    )
    cfg = original.to_config()
    rebuilt = HeadStack.from_config(cfg, original.channel_names)
    assert [h.kind for h in rebuilt.heads] == [h.kind for h in original.heads]
    assert rebuilt.heads[2].lo == -1.0 and rebuilt.heads[2].hi == 2.0
    assert rebuilt.heads[2].beta == 3.0
    assert rebuilt.heads[3].hi == float("inf")
    assert rebuilt.heads[4].n_levels == 3
    assert torch.equal(rebuilt.heads[4].level_values, levels)


def test_head_stack_from_config_defaults_missing_channels_to_regression():
    """Channels absent from the config dict fall back to RegressionHead."""
    stack = HeadStack.from_config({}, ["a", "b"])
    assert all(isinstance(h, RegressionHead) for h in stack.heads)


def test_head_stack_from_config_rejects_unknown_kind():
    try:
        HeadStack.from_config({"x": {"kind": "not_a_thing"}}, ["x"])
    except ValueError:
        return
    raise AssertionError("Unknown head kind must raise")


def test_head_stack_from_config_rejects_unknown_channel():
    try:
        HeadStack.from_config({"typo": {"kind": "binary"}}, ["x", "y"])
    except ValueError as e:
        assert "typo" in str(e)
        return
    raise AssertionError("Unknown channel in head config must raise")


def test_head_stack_from_config_round_trips_half_bounded():
    """Verify half-bounded heads survive serialization via from_config."""
    cfg = {"flow": {"kind": "bounded", "lo": 0.0, "hi": "inf", "beta": 5.0}}
    stack = HeadStack.from_config(cfg, ["flow"])
    head = stack.heads[0]
    assert isinstance(head, BoundedHead)
    assert head.lo == 0.0 and head.hi == float("inf")
    assert head.activate(torch.tensor([[-50.0]])).item() < 1e-10


def test_head_stack_unique_channel_names():
    try:
        HeadStack([RegressionHead(), RegressionHead()], ["a", "a"])
    except ValueError:
        return
    raise AssertionError("Duplicate channel names must raise")
