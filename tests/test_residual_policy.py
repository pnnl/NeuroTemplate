"""Residual DPC: an anchored U column commands a bounded deviation from a rule.

The properties worth pinning are the ones that make the design mean anything:
δ starts at exactly zero, the commanded value never leaves its column's box
around the rule, a box with a zero side does not kill the gradient, the
in-rollout slice of the current step matches the layout the deploy path hands
the rule, and a rule that anchors different columns from the boxes is refused.
"""

import pytest
import torch
import torch.nn as nn
from neuromancer.modules import blocks

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.losses import resolve_objective
from neurotemplate.control.metrics import residual_metrics
from neurotemplate.control.policy import (
    _ResidualPolicy,
    _reference_rule,
    _zero_final_layer,
    build_policy,
    reference_over_horizon,
)
from neurotemplate.control.scenarios import build_window_dict, horizons, scored_steps
from neurotemplate.hooks import load_control_hooks
from control_fixtures import SETPT, full_setup


def _box(below, above):
    return {SETPT: [below, above]}


def _windows(s):
    cfg = s["cfg"]
    return build_window_dict(s["norm_df"], cfg, s["sysid_cfg"], s["nm"],
                              P=cfg.train_pred_steps, stride=cfg.history_steps)


def _rollout(s):
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    return policy, cl(_windows(s))


# ── δ = 0 at initialization ──────────────────────────────────────


@pytest.mark.parametrize("method", ["sigmoid_scale", "relu_clamp"])
def test_a_fresh_residual_policy_emits_the_reference_exactly(method):
    """Zero-initialising the anchored rows of the last layer, plus a symmetric
    net box, means training starts at the rule, not near it."""
    s = full_setup(residual_delta=_box(1.0, 1.0), policy_bound_method=method)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    _, out = _rollout(s)
    u_ref = reference_over_horizon(cfg, out, hmax, nm, ns)[SETPT]
    assert torch.allclose(out["U"][:, scored_steps("U", hmax), 0], u_ref, atol=1e-6)


def test_a_zero_width_box_pins_the_policy_to_the_rule_forever():
    """[0, 0] is the rule itself: train nothing, and score exactly the rule."""
    s = full_setup(residual_delta=_box(0.0, 0.0))
    assert _perturbed_deviation(s, 5.0).abs().max() < 1e-5


# ── The box actually bounds ──────────────────────────────────────


def _perturbed_deviation(s, scale):
    """The deployed deviation from the rule, in °C, after shaking every
    parameter of the policy by `scale` — the box is what holds, not the
    initialisation."""
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * scale)
    out = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)(_windows(s))
    u_ref = reference_over_horizon(cfg, out, hmax, nm, ns)[SETPT]
    deviation = out["U"][:, scored_steps("U", hmax), 0] - u_ref
    return (deviation * ns[SETPT]["stretch"]).detach()


def test_a_one_sided_box_starts_at_the_rule_and_deploys_only_its_open_side():
    """[0, 0.8]: a setpoint anchored at the band floor may move up off it by
    at most 0.8 and never down — and a fresh policy still emits the rule."""
    s = full_setup(residual_delta=_box(0.0, 0.8))
    assert _perturbed_deviation(s, 0.0).abs().max() < 1e-5
    torch.manual_seed(0)
    delta = _perturbed_deviation(s, 20.0)
    assert float(delta.min()) >= -1e-5
    assert float(delta.max()) <= 0.8 + 1e-5
    assert float(delta.max()) > 0.1, "the open side should still be reachable"


def test_a_heating_and_cooling_pair_recover_from_their_closed_sides():
    """A [0, 0.5] heating column and a [0.5, 0] cooling column, held inward in
    opposite directions. Each starts on its rule, which is its closed edge;
    pushed against that edge and then away, each must come off it. A hard
    clamp there zeroes the gradient for good — measured, δ stayed 0 with the
    incentive reversed — so this is the straight-through clamp's test. (A
    long push saturates the net's own sigmoid, from which recovery is slow:
    measured, ~300 steps instead of 40, as at any sigmoid edge.)"""
    torch.manual_seed(0)
    half = torch.tensor([0.5, 0.5])
    net = blocks.MLP_bounds(insize=4, outsize=2, bias=True, linear_map=nn.Linear,
                            nonlin=nn.GELU, hsizes=[8], min=-half, max=half,
                            method="sigmoid_scale")
    _zero_final_layer(net, torch.tensor([True, True]))
    rule = {"heat": torch.tensor(20.0), "cool": torch.tensor(24.0)}
    policy = _ResidualPolicy(
        net, lambda x: rule, ["heat", "cool"],
        umin=torch.tensor([0.0, 0.0]), umax=torch.tensor([50.0, 50.0]),
        below=torch.tensor([0.0, 0.5]), above=torch.tensor([0.5, 0.0]))
    x = torch.randn(32, 4)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    def push(direction, steps):
        """Push heating's δ in `direction` and cooling's against it."""
        for _ in range(steps):
            delta = policy(x) - torch.stack([rule["heat"], rule["cool"]])
            loss = -direction * (delta[:, 0] - delta[:, 1]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return (policy(x) - torch.stack([rule["heat"], rule["cool"]])).detach()

    closed = push(-1.0, 5)                  # into each closed side
    assert closed[:, 0].max() <= 0.0 and closed[:, 1].min() >= 0.0
    opened = push(+1.0, 40)                 # and back out through the rule
    assert opened[:, 0].mean() > 0.4 and opened[:, 0].max() <= 0.5
    assert opened[:, 1].mean() < -0.4 and opened[:, 1].min() >= -0.5


# ── Slicing matches the deploy layout ────────────────────────────


def test_the_in_rollout_current_step_matches_the_bare_rule():
    """The rule inside the policy and the rule handed only the current step
    must agree, or the residual is measured against a fiction."""
    low = "Comfort Low [°C]"
    s = full_setup(residual_delta=_box(1.0, 1.0), future={low: 2})
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    cfg.policy_window[low]["past"] = 1      # R now has history too
    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    ny, nd, n_ref = len(nm["Y"]), len(nm["D"]), 2

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    rule_in_policy = policy.callable.reference
    keys = ["yn", "D", "R"]
    bare = _reference_rule(cfg, load_control_hooks(cfg), nm, ns, keys,
                           now={key: 0 for key in keys},
                           r_columns=resolve_objective(cfg, nm).r_columns)

    b = 3
    yn_win = torch.randn(b, ny * h_ctrl)
    d_win = torch.randn(b, nd * h_ctrl)
    r_win = torch.randn(b, n_ref * 4)       # 1 past, current, 2 previewed

    windowed = rule_in_policy(yn_win, d_win, r_win)[SETPT]
    # The deploy path hands the rule the current step of each signal: the
    # LAST block of the yn/D history, the block after R's one past step.
    current = bare(yn_win[:, -ny:], d_win[:, -nd:], r_win[:, n_ref:2 * n_ref])[SETPT]
    assert torch.allclose(windowed, current)


# ── Reporting ────────────────────────────────────────────────────


def test_residual_metrics_measure_the_deployed_deviation():
    u = torch.tensor([[22.0, 23.0, 24.0, 20.0]])
    u_ref = torch.tensor([[22.0, 22.5, 22.0, 22.0]])
    m = residual_metrics(SETPT, u, u_ref, box=(2.0, 2.0))
    assert m.mean_abs_delta == pytest.approx((0.0 + 0.5 + 2.0 + 2.0) / 4)
    assert m.max_abs_delta == pytest.approx(2.0)
    # One step on each edge, reported separately.
    assert m.frac_at_below == pytest.approx(1 / 4)
    assert m.frac_at_above == pytest.approx(1 / 4)
