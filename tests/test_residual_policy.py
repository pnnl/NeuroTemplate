"""Residual DPC: the policy commands a bounded deviation from a rule.

The properties worth pinning are the ones that make the design mean anything:
δ starts at exactly zero, the commanded setpoint never leaves the ±Δ box
around the rule, the in-rollout slice of the current step matches the layout
the deploy path hands the same Node, and a Δ smaller than the setpoint grid —
the cheapest way to waste a campaign on this feature — raises.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.metrics import residual_metrics
from neurotemplate.control.policy import (
    _ResidualPolicy,
    build_policy,
    build_reference_node,
    reference_over_horizon,
)
from neurotemplate.control.scenarios import build_window_dict, horizons
from control_fixtures import full_setup


def _rollout(s):
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    data = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    )
    return policy, cl(data)


# ── Wiring ───────────────────────────────────────────────────────


def test_residual_keeps_the_node_signature_the_deploy_path_calls():
    """Shape B: the rule is composed inside the policy Node's callable, so the
    Node's input keys, input_map and output are unchanged — which is what lets
    the evaluator rebuild and invoke it with no edits at all."""
    plain = full_setup()
    resid = full_setup(residual_delta=1.0)
    a = build_policy(plain["cfg"], plain["sysid_cfg"], plain["nm"],
                     plain["norm_stats"])
    b = build_policy(resid["cfg"], resid["sysid_cfg"], resid["nm"],
                     resid["norm_stats"])
    assert isinstance(b.callable, _ResidualPolicy)
    assert a.input_keys == b.input_keys
    assert a.output_keys == b.output_keys
    assert a.input_map == b.input_map


# ── δ = 0 at initialization ──────────────────────────────────────


def test_a_fresh_residual_policy_emits_the_reference_exactly():
    """Zero-initialising the last layer plus a symmetric box means training
    starts at the rule, not near it."""
    s = full_setup(residual_delta=1.0, setpoint_step=0.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    _, out = _rollout(s)
    u_ref = reference_over_horizon(cfg, out, hmax, nm, ns)
    assert torch.allclose(out["U"][:, hmax:, :], u_ref, atol=1e-6)


@pytest.mark.parametrize("method", ["sigmoid_scale", "relu_clamp"])
def test_zero_init_holds_for_both_bound_methods(method):
    s = full_setup(residual_delta=1.0, setpoint_step=0.0,
                   policy_bound_method=method)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    _, out = _rollout(s)
    u_ref = reference_over_horizon(cfg, out, hmax, nm, ns)
    assert torch.allclose(out["U"][:, hmax:, :], u_ref, atol=1e-6)


def test_a_zero_width_box_pins_the_policy_to_the_rule_forever():
    """Δ = 0 is the floor: train nothing, and score exactly the rule."""
    s = full_setup(residual_delta=0.0, setpoint_step=0.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    # Perturb the whole network: the box is what holds, not the initialization.
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 5.0)

    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    out = cl(build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    ))
    u_ref = reference_over_horizon(cfg, out, hmax, nm, ns)
    assert torch.allclose(out["U"][:, hmax:, :], u_ref, atol=1e-6)


# ── The box actually bounds ──────────────────────────────────────


def test_a_trained_residual_never_leaves_the_box():
    s = full_setup(residual_delta=1.0, setpoint_step=0.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 20.0)

    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    out = cl(build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    ))
    u_ref = reference_over_horizon(cfg, out, hmax, nm, ns)
    stretch = torch.tensor([ns[c]["stretch"] for c in nm["U"]])
    delta_phys = ((out["U"][:, hmax:, :] - u_ref) * stretch).abs().detach()
    assert float(delta_phys.max()) <= cfg.residual_delta + 1e-4


def test_the_residual_still_learns():
    """A zero start is only useful if the gradient is alive there."""
    s = full_setup(residual_delta=1.0, setpoint_step=0.0, comfort_weight=1.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    out = cl(build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    ))
    out["U"][:, hmax:, :].pow(2).mean().backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert any(float(g.abs().sum()) > 0 for g in grads)


# ── Slicing matches the deploy layout ────────────────────────────


def test_the_in_rollout_current_step_matches_the_bare_node_call():
    """The rule inside the policy and the rule as a standalone Node must see
    the same current step, or the residual is measured against a fiction."""
    s = full_setup(residual_delta=1.0, setpoint_step=0.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, h_ctrl, hmax = horizons(cfg, sysid_cfg)
    ny, nd = len(nm["Y"]), len(nm["D"])
    n_ctrl = 1

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    rule_in_policy = policy.callable.reference
    bare = build_reference_node(cfg, sysid_cfg, nm, ns).callable

    b = 3
    yn_win = torch.randn(b, ny * h_ctrl)
    d_win = torch.randn(b, nd * h_ctrl)
    ymin, ymax = torch.zeros(b, n_ctrl), torch.ones(b, n_ctrl)

    windowed = rule_in_policy(yn_win, ymin, ymax, d_win)
    # The deploy path hands the bare Node the current step of each signal:
    # the LAST block of the yn/D history, the only block of the band.
    current = bare(yn_win[:, -ny:], ymin, ymax, d_win[:, -nd:])
    assert torch.allclose(windowed, current)


# ── Reporting ────────────────────────────────────────────────────


def test_residual_metrics_measure_the_deployed_deviation():
    u = torch.tensor([[[22.0], [23.0], [24.0]]])
    u_ref = torch.tensor([[[22.0], [22.5], [22.0]]])
    m = residual_metrics(u, u_ref, delta_cap=2.0, tol=0.25)
    assert m.mean_abs_delta == pytest.approx((0.0 + 0.5 + 2.0) / 3)
    assert m.max_abs_delta == pytest.approx(2.0)
    assert m.frac_at_cap == pytest.approx(1 / 3)


def test_a_zero_width_box_reports_no_saturation_rather_than_all_of_it():
    """Δ = 0 is the recommended first run — it scores the bare rule.

    Every deviation is trivially at the bound of a zero-width box, so "pinned
    on 100% of steps" is true and meaningless, and it drags in the reportcard's
    warning about a cap that binds. Mean |δ| still earns its place: at Δ = 0 it
    is the assertion that the residual path is inert.
    """
    u = torch.tensor([[[22.0], [23.0]]])
    m = residual_metrics(u, u, delta_cap=0.0, tol=0.25)

    assert m.frac_at_cap is None
    assert m.mean_abs_delta == 0.0
    assert m.max_abs_delta == 0.0


def test_a_grid_aligned_reference_leaves_no_snapping_floor(monkeypatch):
    """Quantization plus an off-grid reference puts a floor of half a grid step
    on mean |delta| that is snapping, not steering. A reference already on the
    grid does not — and the reportcard says which case you are in."""
    import control_user_defined as cud
    from neurotemplate.control.metrics import residual_metrics

    s = full_setup(residual_delta=1.0, setpoint_step=0.5)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    stretch = torch.tensor([ns[c]["stretch"] for c in nm["U"]])
    shift = torch.tensor([ns[c]["shift"] for c in nm["U"]])

    def measure(s):
        _, out = _rollout(s)
        u_ref = reference_over_horizon(cfg, out, hmax, nm, ns)
        return residual_metrics(
            (out["U"][:, hmax:, :] * stretch + shift).detach(),
            (u_ref * stretch + shift).detach(), cfg.residual_delta, 0.25,
        ).mean_abs_delta

    # The fixture's midpoint rule is not grid-aligned, so a delta = 0 policy
    # still shows a non-zero realized residual: the snap, not the network.
    off_grid = measure(s)
    assert off_grid > 0.05          # a real fraction of the 0.25 half-step floor

    midpoint = cud.reference_control

    def grid_aligned(yn, ymin, ymax, d, nm_mapping, norm_stats):
        phys = midpoint(yn, ymin, ymax, d, nm_mapping, norm_stats) * stretch + shift
        return ((phys / 0.5).round() * 0.5 - shift) / stretch

    monkeypatch.setattr(cud, "reference_control", grid_aligned)
    # What is left is float32 round-trip noise through normalize/denormalize,
    # two orders of magnitude below the floor above.
    assert measure(s) < off_grid / 100
