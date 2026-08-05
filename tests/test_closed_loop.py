"""
The gating integration test: policy + frozen dynamics roll out with correct
shapes, and gradients reach ONLY the policy (dynamics stays frozen).
"""

import torch

from neurotemplate.control.scenarios import build_window_dict, build_loaders, horizons
from neurotemplate.control.policy import build_policy
from neurotemplate.control.closed_loop import build_closed_loop, build_control_loss, build_problem
from control_fixtures import (
    full_setup, make_control_cfg, make_frozen_dynamics, make_nm_mapping,
    make_norm_stats, make_raw_df, make_sysid_cfg,
)
from neurotemplate.common.data_utils import normalize_df


def test_closed_loop_rollout_shapes():
    s = full_setup()
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    P = cfg.train_pred_steps

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    data = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, s["norm_stats"],
        P=P, stride=cfg.in_steps,
    )
    out = cl(data)
    n = data["yn"].shape[0]
    # seed (hmax) + P appended predictions / actions
    assert out["yn"].shape == (n, hmax + P, len(nm["Y"]))
    assert out["U"].shape == (n, hmax + P, len(nm["U"]))


def test_gradients_only_reach_policy():
    s = full_setup(comfort_weight=10.0, du_weight=0.0, effort_weight=0.0)
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm, s["norm_stats"])
    problem = build_problem(cl, loss)

    train, _ = build_loaders(
        cfg, sysid_cfg, nm, s["norm_stats"], [s["norm_df"]], [s["norm_df"]],
    )
    batch = next(iter(train))
    out = problem(batch)
    total = out[f'{batch["name"]}_loss']
    total.backward()

    policy_grads = [
        p.grad for p in policy.callable.parameters() if p.requires_grad
    ]
    assert policy_grads, "policy has no trainable params"
    assert any(g is not None and g.abs().sum() > 0 for g in policy_grads), \
        "policy received no gradient signal"

    # Dynamics is frozen: requires_grad False => no grad accumulated.
    for p in s["dyn_node"].callable.parameters():
        assert p.requires_grad is False
        assert p.grad is None


def _gru_setup():
    """full_setup, but with a stateful surrogate instead of an MLP."""
    nm = make_nm_mapping()
    raw = make_raw_df()
    norm_stats = make_norm_stats(raw)
    norm_df = normalize_df(raw, norm_stats)
    sysid_cfg = make_sysid_cfg(model="gru")
    cfg = make_control_cfg()
    _, dyn_node = make_frozen_dynamics(sysid_cfg, nm)
    return nm, norm_df, norm_stats, sysid_cfg, cfg, dyn_node


def test_gru_surrogate_rolls_out_and_warms_up():
    """A stateful surrogate must keep the same shape contract, reset between
    forwards, and enter the rollout warmed over [hmax-h_sys, hmax-2]."""
    nm, norm_df, norm_stats, sysid_cfg, cfg, dyn_node = _gru_setup()
    h_sys, _, hmax = horizons(cfg, sysid_cfg)
    P = cfg.train_pred_steps

    policy = build_policy(cfg, sysid_cfg, nm, norm_stats)
    cl = build_closed_loop(dyn_node, policy, cfg, sysid_cfg)
    data = build_window_dict(
        norm_df, cfg, sysid_cfg, nm, norm_stats, P=P, stride=cfg.in_steps,
    )

    out = cl(dict(data))
    n = data["yn"].shape[0]
    assert out["yn"].shape == (n, hmax + P, len(nm["Y"]))
    assert out["U"].shape == (n, hmax + P, len(nm["U"]))

    # Reset: two forwards on the same window must agree.
    assert torch.allclose(out["yn"], cl(dict(data))["yn"])

    # The surrogate warms from its own h_sys-wide window ending at hmax-1, so
    # feeding that window to a freshly reset block reproduces the first
    # prediction. Control never has to know those indices — SystemPreview's
    # input_map derives the window from the model's declared in_steps.
    model = dyn_node.callable
    model.fy.reset()
    with torch.no_grad():
        expected = model(*[
            data[key][:, hmax - h_sys:hmax, :].reshape(data[key].shape[0], -1)
            for key in ("yn", "U", "D")
        ])
    assert torch.allclose(out["yn"][:, hmax, :], expected, atol=1e-6)


def test_frozen_param_count_is_zero_trainable():
    s = full_setup()
    n_trainable = sum(
        p.numel() for p in s["dyn_sys"].parameters() if p.requires_grad
    )
    assert n_trainable == 0
