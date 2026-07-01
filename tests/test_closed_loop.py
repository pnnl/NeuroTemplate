"""
The gating integration test: policy + frozen dynamics roll out with correct
shapes, and gradients reach ONLY the policy (dynamics stays frozen).
"""

import torch

from neurotemplate.control.scenarios import build_window_dict, build_loaders, horizons
from neurotemplate.control.policy import build_policy
from neurotemplate.control.closed_loop import build_closed_loop, build_control_loss, build_problem
from control_fixtures import full_setup


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


def test_frozen_param_count_is_zero_trainable():
    s = full_setup()
    n_trainable = sum(
        p.numel() for p in s["dyn_sys"].parameters() if p.requires_grad
    )
    assert n_trainable == 0
