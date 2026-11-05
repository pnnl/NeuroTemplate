"""
The gating integration test: policy + frozen dynamics roll out with correct
shapes, and gradients reach ONLY the policy (dynamics stays frozen).
"""

import torch

from neurotemplate.control.scenarios import (
    build_window_dict, build_loaders, horizons, scored_steps,
)
from neurotemplate.control.policy import build_policy
from neurotemplate.control.closed_loop import build_closed_loop, build_control_loss, build_problem
from control_fixtures import (
    ZONE, SETPT, DIST, Y_COLS,
    full_setup, make_frozen_dynamics, make_norm_stats, make_sysid_cfg,
)
from neurotemplate.modeling.dynamics import get_dynamics, stateful_blocks


def _windows(s, cfg=None, sysid_cfg=None):
    cfg = cfg or s["cfg"]
    return build_window_dict(
        s["frame"], cfg, sysid_cfg or s["sysid_cfg"], s["nm"],
        P=cfg.train_pred_steps, stride=cfg.history_steps,
    )


def test_the_first_prediction_already_responds_to_the_policy():
    """The dynamics applies the policy's command in the step it was chosen.

    With U seeded as long as yn, the policy's command landed one index past
    where the dynamics read, so yn[hmax] was driven by recorded data and the
    policy trained against a one-step actuation delay the plant does not have.
    """
    s = full_setup()
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    cl(_windows(s))["yn"][:, hmax, :].sum().backward()

    grads = [p.grad for p in policy.callable.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_gradients_only_reach_policy():
    s = full_setup()
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm)
    problem = build_problem(cl, loss)

    train, _ = build_loaders(cfg, sysid_cfg, nm, [s["frame"]], [s["frame"]])
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
    s = full_setup()
    sysid_cfg = make_sysid_cfg(model="gru")
    _, dyn_node = make_frozen_dynamics(sysid_cfg, s["nm"], make_norm_stats(s["plant"]))
    return s, sysid_cfg, dyn_node


def test_gru_surrogate_rolls_out_and_warms_up():
    """A stateful surrogate must keep the same shape contract, reset between
    forwards, and enter the rollout warmed over [hmax-h_sys, hmax-2]."""
    s, sysid_cfg, dyn_node = _gru_setup()
    cfg, nm = s["cfg"], s["nm"]
    h_sys, _, hmax = horizons(cfg, sysid_cfg)
    P = cfg.train_pred_steps

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(dyn_node, policy, cfg, sysid_cfg)
    data = _windows(s, sysid_cfg=sysid_cfg)

    out = cl(dict(data))
    n = data["yn"].shape[0]
    assert out["yn"].shape == (n, hmax + P, len(nm["Y"]))
    assert out["U"].shape == (n, hmax - 1 + P, len(nm["U"]))

    # Reset: two forwards on the same window must agree.
    assert torch.allclose(out["yn"], cl(dict(data))["yn"])

    # The surrogate warms from its own h_sys-wide window ending at hmax-1, so
    # feeding that window to a freshly reset block reproduces the first
    # prediction. The window is read from `out`, because its last U row is the
    # policy's first command rather than recorded data. Control never has to know those indices — SystemPreview's
    # input_map derives the window from the model's declared history_steps.
    model = dyn_node.callable
    for block in stateful_blocks(model):
        block.set_rollout_state(None)
    with torch.no_grad():
        expected = model(*[
            out[key][:, hmax - h_sys:hmax, :].reshape(out[key].shape[0], -1)
            for key in ("yn", "U", "D")
        ])
    assert torch.allclose(out["yn"][:, hmax, :], expected, atol=1e-6)


def _hinge_setup():
    """full_setup, but the zone is its own block reading the setpoint through a
    hinge, the other channels an MLP block."""
    s = full_setup()
    nm, norm_stats = s["nm"], s["norm_stats"]
    sysid_cfg = make_sysid_cfg(model="mlp")
    spec = {
        "zone": {"Y": [ZONE], "U": [SETPT], "model": "mlp",
                 "inputs": {SETPT: {"target": ZONE, "map": "nrelu"}}},
        "rtu": {"Y": [c for c in Y_COLS if c != ZONE], "model": "mlp"},
    }
    dyn_sys = get_dynamics(sysid_cfg, nm, norm_stats, block_spec=spec)
    for p in dyn_sys.parameters():
        p.requires_grad_(False)
    dyn_sys.eval()
    return s, sysid_cfg, dyn_sys.nodes[0]


def test_a_policy_parked_on_the_inactive_side_still_gets_gradient():
    """The end-to-end form of "no dead zone". The policy is set to command
    the highest recorded setpoint everywhere — above the zone, so the hinge
    is exactly zero and the plant would ignore it — and the raw U it emits
    keeps its width. A hard relu would hand it zero gradient forever; the
    soft clamp's backward lets it learn to start cooling from a coast."""
    s, sysid_cfg, dyn_node = _hinge_setup()
    cfg, nm, norm_stats = s["cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, norm_stats)
    last = policy.callable.base.base.linear[-1]   # _Commands -> _NormalizedNet -> net
    with torch.no_grad():
        last.weight.zero_()
        # Near the top of the bounded net's range without saturating its own
        # sigmoid: the dead zone under test is the thermostat's, not the net's.
        last.bias.fill_(2.0)

    cl = build_closed_loop(dyn_node, policy, cfg, sysid_cfg)
    out = cl(_windows(s, sysid_cfg=sysid_cfg))
    assert out["U"].shape[-1] == len(nm["U"])

    zone_i, sp = nm["Y"].index(ZONE), norm_stats[SETPT]
    zone_phys = out["yn"][:, hmax:, zone_i] * norm_stats[ZONE]["stretch"] + norm_stats[ZONE]["shift"]
    sp_phys = out["U"][:, scored_steps("U", hmax), 0] * sp["stretch"] + sp["shift"]
    assert (sp_phys > zone_phys).all(), "the fixture should sit on the inactive side"

    out["yn"][:, hmax:, zone_i].sum().backward()
    grads = [p.grad for p in policy.callable.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_the_last_steps_preview_reads_real_forecast_rows():
    """A `future` past the rollout's end is read from the data, not padded.

    D and R used to be cut at Hmax + P, so at the late steps of every rollout
    SystemPreview's `nearest` padding repeated the final row for the whole
    preview. They are now F rows longer, and the closed loop trims them back.
    """
    future = 3
    s = full_setup(future={DIST[0]: future})
    cfg, sysid_cfg = s["cfg"], s["sysid_cfg"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    windows = _windows(s)
    rollout = hmax + cfg.train_pred_steps

    assert windows["D"].shape[1] == rollout + future
    tail = windows["D"][:, rollout - 1:, :]
    assert not torch.equal(tail, tail[:, :1, :].expand_as(tail)), \
        "the preview beyond the last step is the final row repeated"

    policy = build_policy(cfg, sysid_cfg, s["nm"], s["norm_stats"])
    out = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)(windows)
    assert out["yn"].shape[1] == rollout
    assert out["D"].shape[1] == out["t"].shape[1] == rollout
