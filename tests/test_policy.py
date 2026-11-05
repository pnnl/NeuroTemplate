import pytest
import torch

from neurotemplate.control.windows import build_window_dict, horizons, scored_steps
from neurotemplate.control.policy import (
    _NormalizedNet,
    _key_gather,
    build_policy,
)
from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.policy import control_box
from control_fixtures import ZONE, full_setup

#: The fixture objective's reference columns: the band's two edges.
N_REF = 2


def _windows(s):
    cfg = s["cfg"]
    return build_window_dict(s["frame"], cfg, s["sysid_cfg"], s["nm"],
                             P=cfg.horizon_steps, stride=cfg.history_steps)


# ── Per-column windows ───────────────────────────────────────────


def test_key_gather_keeps_each_columns_declared_steps():
    # D, col0 {past 2, future 0}, col1 {past 2, future 2}: a 5-row window,
    # both columns on the 3 past/current rows, col1 alone on the 2 future.
    assert _key_gather("D", [(2, 0), (2, 2)]) == [0, 1, 2, 3, 4, 5, 7, 9]
    # U ends at the previous step: col0 {past 2} keeps rows 0 and 1 of a
    # 3-row window, never the current row's padded copy; col1 {past 0} none.
    assert _key_gather("U", [(2, 0), (0, 0)]) == [0, 2]


def test_a_uniform_window_builds_no_gather():
    s = full_setup()
    policy = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    net = next(m for m in policy.callable.modules() if isinstance(m, _NormalizedNet))
    assert net.indices == []


def test_per_column_preview_reaches_only_the_declared_column():
    """A D column with no declared preview must be invisible past its current
    step; a column that declared one must not be."""
    outdoor, irr, occ = full_setup()["nm"]["D"]
    s = full_setup(future={outdoor: 2})
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    d_steps = h_ctrl + 2
    nd = len(nm["D"])

    torch.manual_seed(0)
    yn = torch.randn(2, h_ctrl * len(nm["Y"]))
    D = torch.randn(2, d_steps * nd)
    R = torch.randn(2, N_REF)

    net = policy.callable
    base = net(yn, D, R)

    # Last (deepest-future) timestep: irr never declared a preview, so a
    # change there must not reach the net.
    perturbed_irr = D.clone()
    perturbed_irr[:, (d_steps - 1) * nd + nm["D"].index(irr)] += 5.0
    assert torch.allclose(net(yn, perturbed_irr, R), base)

    # Same timestep, outdoor: it declared a 2-step preview, so it must reach.
    perturbed_outdoor = D.clone()
    perturbed_outdoor[:, (d_steps - 1) * nd + nm["D"].index(outdoor)] += 5.0
    assert not torch.allclose(net(yn, perturbed_outdoor, R), base)


def test_the_reference_reaches_the_policy_and_changes_what_it_commands():
    """An input the policy never reads is a dead window. Moving the band must
    move the commanded setpoint."""
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    cl = build_closed_loop(s["dyn_node"], build_policy(cfg, sysid_cfg, nm, ns),
                           cfg, sysid_cfg)
    data = _windows(s)
    here = cl(dict(data))["U"].clone()
    moved = cl({**data, "R": data["R"] + 1.0})["U"]
    assert not torch.allclose(here, moved)


def test_the_policy_reads_the_commands_applied_and_never_the_padded_copy():
    """At rollout step i a U column with `past: p` sees U[i-p .. i-1], the
    commands already applied, oldest first. U[i] does not exist yet, so
    SystemPreview pads the window's current row with a copy of U[i-1]; the
    gather must drop it."""
    s = full_setup(u_past=5)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    assert hmax == 6                      # the U window sets the seed depth

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    assert policy.input_keys == ["yn", "U", "D", "R"]
    seen = []
    net = next(m for m in policy.callable.modules() if isinstance(m, _NormalizedNet))
    net.base.register_forward_pre_hook(
        lambda _module, args: seen.append(args[1].detach().clone()))
    out = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)(_windows(s))

    # The net sees the commands normalised, so compare them normalised.
    (setpoint,) = nm["U"]
    shift, stretch = ns[setpoint]["shift"], ns[setpoint]["stretch"]
    assert seen
    for k, window in enumerate(seen):
        i = hmax - 1 + k
        applied = (out["U"][:, i - 5:i, :] - shift) / stretch
        assert torch.allclose(window, applied.reshape(len(window), -1))


# ── Policy family registry ───────────────────────────────────────


@pytest.mark.parametrize("family, method", [("mlp", "sigmoid_scale"),
                                            ("linear", "relu_clamp")])
def test_bounds_respected_for_every_family_and_method(family, method):
    """Both bound methods, across the families, must keep U inside the
    actuator limits."""
    s = full_setup(policy=family, policy_bound_method=method)
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    U = cl(_windows(s))["U"][:, scored_steps("U", hmax), :]

    umin, umax = control_box(cfg, nm, s["norm_stats"], "cpu")
    tol = 1e-4
    assert torch.all(U >= umin - tol)
    assert torch.all(U <= umax + tol)
