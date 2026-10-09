from types import SimpleNamespace

import pytest
import torch

from neurotemplate.control.scenarios import build_window_dict, horizons, scored_steps
from neurotemplate.control.policy import (
    POLICY_BOUND_METHODS,
    _PreviewGather,
    _d_preview_by_column,
    _preview_gather,
    build_policy,
    describe_policy,
    policy_catalog,
)
from neurotemplate.control.closed_loop import build_closed_loop
from control_user_defined import actuation_bounds
from control_fixtures import HIGH, LOW, full_setup

_NM = {"Y": ["z"], "U": ["sp"], "D": ["outdoor", "irr"], "t": ["sec"]}
#: The fixture objective's reference columns: the band's two edges.
N_REF = 2


def _windows(s):
    cfg = s["cfg"]
    return build_window_dict(s["norm_df"], cfg, s["sysid_cfg"], s["nm"],
                             P=cfg.train_pred_steps, stride=cfg.in_steps)


def _describe(s):
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    return describe_policy(build_policy(cfg, sysid_cfg, nm, ns), cfg, sysid_cfg, nm, ns)


def test_preview_config_may_name_only_d_columns():
    """A reference column's preview is per run, so it is config, not a hook."""
    hooks = SimpleNamespace(preview_config=lambda nm: {"band": 1})
    with pytest.raises(ValueError, match="not a D column"):
        _d_preview_by_column(hooks, SimpleNamespace(d_preview=0), _NM)


# ── Per-column preview ───────────────────────────────────────────


def test_d_preview_by_column_reads_each_column_own_depth():
    hooks = SimpleNamespace(preview_config=lambda nm: {"outdoor": 2})
    assert _d_preview_by_column(hooks, SimpleNamespace(d_preview=0), _NM) == [2, 0]


def test_d_preview_by_column_falls_back_to_cfg_d_preview_uniformly():
    hooks = SimpleNamespace()  # no preview_config
    assert _d_preview_by_column(hooks, SimpleNamespace(d_preview=3), _NM) == [3, 3]


def test_preview_gather_keeps_only_declared_future_steps():
    # col0: no preview; col1: 2-step preview. history=3 -> 3 past/current
    # positions (both columns kept), then 2 future positions (col1 only).
    idx = _preview_gather(n=2, history=3, depths=[0, 2])
    assert idx.tolist() == [0, 1, 2, 3, 4, 5, 7, 9]


@pytest.mark.parametrize("depths", [[0, 0], [2, 2]])
def test_preview_gather_is_a_no_op_when_every_column_agrees(depths):
    """The one-number case must not build a gather."""
    assert _preview_gather(n=2, history=3, depths=depths) is None


def test_per_column_preview_narrows_insize_to_the_declared_sum(monkeypatch):
    """Two D columns at different declared depths: insize is the per-column
    sum, not nd * (h_ctrl + max_depth)."""
    import control_user_defined as cud
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    outdoor, irr, occ = nm["D"]
    monkeypatch.setattr(cud, "preview_config",
                        lambda nm: {outdoor: 2, irr: 1}, raising=False)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    d = describe_policy(policy, cfg, sysid_cfg, nm, ns)

    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    expected = (
        len(nm["Y"]) * h_ctrl
        + h_ctrl * len(nm["D"]) + (2 + 1 + 0)   # outdoor=2, irr=1, occ=0
        + N_REF
    )
    assert d["insize"] == expected
    assert d["d_preview_by_column"] == {outdoor: 2, irr: 1, occ: 0}
    assert any(isinstance(m, _PreviewGather) for m in policy.callable.modules())


def test_per_column_preview_reaches_only_the_declared_column(monkeypatch):
    """A D column with no declared preview must be invisible past its current
    step; a column that declared one must not be."""
    import control_user_defined as cud
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    outdoor, irr, occ = nm["D"]
    monkeypatch.setattr(cud, "preview_config",
                        lambda nm: {outdoor: 2}, raising=False)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    in_steps = h_ctrl + 2
    nd = len(nm["D"])

    torch.manual_seed(0)
    yn = torch.randn(2, h_ctrl * len(nm["Y"]))
    D = torch.randn(2, in_steps * nd)
    R = torch.randn(2, N_REF)

    net = policy.callable
    base = net(yn, D, R)

    # Last (deepest-future) timestep: irr never declared a preview, so a
    # change there must not reach the net.
    perturbed_irr = D.clone()
    perturbed_irr[:, (in_steps - 1) * nd + nm["D"].index(irr)] += 5.0
    assert torch.allclose(net(yn, perturbed_irr, R), base)

    # Same timestep, outdoor: it declared a 2-step preview, so it must reach.
    perturbed_outdoor = D.clone()
    perturbed_outdoor[:, (in_steps - 1) * nd + nm["D"].index(outdoor)] += 5.0
    assert not torch.allclose(net(yn, perturbed_outdoor, R), base)


def test_reference_preview_widens_the_input_by_each_columns_depth():
    """Each reference column adds its current step plus its own preview."""
    base = _describe(full_setup())
    grown = _describe(full_setup(reference_preview={LOW: 3, HIGH: 1}))
    assert grown["insize"] == base["insize"] + 3 + 1
    assert grown["r_preview_by_column"] == {HIGH: 1, LOW: 3}


def test_policy_node_io():
    s = full_setup()
    policy = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    assert policy.input_keys == ["yn", "D", "R"]
    assert policy.output_keys == ["U"]


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


# ── Policy family registry ───────────────────────────────────────


def test_linear_policy_is_a_single_affine_layer():
    s = full_setup(policy="linear")
    net = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]).callable
    assert len(net.linear) == 1
    assert net.linear[0].out_features == len(s["nm"]["U"])


@pytest.mark.parametrize("method", POLICY_BOUND_METHODS)
@pytest.mark.parametrize("family", list(policy_catalog))
def test_bounds_respected_for_every_family_and_method(family, method):
    """Both bound methods, on both families, must keep U inside the actuator
    limits."""
    s = full_setup(policy=family, policy_bound_method=method)
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    U = cl(_windows(s))["U"][:, scored_steps("U", hmax), :]

    umin, umax = actuation_bounds(nm, s["norm_stats"])
    tol = 1e-4
    assert torch.all(U >= umin - tol)
    assert torch.all(U <= umax + tol)


# ── describe_policy ──────────────────────────────────────────────


def test_describe_policy_reports_shape_and_physical_bounds():
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    d = describe_policy(policy, cfg, sysid_cfg, nm, ns)

    assert d["family"] == "mlp"
    assert d["outsize"] == len(nm["U"])
    assert d["hsizes"] == cfg.policy_layers * [2**cfg.log2_policy_hidden]

    # insize must equal the documented window sum.
    assert d["insize"] == (
        len(nm["Y"]) * d["h_ctrl"]
        + len(nm["D"]) * d["h_ctrl"] + sum(d["d_preview_by_column"].values())
        + N_REF + sum(d["r_preview_by_column"].values())
    )

    # Bounds are reported in physical units, matching the hook's own inputs.
    for i, col in enumerate(nm["U"]):
        assert d["umin_phys"][i] == pytest.approx(ns[col]["min"], rel=1e-4)
        assert d["umax_phys"][i] == pytest.approx(ns[col]["max"], rel=1e-4)
