from types import SimpleNamespace

import pytest
import torch

from neurotemplate.control.scenarios import build_window_dict, horizons
from neurotemplate.control.policy import (
    POLICY_BOUND_METHODS,
    STEQuantize,
    _PreviewGather,
    _d_preview_by_column,
    _d_preview_gather,
    _preview_steps,
    build_policy,
    describe_policy,
    policy_catalog,
    regulated_channels,
)
from neurotemplate.control.closed_loop import build_closed_loop
from control_user_defined import actuation_bounds
from control_fixtures import full_setup, install_price

_NM = {"Y": ["z"], "U": ["sp"], "D": ["outdoor", "irr"], "t": ["sec"]}


def test_preview_steps_falls_back_to_d_preview_without_hook():
    hooks = SimpleNamespace()  # no preview_config attribute
    cfg = SimpleNamespace(d_preview=2, price_preview=0)
    assert _preview_steps(hooks, cfg, _NM) == (2, 0, 0)


def test_preview_steps_reads_hook_and_takes_max_d():
    # outdoor asks for 1 future step, band for 3; the single D window takes the
    # largest per-column D request.
    hooks = SimpleNamespace(preview_config=lambda nm: {"band": 3, "outdoor": 1})
    cfg = SimpleNamespace(d_preview=0, price_preview=0)
    assert _preview_steps(hooks, cfg, _NM) == (1, 3, 0)


def test_price_preview_comes_from_the_config_not_the_hook():
    """It is a per-run knob, so it must not need the hooks file to change."""
    hooks = SimpleNamespace()
    cfg = SimpleNamespace(d_preview=0, price_preview=4)
    assert _preview_steps(hooks, cfg, _NM) == (0, 0, 4)

    with_hook = SimpleNamespace(preview_config=lambda nm: {"band": 2})
    assert _preview_steps(with_hook, cfg, _NM) == (0, 2, 4)


def test_preview_steps_rejects_unknown_signal():
    hooks = SimpleNamespace(preview_config=lambda nm: {"bogus": 1})
    cfg = SimpleNamespace(d_preview=0, price_preview=0)
    with pytest.raises(ValueError, match="not a D column or 'band'"):
        _preview_steps(hooks, cfg, _NM)


def test_regulated_channels_requires_the_hook():
    hooks = SimpleNamespace()
    with pytest.raises(AttributeError, match="regulated"):
        regulated_channels(hooks, _NM)


# ── Per-column D preview ──────────────────────────────────────────


def test_d_preview_by_column_reads_each_column_own_depth():
    hooks = SimpleNamespace(preview_config=lambda nm: {"outdoor": 2})
    cfg = SimpleNamespace(d_preview=0, price_preview=0)
    assert _d_preview_by_column(hooks, cfg, _NM) == [2, 0]


def test_d_preview_by_column_falls_back_to_cfg_d_preview_uniformly():
    hooks = SimpleNamespace()  # no preview_config
    cfg = SimpleNamespace(d_preview=3, price_preview=0)
    assert _d_preview_by_column(hooks, cfg, _NM) == [3, 3]


def test_d_preview_gather_keeps_only_declared_future_steps():
    # col0: no preview; col1: 2-step preview. h_ctrl=3 -> 3 past/current
    # positions (both columns kept), then 2 future positions (col1 only).
    idx = _d_preview_gather(nd=2, h_ctrl=3, d_future=2, d_depths=[0, 2])
    assert idx.tolist() == [0, 1, 2, 3, 4, 5, 7, 9]


@pytest.mark.parametrize("d_future,d_depths", [
    (0, [0, 0]),      # the all-zero default
    (2, [2, 2]),      # uniform nonzero, e.g. cfg.d_preview applied to both
])
def test_d_preview_gather_is_a_no_op_when_every_column_agrees(d_future, d_depths):
    """The one-number case (today's only case) must not build a gather."""
    assert _d_preview_gather(nd=2, h_ctrl=3, d_future=d_future,
                             d_depths=d_depths) is None


def test_per_column_preview_narrows_insize_to_the_declared_sum(monkeypatch):
    """Two D columns at different declared depths: insize is the per-column
    sum, not nd * (h_ctrl + max_depth)."""
    import control_user_defined as cud
    s = full_setup(setpoint_step=0.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    outdoor, irr, occ = nm["D"]
    monkeypatch.setattr(cud, "preview_config",
                        lambda nm: {outdoor: 2, irr: 1}, raising=False)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    d = describe_policy(policy, cfg, sysid_cfg, nm, ns)

    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    n_ctrl = 1  # single controlled zone temp in the fixture
    expected = (
        len(nm["Y"]) * h_ctrl
        + 2 * n_ctrl * 1
        + h_ctrl * len(nm["D"]) + (2 + 1 + 0)   # outdoor=2, irr=1, occ=0
    )
    assert d["insize"] == expected
    assert d["d_preview_by_column"] == {outdoor: 2, irr: 1, occ: 0}
    assert any(isinstance(m, _PreviewGather) for m in policy.callable.modules())


def test_per_column_preview_reaches_only_the_declared_column(monkeypatch):
    """A D column with no declared preview must be invisible past its current
    step; a column that declared one must not be."""
    import control_user_defined as cud
    s = full_setup(setpoint_step=0.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    outdoor, irr, occ = nm["D"]
    monkeypatch.setattr(cud, "preview_config",
                        lambda nm: {outdoor: 2}, raising=False)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    d_future = 2
    in_steps = h_ctrl + d_future
    nd = len(nm["D"])

    torch.manual_seed(0)
    yn = torch.randn(2, h_ctrl * len(nm["Y"]))
    ymin = torch.randn(2, 1)
    ymax = torch.randn(2, 1)
    D = torch.randn(2, in_steps * nd)

    net = policy.callable
    base = net(yn, ymin, ymax, D)

    # Last (deepest-future) timestep: irr never declared a preview, so a
    # change there must not reach the net.
    perturbed_irr = D.clone()
    perturbed_irr[:, (in_steps - 1) * nd + nm["D"].index(irr)] += 5.0
    assert torch.allclose(net(yn, ymin, ymax, perturbed_irr), base)

    # Same timestep, outdoor: it declared a 2-step preview, so it must reach.
    perturbed_outdoor = D.clone()
    perturbed_outdoor[:, (in_steps - 1) * nd + nm["D"].index(outdoor)] += 5.0
    assert not torch.allclose(net(yn, ymin, ymax, perturbed_outdoor), base)


def test_band_preview_widens_policy_input(monkeypatch):
    """A band-preview hook must grow the policy MLP's input width by the extra
    ymin/ymax steps (n_ctrl per side, per previewed step)."""
    import control_user_defined as cud
    s = full_setup(setpoint_step=0.0)
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]

    def insize_of(policy):
        return policy.callable.linear[0].in_features

    base = insize_of(build_policy(cfg, sysid_cfg, nm, s["norm_stats"]))
    monkeypatch.setattr(cud, "preview_config", lambda nm: {"band": 2},
                        raising=False)
    grown = insize_of(build_policy(cfg, sysid_cfg, nm, s["norm_stats"]))

    n_ctrl = 1  # single controlled zone temp in the fixture
    assert grown == base + 2 * n_ctrl * 2


def test_policy_node_io():
    s = full_setup()
    policy = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    assert policy.input_keys == ["yn", "ymin", "ymax", "D"]
    assert policy.output_keys == ["U"]


def test_price_adds_one_policy_input_per_previewed_step(tmp_path):
    """The tariff is a single column, so the policy's input width grows by
    exactly 1 + price_preview."""
    s = full_setup(setpoint_step=0.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]

    def insize_of():
        return build_policy(cfg, sysid_cfg, nm, ns).callable.linear[0].in_features

    base = insize_of()
    install_price(cfg, tmp_path)
    assert insize_of() == base + 1

    cfg.price_preview = 3
    assert insize_of() == base + 4


def test_price_reaches_the_policy_and_changes_what_it_commands(tmp_path):
    """An input the policy never reads is a dead window. Doubling the tariff
    must move the commanded setpoint."""
    s = full_setup(setpoint_step=0.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    install_price(cfg, tmp_path)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    assert policy.input_keys == ["yn", "ymin", "ymax", "D", "price"]

    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    data = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    )
    cheap = cl(data)["U"].clone()
    dear = cl({**data, "price": data["price"] * 2.0})["U"]
    assert not torch.allclose(cheap, dear)


def test_ste_quantize_snaps_forward():
    q = STEQuantize(0.5)
    x = torch.tensor([22.78, 25.7, 30.54, 28.25, 23.0])
    expected = torch.tensor([23.0, 25.5, 30.5, 28.0, 23.0])
    assert torch.allclose(q(x), expected)


def test_ste_quantize_gradient_is_identity():
    q = STEQuantize(0.5)
    x = torch.tensor([24.3, 27.8], requires_grad=True)
    q(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_policy_output_lands_on_physical_grid():
    s = full_setup(setpoint_step=0.5)
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    data = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, s["norm_stats"],
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    )
    out = cl(data)

    # Denormalize the policy-produced U back to °C and assert each value sits
    # on a 0.5 °C grid (modulo numerical noise).
    U = out["U"][:, hmax:, :].detach()
    for j, col in enumerate(nm["U"]):
        s_col = s["norm_stats"][col]
        u_phys = U[:, :, j] * s_col["stretch"] + s_col["shift"]
        residual = (u_phys / 0.5 - (u_phys / 0.5).round()).abs()
        assert residual.max().item() < 1e-3, (
            f"{col}: max grid residual {residual.max().item()}"
        )


# ── Policy family registry ───────────────────────────────────────


def test_linear_policy_is_a_single_affine_layer():
    s = full_setup(policy="linear", setpoint_step=0.0)
    net = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]).callable
    assert len(net.linear) == 1
    assert net.linear[0].out_features == len(s["nm"]["U"])


@pytest.mark.parametrize("method", POLICY_BOUND_METHODS)
@pytest.mark.parametrize("family", list(policy_catalog))
def test_bounds_respected_for_every_family_and_method(family, method):
    """Both bound methods, on both families, must keep U inside the actuator
    limits. setpoint_step=0 because the quantizer sits *after* the bound and
    can legitimately round a hair past it."""
    s = full_setup(policy=family, policy_bound_method=method, setpoint_step=0.0)
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    data = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, s["norm_stats"],
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    )
    U = cl(data)["U"][:, hmax:, :]

    umin, umax = actuation_bounds(nm, s["norm_stats"])
    tol = 1e-4
    assert torch.all(U >= umin - tol)
    assert torch.all(U <= umax + tol)


# ── describe_policy ──────────────────────────────────────────────


def test_describe_policy_reports_shape_and_physical_bounds():
    s = full_setup(setpoint_step=0.5)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    d = describe_policy(policy, cfg, sysid_cfg, nm, ns)

    assert d["family"] == "mlp"
    assert d["quantized"] is True
    assert d["outsize"] == len(nm["U"])
    assert d["hsizes"] == cfg.policy_layers * [2**cfg.log2_policy_hidden]

    # insize must equal the documented window sum.
    n_ctrl = 1  # single controlled zone temp in the fixture
    assert d["reads_price"] is False
    assert d["insize"] == (
        len(nm["Y"]) * d["h_ctrl"]
        + 2 * n_ctrl * (1 + d["band_preview"])
        + len(nm["D"]) * (d["h_ctrl"] + d["d_preview"])
    )

    # Bounds are reported in physical units, matching the hook's own inputs.
    for i, col in enumerate(nm["U"]):
        assert d["umin_phys"][i] == pytest.approx(ns[col]["min"], rel=1e-4)
        assert d["umax_phys"][i] == pytest.approx(ns[col]["max"], rel=1e-4)
