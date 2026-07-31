from types import SimpleNamespace

import pytest
import torch

from neurotemplate.control.scenarios import build_window_dict, horizons
from neurotemplate.control.policy import (
    POLICY_BOUND_METHODS,
    POLICY_NONLIN,
    STEQuantize,
    _preview_steps,
    build_policy,
    describe_policy,
    policy_catalog,
)
from neurotemplate.control.closed_loop import build_closed_loop
from control_user_defined import actuation_bounds
from control_fixtures import full_setup

_NM = {"Y": ["z"], "U": ["sp"], "D": ["outdoor", "irr"], "t": ["sec"]}


def test_preview_steps_falls_back_to_d_preview_without_hook():
    hooks = SimpleNamespace()  # no preview_config attribute
    cfg = SimpleNamespace(d_preview=2)
    assert _preview_steps(hooks, cfg, _NM) == (2, 0)


def test_preview_steps_reads_hook_and_takes_max_d():
    # outdoor asks for 1 future step, band for 3; the single D window takes the
    # largest per-column D request.
    hooks = SimpleNamespace(preview_config=lambda nm: {"band": 3, "outdoor": 1})
    cfg = SimpleNamespace(d_preview=0)
    assert _preview_steps(hooks, cfg, _NM) == (1, 3)


def test_preview_steps_rejects_unknown_signal():
    hooks = SimpleNamespace(preview_config=lambda nm: {"bogus": 1})
    cfg = SimpleNamespace(d_preview=0)
    with pytest.raises(ValueError, match="not a D column or 'band'"):
        _preview_steps(hooks, cfg, _NM)


def test_preview_steps_rejects_negative():
    hooks = SimpleNamespace(preview_config=lambda nm: {"band": -1})
    cfg = SimpleNamespace(d_preview=0)
    with pytest.raises(ValueError, match="non-negative int"):
        _preview_steps(hooks, cfg, _NM)


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


def test_policy_output_within_actuator_bounds():
    s = full_setup(setpoint_step=0.0)  # avoid quantization rounding past bounds
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    policy = build_policy(cfg, sysid_cfg, nm, s["norm_stats"])
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    data = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, s["norm_stats"],
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    )
    out = cl(data)

    umin, umax = actuation_bounds(nm, s["norm_stats"])
    U = out["U"][:, hmax:, :]  # policy-produced portion
    tol = 1e-4
    assert torch.all(U >= umin - tol)
    assert torch.all(U <= umax + tol)


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


def test_ste_quantize_rejects_nonpositive_step():
    with pytest.raises(ValueError, match="step must be > 0"):
        STEQuantize(0.0)
    with pytest.raises(ValueError, match="step must be > 0"):
        STEQuantize(-0.5)


def test_build_policy_skips_quantization_when_step_zero():
    s = full_setup(setpoint_step=0.0)
    policy = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    assert not any(isinstance(m, STEQuantize) for m in policy.callable.modules())


def test_build_policy_includes_quantization_when_step_set():
    s = full_setup(setpoint_step=0.5)
    policy = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    quantizers = [m for m in policy.callable.modules() if isinstance(m, STEQuantize)]
    assert len(quantizers) == 1
    # cfg step is physical °C; STEQuantize holds the normalized step
    # (cfg.setpoint_step / stretch) and offset (shift / stretch) per U channel.
    expected_step = torch.tensor(
        [0.5 / s["norm_stats"][c]["stretch"] for c in s["nm"]["U"]],
        dtype=torch.float32,
    )
    expected_offset = torch.tensor(
        [s["norm_stats"][c]["shift"] / s["norm_stats"][c]["stretch"]
         for c in s["nm"]["U"]],
        dtype=torch.float32,
    )
    assert torch.allclose(quantizers[0].step, expected_step)
    assert torch.allclose(quantizers[0].offset, expected_offset)


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


def test_unknown_policy_lists_available():
    s = full_setup(policy="bogus")
    with pytest.raises(ValueError, match="Unknown policy 'bogus'"):
        build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])


def test_linear_policy_is_a_single_affine_layer():
    s = full_setup(policy="linear", setpoint_step=0.0)
    net = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]).callable
    assert len(net.linear) == 1
    assert net.linear[0].out_features == len(s["nm"]["U"])


def test_linear_policy_has_fewer_params_than_mlp():
    kw = dict(setpoint_step=0.0)
    lin = full_setup(policy="linear", **kw)
    mlp = full_setup(policy="mlp", **kw)

    def n_params(s):
        net = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]).callable
        return sum(p.numel() for p in net.parameters())

    assert n_params(lin) < n_params(mlp)


@pytest.mark.parametrize("nonlin", list(POLICY_NONLIN))
def test_every_nonlin_builds(nonlin):
    s = full_setup(policy_nonlin=nonlin, setpoint_step=0.0)
    policy = build_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    assert isinstance(policy.callable.nonlin[0], POLICY_NONLIN[nonlin])


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
    assert d["insize"] == (
        len(nm["Y"]) * d["h_ctrl"]
        + 2 * n_ctrl * (1 + d["band_preview"])
        + len(nm["D"]) * (d["h_ctrl"] + d["d_preview"])
    )

    # Bounds are reported in physical units, matching the hook's own inputs.
    for i, col in enumerate(nm["U"]):
        assert d["umin_phys"][i] == pytest.approx(ns[col]["min"], rel=1e-4)
        assert d["umax_phys"][i] == pytest.approx(ns[col]["max"], rel=1e-4)


def test_describe_policy_marks_linear_nonlin_not_applicable():
    s = full_setup(policy="linear", setpoint_step=0.0)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    d = describe_policy(build_policy(cfg, sysid_cfg, nm, ns),
                        cfg, sysid_cfg, nm, ns)
    assert d["hsizes"] == []
    assert d["nonlin"] == "n/a (linear)"
    assert d["quantized"] is False
