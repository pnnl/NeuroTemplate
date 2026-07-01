import pytest
import torch

from neurotemplate.control.scenarios import build_window_dict, horizons
from neurotemplate.control.policy import build_policy, STEQuantize
from neurotemplate.control.closed_loop import build_closed_loop
from control_user_defined import actuation_bounds
from control_fixtures import full_setup


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
