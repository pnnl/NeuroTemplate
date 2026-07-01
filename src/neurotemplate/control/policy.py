"""
DPC control policy.

A bounded MLP wrapped in a Neuromancer Node:

    [yn-window, ymin, ymax, D-window]  ->  U   (per-zone cooling setpoints)

Output is squashed into the actuator limits (normalized U space) by
blocks.MLP_bounds. Windowing uses the same input_map convention as
src/models.get_dynamics so the policy composes with the frozen dynamics Node.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from neuromancer.system import Node
from neuromancer.modules import blocks

from neurotemplate.control.scenarios import horizons
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


class STEQuantize(nn.Module):
    """Snap to a setpoint grid in the forward pass; identity in the backward.

    Lets the policy learn against a continuous surrogate while emitting only
    grid-feasible setpoints (e.g. half-degree thermostat steps). The
    `(round(.) - u).detach()` trick zeros the gradient contribution of the
    snap, so ∂out/∂u = 1.

    Snap targets the grid `{k*step - offset : integer k}` in the same space
    as the input. For the policy this is normalized U: caller passes
    `step = step_phys / stretch` and `offset = shift / stretch` so the
    *physical* grid lands at multiples of `step_phys`.

    `step` and `offset` may be scalars or 1-D tensors of shape [nu]
    (per-channel grid, broadcast over batch/time).
    """

    def __init__(self, step, offset=0.0):
        super().__init__()
        if isinstance(step, torch.Tensor):
            if (step <= 0).any():
                raise ValueError(f"STEQuantize step entries must be > 0, got {step}")
            self.register_buffer("step", step.float())
        else:
            if step <= 0:
                raise ValueError(f"STEQuantize step must be > 0, got {step}")
            self.step = float(step)

        if isinstance(offset, torch.Tensor):
            self.register_buffer("offset", offset.float())
        else:
            self.offset = float(offset)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        snapped = ((u + self.offset) / self.step).round() * self.step - self.offset
        return u + (snapped - u).detach()


class _QuantizedPolicy(nn.Module):
    """Compose the bounded MLP with STEQuantize while keeping the variadic
    forward signature the Neuromancer Node expects."""

    def __init__(self, base: nn.Module, step, offset=0.0):
        super().__init__()
        self.base = base
        self.quantize = STEQuantize(step, offset)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.base(*args))


@beartype
def build_policy(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> Node:
    ny = len(nm_mapping["Y"])
    nu = len(nm_mapping["U"])
    nd = len(nm_mapping["D"])
    hooks = load_control_hooks(cfg)
    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    n_ctrl = len(hooks.controlled_channels(nm_mapping)[0])
    device = sysid_cfg.device

    # Flattened window widths (see SystemPreview.get_mapped_data):
    #   yn : past = H_ctrl-1, future = 0           -> ny * H_ctrl
    #   D  : past = H_ctrl-1, future = d_preview   -> nd * (H_ctrl + d_preview)
    #   ymin/ymax: single current step             -> n_ctrl each
    insize = ny * h_ctrl + 2 * n_ctrl + nd * (h_ctrl + cfg.d_preview)

    umin, umax = hooks.actuation_bounds(nm_mapping, norm_stats)
    umin, umax = umin.to(device), umax.to(device)

    net = blocks.MLP_bounds(
        insize=insize,
        outsize=nu,
        bias=True,
        linear_map=nn.Linear,
        nonlin=nn.GELU,
        hsizes=cfg.policy_layers * [2**cfg.log2_policy_hidden],
        min=umin,
        max=umax,
        method="sigmoid_scale",
    ).to(device)

    if cfg.setpoint_step > 0:
        # Quantization is in physical °C, but the policy emits normalized U;
        # convert step + shift per channel so the physical grid lands at
        # multiples of cfg.setpoint_step (not multiples of step shifted by
        # the column mean).
        u_stretch = torch.tensor(
            [norm_stats[c]["stretch"] for c in nm_mapping["U"]],
            dtype=torch.float32, device=device,
        )
        u_shift = torch.tensor(
            [norm_stats[c]["shift"] for c in nm_mapping["U"]],
            dtype=torch.float32, device=device,
        )
        step_norm = cfg.setpoint_step / u_stretch
        offset_norm = u_shift / u_stretch
        net = _QuantizedPolicy(net, step_norm, offset_norm).to(device)

    input_map = {
        "yn": {"past": h_ctrl - 1, "future": 0, "pad_mode": "nearest"},
        "D": {"past": h_ctrl - 1, "future": cfg.d_preview,
              "pad_mode": "nearest"},
    }

    return Node(
        net,
        ["yn", "ymin", "ymax", "D"],
        ["U"],
        name="DPC Policy",
        input_map=input_map,
    )
