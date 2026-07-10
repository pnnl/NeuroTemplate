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


def _preview_steps(hooks, cfg, nm_mapping: dict[str, list[str]]) -> tuple[int, int]:
    """Resolve (disturbance_future, band_future) preview widths for the policy.

    If the use case defines `preview_config(nm_mapping) -> {signal: n_steps}`,
    use it: keys may be D column names (per-column disturbance preview) or the
    literal "band" (future comfort-band preview). The disturbance window is one
    block, so it is sized to the largest per-column D value requested (columns
    asking for less just see harmless extra lookahead). With no hook, fall back
    to the global `cfg.d_preview` for disturbances and no band preview.
    """
    preview = getattr(hooks, "preview_config", None)
    if preview is None:
        return cfg.d_preview, 0

    spec = preview(nm_mapping)
    allowed = set(nm_mapping["D"]) | {"band"}
    for key, steps in spec.items():
        if key not in allowed:
            raise ValueError(
                f"preview_config key '{key}' is not a D column or 'band'. "
                f"Allowed: {sorted(allowed)}"
            )
        if not isinstance(steps, int) or steps < 0:
            raise ValueError(
                f"preview_config['{key}'] must be a non-negative int, got {steps!r}"
            )

    d_future = max([spec.get(col, 0) for col in nm_mapping["D"]], default=0)
    return d_future, spec.get("band", 0)


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

    # How many future steps each previewed signal feeds the policy. A use case
    # may define control_user_defined.preview_config for per-column disturbance
    # preview and comfort-band preview; otherwise fall back to the global
    # cfg.d_preview (disturbances only, no band preview).
    d_future, band_future = _preview_steps(hooks, cfg, nm_mapping)

    # Flattened window widths (see SystemPreview.get_mapped_data):
    #   yn : past = H_ctrl-1, future = 0                 -> ny * H_ctrl
    #   D  : past = H_ctrl-1, future = d_future          -> nd * (H_ctrl + d_future)
    #   ymin/ymax: current + band_future preview steps   -> n_ctrl*(1+band_future) each
    insize = (
        ny * h_ctrl
        + 2 * n_ctrl * (1 + band_future)
        + nd * (h_ctrl + d_future)
    )

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
        "D": {"past": h_ctrl - 1, "future": d_future, "pad_mode": "nearest"},
    }
    # The comfort band is a known future schedule; previewing it lets the policy
    # see the band it is steering into (the dynamics have a one-step lag, so
    # comfort at step i depends on the setpoint chosen at i-1). Only add the
    # window when previewing — future=0 keeps the legacy single-step behaviour.
    if band_future > 0:
        for key in ("ymin", "ymax"):
            input_map[key] = {"past": 0, "future": band_future,
                              "pad_mode": "nearest"}

    return Node(
        net,
        ["yn", "ymin", "ymax", "D"],
        ["U"],
        name="DPC Policy",
        input_map=input_map,
    )
