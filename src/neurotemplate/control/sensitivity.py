"""
Whether the surrogate's outputs move with its commands: the one thing a
controller needs from it.

A policy trains (DPC), and MPC plans, by differentiating the surrogate's
outputs with respect to U. Two failures look alike in a finished run (a
controller that never moves) and mean different things:

  - No gradient path at all: autograd has no edge from y to u, because
    something in the model leaves the graph (numpy, `.item()`, `.detach()`,
    an integer op such as argmax). Nothing can be controlled through that
    model, so `command_sensitivity` raises.
  - A path whose derivative is exactly zero: a saturated clamp or sigmoid, a
    dead ReLU, a rounding op, or a command the model learned to ignore. It
    may be the operating point rather than the model, so it is measured per
    command over windows of recorded data and reported.

Each window is the model's own contract call, from cold: one step, y_next
against the whole u window.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from neurotemplate._typing import beartype
from neurotemplate.modeling.contract import contract_model
from neurotemplate.modeling.dynamics import stateful_blocks


@dataclass(frozen=True)
class CommandSensitivity:
    column: str
    zero_share: float   # of the sampled windows where y_next does not move with it


def _recorded_windows(frames: list, nm_mapping: dict[str, list[str]], history_steps: int,
                      count: int) -> tuple[torch.Tensor, ...]:
    """Up to `count` windows of `history_steps` recorded rows, evenly spaced
    over every frame: (y, u, d), each [windows, history_steps, n]."""
    arrays = [{key: frame[nm_mapping[key]].to_numpy(dtype=np.float32) for key in "YUD"}
              for frame in frames]
    starts = [(i, row) for i, frame in enumerate(frames)
              for row in range(len(frame) - history_steps + 1)]
    picked = starts[::max(1, len(starts) // count)][:count]
    return tuple(
        torch.from_numpy(np.stack([arrays[i][key][row:row + history_steps]
                                   for i, row in picked]))
        for key in "YUD")


@beartype
def command_sensitivity(dyn_sys, frames: list, nm_mapping: dict[str, list[str]],
                        device: torch.device, windows: int = 256) -> list[CommandSensitivity]:
    """Per U column, the share of recorded windows where no output's next
    value moves with any row of that column. Raises when no output has a
    gradient path to U at all."""
    # A cold copy: each window seeds its own state, and the run's model keeps its own.
    model = copy.deepcopy(contract_model(dyn_sys))
    for block in stateful_blocks(model):
        block.set_rollout_state(None)
    history_steps = dyn_sys.nodes[0].callable.history_steps
    y, u, d = (x.to(device) for x in
               _recorded_windows(frames, nm_mapping, history_steps, windows))
    u.requires_grad_(True)

    # |∂y_next[k]/∂u| summed over outputs and window rows: [windows, nu]. One
    # backward per output, so outputs of opposite sign cannot cancel.
    magnitude = torch.zeros(u.shape[0], u.shape[2], device=device)
    connected = False
    with torch.enable_grad():
        y_next = model(y, u, d)
        if y_next.requires_grad:
            for k in range(y_next.shape[1]):
                grad, = torch.autograd.grad(y_next[:, k].sum(), u, retain_graph=True,
                                            allow_unused=True)
                if grad is not None:
                    connected = True
                    magnitude += grad.abs().sum(dim=1)
    if not connected:
        raise ValueError(
            f"The surrogate's outputs have no gradient path to its commands "
            f"{nm_mapping['U']}: something in model {type(model).__name__} leaves "
            "the autograd graph (numpy, .item(), .detach(), an integer op such as "
            "argmax). A policy trains, and MPC plans, through that gradient, so "
            "nothing can be controlled through this model.")
    zero = (magnitude == 0).float().mean(dim=0)
    return [CommandSensitivity(column, float(share))
            for column, share in zip(nm_mapping["U"], zero)]
