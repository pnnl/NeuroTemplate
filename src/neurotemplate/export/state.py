"""
Lift a model's hidden rollout state into explicit tensors.

`RCBlock` keeps its mass temperature and `GRUBlock` its hidden state as plain
attributes that `forward` mutates. A trace that does not lift them bakes
whatever value happened to be there into the graph as a constant, and the
exported step then ignores the state it is handed — a silent error, measured
at ~3e-2 on a GRU. So state lifting is keyed off the stateful-block protocol
(`rollout_state` / `set_rollout_state`, see `dynamics.stateful_blocks`), never
off an attribute name: every block that implements it is lifted, and a block
that does not is stateless by definition.

The model's state is every stateful block's `[batch, width]` state,
concatenated in `modules()` order. `StateSlot` records which stretch of that
vector belongs to which block, so a consumer reading the state trajectory can
tell a GRU's hidden units from a zone's mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from neurotemplate.modeling.dynamics import stateful_blocks


@dataclass(frozen=True)
class StateSlot:
    """One stateful block's stretch of the flat state vector."""
    path: str   # the block's name in `model.named_modules()`
    width: int


def probe_state_slots(model: nn.Module, y: torch.Tensor, u: torch.Tensor,
                      d: torch.Tensor) -> list[StateSlot]:
    """Every stateful block and its state width, from one cold call.

    The width is read off the state the block actually produced rather than
    predicted from its constructor arguments, so a new stateful family needs
    nothing here. The model is left cold afterwards.
    """
    paths = {id(m): name for name, m in model.named_modules()}
    blocks = stateful_blocks(model)
    for block in blocks:
        block.set_rollout_state(None)
    with torch.no_grad():
        model(y, u, d)
    slots = []
    for block in blocks:
        state = block.rollout_state()
        if state is None:
            raise RuntimeError(
                f"stateful block {paths[id(block)]!r} ({type(block).__name__}) "
                "is still cold after a batched forward call; it does not "
                "implement the rollout-state protocol the exporter relies on."
            )
        slots.append(StateSlot(paths[id(block)], int(state.shape[-1])))
        block.set_rollout_state(None)
    return slots


def read_state(blocks: list[nn.Module]) -> torch.Tensor:
    """The blocks' states concatenated, [batch, sum(widths)]."""
    return torch.cat([block.rollout_state() for block in blocks], dim=-1)


def write_state(blocks: list[nn.Module], widths: list[int],
                state: torch.Tensor) -> None:
    """Hand each block its stretch of a flat state vector."""
    offset = 0
    for block, width in zip(blocks, widths):
        block.set_rollout_state(state[:, offset:offset + width])
        offset += width


def clear_state(blocks: list[nn.Module]) -> None:
    """Force every block's next call to be a cold one."""
    for block in blocks:
        block.set_rollout_state(None)
