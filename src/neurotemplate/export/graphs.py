"""
The two graphs an exported dynamics model ships as: the model contract
(`modeling.contract`), traced.

    y  [batch, history_steps, ny]   the last history_steps outputs, oldest first
    u  [batch, history_steps, nu]   the matching inputs; row -1 drives this step
    d  [batch, history_steps, nd]   the matching disturbances
    ->  y_next [batch, ny]

Physical units in, physical units out, as the contract already is: whatever
a model does about units (a built-in family normalises) is inside the graph.
That is the design decision the export rests on: a driver that holds no
numeric logic is short in any language and cannot drift from this one.

Two graphs, because cold start and warm step are structurally different:
`RCBlock` and `GRUBlock` both branch in Python on whether they hold state, so
a trace captures one branch or the other.

  WarmStart  (y, u, d)        -> (y_next, state)       the cold path: seeds
                                                        the state from the window
  Step       (y, u, d, state) -> (y_next, state_next)  state explicit

A stateless model has no state to carry, so both graphs drop the `state`
input and output and compute the same thing. Both are still shipped, so a
runtime never needs to know which kind of model it is driving.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neurotemplate.export.state import StateSlot, clear_state, read_state, write_state


class _Graph(nn.Module):
    """What both graphs share: the contract model and the handles on every
    stateful block."""

    def __init__(self, model: nn.Module, slots: list[StateSlot]):
        super().__init__()
        self.model = model
        self.stateful = [model.get_submodule(slot.path) for slot in slots]
        self.widths = [slot.width for slot in slots]

    def _outputs(self, y_next: torch.Tensor):
        if not self.stateful:
            return y_next
        return y_next, read_state(self.stateful)


class WarmStart(_Graph):
    """(y, u, d) -> (y_next, state): the first step of a rollout."""

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor):
        clear_state(self.stateful)
        return self._outputs(self.model(y, u, d))


class Step(_Graph):
    """(y, u, d, state) -> (y_next, state_next): every step after the first."""

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor,
                state: torch.Tensor | None = None):
        if self.stateful:
            write_state(self.stateful, self.widths, state)
        return self._outputs(self.model(y, u, d))
