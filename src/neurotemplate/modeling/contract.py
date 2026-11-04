"""
The model contract: what a dynamics model is, to everything around it.

    model(y, u, d) -> y_next

    y  [batch, history_steps, ny]   the last history_steps outputs, oldest first
    u  [batch, history_steps, nu]   the matching inputs; row -1 drives this step
    d  [batch, history_steps, nd]   the matching disturbances
    ->  y_next [batch, ny]

Everything is in physical units, the data's own. Training, the physics
checks, control and export all call a model this way and nothing else, so
anything that honours it gets all of them. There is one kind of model: the
built-in families are `stepper.Stepper`s, and a model of your own, registered
by name in the hooks file's `custom_models()`, is any module with this call.

A model says more about itself only if it has more to say, and everything
around it asks with `getattr`:

    rollout_state() / set_rollout_state(state)
                  state past its window, one [batch, width] tensor, None when
                  cold: cold on the first call of a rollout, it seeds the state
                  from its window; warm, it carries it (see `stateful_blocks`)
    describe()    what the report card prints about it
    head_stack    output heads, in the units its `z` is in: the level metrics,
                  threshold calibration and the loss read them
    pre_stack     derived inputs: the physics checks read which U is a demand
    sign_step(y, u, d)
                  the map the gradient sign penalty differentiates, in
                  normalised units, instead of the contract's call

A `BoundedParameter` under it is found wherever it is.

The rollout (`SystemPreview`) hands a node one flat window per key;
`Windowed` reshapes those into the contract's.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Windowed(nn.Module):
    """A contract model as a rollout node's callable.

    `SystemPreview` hands the node `[batch, history_steps * n]` per key; the
    contract takes `[batch, history_steps, n]`.
    """

    def __init__(self, model: nn.Module, widths: tuple[int, int, int], history_steps: int):
        super().__init__()
        self.model = model
        self.widths = tuple(widths)
        self.history_steps = history_steps

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.model(*(x.reshape(x.shape[0], self.history_steps, width)
                            for x, width in zip((y, u, d), self.widths)))


def contract_model(dyn_sys) -> nn.Module:
    """The model behind a dynamics system's node, as the contract sees it."""
    return dyn_sys.nodes[0].callable.model


def stateful_blocks(model: nn.Module) -> list[nn.Module]:
    """Every submodule carrying rollout state, in `modules()` order.

    A block is stateful exactly when it implements the protocol
    `rollout_state()` / `set_rollout_state(state)`: the state as one
    [batch, width] tensor, None when cold. A new stateful part implements those
    two methods and gets both the per-batch reset and ONNX export (see
    `neurotemplate.export.state`) for free.
    """
    # On the class: a wrapper forwarding unknown attributes to the module inside
    # it is not a second stateful block.
    return [m for m in model.modules() if hasattr(type(m), "set_rollout_state")]


def named_columns(nm_mapping: dict[str, list[str]], y: torch.Tensor, u: torch.Tensor,
                  d: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every column of the contract's windows by name, `{column: [batch, rows]}`:
    a model of your own reads `c["Zone Temp"]` rather than `y[:, :, 0]`."""
    columns = {}
    for key, window in zip(("Y", "U", "D"), (y, u, d)):
        columns.update(zip(nm_mapping[key], window.unbind(-1)))
    return columns
