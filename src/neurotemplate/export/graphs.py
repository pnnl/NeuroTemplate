"""
The two graphs an exported dynamics model ships as, with physical units baked in.

A trained `SSMGrad` reads normalised, key-major flattened windows (see
`modeling.windows`). The wrappers here fold everything around it into the
graph — normalisation, window flattening and de-normalisation — so the graph
signature is physical units in, physical units out:

    y  [batch, history_steps, ny]   the last history_steps outputs, oldest first
    u  [batch, history_steps, nu]   the matching inputs; row -1 drives this step
    d  [batch, history_steps, nd]   the matching disturbances
    ->  y_next [batch, ny]

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


def _stats_tensor(norm_stats: dict[str, dict], columns: list[str],
                  field: str) -> torch.Tensor:
    return torch.tensor([float(norm_stats[c][field]) for c in columns],
                        dtype=torch.float32)


class _PhysicalGraph(nn.Module):
    """What both graphs share: the norm stats as buffers, the window layout,
    and the handles on every stateful block."""

    def __init__(self, model: nn.Module, nm_mapping: dict[str, list[str]],
                 norm_stats: dict[str, dict], slots: list[StateSlot]):
        super().__init__()
        self.model = model
        for key in ("Y", "U", "D"):
            name = key.lower()
            columns = nm_mapping[key]
            self.register_buffer(f"{name}_shift", _stats_tensor(norm_stats, columns, "shift"))
            self.register_buffer(f"{name}_stretch", _stats_tensor(norm_stats, columns, "stretch"))
        self.stateful = [model.get_submodule(slot.path) for slot in slots]
        self.widths = [slot.width for slot in slots]

    @staticmethod
    def _window(x: torch.Tensor, shift: torch.Tensor,
                stretch: torch.Tensor) -> torch.Tensor:
        """[batch, history_steps, n] physical -> [batch, history_steps*n] normalised.

        One key's block of the window `modeling.windows` describes: time
        ordered, oldest first, every column of a step together. A row-major
        reshape is exactly that; a transpose first (column-major) would agree
        only when the key has a single column.
        """
        return ((x - shift) / stretch).reshape(x.shape[0], -1)

    def _predict(self, y: torch.Tensor, u: torch.Tensor,
                 d: torch.Tensor) -> torch.Tensor:
        out = self.model(
            self._window(y, self.y_shift, self.y_stretch),
            self._window(u, self.u_shift, self.u_stretch),
            self._window(d, self.d_shift, self.d_stretch),
        )
        return out * self.y_stretch + self.y_shift

    def _outputs(self, y_next: torch.Tensor):
        if not self.stateful:
            return y_next
        return y_next, read_state(self.stateful)


class WarmStart(_PhysicalGraph):
    """(y, u, d) -> (y_next, state): the first step of a rollout."""

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor):
        clear_state(self.stateful)
        return self._outputs(self._predict(y, u, d))


class Step(_PhysicalGraph):
    """(y, u, d, state) -> (y_next, state_next): every step after the first."""

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor,
                state: torch.Tensor | None = None):
        if self.stateful:
            write_state(self.stateful, self.widths, state)
        return self._outputs(self._predict(y, u, d))
