"""
The model contract: what a dynamics model is, to everything around it.

    model(y, u, d) -> y_next

    y  [batch, history_steps, ny]   the last history_steps outputs, oldest first
    u  [batch, history_steps, nu]   the matching inputs; row -1 drives this step
    d  [batch, history_steps, nd]   the matching disturbances
    ->  y_next [batch, ny]

Everything is in physical units, the data's own. Training, the physics
checks, control and export all call a model this way and nothing else, so
anything that honours it gets all of them. A model that keeps state past its
window implements `rollout_state()` / `set_rollout_state(state)` (see
`dynamics.stateful_blocks`): cold on the first call of a rollout, it seeds the
state from its window; warm, it carries it.

The built-in families compute in normalised units. `Normalized` puts one
behind the contract: it normalises the windows with the run's `norm_stats`,
calls the family, and returns physical units. The family's heads and derived
inputs live inside it, in its units (`heads.HeadStack.normalized`).

A model of your own is registered by name in the hooks file's
`model_families()`, `{name: build(cfg, nm_mapping, norm_stats) -> model}`, and
`cfg.model` names it. It is one module over every column, honouring the
contract as written; it may use `Normalized` and the heads itself.

The rollout (`SystemPreview`) hands a node one flat window per key;
`Windowed` reshapes those into the contract's.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_KEYS = ("Y", "U", "D")


def _stats(norm_stats: dict[str, dict], columns: list[str], field: str) -> torch.Tensor:
    return torch.tensor([float(norm_stats[c][field]) for c in columns],
                        dtype=torch.float32)


class Normalized(nn.Module):
    """A model that computes in normalised units, behind the physical contract.

    `core` reads key-major flattened normalised windows (`modeling.windows`)
    and returns the normalised next y; this does the conversions on both
    sides, so `core` never sees a physical value.
    """

    def __init__(self, core: nn.Module, nm_mapping: dict[str, list[str]],
                 norm_stats: dict[str, dict]):
        super().__init__()
        self.core = core
        self.history_steps = core.history_steps
        for key in _KEYS:
            columns = nm_mapping[key]
            self.register_buffer(f"{key.lower()}_shift", _stats(norm_stats, columns, "shift"),
                                 persistent=False)
            self.register_buffer(f"{key.lower()}_stretch", _stats(norm_stats, columns, "stretch"),
                                 persistent=False)

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        """`x` [..., n] of key `key` ("Y", "U" or "D") in the core's units."""
        key = key.lower()
        return (x - getattr(self, f"{key}_shift")) / getattr(self, f"{key}_stretch")

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # A row-major reshape of [batch, steps, n] is the key-major window:
        # time ordered, oldest first, every column of a step together.
        flat = [self.normalize(key, x).reshape(x.shape[0], -1)
                for key, x in zip(_KEYS, (y, u, d))]
        return self.core(*flat) * self.y_stretch + self.y_shift


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


def normalized_core(dyn_sys) -> nn.Module | None:
    """The normalised family inside a built-in model — where its heads, its
    derived inputs and its learned maps live — or None for a model of the
    user's own, which has none of them."""
    model = contract_model(dyn_sys)
    return model.core if isinstance(model, Normalized) else None
