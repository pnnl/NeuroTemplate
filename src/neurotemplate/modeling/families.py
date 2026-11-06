"""
The black-box families' cores: each a plain module over normalised windows.

    core(y, u, d) -> z [batch, ny]      y, u, d: [batch, rows, n]

`mlp` is one network over the flattened windows, `nssm` a separable sum of three,
`lssm` the same sum, linear. Each reads every column at its own lag. `gru` is in
`gru`; the thermal families in `rc_network`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
# Aliased: `blocks` is also this package's per-block module.
from neurotemplate.nm import blocks as nm_blocks

from neurotemplate.modeling import windows


class LagWindow(nn.Module):
    """A `[batch, rows, n]` window flattened to the cells its columns' lags keep:
    row by row, oldest first, each column its newest `lag + 1` rows."""

    def __init__(self, lags: tuple[int, ...], rows: int):
        super().__init__()
        newest = rows - 1
        keep = tuple((j, newest - lag, newest) for j, lag in enumerate(lags))
        self.cells = sum(lag + 1 for lag in lags)
        indices = windows.window_gather((len(lags),), (rows,), (keep,))
        self.gathers = indices != list(range(len(lags) * rows))
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return flat.index_select(1, self.indices) if self.gathers else flat


def mlp_net(insize: int, outsize: int, width: int, depth: int) -> nm_blocks.MLP:
    """The ReLU MLP every family builds."""
    return nm_blocks.MLP(insize=insize, outsize=outsize, bias=True, linear_map=nn.Linear,
                         nonlin=nn.ReLU, hsizes=depth * [width])


def linear_net(insize: int, outsize: int) -> nm_blocks.Linear:
    return nm_blocks.Linear(insize, outsize, bias=True, linear_map=nn.Linear)


def _widths(net) -> list[int]:
    return [int(layer.out_features) for layer in net.linear[:-1]]


class Mlp(nn.Module):
    """One network over the flattened windows."""

    def __init__(self, lags, rows: int, outsize: int, width: int, depth: int):
        super().__init__()
        self.windows = nn.ModuleList(LagWindow(key_lags, rows) for key_lags in lags)
        self.in_features = sum(w.cells for w in self.windows)
        self.net = mlp_net(self.in_features, outsize, width, depth)

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        parts = [w(x) for w, x in zip(self.windows, (y, u, d)) if w.cells]
        return self.net(torch.cat(parts, dim=-1))

    def describe(self) -> dict:
        return {"hsizes": _widths(self.net), "in_features": self.in_features}


class Separable(nn.Module):
    """`fy(y) + fu(u) + fd(d)`: `nssm` with networks, `lssm` with linear maps. A
    key with no column adds nothing."""

    def __init__(self, lags, rows: int, outsize: int, width: Optional[int] = None,
                 depth: Optional[int] = None):
        super().__init__()
        self.windows = nn.ModuleList(LagWindow(key_lags, rows) for key_lags in lags)
        self.maps = nn.ModuleList(
            (linear_net(w.cells, outsize) if width is None
             else mlp_net(w.cells, outsize, width, depth)) if w.cells else nn.Identity()
            for w in self.windows)

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        parts = [f(w(x)) for f, w, x in zip(self.maps, self.windows, (y, u, d)) if w.cells]
        return sum(parts[1:], parts[0])

    def describe(self) -> dict:
        out = {"separable": True, "in_features": sum(w.cells for w in self.windows)}
        if isinstance(self.maps[0], nm_blocks.MLP):
            out["hsizes"] = _widths(self.maps[0])
        return out
