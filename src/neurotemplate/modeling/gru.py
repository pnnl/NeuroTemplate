"""
GRU — the recurrent black-box family, as a block every model can hold.

A GRU block reads a flattened key-major history window (see `windows`), every
column `steps` rows deep, and carries a hidden state so its memory can run past
that window during a rollout. It implements the stateful-block protocol
(`rollout_state` / `set_rollout_state`) that the System's reset and the ONNX
export rely on, as `rc.RCBlock` does.
"""

from typing import Optional

import torch
import torch.nn as nn



class GRUBlock(nn.Module):
    """
    A GRU over normalised `[batch, rows, n]` windows, folding each row into its
    hidden state exactly once:

      - Cold (no hidden state): consume its `steps` newest rows, so the state
        summarises the full context before the first prediction.
      - Warm: the earlier rows are already in the hidden state, so consume only
        the newest.

    The invariant is that the hidden state summarises every row this block has
    already consumed, and the window supplies whatever it has not. That is what
    lets memory run past its window during a rollout while a single cold call is
    still correct on its own — instantiate this block anywhere, hand it one
    window, and you get a properly warmed prediction.

    Call set_rollout_state(None) at a sequence boundary (a new batch); the next
    call re-warms itself from its own window.
    """

    def __init__(self, width: int, steps: int, outsize: int, hidden_size: int,
                 num_layers: int, bias: bool = True):
        super().__init__()
        self.steps = steps
        self.in_features = steps * width
        self.gru = nn.GRU(width, hidden_size, num_layers=num_layers,
                          batch_first=True, bias=bias)
        self.output = nn.Linear(hidden_size, outsize, bias=bias)
        self._h: Optional[torch.Tensor] = None

    # The stateful-block protocol (see `contract.stateful_blocks`). nn.GRU
    # keeps its state layer-first, [num_layers, batch, hidden]; the protocol
    # is batch-first and flat, so the two methods own that reshape.

    def rollout_state(self) -> Optional[torch.Tensor]:
        """The hidden state, [batch, num_layers*hidden], or None if cold."""
        if self._h is None:
            return None
        return self._h.transpose(0, 1).reshape(self._h.shape[1], -1)

    def set_rollout_state(self, state: Optional[torch.Tensor]) -> None:
        """Restore a hidden state, or None to force a cold start."""
        if state is None:
            self._h = None
            return
        layers, hidden = self.gru.num_layers, self.gru.hidden_size
        self._h = state.reshape(-1, layers, hidden).transpose(0, 1).contiguous()

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        rows = self.steps if self._h is None else 1
        window = torch.cat([x[:, -rows:] for x in (y, u, d) if x.shape[-1]], dim=-1)
        out, self._h = self.gru(window, self._h)   # [batch, rows, hidden]
        return self.output(out[:, -1, :])          # [batch, outsize]

    def describe(self) -> dict:
        return {"hidden_size": self.gru.hidden_size, "num_layers": self.gru.num_layers,
                "in_features": self.in_features}
