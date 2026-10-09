"""
GRU — the recurrent black-box family, as a block every model can hold.

A GRU block reads the same flattened `in_steps` history window as every other
block (see `windows`), and carries a hidden state so its memory can run past
that window during a rollout. It implements the stateful-block protocol
(`rollout_state` / `set_rollout_state`) that the System's reset and the ONNX
export rely on, as `rc.RCBlock` does.
"""

from typing import Optional

import torch
import torch.nn as nn

from neurotemplate.modeling import windows


class GRUBlock(nn.Module):
    """
    GRU-based dynamics block. Reads the same flattened in_steps history window
    every other block reads, but folds each timestep into its hidden state
    exactly once:

      - Cold (no hidden state): consume the whole window, so the state
        summarises the full context before the first prediction.
      - Warm: the window's earlier steps are already in the hidden state, so
        consume only its newest step.

    The invariant is that the hidden state summarises every timestep this block
    has already consumed, and the window supplies whatever it has not. That is
    what lets memory run past in_steps during a rollout while a single cold
    call is still correct on its own — instantiate this block anywhere, hand it
    one window, and you get a properly warmed prediction.

    Call reset() at a sequence boundary (a new batch); the next call re-warms
    itself from its own window.

    Handles two input shapes:
      - [batch, in_steps*features]  — normal training/inference path
      - [in_steps*features]         — under vmap (jacrev); always cold, no
                                      state stored
    """

    def __init__(self, ny: int, nu: int, nd: int, in_steps: int, outsize: int,
                 hidden_size: int, num_layers: int, bias: bool = True):
        super().__init__()
        self.widths = (ny, nu, nd)
        self.in_steps = in_steps
        self.in_features = in_steps * (ny + nu + nd)
        self.out_features = outsize
        self.gru = nn.GRU(ny + nu + nd, hidden_size, num_layers=num_layers,
                          batch_first=True, bias=bias)
        self.output = nn.Linear(hidden_size, outsize, bias=bias)
        self._h: Optional[torch.Tensor] = None

        # SSMGrad hands us a key-major window (see `windows`). A GRU reads
        # time-major, and the reorder is a fixed permutation of positions —
        # it depends only on (widths, in_steps), never on the data — so
        # compute it once here instead of on every call.
        self.register_buffer(
            "_seq_perm",
            torch.tensor(windows.time_major(self.widths, in_steps), dtype=torch.long),
            persistent=False,
        )
        self._last_step_slices = windows.last_step(self.widths, in_steps)

    # The stateful-block protocol (see `neurotemplate.export.state`). nn.GRU
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

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """[..., in_steps*(ny+nu+nd)] -> [..., in_steps, ny+nu+nd], key-major
        reordered to time-major via the precomputed permutation."""
        lead = x.shape[:-1]
        seq = x.index_select(x.ndim - 1, self._seq_perm)
        return seq.reshape(*lead, self.in_steps, sum(self.widths))

    def _last_step(self, x: torch.Tensor) -> torch.Tensor:
        """[..., in_steps*(ny+nu+nd)] -> [..., ny+nu+nd]: just the newest
        timestep, without touching the rest of the window."""
        return torch.cat([x[..., s] for s in self._last_step_slices], dim=-1)

    def _manual_gru(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Replay a whole window from a zero hidden state using vmap-safe
        primitives (matmul/sigmoid/tanh). nn.GRU's fused CuDNN kernel cannot be
        batched by vmap, so jacrev under vmap needs this path.

        Variable names follow the PyTorch nn.GRU equation:
            r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)   # reset gate
            z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)   # update gate
            n = tanh   (W_in x + b_in + r*(W_hn h + b_hn))  # candidate
            h = (1 - z) * n + z * h_prev
        See https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html

        Args:
            seq: [in_steps, ny+nu+nd] (1D input, under vmap)
        Returns:
            [outsize] output
        """
        # One hidden state per layer, all starting at zero — the sampled
        # gradient penalty measures d(prediction)/d(window) from cold, the
        # same object it measures for a stateless block.
        hidden = [torch.zeros(self.gru.hidden_size, device=seq.device)
                  for _ in range(self.gru.num_layers)]

        for step in range(self.in_steps):
            x = seq[step]
            for layer in range(self.gru.num_layers):
                w_ih = getattr(self.gru, f'weight_ih_l{layer}')  # [3*H, input]
                w_hh = getattr(self.gru, f'weight_hh_l{layer}')  # [3*H, H]
                b_ih = getattr(self.gru, f'bias_ih_l{layer}')     # [3*H]
                b_hh = getattr(self.gru, f'bias_hh_l{layer}')     # [3*H]

                gi = x @ w_ih.T + b_ih             # [3*H]
                gh = hidden[layer] @ w_hh.T + b_hh  # [3*H]

                i_r, i_z, i_n = gi.chunk(3)
                h_r, h_z, h_n = gh.chunk(3)

                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)
                hidden[layer] = (1 - z) * n + z * hidden[layer]
                x = hidden[layer]  # input to next layer

        return self.output(hidden[-1])  # last layer, last step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            # Under vmap: manual GRU math (fused nn.GRU kernel isn't vmap-safe).
            # Always cold, so it needs the reordered window regardless.
            return self._manual_gru(self._to_sequence(x))

        if self._h is None:
            # Cold: fold the whole window in.
            steps = self._to_sequence(x)
        else:
            # Warm: every step but the newest is already in the hidden state —
            # no need to touch the rest of the window.
            steps = self._last_step(x).unsqueeze(1)  # [batch, 1, ny+nu+nd]

        out, self._h = self.gru(steps, self._h)   # [batch, len(steps), hidden]
        return self.output(out[:, -1, :])          # [batch, outsize]
