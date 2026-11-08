"""
The autoregressive rollout: neuromancer's `SystemPreview`, with the windows
of keys no node writes taken straight from their data tensor.
"""

from __future__ import annotations

import torch
from neurotemplate.nm.system import RolloutBuffers, SystemPreview


class WindowedSystem(SystemPreview):
    """`SystemPreview`, serving a key no node writes as a view of its data.

    `SystemPreview` builds every input window by concatenating the window's
    rows from a list of per-step tensors, one `cat` per key per node per
    step. A key no node writes (U, D and t for the dynamics; D, t and R in
    the closed loop) is the batch's own tensor for the whole rollout, so its
    window at step `i` is `data[:, i - past : i + future + 1]`, which
    flattens to the node's `[batch, rows * dim]` as a view of a contiguous
    batch: no copy at all. At the edges, where the padding applies, it
    defers to `get_mapped_data`, the same indices on the same tensor. A key
    a node writes (yn, and U in the closed loop) is windowed exactly as
    `SystemPreview` does it.

    A node whose callable reads only its newest rows on this step (`rows_wanted`,
    a warm thermal model: see `contract.Windowed`) is handed only those, its
    windows cut to end where they did. Every window of such a node ends at the
    current step.

    The loop is `SystemPreview.forward`'s, unchanged apart from where a
    window comes from.
    """

    def forward(self, input_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data = input_dict.copy()      # never adds keys to the caller's dict
        nsteps = (self.nsteps if self.nsteps is not None
                  else data[self.nstep_key].shape[1] - self.start_iter)
        data = self.init(data)
        rollout = RolloutBuffers(data, self.rollout_keys)
        for i in range(self.start_iter, self.start_iter + nsteps):
            for node in self.nodes:
                input_map = _windows_wanted(node)
                indata = {key: self._window(data, rollout, input_map, key, i)
                          for key in node.input_keys}
                rollout.append(node(indata))
        return rollout.collapse()

    def _window(self, data, rollout, input_map: dict, key: str, i: int) -> torch.Tensor:
        if key not in input_map:
            return rollout[key][i]
        spec = input_map[key]
        if key in self.output_keys:
            return self.get_mapped_steps(rollout[key], i, spec)
        x = data[key]
        first, last = i - spec["past"], i + spec["future"]
        if first >= 0 and last < x.shape[1]:
            return x[:, first:last + 1].reshape(x.shape[0], -1)
        return self.get_mapped_data(x, i, spec)


def _windows_wanted(node) -> dict:
    """`node.input_map`, each window cut to the newest `rows_wanted()` rows when its
    callable reads only those on this step."""
    rows = node.callable.rows_wanted() if hasattr(node.callable, "rows_wanted") else None
    if rows is None:
        return node.input_map
    return {key: {**spec, "past": rows - 1} for key, spec in node.input_map.items()}
