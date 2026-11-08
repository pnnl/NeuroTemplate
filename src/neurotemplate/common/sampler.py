"""
Training batches drawn at random from the records; dev batches fixed.

`WindowSampler` keeps each key's columns, concatenated over the records once, and
the rows a window may start at, so no window crosses from one record into the
next. Each draw picks `batch_size` of those starts and gathers the rows by index:
any row of the records can start a window, and nothing is built per window in
advance. It is an iterator: `next(sampler)` is a batch.

`fixed_batches` cuts a window dict into the batches the dev set is scored on,
the same ones at every evaluation.
"""

from __future__ import annotations

import pandas as pd
import torch


class WindowSampler:
    """Random windows of the records, `batch_size` at a draw.

    `keys` maps each batch key to its columns and how many rows of the window it
    holds, counted from the window's start: `{"yn": (y_columns, 4), "D": (d_columns,
    12)}`. `steps`, the window's length, is at least the longest of them.
    """

    def __init__(self, frames: list[pd.DataFrame], keys: dict[str, tuple[list[str], int]],
                 steps: int, batch_size: int, seed: int, device, name: str = "train"):
        short = [len(frame) for frame in frames if len(frame) < steps]
        if not frames or short:
            raise ValueError(
                f"Every record must hold a whole window of {steps} rows; these hold "
                f"{short or 'nothing'}. Use longer files or shorten the window (`horizon`, "
                "`past`, `context`, and for a policy the deepest `future` it previews).")
        self.rows = {key: rows for key, (_, rows) in keys.items()}
        self.data = {key: torch.cat([torch.tensor(frame[columns].to_numpy(), dtype=torch.float32)
                                     for frame in frames]).to(device)
                     for key, (columns, _) in keys.items()}
        starts, offset = [], 0
        for frame in frames:
            starts.append(torch.arange(offset, offset + len(frame) - steps + 1))
            offset += len(frame)
        self.starts = torch.cat(starts).to(device)
        self.batch_size, self.name, self.device = batch_size, name, device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def __iter__(self) -> WindowSampler:
        return self

    def __next__(self) -> dict:
        pick = torch.randint(len(self.starts), (self.batch_size,), generator=self.generator,
                             device=self.device)
        starts = self.starts[pick]
        batch = {}
        for key, x in self.data.items():
            rows = starts[:, None] + torch.arange(self.rows[key], device=self.device)[None, :]
            batch[key] = x[rows]
        batch["name"] = self.name
        return batch


def fixed_batches(data: dict, batch_size: int, name: str) -> list[dict]:
    """`data`'s windows in order, `batch_size` to a batch, each named `name`."""
    n = next(iter(data.values())).shape[0]
    return [{**{key: x[i:i + batch_size] for key, x in data.items()}, "name": name}
            for i in range(0, n, batch_size)]
