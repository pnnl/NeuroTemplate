"""
Positions inside the flattened history window a dynamics block receives.

`SystemPreview`'s input_map hands every block one flat tensor per call,
laid out **key-major**: the whole Y window, then the whole U window, then
the whole D window, each of them time-ordered oldest-first.

    [ y_0 .. y_{s-1} | u_0 .. u_{s-1} | d_0 .. d_{s-1} ]      s = in_steps

Every block that wants a *part* of that — the newest timestep, one key's
block, one zone's columns — is asking the same question about the same
layout, so the arithmetic lives here once rather than in each block.

These are pure position calculations: they depend only on (widths, in_steps),
never on the data, so blocks compute them once at construction.
"""

from __future__ import annotations


def key_blocks(widths: tuple[int, ...], in_steps: int) -> list[slice]:
    """One slice per key, covering that key's whole `in_steps` window."""
    slices, offset = [], 0
    for width in widths:
        slices.append(slice(offset, offset + in_steps * width))
        offset += in_steps * width
    return slices


def last_step(widths: tuple[int, ...], in_steps: int) -> list[slice]:
    """One slice per key, covering just the newest timestep of that key.

    The newest step is a contiguous chunk at the end of each key's block,
    so this needs no gather — just a slice.
    """
    return [
        slice(block.start + (in_steps - 1) * width, block.stop)
        for block, width in zip(key_blocks(widths, in_steps), widths)
    ]


def time_major(widths: tuple[int, ...], in_steps: int) -> list[int]:
    """Permutation that reorders a key-major window to time-major.

    Result is the index list for `[y_0,u_0,d_0, y_1,u_1,d_1, ...]` — the
    layout a sequence model (an RNN) reads.
    """
    blocks = key_blocks(widths, in_steps)
    return [
        block.start + t * width + j
        for t in range(in_steps)
        for block, width in zip(blocks, widths)
        for j in range(width)
    ]


def per_step(block: slice, width: int, in_steps: int) -> list[slice]:
    """Split one key's block into its `in_steps` per-timestep slices."""
    return [
        slice(block.start + t * width, block.start + (t + 1) * width)
        for t in range(in_steps)
    ]


def gather_columns(
    widths: tuple[int, ...], in_steps: int, keep: tuple[tuple[int, ...], ...]
) -> list[int]:
    """Indices selecting a subset of each key's columns, across all timesteps.

    `keep[k]` lists which of key k's `widths[k]` columns to keep. The result
    is itself a valid key-major window over the narrowed widths, so a block
    reading it can use the helpers above on `tuple(len(c) for c in keep)`.
    """
    return [
        block.start + t * width + j
        for block, width, cols in zip(key_blocks(widths, in_steps), widths, keep)
        for t in range(in_steps)
        for j in cols
    ]
