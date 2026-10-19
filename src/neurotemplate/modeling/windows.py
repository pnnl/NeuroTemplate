"""
Positions inside the flattened windows a dynamics block or a policy receives.

`SystemPreview`'s input_map hands a node one flat tensor per call, laid out
**key-major**: the whole Y window, then the whole U window, then the whole D
window, each of them time-ordered oldest-first.

    [ y_0 .. y_{a-1} | u_0 .. u_{b-1} | d_0 .. d_{c-1} ]      steps = (a, b, c)

Every block that wants a *part* of that — the newest timestep, one key's
block, one zone's columns, each column at its own lag — is asking the same
question about the same layout, so the arithmetic lives here once rather
than in each block.

These are pure position calculations: they depend only on (widths, steps),
never on the data, so blocks compute them once at construction.
"""

from __future__ import annotations


def key_blocks(widths: tuple[int, ...], steps: tuple[int, ...]) -> list[slice]:
    """One slice per key, covering that key's whole window of `steps[k]` rows."""
    slices, offset = [], 0
    for width, rows in zip(widths, steps):
        slices.append(slice(offset, offset + rows * width))
        offset += rows * width
    return slices


def last_step(widths: tuple[int, ...], steps: tuple[int, ...]) -> list[slice]:
    """One slice per key, covering just the newest timestep of that key.

    The newest step is a contiguous chunk at the end of each key's block,
    so this needs no gather — just a slice.
    """
    return [
        slice(block.stop - width, block.stop)
        for block, width in zip(key_blocks(widths, steps), widths)
    ]


def time_major(widths: tuple[int, ...], steps: int) -> list[int]:
    """Permutation that reorders a key-major window, every key `steps` rows
    deep, to time-major.

    Result is the index list for `[y_0,u_0,d_0, y_1,u_1,d_1, ...]` — the
    layout a sequence model (an RNN) reads.
    """
    blocks = key_blocks(widths, (steps,) * len(widths))
    return [
        block.start + t * width + j
        for t in range(steps)
        for block, width in zip(blocks, widths)
        for j in range(width)
    ]


def per_step(block: slice, width: int, steps: int) -> list[slice]:
    """Split one key's block into its `steps` per-timestep slices."""
    return [
        slice(block.start + t * width, block.start + (t + 1) * width)
        for t in range(steps)
    ]


def window_gather(
    widths: tuple[int, ...],
    steps: tuple[int, ...],
    keep: tuple[tuple[tuple[int, int, int], ...], ...],
) -> list[int]:
    """Indices narrowing a rectangular key-major window, column by column.

    Key k's window is `steps[k]` rows of its `widths[k]` columns. `keep[k]`
    lists the columns of key k to keep, in the order wanted, each as
    `(column, first, last)`: the rows it keeps, inclusive, counted from the
    key's oldest row. A column with `first > last` keeps no row. The result
    is key-major again and, inside each key, row by row oldest first, with
    the kept columns in `keep` order on every row they cover — so a column
    that keeps the newest row has it last.

    Keeping every column on every row, in order, gives the identity.
    """
    indices = []
    for block, width, rows, columns in zip(key_blocks(widths, steps), widths,
                                           steps, keep):
        for t, row in enumerate(per_step(block, width, rows)):
            indices.extend(row.start + j for j, first, last in columns
                           if first <= t <= last)
    return indices
