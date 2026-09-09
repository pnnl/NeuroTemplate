"""
Splitting a multi-output model into independent per-block sub-models.

A *block* is a group of Y channels that gets its own sub-model: its own
parameters, its own architecture, and (for stateful families like `2r2c`
and `gru`) its own hidden state. Blocks are independent by construction —
no term in one block's update reads another block's state on the same
step. Inter-block conduction is deliberately out of scope; see `reads`
below for the one kind of cross-block coupling that *is* in scope.

What a block owns
==================

  Y      its own channels, and only its own. Every Y channel belongs to
         exactly one block.
  U      its own actuation, plus any U channel no block claimed. Unclaimed
         U channels are treated as *shared* and handed to every block,
         which is what you want for a single air handler serving several
         blocks.
  D      all of it by default. A block may instead name a subset — D has
         no partition requirement, since it is exogenous and shared by
         nature: two blocks may each claim the same column, or all of it,
         with no conflict.
  reads  other blocks' Y channels this block may also see as an ordinary
         input — see the `Block.y_in_idx` docstring for why this is safe.
  model  the family this block builds under (e.g. "mlp", "2r2c"). None
         defers to `cfg.model`, so a spec that never sets `model` behaves
         exactly like the old zone split: one family for everyone, just
         partitioned by block.

Where the split comes from
===========================

An optional `block_mapping` hook in the application's modeling hooks file:

    def block_mapping(nm_mapping, cfg):
        return {
            "north": {"Y": ["North Zone Air Temperature [°C]"],
                      "U": ["North Cooling Setpoint [°C]"],
                      "model": "2r2c"},
            "power": {"Y": ["Total Power [W]"],
                      "reads": ["North Zone Air Temperature [°C]"],
                      "D": ["Ambient Temperature [degC]"],
                      "model": "mlp"},
        }

With no hook, `build_blocks` returns a single block holding every channel
and reading every D column — which for `2r2c` still means one independent
RC network per Y channel, since that family is per-channel internally. The
hook is what restricts *which setpoint drives which block* (and, now,
which block reads which other block's state); you only need it when that
restriction matters.

`zone_mapping` is kept as a deprecated alias for this hook: a hooks module
that still defines `zone_mapping` and not `block_mapping` is honored
exactly as before (see pipeline.resolve_block_spec).

What this module does NOT do
=============================

There is no per-block override of a family's hyperparameters (hidden_size,
num_layers, ...) — `model` is a family name, nothing more. Every block
built from the same family shares cfg's hyperparameters. CLAUDE.md's YAGNI
rule is the reason: nothing today needs a north zone with a bigger MLP
than the south zone, and adding the knob before something does would be
exactly the speculative extension point the rules warn against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from neurotemplate._typing import beartype


@dataclass(frozen=True)
class Block:
    """One independent sub-model's slice of the Y, U and D channel lists.

    `y_idx`, `u_idx` and `d_idx` index into `nm_mapping["Y"]`,
    `nm_mapping["U"]` and `nm_mapping["D"]`. `reads` also indexes into
    `nm_mapping["Y"]`, naming OTHER blocks' channels this block may see.

    `model` is a family name (e.g. "mlp", "2r2c") or None, meaning "use
    cfg.model" — see the module docstring for why there is no per-block
    hyperparameter override.
    """

    name: str
    y_idx: tuple[int, ...]
    u_idx: tuple[int, ...]
    #: The Y column names behind `y_idx`, carried so a block can label its
    #: per-channel numbers on the reportcard without being handed nm_mapping.
    y_names: tuple[str, ...]
    d_idx: tuple[int, ...] = ()
    reads: tuple[int, ...] = ()
    model: Optional[str] = None

    @property
    def ny(self) -> int:
        """Output width: how many Y channels this block predicts."""
        return len(self.y_idx)

    @property
    def nu(self) -> int:
        return len(self.u_idx)

    @property
    def nd(self) -> int:
        return len(self.d_idx)

    @property
    def y_in_idx(self) -> tuple[int, ...]:
        """Y positions this block reads, as opposed to `y_idx` (predicts).

        Own channels first, then anything read from other blocks (sorted),
        so a family that only ever looks at "its own state" — 2r2c's fixed
        linear term — keeps reading the same leading slice it always has.

        `reads` is an ordinary input for every family except `2r2c`,
        because all block inputs come from the shared history window,
        never from another block's same-step output: the gather that
        builds this block's window and the scatter that reassembles
        nm_mapping["Y"] order are both fixed permutations computed once at
        construction (see `dynamics.BlockStack`), with no evaluation-order
        dependency between blocks — which is what keeps the whole stack
        vmap-safe, and so keeps the sampled gradient penalty alive. The one
        family that cannot accept a `reads` entry is `2r2c`; see the 2r2c
        per-block builder in `dynamics.py` (and `rc.py`'s module docstring)
        for why.
        """
        return tuple(self.y_idx) + tuple(sorted(self.reads))

    @property
    def ny_in(self) -> int:
        """Input width: own channels plus anything read from elsewhere."""
        return len(self.y_in_idx)


def _resolve(columns: list[str], names: list[str], block: str, key: str,
             source: Optional[str] = None) -> list[int]:
    """Map column names to their positions in `columns`, or raise.

    `key` names the spec field being validated (e.g. "Y", "reads"); `source`
    names the nm_mapping list the names must come from, when it differs
    from `key` — `reads` names live in nm_mapping["Y"], not a "reads" role.
    """
    source = source or key
    missing = [n for n in names if n not in columns]
    if missing:
        raise ValueError(
            f"block_mapping: block {block!r} lists {key} column(s) {missing} that "
            f"are not in nm_mapping[{source!r}] ({columns})."
        )
    return [columns.index(n) for n in names]


@beartype
def build_blocks(
    nm_mapping: dict[str, list[str]],
    block_spec: Optional[dict[str, dict]] = None,
) -> list[Block]:
    """Turn a `block_mapping` hook result into validated index slices.

    `block_spec` is `{block_name: {"Y": [...], "U": [...], "D": [...],
    "reads": [...], "model": "..."}}`. Passing None (no hook) gives one
    block holding every Y and U channel and reading every D column.

    Raises ValueError if:
      - Y is not partitioned exactly: every Y channel in one and only one
        block.
      - a U column is claimed by more than one block. (D has no such
        rule — it is shared by nature, so several blocks may each claim
        the same column, or all of it, with no conflict.)
      - a Y, U, D or `reads` name is not in the corresponding nm_mapping
        list.
      - a `reads` entry names a channel the block already owns as Y
        output — reading your own Y is automatic and needs no declaration.
    """
    y_cols, u_cols, d_cols = nm_mapping["Y"], nm_mapping["U"], nm_mapping["D"]

    if block_spec is None:
        return [Block("all", tuple(range(len(y_cols))), tuple(range(len(u_cols))),
                      tuple(y_cols), tuple(range(len(d_cols))))]

    if not block_spec:
        raise ValueError("block_mapping returned an empty mapping; return None instead.")

    claimed_y: list[int] = []
    claimed_u: list[int] = []
    per_block: list[tuple] = []
    for name, spec in block_spec.items():
        y_idx = _resolve(y_cols, spec.get("Y", []), name, "Y")
        u_idx = _resolve(u_cols, spec.get("U", []), name, "U")
        # Absent "D" means all of D; present means exactly the named subset
        # — no partition requirement, D is shared by nature.
        d_idx = (list(range(len(d_cols))) if "D" not in spec
                 else _resolve(d_cols, spec["D"], name, "D"))
        reads_idx = _resolve(y_cols, spec.get("reads", []), name, "reads", source="Y")
        model = spec.get("model")

        if not y_idx:
            raise ValueError(f"block_mapping: block {name!r} claims no Y channels.")
        owned = sorted(set(y_idx) & set(reads_idx))
        if owned:
            raise ValueError(
                f"block_mapping: block {name!r} lists {[y_cols[i] for i in owned]} "
                "in `reads`, but it already owns them as Y output. Reading your "
                "own Y is automatic; only list channels another block produces."
            )

        claimed_y += y_idx
        claimed_u += u_idx
        per_block.append((name, y_idx, u_idx, d_idx, reads_idx, model))

    duplicated = sorted({y_cols[i] for i in claimed_y if claimed_y.count(i) > 1})
    if duplicated:
        raise ValueError(
            f"block_mapping: Y column(s) {duplicated} are claimed by more than "
            "one block. Each Y channel belongs to exactly one block."
        )
    unclaimed_y = [c for i, c in enumerate(y_cols) if i not in claimed_y]
    if unclaimed_y:
        raise ValueError(
            f"block_mapping: Y column(s) {unclaimed_y} are not claimed by any "
            "block. Every Y channel belongs to exactly one block."
        )
    duplicated_u = sorted({u_cols[i] for i in claimed_u if claimed_u.count(i) > 1})
    if duplicated_u:
        raise ValueError(
            f"block_mapping: U column(s) {duplicated_u} are claimed by more than "
            "one block. Leave a U channel out entirely to share it with all blocks."
        )

    # Whatever no block asked for is shared equipment: hand it to everyone.
    shared_u = [i for i in range(len(u_cols)) if i not in claimed_u]
    return [
        Block(name, tuple(y_idx), tuple(sorted(u_idx + shared_u)),
              tuple(y_cols[i] for i in y_idx), tuple(d_idx), tuple(reads_idx), model)
        for name, y_idx, u_idx, d_idx, reads_idx, model in per_block
    ]


# ── Deprecated aliases ────────────────────────────────────────────
#
# "Zone" was the original name; "block" is the correct one — a block is a
# sub-model over a group of signals, and "zone" implied HVAC where none is
# meant. Kept so code and saved specs written against the old name still
# work.
Zone = Block
build_zones = build_blocks
