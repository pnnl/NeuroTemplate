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
         like a plain split: one family for everyone, just partitioned
         by block.
  inputs what each of its U and D columns *is*, per column — see
         "Declaring what a column is" below. Anything undeclared is an
         ordinary network input.
  ambient
         the one D column that is the outdoor temperature, if any. Only
         `2r2c` acts on it (the explicit conduction term); other families
         see the column as an ordinary input either way.
  params the block's physical priors, read by its family's builder. Today
         only `2r2c` reads any: its three time constants, in hours —
         `{"tau_fast_h": [0.3, 1.0], "tau_ambient_h": [2.0, 8.0],
         "tau_slow_h": [20.0, 60.0]}`, a scalar pinning one (see `rc`).

Declaring what a column is
==========================

    "inputs": {
        "Cool SP [°C]":   {"kind": "demand", "sign": -1, "state": "Zone Temp [°C]"},
        "Fan Power [kW]": {"kind": "linear", "sign": +1},
    }

`demand` turns a setpoint into the thermostat's clamped error against one
of the block's *own* Y channels, `relu(sign · (setpoint − state))`, before
any family sees it — see `prestack.Demand`. `sign` is the direction the state
moves while the demand is active (-1 cooling, +1 heating). Pairing with
another block's channel is refused: that is what `reads` is for. A block
with a `demand` entry has exactly one Y channel. Inside a `2r2c` block the
feature feeds a bounded `saturating` branch by default; `"branch": "mlp"`
routes it into the learned residual instead, which gives up the stability
bound and is reported as such.

`linear` says the column enters the zone's heat balance as one sign-fixed
coefficient times the physical value (measured power, solar irradiance).
Only `2r2c` acts on it; other families see the column as usual.

Nothing is inferred: a dataset whose actuator is metered power rather than
a setpoint declares it `linear` (or leaves it to the MLP) and never writes
a `demand` entry.

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

What this module does NOT do
=============================

There is no per-block override of a family's hyperparameters (hidden_size,
num_layers, ...) — `model` is a family name, nothing more. Every block
built from the same family shares cfg's hyperparameters. CLAUDE.md's YAGNI
rule is the reason: nothing today needs a north zone with a bigger MLP
than the south zone, and adding the knob before something does would be
exactly the speculative extension point the rules warn against.

Time constants are the exception, and `params` is where they go: they are
a property of the zone, not a hyperparameter. Two zones with different
envelopes have different priors, and the block is where a zone is declared.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
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
    #: The U and D column names behind `u_idx` / `d_idx`, in the same order,
    #: so per-column declarations can be resolved and labelled without
    #: nm_mapping.
    u_names: tuple[str, ...] = ()
    d_names: tuple[str, ...] = ()
    #: `{column: {"kind": ..., "sign": ..., "state": ...}}` for the declared
    #: columns, validated by `build_blocks`. Empty means every column is an
    #: ordinary input.
    inputs: dict[str, dict] = field(default_factory=dict)
    ambient: Optional[str] = None
    #: The block's physical priors, `{name: value}`, JSON-safe; validated by
    #: the family's builder, which is the one that knows the names.
    params: dict = field(default_factory=dict)

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


#: Every key a block spec may set. Anything else is a typo: the declaration
#: keys change what the block fits without changing its shape, so a
#: misspelled one has no other way of announcing itself.
BLOCK_KEYS = frozenset({"Y", "U", "D", "reads", "model", "inputs", "ambient",
                        "params"})

INPUT_KINDS = ("demand", "linear")
DEMAND_BRANCHES = ("saturating", "mlp")


def _resolve_inputs(spec: dict, block: str, y_names: list[str],
                    u_names: list[str], d_names: list[str],
                    ) -> tuple[dict[str, dict], Optional[str]]:
    """Validated `(inputs, ambient)` for one block, every problem named at once."""
    inputs = spec.get("inputs", {})
    ambient = spec.get("ambient")
    problems = []

    if ambient is not None and ambient not in d_names:
        problems.append(f"`ambient` ({ambient!r}) is not one of this block's D "
                        f"columns {d_names}.")

    for column, entry in inputs.items():
        kind = entry.get("kind")
        if column not in u_names and column not in d_names:
            problems.append(f"`inputs` names {column!r}, which is not one of this "
                            f"block's U {u_names} or D {d_names} columns.")
        if kind not in INPUT_KINDS:
            problems.append(f"`inputs[{column!r}]` has kind {kind!r}; expected one "
                            f"of {list(INPUT_KINDS)}.")
            continue
        if entry.get("sign") not in (-1, 1):
            problems.append(f"`inputs[{column!r}]` needs `sign` of -1 or +1 — the "
                            "direction the column pushes the state; got "
                            f"{entry.get('sign')!r}.")
        if kind == "demand":
            state = entry.get("state")
            if column not in u_names:
                problems.append(f"`inputs[{column!r}]` is `demand`, which only a U "
                                "column (a setpoint) can be.")
            if state not in y_names:
                problems.append(f"`inputs[{column!r}]` pairs with state {state!r}, "
                                f"which is not one of this block's own Y channels "
                                f"{y_names}. A demand may only read the block's own "
                                "state; to see another block's, use `reads`.")
            if len(y_names) != 1:
                problems.append(f"`inputs[{column!r}]` is `demand`, so this block "
                                f"must have exactly one Y channel; it has {y_names}.")
            if entry.get("branch", "saturating") not in DEMAND_BRANCHES:
                problems.append(f"`inputs[{column!r}]` has branch "
                                f"{entry.get('branch')!r}; expected one of "
                                f"{list(DEMAND_BRANCHES)}.")

    if problems:
        raise ValueError(
            f"block_mapping: block {block!r} has an invalid declaration:\n  - "
            + "\n  - ".join(problems)
        )
    return dict(inputs), ambient


@beartype
def build_blocks(
    nm_mapping: dict[str, list[str]],
    block_spec: Optional[dict[str, dict]] = None,
) -> list[Block]:
    """Turn a `block_mapping` hook result into validated index slices.

    `block_spec` is `{block_name: {"Y": [...], "U": [...], "D": [...],
    "reads": [...], "model": "...", "inputs": {...}, "ambient": "...",
    "params": {...}}}`.
    Passing None (no hook) gives one block holding every Y and U channel
    and reading every D column.

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
      - an `inputs` or `ambient` declaration is malformed (see the module
        docstring); every problem in a block is listed in one message.
      - a shared U column is declared `demand` by more than one block.
      - a block sets a key that is not one of `BLOCK_KEYS`.
    """
    y_cols, u_cols, d_cols = nm_mapping["Y"], nm_mapping["U"], nm_mapping["D"]

    if block_spec is None:
        return [Block("all", tuple(range(len(y_cols))), tuple(range(len(u_cols))),
                      tuple(y_cols), tuple(range(len(d_cols))),
                      u_names=tuple(u_cols), d_names=tuple(d_cols))]

    if not block_spec:
        raise ValueError("block_mapping returned an empty mapping; return None instead.")

    claimed_y: list[int] = []
    claimed_u: list[int] = []
    per_block: list[tuple] = []
    for name, spec in block_spec.items():
        unknown_keys = sorted(set(spec) - BLOCK_KEYS)
        if unknown_keys:
            raise ValueError(
                f"block_mapping: block {name!r} has unrecognized key(s) "
                f"{unknown_keys}; expected some of {sorted(BLOCK_KEYS)}. A "
                "misspelled `inputs` or `ambient` would otherwise sit inert "
                "and silently change the physics the block is fitting."
            )
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
        per_block.append((name, y_idx, u_idx, d_idx, reads_idx, model, spec))

    counts = Counter(claimed_y)
    duplicated = sorted({y_cols[i] for i, n in counts.items() if n > 1})
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
    blocks = []
    for name, y_idx, u_idx, d_idx, reads_idx, model, spec in per_block:
        u_idx = sorted(u_idx + shared_u)
        y_names = [y_cols[i] for i in y_idx]
        u_names = [u_cols[i] for i in u_idx]
        d_names = [d_cols[i] for i in d_idx]
        inputs, ambient = _resolve_inputs(spec, name, y_names, u_names, d_names)
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"block_mapping: block {name!r} has `params` "
                             f"{params!r}; expected a mapping.")
        blocks.append(Block(
            name, tuple(y_idx), tuple(u_idx), tuple(y_names), tuple(d_idx),
            tuple(reads_idx), model, tuple(u_names), tuple(d_names), inputs, ambient,
            dict(params),
        ))
    _check_one_demand_per_column(blocks)
    return blocks


def _check_one_demand_per_column(blocks: list[Block]) -> None:
    """The pre-stack rewrites a demand column once, for every block that sees
    it, so a shared column declared `demand` by two blocks can only mean one
    of them — and which one would be decided by dict order, silently."""
    declared_by: dict[str, list[str]] = {}
    for block in blocks:
        for column, entry in block.inputs.items():
            if entry["kind"] == "demand":
                declared_by.setdefault(column, []).append(block.name)
    clashes = {column: names for column, names in declared_by.items() if len(names) > 1}
    if clashes:
        raise ValueError(
            f"block_mapping: U column(s) declared `demand` by more than one "
            f"block: {clashes}. A demand column is rewritten once against one "
            "state for every block that sees it; declare it in one block only."
        )

