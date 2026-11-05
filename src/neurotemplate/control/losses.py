"""
The declared control objective: a table of loss terms over named columns.

    losses:
      comfort_lower: {column: zone_lo,  target: zone_temp, map: relu,  norm: 2}
      comfort_upper: {column: zone_hi,  target: zone_temp, map: nrelu, norm: 2}
      du:            {column: setpoint, map: first_difference,         norm: 1}
      cost:          {column: power,                       norm: 1, weight: tariff}
    gains: {comfort_lower: 5.0, comfort_upper: 5.0, du: 1.0, cost: 1.0}

Each entry is read in one order: the operand `column − target` (`target`
optional), then the `map`, then the `norm` — the power mean
`(mean_t w·|h|^q)^(1/q)` per window with an optional per-step `weight` column
`w`, averaged over the batch — then `gains[name]`.

  - `map: relu` on `column − target` is a lower bound (the target below the
    column costs), `nrelu` an upper bound, so the direction is never encoded
    in a name. `first_difference` is `h[t] − h[t−1]`.
  - `norm` is the power mean, not the raw q-norm: it does not grow with the
    horizon, and it is homogeneous of degree 1 in the operand, so every term's
    gain scales the same way when units change.
  - Everything is in physical units, so a gain is per physical unit of the
    operand; `power` is Watts.
  - A `weight` is data — a reference or disturbance column, in its physical
    units. A tariff is not a kind of reference; it is the `weight` of a power
    term.

Every name resolves to one role, in this order: the reserved `power` (the
`power_metric` hook's output), a Y, U or D column of the surrogate, and
otherwise a column of the record no role claims, a reference. The reference columns
are the control-side `R` key: windowed beside Y/U/D, read by the policy if its window
lists them, never by the surrogate. The objective, the window and `control_bounds` may
each name one.

How the policy normalises an `R` column it reads is declared rather than
inferred. A column bound to a Y target (the `column` of an entry whose `target`
is in Y) uses that target's shift and stretch, so the policy compares like with
like. One band column may bound several targets — one comfort band over every
zone — and then takes the stats of the first of them in Y order. Every other
`R` column — a tariff, a cap on `power` — is divided by its entry in
`reference_scale`, with no shift, so a zero stays a zero.

The regulated set is derived: the Y targets of the `relu`/`nrelu` entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from neurotemplate.common.data_utils import NormStats, parse_column_unit

from neurotemplate._typing import beartype

#: The `power_metric` hook's output, by contract in Watts.
POWER = "power"
MAPS = ("relu", "nrelu", "first_difference")
ENTRY_KEYS = {"column", "target", "map", "norm", "weight"}
#: A term's output key is `<name>_loss`; PenaltyLoss already writes these two.
RESERVED_NAMES = {"objective", "penalty"}
#: The window-dict key each role is read from.
ROLE_KEYS = {"Y": "yn", "U": "U", "D": "D", "R": "R"}


@dataclass(frozen=True)
class LossTerm:
    """One row of the `losses` table, with its gain."""
    name: str
    column: str
    target: Optional[str]
    map: Optional[str]
    norm: float
    weight: Optional[str]
    gain: float

    @property
    def key(self) -> str:
        """The term's key in the problem's output dict and in MLflow."""
        return f"{self.name}_loss"

    @property
    def is_bound(self) -> bool:
        return self.map in ("relu", "nrelu")


@dataclass(frozen=True)
class Band:
    """The bounds on one regulated output: the columns of its `relu` (lower)
    and `nrelu` (upper) entries. A one-sided band leaves the other None."""
    target: str
    lower: Optional[str]
    upper: Optional[str]


@dataclass(frozen=True)
class Objective:
    """The `losses` table resolved against the surrogate's columns."""
    terms: list[LossTerm]
    roles: dict[str, str]      # every named column -> "Y" | "U" | "D" | "R" | POWER
    r_columns: list[str]       # sorted, so YAML order never moves a policy input
    bound_to: dict[str, str]   # reference column -> the Y column whose stats it takes
    bands: list[Band]          # in nm_mapping["Y"] order

    @property
    def regulated(self) -> list[str]:
        return [band.target for band in self.bands]

    @property
    def reads_power(self) -> bool:
        return POWER in self.roles

    def columns_of(self, role: str, nm_mapping: dict[str, list[str]]) -> list[str]:
        return self.r_columns if role == "R" else nm_mapping[role]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@beartype
def parse_losses(losses: dict, gains: dict) -> list[LossTerm]:
    """The `losses` table and `gains` map as terms, checked in their own terms.

    Whether the names exist is a question for `resolve_objective`, which has
    the surrogate's columns.
    """
    _require(bool(losses), "losses is empty: declare at least one control "
             "objective term (see control.yaml's OBJECTIVE section).")
    missing = sorted(set(losses) - set(gains))
    extra = sorted(set(gains) - set(losses))
    _require(not missing, f"losses {missing} have no entry in gains.")
    _require(not extra, f"gains {extra} name no entry in losses.")

    terms = []
    for name, entry in losses.items():
        where = f"losses[{name!r}]"
        _require(name not in RESERVED_NAMES,
                 f"{where}: {sorted(RESERVED_NAMES)} are reserved term names.")
        _require(isinstance(entry, dict), f"{where} must be a mapping, got {entry!r}.")
        unknown = sorted(set(entry) - ENTRY_KEYS)
        _require(not unknown, f"{where} has unknown key(s) {unknown}; allowed: "
                 f"{sorted(ENTRY_KEYS)}.")
        _require(isinstance(entry.get("column"), str),
                 f"{where} needs a `column` naming what it scores.")
        for key in ("target", "weight"):
            _require(entry.get(key) is None or isinstance(entry[key], str),
                     f"{where}: `{key}` must be a column name.")
        norm = entry.get("norm")
        _require(isinstance(norm, (int, float)) and not isinstance(norm, bool)
                 and norm >= 1,
                 f"{where} needs `norm` >= 1, the power-mean exponent "
                 f"(1 = mean, 2 = RMS, larger approaches the max); got {norm!r}.")
        map_ = entry.get("map")
        _require(map_ is None or map_ in MAPS,
                 f"{where}: map must be one of {list(MAPS)} or absent; got {map_!r}.")
        _require(map_ not in ("relu", "nrelu") or entry.get("target") is not None,
                 f"{where}: a `{map_}` bound needs a `target` — the operand is "
                 "`column − target`.")
        gain = gains[name]
        _require(isinstance(gain, (int, float)) and not isinstance(gain, bool)
                 and gain >= 0, f"gains[{name!r}] must be a number >= 0; got {gain!r}.")
        terms.append(LossTerm(
            name=name, column=entry["column"], target=entry.get("target"),
            map=map_, norm=float(norm), weight=entry.get("weight"),
            gain=float(gain),
        ))
    return terms


def _role(name: str, nm_mapping: dict[str, list[str]]) -> str:
    if name == POWER:
        return POWER
    for role in ("Y", "U", "D"):
        if name in nm_mapping[role]:
            return role
    return "R"


def _bands(terms: list[LossTerm], nm_mapping: dict[str, list[str]]) -> list[Band]:
    """One band per Y target of a bound, from at most one lower and one upper."""
    sides: dict[str, dict[str, str]] = {}
    for term in terms:
        if not term.is_bound or term.target == POWER:
            continue
        side = "lower" if term.map == "relu" else "upper"
        target_sides = sides.setdefault(term.target, {})
        _require(side not in target_sides,
                 f"`{term.target}` has two {side} bounds ({target_sides.get(side)!r} "
                 f"and {term.column!r}). Combine them into one column of the "
                 "reference file: the reportcard scores each regulated output "
                 "against one band.")
        target_sides[side] = term.column
    _require(bool(sides),
             "No loss term bounds a Y column (`map: relu` or `nrelu` with a Y "
             "`target`), so there is no regulated output for the reportcard to "
             "score. Declare the band the policy is held to.")
    return [Band(y, sides[y].get("lower"), sides[y].get("upper"))
            for y in nm_mapping["Y"] if y in sides]


@beartype
def resolve_objective(cfg, nm_mapping: dict[str, list[str]]) -> Objective:
    """`cfg.losses` resolved against the surrogate's columns, or a refusal
    naming every way it cannot be. `r_columns` also holds what the window and the
    bounds name beyond the surrogate's columns."""
    terms = parse_losses(cfg.losses, cfg.gains)
    roles: dict[str, str] = {}
    for term in terms:
        for name in (term.column, term.target, term.weight):
            if name is not None:
                roles[name] = _role(name, nm_mapping)
    for name in (*cfg.policy_window, *bound_columns(cfg.control_bounds)):
        roles.setdefault(name, _role(name, nm_mapping))
    r_columns = sorted(name for name, role in roles.items() if role == "R")

    targets_of: dict[str, list[str]] = {}
    for term in terms:
        where = f"losses[{term.name!r}]"
        if term.target is not None:
            _require(roles[term.target] in ("Y", POWER),
                     f"{where}: target {term.target!r} must be a Y column or "
                     f"`{POWER}`; Y = {nm_mapping['Y']}.")
        if term.weight is not None:
            _require(roles[term.weight] in ("D", "R"),
                     f"{where}: weight {term.weight!r} must be data — a reference "
                     "or disturbance column — not something the rollout predicts "
                     "or commands.")
        if roles[term.column] == "R" and term.target is not None \
                and roles[term.target] == "Y":
            targets = targets_of.setdefault(term.column, [])
            if term.target not in targets:
                targets.append(term.target)

    # A column shared by several targets takes the first one's stats.
    bound_to = {column: min(targets, key=nm_mapping["Y"].index)
                for column, targets in targets_of.items()}

    scales = cfg.reference_scale
    for name in r_columns:
        if name in bound_to:
            _require(name not in scales,
                     f"reference_scale[{name!r}] is set, but {name!r} bounds "
                     f"{bound_to[name]!r} and is normalised with its stats.")
        else:
            _require(name in scales,
                     f"Reference column {name!r} bounds no Y column, so it needs "
                     "a `reference_scale` entry: the divisor that gives the "
                     "policy an O(1) input (no shift, so a zero stays a zero).")
    stray = sorted(set(scales) - set(r_columns))
    _require(not stray, f"reference_scale names {stray}, which are not "
             f"reference columns of the objective ({r_columns}).")

    return Objective(terms=terms, roles=roles, r_columns=r_columns,
                     bound_to=bound_to, bands=_bands(terms, nm_mapping))


def bound_columns(bounds: dict[str, list]) -> list[str]:
    """The columns `control_bounds` sides name."""
    named = []
    for pair in bounds.values():
        for side in pair:
            if isinstance(side, str):
                named.append(side)
            elif isinstance(side, list):
                named.append(side[0])
    return named


@beartype
def extend_norm_stats(objective: Objective, cfg,
                      norm_stats: NormStats) -> NormStats:
    """The surrogate's norm_stats plus an entry for every reference column.

    A column bound to a Y target carries that target's entry; any other is
    `x / reference_scale[x]`. Built from the config and the surrogate's stats
    alone, so reloading a control run rebuilds it exactly, with no data.
    """
    stats = dict(norm_stats)
    nan = float("nan")
    for name in objective.r_columns:
        if name in objective.bound_to:
            stats[name] = dict(norm_stats[objective.bound_to[name]])
        else:
            stats[name] = {
                "norm_type": "reference_scale", "shift": 0.0,
                "stretch": float(cfg.reference_scale[name]),
                "min": nan, "max": nan, "mean": nan, "stddev": nan, "median": nan,
                "unit": parse_column_unit(name),
            }
    return stats


# ── Evaluating a term ────────────────────────────────────────────


@beartype
def window_power_mean(h: torch.Tensor, q: float,
                      weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per window `(mean_t w·|h|^q)^(1/q)`, [batch].

        q = 1 -> mean, q = 2 -> RMS, q -> inf -> max

    Reduced per window and then averaged, never pooled: the batch axis holds
    independent windows, so a pooled power mean gives most of its gradient to
    the worst window. The gradient within a window is dense over time and
    concentrated on the large steps by a factor growing with q — a hard max
    would put it on one step.

    The root has an infinite derivative at 0, and an all-zero window (a zone
    inside its band throughout) is the common case, so it is taken only where
    the mean is positive; elsewhere the value and its gradient are 0. With
    q = 1 there is no root, so a negative weight (a market paying to absorb
    load) passes through as a negative cost.
    """
    a = h.abs() if q == 1 else h.abs() ** q
    if weight is not None:
        a = a * weight
    per_window = a.mean(dim=1)
    if q != 1:
        positive = per_window > 0
        safe = torch.where(positive, per_window, torch.ones_like(per_window))
        per_window = torch.where(positive, safe ** (1.0 / q),
                                 torch.zeros_like(per_window))
    return per_window


@beartype
def apply_map(h: torch.Tensor, map_: Optional[str]) -> torch.Tensor:
    if map_ == "relu":
        return torch.relu(h)
    if map_ == "nrelu":
        return torch.relu(-h)
    if map_ == "first_difference":
        return h[:, 1:] - h[:, :-1]
    return h


def check_weights(objective: Objective, frames: list, label: str) -> None:
    """Refuse a negative weight under a root it cannot pass through.

    `window_power_mean` with q > 1 takes a root of `mean(w·|h|^q)`, which a negative
    weight can make negative. q = 1 has no root and keeps the sign.
    """
    for term in objective.terms:
        if term.weight is None or term.norm == 1:
            continue
        for i, frame in enumerate(frames):
            low = float(frame[term.weight].min())
            _require(low >= 0,
                     f"losses[{term.name!r}] has norm {term.norm:g} and weight "
                     f"{term.weight!r}, which goes negative ({low:g}) in "
                     f"{label}[{i}]. A power mean above 1 needs a non-negative "
                     "weight; use norm: 1 for a signed one.")
