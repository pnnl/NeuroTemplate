"""
DPC control policy.

A bounded network wrapped in a Neuromancer Node:

    [yn-window(, U-window), D-window(, R-window)]  ->  U

U is the commands already applied, present only when `policy_window` gives a
U column a `past`; R is the objective's reference columns (control.losses),
present only when it names one. Each column is read over its own `past` and
`future` (see `build_policy`).

Every window arrives in physical units and every command leaves in them.
The learned map alone computes in normalised units (`_NormalizedNet`): its
inputs normalised by their columns' stats, its output squashed into the
column's fixed range by blocks.MLP_bounds (`control_box`). A `control_bounds` side
naming a column maps the command onto a range that moves per step (`_PerStepBounds`),
in physical units. Nothing is rounded to an actuator's resolution: that is the
building's job, and the controller commands through it.
Windowing uses the same input_map convention as modeling/dynamics.get_dynamics
so the policy composes with the frozen dynamics Node.

`policy_catalog` selects the network family the same way `dynamics.FAMILIES`
selects a dynamics family on the sys-id side:

  - "mlp"    — bounded MLP, `policy_layers` hidden layers of width
               2**log2_policy_hidden
  - "linear" — the same bounded map with no hidden layers, i.e. a single
               saturated affine feedback law u = sat(K @ x + b) over the
               flattened feature window. This is the structure of an explicit
               MPC law inside its unconstrained region, and its gain matrix is
               small enough to read.
"""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import Optional

import torch
import torch.nn as nn

from neuromancer.system import Node
from neuromancer.modules import blocks

from neurotemplate.common.data_utils import NormStats
from neurotemplate.control.losses import resolve_objective
from neurotemplate.modeling import windows

from neurotemplate._typing import beartype


# Nonlinearities selectable from YAML. Kept small on purpose — these are the
# activations that actually get used for control policies.
POLICY_NONLIN = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}

class _NormalizedNet(nn.Module):
    """The learned map, in normalised units.

    Each physical input window `[batch, rows * n]` is normalised by its
    columns' stats, then narrowed to the (column, timestep) pairs declared:
    SystemPreview's input_map gives one window per key, sized to the deepest
    `past` and `future` any of its columns declared, and a column that asked
    for less would otherwise see steps it never requested. The narrowing is a
    fixed permutation computed once at construction, applied with
    `index_select` against registered buffers — same discipline as
    `BlockStack` (modeling/dynamics.py): no data-dependent control flow.

    The output is the net's own: normalised U, inside the net's bounds.
    """

    def __init__(self, base: nn.Module, stats: list[tuple[torch.Tensor, torch.Tensor]],
                 gathers: dict[int, torch.Tensor]):
        super().__init__()
        self.base = base
        for i, (shift, stretch) in enumerate(stats):
            self.register_buffer(f"_shift_{i}", shift, persistent=False)
            self.register_buffer(f"_stretch_{i}", stretch, persistent=False)
        self.n_inputs = len(stats)
        self.indices = sorted(gathers)
        for i in self.indices:
            self.register_buffer(f"_gather_{i}", gathers[i], persistent=False)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        normalized = []
        for i, x in enumerate(args):
            shift = getattr(self, f"_shift_{i}")
            if shift.numel():   # a key with no columns has nothing to normalise
                window = x.reshape(x.shape[0], -1, shift.shape[0])
                x = ((window - shift) / getattr(self, f"_stretch_{i}")).reshape(x.shape[0], -1)
            if i in self.indices:
                x = x.index_select(-1, getattr(self, f"_gather_{i}"))
            normalized.append(x)
        return self.base(*normalized)


def _current(x: torch.Tensor, block: int, width: int) -> torch.Tensor:
    """The `block`-th step out of a flattened `(batch, window*width)` window."""
    return x[:, block * width:(block + 1) * width]


class _Commands(nn.Module):
    """The net's normalised output as physical commands: `z · stretch + shift`."""

    def __init__(self, base: nn.Module, u_shift: torch.Tensor, u_stretch: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("u_shift", u_shift, persistent=False)
        self.register_buffer("u_stretch", u_stretch, persistent=False)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.base(*args) * self.u_stretch + self.u_shift


def _bound_side(side) -> tuple[Optional[str], float]:
    """`(column or None, offset)` for one side of a `control_bounds` pair: a
    number is `(None, value)`, a name `(name, 0)`, `[name, offset]` itself."""
    if isinstance(side, list):
        return side[0], float(side[1])
    if isinstance(side, str):
        return side, 0.0
    return None, float(side)


class _PerStepBounds(nn.Module):
    """Map bounded columns' commands from their fixed range onto a per-step range.

    For each column `control_bounds` names with a moving side, the base command `u`, in
    `[umin, umax]`, becomes `low + (u - umin) / (umax - umin) · (high - low)`,
    with `low` and `high` either fixed or read at the current step from a
    column: a reference (R) or a disturbance (D) at its window's present row,
    or another command (U) as this step emits it. `high` is held at least
    `low`. A linear map rather than a clamp, so the whole range keeps a
    gradient.

    Columns are mapped in `order`, each after any bounded command it reads, so
    a side naming a bounded command reads that command's final value.
    """

    def __init__(self, base: nn.Module, u_columns: list[str], bounds: dict[str, list],
                 order: list[str], umin: torch.Tensor, umax: torch.Tensor,
                 sources: dict[str, tuple]):
        super().__init__()
        self.base = base
        self.columns = [u_columns.index(c) for c in order]
        self.register_buffer("umin", umin)
        self.register_buffer("umax", umax)
        # Each side as (source or None, offset): a source is `("U", column)`
        # for a command, or (positional input, present row, window width, column)
        # for a reference; the side is its value plus `offset`. A number is the
        # constant `offset`.
        self.sides = [(None if name is None else sources[name], margin)
                      for column in order
                      for name, margin in map(_bound_side, bounds[column])]

    def _side(self, k: int, args: tuple, columns: list, like: torch.Tensor) -> torch.Tensor:
        source, offset = self.sides[k]
        if source is None:
            return torch.full_like(like, offset)
        if source[0] == "U":
            return columns[source[1]] + offset
        key, row, width, column = source
        return _current(args[key], row, width)[:, column] + offset

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        u = self.base(*args)
        columns = list(u.unbind(-1))
        for n, j in enumerate(self.columns):
            low = self._side(2 * n, args, columns, u[:, j])
            high = torch.maximum(self._side(2 * n + 1, args, columns, u[:, j]), low)
            share = (u[:, j] - self.umin[j]) / (self.umax[j] - self.umin[j])
            columns[j] = low + share * (high - low)
        return torch.stack(columns, dim=-1)

    def base_of(self, u: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        """The base command `forward` maps onto `u` at one step, from that
        step's windows: its inverse. A side naming a command reads that
        command's final value, as in `forward`."""
        columns = list(u.unbind(-1))
        base = list(columns)
        for n, j in enumerate(self.columns):
            low = self._side(2 * n, args, columns, u[:, j])
            high = torch.maximum(self._side(2 * n + 1, args, columns, u[:, j]), low)
            share = ((u[:, j] - low) / (high - low).clamp_min(1e-9)).clamp(0.0, 1.0)
            base[j] = self.umin[j] + share * (self.umax[j] - self.umin[j])
        return torch.stack(base, dim=-1)


def _window_table(cfg, nm_mapping: dict[str, list[str]],
                  r_columns: list[str]) -> dict[str, list[tuple[int, int]]]:
    """`cfg.policy_window` as `(past, future)` per column, per input key, in
    the order each key carries its columns.

    The table must name every Y, U, D and reference column and nothing else.
    A Y column takes no `future` (it is not measured yet), nor does a U
    column (it is the policy's own output); a D or reference column needs
    one, because how far ahead to look at a forecast is the decision.
    """
    columns = {"yn": nm_mapping["Y"], "U": nm_mapping["U"], "D": nm_mapping["D"],
               "R": r_columns}
    listed = [c for cols in columns.values() for c in cols]
    missing = [c for c in listed if c not in cfg.policy_window]
    extra = sorted(set(cfg.policy_window) - set(listed))
    if missing or extra:
        raise ValueError(
            "policy_window must list every Y, U, D and reference column and "
            f"nothing else. Missing: {missing}. Not a policy input: {extra}.")
    wrong = [f"{c} ({'Y' if key == 'yn' else key}: "
             f"{'takes no' if key in ('yn', 'U') else 'needs a'} `future`)"
             for key, cols in columns.items() for c in cols
             if ("future" in cfg.policy_window[c]) != (key in ("D", "R"))]
    if wrong:
        raise ValueError(
            f"policy_window: {', '.join(wrong)}. A Y column is not measured "
            "ahead and a U column is the policy's own output; a D or "
            "reference column is a forecast, and its `future` says how far "
            "ahead the policy sees it.")
    return {key: [(cfg.policy_window[c]["past"], cfg.policy_window[c].get("future", 0))
                  for c in cols]
            for key, cols in columns.items()}


def _input_keys(table: dict[str, list[tuple[int, int]]]) -> list[str]:
    """The policy's positional inputs: U only when some U column has a past,
    R only when the objective names a reference column."""
    reads_u = any(past > 0 for past, _ in table["U"])
    return (["yn"] + (["U"] if reads_u else []) + ["D"]
            + (["R"] if table["R"] else []))


def _input_window(key: str, spans: list[tuple[int, int]]) -> dict:
    """The input_map entry for one key: its deepest `past` and `future`.

    A U window ends at the current step like every other, but U holds only
    the commands already applied, so that last row is SystemPreview's
    `nearest` copy of the previous command; `_key_gather` drops it.
    """
    return {"past": max(past for past, _ in spans),
            "future": 0 if key == "U" else max(future for _, future in spans),
            "pad_mode": "nearest"}


def _key_gather(key: str, spans: list[tuple[int, int]]) -> list[int]:
    """Indices narrowing one key's window to each column's own steps:
    `t` relative to the current step with `-past <= t <= future`, and
    `-past <= t <= -1` for U."""
    window = _input_window(key, spans)
    now = window["past"]
    keep = tuple((j, now - past, now - 1 if key == "U" else now + future)
                 for j, (past, future) in enumerate(spans))
    rows = window["past"] + 1 + window["future"]
    return windows.window_gather((len(spans),), (rows,), (keep,))


def _bounds_follow_device(net: nn.Module) -> nn.Module:
    """Re-register `MLP_bounds`' `min`/`max` as buffers.

    Neuromancer keeps them as plain attributes, which `.to(device)` never
    moves, so a policy on MPS or CUDA squashed against CPU bounds and
    failed on its first call. Non-persistent: they are rebuilt from the
    config on every load, so the checkpoint keys stay as they were.
    """
    for name in ("min", "max"):
        value = getattr(net, name)
        delattr(net, name)
        net.register_buffer(name, value, persistent=False)
    return net


def _build_mlp_net(cfg, insize: int, nu: int, umin, umax) -> nn.Module:
    """Bounded MLP with `policy_layers` hidden layers of equal width."""
    return _bounds_follow_device(blocks.MLP_bounds(
        insize=insize,
        outsize=nu,
        bias=True,
        linear_map=nn.Linear,
        nonlin=POLICY_NONLIN[cfg.policy_nonlin],
        hsizes=cfg.policy_layers * [2**cfg.log2_policy_hidden],
        min=umin,
        max=umax,
        method=cfg.policy_bound_method,
    ))


def _build_linear_net(cfg, insize: int, nu: int, umin, umax) -> nn.Module:
    """Saturated affine feedback law: one linear layer, then the bound squash.

    hsizes=[] makes blocks.MLP build a single linear_map(insize, nu) with an
    Identity nonlinearity, so this is u = sat(K @ x + b) — no hidden layer.
    """
    return _bounds_follow_device(blocks.MLP_bounds(
        insize=insize,
        outsize=nu,
        bias=True,
        linear_map=nn.Linear,
        nonlin=nn.Identity,
        hsizes=[],
        min=umin,
        max=umax,
        method=cfg.policy_bound_method,
    ))


policy_catalog = {
    "mlp": _build_mlp_net,
    "linear": _build_linear_net,
}

# Bound squashing options come from Neuromancer, so read them off the class
# rather than re-listing them here.
POLICY_BOUND_METHODS = list(blocks.MLP_bounds.bound_methods)


@beartype
def build_policy(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> Node:
    if cfg.policy not in policy_catalog:
        raise ValueError(
            f"Unknown policy '{cfg.policy}'. "
            f"Available: {list(policy_catalog.keys())}"
        )

    nu = len(nm_mapping["U"])
    r_columns = resolve_objective(cfg, nm_mapping).r_columns
    device = sysid_cfg.device

    # Each key's window (see SystemPreview.get_mapped_data) is sized to its
    # deepest column, and each column is gathered down to its own steps
    # before the net sees it, so the input width is a per-column sum:
    # past + 1 + future for yn, D and R, and past for U, whose window ends
    # at the previous step.
    table = _window_table(cfg, nm_mapping, r_columns)
    keys = _input_keys(table)
    input_map = {key: _input_window(key, table[key]) for key in keys}
    key_gathers = {key: _key_gather(key, table[key]) for key in keys}
    insize = sum(len(indices) for indices in key_gathers.values())

    # The fixed range, physical, and the same in the net's normalised units.
    u_cols = nm_mapping["U"]
    u_shift, u_stretch = _column_stats(norm_stats, u_cols, device)
    umin, umax = control_box(cfg, nm_mapping, norm_stats, device)
    net = policy_catalog[cfg.policy](cfg, insize, nu, (umin - u_shift) / u_stretch,
                                     (umax - u_shift) / u_stretch).to(device)

    # Each window normalised and gathered to its columns' own steps, in front
    # of whatever net was just built. The per-step bounds read the raw physical
    # windows. A key whose columns all read its whole window gathers nothing.
    columns = {"yn": nm_mapping["Y"], "U": u_cols, "D": nm_mapping["D"], "R": r_columns}
    gathers = {}
    for i, key in enumerate(keys):
        window = input_map[key]
        width = len(table[key]) * (window["past"] + 1 + window["future"])
        if key_gathers[key] != list(range(width)):
            gathers[i] = torch.tensor(key_gathers[key], dtype=torch.long, device=device)
    net = _NormalizedNet(net, [_column_stats(norm_stats, columns[key], device)
                               for key in keys], gathers)

    net = _Commands(net, u_shift, u_stretch)
    net = with_control_bounds(net, cfg, nm_mapping, keys, input_map, umin, umax).to(device)

    return Node(
        net,
        keys,
        ["U"],
        name="DPC Policy",
        input_map=input_map,
    )


def _column_stats(norm_stats: NormStats, columns: list[str],
                  device) -> tuple[torch.Tensor, torch.Tensor]:
    """(shift, stretch) of `columns`, each [n], float32."""
    return tuple(torch.tensor([float(norm_stats[c][field]) for c in columns],
                              dtype=torch.float32, device=device)
                 for field in ("shift", "stretch"))


def control_box(cfg, nm_mapping: dict[str, list[str]], norm_stats: NormStats,
                device) -> tuple[torch.Tensor, torch.Tensor]:
    """(umin, umax), each U column's fixed range, float32, physical units: a numeric
    side of `cfg.control_bounds`, or the column's recorded range. What a controller
    commands inside, and what "at a bound" means on the reportcard."""
    u_columns = nm_mapping["U"]
    unknown = sorted(set(cfg.control_bounds) - set(u_columns))
    if unknown:
        raise ValueError(f"control_bounds names {unknown}, which are not U columns "
                         f"({u_columns}).")
    low, high = [], []
    for column in u_columns:
        pair = cfg.control_bounds.get(column, [None, None])
        low.append(float(pair[0]) if _is_number(pair[0]) else norm_stats[column]["min"])
        high.append(float(pair[1]) if _is_number(pair[1]) else norm_stats[column]["max"])
        if high[-1] <= low[-1] and column in cfg.control_bounds:
            raise ValueError(f"control_bounds[{column!r}] leaves no range: [{low[-1]}, "
                             f"{high[-1]}]. A numeric side beside a column side bounds the "
                             "column's recorded range on the other.")
    return tuple(torch.tensor(side, dtype=torch.float32, device=device)
                 for side in (low, high))


def _is_number(side) -> bool:
    return isinstance(side, (int, float)) and not isinstance(side, bool)


def with_control_bounds(net: nn.Module, cfg, nm_mapping: dict[str, list[str]],
                        keys: list[str], input_map: dict, umin: torch.Tensor,
                        umax: torch.Tensor) -> nn.Module:
    """`net`, commanding over `[umin, umax]`, behind the `cfg.control_bounds` that move
    per step (`_PerStepBounds`); `net` itself when none does. Any controller reading the
    policy's windows can go behind it."""
    moving = {column: pair for column, pair in cfg.control_bounds.items()
              if not all(map(_is_number, pair))}
    if not moving:
        return net
    return _per_step_bounds(net, moving, nm_mapping, keys, input_map,
                            resolve_objective(cfg, nm_mapping).r_columns, umin, umax)


def _per_step_bounds(net, moving: dict[str, list], nm_mapping, keys, input_map, r_columns,
                     umin, umax) -> _PerStepBounds:
    """`_PerStepBounds` for `moving`, every name checked first.

    A named side is a reference column, read from the R window at its present row (the
    key's deepest `past`), or another command. Commands are ordered so each follows the
    bounded commands its sides name; a cycle is refused."""
    u_columns = list(nm_mapping["U"])
    sources, problems = {}, []
    after = {}
    for column, pair in moving.items():
        if column not in u_columns:
            problems.append(f"{column!r} is not a U column ({u_columns})")
        after[column] = set()
        for raw in pair:
            side, _ = _bound_side(raw)
            if side is None:
                continue
            if side in u_columns:
                sources[side] = ("U", u_columns.index(side))
                if side in moving:
                    after[column].add(side)
                continue
            if side not in r_columns or "R" not in keys:
                problems.append(f"{column!r} is bounded by {side!r}, which is not a "
                                f"command or a reference column the policy reads "
                                f"(U {u_columns}, R {r_columns})")
                continue
            sources[side] = (keys.index("R"), input_map["R"]["past"], len(r_columns),
                             r_columns.index(side))
    if problems:
        raise ValueError("control_bounds: " + "; ".join(problems) + ".")
    try:
        order = list(TopologicalSorter(after).static_order())
    except CycleError as e:
        raise ValueError(f"control_bounds: the bounds read each other in a cycle "
                         f"({' -> '.join(e.args[1])}); one of them has to be fixed "
                         "or read from data.") from None
    return _PerStepBounds(net, u_columns, moving, order, umin, umax, sources)


# ── Policy description (for reportcard) ──────────────────────────


@beartype
def describe_policy(
    policy_node: Node,
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
) -> dict:
    """
    Describe the constructed policy for the reportcard.

    Mirrors modeling.dynamics.describe_dynamics: reads what is available on
    the built module rather than re-deriving it from cfg, so a mismatch
    between config and reality shows up on the card.

    The window table is reported alongside `insize` so a reader can re-check
    the sum — past + 1 + future per Y, D and reference column, past per U
    column — by eye: a wrong window is otherwise an invisible failure.
    """
    # Unwrap: Node -> [_PerStepBounds] -> _Commands -> _NormalizedNet -> MLP_bounds
    raw = policy_node.callable
    if isinstance(raw, _PerStepBounds):
        raw = raw.base
    net = raw.base.base

    r_columns = resolve_objective(cfg, nm_mapping).r_columns
    columns = nm_mapping["Y"] + nm_mapping["U"] + nm_mapping["D"] + r_columns

    # Actuator limits in physical units, which is what a reader can sanity
    # check against the building.
    u_cols = nm_mapping["U"]
    umin_phys, umax_phys = ((bound * raw.u_stretch + raw.u_shift).tolist()
                            for bound in (net.min, net.max))

    return {
        "family": cfg.policy,
        "nonlin": cfg.policy_nonlin if cfg.policy == "mlp" else "n/a (linear)",
        "bound_method": cfg.policy_bound_method,
        "insize": int(net.in_features),
        "outsize": int(net.out_features),
        "inputs": list(policy_node.input_keys),
        "ny": len(nm_mapping["Y"]),
        "window": {c: dict(cfg.policy_window[c]) for c in columns},
        "hsizes": [int(lin.out_features) for lin in net.linear[:-1]],
        "param_count": sum(p.numel() for p in net.parameters()),
        "u_cols": list(u_cols),
        "umin_phys": umin_phys,
        "umax_phys": umax_phys,
        "control_bounds": dict(cfg.control_bounds) or None,
    }
