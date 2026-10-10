"""
Derived inputs: the pre-stack.

`PreStack` is the input-side counterpart of `HeadStack`: a closed
taxonomy of small `PreTransform` kinds, persisted by value and reported
through `config()`. Where a head maps one Y channel to itself, a
pre-transform maps a U column and one of the model's own Y channels to a
replacement for that U column — the `demand` kind is the thermostat's
clamped error `relu(sign · (setpoint − zone))`. It runs inside the
autoregressive loop, over the whole history window, so the zone it reads
is recorded data inside the seeded history and the model's own prediction
past it. The dynamics model never sees the raw column. A transform may be
gated by a D column (equipment that runs only on a schedule): the D column,
in physical units, multiplies the feature. See `Demand` for the physics and
`PreStack.forward` for the window layout it relies on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurotemplate._typing import beartype
from neurotemplate.modeling.heads import straight_through


def soft_clamp_min(x: torch.Tensor) -> torch.Tensor:
    """`relu(x)` forward; softplus's derivative `sigmoid(x)` backward.

    A bare relu hands anything parked on the inactive side exactly zero
    gradient toward the active side. The forward is right — the plant's
    dead zone is real — but a controller that can never learn to leave it
    is not, so the backward is softened.
    """
    return straight_through(F.softplus(x), F.relu(x))


# ── Pre-transforms: derived inputs ──────────────────────────────


class PreTransform(nn.Module, ABC):
    """One derived input column, replacing a declared U column.

    Subclasses set `kind` and implement `forward(u, y, gate)` — the declared
    column, its paired Y channel and its gate column (or None), all
    normalised, any matching shape — plus `config()`, the JSON-safe
    constructor kwargs beyond the columns' norm stats, which
    `PreStack.from_config` supplies itself.
    """

    kind: str = ""

    @abstractmethod
    def forward(self, u: torch.Tensor, y: torch.Tensor,
                gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The replacement column, same shape as `u`."""

    def config(self) -> dict:
        return {}

    def describe(self) -> dict:
        """What the reportcard prints: `config()` plus any derived constants."""
        return self.config()


def demand_scale(sign: int, control: dict, state: dict) -> float:
    """The widest demand the recorded ranges permit, in physical units.

    Cooling (`sign = -1`) peaks with the zone at its recorded maximum against
    the setpoint at its minimum; heating mirrors it. Dividing by this maps
    every in-range pair to about [0, 1].
    """
    if sign == -1:
        scale = state["max"] - control["min"]
    else:
        scale = control["max"] - state["min"]
    if scale <= 0:
        raise ValueError(
            f"demand with sign={sign} is never active on the recorded ranges "
            f"(control [{control['min']}, {control['max']}], state "
            f"[{state['min']}, {state['max']}]); the feature would be 0 everywhere."
        )
    return float(scale)


class Demand(PreTransform):
    """`relu(sign · (u − z)) / scale`: the thermostat's error, clamped at zero.

    `sign` is the direction the state moves while the demand is active:
    `-1` gives `relu(z − u)` (cooling), `+1` gives `relu(u − z)` (heating),
    the `SynthSpec.responds` convention. The error is taken in physical
    units, so its zero-crossing sits where the setpoint meets the zone
    whatever each column's normalisation did to them, and it is scaled but
    never shifted, so the clamped atom stays at exactly 0.

    `gate` is the norm stats of a D column the equipment runs on (an
    occupancy schedule an AHU follows): the feature is multiplied by that
    column in physical units, so the thermostat asks for nothing while the
    gate is 0. Its recorded range must lie in [0, 1] — a share of the time
    on, not a gain.

    No learned parameters. The norm stats are persistent buffers, so a
    reloaded checkpoint keeps the numbers it trained with.
    """

    kind = "demand"

    def __init__(self, sign: int, control: dict, state: dict,
                 gate: Optional[dict] = None):
        super().__init__()
        if sign not in (-1, 1):
            raise ValueError(f"Demand requires sign in (-1, 1), got {sign}")
        self.sign = int(sign)
        scale = demand_scale(sign, control, state)
        buffers = [("u_shift", control["shift"]), ("u_stretch", control["stretch"]),
                   ("z_shift", state["shift"]), ("z_stretch", state["stretch"]),
                   ("scale", scale)]
        self.gated = gate is not None
        if self.gated:
            if gate["min"] < 0 or gate["max"] > 1:
                raise ValueError(
                    f"a demand gate must be recorded within [0, 1] (it multiplies "
                    f"the demand); this one spans [{gate['min']}, {gate['max']}]."
                )
            buffers += [("g_shift", gate["shift"]), ("g_stretch", gate["stretch"])]
        for name, value in buffers:
            self.register_buffer(name, torch.tensor(float(value)))

    def forward(self, u: torch.Tensor, y: torch.Tensor,
                gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        u_phys = u * self.u_stretch + self.u_shift
        z_phys = y * self.z_stretch + self.z_shift
        demand = soft_clamp_min(self.sign * (u_phys - z_phys)) / self.scale
        if not self.gated:
            return demand
        return demand * (gate * self.g_stretch + self.g_shift)

    def config(self) -> dict:
        return {"sign": self.sign}

    def describe(self) -> dict:
        return {**self.config(), "scale": round(float(self.scale), 4)}


_PRE_KIND_TO_CLASS: dict[str, type[PreTransform]] = {
    cls.kind: cls for cls in (Demand,)
}


class PreStack(nn.Module):
    """Derived inputs over one (Y, U) window pair.

    Each transform replaces one U column with a value computed from that
    column and one Y channel at the same timestep, across the whole
    `in_steps` window; every other U column passes through untouched, so
    `nu`, the column names, the gradient masks and the controller's view of
    U are all unchanged — the column just means something different.

    Args:
        transforms: one `PreTransform` per replaced column.
        columns: `(u_name, y_name)` per transform — the column it replaces
            and the Y channel it reads.
        y_names / u_names / d_names: the full `nm_mapping` Y, U and D lists.
        in_steps: history window length.
        gates: per transform, the D column that gates it, or None; None
            for the whole list means no transform is gated.
    """

    @beartype
    def __init__(self, transforms: list[PreTransform],
                 columns: list[tuple[str, str]],
                 y_names: list[str], u_names: list[str], d_names: list[str],
                 in_steps: int, gates: Optional[list[Optional[str]]] = None):
        super().__init__()
        gates = [None] * len(transforms) if gates is None else list(gates)
        if not len(transforms) == len(columns) == len(gates):
            raise ValueError(
                f"transforms ({len(transforms)}), columns ({len(columns)}) and "
                f"gates ({len(gates)}) must have the same length"
            )
        replaced = [u for u, _ in columns]
        if len(set(replaced)) != len(replaced):
            raise ValueError(f"a U column may be replaced only once, got {replaced}")
        unknown = [(u, y) for u, y in columns if u not in u_names or y not in y_names]
        unknown += [g for g in gates if g is not None and g not in d_names]
        if unknown:
            raise ValueError(
                f"pre-transform columns {unknown} are not in U={u_names} / "
                f"Y={y_names} / D={d_names}"
            )
        self.transforms = nn.ModuleList(transforms)
        self.columns = list(columns)
        self.gates = gates
        self._pairs = [(u_names.index(u), y_names.index(y)) for u, y in columns]
        self._gate_idx = [None if g is None else d_names.index(g) for g in gates]
        self.ny, self.nu, self.nd = len(y_names), len(u_names), len(d_names)
        self.in_steps = in_steps

    def forward(self, y: torch.Tensor, u: torch.Tensor,
                d: torch.Tensor) -> torch.Tensor:
        """`[..., in_steps·ny], [..., in_steps·nu], [..., in_steps·nd]
        -> [..., in_steps·nu]`.

        Every window is key-major and, within the key, time-major oldest
        first (see `windows`), so reshaping each to `[..., in_steps, n]`
        pairs step `t` of U with step `t` of Y and D. The leading shape is
        kept as-is: a batch during a rollout, nothing under vmap.
        """
        lead = y.shape[:-1]
        y_steps = y.reshape(*lead, self.in_steps, self.ny)
        u_steps = u.reshape(*lead, self.in_steps, self.nu)
        d_steps = d.reshape(*lead, self.in_steps, self.nd)
        out = list(u_steps.unbind(-1))
        for (u_idx, y_idx), g_idx, transform in zip(self._pairs, self._gate_idx,
                                                    self.transforms):
            gate = None if g_idx is None else d_steps[..., g_idx]
            out[u_idx] = transform(u_steps[..., u_idx], y_steps[..., y_idx], gate)
        return torch.stack(out, dim=-1).reshape(*lead, self.in_steps * self.nu)

    # ── Serialization / reporting ─────────────────────────────

    def _columns_of(self, i: int) -> dict:
        """The column names of transform `i`, as its config entry spells them."""
        u, y = self.columns[i]
        gate = self.gates[i]
        return {"state": y} if gate is None else {"state": y, "gate": gate}

    def to_config(self) -> dict[str, dict]:
        """`{u_column: {kind, state[, gate], ...kwargs}}` — the same shape as
        the `inputs` entries of a `block_mapping` spec, which is where these
        come from and where they are persisted (`block_spec.json`)."""
        return {
            u: {"kind": t.kind, **self._columns_of(i), **t.config()}
            for i, ((u, _), t) in enumerate(zip(self.columns, self.transforms))
        }

    def describe(self) -> dict[str, dict]:
        return {
            u: {"kind": t.kind, **self._columns_of(i), **t.describe()}
            for i, ((u, _), t) in enumerate(zip(self.columns, self.transforms))
        }

    @classmethod
    def from_config(
        cls, data: dict[str, dict], y_names: list[str], u_names: list[str],
        d_names: list[str], norm_stats: dict[str, dict], in_steps: int,
    ) -> "PreStack":
        """Inverse of `to_config`. Each entry's column stats come from
        `norm_stats`; `load_state_dict` afterwards restores the buffers the
        checkpoint trained with."""
        transforms, columns, gates = [], [], []
        for u_name, entry in data.items():
            kind = entry["kind"]
            if kind not in _PRE_KIND_TO_CLASS:
                raise ValueError(
                    f"Unknown pre-transform kind '{kind}' for column '{u_name}'. "
                    f"Expected one of: {sorted(_PRE_KIND_TO_CLASS)}."
                )
            y_name, gate = entry["state"], entry.get("gate")
            kwargs = {k: v for k, v in entry.items()
                      if k not in ("kind", "state", "gate")}
            transforms.append(_PRE_KIND_TO_CLASS[kind](
                control=norm_stats[u_name], state=norm_stats[y_name],
                gate=None if gate is None else norm_stats[gate], **kwargs,
            ))
            columns.append((u_name, y_name))
            gates.append(gate)
        return cls(transforms, columns, y_names, u_names, d_names, in_steps, gates)

