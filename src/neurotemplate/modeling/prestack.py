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
past it. The family never sees the raw column. See `Demand` for the physics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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

    Subclasses set `kind` and implement `forward(u, y)` — the declared
    column and its paired Y channel, both normalised, any matching shape —
    plus `config()`, the JSON-safe constructor kwargs beyond the two
    columns' norm stats, which `PreStack.from_config` supplies itself.
    """

    kind: str = ""

    @abstractmethod
    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    No learned parameters. The norm stats are persistent buffers, so a
    reloaded checkpoint keeps the numbers it trained with.
    """

    kind = "demand"

    def __init__(self, sign: int, control: dict, state: dict):
        super().__init__()
        if sign not in (-1, 1):
            raise ValueError(f"Demand requires sign in (-1, 1), got {sign}")
        self.sign = int(sign)
        scale = demand_scale(sign, control, state)
        for name, value in (("u_shift", control["shift"]), ("u_stretch", control["stretch"]),
                            ("z_shift", state["shift"]), ("z_stretch", state["stretch"]),
                            ("scale", scale)):
            self.register_buffer(name, torch.tensor(float(value)))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u_phys = u * self.u_stretch + self.u_shift
        z_phys = y * self.z_stretch + self.z_shift
        error = self.sign * (u_phys - z_phys)
        return soft_clamp_min(error) / self.scale

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
    `history_steps` window; every other U column passes through untouched, so
    `nu`, the column names, the gradient masks and the controller's view of
    U are all unchanged — the column just means something different.

    Args:
        transforms: one `PreTransform` per replaced column.
        columns: `(u_name, y_name)` per transform — the column it replaces
            and the Y channel it reads.
        y_names / u_names: the full `nm_mapping["Y"]` and `["U"]` lists.
    """

    @beartype
    def __init__(self, transforms: list[PreTransform],
                 columns: list[tuple[str, str]],
                 y_names: list[str], u_names: list[str]):
        super().__init__()
        if len(transforms) != len(columns):
            raise ValueError(
                f"transforms ({len(transforms)}) and columns ({len(columns)}) "
                "must have the same length"
            )
        replaced = [u for u, _ in columns]
        if len(set(replaced)) != len(replaced):
            raise ValueError(f"a U column may be replaced only once, got {replaced}")
        unknown = [(u, y) for u, y in columns if u not in u_names or y not in y_names]
        if unknown:
            raise ValueError(
                f"pre-transform columns {unknown} are not in U={u_names} / Y={y_names}"
            )
        self.transforms = nn.ModuleList(transforms)
        self.columns = list(columns)
        self._pairs = [(u_names.index(u), y_names.index(y)) for u, y in columns]

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """`[batch, rows, ny], [batch, rows, nu] -> [batch, rows, nu]`: step `t` of
        U pairs with step `t` of Y."""
        out = list(u.unbind(-1))
        for (u_idx, y_idx), transform in zip(self._pairs, self.transforms):
            out[u_idx] = transform(u[..., u_idx], y[..., y_idx])
        return torch.stack(out, dim=-1)

    # ── Serialization / reporting ─────────────────────────────

    def to_config(self) -> dict[str, dict]:
        """`{u_column: {kind, state, ...kwargs}}` — the same shape as the
        `inputs` entries of a `block_mapping` spec, which is where these
        come from and where they are persisted (`block_spec.json`)."""
        return {
            u: {"kind": t.kind, "state": y, **t.config()}
            for (u, y), t in zip(self.columns, self.transforms)
        }

    def describe(self) -> dict[str, dict]:
        return {
            u: {"kind": t.kind, "state": y, **t.describe()}
            for (u, y), t in zip(self.columns, self.transforms)
        }

    @classmethod
    def from_config(
        cls, data: dict[str, dict], y_names: list[str], u_names: list[str],
        norm_stats: dict[str, dict],
    ) -> "PreStack":
        """Inverse of `to_config`. Each entry's column stats come from
        `norm_stats`; `load_state_dict` afterwards restores the buffers the
        checkpoint trained with."""
        transforms, columns = [], []
        for u_name, entry in data.items():
            kind = entry["kind"]
            if kind not in _PRE_KIND_TO_CLASS:
                raise ValueError(
                    f"Unknown pre-transform kind '{kind}' for column '{u_name}'. "
                    f"Expected one of: {sorted(_PRE_KIND_TO_CLASS)}."
                )
            y_name = entry["state"]
            kwargs = {k: v for k, v in entry.items() if k not in ("kind", "state")}
            transforms.append(_PRE_KIND_TO_CLASS[kind](
                control=norm_stats[u_name], state=norm_stats[y_name], **kwargs,
            ))
            columns.append((u_name, y_name))
        return cls(transforms, columns, y_names, u_names)

