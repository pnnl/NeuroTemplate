"""
The setpoint hinge: a U column read as the thermostat's clamped error.

Declared per block, in the loss table's words, over the block's own zone:

    "inputs": {
        "Cool SP": {"target": "Zone", "map": "nrelu"},   # relu(zone - setpoint)
        "Heat SP": {"target": "Zone", "map": "relu"},    # relu(setpoint - zone)
    }

A setpoint past the zone commands nothing, as on a real thermostat; without this a
black box renders a setpoint as a smooth, always-active push, and a controller trained
on it inherits that. It runs inside the rollout, over the whole window, because the zone
is predicted there. The error is taken in physical units, so its zero sits where
setpoint meets zone whatever each column's normalisation did, and it is scaled but never
shifted, so the clamped side stays at exactly 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Each map, as the sign on `column - target`.
MAP_SIGNS = {"relu": 1, "nrelu": -1}


def straight_through(soft: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """`hard` in the forward pass, `soft`'s gradient in the backward pass."""
    return soft + (hard - soft).detach()


def soft_clamp_min(x: torch.Tensor) -> torch.Tensor:
    """`relu(x)` forward; softplus's derivative `sigmoid(x)` backward.

    A bare relu hands anything parked on the inactive side exactly zero gradient
    toward the active side. The forward is right, the plant's dead zone is real, but a
    controller that can never learn to leave it is not, so the backward is softened.
    """
    return straight_through(F.softplus(x), F.relu(x))


def hinge_scale(column: str, sign: int, setpoint: dict, zone: dict) -> float:
    """The widest error the recorded ranges permit, which maps every in-range pair to
    about [0, 1]."""
    scale = zone["max"] - setpoint["min"] if sign < 0 else setpoint["max"] - zone["min"]
    if scale <= 0:
        raise ValueError(
            f"the hinge on {column!r} is never active on the recorded ranges (setpoint "
            f"[{setpoint['min']}, {setpoint['max']}], zone [{zone['min']}, {zone['max']}]); "
            "it would be 0 everywhere.")
    return float(scale)


class Hinges(nn.Module):
    """Every hinge a model declares, over one (Y, U) window pair in normalised units.

    `declared` is `{U column: {"target": Y column, "map": "relu" | "nrelu"}}`. Every other
    U column passes through, so the column names and the controller's view of U are
    unchanged: the column just means something different to the model.
    """

    def __init__(self, declared: dict[str, dict], nm_mapping: dict[str, list[str]],
                 norm_stats: dict[str, dict]):
        super().__init__()
        self.declared = {column: dict(entry) for column, entry in declared.items()}
        self.pairs = [(nm_mapping["U"].index(column), nm_mapping["Y"].index(entry["target"]))
                      for column, entry in declared.items()]
        self.signs = [MAP_SIGNS[entry["map"]] for entry in declared.values()]
        rows = [(norm_stats[column], norm_stats[entry["target"]], column, MAP_SIGNS[entry["map"]])
                for column, entry in declared.items()]
        for name, values in (
                ("u_shift", [u["shift"] for u, *_ in rows]),
                ("u_stretch", [u["stretch"] for u, *_ in rows]),
                ("z_shift", [z["shift"] for _, z, *_ in rows]),
                ("z_stretch", [z["stretch"] for _, z, *_ in rows]),
                ("scale", [hinge_scale(c, s, u, z) for u, z, c, s in rows])):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32), persistent=False)

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """`[batch, rows, ny], [batch, rows, nu] -> [batch, rows, nu]`: step `t` of U
        pairs with step `t` of Y."""
        out = list(u.unbind(-1))
        for i, ((j, k), sign) in enumerate(zip(self.pairs, self.signs)):
            setpoint = u[..., j] * self.u_stretch[i] + self.u_shift[i]
            zone = y[..., k] * self.z_stretch[i] + self.z_shift[i]
            out[j] = soft_clamp_min(sign * (setpoint - zone)) / self.scale[i]
        return torch.stack(out, dim=-1)

    def describe(self) -> dict[str, dict]:
        return {column: {**entry, "scale": round(float(self.scale[i]), 4)}
                for i, (column, entry) in enumerate(self.declared.items())}
