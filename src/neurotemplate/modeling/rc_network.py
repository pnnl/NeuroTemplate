"""
`thermal`'s parts as blocks: `rc_network` and `thermostat`.

    setpoints ─[thermostat]→ heat ─[rc_network]→ zone temperature
                 heat from a U ────[rc_network]→ zone temperature

Two blocks that know nothing of each other. An `rc_network` block's Y is the
zone temperature and nothing else, and its network's inputs are whatever
columns the declaration names: a `driven_by` column (the heat a thermostat
block produced this step), a U (a commanded heat) or a D (solar, a schedule).
Its window is the run's `context_steps`, the history its filter places the
hidden nodes from. `ambient` is the outdoor temperature the network conducts
to, and at least one of its parameters is pinned: `scale` says which.

A `thermostat` block's Y is the heat, its U the setpoints it follows, and it
`reads` the zone as of the current row.

Everything here is unit conversion and slicing between a block's normalised
flat window and the physical parts in `thermal`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from neurotemplate.modeling.blocks import Block
from neurotemplate.modeling.thermal import RCZone, thermostat


class _Physical(nn.Module):
    """Reads physical columns by name out of a block's flat window, each `rows`
    deep. Key-major, row by row, oldest first (see `windows`)."""

    def __init__(self, block: Block, rows: int, norm_stats: dict):
        super().__init__()
        offset, self._cells = 0, {}
        names = block.y_in_names + block.u_names + block.d_names
        for key_names in (block.y_in_names, block.u_names, block.d_names):
            width = len(key_names)
            for j, column in enumerate(key_names):
                self._cells[column] = (offset + j, offset + width * rows, width)
            offset += width * rows
        for field in ("shift", "stretch"):
            self.register_buffer(f"_{field}", torch.tensor(
                [float(norm_stats[c][field]) for c in names]), persistent=False)
        self._index = {c: i for i, c in enumerate(names)}

    def column(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Column `name` in physical units, [..., rows]."""
        start, stop, width = self._cells[name]
        i = self._index[name]
        return x[..., start:stop:width] * self._stretch[i] + self._shift[i]

    def increment(self, name: str, value: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        return (value - last) / self._stretch[self._index[name]]


def _refuse_scale_mismatch(block: Block, network) -> list[str]:
    """`scale` and the declared network must say the same thing, and the network
    must pin something: without a pin only ratios like gain / C are placed."""
    problems = []
    if not network.pinned:
        problems.append("its network pins nothing, so C, R and the gains are placed only up "
                        "to a common scale. Pin one (the scale column's gain, or a C).")
    for name, value in (block.scale or {}).items():
        keys = [k for k in network.parameter_names() if k == name or
                any(k == f"gain_{c}_{n}" for c, n in network.inputs if c == name)]
        if not keys:
            problems.append(f"`scale` names {name!r}, which is no input or parameter of its "
                            "network.")
        for key in keys:
            if network.pinned.get(key) != value:
                problems.append(f"`scale` holds {name!r} at {value}, but the network has "
                                f"{key} {network.pinned.get(key, 'as a box')}.")
    return problems


class ZoneBlock(_Physical):
    """Y the zone; each network input a `driven_by` column, a U or a D."""

    def __init__(self, block: Block, window: int, dt_h: float, norm_stats: dict):
        super().__init__(block, window, norm_stats)
        problems = []
        if len(block.y_names) != 1:
            problems.append(f"Y must be the zone temperature alone; got {list(block.y_names)}. "
                            "A heat behind a thermostat is its own `thermostat` block")
        if block.ambient is None:
            problems.append("declare `ambient`, the outdoor temperature it conducts to")
        if block.scale is None:
            problems.append("declare `scale`, what sets its units: the input or parameter "
                            "pinned, and its value, e.g. {'Heat [kW]': 1.0}")
        stray_reads = [c for c in block.read_names if c not in block.driven_by]
        if stray_reads or block.inputs:
            problems.append(f"`reads` {stray_reads} and `inputs` have no meaning here: its "
                            "network says where every column goes")
        if "network" not in block.params or set(block.params) != {"network"}:
            problems.append(f"params must be `network` (nt data ladder); got {sorted(block.params)}")
        if problems:
            raise ValueError(f"rc_network block {block.name!r}: " + "; ".join(problems))
        self.zone_name, self.ambient = block.y_names[0], block.ambient
        self.inputs = [c for c in block.driven_by + block.u_names + block.d_names
                       if c != block.ambient]
        self.zone = RCZone(block.params["network"], self.inputs, window, dt_h, block.name)
        problems = _refuse_scale_mismatch(block, self.zone.network)
        if problems:
            raise ValueError(f"rc_network block {block.name!r}: " + "; ".join(problems))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.ndim == 1
        if single:                 # under vmap: always cold, and leaves no state
            x, saved = x[None], self.zone.rollout_state()
            self.zone.set_rollout_state(None)
        zone = self.column(x, self.zone_name)
        next_zone = self.zone(zone, self.column(x, self.ambient),
                              *(self.column(x, c) for c in self.inputs))
        out = self.increment(self.zone_name, next_zone, zone[..., -1])[..., None]
        if single:
            self.zone.set_rollout_state(saved)
            return out[0]
        return out

    def describe(self) -> dict:
        return {"zone": self.zone_name, **self.zone.describe()}


class ThermostatBlock(_Physical):
    """Y the heat; `reads` the zone it closes on; U its setpoints, D its outdoor
    temperature if its form reads one."""

    def __init__(self, block: Block, norm_stats: dict):
        super().__init__(block, 1, norm_stats)
        problems = []
        if len(block.y_names) != 1:
            problems.append(f"Y must be the heat alone; got {list(block.y_names)}")
        if len(block.read_names) != 1 or block.driven_by:
            problems.append("`reads` must be the zone temperature it closes on, as of the "
                            f"current row; got {list(block.read_names)}")
        if set(block.params) != {"thermostat"}:
            problems.append("params must be `thermostat` (nt data thermostat); got "
                            f"{sorted(block.params)}")
        if problems:
            raise ValueError(f"thermostat block {block.name!r}: " + "; ".join(problems))
        self.heat_name, self.zone_name = block.y_names[0], block.read_names[0]
        self.hvac = thermostat(block.params["thermostat"], block.name)
        self.outdoor: Optional[str] = block.ambient if self.hvac.form.outdoor else None
        if sorted(self.hvac.setpoints) != sorted(block.u_names):
            problems.append(f"its setpoints {self.hvac.setpoints} must be the block's U "
                            f"{list(block.u_names)}")
        if self.hvac.form.outdoor and block.ambient is None:
            problems.append(f"form {self.hvac.form.name!r} reads an outdoor temperature: "
                            "declare `ambient`")
        stray = [c for c in block.d_names if c != self.outdoor]
        if stray:
            problems.append(f"D columns {stray} enter nothing: its form "
                            f"{self.hvac.form.name!r} reads no other input. Name the block's D")
        if problems:
            raise ValueError(f"thermostat block {block.name!r}: " + "; ".join(problems))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x if x.ndim == 2 else x[None]
        setpoints = torch.cat([self.column(batch, c) for c in self.hvac.setpoints], dim=-1)
        zone = self.column(batch, self.zone_name)[:, 0]
        outdoor = self.column(batch, self.outdoor)[:, 0] if self.outdoor else None
        heat = self.hvac(setpoints, zone, outdoor)
        out = self.increment(self.heat_name, heat, self.column(batch, self.heat_name)[:, 0])[:, None]
        return out if x.ndim == 2 else out[0]

    def describe(self) -> dict:
        return self.hvac.describe()
