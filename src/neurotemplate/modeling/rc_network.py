"""
`thermal`'s parts as block cores: `rc_network` and `thermostat`.

    setpoints ─[thermostat]→ heat ─[rc_network]→ zone temperature
                 heat from a U ────[rc_network]→ zone temperature

Two blocks that know nothing of each other. An `rc_network` block's Y is the
zone temperature and nothing else, and its network's inputs are whatever
columns the declaration names: a `driven_by` column (the heat a thermostat
block produced this step), a U (a commanded heat) or a D (solar, a schedule).
Its window is the run's `context`, the history its filter places the
hidden nodes from. `ambient` is the outdoor temperature the network conducts
to, and at least one of its parameters is pinned: `scale` says which.

A `thermostat` block's Y is the heat, its U the setpoints it follows, and it
`reads` the zone as of the current row.

Everything here is unit conversion and naming between a block's normalised
windows and the physical parts in `thermal`. Each returns the value itself, so
takes no residual.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from neurotemplate.modeling.blocks import Block
from neurotemplate.modeling.thermal import RCZone, thermostat


class _Physical(nn.Module):
    """Reads a block's columns by name, in physical units."""

    def __init__(self, block: Block, norm_stats: dict):
        super().__init__()
        self._where = {c: (key, j)
                       for key, names in enumerate((block.y_in_names, block.u_names, block.d_names))
                       for j, c in enumerate(names)}
        names = block.y_in_names + block.u_names + block.d_names
        self._index = {c: i for i, c in enumerate(names)}
        for field in ("shift", "stretch"):
            self.register_buffer(f"_{field}", torch.tensor(
                [float(norm_stats[c][field]) for c in names]), persistent=False)

    def column(self, windows: tuple, name: str) -> torch.Tensor:
        """Column `name` in physical units, [batch, rows]."""
        key, j = self._where[name]
        i = self._index[name]
        return windows[key][:, :, j] * self._stretch[i] + self._shift[i]

    def normalised(self, name: str, value: torch.Tensor) -> torch.Tensor:
        i = self._index[name]
        return ((value - self._shift[i]) / self._stretch[i])[:, None]


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
        super().__init__(block, norm_stats)
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
        if "rc_network" not in block.params:
            problems.append("params has no `rc_network` entry: its declaration, from `nt data "
                            "ladder`")
        if problems:
            raise ValueError(f"rc_network block {block.name!r}: " + "; ".join(problems))
        self.zone_name, self.ambient = block.y_names[0], block.ambient
        self.inputs = [c for c in block.driven_by + block.u_names + block.d_names
                       if c != block.ambient]
        self.zone = RCZone(block.params["rc_network"], self.inputs, window, dt_h, block.name)
        problems = _refuse_scale_mismatch(block, self.zone.network)
        if len(self.zone.network.nodes) > 1 and window < 2:
            problems.append(
                "it has hidden nodes, and `context` is what places them: the history "
                "history its filter runs over. There is no default; it is long enough when a "
                "longer one stops changing the day-ahead error")
        if problems:
            raise ValueError(f"rc_network block {block.name!r}: " + "; ".join(problems))

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        windows = (y, u, d)
        next_zone = self.zone(self.column(windows, self.zone_name),
                              self.column(windows, self.ambient),
                              *(self.column(windows, c) for c in self.inputs))
        return self.normalised(self.zone_name, next_zone)

    def describe(self) -> dict:
        return {"zone": self.zone_name, **self.zone.describe()}


class ThermostatBlock(_Physical):
    """Y the heat; `reads` the zone it closes on; U its setpoints; `ambient` the
    outdoor temperature, which only some forms read."""

    def __init__(self, block: Block, norm_stats: dict):
        super().__init__(block, norm_stats)
        problems = []
        if len(block.y_names) != 1:
            problems.append(f"Y must be the heat alone; got {list(block.y_names)}")
        if len(block.read_names) != 1 or block.driven_by:
            problems.append("`reads` must be the zone temperature it closes on, as of the "
                            f"current row; got {list(block.read_names)}")
        if "thermostat" not in block.params:
            problems.append("params has no `thermostat` entry: its declaration, from `nt data "
                            "thermostat`")
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
        stray = [c for c in block.d_names if c != block.ambient]
        if stray:
            problems.append(f"D columns {stray} enter nothing: a thermostat reads at most its "
                            "`ambient`. Name the block's D")
        if problems:
            raise ValueError(f"thermostat block {block.name!r}: " + "; ".join(problems))

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        windows = (y, u, d)
        setpoints = torch.stack([self.column(windows, c)[:, -1] for c in self.hvac.setpoints],
                                dim=-1)
        outdoor = self.column(windows, self.outdoor)[:, -1] if self.outdoor else None
        heat = self.hvac(setpoints, self.column(windows, self.zone_name)[:, -1], outdoor)
        return self.normalised(self.heat_name, heat)

    def describe(self) -> dict:
        return self.hvac.describe()
