"""
A thermal zone's physics as two separate parts, in physical units.

    Thermostat   setpoints, zone temperature, outdoor temperature -> heat (kW)
    RCZone       zone window, outdoor window, input windows -> next zone temperature

Neither knows the other exists, or where its numbers come from. Row j of every
input window is held over the step from j to j + 1, and the last row drives this
step: a caller holding a meter that reports a step late shifts it, and appends
the heat of the step being taken.

`RCZone` is the declared network behind its stationary Kalman filter (see
`network`): cold, the filter runs over the window to place the hidden nodes;
warm, the nodes carry over. In eval mode its matrices are constant buffers, so
the traced step only multiplies, which is what lets it export.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import is_fake

from neurotemplate.modeling import network as nw
from neurotemplate.modeling import thermostat as th
from neurotemplate.modeling.bounded import BoundedParameter, parse_box

#: What `RCZone.discrete` returns, held as buffers in eval mode.
FROZEN = ("a", "b", "gain", "power", "weights_y", "weights_d")


def _linear_box(name: str, value) -> tuple[float, float]:
    pair = value if isinstance(value, (list, tuple)) else [value, value]
    if len(pair) != 2 or pair[0] > pair[1]:
        raise ValueError(f"{name} must be a number or [low, high]; got {value!r}")
    return float(pair[0]), float(pair[1])


def thermostat_form(name: str) -> th.Form:
    parts = name.split("+")
    if parts[0] not in ("tracker", "latched") or not set(parts[1:]) <= {"offset", "outdoor",
                                                                         "stage2"}:
        raise ValueError(f"thermostat form {name!r} is not one of `nt data thermostat`'s")
    return th.Form(latched=parts[0] == "latched", **{p: True for p in parts[1:]})


class RCZone(nn.Module):
    """`inputs` names the network's input columns in the order `forward` takes
    them; they must be the declaration's, every one."""

    def __init__(self, declaration: dict, inputs: list[str], window: int, dt_h: float,
                 name: str = "zone"):
        super().__init__()
        self.network, boxes, noise = nw.parse_declaration(declaration, name)
        self.coarse_taus = declaration.get("time_constants_h")
        declared = [c for c, _ in self.network.inputs]
        if sorted(set(declared)) != sorted(inputs):
            raise ValueError(f"RCZone {name!r} is handed {sorted(inputs)}, but its network "
                             f"declares {sorted(set(declared))}")
        # A column may enter several nodes: the drive has one column per entry.
        self._order = [inputs.index(c) for c in declared]
        self.window, self.dt_h = window, dt_h
        self.physical_names = [k for k in self.network.parameter_names()
                               if not k.startswith("sigma")]
        self.physical = nn.ModuleList(
            BoundedParameter(k, *boxes[k], (name,)) for k in self.physical_names)
        self.register_buffer("noise", torch.tensor(
            [noise[f"sigma_{n}"] for n in self.network.nodes] + [noise["sigma_y"]],
            dtype=torch.float64), persistent=False)
        self._state: Optional[torch.Tensor] = None
        self._cache: Optional[tuple] = None
        for key in FROZEN:
            self.register_buffer(f"_frozen_{key}", None, persistent=False)
        self.register_load_state_dict_post_hook(lambda module, keys: module._refreeze())

    def values(self) -> dict[str, torch.Tensor]:
        out = {k: p.value()[0].double() for k, p in zip(self.physical_names, self.physical)}
        for i, node in enumerate(self.network.nodes):
            out[f"sigma_{node}"] = self.noise.double()[i]
        out["sigma_y"] = self.noise.double()[-1]
        return out

    def discrete(self) -> tuple[torch.Tensor, ...]:
        """`(A, B, K, F^L, W_y, W_d)`: the step, the filter gain, and the filter
        over the window as the linear map it is."""
        if self._frozen_a is not None:
            return tuple(getattr(self, f"_frozen_{k}") for k in FROZEN)
        a, b, q, r = nw.discrete_model(self.network, self.values(), self.dt_h)
        gain = nw.steady_gain(a, q, r)
        n, steps = a.shape[0], self.window - 1
        observe = torch.zeros(n, dtype=a.dtype)
        observe[0] = 1.0
        f = a @ (torch.eye(n, dtype=a.dtype) - torch.outer(gain, observe))
        powers = [torch.eye(n, dtype=a.dtype)]
        for _ in range(steps):
            powers.append(f @ powers[-1])
        back = torch.stack(powers[:steps][::-1]) if steps else torch.zeros(0, n, n, dtype=a.dtype)
        weights_y = back @ (a @ gain)
        weights_d = (back @ b).transpose(1, 2)
        return tuple(t.float() for t in (a, b, gain, powers[-1], weights_y, weights_d))

    def train(self, mode: bool = True):
        super().train(mode)
        self._refreeze()
        return self

    def _refreeze(self) -> None:
        # An exporter toggles the mode while it traces; keep what was frozen.
        if torch.compiler.is_compiling() or any(is_fake(p) for p in self.parameters()):
            return
        for key in FROZEN:
            setattr(self, f"_frozen_{key}", None)
        if self.training:
            return
        with torch.no_grad():
            frozen = self.discrete()
        for key, value in zip(FROZEN, frozen):
            setattr(self, f"_frozen_{key}", value)

    def rollout_state(self) -> Optional[torch.Tensor]:
        return self._state

    def set_rollout_state(self, state: Optional[torch.Tensor]) -> None:
        self._state = state
        if state is None:
            self._cache = None

    def _matrices(self) -> tuple[torch.Tensor, ...]:
        if self._frozen_a is not None:
            return self.discrete()
        if self._state is None or self._cache is None:
            self._cache = self.discrete()
        return self._cache

    def describe(self) -> dict:
        """The network's time constants now and from the ladder."""
        with torch.no_grad():
            values = {k: float(v) for k, v in self.values().items()}
            taus = nw.time_constants_h(self.network, nw.as_tensors(values)).tolist()
        out = {"network": "+".join(self.network.nodes),
               "network_time_constants_h": [round(t, 3) for t in taus]}
        if self.coarse_taus is not None:
            out["coarse_time_constants_h"] = self.coarse_taus
            out["left_coarse_bar"] = [
                round(t, 3) for t, (centre, se) in zip(taus, self.coarse_taus)
                if se is not None and abs(t - centre) > 3 * se]
        return out

    def forward(self, zone: torch.Tensor, ambient: torch.Tensor, *inputs: torch.Tensor
                ) -> torch.Tensor:
        """The next zone temperature [B] from windows [B, window]."""
        a, b, gain, power, weights_y, weights_d = self._matrices()
        drive = torch.stack([ambient, *(inputs[i] for i in self._order)], dim=-1)
        nodes = self._state
        if nodes is None:
            batch, n, steps = zone.shape[0], a.shape[0], zone.shape[1] - 1
            nodes = zone[:, :1].expand(-1, n) @ power.T
            if steps:
                nodes = (nodes + zone[:, :steps] @ weights_y
                         + drive[:, :steps].reshape(batch, -1) @ weights_d.reshape(-1, n))
        nodes = nodes + gain * (zone[:, -1:] - nodes[:, :1])
        nodes = nodes @ a.T + drive[:, -1] @ b.T
        self._state = nodes
        return nodes[:, 0]


class Thermostat(nn.Module):
    """Setpoints [B, sides], in the order of `self.sides`, to heat [B] in kW.

    Its declared `gain_limit` (kW/K) keeps the loop it closes on a zone from
    ringing: `C_zone / dt`, the one thing it needs to know about the zone.
    """

    def __init__(self, declaration: dict, name: str = "thermostat"):
        super().__init__()
        self.form = thermostat_form(declaration.get("form", ""))
        if self.form.latched:
            raise ValueError("Thermostat is stateless; use LatchedThermostat")
        self._setup(declaration, name)

    def _setup(self, declaration, name):
        if "gain_limit" not in declaration:
            raise ValueError(f"thermostat {name!r} declares no gain_limit (kW/K): `C_zone / dt`, "
                             "the gain past which the loop it closes on the zone rings. "
                             "`nt data thermostat` writes it.")
        self.sides = tuple(s for s in th.SIDES if s in declaration)
        self.setpoints = [declaration[s]["setpoint"] for s in self.sides]
        self.gain_limit = float(declaration["gain_limit"])
        self.reference = float(declaration.get("outdoor_reference", 0.0))
        self.names, parameters = [], []
        for side in self.sides:
            for key, positive in self.form.side_parameters():
                value = declaration[side].get(key)
                if value is None:
                    raise ValueError(
                        f"thermostat {name!r}: {side}.{key} is not declared (null where the "
                        "record did not place it): give it a box from what you know of the "
                        "equipment.")
                box = (parse_box(f"{side}.{key}", value) if positive
                       else _linear_box(f"{side}.{key}", value))
                self.names.append(f"{side}.{key}")
                parameters.append(BoundedParameter(f"{side}.{key}", *box, (name,), log=positive))
        self.parameter_boxes = nn.ModuleList(parameters)

    def values(self) -> dict[str, torch.Tensor]:
        p = {k: q.value()[0] for k, q in zip(self.names, self.parameter_boxes)}
        for side in self.sides:
            gain = p[f"{side}.gain"]
            p[f"{side}.gain"] = gain * (1 + (gain / self.gain_limit) ** 8) ** (-1 / 8)
        return p

    def _heat(self, setpoints, zone, outdoor, stages=None):
        if self.form.outdoor and outdoor is None:
            raise ValueError(f"thermostat form {self.form.name!r} reads the outdoor temperature")
        p = self.values()
        total, new_stages = torch.zeros_like(zone), []
        for i, side in enumerate(self.sides):
            error = th.SIDES[side] * (setpoints[:, i] - zone)
            carried = stages[:, i] if stages is not None else None
            q, stage = th.side_heat(p, side, self.form, error, outdoor, self.reference,
                                    smooth=False, stage=carried)
            total = total + q
            new_stages.append(stage)
        return total, new_stages

    def describe(self) -> dict:
        """The form, and how much of its gain limit each side uses."""
        with torch.no_grad():
            p = self.values()
        return {"thermostat": self.form.name,
                "gain_share": {side: round(float(p[f"{side}.gain"]) / self.gain_limit, 3)
                               for side in self.sides}}

    def forward(self, setpoints: torch.Tensor, zone: torch.Tensor,
                outdoor: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._heat(setpoints, zone, outdoor)[0]


class LatchedThermostat(Thermostat):
    """A thermostat whose stages hold between its switch points: one carried
    stage per side, as rollout state."""

    def __init__(self, declaration: dict, name: str = "thermostat"):
        nn.Module.__init__(self)
        self.form = thermostat_form(declaration.get("form", ""))
        if not self.form.latched:
            raise ValueError("LatchedThermostat needs a latched form")
        self._setup(declaration, name)
        self._stages: Optional[torch.Tensor] = None

    def rollout_state(self) -> Optional[torch.Tensor]:
        return self._stages

    def set_rollout_state(self, state: Optional[torch.Tensor]) -> None:
        self._stages = state

    def forward(self, setpoints, zone, outdoor=None):
        total, stages = self._heat(setpoints, zone, outdoor, self._stages)
        self._stages = torch.stack(stages, dim=-1)
        return total


def thermostat(declaration: dict, name: str = "thermostat") -> Thermostat:
    """The thermostat its declared form calls for."""
    latched = thermostat_form(declaration.get("form", "")).latched
    return (LatchedThermostat if latched else Thermostat)(declaration, name)
