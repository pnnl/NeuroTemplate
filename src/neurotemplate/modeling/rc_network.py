"""
`rc_network` — a thermal zone as a declared RC network behind a thermostat.

    setpoints ─[thermostat]→ heat delivered (kW) ─[RC network]→ zone temperature

The block's Y are the zone temperature and the heat delivered to it (the
block's `heat`); its U the setpoints its thermostat follows; its D the
outdoor temperature (`ambient`) and every other input the network takes.
`params` holds the two declarations, `network` (`nt data ladder`, see
`network`) and `thermostat` (`nt data thermostat`, see `thermostat`), with
every C, R, gain and thermostat parameter fitted inside its box. Everything
inside is in physical units: kWh/K, K/kW, kW, °C, hours.

One step, from the history window to the next row
=================================================

- **Cold** (the first call of a rollout): a Kalman filter runs over the
  window, the zone measured and the measured heat driving the network, and
  leaves every node's temperature at the current row. This is what places
  the hidden nodes: on two plant buildings a day of history did it as well
  as the whole record. **Warm**: the nodes carry over.
- The thermostat turns the current setpoints and the zone node into the heat
  delivered over the step, which is also the block's heat output: the meter
  reports the heat over the step at the row that ends it.
- The network steps once with that heat, the current outdoor temperature
  and the other inputs, `x ← A x + B u`, with A and B the exact
  zero-order hold at the data's step.

The filter's gain is the stationary one the declared noise implies; the
noise is a declaration, not a fitted parameter, here.

Stability: the network is stable for any positive R and C. The thermostat
closes a loop on the zone, stable and free of oscillation while its gain
stays under `C_zone / dt`; the block holds it there smoothly, `K_eff =
bound · (1 − exp(−K / bound))`, so a gain fitted from data that switches
faster than the step can never make the loop ring.

Export: the discretisation and the gain need a matrix exponential and
linear solves, which do not export. In eval mode the block holds them as
constant buffers, recomputed on `eval()` and after a state-dict load, so
the traced step only multiplies.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import is_fake

from neurotemplate.modeling import network as nw
from neurotemplate.modeling import thermostat as th
from neurotemplate.modeling.bounded import BoundedParameter, parse_box

#: What `discrete` returns, held as buffers in eval mode.
FROZEN = ("a", "b", "gain", "power", "weights_y", "weights_d")


def _form(name: str) -> th.Form:
    parts = name.split("+")
    if parts[0] not in ("tracker", "latched") or not set(parts[1:]) <= {"offset", "outdoor",
                                                                         "stage2"}:
        raise ValueError(f"thermostat form {name!r} is not one of `nt data thermostat`'s")
    return th.Form(latched=parts[0] == "latched", **{p: True for p in parts[1:]})


class NetworkBlock(nn.Module):
    """One thermal zone: its network, its thermostat and its filter.

    Reads a key-major window of `window` rows of its Y (zone and heat) and D,
    and the current row of U, all normalised; returns the increment of both
    Y channels, normalised (`residual=True` adds the current row back).
    """

    def __init__(self, name: str, y_names: tuple[str, ...], u_names: tuple[str, ...],
                 d_names: tuple[str, ...], heat: str, ambient: str, params: dict,
                 window: int, dt_h: float, norm_stats: dict[str, dict]):
        super().__init__()
        if set(params) != {"network", "thermostat"}:
            raise ValueError(f"rc_network block {name!r}: params must be `network` "
                             f"(nt data ladder) and `thermostat` (nt data thermostat); got "
                             f"{sorted(params)}")
        self.name = name
        self.network, boxes, noise = nw.parse_declaration(params["network"], name)
        self.coarse_taus = params["network"].get("time_constants_h")
        self.y_names, self.u_names, self.d_names = y_names, u_names, d_names
        self.window, self.dt_h = window, dt_h
        self.zone = next(c for c in y_names if c != heat)
        self.heat_name = heat

        # The drive's columns: ambient, then the network's inputs, each read
        # from the D window, except the heat, which is the Y channel.
        inputs = [c for c, _ in self.network.inputs]
        problems = []
        if heat not in inputs:
            problems.append(f"the network takes no {heat!r}; the heat enters the zone")
        stray = [c for c in inputs if c != heat and c not in d_names]
        if stray:
            problems.append(f"network inputs {stray} are not D columns of the block")
        unused = [c for c in d_names if c != ambient and c not in inputs]
        if unused:
            problems.append(f"D columns {unused} enter no node of the network")
        self._ambient = d_names.index(ambient)
        self._drive = [(None if c == heat else d_names.index(c)) for c in inputs]

        # The thermostat: each side's setpoint must be a U column, and every U
        # column some side's setpoint.
        decl = params["thermostat"]
        self.form = _form(decl.get("form", ""))
        self.sides = tuple(s for s in th.SIDES if s in decl)
        setpoints = [decl[s]["setpoint"] for s in self.sides]
        if sorted(setpoints) != sorted(u_names):
            problems.append(f"the thermostat's setpoints {setpoints} must be the block's U "
                            f"{list(u_names)}")
        if problems:
            raise ValueError(f"rc_network block {name!r}:\n  - " + "\n  - ".join(problems))
        self._setpoints = [u_names.index(decl[s]["setpoint"]) for s in self.sides]
        self.reference = float(decl.get("outdoor_reference", 0.0))

        self.physical_names = list(self.network.parameter_names())
        self.physical_names = [k for k in self.physical_names if not k.startswith("sigma")]
        self.physical = nn.ModuleList(
            BoundedParameter(k, *boxes[k], (name,)) for k in self.physical_names)
        self.register_buffer("noise", torch.tensor(
            [noise[f"sigma_{n}"] for n in self.network.nodes] + [noise["sigma_y"]],
            dtype=torch.float64), persistent=False)

        self.thermostat_names, thermostat = [], []
        for side in self.sides:
            for key, positive in self.form.side_parameters():
                value = decl[side].get(key)
                if value is None:
                    raise ValueError(
                        f"rc_network block {name!r}: thermostat {side}.{key} is not "
                        "declared (null where the record did not place it): give it a "
                        "box from what you know of the equipment.")
                box = (parse_box(f"{side}.{key}", value) if positive
                       else _linear_box(f"{side}.{key}", value))
                self.thermostat_names.append(f"{side}.{key}")
                thermostat.append(BoundedParameter(f"{side}.{key}", *box, (name,),
                                                   log=positive))
        self.thermostat = nn.ModuleList(thermostat)

        for key, columns in (("y", y_names), ("u", u_names), ("d", d_names)):
            for field in ("shift", "stretch"):
                self.register_buffer(f"_{key}_{field}", torch.tensor(
                    [float(norm_stats[c][field]) for c in columns]), persistent=False)
        self._zone_col, self._heat_col = y_names.index(self.zone), y_names.index(heat)
        self.n_latched = len(self.sides) if self.form.latched else 0

        self._state: Optional[torch.Tensor] = None
        self._cache: Optional[tuple] = None
        for key in FROZEN:
            self.register_buffer(f"_frozen_{key}", None, persistent=False)
        self.register_load_state_dict_post_hook(lambda module, keys: module._refreeze())

    # ── parameters ──────────────────────────────────────────────

    def values(self) -> dict[str, torch.Tensor]:
        """The network's parameters by name, float64 scalars, noise included."""
        out = {k: p.value()[0].double() for k, p in zip(self.physical_names, self.physical)}
        noise = self.noise.double()
        for i, node in enumerate(self.network.nodes):
            out[f"sigma_{node}"] = noise[i]
        out["sigma_y"] = noise[-1]
        return out

    def thermostat_values(self) -> dict[str, torch.Tensor]:
        """The thermostat's parameters by name, each gain held under its bound."""
        p = {k: q.value()[0] for k, q in zip(self.thermostat_names, self.thermostat)}
        bound = self.gain_bound()
        for side in self.sides:
            gain = p[f"{side}.gain"]
            p[f"{side}.gain"] = bound * (1 - torch.exp(-gain / bound))
        return p

    def gain_bound(self) -> torch.Tensor:
        """`C_zone / dt`, kW/K: the gain past which the discrete loop rings."""
        zone = self.network.zone
        index = self.physical_names.index(f"C_{zone}")
        return self.physical[index].value()[0] / self.dt_h

    def discrete(self) -> tuple[torch.Tensor, ...]:
        """`(A, B, K, F^L, W_y, W_d)` at the data's step, float32: the
        network's step, the stationary filter gain, and the filter over the
        window as the linear map it is (see `_filter`). Frozen in eval mode,
        else computed."""
        if self._frozen_a is not None:
            return tuple(getattr(self, f"_frozen_{k}") for k in FROZEN)
        a, b, q, r = nw.discrete_model(self.network, self.values(), self.dt_h)
        gain = nw.steady_gain(a, q, r)
        # x ← F x + A K y_j + B u_j over the window's steps, F = A (I − K e₀ᵀ):
        # the state after L steps is F^L x₀ + Σ_j F^(L−1−j) (A K y_j + B u_j).
        n, steps = a.shape[0], self.window - 1
        observe = torch.zeros(n, dtype=a.dtype)
        observe[0] = 1.0
        f = a @ (torch.eye(n, dtype=a.dtype) - torch.outer(gain, observe))
        powers = [torch.eye(n, dtype=a.dtype)]
        for _ in range(steps):
            powers.append(f @ powers[-1])
        back = torch.stack(powers[:steps][::-1]) if steps else torch.zeros(0, n, n,
                                                                           dtype=a.dtype)
        weights_y = back @ (a @ gain)                        # [L, n]
        weights_d = (back @ b).transpose(1, 2)               # [L, 1 + m, n]
        return tuple(t.float() for t in (a, b, gain, powers[-1], weights_y, weights_d))

    def train(self, mode: bool = True):
        super().train(mode)
        self._refreeze()
        return self

    def _refreeze(self) -> None:
        # An exporter toggles the mode while it traces; the constants frozen
        # before it started are the ones to keep.
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

    # ── the rollout-state protocol (see `dynamics.stateful_blocks`) ──

    def rollout_state(self) -> Optional[torch.Tensor]:
        """Node temperatures, then any latched stages: [batch, n + n_latched]."""
        return self._state

    def set_rollout_state(self, state: Optional[torch.Tensor]) -> None:
        self._state = state
        if state is None:
            self._cache = None

    # ── forward ─────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return self.step(x[None], None, self.discrete())[0][0]
        if self._frozen_a is not None:
            matrices = self.discrete()
        else:
            # Training: the parameters hold still through a rollout, so its
            # cold call computes the matrices once for every step after it.
            if self._state is None or self._cache is None:
                self._cache = self.discrete()
            matrices = self._cache
        out, self._state = self.step(x, self._state, matrices)
        return out

    def step(self, x: torch.Tensor, state: Optional[torch.Tensor], matrices: tuple
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(increment, state)` after one step; `state` None is a cold call."""
        a, b, gain = matrices[:3]
        rows, ny, nu, nd = self.window, len(self.y_names), len(self.u_names), len(self.d_names)
        y = x[..., :rows * ny].reshape(-1, rows, ny) * self._y_stretch + self._y_shift
        u = x[..., rows * ny:rows * ny + nu] * self._u_stretch + self._u_shift
        d = x[..., rows * ny + nu:].reshape(-1, rows, nd) * self._d_stretch + self._d_shift
        zone, heat = y[..., self._zone_col], y[..., self._heat_col]
        n = len(self.network.nodes)
        if state is None:
            nodes, stages = self._filter(zone, heat, d, matrices[3:]), None
        else:
            nodes, stages = state[:, :n], state[:, n:] if self.n_latched else None
        # The measurement at the current row: a no-op on a warm call, whose
        # zone row is this block's own last prediction.
        nodes = nodes + gain * (zone[:, -1:] - nodes[:, :1])

        p = self.thermostat_values()
        delivered = torch.zeros_like(nodes[:, 0])
        new_stages = []
        outdoor = d[:, -1, self._ambient]
        for i, side in enumerate(self.sides):
            error = th.SIDES[side] * (u[:, self._setpoints[i]] - nodes[:, 0])
            carried = stages[:, i] if stages is not None else None
            q, stage = th.side_heat(p, side, self.form, error, outdoor, self.reference,
                                    smooth=False, stage=carried)
            delivered = delivered + q
            if self.form.latched:
                new_stages.append(stage)
        drive = self._drive_row(d[:, -1], delivered)
        nodes = nodes @ a.T + drive @ b.T

        change = {self._zone_col: nodes[:, 0] - zone[:, -1],
                  self._heat_col: delivered - heat[:, -1]}
        increment = torch.stack([change[i] for i in range(2)], dim=-1) / self._y_stretch
        state = torch.cat([nodes, torch.stack(new_stages, dim=-1)], dim=-1) \
            if new_stages else nodes
        return increment, state

    def _drive_row(self, d_row: torch.Tensor, heat: torch.Tensor) -> torch.Tensor:
        """The network's drive, [..., 1 + m], from D rows [..., nd] and the heat
        over each step [...]."""
        columns = [d_row[..., self._ambient]]
        columns += [heat if pos is None else d_row[..., pos] for pos in self._drive]
        return torch.stack(columns, dim=-1)

    def _filter(self, zone, heat, d, weights) -> torch.Tensor:
        """The nodes predicted for the current row, before its measurement:
        the stationary filter over the window from every node at its first
        zone reading, the step from row j driven by row j's inputs and row
        j+1's heat. Linear in the window, so it is the fixed map `weights`
        (`discrete`) rather than a loop or a scan, which is also what lets it
        export as plain matrix products."""
        power, weights_y, weights_d = weights
        steps, n = zone.shape[1] - 1, power.shape[0]
        nodes = zone[:, :1].expand(-1, n) @ power.T
        if steps == 0:
            return nodes
        drive = self._drive_row(d[:, :steps], heat[:, 1:]).reshape(zone.shape[0], -1)
        return nodes + zone[:, :steps] @ weights_y + drive @ weights_d.reshape(-1, n)

    # ── description ─────────────────────────────────────────────

    def describe(self) -> dict:
        """The network's time constants now and from the ladder, and how much
        of its stability bound each thermostat gain uses."""
        with torch.no_grad():
            values = {k: float(v) for k, v in self.values().items()}
            taus = nw.time_constants_h(self.network, nw.as_tensors(values)).tolist()
            p = self.thermostat_values()
            bound = float(self.gain_bound())
        out = {
            "network": "+".join(self.network.nodes),
            "network_time_constants_h": [round(t, 3) for t in taus],
            "thermostat": self.form.name,
            "gain_share": {side: round(float(p[f"{side}.gain"]) / bound, 3)
                           for side in self.sides},
        }
        if self.coarse_taus is not None:
            out["coarse_time_constants_h"] = self.coarse_taus
            out["left_coarse_bar"] = [
                round(t, 3) for t, (centre, se) in zip(taus, self.coarse_taus)
                if se is not None and abs(t - centre) > 3 * se]
        return out


def _linear_box(name: str, value) -> tuple[float, float]:
    pair = value if isinstance(value, (list, tuple)) else [value, value]
    if len(pair) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                 for v in pair) or pair[0] > pair[1]:
        raise ValueError(f"{name} must be a number or [low, high]; got {value!r}")
    return float(pair[0]), float(pair[1])
