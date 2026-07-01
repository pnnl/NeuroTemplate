"""
Gray-box HVAC dynamics for the modeling catalog.

Registers two physics models from neuromancer-lite alongside the neural models
(`mlp`/`nssm`/...), so the framework can fit their *physical* parameters and
benchmark gray-box vs. black-box on the same data with a one-word config toggle:

    model: nssm          # black box
    model: hvac_envelope # gray box: RC envelope, Q_hvac supplied as a column
    model: hvac_rtu      # gray box: StagingController -> StagedDXPlant -> Envelope

Both wrap a single physical step as the inseparable `SSMGrad.fy(cat(y,u,d))`
callable, so the framework's SystemPreview owns the autoregressive rollout (no
nested neuromancer-lite rollout). See `hvac_integration.md`.

Importing this module registers the two factories in `model_catalog`. The import
is opt-in (the library does not depend on neuromancer-lite); the idiomatic place
to trigger it is the top of `modeling_user_defined.py`, which also supplies the
dataset-specific `building_spec` hook this module consumes.

`building_spec(cfg, nm_mapping) -> dict` contract (lives in user hooks):
    dt           float, integration step [s]
    n_zones      int, building zones (single-zone only, == len(nm_mapping['Y']))
    to_kelvin    list[str], component variables given in degC that need +273.15
    channel_map  dict, component variable -> column name present in nm_mapping
    envelope     dict, kwargs for Envelope (incl. "learnable": list[str])
    plant        dict, kwargs for StagedDXPlant (hvac_rtu only)
    controller   dict, kwargs for StagingController (hvac_rtu only)
"""
import torch
import torch.nn as nn

from neuromancer.hvac.building_components import Envelope, StagedDXPlant, StagingController
from neuromancer.hvac.building import CompositeDynamics

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.dynamics import SSMGrad, model_catalog
from neurotemplate.hooks import load_modeling_hooks

from neurotemplate._typing import beartype

_KELVIN_OFFSET = 273.15

# Ordered observable channels for hvac_rtu. Zone temp is first because it is
# the only autoregressive state; the rest are algebraic outputs fed back but
# ignored by the physics on the next step.
RTU_OBSERVABLE_CHANNELS = ["T_zones", "T_supply", "supply_airflow", "fan_power", "dx_power", "stage"]
_RTU_KELVIN_OUTPUTS = {"T_zones", "T_supply"}


class HVACStepAdapter(nn.Module):
    """
    Bridge a single physical step to the `SSMGrad.fy(cat(y,u,d))` contract.

    The framework hands `fy` a flattened `[batch, ny+nu+nd]` row (in_steps=1, so
    one timestep per channel, ordered Y then U then D). This adapter slices out
    the physical variables the component needs by name, converts the temperature
    channels degC -> K, and delegates to `step_fn` which returns the final
    `[batch, ny]` tensor in framework units (degC for temperatures).
    """

    def __init__(self, component, var_to_index, kelvin_vars, dt, step_fn):
        super().__init__()
        self.component = component               # registers physical params
        self.var_to_index = dict(var_to_index)   # component var -> col index in cat
        self.kelvin_vars = set(kelvin_vars)
        self.dt = float(dt)
        self.step_fn = step_fn                   # (component, vars, dt) -> [batch, ny] in framework units

    def forward(self, cat_yud: torch.Tensor) -> torch.Tensor:
        vals = {}
        for var, idx in self.var_to_index.items():
            v = cat_yud[:, idx:idx + 1]
            if var in self.kelvin_vars:
                v = v + _KELVIN_OFFSET
            vals[var] = v
        return self.step_fn(self.component, vals, self.dt)


def _column_index(nm_mapping: dict[str, list[str]]) -> dict[str, int]:
    """Map each column to its index in the flattened [Y | U | D] feature row."""
    ordered = nm_mapping["Y"] + nm_mapping["U"] + nm_mapping["D"]
    return {col: i for i, col in enumerate(ordered)}


def _resolve_indices(spec: dict, nm_mapping: dict, needed: list[str]) -> dict[str, int]:
    """Look up the cat-row index of every component variable the model consumes."""
    channel_map = spec["channel_map"]
    col_index = _column_index(nm_mapping)
    var_to_index = {}
    for var in needed:
        if var not in channel_map:
            raise ValueError(
                f"building_spec channel_map is missing required variable '{var}'. "
                f"Needed: {needed}"
            )
        col = channel_map[var]
        if col not in col_index:
            raise ValueError(
                f"building_spec maps '{var}' -> column '{col}', which is not in "
                f"nm_mapping (Y/U/D). Available columns: {list(col_index)}"
            )
        var_to_index[var] = col_index[col]
    return var_to_index


def _check_envelope(cfg, spec, nm_mapping):
    ny = len(nm_mapping["Y"])
    if cfg.in_steps != 1:
        raise ValueError(
            f"hvac models are single-step physics; set in_steps=1 (got {cfg.in_steps})."
        )
    if spec.get("n_zones", 1) != 1 or ny != 1:
        raise ValueError(
            "hvac_envelope is single-zone with one Y channel: building_spec['n_zones'] and "
            f"len(nm_mapping['Y']) must both be 1 (got n_zones={spec.get('n_zones', 1)}, "
            f"ny={ny})."
        )
    if spec["channel_map"]["T_zones"] != nm_mapping["Y"][0]:
        raise ValueError(
            f"building_spec maps T_zones -> '{spec['channel_map']['T_zones']}', but the "
            f"only Y column is '{nm_mapping['Y'][0]}'. The predicted state must be Y."
        )


def _check_rtu(cfg, spec, nm_mapping):
    ny = len(nm_mapping["Y"])
    n_obs = len(RTU_OBSERVABLE_CHANNELS)
    if cfg.in_steps != 1:
        raise ValueError(
            f"hvac models are single-step physics; set in_steps=1 (got {cfg.in_steps})."
        )
    if spec.get("n_zones", 1) != 1:
        raise ValueError(
            f"hvac_rtu is single-zone; building_spec['n_zones'] must be 1 "
            f"(got {spec.get('n_zones', 1)})."
        )
    if ny != n_obs:
        raise ValueError(
            f"hvac_rtu requires exactly {n_obs} Y channels, one per observable "
            f"({RTU_OBSERVABLE_CHANNELS}); got ny={ny}."
        )
    if spec["channel_map"]["T_zones"] != nm_mapping["Y"][0]:
        raise ValueError(
            f"hvac_rtu: T_zones must be the first Y column (it is the autoregressive state). "
            f"building_spec maps T_zones -> '{spec['channel_map']['T_zones']}' but "
            f"nm_mapping['Y'][0] = '{nm_mapping['Y'][0]}'."
        )


def _device(cfg) -> torch.device:
    return torch.device(cfg.device) if cfg.device else torch.device("cpu")


def _ssm(adapter, nm_mapping) -> SSMGrad:
    return SSMGrad(
        fy=adapter, fu=None, fd=None,
        ny=len(nm_mapping["Y"]), nu=len(nm_mapping["U"]), nd=len(nm_mapping["D"]),
        in_steps=1, residual=False, separable=False,
    )


def _env_step(component, vals, dt):
    out = component(
        t=0.0, dt=dt,
        T_zones=vals["T_zones"], T_outdoor=vals["T_outdoor"],
        irradiance=vals["irradiance"], occupancy=vals["occupancy"],
        Q_hvac=vals["Q_hvac"],
    )
    return out["T_zones"] - _KELVIN_OFFSET  # K -> degC


def _rtu_step(component, vals, dt):
    state = {"T_zones": vals["T_zones"]}
    controls = {"T_setpoint": vals["T_setpoint"]}
    disturbances = {
        "T_outdoor": vals["T_outdoor"],
        "irradiance": vals["irradiance"],
        "occupancy": vals["occupancy"],
    }
    _, aux = component.step(state, controls, disturbances, t=0.0, dt=dt)
    # aux contains all component outputs including T_zones (next state in K)
    return torch.cat([
        aux[ch] - _KELVIN_OFFSET if ch in _RTU_KELVIN_OUTPUTS else aux[ch]
        for ch in RTU_OBSERVABLE_CHANNELS
    ], dim=-1)  # [batch, 6] in framework units


def _build_envelope(spec, device) -> Envelope:
    kwargs = dict(spec.get("envelope", {}))
    learnable = set(kwargs.pop("learnable", []))
    return Envelope(n_zones=1, learnable=learnable, device=device, **kwargs)


@beartype
def get_hvac_envelope(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    """RC envelope as a state transition; Q_hvac supplied as a measured column."""
    spec = load_modeling_hooks(cfg).building_spec(cfg, nm_mapping)
    _check_envelope(cfg, spec, nm_mapping)
    device = _device(cfg)

    envelope = _build_envelope(spec, device)
    needed = ["T_zones", "T_outdoor", "irradiance", "occupancy", "Q_hvac"]
    var_to_index = _resolve_indices(spec, nm_mapping, needed)
    adapter = HVACStepAdapter(
        envelope, var_to_index, spec.get("to_kelvin", []), spec["dt"], _env_step,
    )
    return _ssm(adapter, nm_mapping)


@beartype
def get_hvac_rtu(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    """Controller -> StagedDXPlant -> Envelope; emits all 6 observable channels."""
    spec = load_modeling_hooks(cfg).building_spec(cfg, nm_mapping)
    _check_rtu(cfg, spec, nm_mapping)
    device = _device(cfg)

    ctrl_kwargs = dict(spec.get("controller", {}))
    plant_kwargs = dict(spec.get("plant", {}))
    controller = StagingController(
        learnable=set(ctrl_kwargs.pop("learnable", [])), device=device, **ctrl_kwargs,
    )
    plant = StagedDXPlant(
        learnable=set(plant_kwargs.pop("learnable", [])), device=device, **plant_kwargs,
    )
    envelope = _build_envelope(spec, device)
    composite = CompositeDynamics(controller, plant, envelope)

    needed = ["T_zones", "T_setpoint", "T_outdoor", "irradiance", "occupancy"]
    var_to_index = _resolve_indices(spec, nm_mapping, needed)
    adapter = HVACStepAdapter(
        composite, var_to_index, spec.get("to_kelvin", []), spec["dt"], _rtu_step,
    )
    return _ssm(adapter, nm_mapping)


model_catalog["hvac_envelope"] = get_hvac_envelope
model_catalog["hvac_rtu"] = get_hvac_rtu
