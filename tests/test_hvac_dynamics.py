"""
Smoke test for the gray-box HVAC models (`hvac_envelope` / `hvac_rtu`).

Importing `hvac_adapter` registers the two factories in `model_catalog`; from
there they go through the same `get_dynamics` path as the neural models. The test
runs in physical units (`norm_type=None`) and checks: registration, an
autoregressive rollout with finite output of the right shape, and that gradients
reach the Envelope's learnable physical parameters.
"""
import os

import pytest
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import get_dynamics, model_catalog
from neurotemplate.modeling.hvac_adapter import RTU_OBSERVABLE_CHANNELS
import neurotemplate.modeling.hvac_adapter  # noqa: F401  (registers hvac_* factories)

# Column names must match the fixture's building_spec channel_map.
ZONE = "Air Temperature [°C]"
SETPT = "Cooling Setpoint [°C]"
QHVAC = "HVAC Heat Delivered [W]"
OUT = "Outdoor Dry-Bulb Temperature [°C]"
IRR = "Global Horizontal Irradiance [W/m²]"
OCC = "Occupancy Signal [-]"

# hvac_rtu emits 6 observable channels; column names for Y[1:] are placeholders
# (the adapter computes them from physics and never reads them as inputs).
RTU_Y_COLS = [ZONE, "Supply Air Temperature [°C]", "Supply Airflow [kg/s]",
              "Fan Power [W]", "DX Power [W]", "Cooling Stage [-]"]

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "modeling_user_defined.py")


def make_envelope_nm_mapping() -> dict[str, list[str]]:
    return {
        "Y": [ZONE],
        "U": [SETPT, QHVAC],
        "D": [OUT, IRR, OCC],
        "t": ["sec_elapsed"],
    }


def make_rtu_nm_mapping() -> dict[str, list[str]]:
    return {
        "Y": RTU_Y_COLS,
        "U": [SETPT],
        "D": [OUT, IRR, OCC],
        "t": ["sec_elapsed"],
    }


def make_cfg(model: str) -> Config:
    cfg = Config()
    cfg.model = model
    cfg.in_steps = 1          # single-step physics
    cfg.norm_type = None      # physical units
    cfg.device = "cpu"
    cfg.user_hooks_path = FIXTURE
    return cfg


def make_inputs(batch: int = 4, horizon: int = 6, ny: int = 1):
    """Physical-unit windowed inputs for an autoregressive rollout."""
    torch.manual_seed(0)
    yn0 = torch.zeros(batch, 1, ny)
    yn0[:, :, 0] = 24.0 + torch.randn(batch, 1)  # zone temp seed [°C]; other channels zero
    setpt = 24.0 + 0.5 * torch.randn(batch, horizon, 1)        # [°C]
    out_t = 25.0 + 3.0 * torch.randn(batch, horizon, 1)        # [°C]
    irr = torch.clamp(200.0 + 290.0 * torch.randn(batch, horizon, 1), min=0.0)
    occ = (torch.rand(batch, horizon, 1) > 0.5).float()
    D = torch.cat([out_t, irr, occ], dim=-1)                   # [batch, horizon, 3]
    Y = torch.zeros(batch, horizon, ny)                         # only sets rollout length
    return {"yn": yn0, "D": D, "Y": Y}, horizon


def make_envelope_inputs(batch: int = 4, horizon: int = 6):
    data, horizon = make_inputs(batch, horizon, ny=1)
    qhvac = -2000.0 + 200.0 * torch.randn(batch, horizon, 1)
    setpt = 24.0 + 0.5 * torch.randn(batch, horizon, 1)
    data["U"] = torch.cat([setpt, qhvac], dim=-1)
    return data, horizon


def make_rtu_inputs(batch: int = 4, horizon: int = 6):
    data, horizon = make_inputs(batch, horizon, ny=len(RTU_OBSERVABLE_CHANNELS))
    setpt = 24.0 + 0.5 * torch.randn(batch, horizon, 1)
    data["U"] = setpt
    return data, horizon


def test_registration():
    assert "hvac_envelope" in model_catalog
    assert "hvac_rtu" in model_catalog


def test_envelope_rollout_shape_and_finite():
    cfg = make_cfg("hvac_envelope")
    system = get_dynamics(cfg, make_envelope_nm_mapping())
    data, horizon = make_envelope_inputs()

    out = system(data)

    yn = out["yn"]
    assert yn.shape == (4, horizon + 1, 1)
    assert torch.isfinite(yn).all()


def test_rtu_rollout_shape_and_finite():
    cfg = make_cfg("hvac_rtu")
    system = get_dynamics(cfg, make_rtu_nm_mapping())
    data, horizon = make_rtu_inputs()

    out = system(data)

    yn = out["yn"]
    assert yn.shape == (4, horizon + 1, len(RTU_OBSERVABLE_CHANNELS))
    assert torch.isfinite(yn).all()


@pytest.mark.parametrize("model,nm_fn,inputs_fn", [
    ("hvac_envelope", make_envelope_nm_mapping, make_envelope_inputs),
    ("hvac_rtu", make_rtu_nm_mapping, make_rtu_inputs),
])
def test_gradients_reach_physical_params(model, nm_fn, inputs_fn):
    cfg = make_cfg(model)
    system = get_dynamics(cfg, nm_fn())
    data, _ = inputs_fn()

    out = system(data)
    out["yn"].sum().backward()

    learned = {
        name: p for name, p in system.named_parameters()
        if ("solar_gain" in name or "internal_gain" in name)
    }
    assert learned, "Envelope learnable params not found in the model"
    for name, p in learned.items():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all()
