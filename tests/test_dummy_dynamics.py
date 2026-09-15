"""
Tests for the content-free `dummy` surrogate (zone temp = cooling setpoint).

The dummy goes through the same `get_dynamics` path as every other model
and must reproduce the setpoint exactly as the next state (passthrough).
"""
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import get_dummy, get_dynamics

ZONE = "Zone Air Temperature [°C]"
SETPT = "Zone Cooling Setpoint [°C]"
OUT = "Outdoor Dry-Bulb Temperature [°C]"
IRR = "Global Horizontal Irradiance [W/m²]"
OCC = "Occupancy Signal [-]"


def single_zone_mapping() -> dict[str, list[str]]:
    return {"Y": [ZONE], "U": [SETPT], "D": [OUT, IRR, OCC], "t": ["sec_elapsed"]}


def make_cfg(in_steps: int = 1) -> Config:
    cfg = Config()
    cfg.model = "dummy"
    cfg.in_steps = in_steps   # single-step passthrough
    cfg.norm_type = None      # physical units — setpoint and zone temp share °C
    cfg.device = "cpu"
    return cfg


def test_forward_returns_setpoint_exactly():
    """SSMGrad.forward(y, u, d) must return the setpoint u with the bias at 0."""
    mapping = single_zone_mapping()
    model = get_dummy(make_cfg(), mapping)
    y = 24.0 + torch.randn(5, 1)      # zone temp [batch, ny]
    u = 22.0 + torch.randn(5, 1)      # setpoint  [batch, nu]
    d = torch.randn(5, 3)             # disturbances [batch, nd]

    out = model(y, u, d)

    assert out.shape == (5, 1)
    assert torch.equal(out, u)        # exact passthrough, no bias drift


def test_rollout_tracks_constant_setpoint():
    """A constant setpoint must appear unchanged as the predicted zone temp,
    independent of rollout-step alignment."""
    system = get_dynamics(make_cfg(), single_zone_mapping())
    batch, horizon = 4, 6
    setpoint = 23.0
    data = {
        "yn": torch.full((batch, 1, 1), 30.0),          # seed far from setpoint
        "U": torch.full((batch, horizon, 1), setpoint),
        "D": torch.randn(batch, horizon, 3),
        "Y": torch.zeros(batch, horizon, 1),            # only sets rollout length
    }

    out = system(data)

    yn = out["yn"]
    assert yn.shape == (batch, horizon + 1, 1)
    assert torch.allclose(yn[:, 1:, 0], torch.full((batch, horizon), setpoint))
