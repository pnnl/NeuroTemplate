"""
Tests for the content-free `dummy` surrogate (zone temp = cooling setpoint).

The dummy goes through the same `get_dynamics` path as every other model. It
must (a) be registered, (b) reproduce the setpoint exactly as the next state
(passthrough), and (c) refuse a mapping where the number of zone temps and
setpoints disagree.
"""
import pytest
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import get_dummy, get_dynamics, model_catalog

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


def test_registration():
    assert "dummy" in model_catalog


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


def test_passthrough_ignores_in_steps():
    """The passthrough declares a width-1 input, so it must never be handed a
    window — otherwise it slices the wrong columns out of the flattened cat."""
    in_steps, batch, horizon = 4, 4, 6
    setpoint = 23.0
    system = get_dynamics(make_cfg(in_steps), single_zone_mapping())
    assert system.nodes[0].input_map == {}

    data = {
        "yn": torch.full((batch, in_steps, 1), 30.0),   # seed far from setpoint
        "U": torch.full((batch, horizon, 1), setpoint),
        "D": torch.randn(batch, horizon, 3),
        "Y": torch.zeros(batch, horizon, 1),            # only sets rollout length
    }

    yn = system(data)["yn"]

    n_pred = yn.shape[1] - in_steps
    assert torch.allclose(
        yn[:, in_steps:, 0], torch.full((batch, n_pred), setpoint)
    )


def test_requires_ny_equals_nu():
    """Two zone temps but one setpoint is ambiguous and must fail fast."""
    bad = {"Y": [ZONE, "Zone 2 Air Temperature [°C]"], "U": [SETPT],
           "D": [OUT], "t": ["sec_elapsed"]}
    with pytest.raises(ValueError, match="one setpoint per zone temp"):
        get_dummy(make_cfg(), bad)
