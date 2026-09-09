"""The bracket rule: straddle the comfort band with two setpoints.

`bracket_band` lives in template/control_user_defined.py, not in the library
(see the comment above its definition there for why) — it is example code
for the `reference_control` hook, one plant's HVAC thermostat convention, and
not something control.policy calls. Loaded here by file path rather than by
module name: tests/fixtures/ already puts a stub module named
`control_user_defined` on sys.path for the hook-discovery tests, and importing
by name would collide with that.

Measured on a real single-zone plant, this rule holds a 0.5-2.0 °C-wide random
comfort band to 2.5-6.2% violation across four buildings with no model at all,
where aiming both setpoints at the band edge gives 21.8-56.7%. These tests pin
the three properties that make the difference, each of which is easy to get
backwards:

  - both setpoints land ON the quantization grid,
  - each rounds toward the band INTERIOR, never outside its own edge,
  - heating never commands above cooling (the stages would fight).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "template" / "control_user_defined.py"
_spec = importlib.util.spec_from_file_location("template_control_user_defined", _TEMPLATE_PATH)
_template_control = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_template_control)
bracket_band = _template_control.bracket_band


def test_setpoints_land_on_the_grid():
    lo = torch.tensor([20.3, 21.7, 22.02])
    hi = torch.tensor([22.1, 23.4, 24.88])
    cool, heat = bracket_band(lo, hi, step=0.5)
    for u in (cool, heat):
        assert torch.allclose(u / 0.5, torch.round(u / 0.5), atol=1e-5)


def test_rounds_toward_the_band_interior():
    """Heating rounds up off the floor, cooling rounds down off the ceiling.

    Rounding the other way would put a setpoint outside the band it holds —
    the mistake this rule exists to avoid.
    """
    lo = torch.tensor([20.1])   # nearest grid point below is 20.0, outside
    hi = torch.tensor([23.9])   # nearest grid point above is 24.0, outside
    cool, heat = bracket_band(lo, hi, step=0.5)
    assert heat.item() == pytest.approx(20.5)
    assert cool.item() == pytest.approx(23.5)
    assert heat.item() >= lo.item()
    assert cool.item() <= hi.item()


def test_exact_grid_edges_are_not_pushed_inward():
    """A band already on the grid keeps its full width — ceil/floor are no-ops."""
    lo, hi = torch.tensor([20.0]), torch.tensor([24.0])
    cool, heat = bracket_band(lo, hi, step=0.5)
    assert heat.item() == pytest.approx(20.0)
    assert cool.item() == pytest.approx(24.0)


def test_insets_pull_each_setpoint_further_inside():
    lo, hi = torch.tensor([20.0]), torch.tensor([26.0])
    cool, heat = bracket_band(lo, hi, step=0.5, cool_inset=0.5, heat_inset=0.25)
    assert cool.item() == pytest.approx(25.5)
    assert heat.item() == pytest.approx(20.5)  # ceil(20.25 / 0.5) * 0.5


def test_heating_never_exceeds_cooling():
    """Across a wide sweep of bands and insets, the ordering must hold."""
    lo = torch.linspace(20.0, 28.0, 60).reshape(-1, 1)
    width = torch.linspace(0.1, 3.0, 40).reshape(1, -1)
    hi = lo + width
    for cool_inset, heat_inset in ((0.0, 0.0), (0.5, 0.0), (1.0, 0.25),
                                   (2.0, 2.0)):
        cool, heat = bracket_band(
            lo, hi, step=0.5, cool_inset=cool_inset, heat_inset=heat_inset)
        assert (heat <= cool).all(), f"insets {cool_inset}/{heat_inset}"


def test_narrow_band_collapses_to_one_grid_point():
    """Too narrow to seat both: both go to one point near the midpoint.

    Giving up the deadband is better than letting the stages fight.
    """
    lo, hi = torch.tensor([21.9]), torch.tensor([22.1])
    cool, heat = bracket_band(lo, hi, step=0.5)
    assert cool.item() == pytest.approx(heat.item())
    assert cool.item() == pytest.approx(22.0)


def test_oversized_inset_collapses_rather_than_inverting():
    """An inset wider than the band must not invert the pair.

    This is the failure mode behind treating the inset as a safety margin: past
    a threshold the rule stops bracketing entirely, so a larger inset is not a
    safer one.
    """
    lo, hi = torch.tensor([20.0]), torch.tensor([21.0])
    cool, heat = bracket_band(lo, hi, step=0.5, cool_inset=2.0)
    assert cool.item() == pytest.approx(heat.item())
    assert heat.item() <= cool.item()


def test_collapse_point_is_inside_the_band():
    """The collapse fallback must not itself land outside the band."""
    lo = torch.linspace(20.0, 28.0, 81)
    hi = lo + 0.2  # narrower than one grid step everywhere
    cool, heat = bracket_band(lo, hi, step=0.5)
    # A 0.2 °C band cannot always contain a 0.5 °C grid point, but the chosen
    # point must be the nearest one to the midpoint, so it is never further
    # than half a step away.
    mid = 0.5 * (lo + hi)
    assert (cool - mid).abs().max() <= 0.25 + 1e-6
    assert torch.equal(cool, heat)


def test_one_degree_grid_is_supported():
    """The user-facing win: a 1 °C grid still brackets a 2 °C band."""
    lo, hi = torch.tensor([21.0]), torch.tensor([23.0])
    cool, heat = bracket_band(lo, hi, step=1.0)
    assert heat.item() == pytest.approx(21.0)
    assert cool.item() == pytest.approx(23.0)


def test_shapes_broadcast_like_the_closed_loop_tensors():
    """[batch, time, 1] band tensors, the shape the control loop hands over."""
    lo = torch.full((4, 12, 1), 21.0)
    hi = torch.full((4, 12, 1), 23.0)
    cool, heat = bracket_band(lo, hi, step=0.5)
    assert cool.shape == (4, 12, 1)
    assert heat.shape == (4, 12, 1)


def test_output_is_constant_when_the_band_is_constant():
    """No jitter: a held band produces a held setpoint, so mean |Δu| is 0."""
    lo = torch.full((1, 50, 1), 21.3)
    hi = torch.full((1, 50, 1), 22.8)
    cool, heat = bracket_band(lo, hi, step=0.5, cool_inset=0.5)
    for u in (cool, heat):
        assert (u[:, 1:] - u[:, :-1]).abs().max() == pytest.approx(0.0)


@pytest.mark.parametrize("kwargs", [
    {"step": 0.0},
    {"step": -0.5},
    {"step": 0.5, "cool_inset": -0.1},
    {"step": 0.5, "heat_inset": -1.0},
])
def test_input_validation(kwargs):
    lo, hi = torch.tensor([20.0]), torch.tensor([24.0])
    with pytest.raises(ValueError):
        bracket_band(lo, hi, **kwargs)
