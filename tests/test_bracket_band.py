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
