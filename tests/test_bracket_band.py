"""The bracket rule: straddle the comfort band with two setpoints.

`bracket_band` lives in the template's nt_control.py, not in the library
(see the comment above its definition there for why) — it is example code
for the `reference_control` hook, one plant's HVAC thermostat convention, and
not something control.policy calls. Loaded here by file path rather than by
module name: tests/fixtures/ already puts a stub module named
`nt_control` on sys.path for the hook-discovery tests, and importing
by name would collide with that.

Measured on a real single-zone plant, this rule holds a 0.5-2.0 °C-wide random
comfort band to 2.5-6.2% violation across four buildings with no model at all,
where aiming both setpoints at the band edge gives 21.8-56.7% (both measured
with setpoints rounded to a half-degree grid, which the rule no longer does:
resolution is the building's). These tests pin the property that is easy to
get backwards: heating never commands above cooling (the stages would fight).
"""

import importlib.util

import pytest
import torch

from neurotemplate.init import TEMPLATE

_spec = importlib.util.spec_from_file_location("template_nt_control", TEMPLATE / "nt_control.py")
_template_control = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_template_control)
bracket_band = _template_control.bracket_band


def test_oversized_inset_collapses_rather_than_inverting():
    """An inset wider than the band must not invert the pair.

    This is the failure mode behind treating the inset as a safety margin: past
    a threshold the rule stops bracketing entirely, so a larger inset is not a
    safer one.
    """
    lo, hi = torch.tensor([20.0]), torch.tensor([21.0])
    cool, heat = bracket_band(lo, hi, cool_inset=2.0)
    assert cool.item() == pytest.approx(heat.item())
    assert heat.item() <= cool.item()
