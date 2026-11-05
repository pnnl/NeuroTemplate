"""
Tests for the two behavioural control diagnostics: per-control saturation and
the reference-band jump audit.

Both exist because pooled comfort numbers cannot see behaviour, so the cases
here are the ones the pooled numbers get wrong: an actuator pinned at a bound,
and a band jump the controller ignores.
"""

import pytest
import torch

from neurotemplate.metrics.control import (
    response_audit,
)

COOL, HEAT = "cool sp [°C]", "heat sp [°C]"
ZONE = "zone [°C]"
U_NAMES = [COOL, HEAT]
U_LO = torch.tensor([20.0, 15.0])
U_HI = torch.tensor([26.0, 22.0])


# ── response_audit ───────────────────────────────────────────────


def _jump_case(cool_after: float, heat_after: float, y_after: float):
    """One window, one band jump at step 3 that leaves the zone too hot.

    Before the jump the zone sits inside a wide band; the jump narrows the
    ceiling down past the zone, so the demand is "fall". `y_after` is where
    the zone ends up two steps later, i.e. whether the loop closed it.
    """
    n = 8
    y = torch.full((1, n, 1), 24.0)
    y[0, 5:, 0] = y_after
    ymin = torch.full((1, n, 1), 20.0)
    ymax = torch.full((1, n, 1), 26.0)
    ymax[0, 3:, 0] = 23.0            # the jump: ceiling drops below the zone
    u = torch.zeros(1, n, 2)
    u[0, :, 0], u[0, :, 1] = 24.0, 18.0
    u[0, 3:, 0], u[0, 3:, 1] = cool_after, heat_after
    return y, ymin, ymax, u


def _audit(*args):
    return response_audit(*args, [ZONE], U_NAMES, U_LO, U_HI)


def test_a_jump_the_controller_answers():
    y, ymin, ymax, u = _jump_case(cool_after=21.0, heat_after=18.0,
                                  y_after=22.0)

    (r,) = _audit(y, ymin, ymax, u)

    assert (r.name, r.direction, r.n_jumps) == (ZONE, "fall", 1)
    assert r.frac_actuated == 1.0            # the cooling setpoint dropped 3 °C
    assert r.frac_closed == 1.0              # 24.0 -> 22.0 is the right way
    assert r.frac_resolved == 1.0            # and 22.0 is inside [20, 23]
    assert r.mean_du[COOL] == pytest.approx(-3.0)
    assert r.mean_du[HEAT] == pytest.approx(0.0)


def test_a_jump_the_controller_ignores():
    """The exact silent failure this audit exists for: nothing moves."""
    y, ymin, ymax, u = _jump_case(cool_after=24.0, heat_after=18.0,
                                  y_after=24.0)

    (r,) = _audit(y, ymin, ymax, u)

    assert r.frac_actuated == 0.0
    assert r.frac_closed == 0.0
    assert r.frac_resolved == 0.0
    assert r.mean_du[COOL] == pytest.approx(0.0)
