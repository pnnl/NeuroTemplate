"""
Tests for the two behavioural control diagnostics: per-control saturation and
the reference-band jump audit.

Both exist because pooled comfort numbers cannot see behaviour, so the cases
here are the ones the pooled numbers get wrong: an actuator pinned at a bound,
and a band jump the controller ignores.
"""

import pytest
import torch

from neurotemplate.control.metrics import (
    SATURATION_TOL,
    actuation_metrics,
    actuation_to_dict,
    actuation_to_markdown,
    response_audit,
    response_to_dict,
    response_to_markdown,
)

COOL, HEAT = "cool sp [°C]", "heat sp [°C]"
ZONE = "zone [°C]"
U_NAMES = [COOL, HEAT]
U_LO = torch.tensor([20.0, 15.0])
U_HI = torch.tensor([26.0, 22.0])


# ── actuation_metrics ────────────────────────────────────────────


def test_saturation_fraction_counts_steps_on_a_bound():
    """One control pinned low the whole time, one parked in the middle."""
    u = torch.empty(2, 10, 2)
    u[:, :, 0] = 20.0        # exactly the lower bound
    u[:, :, 1] = 18.5        # mid-range

    cool, heat = actuation_metrics(u, U_NAMES, U_LO, U_HI)

    assert cool.frac_at_min == pytest.approx(1.0)
    assert cool.frac_at_max == pytest.approx(0.0)
    assert cool.frac_saturated == pytest.approx(1.0)
    assert heat.frac_saturated == pytest.approx(0.0)
    assert heat.mean_value == pytest.approx(18.5)


def test_a_bound_is_approached_not_reached():
    """`sigmoid_scale` never lands on a limit, so the check is a tolerance."""
    u = torch.full((1, 4, 2), 0.0)
    u[:, :, 0] = 20.0 + 0.5 * SATURATION_TOL * (26.0 - 20.0)   # just inside
    u[:, :, 1] = 15.0 + 2.0 * SATURATION_TOL * (22.0 - 15.0)   # clearly off it

    cool, heat = actuation_metrics(u, U_NAMES, U_LO, U_HI)

    assert cool.frac_at_min == pytest.approx(1.0)
    assert heat.frac_at_min == pytest.approx(0.0)


def test_mean_abs_du_is_per_channel():
    u = torch.zeros(1, 3, 2)
    u[0, :, 0] = torch.tensor([20.0, 21.0, 22.0])   # steps of 1.0
    u[0, :, 1] = torch.tensor([18.0, 18.0, 18.0])   # never moves

    cool, heat = actuation_metrics(u, U_NAMES, U_LO, U_HI)

    assert cool.mean_abs_du == pytest.approx(1.0)
    assert heat.mean_abs_du == pytest.approx(0.0)


def test_a_degenerate_box_saturates_once_not_twice():
    """u_min == u_max puts every command on both bounds; don't report 200%."""
    u = torch.full((1, 4, 2), 20.0)
    (cool, _) = actuation_metrics(u, U_NAMES, torch.tensor([20.0, 15.0]),
                                  torch.tensor([20.0, 22.0]))

    assert cool.frac_at_min == cool.frac_at_max == pytest.approx(1.0)
    assert cool.frac_saturated == pytest.approx(1.0)


def test_actuation_rejects_mismatched_names_and_bounds():
    u = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError, match="u_names has 1 entries"):
        actuation_metrics(u, [COOL], U_LO, U_HI)
    with pytest.raises(ValueError, match="one entry per control"):
        actuation_metrics(u, U_NAMES, torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError, match="3D"):
        actuation_metrics(torch.zeros(3, 2), U_NAMES, U_LO, U_HI)


def test_actuation_renderers():
    u = torch.full((1, 4, 2), 20.0)
    metrics = actuation_metrics(u, U_NAMES, U_LO, U_HI)

    table = actuation_to_markdown(metrics)
    assert "⚠️ pinned" in table            # cool sits on its floor throughout
    assert table.count("\n") == 3          # header + separator + 2 rows

    payload = actuation_to_dict(metrics)
    assert payload[COOL]["frac_saturated"] == pytest.approx(1.0)


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


def test_a_jump_that_asks_for_nothing_is_not_scored():
    """The channel is already inside the new band, so there is no demand."""
    y, ymin, ymax, u = _jump_case(cool_after=24.0, heat_after=18.0,
                                  y_after=24.0)
    ymax[0, 3:, 0] = 26.0            # ceiling moves but stays above the zone
    ymin[0, 3:, 0] = 21.0            # still a jump, still no demand

    assert _audit(y, ymin, ymax, u) == []


def test_a_constant_band_has_nothing_to_audit():
    y = torch.full((1, 6, 1), 30.0)          # wildly out of band, but no jumps
    ymin = torch.full((1, 6, 1), 20.0)
    ymax = torch.full((1, 6, 1), 25.0)
    u = torch.zeros(1, 6, 2)

    assert _audit(y, ymin, ymax, u) == []


def test_rise_and_fall_are_reported_separately():
    n = 9
    y = torch.full((2, n, 1), 24.0)
    ymin = torch.full((2, n, 1), 20.0)
    ymax = torch.full((2, n, 1), 26.0)
    # Window 0 gets a ceiling drop (zone must fall); window 1 a floor rise.
    ymax[0, 4:, 0] = 22.0
    ymin[1, 4:, 0] = 27.0
    ymax[1, 4:, 0] = 30.0
    u = torch.zeros(2, n, 2)

    results = _audit(y, ymin, ymax, u)

    assert [(r.name, r.direction, r.n_jumps) for r in results] == [
        (ZONE, "rise", 1), (ZONE, "fall", 1),
    ]


def test_response_renderers():
    y, ymin, ymax, u = _jump_case(21.0, 18.0, 22.0)
    results = _audit(y, ymin, ymax, u)

    table = response_to_markdown(results, U_NAMES)
    assert f"Δ`{COOL}`" in table and f"Δ`{HEAT}`" in table
    assert table.count("\n") == 1 + len(results)   # header + separator + rows

    payload = response_to_dict(results)
    assert list(payload) == [f"{ZONE} | fall"]
    assert payload[f"{ZONE} | fall"]["mean_du"][COOL] == pytest.approx(-3.0)


def test_response_rejects_mismatched_shapes():
    y, ymin, ymax, u = _jump_case(21.0, 18.0, 22.0)
    with pytest.raises(ValueError, match="Shape mismatch"):
        _audit(y, ymin[:, :4, :], ymax, u)
    with pytest.raises(ValueError, match="share batch/time"):
        _audit(y, ymin, ymax, u[:, :4, :])
