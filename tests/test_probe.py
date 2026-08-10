"""
Tests for the counterfactual actuator probe.

The `dummy` surrogate is a passthrough — next state = setpoint, in physical
units — so the probe's answers are known in closed form: swinging the setpoint
across its range must move the channel by exactly that range, the achievable
envelope must be exactly the actuation box, and a control the model ignores
must come back with zero authority.
"""

import pytest
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.probe import (
    ActuatorAuthority,
    authority_to_dict,
    authority_to_markdown,
    envelope_to_markdown,
    probe_authority,
    u_bounds_from_norm_stats,
)

ZONE = "zone [°C]"
ZONE2 = "zone 2 [°C]"
SETPT = "setpoint [°C]"
SETPT2 = "setpoint 2 [°C]"
OUT = "outdoor [°C]"

U_LO, U_HI = 20.0, 26.0
WINDOW = 12
IN_STEPS = 1


def _mapping(n_zones: int = 1) -> dict[str, list[str]]:
    y = [ZONE] if n_zones == 1 else [ZONE, ZONE2]
    u = [SETPT] if n_zones == 1 else [SETPT, SETPT2]
    return {"Y": y, "U": u, "D": [OUT], "t": ["sec_elapsed"]}


def _norm_stats(nm: dict) -> dict:
    """Identity normalization (dummy runs in physical units), U box 20–26 °C."""
    stats = {}
    for c in nm["Y"] + nm["U"] + nm["D"]:
        lo, hi = (U_LO, U_HI) if c in nm["U"] else (18.0, 30.0)
        stats[c] = {"shift": 0.0, "stretch": 1.0, "min": lo, "max": hi}
    return stats


def _cfg() -> Config:
    cfg = Config()
    cfg.model = "dummy"
    cfg.in_steps = IN_STEPS
    cfg.norm_type = None
    cfg.device = "cpu"
    return cfg


def _data(nm: dict, batch: int = 3) -> dict:
    ny, nu, nd = len(nm["Y"]), len(nm["U"]), len(nm["D"])
    return {
        "Y": torch.zeros(batch, WINDOW, ny),   # only sets the rollout length
        "yn": torch.full((batch, IN_STEPS, ny), 30.0),
        "U": torch.full((batch, WINDOW, nu), 23.0),
        "D": torch.randn(batch, WINDOW, nd),
        "t": torch.arange(WINDOW, dtype=torch.float32).expand(batch, WINDOW)[..., None],
    }


def _probe(n_zones: int = 1, **kwargs):
    nm = _mapping(n_zones)
    ns = _norm_stats(nm)
    dyn = get_dynamics(_cfg(), nm)
    umin, umax = u_bounds_from_norm_stats(nm, ns)
    return probe_authority(
        dyn, _data(nm), IN_STEPS, nm, ns, umin, umax, **kwargs
    ), nm


def test_authority_equals_the_swept_range_for_a_passthrough():
    """y_next = u, so a full setpoint swing moves the channel by exactly it."""
    report, _ = _probe()

    assert len(report.authority) == 1
    a = report.authority[0]
    assert a.y_name == ZONE and a.u_name == SETPT
    assert a.delta == pytest.approx(U_HI - U_LO)
    assert (a.u_lo, a.u_hi) == (U_LO, U_HI)
    assert not a.is_dead()


def test_envelope_is_the_actuation_box_for_a_passthrough():
    report, _ = _probe()

    e = report.envelope_for(ZONE)
    assert e.lo == pytest.approx(U_LO)
    assert e.hi == pytest.approx(U_HI)
    assert e.covers(21.0, 25.0)
    assert e.covers(19.0, 21.0)          # partial overlap still counts
    assert not e.covers(27.0, 29.0)      # band above anything achievable


def test_cross_channel_authority_is_zero_and_flagged_dead():
    """Setpoint 2 drives zone 2 only, so its authority over zone 1 is nil."""
    report, _ = _probe(n_zones=2)

    by_pair = {(a.y_name, a.u_name): a for a in report.authority}
    assert by_pair[(ZONE, SETPT)].delta == pytest.approx(U_HI - U_LO)
    assert by_pair[(ZONE, SETPT2)].delta == pytest.approx(0.0)
    assert by_pair[(ZONE, SETPT2)].is_dead()
    # Each setpoint still owns its own zone, so neither is dead outright.
    assert report.dead_actuators() == []


def test_dead_actuators_lists_a_control_with_no_lever_anywhere():
    a_live = ActuatorAuthority(ZONE, SETPT, delta=5.0, y_range=10.0,
                               u_lo=20.0, u_hi=26.0)
    a_dead = ActuatorAuthority(ZONE, SETPT2, delta=0.01, y_range=10.0,
                               u_lo=20.0, u_hi=26.0)
    from neurotemplate.modeling.probe import AuthorityReport
    report = AuthorityReport([a_live, a_dead], [], settle_steps=0, n_windows=1)

    assert report.dead_actuators() == [SETPT2]


def test_settle_steps_defaults_to_half_the_rollout():
    report, _ = _probe()
    assert report.settle_steps == (WINDOW - IN_STEPS) // 2

    explicit, _ = _probe(settle_steps=2)
    assert explicit.settle_steps == 2


def test_window_count_is_capped():
    nm = _mapping()
    ns = _norm_stats(nm)
    dyn = get_dynamics(_cfg(), nm)
    umin, umax = u_bounds_from_norm_stats(nm, ns)

    report = probe_authority(
        dyn, _data(nm, batch=50), IN_STEPS, nm, ns, umin, umax, max_windows=8,
    )
    assert report.n_windows == 8


def test_rejects_a_mapping_with_no_controls():
    nm = {"Y": [ZONE], "U": [], "D": [OUT], "t": ["sec_elapsed"]}
    ns = _norm_stats(_mapping())
    dyn = get_dynamics(_cfg(), _mapping())
    with pytest.raises(ValueError, match="nm_mapping\\['U'\\] is empty"):
        probe_authority(dyn, _data(_mapping()), IN_STEPS, nm, ns,
                        torch.zeros(0), torch.zeros(0))


def test_rejects_bounds_of_the_wrong_width():
    nm = _mapping()
    ns = _norm_stats(nm)
    dyn = get_dynamics(_cfg(), nm)
    with pytest.raises(ValueError, match="one entry per U column"):
        probe_authority(dyn, _data(nm), IN_STEPS, nm, ns,
                        torch.zeros(2), torch.ones(2))


def test_rejects_a_window_with_no_room_to_roll_out():
    nm = _mapping()
    ns = _norm_stats(nm)
    dyn = get_dynamics(_cfg(), nm)
    umin, umax = u_bounds_from_norm_stats(nm, ns)
    with pytest.raises(ValueError, match="at least 2 rollout steps"):
        probe_authority(dyn, _data(nm), WINDOW - 1, nm, ns, umin, umax)


def test_renderers_cover_every_row_and_flag_dead_actuators():
    report, _ = _probe(n_zones=2)

    table = authority_to_markdown(report)
    assert table.count("\n") == 1 + len(report.authority)  # header + separator
    assert "⚠️ no authority" in table                      # the cross pairs

    envelope = envelope_to_markdown(report)
    assert ZONE in envelope and ZONE2 in envelope

    payload = authority_to_dict(report)
    assert len(payload["authority"]) == 4
    assert set(payload["envelope"]) == {ZONE, ZONE2}
    assert payload["n_windows"] == 3
