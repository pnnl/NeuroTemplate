"""
Tests for the objective diagnostics: term balance, and where the power
signal's clamp cuts the gradient.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem,
)
from neurotemplate.control.objective import (
    DOMINANCE_FACTOR,
    objective_balance,
    power_diagnostics,
    run_objective_pass,
)
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import build_loaders
from control_fixtures import BAND_LOSSES, POWER_LOSSES, full_setup


def _problem(**gains):
    """The band, du and energy terms at the given gains (default 1 each)."""
    s = full_setup(
        losses={**BAND_LOSSES, "energy": POWER_LOSSES["energy"]},
        gains={"comfort_lower": 1.0, "comfort_upper": 1.0, "du": 1.0,
               "energy": 1.0, **gains},
    )
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm, ns)
    problem = build_problem(cl, loss)
    train, _ = build_loaders(cfg, sysid_cfg, nm, [s["norm_df"]], [s["norm_df"]])
    return problem, loss, next(iter(train)), s


# ── objective_balance ────────────────────────────────────────────


def test_balance_covers_every_term_and_sums_to_the_total():
    problem, loss, batch, _ = _problem()
    out = run_objective_pass(problem, batch)

    balance = objective_balance(loss, out)

    assert [t.name for t in balance.terms] == [o.name for o in loss.objectives]
    assert sum(t.value for t in balance.terms) == pytest.approx(balance.total)
    assert sum(t.share for t in balance.terms) == pytest.approx(1.0)
    assert balance.total == pytest.approx(float(out["loss"]), rel=1e-5)


def test_a_pass_leaves_the_module_in_the_mode_it_found_it():
    problem, _, batch, _ = _problem()
    problem.train()
    run_objective_pass(problem, batch)
    assert problem.training


def test_a_lopsided_weight_is_flagged_as_dominant():
    """The measured failure: an energy gain suited to the last dataset."""
    problem, loss, batch, _ = _problem(du=0.0, energy=1e4)
    out = run_objective_pass(problem, batch)

    balance = objective_balance(loss, out)

    dominant = balance.dominant()
    assert [t.name for t in dominant] == ["energy_loss"]
    assert dominant[0].ratio_to_rest > DOMINANCE_FACTOR


# ── power_diagnostics ────────────────────────────────────────────


HMAX = 2

# A hand-built rollout rather than a real one: these are tests of a pure
# function, and the point is a known mix of positive and negative power. Half
# the scored steps are negative, so a clamp has something to bite on.
_SCORED = torch.tensor([[-2.0, -1.0, 1.0, 2.0]])


def _fake_out():
    yn = torch.cat([torch.zeros(1, HMAX), _SCORED], dim=1)[:, :, None]
    return {"yn": yn, "U": torch.ones(1, HMAX - 1 + 4, 1),
            "D": torch.ones(1, HMAX + 4, 1)}


def _diag(term_fn):
    return power_diagnostics(term_fn, _fake_out(), HMAX)


def _clamped(yn, u, d):
    """A regression head that goes negative, clamped: no gradient below zero."""
    return yn[:, :, 0].clamp_min(0.0)


def _snapped(yn, u, d):
    """The same forward values through a straight-through estimator.

    This is what an ordinal head does at a predicted zero: the value is hard,
    the backward path is smooth, so gradient still flows.
    """
    z = yn[:, :, 0]
    return z + (z.clamp_min(0.0) - z).detach()


def test_a_clamped_zero_is_dead_and_a_snapped_zero_is_not():
    """The whole point: identical forward values, opposite gradient verdicts.

    Reporting the forward value alone calls a correct "compressor off"
    prediction a defect, which is what the v6 ordinal-head run ran into.
    """
    clamped, snapped = _diag(_clamped), _diag(_snapped)

    # Half the scored steps are negative and both terms clamp them to zero.
    assert clamped.frac_zero == snapped.frac_zero == 0.5
    assert clamped.frac_dead == 0.5, "the clamp cut the chain at every zero"
    assert snapped.frac_dead == 0.0, "the straight-through path still carries"
