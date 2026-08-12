"""
Tests for the objective diagnostics — term balance and term shape.

These are the two answers to "the weight is wrong": mis-scaled, which a
balance report catches, and mis-shaped, which only a sweep catches because no
weight fixes it.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem, resolve_du_weights,
)
from neurotemplate.control.objective import (
    DOMINANCE_FACTOR,
    ObjectiveBalance,
    TermBalance,
    TermShape,
    balance_to_markdown,
    format_objective_report,
    objective_balance,
    objective_to_dict,
    run_objective_pass,
    shapes_to_markdown,
    term_shapes,
)
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import build_loaders, horizons
from control_fixtures import full_setup

SETPT = "Zone Cooling Setpoint [°C]"


def _problem(**overrides):
    s = full_setup(**overrides)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm, ns)
    problem = build_problem(cl, loss)
    train, _ = build_loaders(cfg, sysid_cfg, nm, ns, [s["norm_df"]],
                             [s["norm_df"]])
    return problem, loss, next(iter(train)), s


# ── objective_balance ────────────────────────────────────────────


def test_balance_covers_every_term_and_sums_to_the_total():
    problem, loss, batch, _ = _problem(
        comfort_weight=1.0, du_weight=1.0, energy_weight=1.0,
    )
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
    """The measured failure: an energy weight suited to the last dataset."""
    problem, loss, batch, _ = _problem(
        comfort_weight=1.0, du_weight=0.0, energy_weight=1e4,
    )
    out = run_objective_pass(problem, batch)

    balance = objective_balance(loss, out)

    dominant = balance.dominant()
    assert [t.name for t in dominant] == ["energy_loss"]
    assert dominant[0].ratio_to_rest > DOMINANCE_FACTOR
    assert "⚠️ dominates" in balance_to_markdown(balance)
    assert "WARNING" in format_objective_report(balance, [])


def test_a_balanced_objective_is_not_flagged():
    terms = [TermBalance("a", 1.0, 0.5, 1.0), TermBalance("b", 1.0, 0.5, 1.0)]
    assert ObjectiveBalance(terms, total=2.0).dominant() == []


def test_an_all_zero_objective_is_not_reported_as_dominated():
    """A policy that starts perfectly in band must not trip the warning."""
    problem, loss, batch, _ = _problem(
        comfort_weight=1.0, du_weight=0.0, energy_weight=0.0,
    )
    out = run_objective_pass(problem, batch)
    zeroed = {**out, "comfort_lower_loss": torch.tensor(0.0),
              "comfort_upper_loss": torch.tensor(0.0)}

    balance = objective_balance(loss, zeroed)

    assert balance.total == 0.0
    assert all(t.ratio_to_rest == 0.0 for t in balance.terms)
    assert balance.dominant() == []


def test_a_lone_term_reports_an_infinite_ratio():
    problem, loss, batch, _ = _problem(
        comfort_weight=0.0, du_weight=1.0, energy_weight=0.0,
    )
    out = run_objective_pass(problem, batch)

    (term,) = objective_balance(loss, out).terms

    assert term.ratio_to_rest == float("inf")
    assert "∞" in balance_to_markdown(ObjectiveBalance([term], term.value))


# ── term_shapes ──────────────────────────────────────────────────


def _shapes(term_fn, n_grid=9):
    problem, _, batch, s = _problem(energy_weight=1.0)
    out = run_objective_pass(problem, batch)
    _, _, hmax = horizons(s["cfg"], s["sysid_cfg"])
    nm, ns = s["nm"], s["norm_stats"]
    lo = torch.tensor([(ns[c]["min"] - ns[c]["shift"]) / ns[c]["stretch"]
                       for c in nm["U"]])
    hi = torch.tensor([(ns[c]["max"] - ns[c]["shift"]) / ns[c]["stretch"]
                       for c in nm["U"]])
    return term_shapes(term_fn, out, hmax, nm["U"], lo, hi, ns, n_grid=n_grid)


def test_a_monotone_term_is_reported_without_a_verdict():
    """A term minimised on an endpoint is normal, not a defect.

    Here the term is the setpoint itself, so it is minimised at the floor. An
    honest power metric behaves the same way — it costs more as the actuator
    works harder — so the sweep reports where the minimum is and says nothing
    about whether that is wrong.
    """
    (shape,) = _shapes(lambda yn, u, d: u[:, :, 0])

    assert shape.u_name == SETPT
    assert shape.argmin == 0
    assert not shape.is_flat
    assert shape.minimised_at == pytest.approx(shape.grid[0])

    table = shapes_to_markdown([shape])
    assert f"{shape.minimised_at:.4g}" in table
    assert "⚠️" not in table

    report = format_objective_report(ObjectiveBalance([], 0.0), [shape])
    assert "WARNING" not in report
    assert "no verdict attached" in report


def test_an_interior_optimum_is_reported_too():
    """A term that costs distance from a target reports its interior minimum."""
    target = 24.0
    ns_shift, ns_stretch = None, None

    def term(yn, u, d):
        return (u[:, :, 0] - _norm(target)) ** 2

    def _norm(x):
        return (x - ns_shift) / ns_stretch

    s = full_setup()
    ns_shift = s["norm_stats"][SETPT]["shift"]
    ns_stretch = s["norm_stats"][SETPT]["stretch"]

    (shape,) = _shapes(term)

    assert 0 < shape.argmin < len(shape.values) - 1
    assert shape.minimised_at == pytest.approx(target, abs=1.0)


def test_a_term_that_ignores_the_control_reads_as_flat():
    """Metered power lives in Y, so the sweep must not cry wolf about it."""
    (shape,) = _shapes(lambda yn, u, d: yn[:, :, 0])

    assert shape.is_flat
    assert "flat (no direct effect)" in shapes_to_markdown([shape])


def test_the_grid_spans_the_actuation_box():
    (shape,) = _shapes(lambda yn, u, d: u[:, :, 0], n_grid=5)
    s = full_setup()
    stats = s["norm_stats"][SETPT]

    assert len(shape.grid) == len(shape.values) == 5
    assert shape.grid[0] == pytest.approx(stats["min"], abs=1e-4)
    assert shape.grid[-1] == pytest.approx(stats["max"], abs=1e-4)


def test_objective_to_dict_is_serializable():
    import json

    problem, loss, batch, _ = _problem(energy_weight=1.0)
    out = run_objective_pass(problem, batch)
    balance = objective_balance(loss, out)
    shape = TermShape(SETPT, grid=[20.0, 23.0, 26.0], values=[3.0, 2.0, 1.0],
                      argmin=2)

    payload = objective_to_dict(balance, [shape])

    assert payload["power_shape"][SETPT]["minimised_at"] == 26.0
    assert payload["power_shape"][SETPT]["values"] == [3.0, 2.0, 1.0]
    json.dumps(payload)  # must not raise


# ── per-channel du_weight ────────────────────────────────────────


def test_scalar_du_weight_broadcasts_to_every_control():
    assert resolve_du_weights(2.5, ["a", "b"]) == [2.5, 2.5]


def test_list_du_weight_is_positional():
    assert resolve_du_weights([1.0, 4.0], ["a", "b"]) == [1.0, 4.0]
    with pytest.raises(ValueError, match="has 1 entries but there are 2"):
        resolve_du_weights([1.0], ["a", "b"])


def test_dict_du_weight_defaults_omitted_controls_to_zero():
    assert resolve_du_weights({"b": 3.0}, ["a", "b"]) == [0.0, 3.0]
    with pytest.raises(ValueError, match="not controls"):
        resolve_du_weights({"c": 1.0}, ["a", "b"])


def test_scalar_du_weight_reproduces_the_pooled_penalty():
    """The per-channel form must be a strict generalization, not a change."""
    s = full_setup(comfort_weight=0.0, du_weight=3.0, energy_weight=0.0)
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm, s["norm_stats"])

    b, W, nu = 2, hmax + cfg.train_pred_steps, len(nm["U"])
    u = torch.randn(b, W, nu)
    data = {
        "yn": torch.zeros(b, W, len(nm["Y"])), "U": u,
        "D": torch.zeros(b, W, len(nm["D"])),
        "ymin": torch.zeros(b, W, 1), "ymax": torch.ones(b, W, 1),
    }
    u_h = u[:, hmax:, :]
    expected = 3.0 * (u_h[:, 1:, :] - u_h[:, :-1, :]).abs().mean()

    assert float(loss(data)["loss"]) == pytest.approx(float(expected), rel=1e-5)


def test_a_zero_weight_on_every_control_drops_the_term():
    s = full_setup(comfort_weight=1.0, du_weight={}, energy_weight=0.0)
    loss = build_control_loss(s["cfg"], s["sysid_cfg"], s["nm"],
                              s["norm_stats"])
    assert "du_loss" not in [o.name for o in loss.objectives]
