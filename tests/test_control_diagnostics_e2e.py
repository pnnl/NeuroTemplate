"""
End-to-end wiring test for the control diagnostics.

`nt control train` runs several checks the pooled metrics cannot make — the
objective balance, the power-signal sweep, the closed-loop behaviour audit,
the residual box and the learned-vs-reference call — and threads all of them
into one reportcard. Each piece has unit tests; this walks the same sequence
the CLI does, so a signature that drifts out of step fails here rather than an
hour into a real run.

Everything above MLflow and argparse is exercised; those two are the CLI's
own business.
"""

from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem, power_term,
)
from neurotemplate.control.losses import resolve_objective
from neurotemplate.control.objective import (
    power_diagnostics,
    objective_balance,
    objective_notes,
    run_objective_pass,
    term_shapes,
)
from neurotemplate.control.pipeline import (
    evaluate_closed_loop,
)
from neurotemplate.control.policy import build_policy, describe_policy
from neurotemplate.control.reportcard import render_control_reportcard
from neurotemplate.control.scenarios import build_loaders, horizons
from control_fixtures import (
    BAND_GAINS, BAND_LOSSES, POWER_LOSSES, TARIFF, TARIFF_SCALE, full_setup,
)
import control_user_defined as hooks


def _with_power(*names, **overrides):
    """The band and du, plus the named power terms, every gain 1."""
    losses = {**BAND_LOSSES, **{n: POWER_LOSSES[n] for n in names}}
    return dict(losses=losses, gains={**BAND_GAINS, **{n: 1.0 for n in names}},
                reference_scale={TARIFF: TARIFF_SCALE} if "cost" in names else {},
                **overrides)


def _run_diagnostics(reference=None, **overrides):
    """The CLI's diagnostic sequence, minus MLflow and argparse."""
    s = full_setup(reference=reference, **overrides)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    objective = resolve_objective(cfg, nm)
    test_dfs = [s["norm_df"]]

    u_bounds = hooks.actuation_bounds(nm, ns)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm, ns)
    problem = build_problem(cl, loss)

    train_loader, _ = build_loaders(cfg, sysid_cfg, nm, test_dfs, test_dfs)
    out = run_objective_pass(problem, next(iter(train_loader)))
    balance = objective_balance(loss, out)
    term = power_term(cfg, nm, ns)
    if term is None or not objective.reads_power:
        shapes, power_diag = [], None
    else:
        shapes = term_shapes(term, problem, policy, next(iter(train_loader)), hmax,
                             nm["U"], *u_bounds, ns)
        power_diag = power_diagnostics(term, out, hmax)

    arch = describe_policy(policy, cfg, sysid_cfg, nm, ns)
    notes = objective_notes(objective, cfg)
    results, _ = evaluate_closed_loop(cfg, sysid_cfg, cl, test_dfs, nm, ns)
    md = render_control_reportcard(
        cfg, sysid_cfg, arch, nm, results,
        balance=balance, shapes=shapes,
        power=power_diag, objective_notes=notes,
    )
    return dict(results=results, balance=balance,
                shapes=shapes, notes=notes, power=power_diag,
                md=md, u_cols=nm["U"])


def test_a_stepping_test_band_produces_something_to_audit():
    """A band that steps inside the test windows is what makes the response
    audit non-empty; the fixture's steps every 6 rows."""
    d = _run_diagnostics()
    assert d["results"].response, "band jumps should have been scored"
    for r in d["results"].response:
        assert r.n_jumps > 0
        assert 0.0 <= r.frac_actuated <= 1.0
        assert r.mean_span >= 1


# ── Grid response and residual control, through the same sequence ─


def test_the_grid_terms_reach_every_diagnostic():
    """cost and peak show up on the balance report, in the reportcard and in
    the energy section with nothing told about them by name."""
    d = _run_diagnostics(**_with_power("energy", "cost", "peak"))

    assert {t.name for t in d["balance"].terms} == {
        "comfort_lower_loss", "comfort_upper_loss", "du_loss",
        "energy_loss", "cost_loss", "peak_loss",
    }
    assert set(d["results"].energy.weighted_energy) == {TARIFF}
    assert TARIFF in d["md"]
    # The power sweep is bound to the shared signal, not to the three terms.
    assert [s.u_name for s in d["shapes"]] == d["u_cols"]
    assert d["power"] is not None
    # Metered power reads the surrogate's outputs, so the flat U-sweep must be
    # reported as that rather than as an inert term.
    assert d["power"].reads_outputs


def test_an_unpreviewed_tariff_is_flagged_as_reactive_only():
    d = _run_diagnostics(**_with_power("cost"))
    assert any("previews 0 future steps" in n for n in d["notes"])
    d = _run_diagnostics(**_with_power("cost", future={TARIFF: 4}))
    assert not any("previews 0 future steps" in n for n in d["notes"])
