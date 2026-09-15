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
from neurotemplate.control.objective import (
    power_diagnostics,
    objective_balance,
    objective_notes,
    objective_to_dict,
    run_objective_pass,
    term_shapes,
)
from neurotemplate.control.pipeline import (
    EVAL_BAND_SEED, clears_reference, evaluate_baseline, evaluate_closed_loop,
    log_control_results,
)
from neurotemplate.control.policy import build_policy, describe_policy
from neurotemplate.control.reportcard import render_control_reportcard
from neurotemplate.control.scenarios import build_loaders, horizons
from control_fixtures import full_setup, install_price
import control_user_defined as hooks


class _FakeLogger:
    def __init__(self):
        self.metrics, self.dicts = {}, {}

    def log_metrics(self, d):
        self.metrics.update(d)

    def log_dict(self, d, name):
        self.dicts[name] = d


def _run_diagnostics(tmp_path=None, **overrides):
    """The CLI's diagnostic sequence, minus MLflow and argparse."""
    s = full_setup(**overrides)
    if tmp_path is not None:
        install_price(s["cfg"], tmp_path)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    test_dfs = [s["norm_df"]]

    u_bounds = hooks.actuation_bounds(nm, ns)

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm, ns)
    problem = build_problem(cl, loss)

    train_loader, _ = build_loaders(cfg, sysid_cfg, nm, ns, test_dfs, test_dfs)
    out = run_objective_pass(problem, next(iter(train_loader)))
    balance = objective_balance(loss, out)
    term = power_term(cfg, nm, ns)
    weighted = cfg.energy_weight > 0 or cfg.cost_weight > 0 or cfg.peak_weight > 0
    if term is None or not weighted:
        shapes, power_diag = [], None
    else:
        shapes = term_shapes(term, out, hmax, nm["U"], *u_bounds, ns)
        power_diag = power_diagnostics(term, out, hmax)

    arch = describe_policy(policy, cfg, sysid_cfg, nm, ns)
    notes = objective_notes(cfg, arch)
    results, _ = evaluate_closed_loop(cfg, sysid_cfg, cl, test_dfs, nm, ns)
    baseline = evaluate_baseline(cfg, sysid_cfg, s["dyn_node"], test_dfs, nm, ns)
    md = render_control_reportcard(
        cfg, sysid_cfg, arch, nm, hooks.regulated(nm)[0], results,
        eval_band_seed=EVAL_BAND_SEED, baseline=baseline,
        clears_reference=clears_reference(results, baseline),
        balance=balance, shapes=shapes,
        power=power_diag, objective_notes=notes,
    )
    return dict(results=results, balance=balance, baseline=baseline,
                shapes=shapes, notes=notes, power=power_diag,
                md=md, u_cols=nm["U"])


def test_the_diagnostics_describe_the_run_they_were_measured_on():
    d = _run_diagnostics(energy_weight=1.0)
    (u_col,) = d["u_cols"]

    assert [m.name for m in d["results"].actuation] == [u_col]
    for r in d["results"].response:
        assert set(r.mean_du) == {u_col}

    # And the balance must cover exactly the terms the loss actually built.
    assert {t.name for t in d["balance"].terms} == {
        "comfort_lower_loss", "comfort_upper_loss", "du_loss", "energy_loss",
    }
    assert [s.u_name for s in d["shapes"]] == [u_col]


def test_a_jumpy_evaluation_band_produces_something_to_audit():
    """`test_band_sub_samples > 1` is what makes the response audit non-empty."""
    d = _run_diagnostics(test_band_sub_samples=4, band_width_min=0.1,
                         band_width_max=0.3)
    assert d["results"].response, "band jumps should have been scored"
    for r in d["results"].response:
        assert r.n_jumps > 0
        assert 0.0 <= r.frac_actuated <= 1.0
        assert r.mean_span >= 1


def test_a_constant_evaluation_band_has_nothing_to_audit():
    d = _run_diagnostics(test_band_sub_samples=1)
    assert d["results"].response == []


def test_everything_logged_is_json_serializable():
    import json

    d = _run_diagnostics(energy_weight=1.0)
    logger = _FakeLogger()
    log_control_results(logger, d["results"], baseline=d["baseline"])
    logger.log_dict(
        objective_to_dict(d["balance"], d["shapes"], d["power"],
                          d["notes"]),
        name="objective_diagnostics.json",
    )

    for payload in logger.dicts.values():
        json.dumps(payload)
    assert all(isinstance(v, (int, float)) for v in logger.metrics.values())
    # The headline behavioural keys, present without anyone asking.
    assert any(k.endswith("_frac_saturated") for k in logger.metrics)
    assert "control_mean_exceedance" in logger.metrics
    assert "control_clears_reference" in logger.metrics


# ── Grid response and residual control, through the same sequence ─


def test_the_grid_terms_reach_every_diagnostic(tmp_path):
    """cost and peak show up on the balance report, in the reportcard and in
    the energy section with nothing told about them by name."""
    d = _run_diagnostics(tmp_path, energy_weight=1.0, cost_weight=1.0,
                         peak_weight=1.0)

    assert {t.name for t in d["balance"].terms} == {
        "comfort_lower_loss", "comfort_upper_loss", "du_loss",
        "energy_loss", "cost_loss", "peak_loss",
    }
    assert d["results"].energy.energy_cost is not None
    # The power sweep is bound to the shared signal, not to the three terms.
    assert [s.u_name for s in d["shapes"]] == d["u_cols"]
    assert d["power"] is not None
    # Metered power reads the surrogate's outputs, so the flat U-sweep must be
    # reported as that rather than as an inert term.
    assert d["power"].reads_outputs


def test_peak_alone_is_flagged_as_raising_energy(tmp_path):
    d = _run_diagnostics(tmp_path, energy_weight=0.0, peak_weight=1.0)
    assert any("only power term" in n for n in d["notes"])
