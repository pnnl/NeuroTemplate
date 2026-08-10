"""
End-to-end wiring test for the control diagnostics.

`nt control train` now runs three checks the pooled metrics cannot make — the
surrogate probe, the objective balance, and the closed-loop behaviour audit —
and threads all three into one reportcard. Each piece has unit tests; this
walks the same sequence the CLI does, so a signature that drifts out of step
fails here rather than an hour into a real run.

Everything above MLflow and argparse is exercised; those two are the CLI's
own business.
"""

import torch

from neurotemplate.common.data_utils import get_sw_dict_multi
from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem, power_term,
)
from neurotemplate.control.objective import (
    format_objective_report,
    objective_balance,
    objective_to_dict,
    run_objective_pass,
    term_shapes,
)
from neurotemplate.control.pipeline import (
    EVAL_BAND_SEED, evaluate_closed_loop, log_control_results,
)
from neurotemplate.control.policy import build_policy, describe_policy
from neurotemplate.control.reportcard import render_control_reportcard
from neurotemplate.control.scenarios import build_loaders, horizons
from neurotemplate.modeling.probe import authority_to_dict, probe_authority
from control_fixtures import full_setup
import control_user_defined as hooks


class _FakeLogger:
    def __init__(self):
        self.metrics, self.dicts = {}, {}

    def log_metrics(self, d):
        self.metrics.update(d)

    def log_dict(self, d, name):
        self.dicts[name] = d


def _run_diagnostics(**overrides):
    """The CLI's diagnostic sequence, minus MLflow and argparse."""
    s = full_setup(**overrides)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    test_dfs = [s["norm_df"]]

    u_bounds = hooks.actuation_bounds(nm, ns)
    probe_data = get_sw_dict_multi(
        dfs=test_dfs, cfg=sysid_cfg, nm_mapping=nm,
        steps=cfg.test_window, stride=cfg.test_stride,
    )
    probe_data.pop("source_idx")
    authority = probe_authority(
        s["dyn_sys"], probe_data, sysid_cfg.in_steps, nm, ns, *u_bounds,
    )

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm, ns)
    problem = build_problem(cl, loss)

    train_loader, _ = build_loaders(cfg, sysid_cfg, nm, ns, test_dfs, test_dfs)
    out = run_objective_pass(problem, next(iter(train_loader)))
    balance = objective_balance(loss, out)
    term = power_term(cfg, nm, ns)
    shapes = [] if (term is None or cfg.effort_weight == 0) else term_shapes(
        term, out, hmax, nm["U"], *u_bounds, ns,
    )

    results, _ = evaluate_closed_loop(cfg, sysid_cfg, cl, test_dfs, nm, ns)
    md = render_control_reportcard(
        cfg, sysid_cfg, describe_policy(policy, cfg, sysid_cfg, nm, ns),
        nm, hooks.controlled_channels(nm)[0], results,
        eval_band_seed=EVAL_BAND_SEED,
        authority=authority, balance=balance, shapes=shapes,
    )
    return dict(results=results, authority=authority, balance=balance,
                shapes=shapes, md=md, u_cols=nm["U"])


def test_every_diagnostic_section_reaches_the_reportcard():
    d = _run_diagnostics(effort_weight=1.0)

    for title in ("## Achievable envelope vs. comfort band",
                  "## Objective balance",
                  "## Actuation",
                  "## Band-jump response"):
        assert title in d["md"], f"missing section: {title}"


def test_the_diagnostics_describe_the_run_they_were_measured_on():
    d = _run_diagnostics(effort_weight=1.0)
    (u_col,) = d["u_cols"]

    # The probe and the closed-loop audit must agree on what the controls are.
    assert [a.u_name for a in d["authority"].authority] == [u_col] * len(
        d["authority"].authority
    )
    assert [m.name for m in d["results"].actuation] == [u_col]
    for r in d["results"].response:
        assert set(r.mean_du) == {u_col}

    # And the balance must cover exactly the terms the loss actually built.
    assert {t.name for t in d["balance"].terms} == {
        "comfort_lower_loss", "comfort_upper_loss", "du_loss", "effort_loss",
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


def test_a_constant_evaluation_band_says_so_rather_than_failing():
    d = _run_diagnostics(test_band_sub_samples=1)
    assert d["results"].response == []
    assert "nothing to audit" in d["md"]


def test_everything_logged_is_json_serializable():
    import json

    d = _run_diagnostics(effort_weight=1.0)
    logger = _FakeLogger()
    log_control_results(logger, d["results"])
    logger.log_dict(objective_to_dict(d["balance"], d["shapes"]),
                    name="objective_diagnostics.json")
    logger.log_dict(authority_to_dict(d["authority"]),
                    name="actuator_authority.json")

    for payload in logger.dicts.values():
        json.dumps(payload)
    assert all(isinstance(v, (int, float)) for v in logger.metrics.values())
    # The three headline behavioural keys, present without anyone asking.
    assert any(k.endswith("_frac_saturated") for k in logger.metrics)
    assert "control_mean_exceedance" in logger.metrics


def test_the_stdout_report_is_printable_before_training():
    d = _run_diagnostics(effort_weight=1.0)
    text = format_objective_report(d["balance"], d["shapes"])
    for term in ("comfort_lower_loss", "du_loss", "effort_loss"):
        assert term in text


def test_the_probe_does_not_disturb_the_surrogate():
    """Diagnostics run on a frozen model and must leave no trace on it."""
    s = full_setup()
    before = [p.clone() for p in s["dyn_sys"].parameters()]
    probe_data = get_sw_dict_multi(
        dfs=[s["norm_df"]], cfg=s["sysid_cfg"], nm_mapping=s["nm"],
        steps=s["cfg"].test_window, stride=s["cfg"].test_stride,
    )
    probe_data.pop("source_idx")

    probe_authority(
        s["dyn_sys"], probe_data, s["sysid_cfg"].in_steps, s["nm"],
        s["norm_stats"], *hooks.actuation_bounds(s["nm"], s["norm_stats"]),
    )

    for p, p0 in zip(s["dyn_sys"].parameters(), before):
        assert torch.equal(p, p0)
        assert p.grad is None
