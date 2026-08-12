"""One real optimisation run per feature, through the trainer the CLI uses.

The unit tests check each piece; this checks that the pieces compose under
`train_policy` — a real optimizer, a real dev loader, real checkpointing —
because a term that builds and a term that trains are different claims.
"""

import torch

from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem,
)
from neurotemplate.control.pipeline import (
    clears_reference, evaluate_baseline, evaluate_closed_loop, train_policy,
)
from neurotemplate.control.policy import build_policy, reference_over_horizon
from neurotemplate.control.scenarios import build_loaders, horizons
from control_fixtures import full_setup, install_price_hook


def _train(s):
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    problem = build_problem(cl, build_control_loss(cfg, sysid_cfg, nm, ns))
    train_loader, dev_loader = build_loaders(
        cfg, sysid_cfg, nm, ns, [s["norm_df"]], [s["norm_df"]],
    )
    trained, best_dev, _ = train_policy(cfg, problem, train_loader, dev_loader)
    return policy, trained, best_dev


def test_a_grid_response_objective_trains(monkeypatch):
    install_price_hook(monkeypatch)
    s = full_setup(epochs=3, comfort_weight=10.0, du_weight=1.0,
                   energy_weight=0.1, cost_weight=0.1, peak_weight=0.1)
    _, trained, best_dev = _train(s)

    assert torch.isfinite(torch.tensor(best_dev))
    results, out = evaluate_closed_loop(
        s["cfg"], s["sysid_cfg"], trained, [s["norm_df"]], s["nm"],
        s["norm_stats"],
    )
    assert results.energy.energy_cost is not None
    assert torch.isfinite(out["U"]).all()


def test_a_residual_policy_trains_and_stays_in_its_box(monkeypatch):
    install_price_hook(monkeypatch)
    s = full_setup(epochs=3, residual_delta=1.0, setpoint_step=0.5,
                   comfort_weight=10.0, cost_weight=0.1)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    _, trained, best_dev = _train(s)

    assert torch.isfinite(torch.tensor(best_dev))
    results, out = evaluate_closed_loop(
        cfg, sysid_cfg, trained, [s["norm_df"]], nm, ns,
    )
    # The commanded setpoint never leaves the box, after training and after
    # quantization — the guarantee the residual exists to give.
    u_ref = reference_over_horizon(cfg, out, hmax, nm, ns)
    stretch = torch.tensor([ns[c]["stretch"] for c in nm["U"]])
    delta = ((out["U"][:, hmax:, :] - u_ref) * stretch).abs()
    assert float(delta.max()) <= cfg.residual_delta + 1e-4
    assert results.residual.max_abs_delta <= cfg.residual_delta + 1e-4

    # ...and the run answers the question the residual makes answerable.
    baseline = evaluate_baseline(
        cfg, sysid_cfg, s["dyn_node"], [s["norm_df"]], nm, ns,
    )
    assert clears_reference(results, baseline) in (True, False)
