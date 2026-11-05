"""Planning through the surrogate: the best open-loop plan, no worse than the
policy it starts from, and MPC, which re-plans at every step. Both inside the
bounds a policy is held to."""

import torch

from neurotemplate.control.closed_loop import build_closed_loop, build_control_loss
from neurotemplate.control.controller import build_controller
from neurotemplate.control.losses import resolve_objective
from neurotemplate.control.pipeline import evaluate_closed_loop, evaluation_windows
from neurotemplate.control.plan import plan_on_surrogate
from neurotemplate.control.policy import build_policy
from neurotemplate.control.windows import horizons, scored_steps
from control_fixtures import HIGH, LOW, SETPT, full_setup, make_nm_mapping, make_tracking_plant
from model_fixtures import steps


def test_the_plan_improves_on_the_policy_inside_its_bounds():
    torch.manual_seed(0)   # the fixture surrogate's weights
    s = full_setup(control_bounds={SETPT: [LOW, [HIGH, -0.5]]}, solver={"max_iters": 100},
                   test_horizon=steps(26), test_stride=steps(30))
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    torch.manual_seed(0)

    _, out = evaluate_closed_loop(cfg, sysid_cfg, cl, [s["frame"]], nm, ns)
    plan = plan_on_surrogate(cfg, sysid_cfg, cl, policy, [s["frame"]], out, nm, ns)

    assert plan.plan.total < plan.policy.total
    assert 0 < plan.gap < 1
    assert plan.converged and plan.starts == 4
    # The plan's commands obey control_bounds at every step, as the policy's do.
    _, _, hmax = horizons(cfg, sysid_cfg)
    r = resolve_objective(cfg, nm).r_columns
    u = plan.out["U"][:, scored_steps("U", hmax), 0]
    low, high = (plan.out["R"][:, hmax - 1:hmax - 1 + u.shape[1], r.index(c)] for c in (LOW, HIGH))
    assert (u >= low - 1e-4).all() and (u <= high - 0.5 + 1e-4).all()


def test_mpc_beats_an_untrained_policy_inside_its_bounds_and_starts_each_rollout_cold():
    future = {c: 4 for c in (LOW, HIGH, *make_nm_mapping()["D"])}
    s = full_setup(control_bounds={SETPT: [LOW, [HIGH, -0.5]]}, future=future,
                   solver={"max_iters": 200})
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, plant = make_tracking_plant(sysid_cfg, nm)   # a zone that follows its setpoint
    data, _ = evaluation_windows(cfg, sysid_cfg, [s["frame"]], nm)
    loss = build_control_loss(cfg, sysid_cfg, nm)

    def objective(policy):
        cl = build_closed_loop(plant, policy, cfg, sysid_cfg)
        cl.nsteps = cfg.test_horizon_steps
        with torch.no_grad():
            out = cl(dict(data))
        return float(loss(out)["loss"]), out, cl

    torch.manual_seed(0)
    untrained, _, _ = objective(build_policy(cfg, sysid_cfg, nm, ns))
    cfg.policy, cfg.horizon = "mpc", steps(4)
    mpc = build_controller(cfg, sysid_cfg, nm, ns, plant)
    planned, out, cl = objective(mpc)
    assert planned < untrained
    # Every solve stopped on patience, before the cap.
    assert max(mpc.callable.solves) < cfg.solver["max_iters"]

    _, _, hmax = horizons(cfg, sysid_cfg)
    r = resolve_objective(cfg, nm).r_columns
    u = out["U"][:, scored_steps("U", hmax), 0]
    low, high = (out["R"][:, hmax - 1:hmax - 1 + u.shape[1], r.index(c)] for c in (LOW, HIGH))
    assert (u >= low - 1e-4).all() and (u <= high - 0.5 + 1e-4).all()

    # The last plan is rollout state: a second rollout on the same windows
    # starts cold and commands the same.
    with torch.no_grad():
        again = cl(dict(data))
    assert torch.allclose(again["U"], out["U"])
