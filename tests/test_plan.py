"""The best open-loop plan: no worse than the policy it starts from, and
inside the bounds a policy is held to."""

import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.plan import plan_on_surrogate
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import horizons, scored_steps
from neurotemplate.control.losses import resolve_objective
from control_fixtures import HIGH, LOW, SETPT, full_setup


def test_the_plan_improves_on_the_policy_inside_its_bounds():
    s = full_setup(command_bounds={SETPT: [LOW, [HIGH, -0.5]]}, plan_iters=30,
                   test_window=30, test_stride=30)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    torch.manual_seed(0)

    plan = plan_on_surrogate(cfg, sysid_cfg, cl, policy, [s["frame"]], nm, ns)

    assert plan.plan.total < plan.policy.total
    assert 0 < plan.gap < 1
    # The plan's commands obey command_bounds at every step, as the policy's do.
    _, _, hmax = horizons(cfg, sysid_cfg)
    r = resolve_objective(cfg, nm).r_columns
    u = plan.out["U"][:, scored_steps("U", hmax), 0]
    low, high = (plan.out["R"][:, hmax - 1:hmax - 1 + u.shape[1], r.index(c)] for c in (LOW, HIGH))
    assert (u >= low - 1e-4).all() and (u <= high - 0.5 + 1e-4).all()
