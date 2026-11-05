"""
The best open-loop plan on the surrogate, and the controller gap.

A free command per scored step, inside `control_bounds`, optimised through the
frozen surrogate against the run's
own objective (`control.solver`), on the very windows the policy is scored
on. With the model as the plant and the forecast perfect, nothing that
commands through the same bounds can do better on this surrogate, to the
optimiser's reach. It is in sample and sees the whole window ahead, so it is
a bar, not a controller.

The objective is not convex, so the plan is solved from several starts: the
controller's own commands, which keep the plan no worse than
the controller, and random ones. Each window keeps its best start. How far
the starts land apart says how much one start would have understated the
bar.

Read the policy against it:

  - the gap between them is the controller's: the policy's structure, inputs
    or training leave that much of the objective on the table;
  - what the plan itself does not do (pre-cool at night, ride the band) is
    the model's or the objective's: the surrogate does not make it pay.

It runs on demand, `nt control gap`, because it costs more than the training.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter as tic

import torch

from neuromancer.system import Node

from neurotemplate._typing import beartype
from neurotemplate.common.data_utils import NormStats
from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_window_objective,
)
from neurotemplate.control.objective import ObjectiveBalance, objective_balance
from neurotemplate.control.pipeline import (
    ControlResults, evaluation_windows, score_rollout,
)
from neurotemplate.control.policy import control_box, with_control_bounds
from neurotemplate.control.scenarios import eval_pred_steps, horizons
from neurotemplate.control.solver import SharePlan, share_of, solve


@dataclass(frozen=True)
class PlanResult:
    """The plan on the evaluation windows, beside the policy on the same ones."""
    results: ControlResults        # the plan's rollout, scored as the policy's is
    out: dict                      # ...and the rollout itself
    policy: ObjectiveBalance       # the run's objective, per term
    plan: ObjectiveBalance
    steps: int                     # Adam steps the solve took
    converged: bool                # every window stopped before plan_max_iters
    starts: int
    own_start_best: float          # share of windows the controller's start won
    start_spread: float            # median over windows of (worst − best) / |best| start
    time_s: float

    @property
    def gap(self) -> float:
        """How far below the policy's objective the plan's is, as a share of
        the policy's."""
        return (self.policy.total - self.plan.total) / self.policy.total


def _plan_loop(plan: SharePlan, cfg, sysid_cfg, nm_mapping, policy_node: Node, plant: Node,
               umin: torch.Tensor, umax: torch.Tensor, steps: int):
    """The closed loop with `plan` commanding, behind `control_bounds` as the
    policy is, and that wrapper (for `base_of`)."""
    bounded = with_control_bounds(plan, cfg, nm_mapping, list(policy_node.input_keys),
                                  policy_node.input_map, umin, umax)
    node = Node(bounded, policy_node.input_keys, ["U"], name="plan",
                input_map=policy_node.input_map)
    loop = build_closed_loop(plant, node, cfg, sysid_cfg)
    loop.nsteps = steps
    return loop, bounded


@beartype
def plan_on_surrogate(cfg, sysid_cfg, cl_system, policy_node: Node, test_dfs: list,
                      policy_out: dict, nm_mapping: dict[str, list[str]],
                      norm_stats: NormStats, starts: int = 4) -> PlanResult:
    """Solve the free plan from `starts` starts, keep each window's
    best, and score it as the policy is scored.

    `policy_out` is the policy's rollout on the same windows, from
    `evaluate_closed_loop`, with the commands it issued."""
    t0 = tic()
    data, source_idx = evaluation_windows(cfg, sysid_cfg, test_dfs, nm_mapping)
    steps = eval_pred_steps(cfg, horizons(cfg, sysid_cfg)[2])
    loss = build_control_loss(cfg, sysid_cfg, nm_mapping)
    window_objective = build_window_objective(cfg, sysid_cfg, nm_mapping)
    with torch.no_grad():
        policy_balance = objective_balance(loss, loss(policy_out))

    device = sysid_cfg.device
    umin, umax = control_box(cfg, nm_mapping, norm_stats, device)
    plant = cl_system.nodes[1]
    windows, nu = len(data["yn"]), len(umin)

    # Start 0 is the controller's own commands: behind control_bounds, at the
    # base commands the bounds map onto them. The rest are random, seeded.
    _, bounded = _plan_loop(SharePlan(umin, umax, torch.zeros(windows, steps, nu)),
                            cfg, sysid_cfg, nm_mapping, policy_node, plant, umin, umax, steps)
    own = torch.stack([bounded.base_of(u, *args) if hasattr(bounded, "base_of") else u
                       for args, u in policy_out["issued"]], dim=1)
    generator = torch.Generator().manual_seed(cfg.seed)
    first_shares = [share_of(own, umin, umax)] + [
        torch.rand(windows, steps, nu, generator=generator).to(device)
        for _ in range(starts - 1)]

    # Every start of every window is one problem of one batch, start-major.
    plan = SharePlan(umin, umax, torch.cat(first_shares))
    loop, _ = _plan_loop(plan, cfg, sysid_cfg, nm_mapping, policy_node, plant,
                         umin, umax, steps)
    batch = {key: x.repeat(starts, *[1] * (x.dim() - 1)) for key, x in data.items()}
    result = solve(plan, lambda: window_objective(loop(dict(batch))), cfg.plan_lr,
                   cfg.plan_rtol, cfg.plan_patience, cfg.plan_max_iters)

    by_start = result.best.reshape(starts, windows)
    winner = by_start.argmin(dim=0)
    best_share = plan.share.detach().reshape(starts, windows, steps, nu)[
        winner, torch.arange(windows, device=device)]
    spread = ((by_start.max(dim=0).values - by_start.min(dim=0).values)
              / by_start.min(dim=0).values.abs().clamp_min(1e-12))

    final, _ = _plan_loop(SharePlan(umin, umax, best_share), cfg, sysid_cfg, nm_mapping,
                          policy_node, plant, umin, umax, steps)
    with torch.no_grad():
        out = final(dict(data))
    out["source_idx"] = source_idx
    results = score_rollout(cfg, sysid_cfg, out, nm_mapping, norm_stats, eval_time_s=0.0)
    return PlanResult(results=results, out=out, policy=policy_balance,
                      plan=objective_balance(loss, loss(out)),
                      steps=result.steps, converged=result.converged,
                      starts=starts,
                      own_start_best=float((winner == 0).float().mean()),
                      start_spread=float(spread.median()), time_s=tic() - t0)
