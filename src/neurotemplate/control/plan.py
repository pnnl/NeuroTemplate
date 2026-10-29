"""
The best open-loop plan on the surrogate, and the controller gap.

A free command per scored step, inside the actuation bounds and
`command_bounds`, optimised with Adam through the frozen surrogate against the
run's own objective, on the very windows the policy is scored on. With the
model as the plant and the forecast perfect, nothing that commands through
the same bounds can do better on this surrogate, to the optimiser's
convergence. It is in sample and sees the whole window ahead, so it is a bar,
not a controller.

Read the policy against it:

  - the gap between them is the controller's: the policy's structure, inputs
    or training leave that much of the objective on the table;
  - what the plan itself does not do (pre-cool at night, ride the band) is
    the model's or the objective's: the surrogate does not make it pay.

A residual box bounds the policy and not the plan, so with one the gap also
holds what the box costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter as tic

import torch
import torch.nn as nn

from neuromancer.system import Node

from neurotemplate._typing import beartype
from neurotemplate.common.data_utils import NormStats
from neurotemplate.control.closed_loop import build_closed_loop, build_control_loss
from neurotemplate.control.objective import ObjectiveBalance, objective_balance
from neurotemplate.control.pipeline import (
    ControlResults, evaluation_windows, score_rollout,
)
from neurotemplate.control.policy import _ReferenceBounds, with_command_bounds
from neurotemplate.control.scenarios import eval_pred_steps, horizons
from neurotemplate.hooks import load_control_hooks


class _FreePlan(nn.Module):
    """One command per rollout step, `umin + σ(z) · (umax − umin)`.

    The closed loop calls it once per step, with the policy's windows, which
    it does not read; `step` counts the calls and is reset before a rollout.
    """

    def __init__(self, umin: torch.Tensor, umax: torch.Tensor, start: torch.Tensor):
        super().__init__()
        self.register_buffer("umin", umin)
        self.register_buffer("umax", umax)
        share = ((start - umin) / (umax - umin).clamp_min(1e-9)).clamp(1e-3, 1 - 1e-3)
        self.z = nn.Parameter(torch.log(share / (1 - share)))     # [b, steps, nu]
        self.step = 0

    def forward(self, *windows: torch.Tensor) -> torch.Tensor:
        k = min(self.step, self.z.shape[1] - 1)
        self.step += 1
        return self.umin + torch.sigmoid(self.z[:, k]) * (self.umax - self.umin)


@dataclass(frozen=True)
class PlanResult:
    """The plan on the evaluation windows, beside the policy on the same ones."""
    results: ControlResults        # the plan's rollout, scored as the policy's is
    out: dict                      # ...and the rollout itself
    policy: ObjectiveBalance       # the run's objective, per term
    plan: ObjectiveBalance
    iterations: int
    time_s: float

    @property
    def gap(self) -> float:
        """How far below the policy's objective the plan's is, as a share of
        the policy's."""
        return (self.policy.total - self.plan.total) / self.policy.total


def _commands_before_bounds(cl_system, policy_node: Node, data: dict) -> torch.Tensor:
    """The policy's commands at every step of its rollout on `data`, before
    `command_bounds` maps them, [b, steps, nu]: the plan starts there."""
    policy = policy_node.callable
    source = policy.base if isinstance(policy, _ReferenceBounds) else policy
    seen = []
    handle = source.register_forward_hook(lambda _m, _args, out: seen.append(out.detach()))
    try:
        with torch.no_grad():
            cl_system(dict(data))
    finally:
        handle.remove()
    return torch.stack(seen, dim=1)


@beartype
def plan_on_surrogate(cfg, sysid_cfg, cl_system, policy_node: Node, test_dfs: list,
                      nm_mapping: dict[str, list[str]],
                      norm_stats: NormStats) -> PlanResult:
    """Optimise the free plan for `cfg.plan_iters` Adam steps at `cfg.plan_lr`,
    from the policy's own commands, and score it as the policy is scored."""
    t0 = tic()
    data, source_idx = evaluation_windows(cfg, sysid_cfg, test_dfs, nm_mapping)
    steps = eval_pred_steps(cfg, horizons(cfg, sysid_cfg)[2])
    cl_system.nsteps = steps
    loss = build_control_loss(cfg, sysid_cfg, nm_mapping)
    with torch.no_grad():
        policy_balance = objective_balance(loss, loss(cl_system(dict(data))))

    device = sysid_cfg.device
    umin, umax = (b.to(device=device, dtype=torch.float32)
                  for b in load_control_hooks(cfg).actuation_bounds(nm_mapping, norm_stats))
    plan = _FreePlan(umin, umax, _commands_before_bounds(cl_system, policy_node, data))
    node = Node(with_command_bounds(plan, cfg, nm_mapping, list(policy_node.input_keys),
                                    policy_node.input_map, umin, umax),
                policy_node.input_keys, ["U"], name="plan",
                input_map=policy_node.input_map)
    loop = build_closed_loop(cl_system.nodes[1], node, cfg, sysid_cfg)
    loop.nsteps = steps

    optimizer = torch.optim.Adam([plan.z], lr=cfg.plan_lr)
    for _ in range(cfg.plan_iters):
        plan.step = 0
        total = loss(loop(dict(data)))["loss"]
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    plan.step = 0
    with torch.no_grad():
        out = loop(dict(data))
    out["source_idx"] = source_idx
    results = score_rollout(cfg, sysid_cfg, out, nm_mapping, norm_stats,
                            eval_time_s=0.0, residual=False)
    return PlanResult(results=results, out=out, policy=policy_balance,
                      plan=objective_balance(loss, loss(out)),
                      iterations=cfg.plan_iters, time_s=tic() - t0)
