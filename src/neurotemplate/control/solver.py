"""
The plan both planners optimise (`control.plan`, `control.mpc`), and how.

A plan is a free command per step, held as its share of the actuation range,
[batch, steps, nu] in [0, 1]. Adam steps the share, and every step is
clamped back into the box: projected gradient. (Held as σ(z) instead, a
command at a bound sits where σ' is ~0, so a step barely moves it even when
the gradient pulls it off: that read as converged.)

Each window of the batch is its own problem. The objective is summed over
windows for the gradient, so a window's steps do not depend on its
batch-mates (Adam's epsilon sees the gradient's scale), and each window
stops on its own, as early stopping does: once `patience` steps pass
without its objective improving by more than `rtol` of it. A stopped window
holds its best plan while the rest go on; the solve ends when every window
has stopped, or at `max_iters`. Every window ends at the best plan it saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


class SharePlan(nn.Module):
    """One command per rollout step, `umin + share · (umax − umin)`.

    The closed loop calls it once per step, with the policy's windows, which
    it does not read; `step` counts the calls and is reset before a rollout.
    """

    def __init__(self, umin: torch.Tensor, umax: torch.Tensor, share: torch.Tensor):
        super().__init__()
        self.register_buffer("umin", umin)
        self.register_buffer("umax", umax)
        self.share = nn.Parameter(share.detach().clamp(0, 1).clone())   # [b, steps, nu]
        self.step = 0

    def forward(self, *windows: torch.Tensor) -> torch.Tensor:
        k = min(self.step, self.share.shape[1] - 1)
        self.step += 1
        return self.umin + self.share[:, k] * (self.umax - self.umin)


def share_of(u: torch.Tensor, umin: torch.Tensor, umax: torch.Tensor) -> torch.Tensor:
    """Commands as their share of the actuation range, in [0, 1]."""
    return ((u - umin) / (umax - umin).clamp_min(1e-9)).clamp(0, 1)


@dataclass(frozen=True)
class Solve:
    steps: int              # Adam steps taken
    converged: bool         # every window stopped on patience, before max_iters
    best: torch.Tensor      # each window's best objective, [batch]


def solve(plan: SharePlan, window_objective: Callable[[], torch.Tensor], lr: float,
          rtol: float, patience: int, max_iters: int) -> Solve:
    """Optimise `plan` in place; see the module docstring. `window_objective`
    rolls the plan out and returns its objective per window, [batch]."""
    share = plan.share
    n = share.shape[0]
    optimizer = torch.optim.Adam([share], lr=lr)
    inf = torch.full((n,), torch.inf, device=share.device)
    best, reference = inf.clone(), inf.clone()   # best seen; last real improvement
    best_share = share.detach().clone()
    wait = torch.zeros(n, dtype=torch.long, device=share.device)
    stopped = torch.zeros(n, dtype=torch.bool, device=share.device)
    steps = 0
    while True:
        plan.step = 0
        values = window_objective()
        with torch.no_grad():
            v = values.detach()
            better = ~stopped & (v < best)
            best = torch.where(better, v, best)
            best_share[better] = share[better]
            improved = ~stopped & (torch.isinf(reference)
                                   | (v < reference - rtol * reference.abs()))
            reference = torch.where(improved, v, reference)
            wait = torch.where(improved, torch.zeros_like(wait), wait + 1)
            stopped |= wait >= patience
        if stopped.all() or steps == max_iters:
            break
        optimizer.zero_grad()
        values.sum().backward()
        optimizer.step()
        steps += 1
        with torch.no_grad():
            share.clamp_(0, 1)
            share[stopped] = best_share[stopped]
    with torch.no_grad():
        share.copy_(best_share)
    return Solve(steps=steps, converged=bool(stopped.all()), best=best)
