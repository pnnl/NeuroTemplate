"""
Receding-horizon MPC through the frozen surrogate (`policy: mpc`).

At every step the controller solves a plan over the next `horizon`
steps (`control.solver`): a free command per step through its own copy of
the surrogate, inside `control_bounds`, against the
run's objective over the horizon. It starts from the previous step's plan,
shifted by one, and commands the first step. Nothing is trained: the
surrogate is the controller.

A window's solve stops once the `solver`'s `patience` Adam steps pass without its
objective improving by more than its `rtol` of it; its `max_iters` caps it,
so every solve ends, and ends the same on any machine. Every solve's step
count is kept (`solves`) for the reportcard. A deadline is the
deployment's: the exported runtime takes one per call.

It reads what a policy reads, the windows of `yn`, `U`, `D` and `R`, `hmax`
steps back and `horizon` ahead for `D` and `R`: those previews are its
forecast, so every D and reference column's `future` in `policy_window` must
cover the horizon. Every solve seeds the surrogate's state from the window,
as the closed loop does at its first step.

The previous plan is its rollout state (`rollout_state()`), cleared at every
sequence boundary like a surrogate's.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from neurotemplate.nm.system import Node

from neurotemplate._typing import beartype
from neurotemplate.common.data_utils import NormStats
from neurotemplate.control.closed_loop import build_closed_loop, build_window_objective
from neurotemplate.control.losses import resolve_objective
from neurotemplate.control.policy import control_box, with_control_bounds
from neurotemplate.control.windows import horizons
from neurotemplate.control.solver import SharePlan, share_of, solve


def _inner_keys(r_columns: list[str]) -> tuple[list[str], dict]:
    """The plan's inputs inside a solve: the current step of each key, which
    is all `control_bounds` reads."""
    keys = ["yn", "D"] + (["R"] if r_columns else [])
    return keys, {key: {"past": 0, "future": 0, "pad_mode": "nearest"} for key in keys}


class MPC(nn.Module):
    """The controller; see the module docstring."""

    def __init__(self, dyn_node: Node, cfg, sysid_cfg, nm_mapping: dict[str, list[str]],
                 umin: torch.Tensor, umax: torch.Tensor):
        super().__init__()
        self.cfg, self.sysid_cfg, self.nm_mapping = cfg, sysid_cfg, nm_mapping
        # Its own copy: a solve's rollouts must not touch the closed loop's
        # surrogate state.
        self.plant = copy.deepcopy(dyn_node)
        self.register_buffer("umin", umin)
        self.register_buffer("umax", umax)
        self.r_columns = resolve_objective(cfg, nm_mapping).r_columns
        self.keys = ["yn", "U", "D"] + (["R"] if self.r_columns else [])
        self.hmax = horizons(cfg, sysid_cfg)[2]
        self.horizon = cfg.horizon_steps
        self.lr, self.rtol = cfg.solver["lr"], cfg.solver["rtol"]
        self.patience, self.max_iters = cfg.solver["patience"], cfg.solver["max_iters"]
        self.objective = build_window_objective(cfg, sysid_cfg, nm_mapping)
        self.widths = {"yn": len(nm_mapping["Y"]), "U": len(nm_mapping["U"]),
                       "D": len(nm_mapping["D"]), "R": len(self.r_columns)}
        self.previous = None    # the last solve's plan, share [b, horizon, nu]
        self.solves: list[int] = []     # Adam steps each solve took

    def input_map(self) -> dict:
        """Each key `hmax` steps back; D and R `horizon` ahead."""
        return {key: {"past": self.hmax - 1,
                      "future": self.horizon if key in ("D", "R") else 0,
                      "pad_mode": "nearest"}
                for key in self.keys}

    def rollout_state(self):
        return None if self.previous is None else self.previous.flatten(1)

    def set_rollout_state(self, state):
        self.previous = None if state is None else state.reshape(
            state.shape[0], self.horizon, -1)

    def _windows(self, args) -> dict:
        """The node's flat windows as the closed loop's own keys: `hmax` rows
        of history (U without its padded current row), D and R ahead."""
        out = {}
        for key, x in zip(self.keys, args):
            window = x.reshape(x.shape[0], -1, self.widths[key])
            out[key] = window[:, :self.hmax - 1] if key == "U" else window
        return out

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        data = self._windows(args)
        if self.previous is None:   # cold: hold the last command applied
            start = share_of(data["U"][:, -1:], self.umin, self.umax).expand(
                -1, self.horizon, -1)
        else:
            start = torch.cat([self.previous[:, 1:], self.previous[:, -1:]], dim=1)
        plan = SharePlan(self.umin, self.umax, start)
        keys, input_map = _inner_keys(self.r_columns)
        node = Node(with_control_bounds(plan, self.cfg, self.nm_mapping, keys, input_map,
                                        self.umin, self.umax),
                    keys, ["U"], name="mpc plan", input_map=input_map)
        loop = build_closed_loop(self.plant, node, self.cfg, self.sysid_cfg)
        loop.nsteps = self.horizon
        with torch.enable_grad():
            result = solve(plan, lambda: self.objective(loop(dict(data))), self.lr,
                           self.rtol, self.patience, self.max_iters)
        self.solves.append(result.steps)
        plan.step = 0
        with torch.no_grad():
            first = loop(dict(data))["U"][:, self.hmax - 1]
        self.previous = plan.share.detach()
        return first


@beartype
def build_mpc(cfg, sysid_cfg, nm_mapping: dict[str, list[str]], norm_stats: NormStats,
              dyn_node: Node) -> Node:
    """The MPC controller as the closed loop's policy node."""
    short = [column for column, entry in cfg.window_steps.items()
             if "future" in entry and entry["future"] < cfg.horizon_steps]
    if short:
        raise ValueError(
            f"policy: mpc plans {cfg.horizon} ahead (`horizon`), so every D "
            f"and reference column's `future` in policy_window must reach that far: "
            f"{short} do not. The previews are the controller's forecast.")
    device = sysid_cfg.device
    umin, umax = control_box(cfg, nm_mapping, norm_stats, device)
    mpc = MPC(dyn_node, cfg, sysid_cfg, nm_mapping, umin, umax)
    return Node(mpc, mpc.keys, ["U"], name="MPC", input_map=mpc.input_map())
