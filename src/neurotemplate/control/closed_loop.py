"""
Closed-loop assembly + DPC objective.

    SystemPreview([policy_node, frozen_dyn_node])

At rollout step i: the policy writes U (appended to data['U']); the frozen
dynamics consumes a U/D/yn window ending at i and writes the next yn (fed
back). With start_iter = Hmax-1 and explicit nsteps = P, predicted yn occupies
indices [Hmax : Hmax+P], aligned with the comfort-band signal slice
[Hmax : Hmax+P].
"""

from __future__ import annotations


import torch
from torch.nn.functional import relu

from neuromancer.system import SystemPreview
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss

from neurotemplate.control.scenarios import horizons
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


@beartype
def build_closed_loop(dyn_node, policy_node, cfg, sysid_cfg) -> SystemPreview:
    """Policy + frozen dynamics, rolled out for cfg.train_pred_steps steps."""
    _, _, hmax = horizons(cfg, sysid_cfg)

    cl = SystemPreview(
        [policy_node, dyn_node],
        name="closed_loop",
        nstep_key="D",
        nsteps=cfg.train_pred_steps,
        start_iter=hmax - 1,
    )

    # Match src/models.get_dynamics: stateful GRU surrogates must reset their
    # hidden state at the start of each batch.
    raw = dyn_node.callable
    if hasattr(raw, "fy") and hasattr(raw.fy, "reset"):
        cl.register_forward_pre_hook(lambda m, inp: raw.fy.reset())

    return cl


@beartype
def build_control_loss(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> PenaltyLoss:
    """
    Build all control losses as scalar penalties wrapped in (expr == 0)
    Constraints so they show up under their `update_name` labels in the
    Problem output dict (Objective.minimize(name=...) does NOT propagate to
    the output keys — only Constraint.update_name does).

    All terms go in PenaltyLoss.objectives; constraints=[] is intentional.
    The effort term needs control_user_defined.power_metric; if requested
    (effort_weight > 0) but unimplemented, fail fast.
    """
    hooks = load_control_hooks(cfg)
    _, _, hmax = horizons(cfg, sysid_cfg)
    _, ctrl_idx = hooks.controlled_channels(nm_mapping)

    yn = variable("yn")
    ymin = variable("ymin")
    ymax = variable("ymax")
    u = variable("U")
    d = variable("D")

    # Predicted controlled channels over the horizon vs. the band.
    y_ctrl = yn[:, hmax:, ctrl_idx]      # [b, P, n_ctrl]
    ymin_h = ymin[:, hmax:, :]           # [b, P, n_ctrl]
    ymax_h = ymax[:, hmax:, :]

    objectives = []

    if cfg.comfort_weight > 0:
        lower_violation = torch.mean(relu(ymin_h - y_ctrl)**2)
        comfort_lower_loss = cfg.comfort_weight * lower_violation == 0
        comfort_lower_loss.update_name("comfort_lower_loss")

        upper_violation = torch.mean(relu(y_ctrl - ymax_h)**2)
        comfort_upper_loss = cfg.comfort_weight * upper_violation == 0
        comfort_upper_loss.update_name("comfort_upper_loss")

        objectives += [comfort_lower_loss, comfort_upper_loss]

    if cfg.du_weight > 0:
        u_h = u[:, hmax:, :]            # policy-produced control [b, P, nu]
        du = torch.mean((u_h[:, 1:, :] - u_h[:, :-1, :]) ** 2)
        du_loss = cfg.du_weight * du == 0
        du_loss.update_name("du_loss")
        objectives.append(du_loss)

    if cfg.effort_weight > 0:
        if not getattr(hooks, "POWER_METRIC_IMPLEMENTED", False):
            raise NotImplementedError(
                "effort_weight > 0 requires control_user_defined.power_metric "
                "to be implemented (set POWER_METRIC_IMPLEMENTED=True). Run "
                "with --set effort_weight=0 until a power model exists."
            )

        def _power(yn_v, u_v, d_v):
            return hooks.power_metric(
                yn_v[:, hmax:, :], u_v[:, hmax:, :], d_v[:, hmax:, :],
                nm_mapping, norm_stats,
            )

        power = variable([yn, u, d], _power, display_name="power")
        mean_power = torch.mean(power)
        effort_loss = cfg.effort_weight * mean_power == 0
        effort_loss.update_name("effort_loss")
        objectives.append(effort_loss)

    if not objectives:
        raise ValueError(
            "No active control objective. Set at least one of "
            "comfort_weight / du_weight / effort_weight > 0."
        )

    return PenaltyLoss(objectives, constraints=[])


@beartype
def build_problem(cl_system: SystemPreview, loss: PenaltyLoss) -> Problem:
    return Problem([cl_system], loss)
