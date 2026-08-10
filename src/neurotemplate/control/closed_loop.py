"""
Closed-loop assembly + DPC objective.

    SystemPreview([policy_node, frozen_dyn_node])

At rollout step i: the policy writes U (appended to data['U']); the frozen
dynamics consumes a U/D/yn window ending at i and writes the next yn (fed
back). With start_iter = Hmax-1 and explicit nsteps = P, predicted yn occupies
indices [Hmax : Hmax+P], aligned with the comfort-band signal slice
[Hmax : Hmax+P].

A stateful surrogate needs its hidden state cleared between batches, which is
all init_func does here — it warms itself from the same H_sys-wide window a
stateless surrogate reads, so the two enter the rollout on equal footing.
"""

from __future__ import annotations


import torch
from torch.nn.functional import relu

from neuromancer.system import SystemPreview
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss

from neurotemplate.control.scenarios import horizons
from neurotemplate.modeling.dynamics import make_reset_init
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


@beartype
def build_closed_loop(dyn_node, policy_node, cfg, sysid_cfg) -> SystemPreview:
    """Policy + frozen dynamics, rolled out for cfg.train_pred_steps steps."""
    _, _, hmax = horizons(cfg, sysid_cfg)

    return SystemPreview(
        [policy_node, dyn_node],
        name="closed_loop",
        nstep_key="D",
        nsteps=cfg.train_pred_steps,
        start_iter=hmax - 1,
        # Stateful surrogates only: clear the hidden state so it cannot leak
        # between batches. None for stateless ones.
        init_func=make_reset_init(dyn_node.callable),
    )


@beartype
def resolve_du_weights(du_weight, u_cols: list[str]) -> list[float]:
    """Broadcast `cfg.du_weight` to one weight per control column.

    A scalar applies to every actuator, which is the old behaviour exactly.
    A list must be one entry per U column, in `nm_mapping["U"]` order; a dict
    is keyed by column name and any column left out gets 0 (no smoothing).

    Per-channel weights matter as soon as two actuators have different
    authority: on a plant where one moves the controlled channel 5x further
    per step than the other, a single pooled penalty is necessarily
    mis-weighted for one of them.
    """
    if isinstance(du_weight, (int, float)):
        return [float(du_weight)] * len(u_cols)
    if isinstance(du_weight, dict):
        unknown = [c for c in du_weight if c not in u_cols]
        if unknown:
            raise ValueError(
                f"du_weight names columns that are not controls: {unknown}. "
                f"U = {u_cols}"
            )
        return [float(du_weight.get(c, 0.0)) for c in u_cols]
    if len(du_weight) != len(u_cols):
        raise ValueError(
            f"du_weight has {len(du_weight)} entries but there are "
            f"{len(u_cols)} control columns ({u_cols})"
        )
    return [float(w) for w in du_weight]


@beartype
def power_term(cfg, nm_mapping: dict[str, list[str]],
               norm_stats: dict[str, dict[str, float]]):
    """The effort term as a plain callable, or None when unimplemented.

    Signature is `(yn, U, D) -> [batch, time]` over the *scored horizon*
    slices. Shared with `control.objective` so the shape check sweeps exactly
    the quantity the loss minimizes, not a re-derivation of it.
    """
    hooks = load_control_hooks(cfg)
    if not getattr(hooks, "POWER_METRIC_IMPLEMENTED", False):
        return None

    def term(yn, u, d):
        return hooks.power_metric(yn, u, d, nm_mapping, norm_stats)

    return term


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

    Nothing here checks that a weight suits the magnitude of what it
    multiplies — that is `control.objective`, run once on a real batch before
    training, because it takes a forward pass to know.
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

    du_weights = resolve_du_weights(cfg.du_weight, nm_mapping["U"])
    if any(w > 0 for w in du_weights):
        u_h = u[:, hmax:, :]            # policy-produced control [b, P, nu]
        du_abs = torch.abs(u_h[:, 1:, :] - u_h[:, :-1, :])
        # Weighted per channel, then averaged over channels — so a scalar
        # weight reproduces the pooled mean exactly.
        du = sum(w * torch.mean(du_abs[:, :, j])
                 for j, w in enumerate(du_weights)) / len(du_weights)
        du_loss = du == 0
        du_loss.update_name("du_loss")
        objectives.append(du_loss)

    if cfg.effort_weight > 0:
        term = power_term(cfg, nm_mapping, norm_stats)
        if term is None:
            raise NotImplementedError(
                "effort_weight > 0 requires control_user_defined.power_metric "
                "to be implemented (set POWER_METRIC_IMPLEMENTED=True). Run "
                "with --set effort_weight=0 until a power model exists."
            )

        def _power(yn_v, u_v, d_v):
            return term(
                yn_v[:, hmax:, :], u_v[:, hmax:, :], d_v[:, hmax:, :],
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
