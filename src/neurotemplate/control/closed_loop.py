"""
Closed-loop assembly + DPC objective.

    SystemPreview([policy_node, frozen_dyn_node])

At rollout step i: the policy writes U[i]; the frozen dynamics consumes a
U/D/yn window ending at i, so it applies that fresh command, and writes
yn[i+1] (fed back). With start_iter = Hmax-1 and explicit nsteps = P,
predicted yn occupies indices [Hmax : Hmax+P], aligned with the reference
columns' slice [Hmax : Hmax+P], and the commands that produced it occupy
[Hmax-1 : Hmax-1+P]. `scenarios.scored_steps` is that alignment, in one place.

A stateful surrogate needs its hidden state cleared between batches, which is
all init_func does here — it warms itself from the same H_sys-wide window a
stateless surrogate reads, so the two enter the rollout on equal footing.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from neuromancer.system import SystemPreview
from neuromancer.problem import Problem
from neuromancer.constraint import Loss
from neuromancer.loss import PenaltyLoss

from neurotemplate.control.losses import (
    POWER, LossTerm, apply_map, power_mean, resolve_objective,
)
from neurotemplate.control.scenarios import horizons, scored_series, scored_steps
from neurotemplate.modeling.dynamics import make_reset_init
from neurotemplate.modeling.rollout import WindowedSystem
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


class ForecastSystem(WindowedSystem):
    """A WindowedSystem whose forecast keys are trimmed to the rollout after it runs.

    D, t and R arrive `forecast_tail` rows longer than the rollout so the policy's
    preview reads real rows to the end (see `scenarios`). Every loss, metric and plot
    slices them with `scored_steps`, which runs to the end of the key, so they are cut
    back to yn's length here, once.
    """

    def forward(self, input_dict):
        out = super().forward(input_dict)
        end = out["yn"].shape[1]
        for key in ("D", "t", "R"):
            if key in out:
                out[key] = out[key][:, :end]
        return out


@beartype
def build_closed_loop(dyn_node, policy_node, cfg, sysid_cfg) -> SystemPreview:
    """Policy + frozen dynamics, rolled out for cfg.train_pred_steps steps."""
    _, _, hmax = horizons(cfg, sysid_cfg)

    return ForecastSystem(
        [policy_node, dyn_node],
        name="closed_loop",
        nstep_key="D",
        nsteps=cfg.train_pred_steps,
        start_iter=hmax - 1,
        # Stateful surrogates and controllers (MPC's last plan) only: clear
        # the state so it cannot leak between batches. None for stateless ones.
        init_func=make_reset_init(nn.ModuleList([dyn_node.callable, policy_node.callable])),
    )


@beartype
def power_term(cfg, nm_mapping: dict[str, list[str]]):
    """The per-step power signal as a plain callable, or None when unimplemented.

    Signature is `(yn, U, D) -> [batch, time]` over the *scored horizon*
    slices, physical units. Shared with `control.objective` so the shape check
    sweeps exactly the quantity the loss minimizes, not a re-derivation of it.
    """
    hooks = load_control_hooks(cfg)
    if not getattr(hooks, "POWER_METRIC_IMPLEMENTED", False):
        return None

    def term(yn, u, d):
        return hooks.power_metric(yn, u, d, nm_mapping)

    return term


@beartype
def build_control_loss(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
) -> PenaltyLoss:
    """One scalar term per entry of `cfg.losses`, scaled by its gain.

    Each term reads the scored horizon of the rollout — the P steps after the
    warm-up, with U read one index earlier (see `scenarios.scored_steps`) —
    and is reduced by `losses.power_mean`, so a gain tuned at P=48 means the
    same thing at P=288. Every series is in physical units.

    Every term that names `power` evaluates the same `power_metric` hook on
    the same slices, so a cost cannot price a different power signal than the
    energy term measures. Naming `power` without the hook implemented fails
    here rather than substituting a proxy.

    Nothing here checks that a gain suits the magnitude of what it multiplies
    — that is `control.objective`, run once on a real batch before training,
    because it takes a forward pass to know.
    """
    objective = resolve_objective(cfg, nm_mapping)
    _, _, hmax = horizons(cfg, sysid_cfg)
    keys = ["yn", "U", "D"] + (["R"] if objective.r_columns else [])

    power = power_term(cfg, nm_mapping)
    if objective.reads_power and power is None:
        names = [t.name for t in objective.terms
                 if POWER in (t.column, t.target)]
        raise NotImplementedError(
            f"losses {names} read `{POWER}`, which needs "
            "control_user_defined.power_metric implemented (set "
            "POWER_METRIC_IMPLEMENTED = True)."
        )

    def series(name: str, data: dict) -> torch.Tensor:
        """`name` over the scored horizon, [batch, P]."""
        if name == POWER:
            return power(*(data[k][:, scored_steps(k, hmax), :]
                           for k in ("yn", "U", "D")))
        return scored_series(objective, name, data, hmax, nm_mapping)

    def term_loss(term: LossTerm):
        def loss(*tensors):
            data = dict(zip(keys, tensors))
            h = series(term.column, data)
            if term.target is not None:
                h = h - series(term.target, data)
            h = apply_map(h, term.map)
            weight = None
            if term.weight is not None:
                weight = series(term.weight, data)
                if term.map == "first_difference":
                    weight = weight[:, 1:]
            return power_mean(h, term.norm, weight)
        return loss

    return PenaltyLoss(
        [Loss(keys, term_loss(term), weight=term.gain, name=term.key)
         for term in objective.terms],
        constraints=[],
    )


@beartype
def build_problem(cl_system: SystemPreview, loss: PenaltyLoss) -> Problem:
    return Problem([cl_system], loss)
