"""
Closed-loop assembly + DPC objective.

    SystemPreview([policy_node, frozen_dyn_node])

At rollout step i: the policy writes U[i]; the frozen dynamics consumes a
U/D/yn window ending at i, so it applies that fresh command, and writes
yn[i+1] (fed back). With start_iter = Hmax-1 and explicit nsteps = P,
predicted yn occupies indices [Hmax : Hmax+P], aligned with the reference
columns' slice [Hmax : Hmax+P], and the commands that produced it occupy
[Hmax-1 : Hmax-1+P]. `windows.scored_steps` is that alignment, in one place.

A stateful surrogate needs its hidden state cleared between batches, which is
all init_func does here — it warms itself from the same H_sys-wide window a
stateless surrogate reads, so the two enter the rollout on equal footing.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from neurotemplate.nm.system import SystemPreview
from neurotemplate.nm.problem import Problem
from neurotemplate.nm.loss import Loss, PenaltyLoss

from neurotemplate.control.losses import (
    POWER, LossTerm, apply_map, resolve_objective, window_power_mean,
)
from neurotemplate.control.windows import horizons, scored_series, scored_steps
from neurotemplate.modeling.dynamics import make_reset_init
from neurotemplate.modeling.rollout import WindowedSystem
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


class ForecastSystem(WindowedSystem):
    """A WindowedSystem whose forecast keys are trimmed to the rollout after it runs.

    D, t and R arrive `forecast_tail` rows longer than the rollout so the policy's
    preview reads real rows to the end (see `windows`). Every loss, metric and plot
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
    """Policy + frozen dynamics, rolled out over `cfg.horizon`."""
    _, _, hmax = horizons(cfg, sysid_cfg)

    return ForecastSystem(
        [policy_node, dyn_node],
        name="closed_loop",
        nstep_key="D",
        nsteps=cfg.horizon_steps,
        start_iter=hmax - 1,
        # Stateful surrogates and controllers (MPC's last plan) only: clear
        # the state so it cannot leak between batches. None for stateless ones.
        init_func=make_reset_init(nn.ModuleList([dyn_node.callable, policy_node.callable])),
    )


@beartype
def power_term(cfg, nm_mapping: dict[str, list[str]]):
    """The per-step power signal as a plain callable, or None when no hook says what it
    is: `power_metric`, or else the sum of the `power_channels` the surrogate predicts.

    Signature is `(yn, U, D) -> [batch, time]` over the *scored horizon*
    slices, physical units. Shared with `metrics.objective` so the shape check
    sweeps exactly the quantity the loss minimizes, not a re-derivation of it.
    """
    hooks = load_control_hooks(cfg)
    if hasattr(hooks, "power_metric"):
        return lambda yn, u, d: hooks.power_metric(yn, u, d, nm_mapping)
    channels = power_channels(hooks, nm_mapping)
    if not channels:
        return None
    indices = [nm_mapping["Y"].index(c) for c in channels]
    return lambda yn, u, d: sum(yn[:, :, i] for i in indices)


def power_channels(hooks, nm_mapping: dict[str, list[str]]) -> list[str]:
    """The Y columns that meter power, in plot order; none without the hook."""
    return hooks.power_channels(nm_mapping) if hasattr(hooks, "power_channels") else []


@beartype
def build_control_loss(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
) -> PenaltyLoss:
    """One scalar term per entry of `cfg.losses`, scaled by its gain: the
    batch mean of `build_window_objective`'s terms.

    Each term reads the scored horizon of the rollout — the P steps after the
    warm-up, with U read one index earlier (see `windows.scored_steps`) —
    and is reduced by `losses.window_power_mean`, so a gain tuned at P=48 means the
    same thing at P=288. Every series is in physical units.

    Every term that names `power` evaluates the same `power_metric` hook on
    the same slices, so a cost cannot price a different power signal than the
    energy term measures. Naming `power` without the hook implemented fails
    here rather than substituting a proxy.

    Nothing here checks that a gain suits the magnitude of what it multiplies
    — that is `metrics.objective`, run once on a real batch before training,
    because it takes a forward pass to know.
    """
    objective, keys, term_value = _term_values(cfg, sysid_cfg, nm_mapping)

    def term_loss(term: LossTerm):
        def loss(*tensors):
            return term_value(term, dict(zip(keys, tensors))).mean()
        return loss

    return PenaltyLoss(
        [Loss(keys, term_loss(term), weight=term.gain, name=term.key)
         for term in objective.terms],
        constraints=[],
    )


@beartype
def build_window_objective(cfg, sysid_cfg, nm_mapping: dict[str, list[str]]):
    """The run's objective per window, `rollout -> [batch]`: each term's
    window power mean times its gain, summed. Its batch mean is
    `build_control_loss`'s total. A planner needs it per window: each window
    is its own problem, and stops and keeps its best on its own."""
    objective, keys, term_value = _term_values(cfg, sysid_cfg, nm_mapping)

    def window_objective(data: dict) -> torch.Tensor:
        return sum(term.gain * term_value(term, data) for term in objective.terms)
    return window_objective


def _term_values(cfg, sysid_cfg, nm_mapping: dict[str, list[str]]):
    """(objective, the rollout keys it reads, `term_value(term, data) ->
    [batch]`, one term's power mean per window, before its gain)."""
    objective = resolve_objective(cfg, nm_mapping)
    _, _, hmax = horizons(cfg, sysid_cfg)
    keys = ["yn", "U", "D"] + (["R"] if objective.r_columns else [])

    power = power_term(cfg, nm_mapping)
    if objective.reads_power and power is None:
        names = [t.name for t in objective.terms
                 if POWER in (t.column, t.target)]
        raise NotImplementedError(
            f"losses {names} read `{POWER}`, and nothing says what it is: define "
            "`power_channels` in nt_control.py, or `power_metric` for a proxy "
            "(`nt docs power_channels`)."
        )

    def series(name: str, data: dict) -> torch.Tensor:
        """`name` over the scored horizon, [batch, P]."""
        if name == POWER:
            return power(*(data[k][:, scored_steps(k, hmax), :]
                           for k in ("yn", "U", "D")))
        return scored_series(objective, name, data, hmax, nm_mapping)

    def term_value(term: LossTerm, data: dict) -> torch.Tensor:
        h = series(term.column, data)
        if term.target is not None:
            h = h - series(term.target, data)
        h = apply_map(h, term.map)
        weight = None
        if term.weight is not None:
            weight = series(term.weight, data)
            if term.map == "first_difference":
                weight = weight[:, 1:]
        return window_power_mean(h, term.norm, weight)

    return objective, keys, term_value


@beartype
def build_problem(cl_system: SystemPreview, loss: PenaltyLoss) -> Problem:
    return Problem([cl_system], loss)
