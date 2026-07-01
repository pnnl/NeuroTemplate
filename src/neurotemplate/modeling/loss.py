"""
Loss construction for the physics-informed training pipeline.

Builds a single Neuromancer PenaltyLoss containing:
  1. Tracking objectives (ref, velocity, sign) — standard Variable expressions
  2. Sampled gradient sign penalty — Neuromancer Loss
  3. Synthetic scenario penalty — Neuromancer Loss

All loss terms are proper Neuromancer objectives so they flow through
Problem.forward() and show up with correct labels in the Trainer's
output dict. Physics losses are active in both train and eval so that
dev_loss (and early stopping) accounts for physics compliance.
"""

import warnings

import torch
from torch.nn.functional import softplus

from neurotemplate.modeling.config import ModelingConfig
from neuromancer.constraint import variable, Loss
from neuromancer.loss import PenaltyLoss

from typing import Optional

from neurotemplate.modeling.gradient_sampling import SampledGradientPenalty
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.scenarios import build_scenarios

from neurotemplate._typing import beartype


@beartype
def build_loss(
    cfg: ModelingConfig,
    raw_model: torch.nn.Module,
    system: torch.nn.Module,
    grad_masks: dict[str, torch.Tensor],
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    synth_cfg: Optional[dict[str, dict]],
    head_stack: Optional[HeadStack] = None,
) -> PenaltyLoss:
    """
    Build the complete PenaltyLoss with tracking + physics objectives.

    Args:
        cfg:          ModelingConfig with loss gains and physics parameters
        raw_model:    SSMGrad instance (the raw dynamics module inside the System)
        system:       Neuromancer SystemPreview wrapping the dynamics model
        grad_masks:   {key: Tensor[1,1,ny,nx]} with sign constraints, or empty dict
        nm_mapping:   {"Y": [...], "U": [...], "D": [...]}
        norm_stats:   {col: {"shift", "stretch", "min", "max", ...}} per-column stats
        synth_cfg:    Per-column physical values for synth scenarios, or None
                      to indicate the synth subsystem is disabled for this run.

    Returns:
        PenaltyLoss combining all active loss terms
    """

    # ── Tracking losses (Neuromancer Variable expressions) ──────────
    #
    # These operate on the System's output dict:
    #   'Y'  = ground truth states from the data loader [batch, time, ny]
    #   'yn' = model predictions accumulated by System [batch, time+1, ny]
    #
    # yn has one extra timestep (the initial condition), so we slice :-1
    # to align predictions with ground truth.

    y = variable("Y")
    yhat = variable("yn")[:, :-1, :]

    # Reference tracking
    #
    # Without heads (or with all-regression heads), this is the original
    # blanket MSE — preserved as a Variable expression so it still shows
    # up labeled "ref_loss" in the Trainer output dict.
    #
    # With non-trivial heads, the per-channel loss is computed via the
    # head_stack callable so each channel uses its own activation/loss
    # pairing (e.g. BCE for binary). The result is wrapped in a Loss so
    # it still appears as a labeled objective alongside the others.
    if head_stack is None:
        mse = torch.mean((yhat - y) ** 2)
        ref_loss = cfg.ref_gain * mse == 0
        ref_loss.update_name("ref_loss")
    else:
        def head_compute(yn, Y):
            return head_stack.loss(yn[:, :-1, :], Y)

        ref_loss = Loss(
            input_keys=["yn", "Y"],
            loss=head_compute,
            weight=cfg.ref_gain,
            name="ref_loss",
        )

    objectives = [ref_loss]

    # Velocity tracking: R2 style on finite differences
    if cfg.vel_gain > 0:
        # Finite differences for velocity tracking
        v = y[:, 1:, :] - y[:, :-1, :]
        vhat = yhat[:, 1:, :] - yhat[:, :-1, :]
        vel_sum_res = torch.mean((vhat - v) ** 2)
        # vel_sum_tot = torch.sum((v - torch.mean(v)) ** 2)
        vel_loss = cfg.vel_gain * vel_sum_res == 0
        vel_loss.update_name("vel_loss")
        objectives.append(vel_loss)

    # Sign agreement: penalises predicted velocity with wrong sign
    if cfg.sgn_gain > 0:
        sgn_loss = (
            cfg.sgn_gain * softplus(-v * vhat, beta=cfg.sgn_steepness) 
            == 
            torch.zeros_like(v)
        )
        sgn_loss.update_name("sgn_loss")
        objectives.append(sgn_loss)

    # ── Sampled gradient sign penalty ───────────────────────────────
    #
    # Samples random points in [0,1] input space, computes Jacobians,
    # and penalizes wrong-sign entries. Decoupled from System rollout.
    #
    # Uses Neuromancer's Loss class with input_keys=[] since it doesn't
    # read from the data dict — it generates its own synthetic inputs.

    if grad_masks and cfg.grad_gain > 0:
        # Linear models have a constant Jacobian; sampling more than one input
        # point is wasted compute.
        n_samples = cfg.grad_n_samples
        if cfg.model == "lssm" and n_samples != 1:
            warnings.warn(
                f"Linear model (lssm) has a constant Jacobian; pinning "
                f"grad_n_samples to 1 (was {n_samples}).",
                stacklevel=2,
            )
            n_samples = 1

        penalty = SampledGradientPenalty(
            model=raw_model,
            grad_mapping=grad_masks,
            n_samples=n_samples,
            device=cfg.device,
            cfg=cfg,
        )

        def grad_compute():
            if not system.training:
                return torch.tensor(0.0, device=cfg.device)
            return penalty.compute()

        grad_loss = Loss(
            input_keys=[],
            loss=grad_compute,
            weight=cfg.grad_gain,
            name="grad_penalty",
        )
        objectives.append(grad_loss)

    # ── Synthetic scenario penalties ────────────────────────────────
    #
    # Built by synthetic_scenarios.build_scenarios from the user-provided
    # synth_cfg. Each scenario contributes a differentiable scalar penalty
    # summed into one `synth_penalty` term. All share synth_gain /
    # synth_n_scenarios / synth_n_steps. See synthetic_scenarios.py for
    # the scenario catalog.

    if cfg.synth_gain > 0:
        # synth_gain > 0 with no synth_cfg is a configuration mistake; fail fast.
        if synth_cfg is None:
            raise ValueError(
                "cfg.synth_gain > 0 but synth_cfg is None. Either set "
                "synth_gain=0 or have user_defined.synth_cfg return a populated dict."
            )
        scenarios = build_scenarios(
            system=system,
            nm_mapping=nm_mapping,
            norm_stats=norm_stats,
            synth_cfg=synth_cfg,
            in_steps=cfg.in_steps,
            n_scenarios=cfg.synth_n_scenarios,
            n_steps=cfg.synth_n_steps,
            device=cfg.device,
        )

        def synth_compute():
            if not system.training:
                return torch.tensor(0.0, device=cfg.device)
            total = torch.tensor(0.0, device=cfg.device)
            for s in scenarios:
                total = total + s.compute()
            return total

        synth_loss = Loss(
            input_keys=[],
            loss=synth_compute,
            weight=cfg.synth_gain,
            name="synth_penalty",
        )
        objectives.append(synth_loss)

    return PenaltyLoss(objectives, constraints=[])
