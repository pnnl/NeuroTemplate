"""
Loss construction for the physics-informed training pipeline.

Builds a single Neuromancer PenaltyLoss containing:
  1. Tracking objectives (ref, velocity, sign) — standard Variable expressions
  2. Sampled gradient sign penalty — Neuromancer Loss

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

from neurotemplate._typing import beartype


@beartype
def build_loss(
    cfg: ModelingConfig,
    raw_model: torch.nn.Module,
    system: torch.nn.Module,
    grad_masks: dict[str, torch.Tensor],
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
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
    # to align predictions with ground truth. The leading in_steps positions
    # of yn are the seeded history — copies of Y that contribute exactly zero
    # error and would only dilute the mean — so both sides drop them. This is
    # the same slice pipeline.evaluate_and_plot uses, which keeps the training
    # loss and the reported test metric measuring the same span.

    y = variable("Y")[:, cfg.in_steps:, :]
    yhat = variable("yn")[:, cfg.in_steps:-1, :]

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
            return head_stack.loss(
                yn[:, cfg.in_steps:-1, :], Y[:, cfg.in_steps:, :]
            )

        ref_loss = Loss(
            input_keys=["yn", "Y"],
            loss=head_compute,
            weight=cfg.ref_gain,
            name="ref_loss",
        )

    objectives = [ref_loss]

    # Finite differences over the prediction span only — y/yhat already have
    # the seeded history sliced off. Symbolic Variable expressions, so this is
    # cheap to build even when neither cfg.vel_gain nor cfg.sgn_gain is set;
    # both the velocity and sign-agreement terms below read from it.
    v = y[:, 1:, :] - y[:, :-1, :]
    vhat = yhat[:, 1:, :] - yhat[:, :-1, :]

    # Velocity tracking: R2 style on finite differences
    if cfg.vel_gain > 0:
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

    # The physics scenarios (modeling/scenarios.py) deliberately contribute
    # nothing here. They are assertions run against the trained surrogate, not
    # a training signal — see that module's header. Physicality *is* enforced
    # during training, by the gradient sign penalty above.

    return PenaltyLoss(objectives, constraints=[])
