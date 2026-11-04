"""
Loss construction for the physics-informed training pipeline.

Builds a single Neuromancer PenaltyLoss containing:
  1. The tracking objective (ref_loss) — MSE, or each head's own loss
  2. Sampled gradient sign penalty — Neuromancer Loss

All loss terms are proper Neuromancer objectives so they flow through
Problem.forward() and show up with correct labels in the Trainer's
output dict. The gradient penalty is train-only: it samples fresh random
points on every call, so scoring it in eval would make dev_loss (and so
early stopping) noisy without measuring anything about the fit.
"""

from typing import Optional

import torch

from neuromancer.constraint import Loss
from neuromancer.loss import PenaltyLoss

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.contract import contract_model
from neurotemplate.modeling.gradient_sampling import SampledGradientPenalty, signed_model
from neurotemplate.modeling.heads import HeadStack

from neurotemplate._typing import beartype


@beartype
def build_loss(
    cfg: ModelingConfig,
    system: torch.nn.Module,
    grad_masks: dict[str, torch.Tensor],
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict],
    head_stack: Optional[HeadStack] = None,
) -> PenaltyLoss:
    """
    Build the complete PenaltyLoss with tracking + physics objectives.

    Args:
        cfg:          ModelingConfig with loss gains and physics parameters
        system:       Neuromancer SystemPreview wrapping the dynamics model
        grad_masks:   {key: Tensor[1,1,ny,nx]} with sign constraints, or empty dict
        nm_mapping, norm_stats: the run's; the tracking loss is scored with
                      each Y channel normalised by its stats, so channels in
                      different units weigh alike
        head_stack:   per-channel output heads in those normalised units, or
                      None for blanket MSE

    Returns:
        PenaltyLoss combining all active loss terms
    """

    # ── Tracking loss ───────────────────────────────────────────────
    #
    #   'Y'  = ground truth from the data loader [batch, time, ny]
    #   'yn' = model predictions accumulated by System [batch, time+1, ny]
    #
    # Both in physical units. yn has one extra timestep (the initial
    # condition), so it is sliced :-1 to align with the truth. The leading
    # history_steps positions are the seeded history — copies of Y that
    # contribute exactly zero error and would only dilute the mean — so both
    # sides drop them. This is the same span pipeline.evaluate_and_plot scores.
    #
    # Each channel is scored normalised, with its own head's loss (MSE for a
    # regression head, BCE for binary, ...), or blanket MSE without heads.
    shift = torch.tensor([norm_stats[c]["shift"] for c in nm_mapping["Y"]],
                         dtype=torch.float32, device=cfg.device)
    stretch = torch.tensor([norm_stats[c]["stretch"] for c in nm_mapping["Y"]],
                           dtype=torch.float32, device=cfg.device)
    h = cfg.history_steps

    def tracking(yn, Y):
        y_hat = (yn[:, h:-1, :] - shift) / stretch
        y = (Y[:, h:, :] - shift) / stretch
        if head_stack is None:
            return torch.mean((y_hat - y) ** 2)
        return head_stack.loss(y_hat, y)

    ref_loss = Loss(input_keys=["yn", "Y"], loss=tracking, weight=cfg.ref_gain,
                    name="ref_loss")

    objectives = [ref_loss]

    # ── Sampled gradient sign penalty ───────────────────────────────
    #
    # Samples random points in [0,1] input space, computes Jacobians,
    # and penalizes wrong-sign entries. Decoupled from System rollout.
    #
    # Uses Neuromancer's Loss class with input_keys=[] since it doesn't
    # read from the data dict — it generates its own synthetic inputs.

    if grad_masks and cfg.grad_gain > 0:
        penalty = SampledGradientPenalty(
            model=signed_model(contract_model(system), nm_mapping, norm_stats,
                               cfg.history_steps),
            grad_mapping=grad_masks,
            n_samples=cfg.grad_n_samples,
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
