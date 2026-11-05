"""
Sampled Gradient Sign Penalty
=============================
Computes gradient sign violations at randomly sampled input points,
decoupled from the rollout.

Instead of computing Jacobians along the training trajectory, we sample random
points in the normalized input space and evaluate gradient sign constraints
there. Any model gets this: a `Stepper` is differentiated at its `sign_step`, a
model of your own through its contract call, in normalised units either way.

Sampling distribution is chosen to match the normalization strategy:
  - min_max  → Uniform[0, 1]
  - abs_max  → Uniform[-1, 1]
  - z_score  → Normal(0, 1)
  - None     → not supported (raises at init)

This module provides the raw computation. It is wrapped as a Neuromancer
Loss object in loss.py so it integrates cleanly with PenaltyLoss.
"""

import torch
import torch.nn as nn
from torch import relu

from neurotemplate.modeling.contract import stateful_blocks
from neurotemplate.modeling.stepper import Scaler


class _ContractSigns(nn.Module):
    """A model with no `sign_step`: its contract call, in normalised units."""

    def __init__(self, model, nm_mapping, norm_stats, history_steps):
        super().__init__()
        self.model = model
        self.scale = Scaler(nm_mapping, norm_stats)
        self.history_steps = history_steps
        self.ny, self.nu, self.nd = (len(nm_mapping[key]) for key in ("Y", "U", "D"))
        self.hinges = None

    def _physical(self, x, key):
        key = key.lower()
        return x * getattr(self.scale, f"{key}_stretch") + getattr(self.scale, f"{key}_shift")

    def sign_step(self, y, u, d):
        y_next = self.model(self._physical(y, "Y"), self._physical(u, "U"),
                            self._physical(d, "D"))
        return self.scale.normalize("Y", y_next)


def signed_model(model, nm_mapping, norm_stats, history_steps):
    """`model`, ready for the penalty: as it is if it has a `sign_step`."""
    if hasattr(model, "sign_step"):
        return model
    return _ContractSigns(model, nm_mapping, norm_stats, history_steps)


class SampledGradientPenalty:
    """
    Computes gradient sign violation at randomly sampled input points.

    On each call to compute():
      1. Samples n_samples random points in the normalized input space
      2. Differentiates each constrained output of the dynamics step with
         respect to its inputs, from cold; the batch's samples are independent,
         so the gradient of an output's sum is that output's Jacobian per sample
      3. Sums each Jacobian over the history_steps lags and returns the mean sign
         violation of that sum per constrained (output, input) pair
         (unweighted scalar)

    The mean is over the constrained entries only, not the whole `ny × nx`
    sign matrix, so the same constraints weigh the same however many
    unconstrained channels a system has, and `grad_gain` ports between
    systems.

    The Jacobian is of `model.sign_step` — the core plus the residual, per
    channel exactly as in a rollout — so it is d(y_{k+1})/d(input) up to each
    head's monotone activation, which cannot change a sign, rather than just
    d(learned_correction)/d(input).

    The Jacobian is taken downstream of the model's setpoint hinges: sampled raw
    inputs are pushed through them first, so a hinged column is differentiated as
    the hinge, not the setpoint. That is what makes a sign on it meaningful —
    `∂z/∂setpoint` is 0 on the inactive side and has no universal sign, `∂z/∂hinge`
    does — and it turns the Y half into `∂y_{k+1}/∂y_k` at a *fixed* hinge, the zone's
    own self-coefficient rather than the total derivative through the thermostat.

    Args:
        model:        anything `signed_model` returns
        grad_mapping: dict like {"Y": Tensor[1,1,ny,ny], "U": ...}
                      with entries in {-1, 0, +1}
        n_samples:    number of random points per call
        device:       torch device
        cfg:          ModelingConfig object; norm_type must be set (not None)
    """

    def __init__(self, model, grad_mapping, n_samples, device, cfg):
        if not grad_mapping:
            raise ValueError(
                "SampledGradientPenalty requires a non-empty grad_mapping. "
                "If you don't want gradient penalties, set grad_gain=0."
            )
        if cfg.norm_type is None:
            raise ValueError(
                "SampledGradientPenalty requires normalization. "
                "Set norm_type to 'min_max', 'abs_max', or 'z_score' in config."
            )

        widths = {"Y": model.ny, "U": model.nu, "D": model.nd}
        for key, mask in grad_mapping.items():
            if tuple(mask.shape[-2:]) != (model.ny, widths[key]):
                raise ValueError(
                    f"grad_mapping[{key!r}] has shape {tuple(mask.shape)}; the "
                    f"model expects (..., {model.ny}, {widths[key]})."
                )

        self.model = model
        # [ny, nx] float sign matrices, on the device the Jacobians land on.
        self.signs = {key: mask[0, 0].to(device=device, dtype=torch.float32)
                      for key, mask in grad_mapping.items()}
        self.n_samples = n_samples
        self.n_constrained = max(
            sum(int((m != 0).sum()) for m in self.signs.values()), 1)
        self.device = device
        self.norm_type = cfg.norm_type
        # Only the outputs a sign is declared on need differentiating.
        self.outputs = sorted({i for m in self.signs.values()
                               for i in (m != 0).any(dim=1).nonzero().squeeze(1).tolist()})
        self.stateful = stateful_blocks(model) if isinstance(model, nn.Module) else []

    def _repeat_sample(self, nx: int) -> torch.Tensor:
        """Return [n_samples, history_steps, nx] random inputs matching the
        normalization range."""
        if self.norm_type == "min_max":
            sample = torch.rand(self.n_samples, nx, device=self.device)
        elif self.norm_type == "abs_max":
            sample = 2 * torch.rand(self.n_samples, nx, device=self.device) - 1
        else:  # z_score
            sample = torch.randn(self.n_samples, nx, device=self.device)

        return sample[:, None].repeat(1, self.model.history_steps, 1)

    def compute(self):
        """
        Sample random inputs, compute Jacobians, return the mean sign
        violation per sample and constrained entry.

        Returns:
            Scalar tensor, differentiable w.r.t. model parameters.
        """

        # Random inputs in normalized space (distribution matches norm_type).
        # Buildings are slow-moving: sample a single state per trajectory and
        # tile it across history_steps, so the building has been at that state for
        # the entire history window rather than jumping randomly each step.
        y = self._repeat_sample(self.model.ny)
        u = self._repeat_sample(self.model.nu)
        d = self._repeat_sample(self.model.nd)
        if self.model.hinges is not None:
            u = self.model.hinges(y, u)
        inputs = {"Y": y, "U": u, "D": d}
        leaves = {key: inputs[key].detach().requires_grad_(True) for key in self.signs}
        windows = [leaves.get(key, inputs[key]) for key in ("Y", "U", "D")]

        # From cold, and leaving the rollout's state as it found it.
        saved = [block.rollout_state() for block in self.stateful]
        for block in self.stateful:
            block.set_rollout_state(None)
        # cuDNN's fused kernels have no second derivative.
        with torch.backends.cudnn.flags(enabled=False):
            out = self.model.sign_step(*windows)
            total = torch.tensor(0.0, device=self.device)
            for i in self.outputs:
                grads = torch.autograd.grad(out[:, i].sum(), list(leaves.values()),
                                            create_graph=True, retain_graph=True,
                                            allow_unused=True)
                for key, grad in zip(leaves, grads):
                    if grad is None:     # an input this output does not read
                        continue
                    # Sum over the lags: the response to shifting the whole
                    # (constant) window, i.e. a sustained change. Individual lag
                    # coefficients have no physical sign — an older lag may go
                    # negative to carry rate information — but their sum does.
                    #   mask=+1 → penalize negative Jacobian entries
                    #   mask=-1 → penalize positive Jacobian entries
                    #   mask= 0 → no penalty (relu of 0 is 0)
                    total = total + relu(-grad.sum(dim=1) * self.signs[key][i]).sum()
        for block, state in zip(self.stateful, saved):
            block.set_rollout_state(state)

        return total / (self.n_samples * self.n_constrained)
