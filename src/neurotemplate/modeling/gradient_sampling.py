"""
Sampled Gradient Sign Penalty
=============================
Computes gradient sign violations at randomly sampled input points,
decoupled from the Neuromancer System rollout.

Instead of computing Jacobians along the training trajectory (coupled to
the System, tricky shapes, TCN issues), we sample random points in the
normalized input space and evaluate gradient sign constraints there.

Sampling distribution is chosen to match the normalization strategy:
  - min_max  → Uniform[0, 1]
  - abs_max  → Uniform[-1, 1]
  - z_score  → Normal(0, 1)
  - None     → not supported (raises at init)

This module provides the raw computation. It is wrapped as a Neuromancer
Loss object in loss.py so it integrates cleanly with PenaltyLoss.
"""

import torch
from torch import relu


class SampledGradientPenalty:
    """
    Computes gradient sign violation at randomly sampled input points.

    On each call to compute():
      1. Samples n_samples random points in the normalized input space
      2. Computes Jacobians of the full dynamics step via jacrev + vmap
      3. Returns the mean sign violation (unweighted scalar)

    The Jacobian includes the residual connection so we get
    d(y_{k+1})/d(input), not just d(learned_correction)/d(input).

    Args:
        model:        SSMGrad instance (the raw dynamics module)
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

        self.model = model
        self.grad_mapping = grad_mapping
        self.n_samples = n_samples
        self.device = device
        self.norm_type = cfg.norm_type

        # Determine which arguments (y=0, u=1, d=2) to differentiate w.r.t.
        argnums = []
        self.keys = []
        for i, key in enumerate(["Y", "U", "D"]):
            if key in grad_mapping:
                argnums.append(i)
                self.keys.append(key)

        ny = model.ny

        # Build the function to differentiate.
        # Includes residual so we get d(y_{k+1})/d(input).
        def full_step(y, u, d):
            out = model.compute_output(y, u, d)
            if model.residual:
                if model.is_tcn:
                    # Under vmap y is 2D: [in_steps, ny]. Last row = most recent state.
                    out = out + y[-1]
                else:
                    # Under vmap y is 1D: [in_steps * ny]. Last ny elements = most recent state.
                    out = out + y[-ny:]
            return out

        self.jac_fn = torch.vmap(
            torch.func.jacrev(full_step, argnums=tuple(argnums))
        )

    def _repeat_sample(self, nx: int) -> torch.Tensor:
        """Return [n_samples, nx] random inputs matching the normalization range."""
        if self.norm_type == "min_max":
            sample = torch.rand(self.n_samples, nx, device=self.device)
        elif self.norm_type == "abs_max":
            sample = 2 * torch.rand(self.n_samples, nx, device=self.device) - 1
        else:  # z_score
            sample = torch.randn(self.n_samples, nx, device=self.device)

        if self.model.is_tcn:
            return sample.unsqueeze(1).expand(-1, self.model.in_steps, -1)
        else:
            return sample.repeat(1, self.model.in_steps)

    def compute(self):
        """
        Sample random inputs, compute Jacobians, return mean sign violation.

        Returns:
            Scalar tensor (differentiable w.r.t. model parameters via
            create_graph behavior of jacrev).
        """

        # Random inputs in normalized space (distribution matches norm_type).
        # Buildings are slow-moving: sample a single state per trajectory and
        # tile it across in_steps, so the building has been at that state for
        # the entire history window rather than jumping randomly each step.
        y = self._repeat_sample(self.model.ny)
        u = self._repeat_sample(self.model.nu)
        d = self._repeat_sample(self.model.nd)

        # Each J_i has shape [n_samples, ny, in_steps * nx_i]
        J = self.jac_fn(y, u, d)
        if not isinstance(J, tuple):
            J = (J,)

        total = torch.tensor(0.0, device=self.device)
        for j_mat, key in zip(J, self.keys):
            mask = self.grad_mapping[key]       # [1, ny, nx]
            nx = mask.shape[-1]

            # Only care about the most recent input step.
            # TCN Jacobian: [n_samples, ny, in_steps, nx] — slice last timestep.
            # MLP Jacobian: [n_samples, ny, in_steps*nx] — slice last nx columns.
            if self.model.is_tcn:
                j_last = j_mat[:, :, -1, :]     # [n_samples, ny, nx]
            else:
                j_last = j_mat[:, :, -nx:]      # [n_samples, ny, nx]
            m = mask.squeeze()                   # [ny, nx]

            # Penalize sign violations:
            #   mask=+1 → penalize negative Jacobian entries
            #   mask=-1 → penalize positive Jacobian entries
            #   mask= 0 → no penalty (relu of 0 is 0)
            violations = relu(-j_last * m)       # [n_samples, ny, nx]
            total = total + violations.mean()

        return total
