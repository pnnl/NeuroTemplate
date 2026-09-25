"""
Sampled Gradient Sign Penalty
=============================
Computes gradient sign violations at randomly sampled input points,
decoupled from the Neuromancer System rollout.

Instead of computing Jacobians along the training trajectory (coupled to
the System, tricky shapes), we sample random points in the normalized
input space and evaluate gradient sign constraints there.

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
      3. Sums each Jacobian over the in_steps lags and returns the mean sign
         violation of that sum (unweighted scalar)

    The Jacobian is of `model.pre_activation` — the learned map plus the
    residual, applied per channel exactly as in a rollout — so it is
    d(y_{k+1})/d(input) up to each head's monotone activation, which cannot
    change a sign, rather than just d(learned_correction)/d(input).

    The Jacobian is taken downstream of the model's pre-stack: sampled raw
    inputs are pushed through it outside `jac_fn`, so a `demand` column is
    differentiated as the demand feature, not the setpoint. That is what
    makes a sign on it meaningful — `∂z/∂setpoint` is 0 on the inactive side
    and has no universal sign, `∂z/∂demand` does — and it turns the Y half
    into `∂y_{k+1}/∂y_k` at *fixed* demand, the zone's own self-coefficient
    rather than the total derivative through the thermostat.

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
        self.device = device
        self.norm_type = cfg.norm_type

        # Determine which arguments (y=0, u=1, d=2) to differentiate w.r.t.
        argnums = []
        self.keys = []
        for i, key in enumerate(["Y", "U", "D"]):
            if key in grad_mapping:
                argnums.append(i)
                self.keys.append(key)

        self.jac_fn = torch.vmap(
            torch.func.jacrev(model.pre_activation, argnums=tuple(argnums))
        )

    def _repeat_sample(self, nx: int) -> torch.Tensor:
        """Return [n_samples, nx] random inputs matching the normalization range."""
        if self.norm_type == "min_max":
            sample = torch.rand(self.n_samples, nx, device=self.device)
        elif self.norm_type == "abs_max":
            sample = 2 * torch.rand(self.n_samples, nx, device=self.device) - 1
        else:  # z_score
            sample = torch.randn(self.n_samples, nx, device=self.device)

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
        if self.model.pre_stack is not None:
            u = self.model.pre_stack(y, u)

        # Each J_i has shape [n_samples, ny, in_steps * nx_i]
        J = self.jac_fn(y, u, d)
        if not isinstance(J, tuple):
            J = (J,)

        total = torch.tensor(0.0, device=self.device)
        for j_mat, key in zip(J, self.keys):
            m = self.signs[key]                  # [ny, nx]
            nx = m.shape[-1]

            # Sum the Jacobian over the in_steps lags: the response to shifting
            # the whole (constant) history window, i.e. a sustained change.
            # Individual lag coefficients have no physical sign — an older lag
            # may go negative to carry rate information — but their sum does.
            # Jacobian: [n_samples, ny, in_steps*nx], flattened oldest→newest.
            j_sustained = j_mat.reshape(
                j_mat.shape[0], j_mat.shape[1], self.model.in_steps, nx
            ).sum(dim=2)                         # [n_samples, ny, nx]

            # Penalize sign violations:
            #   mask=+1 → penalize negative Jacobian entries
            #   mask=-1 → penalize positive Jacobian entries
            #   mask= 0 → no penalty (relu of 0 is 0)
            violations = relu(-j_sustained * m)  # [n_samples, ny, nx]
            total = total + violations.mean()

        return total
