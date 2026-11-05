"""The sampled gradient sign penalty scores each declared sign against its own
(output, input) pair and nothing else, summed over the history_steps lags, and
averaged over the declared pairs only."""

import torch

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.gradient_sampling import SampledGradientPenalty


class _Linear:
    """y_next = sum_k W[k] @ u_k over a two-step window (oldest first), so
    d(y_i)/d(u_j) at lag k = W[k, i, j] exactly. A [ny, nu] weight acts on
    the newest step only."""

    def __init__(self, weights: torch.Tensor):
        if weights.dim() == 2:
            weights = torch.stack([torch.zeros_like(weights), weights])
        self.w = weights
        self.history_steps, self.ny, self.nu = weights.shape
        self.nd, self.hinges = 1, None

    def sign_step(self, y, u, d):
        return torch.einsum("kij,bkj->bi", self.w, u)


def _penalty(weights: torch.Tensor, mask: torch.Tensor) -> float:
    cfg = ModelingConfig(train_data_path="x", dev_data_path="x", test_data_path="x")
    return float(SampledGradientPenalty(_Linear(weights), {"U": mask}, 8, "cpu", cfg).compute())


def _mask(ny: int, nu: int, entries: dict) -> torch.Tensor:
    mask = torch.zeros(1, 1, ny, nu, dtype=torch.int8)
    for (i, j), sign in entries.items():
        mask[0, 0, i, j] = sign
    return mask


def test_a_single_input_column_constrains_only_its_declared_output():
    # One U column, two outputs; only output 0 has a declared sign (+1), and
    # it is obeyed. Output 1 responds negatively, which nothing forbids.
    weights = torch.tensor([[2.0], [-3.0]])
    assert _penalty(weights, _mask(2, 1, {(0, 0): 1})) == 0.0


def test_a_negative_older_lag_is_allowed_when_the_sum_obeys_the_sign():
    # Older lag -1, newest +3: the sustained response is +2, which obeys +1.
    weights = torch.tensor([[[-1.0]], [[3.0]]])
    assert _penalty(weights, _mask(1, 1, {(0, 0): 1})) == 0.0


def test_a_sum_over_lags_that_violates_the_sign_is_penalised():
    # Newest lag alone obeys +1, but the older lag drags the sum negative.
    weights = torch.tensor([[[-3.0]], [[1.0]]])
    assert _penalty(weights, _mask(1, 1, {(0, 0): 1})) > 0.0
