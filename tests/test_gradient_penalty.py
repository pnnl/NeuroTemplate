"""The sampled gradient sign penalty scores each declared sign against its own
(output, input) pair and nothing else."""

import torch

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.gradient_sampling import SampledGradientPenalty


class _Linear:
    """y_next = W @ u_newest, so d(y_i)/d(u_j) = W[i, j] exactly."""

    def __init__(self, weights: torch.Tensor):
        self.w = weights
        self.ny, self.nu = weights.shape
        self.nd, self.in_steps, self.pre_stack = 1, 2, None

    def pre_activation(self, y, u, d):
        return self.w @ u[-self.nu:]


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


def test_a_violated_sign_is_penalised():
    weights = torch.tensor([[-2.0], [5.0]])
    assert _penalty(weights, _mask(2, 1, {(0, 0): 1})) > 0.0
