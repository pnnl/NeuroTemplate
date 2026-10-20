"""Per-channel metric shapes, ranges, and edge cases."""

import math

import pytest
import torch

from neurotemplate.common.metrics import per_channel_metrics


def test_constant_truth_marks_nrmse_nan():
    """When channel is constant (range = 0), NRMSE should be NaN."""
    y_true = torch.ones(2, 10, 1)
    y_pred = torch.zeros(2, 10, 1)
    metrics = per_channel_metrics(y_pred, y_true, ["constant"])
    assert metrics[0].rmse == pytest.approx(1.0)
    assert math.isnan(metrics[0].nrmse)


# ── Level metrics: the classification view of a staged channel ────

from neurotemplate.common.metrics import level_metrics  # noqa: E402


def test_a_constant_predictor_scores_the_balanced_accuracy_floor():
    """The head-bake-off failure: never firing the furnace posts the best
    NRMSE on the page. Balanced accuracy must put it at the floor (1/K),
    not near the top."""
    true = torch.tensor([0] * 97 + [1] * 3)
    never = torch.zeros(100, dtype=torch.long)
    m = level_metrics(never, true, [0.0, 61728.4], "furnace")
    assert m.balanced_accuracy == pytest.approx(0.5)
    assert m.recall == [1.0, 0.0]
    assert m.pred_share == [1.0, 0.0]
    assert m.true_share == [0.97, 0.03]


def test_recall_is_per_level_and_the_mean_is_the_balanced_accuracy():
    true = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    pred = torch.tensor([0, 0, 1, 1, 1, 1, 2, 0])
    m = level_metrics(pred, true, [0.0, 483.0, 1660.0], "fan")
    assert m.recall == [0.5, 1.0, 0.5]
    assert m.balanced_accuracy == pytest.approx(2.0 / 3.0)
    assert m.n_samples == 8
