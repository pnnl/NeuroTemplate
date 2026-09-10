"""Per-channel metric shapes, ranges, and edge cases."""

import math

import pytest
import torch

from neurotemplate.common.metrics import (
    ChannelMetrics,
    metrics_to_dict,
    metrics_to_markdown,
    per_channel_metrics,
)


def test_perfect_prediction():
    """Identical pred and true → zero error, R²=1, NRMSE=0 if range>0."""
    y = torch.linspace(0, 10, 50).reshape(1, 50, 1)
    metrics = per_channel_metrics(y, y, ["y0"])
    assert len(metrics) == 1
    m = metrics[0]
    assert m.name == "y0"
    assert m.rmse == pytest.approx(0.0)
    assert m.mae == pytest.approx(0.0)
    assert m.r2 == pytest.approx(1.0)
    assert m.nrmse == pytest.approx(0.0)


def test_constant_truth_marks_nrmse_nan():
    """When channel is constant (range = 0), NRMSE should be NaN."""
    y_true = torch.ones(2, 10, 1)
    y_pred = torch.zeros(2, 10, 1)
    metrics = per_channel_metrics(y_pred, y_true, ["constant"])
    assert metrics[0].rmse == pytest.approx(1.0)
    assert math.isnan(metrics[0].nrmse)


def test_per_channel_independence():
    """Channels are reported separately and don't bleed into each other."""
    y_true = torch.zeros(1, 10, 2)
    y_pred = torch.zeros(1, 10, 2)
    y_pred[..., 0] = 1.0  # 1.0 error on ch0, 0.0 on ch1
    y_true[..., 0] = torch.linspace(0, 9, 10)  # give it a range
    y_pred[..., 0] += y_true[..., 0]  # offset by 1
    y_true[..., 1] = torch.linspace(0, 9, 10)
    y_pred[..., 1] = y_true[..., 1]  # perfect

    metrics = per_channel_metrics(y_pred, y_true, ["a", "b"])
    assert metrics[0].rmse == pytest.approx(1.0)
    assert metrics[1].rmse == pytest.approx(0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        per_channel_metrics(
            torch.zeros(1, 10, 2), torch.zeros(1, 10, 3), ["a", "b"]
        )


def test_wrong_channel_names_count_raises():
    with pytest.raises(ValueError):
        per_channel_metrics(
            torch.zeros(1, 10, 2), torch.zeros(1, 10, 2), ["only_one"]
        )


def test_to_dict_keys():
    y = torch.zeros(1, 5, 1)
    metrics = per_channel_metrics(y, y, ["x"])
    d = metrics_to_dict(metrics)
    assert "x" in d
    assert {"rmse", "mae", "r2", "nrmse", "n_samples"} <= set(d["x"].keys())


def test_markdown_table_has_header_and_row():
    y_true = torch.linspace(0, 10, 50).reshape(1, 50, 1)
    y_pred = y_true + 0.1
    md = metrics_to_markdown(per_channel_metrics(y_pred, y_true, ["temp"]))
    lines = md.splitlines()
    assert lines[0].startswith("| Channel")
    assert any("temp" in line for line in lines[2:])


# ── Level metrics: the classification view of a staged channel ────

from neurotemplate.common.metrics import (  # noqa: E402
    level_metrics,
    level_metrics_to_dict,
    level_metrics_to_markdown,
)


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


def test_a_level_the_truth_never_visits_has_no_recall_and_is_left_out():
    true = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([0, 1, 1, 1])
    m = level_metrics(pred, true, [0.0, 1.0, 2.0], "three-level")
    assert math.isnan(m.recall[2])
    assert m.balanced_accuracy == pytest.approx(0.75)   # mean of 0.5 and 1.0


def test_level_metrics_accept_any_matching_shape_and_reject_mismatch():
    true = torch.zeros(2, 5, dtype=torch.long)
    m = level_metrics(torch.zeros(2, 5, dtype=torch.long), true, [0.0, 1.0], "x")
    assert m.n_samples == 10
    with pytest.raises(ValueError, match="shape"):
        level_metrics(torch.zeros(3, dtype=torch.long), true, [0.0, 1.0], "x")


def test_level_metrics_render_beside_a_reference_and_serialize():
    true = torch.tensor([0, 0, 0, 1])
    model = level_metrics(torch.tensor([0, 0, 1, 1]), true, [0.0, 1.0], "dx")
    persist = level_metrics(torch.tensor([0, 0, 0, 0]), true, [0.0, 1.0], "dx")
    md = level_metrics_to_markdown([model], [persist])
    assert "| dx | 0.833 | 0.500 |" in md          # model beside persistence
    assert "| dx | 1 | 25.0% | 50.0% | 1.000 |" in md
    d = level_metrics_to_dict([model])
    assert d["dx"]["balanced_accuracy"] == pytest.approx(0.8333, abs=1e-3)
    assert d["dx"]["levels"] == [0.0, 1.0]
