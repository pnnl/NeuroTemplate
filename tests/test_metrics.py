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
