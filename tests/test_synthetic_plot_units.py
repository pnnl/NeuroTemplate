"""plot_synthetic's y-axis label: per-trace units from norm_stats, with
cfg.synth_y_unit (renamed `fallback_unit` here) as the deprecated fallback
for a state column whose name carries no unit of its own.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from neurotemplate.modeling.plotting import plot_synthetic


def _trace(state_names, channel_names, n=1, steps=4):
    """A minimal stand-in for `scenario.trace(n)` -- only the keys
    plot_synthetic reads."""
    return {
        "name": "test-scenario",
        "predictions": torch.zeros(n, steps, len(channel_names)),
        "ambient_traj": torch.zeros(n, steps - 1),
        "channel_names": channel_names,
        "state_names": state_names,
        "seed_steps": 1,
    }


def _labels_from(traces, norm_stats, tmp_path, **kw):
    captured = []
    real_close = plt.close

    def close_and_capture(fig):
        captured.append(fig.axes[0].get_ylabel())
        real_close(fig)

    orig = plt.close
    plt.close = close_and_capture
    try:
        plot_synthetic(traces, run_dir=str(tmp_path), norm_stats=norm_stats, **kw)
    finally:
        plt.close = orig
    return captured


def test_label_uses_the_states_own_parsed_unit(tmp_path):
    col = "Zone Air Temperature [°C]"
    norm_stats = {col: {"unit": "°C"}}
    labels = _labels_from([_trace([col], [col])], norm_stats, tmp_path)
    assert labels == ["Temperature [°C]"]


def test_mixed_units_across_states_render_unit_less(tmp_path):
    """A per-block Y can mix a temperature state and a power state; printing
    either one's unit on the shared axis would be a wrong label, not a
    guess worth making."""
    cols = ["Zone Air Temperature [°C]", "Total Power [W]"]
    norm_stats = {cols[0]: {"unit": "°C"}, cols[1]: {"unit": "W"}}
    labels = _labels_from([_trace(cols, cols)], norm_stats, tmp_path)
    assert labels == ["Temperature"]


def test_fallback_unit_only_covers_a_column_with_no_unit_of_its_own(tmp_path):
    """`fallback_unit` is the deprecated cfg.synth_y_unit: it fills in for a
    column whose name carries no unit, but a column that already has one
    keeps it."""
    col = "Tank Temperature"  # no bracketed unit in the name
    norm_stats = {col: {}}
    labels = _labels_from(
        [_trace([col], [col])], norm_stats, tmp_path, fallback_unit="degC",
    )
    assert labels == ["Temperature [degC]"]


def test_no_fallback_and_no_parsed_unit_is_unit_less(tmp_path):
    col = "Tank Temperature"
    norm_stats = {col: {}}
    labels = _labels_from([_trace([col], [col])], norm_stats, tmp_path)
    assert labels == ["Temperature"]
