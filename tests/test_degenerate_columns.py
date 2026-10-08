"""Columns that are constant in training carry raw units into the model.

`compute_norm_stats` guards against dividing by zero on a constant column by
setting `stretch = 1`. That makes normalization the identity for that column, so
it stays in raw physical units — harmless while the column stays constant, and
catastrophic the moment it does not.

Measured on real submissions: two of four NYC buildings trained on an excitation
that never fired the furnace, so `Gas Energy [J]` and `Furnace Gas Power [W]`
were constant and got `stretch = 1`. Their surrogates scored 1.03 and 0.66 °C
zone RMSE on their own test splits and diverged to 1e17 within 48 steps of the
evaluator's trajectory, where the furnace does fire and 1.6e9 J arrives as an
input of 1.5e9. Zeroing just those two columns brought the same models back to
0.90 and 1.18 °C. The other two buildings had no constant columns and did not
diverge that way.

These tests pin the detector and the warning that surfaces it.
"""

import pandas as pd
import pytest

from neurotemplate.common.data_utils import (
    compute_norm_stats,
    degenerate_columns,
    degenerate_mapped_columns,
    warn_degenerate_columns,
)

MAPPING = {"Y": ["zone [°C]", "gas [J]"], "U": ["cool_sp [°C]"], "D": ["oat [°C]"]}


def _frame(gas: list[float]) -> pd.DataFrame:
    """Four mapped columns; `gas` decides whether one of them is constant."""
    n = len(gas)
    return pd.DataFrame({
        "zone [°C]": [22.0 + i * 0.1 for i in range(n)],
        "gas [J]": gas,
        "cool_sp [°C]": [24.0] * n,       # constant too, and in U
        "oat [°C]": [18.0 + i * 0.5 for i in range(n)],
    })


@pytest.mark.parametrize("norm_type", ["min_max", "abs_max", "z_score"])
def test_a_constant_column_is_detected_under_every_norm_type(norm_type):
    """The fallback fires for all three strategies, so detection must too."""
    stats = compute_norm_stats(_frame([1e9] * 6), norm_type=norm_type)
    assert "gas [J]" in degenerate_columns(stats)
    assert "zone [°C]" not in degenerate_columns(stats)


def test_unit_spread_is_not_mistaken_for_degenerate():
    """Detection is `min == max`, not `stretch == 1` — a column may span 1.0."""
    df = pd.DataFrame({"x": [0.0, 1.0]})
    stats = compute_norm_stats(df, norm_type="min_max")
    assert stats["x"]["stretch"] == pytest.approx(1.0)
    assert degenerate_columns(stats) == []


def test_the_fallback_leaves_the_column_in_raw_units():
    """The mechanism itself: stretch 1 and shift equal to the constant value.

    This is what makes a later non-constant value land orders of magnitude out
    of range instead of near zero.
    """
    stats = compute_norm_stats(_frame([1e9] * 6), norm_type="min_max")
    assert stats["gas [J]"]["stretch"] == 1
    assert stats["gas [J]"]["shift"] == pytest.approx(1e9)
    # A later value of 1.6e9 normalizes to 6e8, not to something near zero.
    later = (1.6e9 - stats["gas [J]"]["shift"]) / stats["gas [J]"]["stretch"]
    assert later == pytest.approx(6e8)


@pytest.mark.parametrize("norm_type", ["min_max", "abs_max", "z_score"])
def test_a_column_that_varies_on_a_tiny_scale_is_normalised_not_left_raw(norm_type):
    """A µA current varies; it must be scaled by its own spread. Left raw it
    reaches the model a million times smaller than its neighbours, and it is
    not constant, so no warning would say so."""
    df = pd.DataFrame({"current [A]": [-3e-7, -1e-7, 1e-7, 2e-7]})
    stats = compute_norm_stats(df, norm_type=norm_type)
    normalised = (df["current [A]"] - stats["current [A]"]["shift"]) / stats["current [A]"]["stretch"]
    assert normalised.abs().max() >= 0.5
    if norm_type != "z_score":
        assert normalised.abs().max() <= 1.0     # abs_max scales by |min| here
    assert degenerate_columns(stats) == []


# ── The startup warning ──────────────────────────────────────────
#
# The reportcard only exists once the run has finished. The condition is
# knowable when norm_stats is frozen, so it is also printed there — someone
# watching a loss curve should not have to open a file to learn about it.


def test_warn_ignores_columns_the_mapping_never_references(capsys):
    """Same filtering as the reportcard — both read `degenerate_mapped_columns`."""
    df = _frame([1e9, 2e9, 3e9, 4e9, 5e9, 6e9])
    df["cool_sp [°C]"] = [23.0, 24.0, 25.0, 24.0, 23.0, 25.0]
    df["ignored [-]"] = [7.0] * 6
    stats = compute_norm_stats(df)

    assert "ignored [-]" in degenerate_columns(stats)
    assert degenerate_mapped_columns(stats, MAPPING) == []
    assert warn_degenerate_columns(stats, MAPPING) == []
    assert capsys.readouterr().out == ""
