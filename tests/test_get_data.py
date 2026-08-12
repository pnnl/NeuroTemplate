"""CSV loading: timestamp parsing, frequency inference, elapsed time."""

import numpy as np
import pandas as pd
import pytest

from neurotemplate.common.data_utils import get_data


def write_csv(tmp_path, freq, n=48, name="series.csv"):
    """An n-row CSV at `freq` with one value column."""
    path = tmp_path / name
    pd.DataFrame({
        "timestamp": pd.date_range("2024-08-02 00:00", periods=n, freq=freq),
        "value": np.arange(n, dtype=float),
    }).to_csv(path, index=False)
    return str(path)


# "h" and "D" are what `pd.infer_freq` returns for hourly and daily files, and
# an hourly tariff over 5-minute plant data is the documented price recipe.
@pytest.mark.parametrize("freq,expected", [
    ("5min", 300.0),
    ("15min", 900.0),
    ("min", 60.0),
    ("h", 3600.0),
    ("2h", 7200.0),
    ("D", 86400.0),
])
def test_freq_seconds_covers_unit_and_multiplied_frequencies(
    tmp_path, freq, expected,
):
    _, freq_seconds = get_data(write_csv(tmp_path, freq))
    assert freq_seconds == expected


def test_sec_elapsed_counts_from_the_first_row(tmp_path):
    """Elapsed, not POSIX — plots that want wall-clock need the df index."""
    df, _ = get_data(write_csv(tmp_path, "h", n=5))
    assert df["sec_elapsed"].tolist() == [0.0, 3600.0, 7200.0, 10800.0, 14400.0]
    assert df.index[0] == pd.Timestamp("2024-08-02 00:00")


def test_irregular_timestamps_fail_loudly(tmp_path):
    path = tmp_path / "gappy.csv"
    pd.DataFrame({
        "timestamp": ["2024-08-02 00:00", "2024-08-02 01:00", "2024-08-02 03:30"],
        "value": [1.0, 2.0, 3.0],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Could not infer frequency"):
        get_data(str(path))
