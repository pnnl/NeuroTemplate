"""`nt data plot`: channels grouped by role, and a gap drawn as a break."""

import numpy as np
import pandas as pd

from neurotemplate.modeling.cli.data_plot import UNMAPPED, break_at_gaps, group_channels


def test_channels_group_by_role_and_the_rest_are_unmapped():
    mapping = {"Y": ["Zone [°C]"], "U": ["Setpoint [°C]"], "D": ["Not In File"],
               "t": ["sec_elapsed"]}
    columns = ["Setpoint [°C]", "Comfort Low [°C]", "Zone [°C]"]
    assert group_channels(columns, mapping) == {
        "Y": ["Zone [°C]"], "U": ["Setpoint [°C]"], UNMAPPED: ["Comfort Low [°C]"]}
    assert group_channels(columns, None) == {"columns": columns}


def test_a_gap_breaks_the_line_one_step_after_it_starts():
    index = pd.date_range("2024-08-01", periods=10, freq="5min").delete([4, 5, 6])
    line, gaps = break_at_gaps(pd.Series(np.arange(7.0), index=index))
    assert gaps == 1
    assert line.isna().sum() == 1
    assert np.isnan(line[pd.Timestamp("2024-08-01 00:20")])
