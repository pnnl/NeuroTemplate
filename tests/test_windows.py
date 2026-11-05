"""
Per-column windows: the gather that narrows a rectangular window to each
column's own rows. A wrong index here is a silent misalignment — the model
reads the wrong step of the wrong column and trains anyway.
"""

import torch

from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.windows import window_gather
from model_fixtures import model_cfg, unit_stats
from neurotemplate.modeling.contract import contract_model

NM = {"Y": ["y0", "y1"], "U": ["u0"], "D": ["d0", "d1"], "t": ["sec_elapsed"]}


def test_two_keys_are_gathered_key_major_from_their_own_offsets():
    # Key 0: 2 rows of 1 column; key 1: 1 row of 2 columns, starting at 2.
    keep = (((0, 0, 1),), ((1, 0, 0), (0, 0, 0)))
    assert window_gather((1, 2), (2, 1), keep) == [0, 1, 3, 2]


def test_a_ragged_table_feeds_each_column_at_its_lag_and_a_uniform_one_gathers_nothing():
    past = {"y0": "10m", "y1": 0, "u0": "5m", "d0": 0, "d1": "10m"}
    model = contract_model(get_dynamics(model_cfg(model="lssm", past=past), NM,
                                        unit_stats(NM)))
    core = model.core.cores[0]
    # Window 3 deep. Y rows: [y0 y1] x 3; y0 keeps all three, y1 the newest.
    y = torch.arange(6.0).reshape(1, 3, 2)
    assert core.windows[0](y).tolist() == [[0.0, 2.0, 4.0, 5.0]]
    assert core.maps[0].linear.in_features == 4       # (2+1) + (0+1)
    assert core.maps[1].linear.in_features == 2
    assert core.maps[2].linear.in_features == 4

    uniform = {c: "10m" for c in past}
    model = contract_model(get_dynamics(model_cfg(model="lssm", past=uniform), NM,
                                        unit_stats(NM)))
    assert not any(w.gathers for w in model.core.cores[0].windows)
