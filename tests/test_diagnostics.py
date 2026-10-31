"""
Tests for the `nt data heads` statistics.

`diagnostics.py` is deliberately pure (numpy in, dataclasses/strings out) so
these tests build synthetic arrays instead of a dataset and a config file.

The head-recommendation tests are built around one claim in particular: the
statistic must key off the SHAPE of the distribution (modes with unoccupied
gaps between them), not the number of distinct values. `test_ordinal_is_
not_driven_by_distinct_value_counting` is the test that would fail if
`recommend_head` were secretly doing `len(np.unique(x))` instead.
"""


import numpy as np
import pytest

from neurotemplate.modeling.diagnostics import recommend_head


# ── Synthetic column generators ───────────────────────────────────

def _staged(seed=0, n=3000, centers=(0.0, 4210.0, 8070.0), noise=150.0):
    """A staged signal: noise on every sample, so cardinality alone looks
    continuous, but the values cluster around `centers` with empty gaps
    between them -- the DX-compressor motivating case from §4.2."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(centers), n)
    return np.asarray(centers)[labels] + rng.normal(0, noise, n)


def _binary(seed=0, n=500):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n).astype(float)


def _floor_pinned(seed=0, n_zero=400, n_tail=1600):
    """Mass piled at zero, continuous positive tail -- the BoundedHead case."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(n_zero), rng.exponential(5.0, n_tail) + 0.01])
    rng.shuffle(x)
    return x


# ── recommend_head: one shape, one kind ───────────────────────────

@pytest.mark.parametrize("make_x,expected_kind", [
    pytest.param(_staged, "ordinal", id="staged-three-level"),
    pytest.param(_binary, "binary", id="zero-one-flag"),
    pytest.param(_floor_pinned, "bounded", id="floor-pinned-continuous-tail"),
])
def test_recommend_head_kind_from_shape(make_x, expected_kind):
    rec = recommend_head(make_x(), "col")
    assert rec.kind == expected_kind


def test_staged_signal_recommends_the_right_number_of_levels():
    rec = recommend_head(_staged(), "Total DX Power [W]")
    assert rec.kind == "ordinal"
    assert "n_levels=3" in rec.snippet
    assert "level_values=[" in rec.snippet
