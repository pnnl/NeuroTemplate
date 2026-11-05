"""Per-channel metric shapes, ranges, and edge cases."""

import math

import pytest
import torch

from neurotemplate.common.metrics import skill


def test_skill_is_measured_against_persistence():
    """0 is no better than holding the last value, below 0 is worse, and a
    channel persistence already predicts exactly has no skill to score."""
    assert skill(1.0, 1.0) == 0.0
    assert skill(0.25, 1.0) == pytest.approx(0.75)
    assert skill(2.0, 1.0) < 0
    assert math.isnan(skill(0.1, 0.0))
