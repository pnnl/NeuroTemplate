"""
Tests for the Y/U partition behind per-zone models.

`build_zones` is the whole contract between an application's `zone_mapping`
hook and every per-zone block: Y is partitioned exactly, U may be left
unclaimed (and is then shared), and anything else is an error the user sees
before a single epoch runs.
"""
import pytest

from neurotemplate.modeling.zones import build_zones

MAPPING = {
    "Y": ["north T", "south T"],
    "U": ["north sp", "south sp", "ahu fan"],
    "D": ["oat", "ghi"],
    "t": ["sec_elapsed"],
}

SPLIT = {
    "north": {"Y": ["north T"], "U": ["north sp"]},
    "south": {"Y": ["south T"], "U": ["south sp"]},
}


def test_no_hook_gives_one_zone_holding_everything():
    """The single-zone building needs no hook, and pays nothing for one."""
    zones = build_zones(MAPPING)
    assert len(zones) == 1
    assert zones[0].y_idx == (0, 1)
    assert zones[0].u_idx == (0, 1, 2)
    assert zones[0].y_names == ("north T", "south T")


def test_each_zone_gets_its_own_channels_and_its_names():
    zones = build_zones(MAPPING, SPLIT)
    assert [z.name for z in zones] == ["north", "south"]
    assert [z.y_idx for z in zones] == [(0,), (1,)]
    assert [z.y_names for z in zones] == [("north T",), ("south T",)]


def test_unclaimed_controls_are_shared_with_every_zone():
    """One air handler serving several zones is the reason U is not required
    to partition: leave it out and every zone gets it."""
    zones = build_zones(MAPPING, SPLIT)
    fan = MAPPING["U"].index("ahu fan")
    assert all(fan in z.u_idx for z in zones)
    # ...and each zone still sees only its own setpoint.
    assert [z.u_idx for z in zones] == [(0, 2), (1, 2)]


def test_a_zone_may_hold_several_states():
    zones = build_zones(MAPPING, {"whole": {"Y": MAPPING["Y"], "U": ["ahu fan"]}})
    assert zones[0].ny == 2 and zones[0].nu == 3


@pytest.mark.parametrize("spec,message", [
    ({}, "empty mapping"),
    ({"a": {"Y": ["north T"], "U": []}}, "not claimed by any zone"),
    ({"a": {"Y": ["north T"]}, "b": {"Y": ["north T", "south T"]}},
     "claimed by more than one zone"),
    ({"a": {"Y": ["north T"], "U": ["north sp"]},
      "b": {"Y": ["south T"], "U": ["north sp"]}}, "claimed by more than one"),
    ({"a": {"Y": ["typo T", "south T"]}}, "not in nm_mapping"),
    ({"a": {"Y": ["north T", "south T"], "U": ["typo sp"]}}, "not in nm_mapping"),
    ({"a": {"Y": [], "U": ["north sp"]}, "b": {"Y": MAPPING["Y"]}},
     "claims no Y channels"),
])
def test_a_broken_split_is_named_before_training_starts(spec, message):
    with pytest.raises(ValueError, match=message):
        build_zones(MAPPING, spec)
