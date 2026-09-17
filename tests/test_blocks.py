"""
Tests for the Y/U/D partition behind per-block models.

`build_blocks` is the whole contract between an application's
`block_mapping` hook and every per-block model: Y is partitioned exactly,
U may be left unclaimed (and is then shared), D has no partition
requirement at all, `reads` may name another block's Y channel but not the
block's own, a per-column declaration names only the block's own columns
and pairs a `demand` only with the block's own state, and anything else is
an error the user sees before a single epoch runs.
"""
import pytest

from neurotemplate.modeling.blocks import Block, build_blocks

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


# ── The basic partition (Y, U) ───────────────────────────────────

def test_no_hook_gives_one_block_holding_everything():
    """The single-block building needs no hook, and pays nothing for one."""
    blocks = build_blocks(MAPPING)
    assert len(blocks) == 1
    assert blocks[0].y_idx == (0, 1)
    assert blocks[0].u_idx == (0, 1, 2)
    assert blocks[0].d_idx == (0, 1)
    assert blocks[0].y_names == ("north T", "south T")
    assert blocks[0].model is None
    assert blocks[0].reads == ()


def test_each_block_gets_its_own_channels_and_its_names():
    blocks = build_blocks(MAPPING, SPLIT)
    assert [b.name for b in blocks] == ["north", "south"]
    assert [b.y_idx for b in blocks] == [(0,), (1,)]
    assert [b.y_names for b in blocks] == [("north T",), ("south T",)]


def test_unclaimed_controls_are_shared_with_every_block():
    """One air handler serving several blocks is the reason U is not
    required to partition: leave it out and every block gets it."""
    blocks = build_blocks(MAPPING, SPLIT)
    fan = MAPPING["U"].index("ahu fan")
    assert all(fan in b.u_idx for b in blocks)
    # ...and each block still sees only its own setpoint.
    assert [b.u_idx for b in blocks] == [(0, 2), (1, 2)]


def test_a_block_may_hold_several_states():
    blocks = build_blocks(MAPPING, {"whole": {"Y": MAPPING["Y"], "U": ["ahu fan"]}})
    assert blocks[0].ny == 2 and blocks[0].nu == 3


# ── D: no partition requirement, shared by nature ────────────────

def test_absent_d_means_every_block_sees_all_of_it():
    blocks = build_blocks(MAPPING, SPLIT)
    assert all(b.d_idx == (0, 1) for b in blocks)


def test_a_named_d_subset_narrows_just_that_block():
    spec = {
        "north": {"Y": ["north T"], "U": ["north sp"], "D": ["oat"]},
        "south": {"Y": ["south T"], "U": ["south sp"]},
    }
    blocks = build_blocks(MAPPING, spec)
    north, south = blocks
    assert north.d_idx == (0,)          # named subset
    assert south.d_idx == (0, 1)        # absent -> all of D


def test_two_blocks_may_claim_the_same_d_column_with_no_conflict():
    """D is exogenous and shared by nature, unlike Y and U — several
    blocks naming the same column (or all of it) is not an error."""
    spec = {
        "north": {"Y": ["north T"], "U": ["north sp"], "D": ["oat"]},
        "south": {"Y": ["south T"], "U": ["south sp"], "D": ["oat"]},
    }
    blocks = build_blocks(MAPPING, spec)
    assert all(b.d_idx == (0,) for b in blocks)


# ── reads: an ordinary input, gathered from the shared window ───

def test_reads_names_another_blocks_y_channel():
    spec = {
        "north": {"Y": ["north T"], "U": ["north sp"]},
        "south": {"Y": ["south T"], "U": ["south sp"], "reads": ["north T"]},
    }
    blocks = build_blocks(MAPPING, spec)
    north, south = blocks
    assert north.reads == ()
    assert south.reads == (0,)          # index of "north T" in nm_mapping["Y"]


def test_y_in_idx_puts_own_channels_first_then_reads_sorted():
    """Own channels come first so a family that only ever looks at 'its own
    state' — 2r2c's fixed linear term — keeps reading the same leading slice
    it always has."""
    block = Block("power", y_idx=(3,), u_idx=(), y_names=("power",),
                  d_idx=(), reads=(0, 1))
    assert block.y_in_idx == (3, 0, 1)
    assert block.ny == 1
    assert block.ny_in == 3


def test_model_defaults_to_none_meaning_cfg_model():
    blocks = build_blocks(MAPPING, SPLIT)
    assert all(b.model is None for b in blocks)


def test_model_is_recorded_per_block():
    spec = {
        "north": {"Y": ["north T"], "U": ["north sp"], "model": "2r2c"},
        "south": {"Y": ["south T"], "U": ["south sp"], "model": "mlp"},
    }
    blocks = build_blocks(MAPPING, spec)
    assert [b.model for b in blocks] == ["2r2c", "mlp"]


# ── Validation: named before the first epoch ─────────────────────

@pytest.mark.parametrize("spec,message", [
    ({"a": {"Y": ["north T"], "U": []}}, "not claimed by any block"),
    ({"a": {"Y": ["north T"]}, "b": {"Y": ["north T", "south T"]}},
     "claimed by more than one block"),
    ({"a": {"Y": ["typo T", "south T"]}}, "not in nm_mapping"),
    # A block reading a channel it already owns — automatic, so a spec error.
    ({"a": {"Y": ["north T"], "reads": ["north T"]},
      "b": {"Y": ["south T"]}}, "already owns them as Y output"),
])
def test_a_broken_split_is_named_before_training_starts(spec, message):
    with pytest.raises(ValueError, match=message):
        build_blocks(MAPPING, spec)


# ── Declarations: what a column is ───────────────────────────────

def test_declarations_are_carried_with_the_blocks_own_column_names():
    spec = {
        "north": {"Y": ["north T"], "U": ["north sp"], "ambient": "oat",
                  "inputs": {"north sp": {"kind": "demand", "sign": -1,
                                          "state": "north T"},
                             "ahu fan": {"kind": "linear", "sign": +1}}},
        "south": {"Y": ["south T"], "U": ["south sp"]},
    }
    north, south = build_blocks(MAPPING, spec)
    assert north.u_names == ("north sp", "ahu fan")     # shared fan included
    assert north.d_names == ("oat", "ghi")
    assert north.ambient == "oat"
    assert set(north.inputs) == {"north sp", "ahu fan"}
    assert south.inputs == {} and south.ambient is None


def test_no_hook_carries_every_column_name_and_no_declaration():
    (block,) = build_blocks(MAPPING)
    assert block.u_names == tuple(MAPPING["U"])
    assert block.d_names == tuple(MAPPING["D"])
    assert block.inputs == {} and block.ambient is None


def test_a_demand_may_only_read_the_blocks_own_state():
    """Cross-block pairing is `reads`, which 2r2c refuses for the stability
    reason and the other families should declare explicitly."""
    spec = {
        "north": {"Y": ["north T"], "U": ["north sp"],
                  "inputs": {"north sp": {"kind": "demand", "sign": -1,
                                          "state": "south T"}}},
        "south": {"Y": ["south T"], "U": ["south sp"]},
    }
    with pytest.raises(ValueError, match="own Y channels.*use `reads`"):
        build_blocks(MAPPING, spec)


def test_ambient_names_exactly_one_of_the_blocks_d_columns():
    spec = {"all": {"Y": MAPPING["Y"], "ambient": ["oat", "ghi"]}}
    with pytest.raises(ValueError, match="`ambient`.*not one of this block's D"):
        build_blocks(MAPPING, spec)


def test_every_problem_in_a_declaration_is_listed_at_once():
    spec = {"all": {"Y": MAPPING["Y"], "ambient": "typo",
                    "inputs": {"north sp": {"kind": "demand", "sign": 2,
                                            "state": "north T"},
                               "oat": {"kind": "demand", "sign": -1,
                                       "state": "north T"},
                               "nope": {"kind": "linear", "sign": +1},
                               "ghi": {"kind": "quadratic"}}}}
    with pytest.raises(ValueError) as err:
        build_blocks(MAPPING, spec)
    message = str(err.value)
    for expected in ("`ambient` ('typo')", "`sign` of -1 or +1",
                     "exactly one Y channel", "only a U column",
                     "names 'nope'", "kind 'quadratic'"):
        assert expected in message, message


# ── family_by_channel: which family builds which channel ──────────

def _mixed_nm():
    return {"Y": ["Zone Temp [°C]", "Total Power [W]"],
            "U": ["Cooling SP [°C]"],
            "D": ["Ambient [°C]", "Solar [W/m2]"],
            "t": ["sec_elapsed"]}


def test_family_by_channel_is_per_block_not_per_run():
    """The defect this pins: `cfg.model` is the wrong answer for any block
    that declares its own family, and `nt data heads` used to warn off a
    perfectly legal ordinal head on an mlp-built channel because of it."""
    from neurotemplate.modeling.dynamics import family_by_channel
    from neurotemplate.modeling.config import ModelingConfig

    cfg = ModelingConfig(train_data_path="a.csv", dev_data_path="b.csv",
                         test_data_path="c.csv", model="2r2c")
    spec = {
        "zone": {"Y": ["Zone Temp [°C]"], "U": ["Cooling SP [°C]"],
                 "model": "2r2c"},
        "power": {"Y": ["Total Power [W]"], "model": "mlp"},
    }
    assert family_by_channel(cfg, _mixed_nm(), spec) == {
        "Zone Temp [°C]": "2r2c",
        "Total Power [W]": "mlp",
    }


def test_family_by_channel_defers_to_cfg_model_without_a_spec():
    """No block spec: one block holding everything, so every channel is
    cfg.model — the no-regression case."""
    from neurotemplate.modeling.dynamics import family_by_channel
    from neurotemplate.modeling.config import ModelingConfig

    cfg = ModelingConfig(train_data_path="a.csv", dev_data_path="b.csv",
                         test_data_path="c.csv", model="2r2c")
    assert family_by_channel(cfg, _mixed_nm(), None) == {
        "Zone Temp [°C]": "2r2c",
        "Total Power [W]": "2r2c",
    }
