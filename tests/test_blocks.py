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


# ── D: no partition requirement, shared by nature ────────────────

def test_a_named_d_subset_narrows_just_that_block():
    spec = {
        "north": {"Y": ["north T"], "U": ["north sp"], "D": ["oat"]},
        "south": {"Y": ["south T"], "U": ["south sp"]},
    }
    blocks = build_blocks(MAPPING, spec)
    north, south = blocks
    assert north.d_idx == (0,)          # named subset
    assert south.d_idx == (0, 1)        # absent -> all of D


# ── reads: an ordinary input, gathered from the shared window ───

def test_y_in_idx_puts_own_channels_first_then_reads_sorted():
    """Own channels come first so a family that only ever looks at 'its own
    state' keeps reading the same leading slice it always has."""
    block = Block("power", y_idx=(3,), u_idx=(), y_names=("power",),
                  d_idx=(), reads=(0, 1))
    assert block.y_in_idx == (3, 0, 1)
    assert block.ny == 1
    assert block.ny_in == 3


# ── Declarations: what a column is ───────────────────────────────

def test_a_shared_setpoint_is_a_demand_in_one_block_only():
    """Two blocks each pairing the shared column with their own state would
    collapse to one entry in the pre-stack, and the other block would get a
    demand computed against the wrong zone."""
    spec = {
        "north": {"Y": ["north T"], "U": ["north sp"],
                  "inputs": {"ahu fan": {"kind": "demand", "sign": -1,
                                         "state": "north T"}}},
        "south": {"Y": ["south T"], "U": ["south sp"],
                  "inputs": {"ahu fan": {"kind": "demand", "sign": -1,
                                         "state": "south T"}}},
    }
    with pytest.raises(ValueError, match=r"'ahu fan': \['north', 'south'\]"):
        build_blocks(MAPPING, spec)


def test_a_misspelled_block_key_is_refused_rather_than_ignored():
    """A declaration key changes what the block fits, not its shape, so a
    typo has no other way of showing up."""
    spec = {"all": {"Y": MAPPING["Y"], "Ambient": "oat", "input": {}}}
    with pytest.raises(ValueError, match=r"unrecognized key\(s\) \['Ambient', 'input'\]"):
        build_blocks(MAPPING, spec)


# ── One line of config picks a family ───────────────────────────

import torch  # noqa: E402

def _swap_system(**over):
    from model_fixtures import model_cfg, unit_stats
    from neurotemplate.modeling.dynamics import get_dynamics
    mapping = {"Y": ["a", "b"], "U": ["u"], "D": ["d"], "t": ["sec_elapsed"]}
    spec = over.pop("spec", {"one": {"Y": ["a"], "U": ["u"]}, "two": {"Y": ["b"], "U": []}})
    cfg = model_cfg(history={c: 2 for c in ("a", "b", "u", "d")}, **over)
    return cfg, mapping, get_dynamics(cfg, mapping, unit_stats(mapping), block_spec=spec)


@pytest.mark.parametrize("family", ["mlp", "nssm", "lssm", "gru"])
def test_the_config_alone_swaps_every_block_to_any_family(family):
    from neurotemplate.modeling.dynamics import describe_dynamics
    cfg, mapping, system = _swap_system(model=family)
    arch = describe_dynamics(system, cfg)
    assert list(arch["blocks"]) == ["one", "two"]
    assert {info["model"] for info in arch["blocks"].values()} == {family}
    data = {"Y": torch.randn(2, 6, 2), "U": torch.randn(2, 6, 1), "D": torch.randn(2, 6, 1)}
    data["yn"] = data["Y"][:, :3]
    assert system(data)["yn"].shape == (2, 7, 2)


def test_a_family_is_settled_by_block_models_then_the_hook_then_model():
    from neurotemplate.modeling.dynamics import describe_dynamics, family_lines
    spec = {"one": {"Y": ["a"], "U": ["u"], "model": "gru"}, "two": {"Y": ["b"], "U": []}}
    cfg, mapping, system = _swap_system(model="mlp", spec=spec, block_models={"two": "lssm"})
    blocks = describe_dynamics(system, cfg)["blocks"]
    assert (blocks["one"]["model"], blocks["two"]["model"]) == ("gru", "lssm")
    assert family_lines(cfg, mapping, spec) == ["  block 'one': gru (from block_mapping)",
                                                "  block 'two': lssm (from block_models)"]
    cfg, mapping, system = _swap_system(model="nssm", spec=spec)
    assert describe_dynamics(system, cfg)["blocks"]["two"]["model"] == "nssm"


def test_block_models_naming_no_block_is_refused():
    with pytest.raises(ValueError, match="block_models names"):
        _swap_system(model="mlp", block_models={"nope": "gru"})
