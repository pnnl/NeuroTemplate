"""
The model whitelist: `model_catalog` vs. `VALID_MODEL_NAMES`.

`model_catalog` only holds families with a monolithic factory — one network
over the whole Y vector at once. `2r2c` has no monolithic form: an RC
network describes *a* block, never all of Y jointly, so it lives only in
`BLOCK_BUILDERS`. Before this split, `model_catalog` did double duty as the
model whitelist too, which forced a placeholder entry for `2r2c` —
`get_2r2c`, a stub that only ever raised RuntimeError, kept alive purely so
the key existed. `VALID_MODEL_NAMES` (the union of `model_catalog` and
`BLOCK_BUILDERS`) is now the single whitelist `ModelingConfig.validate()`
and `get_dynamics` both read, so `get_2r2c` and its stub are gone entirely.
These tests are the behaviour that stub used to protect, checked without it.
"""

import pytest
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import (
    BLOCK_BUILDERS, VALID_MODEL_NAMES, get_dynamics, model_catalog,
)

MAPPING = {
    "Y": ["zone T"], "U": ["zone sp"], "D": ["oat"], "t": ["sec_elapsed"],
}


def make_cfg(**over) -> Config:
    cfg = Config()
    cfg.model = "2r2c"
    cfg.in_steps = 6
    cfg.log2_hidden_size = 4
    cfg.num_layers = 1
    cfg.steps_per_hour = 12.0
    cfg.device = torch.device("cpu")
    cfg.verbose = False
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


# ── The two concerns are separate dicts now ───────────────────────

def test_2r2c_has_no_monolithic_factory():
    """The whole reason model_catalog and the whitelist had to split."""
    assert "2r2c" not in model_catalog
    assert "2r2c" in BLOCK_BUILDERS


def test_get_2r2c_no_longer_exists():
    """The unreachable stub is gone, not just unreachable."""
    import neurotemplate.modeling.dynamics as dynamics
    assert not hasattr(dynamics, "get_2r2c")


def test_valid_model_names_is_the_union_of_both_dicts():
    assert set(VALID_MODEL_NAMES) == set(model_catalog) | set(BLOCK_BUILDERS)


# ── The whitelist is what validate() and get_dynamics both read ──

@pytest.mark.parametrize("name", list(VALID_MODEL_NAMES))
def test_every_whitelisted_name_passes_validate(name):
    """check_one_of reads VALID_MODEL_NAMES, not model_catalog: `2r2c` (which
    isn't in model_catalog) has to pass exactly like every other family."""
    cfg = make_cfg(model=name, train_data_path="x.csv", dev_data_path="x.csv",
                   test_data_path="x.csv")
    cfg.validate()  # must not raise


def test_an_unknown_model_is_refused_by_validate_naming_every_family():
    """The error the removed stub used to protect: an unknown model must be
    refused, and the message must still name every real family, 2r2c
    included — that is the behaviour `get_2r2c` existed to keep true."""
    cfg = make_cfg(model="banana")
    with pytest.raises(ValueError, match="banana"):
        cfg.validate()
    with pytest.raises(ValueError, match="2r2c"):
        cfg.validate()


def test_get_dynamics_also_names_2r2c_for_an_unknown_model():
    """get_dynamics has its own 'unknown model' gate, ahead of build_blocks;
    it must read the same whitelist as ModelingConfig.validate()."""
    cfg = make_cfg(model="banana")
    with pytest.raises(ValueError, match="2r2c"):
        get_dynamics(cfg, MAPPING)


# ── Regression guard: 2r2c's real (per-block) path is untouched ──

def test_2r2c_still_builds_and_reports_tau_slow():
    """Splitting the whitelist from model_catalog must not disturb 2r2c's
    actual per-block construction path or the number the reportcard reads."""
    system = get_dynamics(make_cfg(), MAPPING)
    fy = system.nodes[0].callable.fy
    taus = fy.blocks[0].describe()["time_constants_h"]["zone T"]
    assert taus["slow"] == pytest.approx(13.40, abs=0.05)
