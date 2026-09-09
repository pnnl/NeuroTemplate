"""
Loading and validating a ControlConfig.

Almost everything here is one shape: set some fields, then assert that
`validate()` either names the problem or stays quiet. Those cases live in
two tables rather than two dozen near-identical functions — the table is
the readable form of "here is every way a control config can be wrong."
"""
import warnings

import pytest

from neurotemplate.common.config import ConfigError, CONTROL_CONFIG_DIR
from neurotemplate.control.config import ControlConfig, load_config_from_cli

_YAML = """
sysid_run_id: abc123
train_pred_steps: 24
run_dir: none
train_data_path: train.csv
dev_data_path:   dev.csv
test_data_path:  test.csv
band_lo: 20.0
band_hi: 26.0
"""

_TARIFF = dict(train_price_path="train_price.csv",
               dev_price_path="dev_price.csv",
               test_price_path="test_price.csv")


def _load(tmp_path, text, *overrides):
    path = tmp_path / "control.yaml"
    path.write_text(text)
    args = ["--config", str(path)]
    for override in overrides:
        args += ["--set", override]
    cfg, _ = load_config_from_cli(
        args, default_config_path=None, cls=ControlConfig,
        config_dir=CONTROL_CONFIG_DIR,
    )
    return cfg


def _runnable(**fields) -> ControlConfig:
    """A config that reaches every check past the required-field ones."""
    cfg = ControlConfig()
    cfg.sysid_run_id = "x"
    cfg.train_data_path = "train.csv"
    cfg.dev_data_path = "dev.csv"
    cfg.test_data_path = "test.csv"
    # band_lo/band_hi have no default, so every case not itself testing that
    # requirement needs a valid pair to get past it.
    cfg.band_lo = 20.0
    cfg.band_hi = 26.0
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


# ── Loading ──────────────────────────────────────────────────────

def test_yaml_fills_what_it_names_and_defaults_the_rest(tmp_path):
    cfg = _load(tmp_path, _YAML)
    assert cfg.sysid_run_id == "abc123"
    assert cfg.train_pred_steps == 24
    assert cfg.run_dir is None                      # "none" -> None on Optional
    assert cfg.comfort_weight == ControlConfig.comfort_weight


def test_set_overrides_the_yaml(tmp_path):
    cfg = _load(tmp_path, _YAML, "energy_weight=0", "epochs=3")
    assert cfg.energy_weight == 0 and cfg.epochs == 3


def test_an_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        _load(tmp_path, _YAML + "not_a_key: 1\n")


# ── Every way a control config can be wrong ──────────────────────

@pytest.mark.parametrize("fields,message", [
    # Required fields and ordering.
    ({"sysid_run_id": None}, "sysid_run_id"),
    # band_lo/band_hi are physical-unit constants with no library default;
    # missing either one names both fields, not a bare TypeError.
    ({"band_lo": None, "band_hi": None}, "band_lo and band_hi must both be set"),
    ({"band_lo": None}, "band_lo and band_hi must both be set"),
    ({"band_lo": 24.0, "band_hi": 20.0}, "band_lo"),
    # Policy options are closed sets.
    ({"policy": "bogus"}, "policy"),
    ({"policy_nonlin": "swish"}, "policy_nonlin"),
    ({"policy_bound_method": "clip"}, "policy_bound_method"),
    # Below 1, M_q rewards spiky profiles — the opposite of the request.
    ({"peak_q": 0.5}, "peak_q must be >= 1"),
    ({"cost_weight": -1.0}, "cost_weight"),
    ({"peak_weight": -1.0}, "peak_weight"),
    # The single most likely way to waste a campaign on the residual: every
    # delta snaps back to zero and nothing on any report says so.
    ({"setpoint_step": 0.5, "residual_delta": 0.2}, "smaller than setpoint_step"),
    ({"residual_delta": -0.5}, "residual_delta must be >= 0"),
    # A tariff is all-or-nothing: a split with no price file would silently
    # see no cost term.
    ({**_TARIFF, "train_price_path": None}, "together or not at all"),
    ({**_TARIFF}, "price_scale"),
    ({"price_scale": 0.30}, "price_scale is set"),
    ({**_TARIFF, "price_scale": 0.0}, "price_scale must be positive"),
    # Previewing a signal that never reaches the rollout is a dead input
    # width — caught at config load rather than at policy build.
    ({"price_preview": 48}, "no tariff reaches the rollout"),
    ({"price_preview": -1}, "price_preview must be >= 0"),
    ({"price_align": "hourly"}, "price_align must be one of"),
])
def test_validate_names_the_problem(fields, message):
    with pytest.raises(ValueError, match=message):
        _runnable(**fields).validate()


@pytest.mark.parametrize("fields", [
    pytest.param({"policy": "linear", "policy_bound_method": "relu_clamp"},
                 id="linear policy"),
    # Explicit band_lo/band_hi past the missing-field check works unchanged.
    pytest.param({"band_lo": 18.0, "band_hi": 24.0}, id="band explicitly set"),
    pytest.param({"setpoint_step": 0.5, "residual_delta": 0.5},
                 id="residual at exactly one grid step"),
    # 0 means "emit the reference exactly" — a floor to train nothing and
    # score, and not the same thing as a residual that quantizes away.
    pytest.param({"residual_delta": 0.0}, id="zero-width residual box"),
    pytest.param({**_TARIFF, "price_scale": 0.30, "price_preview": 48},
                 id="complete tariff"),
])
def test_validate_stays_quiet_on_a_legal_config(fields):
    _runnable(**fields).validate()


# ── in_steps: advisory, matching ModelingConfig ──────────────────


def test_in_steps_above_the_threshold_warns_instead_of_raising():
    """`in_steps` used to be a hard ValueError ceiling here (`check_max`
    against the modeling side's `MAX_IN_STEPS` alias) while the same value
    was merely advisory on ModelingConfig — fatal on one side of the library,
    fine on the other, for the same number. Now both warn for the same
    reason (memory per batch; history the model never trained on) and
    neither raises."""
    with pytest.warns(UserWarning, match=r"in_steps=13.*above the usual ceiling of 12"):
        _runnable(in_steps=13).validate()


def test_in_steps_at_or_below_the_threshold_is_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _runnable(in_steps=12).validate()  # must not raise or warn
