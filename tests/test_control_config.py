"""
Loading and validating a ControlConfig and its declared objective.

Almost everything here is one shape: set some fields, then assert that
`validate()` (the table's own shape) or `resolve_objective` (its names,
against the surrogate's columns) either names the problem or stays quiet.
The rows are the checks whose absence would let a run train silently on the
wrong objective; closed-set and sign checks fail loudly at build time
regardless and are not tabulated.
"""
import pytest

from neurotemplate.common.config import CONTROL_CONFIG_DIR, ConfigError, time_steps
from neurotemplate.common.config import load_config_from_cli
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.losses import fill_reference_scale, resolve_objective
from control_fixtures import (
    BAND_GAINS, BAND_LOSSES, HIGH, LOW, SUPPLY_T, TARIFF, ZONE, make_control_cfg, make_nm_mapping,
)

_YAML = """
sysid_run_id: abc123
horizon: 2h
steps_per_hour: 12
run_dir: none
losses:
  comfort_lower: {column: lo, target: zone, map: relu, norm: 2}
  du:            {column: setpoint, map: first_difference, norm: 1}
gains: {comfort_lower: 5.0, du: 1.0e3}
policy_window:
  zone:     {past: 15m}
  setpoint: {past: 5m}
  oat:      {past: 0, future: 1h}
  lo:       {past: 0, future: 0}
"""


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


# ── Loading ──────────────────────────────────────────────────────

def test_yaml_fills_what_it_names_and_defaults_the_rest(tmp_path):
    cfg = _load(tmp_path, _YAML)
    assert cfg.sysid_run_id == "abc123"
    assert cfg.horizon == "2h" and cfg.horizon_steps == 24
    assert cfg.run_dir is None                      # "none" -> None on Optional
    assert cfg.losses["comfort_lower"]["map"] == "relu"
    assert cfg.gains["du"] == 1000.0


def test_a_preview_past_the_training_rollout_is_refused(tmp_path):
    cfg = _load(tmp_path, _YAML, "horizon=30m")   # `oat` previews an hour
    cfg.fill_common_windows(288)
    with pytest.raises(ValueError, match="never rewarded"):
        cfg.check_windows()      # once the surrogate has said the timestep


@pytest.mark.parametrize("value,steps", [("5h", 60), ("20m", 4), ("300s", 1), (0, 0), ("0", 0),
                                         ("1.5h", 18), ("23h55m", 287)])
def test_a_window_is_a_time(value, steps):
    assert time_steps(value, 12.0, "horizon") == steps


@pytest.mark.parametrize("value,message", [
    ("7m", "not a whole number"), (3, "is a time"), ("3 h", "is a time"), ("3d", "is a time"),
])
def test_a_window_off_the_timestep_or_without_a_unit_is_refused(value, message):
    with pytest.raises(ConfigError, match=message):
        time_steps(value, 12.0, "horizon")


def test_the_table_survives_the_mlflow_param_round_trip():
    """A run's config is rebuilt from its logged params — str() of each field
    — on reload, so the table must parse back to itself."""
    from neurotemplate.common.config import _parse_value, dict_to_dataclass
    cfg = make_control_cfg(reference_scale={TARIFF: 0.3},
                           policy_window={ZONE: {"past": "10m"}, LOW: {"past": 0, "future": "10m"}})
    logged = {k: _parse_value(str(v)) for k, v in cfg.asdict().items()
              if k != "device"}
    rebuilt = dict_to_dataclass(logged, ControlConfig)
    for name in ("losses", "gains", "reference_scale", "policy_window",
                 "draws"):
        assert getattr(rebuilt, name) == getattr(cfg, name)


# ── The table's own shape (validate) ─────────────────────────────

def _band(**changes):
    """The fixture's band objective with one entry's fields replaced."""
    losses = {k: dict(v) for k, v in BAND_LOSSES.items()}
    for name, fields in changes.items():
        losses[name] = {**losses.get(name, {}), **fields}
    return losses


# ── The names, against the surrogate (resolve_objective) ─────────

_COST = {"column": "power", "norm": 1, "weight": TARIFF}
#: The same comfort band over a second Y channel.
_SUPPLY_BAND = {
    "supply_lower": {"column": LOW, "target": SUPPLY_T, "map": "relu", "norm": 2},
    "supply_upper": {"column": HIGH, "target": SUPPLY_T, "map": "nrelu", "norm": 2},
}
_SUPPLY_GAINS = {**BAND_GAINS, "supply_lower": 1.0, "supply_upper": 1.0}


@pytest.mark.parametrize("fields,message", [
    # A weight is data: one the rollout predicts would carry gradient.
    ({"losses": _band(comfort_lower={"weight": ZONE})}, "must be data"),
])
def test_resolve_names_the_problem(fields, message):
    cfg = make_control_cfg(policy_window={}, **fields)   # not what is under test
    with pytest.raises(ValueError, match=message):
        resolve_objective(cfg, make_nm_mapping())


def test_a_tariffs_scale_is_its_largest_magnitude_in_the_training_record():
    """Set by a rule and written back, so the run logs it and a reload reads it; a band
    borrows its target's stats instead, and one you set stays yours."""
    import pandas as pd
    cfg = make_control_cfg(losses=_band(cost=_COST), gains={**BAND_GAINS, "cost": 1.0},
                           policy_window={})
    objective = resolve_objective(cfg, make_nm_mapping())
    frame = pd.DataFrame({TARIFF: [0.1, -0.6, 0.2], LOW: [20.0] * 3, HIGH: [25.0] * 3})
    fill_reference_scale(objective, cfg, [frame])
    assert cfg.reference_scale == {TARIFF: 0.6}
    cfg.reference_scale[TARIFF] = 0.3
    fill_reference_scale(objective, cfg, [frame])
    assert cfg.reference_scale == {TARIFF: 0.3}


def test_one_band_column_may_bound_several_outputs():
    """One comfort band over every zone, not 2N copies of it (field report:
    5 and 17 zones). It takes the first target's stats, in Y order."""
    cfg = make_control_cfg(losses={**BAND_LOSSES, **_SUPPLY_BAND}, gains=_SUPPLY_GAINS)
    objective = resolve_objective(cfg, make_nm_mapping())
    assert objective.regulated == [ZONE, SUPPLY_T]
    assert objective.r_columns == [HIGH, LOW]
    assert objective.bound_to == {LOW: ZONE, HIGH: ZONE}


def test_the_regulated_set_and_reference_order_are_derived():
    """Regulated outputs are the bound targets, in Y order; the reference
    columns are sorted, so reordering the YAML never moves a policy input."""
    reordered = dict(reversed(list(BAND_LOSSES.items())))
    a = resolve_objective(make_control_cfg(), make_nm_mapping())
    b = resolve_objective(make_control_cfg(losses=reordered), make_nm_mapping())
    assert a.regulated == b.regulated == [ZONE]
    assert a.r_columns == b.r_columns == [HIGH, LOW]
