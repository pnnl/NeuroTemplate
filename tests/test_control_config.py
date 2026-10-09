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

from neurotemplate.common.config import ConfigError, CONTROL_CONFIG_DIR
from neurotemplate.common.config import load_config_from_cli
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.losses import resolve_objective
from control_fixtures import (
    BAND_GAINS, BAND_LOSSES, HIGH, LOW, SETPT, TARIFF, ZONE, DIST,
    make_control_cfg, make_nm_mapping,
)

_YAML = """
sysid_run_id: abc123
train_pred_steps: 24
run_dir: none
train_data_path: train.csv
dev_data_path:   dev.csv
test_data_path:  test.csv
train_reference_path: train_ref.csv
dev_reference_path:   dev_ref.csv
test_reference_path:  test_ref.csv
losses:
  comfort_lower: {column: lo, target: zone, map: relu, norm: 2}
  du:            {column: setpoint, map: first_difference, norm: 1}
gains: {comfort_lower: 5.0, du: 1.0e3}
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
    assert cfg.train_pred_steps == 24
    assert cfg.run_dir is None                      # "none" -> None on Optional
    assert cfg.losses["comfort_lower"]["map"] == "relu"
    assert cfg.gains["du"] == 1000.0


def test_a_gain_sweeps_with_set_without_restating_the_table(tmp_path):
    cfg = _load(tmp_path, _YAML, "gains={'comfort_lower': 2.0, 'du': 10}",
                "epochs=3")
    assert cfg.gains == {"comfort_lower": 2.0, "du": 10} and cfg.epochs == 3
    assert cfg.losses["comfort_lower"]["column"] == "lo"


def test_the_table_survives_the_mlflow_param_round_trip():
    """A run's config is rebuilt from its logged params — str() of each field
    — on reload, so the table must parse back to itself."""
    from neurotemplate.common.config import _parse_value, dict_to_dataclass
    cfg = make_control_cfg(reference_scale={TARIFF: 0.3},
                           reference_preview={LOW: 2})
    logged = {k: _parse_value(str(v)) for k, v in cfg.asdict().items()
              if k != "device"}
    rebuilt = dict_to_dataclass(logged, ControlConfig)
    for name in ("losses", "gains", "reference_scale", "reference_preview",
                 "train_reference_path"):
        assert getattr(rebuilt, name) == getattr(cfg, name)


def test_an_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        _load(tmp_path, _YAML + "not_a_key: 1\n")


# ── The table's own shape (validate) ─────────────────────────────

def _band(**changes):
    """The fixture's band objective with one entry's fields replaced."""
    losses = {k: dict(v) for k, v in BAND_LOSSES.items()}
    for name, fields in changes.items():
        losses[name] = {**losses.get(name, {}), **fields}
    return losses


@pytest.mark.parametrize("fields,message", [
    ({"losses": {}}, "losses is empty"),
    # A term with no gain would silently weigh nothing.
    ({"gains": {"comfort_lower": 1.0, "comfort_upper": 1.0}}, r"no entry in gains"),
    # `relu(column)` with no target is a bound on nothing.
    ({"losses": _band(comfort_lower={"target": None})}, "needs a `target`"),
    # Below 1 a power mean rewards spiky profiles — the opposite of a peak.
    ({"losses": _band(du={"norm": 0.5})}, "norm"),
    # A split with no reference file has no value for a reference column.
    ({"dev_reference_path": None}, "set together or not at all"),
    # The single most likely way to waste a campaign on the residual: every
    # delta snaps back to zero and nothing on any report says so.
    ({"setpoint_step": 0.5, "residual_delta": 0.2}, "smaller than setpoint_step"),
])
def test_validate_names_the_problem(fields, message):
    with pytest.raises(ValueError, match=message):
        make_control_cfg(**fields).validate()


@pytest.mark.parametrize("fields", [
    pytest.param({"setpoint_step": 0.5, "residual_delta": 0.5},
                 id="residual at exactly one grid step"),
    # 0 means "emit the reference exactly" — a floor to train nothing and
    # score, and not the same thing as a residual that quantizes away.
    pytest.param({"residual_delta": 0.0}, id="zero-width residual box"),
])
def test_validate_stays_quiet_on_a_legal_config(fields):
    make_control_cfg(**fields).validate()


# ── The names, against the surrogate (resolve_objective) ─────────

_COST = {"column": "power", "norm": 1, "weight": TARIFF}


@pytest.mark.parametrize("fields,message", [
    # A tariff has no target to borrow stats from, so it must say its scale.
    ({"losses": _band(cost=_COST), "gains": {**BAND_GAINS, "cost": 1.0}},
     "needs a `reference_scale` entry"),
    ({"reference_scale": {LOW: 1.0}}, "normalised with its stats"),
    # A weight is data: one the rollout predicts would carry gradient.
    ({"losses": _band(comfort_lower={"weight": ZONE})}, "must be data"),
    ({"losses": _band(comfort_lower={"target": SETPT})}, "must be a Y column"),
    # `units: normalized` compares a setpoint and a zone in two scales.
    ({"losses": _band(comfort_lower={"column": SETPT, "units": "normalized"})},
     "one scale only"),
    ({"losses": _band(comfort_upper={"column": LOW, "map": "relu"})},
     "two lower bounds"),
    ({"losses": {"du": BAND_LOSSES["du"]}, "gains": {"du": 1.0},
      "train_reference_path": None, "dev_reference_path": None,
      "test_reference_path": None},
     "No loss term bounds a Y column"),
    ({"train_reference_path": None, "dev_reference_path": None,
      "test_reference_path": None}, "no reference files are set"),
    ({"reference_preview": {DIST[0]: 2}}, "not reference columns"),
])
def test_resolve_names_the_problem(fields, message):
    cfg = make_control_cfg(**fields)
    with pytest.raises(ValueError, match=message):
        resolve_objective(cfg, make_nm_mapping())


def test_the_regulated_set_and_reference_order_are_derived():
    """Regulated outputs are the bound targets, in Y order; the reference
    columns are sorted, so reordering the YAML never moves a policy input."""
    reordered = dict(reversed(list(BAND_LOSSES.items())))
    a = resolve_objective(make_control_cfg(), make_nm_mapping())
    b = resolve_objective(make_control_cfg(losses=reordered), make_nm_mapping())
    assert a.regulated == b.regulated == [ZONE]
    assert a.r_columns == b.r_columns == [HIGH, LOW]
