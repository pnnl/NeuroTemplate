"""YAML null coercion: string 'none'/'null'/'None' → Python None on Optional fields."""

import pytest

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.common.config import ConfigError, dict_to_dataclass


def test_explicit_none_passes_through():
    cfg = dict_to_dataclass({
        "norm_type": None,
        "device": None,
        "grad_path": None,
    }, Config)
    assert cfg.norm_type is None
    assert cfg.device is None
    assert cfg.grad_path is None


@pytest.mark.parametrize("text", ["none", "null", "None", "NULL", "  None  "])
def test_string_null_coerced_on_optional(text):
    cfg = dict_to_dataclass({
        "norm_type": text,
        "device": text,
        "grad_path": text,
    }, Config)
    assert cfg.norm_type is None
    assert cfg.device is None
    assert cfg.grad_path is None


def test_real_string_not_coerced():
    cfg = dict_to_dataclass({"norm_type": "min_max"}, Config)
    assert cfg.norm_type == "min_max"


# ── Float format normalization ───────────────────────────────────
# PyYAML leaves 1e5/1E5/1.0e5 as strings and 10000 as int; all should
# resolve to the same float for a float-typed field.


@pytest.mark.parametrize("value", [100000, 100000.0, "10000.0", "1e5", "1E5", "1.0e5", "1.0E5"])
def test_float_formats_resolve_equal(value):
    cfg = dict_to_dataclass({"ref_gain": value}, Config)
    assert isinstance(cfg.ref_gain, float)
    assert cfg.ref_gain == float(value)


def test_optional_float_still_coerced():
    from neurotemplate.control.config import ControlConfig
    cfg = dict_to_dataclass({"test_band_lo": "1e1"}, ControlConfig)
    assert cfg.test_band_lo == 10.0


def test_optional_float_null_stays_none():
    from neurotemplate.control.config import ControlConfig
    cfg = dict_to_dataclass({"test_band_lo": "none"}, ControlConfig)
    assert cfg.test_band_lo is None


def test_bad_float_raises_config_error():
    with pytest.raises(ConfigError, match="float"):
        dict_to_dataclass({"ref_gain": "not_a_number"}, Config)


def test_set_override_float_scientific_notation():
    """--set ref_gain=1e5 → _parse_value floats it; --set ref_gain=100000
    parses to int, then dict_to_dataclass coerces the float field."""
    from neurotemplate.common.config import apply_overrides
    for text in ("1e5", "1E5", "1.0e5", "100000", "100000.0"):
        merged = apply_overrides({"ref_gain": 1.0}, [f"ref_gain={text}"])
        cfg = dict_to_dataclass(merged, Config)
        assert cfg.ref_gain == 100000.0


def test_unknown_key_raises():
    with pytest.raises(ConfigError):
        dict_to_dataclass({"not_a_real_field": 1}, Config)


def test_empty_split_paths_fail_validate():
    # Config has defaults for everything, but validate() rejects empty split paths.
    cfg = dict_to_dataclass({}, Config)
    with pytest.raises(ValueError, match="train_data_path"):
        cfg.validate()


# ── Multi-path data fields ───────────────────────────────────────


def _filled_cfg(**overrides):
    """Config with all required-for-validate fields populated."""
    base = {
        "train_data_path": "train.csv",
        "dev_data_path": "dev.csv",
        "test_data_path": "test.csv",
    }
    base.update(overrides)
    return dict_to_dataclass(base, Config)


def test_string_path_accepted():
    cfg = _filled_cfg()
    cfg.validate()  # must not raise
    assert cfg.train_data_path == "train.csv"


def test_list_path_accepted():
    cfg = _filled_cfg(train_data_path=["a.csv", "b.csv"])
    cfg.validate()
    assert cfg.train_data_path == ["a.csv", "b.csv"]


def test_paths_as_list_normalizes_both_shapes():
    from neurotemplate.common.config import paths_as_list
    assert paths_as_list("a.csv") == ["a.csv"]
    assert paths_as_list(["a.csv", "b.csv"]) == ["a.csv", "b.csv"]


def test_empty_list_rejected():
    cfg = _filled_cfg(train_data_path=[])
    with pytest.raises(ValueError, match="empty list"):
        cfg.validate()


def test_list_with_empty_string_rejected():
    cfg = _filled_cfg(train_data_path=["a.csv", ""])
    with pytest.raises(ValueError, match=r"train_data_path\[1\]"):
        cfg.validate()


def test_set_override_parses_json_list():
    """--set train_data_path='[...]' goes through apply_overrides →
    _parse_value, which JSON-decodes bracketed values into Python lists.
    Round-trips from MLflow's Python-repr form ('[...]' with single
    quotes) work through the ast.literal_eval fallback."""
    from neurotemplate.common.config import apply_overrides
    base = {
        "train_data_path": "old.csv",
        "dev_data_path": "dev.csv",
        "test_data_path": "test.csv",
    }
    merged = apply_overrides(base, ['train_data_path=["a.csv","b.csv"]'])
    cfg = dict_to_dataclass(merged, Config)
    cfg.validate()
    assert cfg.train_data_path == ["a.csv", "b.csv"]
