"""
Config loading: YAML's null spellings, float formats, and multi-path fields.

PyYAML hands us whatever the file said — `none` as a string, `1e5` as a
string, `10000` as an int — so `dict_to_dataclass` is what makes a field's
declared type true. These tests are that coercion, plus the split-path
fields, which accept one path or many.
"""

import warnings

import pytest

from neurotemplate.common.config import (
    ConfigError, apply_overrides, dict_to_dataclass, paths_as_list,
)
from neurotemplate.control.config import ControlConfig
from neurotemplate.modeling.config import ModelingConfig as Config

_OPTIONAL = ("norm_type", "device", "grad_path")


def _filled(**overrides):
    """Config with every required-for-validate field populated."""
    base = {"train_data_path": "train.csv", "dev_data_path": "dev.csv",
            "test_data_path": "test.csv"}
    return dict_to_dataclass({**base, **overrides}, Config)


# ── Null coercion on Optional fields ─────────────────────────────

@pytest.mark.parametrize("text", [None, "none", "null", "None", "NULL", "  None  "])
def test_every_spelling_of_null_lands_as_none(text):
    cfg = dict_to_dataclass({name: text for name in _OPTIONAL}, Config)
    assert all(getattr(cfg, name) is None for name in _OPTIONAL)


def test_a_real_string_is_not_coerced():
    assert dict_to_dataclass({"norm_type": "min_max"}, Config).norm_type == "min_max"


def test_an_optional_float_keeps_both_readings():
    assert dict_to_dataclass({"test_band_lo": "1e1"}, ControlConfig).test_band_lo == 10.0
    assert dict_to_dataclass({"test_band_lo": "none"}, ControlConfig).test_band_lo is None


# ── Float formats ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "value", [100000, 100000.0, "100000.0", "1e5", "1E5", "1.0e5", "1.0E5"])
def test_every_float_format_resolves_to_the_same_number(value):
    """PyYAML leaves 1e5 a string and 100000 an int; a float field is a float
    either way, whether it arrives from the file or from `--set`."""
    cfg = dict_to_dataclass({"ref_gain": value}, Config)
    assert isinstance(cfg.ref_gain, float) and cfg.ref_gain == 100000.0

    merged = apply_overrides({"ref_gain": 1.0}, [f"ref_gain={value}"])
    assert dict_to_dataclass(merged, Config).ref_gain == 100000.0


def test_a_number_that_is_not_one_says_so():
    with pytest.raises(ConfigError, match="float"):
        dict_to_dataclass({"ref_gain": "not_a_number"}, Config)


def test_an_unknown_key_is_rejected():
    with pytest.raises(ConfigError):
        dict_to_dataclass({"not_a_real_field": 1}, Config)


# ── Multi-path data fields ───────────────────────────────────────

def test_a_split_takes_one_path_or_many():
    assert _filled().train_data_path == "train.csv"
    assert _filled(train_data_path=["a.csv", "b.csv"]).train_data_path == ["a.csv", "b.csv"]
    for cfg in (_filled(), _filled(train_data_path=["a.csv", "b.csv"])):
        cfg.validate()                                   # must not raise
    # Downstream code only ever sees the list form.
    assert paths_as_list("a.csv") == ["a.csv"]
    assert paths_as_list(["a.csv", "b.csv"]) == ["a.csv", "b.csv"]


@pytest.mark.parametrize("value,message", [
    ("", "train_data_path"),
    ([], "empty list"),
    (["a.csv", ""], r"train_data_path\[1\]"),
])
def test_a_split_with_no_usable_path_says_which(value, message):
    with pytest.raises(ValueError, match=message):
        _filled(train_data_path=value).validate()


def test_set_parses_a_json_list():
    """`--set train_data_path='[...]'` goes through `_parse_value`, which
    JSON-decodes bracketed values. MLflow's Python-repr form (single quotes)
    round-trips via the `ast.literal_eval` fallback."""
    merged = apply_overrides(
        {"train_data_path": "old.csv", "dev_data_path": "dev.csv",
         "test_data_path": "test.csv"},
        ['train_data_path=["a.csv","b.csv"]'])
    cfg = dict_to_dataclass(merged, Config)
    cfg.validate()
    assert cfg.train_data_path == ["a.csv", "b.csv"]


# ── in_steps: an advisory warning, not a hard ceiling ─────────────

def test_in_steps_above_the_threshold_warns_instead_of_raising():
    """MAX_IN_STEPS used to be a hard ValueError ceiling (check_max); it is
    now advisory, since the real risks — memory, a window longer than the
    training horizon — don't justify refusing a value the user set on
    purpose. `validate()` must not raise, and training can proceed."""
    with pytest.warns(UserWarning, match=r"in_steps=13.*above the usual ceiling of 12"):
        _filled(in_steps=13).validate()


def test_in_steps_at_or_below_the_threshold_is_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _filled(in_steps=12).validate()  # must not raise or warn
