"""
Config loading: YAML's null spellings, float formats, and multi-path fields.

PyYAML hands us whatever the file said — `none` as a string, `1e5` as a
string, `10000` as an int — so `dict_to_dataclass` is what makes a field's
declared type true. These tests are that coercion, plus the split-path
fields, which accept one path or many.
"""


import pytest

from neurotemplate.common.config import (
    ConfigError, apply_overrides, dict_to_dataclass, parse_set_args,
    paths_as_list,
)
from neurotemplate.modeling.config import ModelingConfig as Config

_OPTIONAL = ("norm_type", "device", "grad_path")


def _filled(**overrides):
    """Config with every required-for-validate field populated."""
    base = {"train_data_path": "train.csv", "dev_data_path": "dev.csv",
            "test_data_path": "test.csv", "history": {"y": 0}}
    return dict_to_dataclass({**base, **overrides}, Config)


# ── Null coercion on Optional fields ─────────────────────────────

@pytest.mark.parametrize("text", [None, "none", "  None  "])
def test_every_spelling_of_null_lands_as_none(text):
    cfg = dict_to_dataclass({name: text for name in _OPTIONAL}, Config)
    assert all(getattr(cfg, name) is None for name in _OPTIONAL)


def test_a_real_string_is_not_coerced():
    assert dict_to_dataclass({"norm_type": "min_max"}, Config).norm_type == "min_max"


def test_an_optional_float_keeps_both_readings():
    assert dict_to_dataclass({"steps_per_hour": "1.2e1"}, Config).steps_per_hour == 12.0
    assert dict_to_dataclass({"steps_per_hour": "none"}, Config).steps_per_hour is None


# ── Float formats ────────────────────────────────────────────────

@pytest.mark.parametrize("value", [100000, "1e5", "1.0E5"])
def test_every_float_format_resolves_to_the_same_number(value):
    """PyYAML leaves 1e5 a string and 100000 an int; a float field is a float
    either way, whether it arrives from the file or from `--set`."""
    cfg = dict_to_dataclass({"ref_gain": value}, Config)
    assert isinstance(cfg.ref_gain, float) and cfg.ref_gain == 100000.0

    merged = apply_overrides({"ref_gain": 1.0}, parse_set_args([f"ref_gain={value}"]))
    assert dict_to_dataclass(merged, Config).ref_gain == 100000.0


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


def test_set_parses_a_json_list():
    """`--set train_data_path='[...]'` goes through `_parse_value`, which
    JSON-decodes bracketed values. MLflow's Python-repr form (single quotes)
    round-trips via the `ast.literal_eval` fallback."""
    merged = apply_overrides(
        {"train_data_path": "old.csv", "dev_data_path": "dev.csv",
         "test_data_path": "test.csv", "history": {"y": 0}},
        parse_set_args(['train_data_path=["a.csv","b.csv"]']))
    cfg = dict_to_dataclass(merged, Config)
    cfg.validate()
    assert cfg.train_data_path == ["a.csv", "b.csv"]
