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
)
from neurotemplate.control.config import ControlConfig
from neurotemplate.modeling.config import ModelingConfig as Config

_OPTIONAL = ("norm_type", "device", "grad_path")


# ── Null coercion on Optional fields ─────────────────────────────

@pytest.mark.parametrize("text", ["none"])
def test_every_spelling_of_null_lands_as_none(text):
    cfg = dict_to_dataclass({name: text for name in _OPTIONAL}, Config)
    assert all(getattr(cfg, name) is None for name in _OPTIONAL)


# ── Float formats ────────────────────────────────────────────────

@pytest.mark.parametrize("value", [100000, "1e5"])
def test_every_float_format_resolves_to_the_same_number(value):
    """PyYAML leaves 1e5 a string and 100000 an int; a float field is a float
    either way, whether it arrives from the file or from `--set`."""
    cfg = dict_to_dataclass({"ref_gain": value}, Config)
    assert isinstance(cfg.ref_gain, float) and cfg.ref_gain == 100000.0

    merged = apply_overrides({"ref_gain": 1.0}, parse_set_args([f"ref_gain={value}"]))
    assert dict_to_dataclass(merged, Config).ref_gain == 100000.0


# ── Multi-path data fields ───────────────────────────────────────

def test_set_parses_a_json_list():
    """`--set draws='[...]'` goes through `_parse_value`, which JSON-decodes bracketed
    values. MLflow's Python-repr form (single quotes) round-trips via the
    `ast.literal_eval` fallback."""
    merged = apply_overrides({"draws": ["old.csv"]},
                             parse_set_args(['draws=["a.csv","b.csv"]']))
    assert dict_to_dataclass(merged, ControlConfig).draws == ["a.csv", "b.csv"]


def test_a_dotted_override_sets_one_entry_and_keeps_the_rest():
    merged = apply_overrides({"gains": {"cost": 1.0, "du": 0.5}},
                             parse_set_args(["gains.cost=2.0"]))
    assert merged == {"gains": {"cost": 2.0, "du": 0.5}}
    with pytest.raises(ConfigError, match="gains.typo"):
        apply_overrides({"gains": {"cost": 1.0}}, {"gains.typo": 2.0})
