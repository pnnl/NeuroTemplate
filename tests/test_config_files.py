"""The template's default config files: every setting, at its default, as the
dataclass says; and a custom file that sets nothing."""
from dataclasses import fields

import pytest

from neurotemplate.common.config import dict_to_dataclass, load_yaml
from neurotemplate.common.config_docs import default_config_files
from neurotemplate.control.config import ControlConfig
from neurotemplate.init import TEMPLATE
from neurotemplate.modeling.config import ModelingConfig

DEFAULTS = {"model_configs/modeling.yaml": ModelingConfig,
            "control_configs/control.yaml": ControlConfig}
NOT_WRITTEN = {"savedir", "steps_per_hour"}


def test_the_default_files_are_what_the_dataclasses_generate():
    for name, text in default_config_files().items():
        assert (TEMPLATE / name).read_text(encoding="utf-8") == text, (
            f"{name} is stale: `python -m neurotemplate.common.config_docs` rewrites it.")


@pytest.mark.parametrize("name", DEFAULTS)
def test_a_default_file_names_every_setting_at_its_default(name):
    cls = DEFAULTS[name]
    written = load_yaml(TEMPLATE / name)
    assert set(written) == {f.name for f in fields(cls)} - NOT_WRITTEN
    loaded, fresh = dict_to_dataclass(written, cls), cls()
    for f in fields(cls):
        if f.name not in NOT_WRITTEN:
            assert getattr(loaded, f.name) == getattr(fresh, f.name), f.name


@pytest.mark.parametrize("side", ["model_configs", "control_configs"])
def test_a_custom_file_sets_nothing_and_fits_the_width(side):
    path = TEMPLATE / side / "custom.yaml"
    assert load_yaml(path) == {}
    assert all(len(line) <= 80 for line in path.read_text().splitlines())
