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
"""


def _write(tmp_path, text):
    p = tmp_path / "control.yaml"
    p.write_text(text)
    return p


def test_load_defaults_and_yaml(tmp_path):
    path = _write(tmp_path, _YAML)
    cfg, _ = load_config_from_cli(
        ["--config", str(path)], default_config_path=None, cls=ControlConfig, config_dir=CONTROL_CONFIG_DIR,
    )
    assert isinstance(cfg, ControlConfig)
    assert cfg.sysid_run_id == "abc123"
    assert cfg.train_pred_steps == 24
    assert cfg.run_dir is None  # "none" -> None on Optional field
    assert cfg.comfort_weight == ControlConfig.comfort_weight


def test_set_override(tmp_path):
    path = _write(tmp_path, _YAML)
    cfg, _ = load_config_from_cli(
        ["--config", str(path), "--set", "effort_weight=0",
         "--set", "epochs=3"],
        default_config_path=None, cls=ControlConfig, config_dir=CONTROL_CONFIG_DIR,
    )
    assert cfg.effort_weight == 0
    assert cfg.epochs == 3


def test_unknown_key_rejected(tmp_path):
    path = _write(tmp_path, _YAML + "not_a_key: 1\n")
    with pytest.raises(ConfigError):
        load_config_from_cli(
            ["--config", str(path)], default_config_path=None,
            cls=ControlConfig, config_dir=CONTROL_CONFIG_DIR,
        )


def test_validate_requires_run_id():
    with pytest.raises(ValueError, match="sysid_run_id"):
        ControlConfig().validate()


def test_validate_band_order():
    cfg = ControlConfig()
    cfg.sysid_run_id = "x"
    # Data paths must be set for validate() to reach the band check.
    cfg.train_data_path = "train.csv"
    cfg.dev_data_path = "dev.csv"
    cfg.test_data_path = "test.csv"
    cfg.band_lo, cfg.band_hi = 24.0, 20.0
    with pytest.raises(ValueError, match="band_lo"):
        cfg.validate()


def _runnable_cfg() -> ControlConfig:
    cfg = ControlConfig()
    cfg.sysid_run_id = "x"
    cfg.train_data_path = "train.csv"
    cfg.dev_data_path = "dev.csv"
    cfg.test_data_path = "test.csv"
    return cfg


@pytest.mark.parametrize("field,bad", [
    ("policy", "bogus"),
    ("policy_nonlin", "swish"),
    ("policy_bound_method", "clip"),
])
def test_validate_rejects_unknown_policy_options(field, bad):
    cfg = _runnable_cfg()
    setattr(cfg, field, bad)
    with pytest.raises(ValueError, match=field):
        cfg.validate()


def test_validate_accepts_linear_policy():
    cfg = _runnable_cfg()
    cfg.policy = "linear"
    cfg.policy_bound_method = "relu_clamp"
    cfg.validate()  # must not raise
