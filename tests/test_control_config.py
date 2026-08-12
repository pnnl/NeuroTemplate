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
        ["--config", str(path), "--set", "energy_weight=0",
         "--set", "epochs=3"],
        default_config_path=None, cls=ControlConfig, config_dir=CONTROL_CONFIG_DIR,
    )
    assert cfg.energy_weight == 0
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


# ── Grid-response and residual knobs ─────────────────────────────


@pytest.mark.parametrize("q", [0.0, 0.5, -1.0])
def test_validate_rejects_peak_q_below_one(q):
    """Below 1, M_q rewards spiky profiles — the opposite of the request."""
    cfg = _runnable_cfg()
    cfg.peak_q = q
    with pytest.raises(ValueError, match="peak_q must be >= 1"):
        cfg.validate()


@pytest.mark.parametrize("field", ["cost_weight", "peak_weight"])
def test_validate_rejects_negative_grid_weights(field):
    cfg = _runnable_cfg()
    setattr(cfg, field, -1.0)
    with pytest.raises(ValueError, match=field):
        cfg.validate()


def test_validate_rejects_a_residual_narrower_than_the_setpoint_grid():
    """The single most likely way to waste a campaign on the residual: every
    delta snaps back to zero and nothing on any report says so."""
    cfg = _runnable_cfg()
    cfg.setpoint_step = 0.5
    cfg.residual_delta = 0.2
    with pytest.raises(ValueError, match="smaller than setpoint_step"):
        cfg.validate()


def test_validate_accepts_a_residual_at_or_above_one_grid_step():
    cfg = _runnable_cfg()
    cfg.setpoint_step = 0.5
    cfg.residual_delta = 0.5
    cfg.validate()  # must not raise


def test_validate_accepts_a_zero_width_residual_box():
    """0 means 'emit the reference exactly' — a floor to train nothing and
    score, and not the same thing as a residual that quantizes away."""
    cfg = _runnable_cfg()
    cfg.residual_delta = 0.0
    cfg.validate()  # must not raise


def test_validate_rejects_a_negative_residual():
    cfg = _runnable_cfg()
    cfg.residual_delta = -0.5
    with pytest.raises(ValueError, match="residual_delta must be >= 0"):
        cfg.validate()


# ── Energy tariff (train/dev/test_price_path + price_scale) ──────


@pytest.mark.parametrize("field", [
    "train_price_path", "dev_price_path", "test_price_path",
])
def test_validate_rejects_a_partially_set_tariff(field):
    """A split with no price file would silently see no cost term."""
    cfg = _runnable_cfg()
    cfg.train_price_path = "train_price.csv"
    cfg.dev_price_path = "dev_price.csv"
    cfg.test_price_path = "test_price.csv"
    setattr(cfg, field, None)
    with pytest.raises(ValueError, match="together or not at all"):
        cfg.validate()


def test_validate_rejects_a_tariff_with_no_scale():
    cfg = _runnable_cfg()
    cfg.train_price_path = "train_price.csv"
    cfg.dev_price_path = "dev_price.csv"
    cfg.test_price_path = "test_price.csv"
    with pytest.raises(ValueError, match="price_scale"):
        cfg.validate()


def test_validate_rejects_a_scale_with_no_tariff():
    cfg = _runnable_cfg()
    cfg.price_scale = 0.30
    with pytest.raises(ValueError, match="price_scale is set"):
        cfg.validate()


def test_validate_rejects_a_non_positive_price_scale():
    cfg = _runnable_cfg()
    cfg.train_price_path = "train_price.csv"
    cfg.dev_price_path = "dev_price.csv"
    cfg.test_price_path = "test_price.csv"
    cfg.price_scale = 0.0
    with pytest.raises(ValueError, match="price_scale must be positive"):
        cfg.validate()


def test_validate_accepts_a_complete_tariff():
    cfg = _runnable_cfg()
    cfg.train_price_path = "train_price.csv"
    cfg.dev_price_path = "dev_price.csv"
    cfg.test_price_path = "test_price.csv"
    cfg.price_scale = 0.30
    cfg.price_preview = 48
    cfg.validate()  # must not raise


def test_validate_rejects_a_price_preview_with_no_tariff():
    """Previewing a signal that never reaches the rollout is a dead input
    width — caught at config load rather than at policy build."""
    cfg = _runnable_cfg()
    cfg.price_preview = 48
    with pytest.raises(ValueError, match="no tariff reaches the rollout"):
        cfg.validate()


def test_validate_rejects_an_unknown_price_alignment():
    cfg = _runnable_cfg()
    cfg.price_align = "hourly"
    with pytest.raises(ValueError, match="price_align must be one of"):
        cfg.validate()


def test_validate_rejects_a_negative_price_preview():
    cfg = _runnable_cfg()
    cfg.price_preview = -1
    with pytest.raises(ValueError, match="price_preview must be >= 0"):
        cfg.validate()
