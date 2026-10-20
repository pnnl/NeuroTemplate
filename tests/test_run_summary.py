"""
`run_summary.json` — the machine-readable record of what a run actually ran.

The point of the file is that a downstream audit compares a value to itself
instead of to a retyped copy, so the tests that matter are: the params in
the summary are the params MLflow logged.
"""


from neurotemplate.common.run_summary import (
    build_run_summary,
)
from neurotemplate.modeling.config import ModelingConfig
from control_fixtures import make_control_cfg

_NM = {"Y": ["zone [°C]"], "U": ["setpoint [°C]"], "D": ["outdoor [°C]"],
       "t": ["sec_elapsed"]}


def _mlflow_params(cfg) -> dict:
    """What RunLogger.log_parameters hands to mlflow.log_params."""
    return {k: getattr(cfg, k) for k in vars(cfg)}


def test_model_params_are_the_params_mlflow_logged():
    cfg = ModelingConfig()
    summary = build_run_summary(
        stage="model", run_id="abc", config_path="model_configs/modeling.yaml",
        cfg=cfg, metrics={"test_rmse": 0.63, "worst_channel_r2": 0.71},
        mapping=_NM,
    )
    logged = _mlflow_params(cfg)
    assert summary["params"] == {k: logged[k] for k in summary["params"]}


def test_control_params_are_the_params_mlflow_logged():
    cfg = make_control_cfg()
    summary = build_run_summary(
        stage="control", run_id="abc",
        config_path="control_configs/control.yaml", cfg=cfg,
        metrics={"control_violation_rate": 0.02, "control_mean_abs_du": 0.31},
    )
    logged = _mlflow_params(cfg)
    assert summary["params"] == {k: logged[k] for k in summary["params"]}
    # Control names its surrogate and carries no channel mapping of its own.
    assert summary["params"]["sysid_run_id"] == cfg.sysid_run_id
    assert "mapping" not in summary
