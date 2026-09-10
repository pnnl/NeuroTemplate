"""
`run_summary.json` — the machine-readable record of what a run actually ran.

The point of the file is that a downstream audit compares a value to itself
instead of to a retyped copy, so the tests that matter are: the params in
the summary are the params MLflow logged, and the envelope's shape is
pinned.

Includes a snapshot test against a golden file. To regenerate it (e.g. after
an intentional contract change), run:

    REGEN_GOLDEN=1 uv run pytest tests/test_run_summary.py -k snapshot
"""

import json
import os
from pathlib import Path

from neurotemplate.common.run_summary import (
    RUN_SUMMARY_NAME,
    build_run_summary,
    write_run_summary,
)
from neurotemplate.modeling.config import ModelingConfig
from control_fixtures import make_control_cfg

GOLDEN_DIR = Path(__file__).parent / "golden"

_NM = {"Y": ["zone [°C]"], "U": ["setpoint [°C]"], "D": ["outdoor [°C]"],
       "t": ["sec_elapsed"]}


class _FakeLogger:
    """Stands in for MLFlowLogger: a run id and a dict sink."""

    class _Run:
        class info:
            run_id = "3f2a1c"

    def __init__(self):
        self.run = self._Run()
        self.logged = {}

    def log_dict(self, data, name):
        self.logged[name] = data


def _mlflow_params(cfg) -> dict:
    """What MLFlowLogger.log_parameters hands to mlflow.log_params."""
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


def test_write_logs_a_json_parseable_artifact_under_the_run():
    logger = _FakeLogger()
    written = write_run_summary(
        logger, stage="model", config_path="model_configs/modeling.yaml",
        cfg=ModelingConfig(), metrics={"test_rmse": 0.63,
                                       "worst_channel_r2": 0.71},
        mapping=_NM,
    )
    assert logger.logged[RUN_SUMMARY_NAME] == written
    assert json.loads(json.dumps(written))["run_id"] == "3f2a1c"


def test_snapshot_matches_golden():
    """Pin the envelope. Update via REGEN_GOLDEN=1."""
    summary = build_run_summary(
        stage="model", run_id="3f2a1c",
        config_path="model_configs/modeling.yaml", cfg=ModelingConfig(),
        metrics={"test_rmse": 0.63, "worst_channel_r2": 0.71}, mapping=_NM,
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / "run_summary_model.json"

    if os.environ.get("REGEN_GOLDEN") or not golden_path.exists():
        golden_path.write_text(text)
        return

    assert text == golden_path.read_text(), (
        "run_summary.json envelope changed.\n"
        f"If intentional, regenerate via: REGEN_GOLDEN=1 pytest {__file__}"
    )
