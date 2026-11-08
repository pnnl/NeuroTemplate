"""What a person actually types: a project on synthetic data with zero edits, and one on
your own data with column roles and, for a controller, an objective. Nothing else."""

import json
import sys

import mlflow
import pytest
import yaml

from neurotemplate.cli import main as nt
from neurotemplate.example import (
    BAND_HIGH, BAND_LOW, HEAT, OUTDOOR, POWER, SETPOINT, SOLAR, SPLITS, TARIFF, ZONE, thermal_plant,
)


@pytest.fixture(autouse=True)
def _isolated_hooks():
    """The project's own `nt_model` and `nt_control`, for this test only."""
    previous_uri = mlflow.get_tracking_uri()
    saved = {name: sys.modules.pop(name, None) for name in ("nt_model", "nt_control")}
    saved_path = list(sys.path)
    yield
    sys.path[:] = saved_path
    for name, module in saved.items():
        sys.modules.pop(name, None)
        if module is not None:
            sys.modules[name] = module
    mlflow.set_tracking_uri(previous_uri)


def _latest_run() -> str:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    return mlflow.search_runs(search_all_experiments=True, order_by=["start_time DESC"],
                              max_results=1).iloc[0].run_id


QUICK = ["--set", "iters=4", "--set", "plot=false"]


def test_the_thermal_example_runs_model_control_and_export_with_zero_edits(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    nt(["example", "thermal", "project"])
    monkeypatch.chdir(tmp_path / "project")
    nt(["model", "train", *QUICK])
    nt(["control", "train", *QUICK])                  # takes the latest model run
    out = capsys.readouterr().out
    assert "You set:" in out and "iters: 4" in out and "Defaults used:" in out
    assert "Not read by the policy" in out and HEAT in out
    nt(["export", "policy", "--run_id", _latest_run(), "--out", "delivery"])
    inputs = json.loads((tmp_path / "project/delivery/policy/policy.json").read_text())["inputs"]
    assert inputs["y"]["columns"] == [ZONE]          # no heat or meter for anyone to rebuild


def test_a_project_runs_on_column_roles_alone_and_a_controller_on_an_objective(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    nt(["init"])
    for split, days, seed, start in SPLITS:
        thermal_plant(days, seed, start).to_csv(tmp_path / "data" / f"{split}.csv")

    # No roles yet: it stops and lists every column, ready to paste.
    with pytest.raises(SystemExit, match="assigns no columns") as stop:
        nt(["model", "train"])
    assert POWER in str(stop.value)

    data = yaml.safe_load((tmp_path / "data.yaml").read_text())
    data["columns"] = {"Y": [ZONE, HEAT, POWER], "U": [SETPOINT], "D": [OUTDOOR, SOLAR]}
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True))
    nt(["model", "train", *QUICK])
    out = capsys.readouterr().out
    assert "No physics checks will run: define `synth_cfg`" in out
    assert TARIFF in out                             # listed as not read by the model

    objective = {
        "losses": {"low": {"column": BAND_LOW, "target": ZONE, "map": "relu", "norm": 1},
                   "high": {"column": BAND_HIGH, "target": ZONE, "map": "nrelu", "norm": 1}},
        "gains": {"low": 1.0, "high": 1.0},
    }
    (tmp_path / "control_configs/control.yaml").write_text(
        yaml.safe_dump(objective, allow_unicode=True))
    nt(["control", "train", *QUICK])
    assert "Finished control run" in capsys.readouterr().out
