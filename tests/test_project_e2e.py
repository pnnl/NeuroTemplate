"""What a person actually types: the guides as written, on the practice record their
examples use, and a project on column roles alone with, for a controller, an objective."""

import json
import subprocess
import sys

import mlflow
import pandas as pd
import pytest
import yaml

from neurotemplate.cli import main as nt

ZONE, HEAT, POWER = "Zone 1 Temp", "Zone 1 Heat", "Zone 1 Power"
COOL_SP, HEAT_SP = "Zone 1 Cooling SP", "Zone 1 Heating SP"
OUTDOOR, SOLAR = "Outdoor Temp", "Solar"


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


def _assign(columns: dict) -> None:
    data = yaml.safe_load(open("data.yaml").read())
    data["columns"] = columns
    open("data.yaml", "w").write(yaml.safe_dump(data, allow_unicode=True))


def _add_band() -> None:
    """7-first-controller.md, step 1: the band in every split, in no role."""
    for split in ("train", "dev", "test"):
        df = pd.read_csv(f"data/{split}.csv")
        df["Band Low"], df["Band High"] = 21.0, 24.0
        df.to_csv(f"data/{split}.csv", index=False)


QUICK = ["--set", "iters=4", "--set", "plot=false"]


@pytest.fixture
def practice_project(tmp_path, monkeypatch):
    """`nt init`, then the project's own `guides/make_data.py`, run as guide 1 says."""
    monkeypatch.chdir(tmp_path)
    nt(["init", "--no-warmup"])
    subprocess.run([sys.executable, "guides/make_data.py"], check=True, capture_output=True)
    return tmp_path


def test_the_practice_record_refuses_to_overwrite_a_record(practice_project):
    before = (practice_project / "data/train.csv").read_bytes()
    rerun = subprocess.run([sys.executable, "guides/make_data.py"], capture_output=True,
                           text=True)
    assert rerun.returncode != 0 and "nothing written" in rerun.stderr
    assert (practice_project / "data/train.csv").read_bytes() == before


def test_the_guides_run_as_written_from_a_first_model_to_an_export(practice_project):
    # 1-first-model.md: no roles yet, so it stops and lists every column.
    with pytest.raises(SystemExit, match="assigns no columns") as stop:
        nt(["model", "train"])
    assert POWER in str(stop.value)
    _assign({"Y": [ZONE], "U": [HEAT], "D": [OUTDOOR, SOLAR]})
    nt(["model", "train", *QUICK])

    # 7-first-controller.md: the band in the record, the objective in custom.yaml.
    _add_band()
    with open("control_configs/custom.yaml", "a") as custom:
        custom.write("\nlosses:\n"
                     f"  too_cold: {{column: Band Low, target: {ZONE}, map: relu, norm: 1}}\n"
                     f"  too_hot: {{column: Band High, target: {ZONE}, map: nrelu, norm: 1}}\n"
                     f"  smooth: {{column: {HEAT}, map: first_difference, norm: 1}}\n"
                     "gains: {too_cold: 10.0, too_hot: 10.0, smooth: 0.1}\n")
    nt(["control", "train", "--config", "custom", *QUICK])   # takes the latest model run

    # 8-export.md.
    nt(["export", "policy", "--run_id", _latest_run(), "--out", "delivery"])
    inputs = json.loads((practice_project / "delivery/policy/policy.json").read_text())["inputs"]
    assert inputs["y"]["columns"] == [ZONE]


def test_a_project_runs_on_column_roles_alone_and_a_controller_on_an_objective(
        practice_project):
    _add_band()
    _assign({"Y": [ZONE, POWER], "U": [COOL_SP, HEAT_SP], "D": [OUTDOOR, SOLAR]})
    nt(["model", "train", *QUICK])

    objective = {
        "losses": {"low": {"column": "Band Low", "target": ZONE, "map": "relu", "norm": 1},
                   "high": {"column": "Band High", "target": ZONE, "map": "nrelu", "norm": 1}},
        "gains": {"low": 1.0, "high": 1.0},
    }
    (practice_project / "control_configs/control.yaml").write_text(
        yaml.safe_dump(objective, allow_unicode=True))
    nt(["control", "train", *QUICK])
