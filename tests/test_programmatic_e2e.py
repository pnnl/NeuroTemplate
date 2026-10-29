"""The Python entry points chain the way the CLI commands do.

A surrogate `train_model` trains is the `sysid_run_id` a `train_control` run
loads, both runs reload by the ids they return, and `evaluate_run` scores
the surrogate as its training run did — one real MLflow store, one tiny run
each. The CLIs are thin wrappers over these calls, so this is also the
wiring `nt model train`, `nt model load` and `nt control train` depend on.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest

from neurotemplate.common.config import load_config
from neurotemplate.control.load import load_trained_policy
from neurotemplate.control.train import train_control
from neurotemplate.common.data_utils import get_data
from neurotemplate.export.dynamics import export_model
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.contract import contract_model
from neurotemplate.modeling.load import evaluate_run, load_trained_dynamics
from neurotemplate.modeling.train import train_model
from control_fixtures import (
    make_control_cfg, make_nm_mapping, make_raw_df, make_reference_df, write_splits,
)
from model_fixtures import uniform_history

FIXTURES = Path(__file__).parent / "fixtures"

# Every split reads the same plant file.
pytestmark = pytest.mark.filterwarnings("ignore:.*timestamps from training")


@pytest.fixture(autouse=True)
def _restore_tracking_uri():
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)
    for name in list(sys.modules):
        if name.startswith(("modeling_user_defined_", "control_user_defined_")):
            del sys.modules[name]


#: A model of the user's own, appended to the fixture hooks: stateful, in
#: physical units, behind the contract as written (`modeling.contract`).
USER_MODEL = """

import torch.nn as nn


class EwmaZone(nn.Module):
    \"\"\"next y = last y + rate * (memory - last y) + gain(u, d), the memory
    an EWMA of y carried as rollout state.\"\"\"

    def __init__(self, ny, nu, nd):
        super().__init__()
        self.rate = nn.Parameter(torch.full((ny,), 0.1))
        self.gain = nn.Linear(nu + nd, ny)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)
        self.memory = None

    def rollout_state(self):
        return self.memory

    def set_rollout_state(self, state):
        self.memory = state

    def forward(self, y, u, d):
        last = y[:, -1]
        memory = y.mean(dim=1) if self.memory is None else self.memory
        self.memory = 0.9 * memory + 0.1 * last
        return last + self.rate * (memory - last) + self.gain(torch.cat([u[:, -1], d[:, -1]], -1))


def model_families():
    return {"ewma_zone": lambda cfg, nm, stats: EwmaZone(len(nm["Y"]), len(nm["U"]), len(nm["D"]))}


def data_head_config(nm_mapping, norm_stats):
    return HeadStack([RegressionHead() for _ in nm_mapping["Y"]], nm_mapping["Y"])
"""


def _surrogate_then_policy(tmp_path, model: str, hooks_path: Path):
    """Train a surrogate and a policy on it through the Python entry points,
    as `nt model train` and `nt control train` do; return both runs."""
    location = f"sqlite:///{tmp_path / 'mlflow.db'}"
    plant = make_raw_df(n_rows=160)
    plant_path = tmp_path / "plant.csv"
    plant.drop(columns="sec_elapsed").rename_axis("timestamp").to_csv(plant_path)

    # A JSON object is a YAML flow mapping.
    history = json.dumps(uniform_history(make_nm_mapping(), 4), ensure_ascii=False)
    config_path = tmp_path / "modeling.yaml"
    config_path.write_text(f"""
train_data_path: {plant_path}
dev_data_path: {plant_path}
test_data_path: {plant_path}
user_hooks_path: {hooks_path}
model: {model}
history: {history}
log2_hidden_size: 2
num_layers: 1
epochs: 50
warmup: 0
train_pred_steps: 4
dev_pred_steps: 4
test_window: 24
test_stride: 24
batch_size: 8
norm_type: z_score
plot: false
progress: silent
device: cpu
""")
    model_cfg = load_config(config_path, ModelingConfig, {
        "epochs": 2, "location": location, "savedir": str(tmp_path / "model"),
    })
    as_loaded = replace(model_cfg)
    model_run = train_model(model_cfg, config_path=str(config_path))
    assert model_cfg == as_loaded   # the run resolves device, seed, name on a copy

    control_cfg = make_control_cfg(
        sysid_run_id=model_run.run_id, location=location,
        savedir=str(tmp_path / "control"), progress="silent", plot=False,
        user_hooks_path=str(FIXTURES / "control_user_defined.py"),
    )
    write_splits(control_cfg, tmp_path, plant, [make_reference_df(plant)])
    return model_run, train_control(control_cfg), plant_path


def test_a_trained_surrogate_feeds_a_policy_run_and_both_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # MLflow's default artifact root is ./mlruns
    model_run, policy_run, plant_path = _surrogate_then_policy(
        tmp_path, "mlp", FIXTURES / "modeling_user_defined.py")

    dyn_sys, sysid_cfg, nm, _ = load_trained_dynamics(model_run.run_id)
    assert sysid_cfg.epochs == 2
    assert [m.name for m in model_run.test_results.metrics] == nm["Y"]

    # Scored again from its id on its own test split, the run reports what
    # training reported: the run's own scaling, the topology it trained.
    rescored = evaluate_run(model_run.run_id, [str(plant_path)])
    assert [m.rmse for m in rescored.metrics] == pytest.approx(
        [m.rmse for m in model_run.test_results.metrics])

    _, _, cfg, reloaded_sysid_cfg, _, _ = load_trained_policy(policy_run.run_id)
    assert cfg.sysid_run_id == model_run.run_id
    assert reloaded_sysid_cfg.model == "mlp"
    assert 0.0 <= policy_run.results.comfort_violation_rate <= 1.0


def test_a_model_of_your_own_trains_reloads_controls_and_exports(tmp_path, monkeypatch):
    """Registered in the hooks file, a stateful physical-units model gets the
    whole backend: training, reload from the run's own hooks copy, a policy
    trained against it, and an export that agrees with it."""
    monkeypatch.chdir(tmp_path)
    hooks_path = tmp_path / "modeling_user_defined.py"
    hooks_path.write_text((FIXTURES / "modeling_user_defined.py").read_text() + USER_MODEL)
    model_run, policy_run, plant_path = _surrogate_then_policy(tmp_path, "ewma_zone", hooks_path)

    hooks_path.unlink()   # reloading reads the run's own copy
    dyn_sys, sysid_cfg, nm, norm_stats = load_trained_dynamics(model_run.run_id)
    assert type(contract_model(dyn_sys)).__name__ == "EwmaZone"
    rescored = evaluate_run(model_run.run_id, [str(plant_path)], physics_checks=False)
    assert [m.rmse for m in rescored.metrics] == pytest.approx(
        [m.rmse for m in model_run.test_results.metrics])
    assert 0.0 <= policy_run.results.comfort_violation_rate <= 1.0

    frame, timestep = get_data(str(plant_path))
    exported = export_model(
        dyn_sys=dyn_sys, cfg=sysid_cfg, nm_mapping=nm, norm_stats=norm_stats,
        frame=frame, timestep_seconds=timestep, run_id=model_run.run_id,
        out_dir=tmp_path / "pkg", exporter="legacy")
    assert exported.state_width == len(nm["Y"])
