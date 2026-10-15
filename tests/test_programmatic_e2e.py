"""The Python entry points chain the way the CLI commands do.

A surrogate `train_model` trains is the `sysid_run_id` a `train_control` run
loads, both runs reload by the ids they return, and `evaluate_run` scores
the surrogate as its training run did — one real MLflow store, one tiny run
each. The CLIs are thin wrappers over these calls, so this is also the
wiring `nt model train`, `nt model load` and `nt control train` depend on.
"""

import sys
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest

from neurotemplate.common.config import load_config
from neurotemplate.control.load import load_trained_policy
from neurotemplate.control.train import train_control
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.load import evaluate_run, load_trained_dynamics
from neurotemplate.modeling.train import train_model
from control_fixtures import (
    make_control_cfg, make_raw_df, make_reference_df, write_splits,
)

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


def test_a_trained_surrogate_feeds_a_policy_run_and_both_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # MLflow's default artifact root is ./mlruns
    location = f"sqlite:///{tmp_path / 'mlflow.db'}"
    plant = make_raw_df(n_rows=160)
    plant_path = tmp_path / "plant.csv"
    plant.drop(columns="sec_elapsed").rename_axis("timestamp").to_csv(plant_path)

    config_path = tmp_path / "modeling.yaml"
    config_path.write_text(f"""
train_data_path: {plant_path}
dev_data_path: {plant_path}
test_data_path: {plant_path}
user_hooks_path: {FIXTURES / "modeling_user_defined.py"}
model: mlp
in_steps: 4
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
verbose: false
device: cpu
""")
    model_cfg = load_config(config_path, ModelingConfig, {
        "epochs": 2, "location": location, "savedir": str(tmp_path / "model"),
    })
    as_loaded = replace(model_cfg)
    model_run = train_model(model_cfg, config_path=str(config_path))

    assert model_cfg == as_loaded   # the run resolves device, seed, name on a copy
    dyn_sys, sysid_cfg, nm, _ = load_trained_dynamics(model_run.run_id)
    assert sysid_cfg.epochs == 2
    assert [m.name for m in model_run.test_results.metrics] == nm["Y"]

    # Scored again from its id on its own test split, the run reports what
    # training reported: the run's own scaling, the topology it trained.
    rescored = evaluate_run(model_run.run_id, [str(plant_path)])
    assert [m.rmse for m in rescored.metrics] == pytest.approx(
        [m.rmse for m in model_run.test_results.metrics])

    control_cfg = make_control_cfg(
        sysid_run_id=model_run.run_id, location=location,
        savedir=str(tmp_path / "control"), verbose=False, plot=False,
        user_hooks_path=str(FIXTURES / "control_user_defined.py"),
    )
    write_splits(control_cfg, tmp_path, plant, [make_reference_df(plant)])
    policy_run = train_control(control_cfg)

    _, _, cfg, reloaded_sysid_cfg, _, _ = load_trained_policy(policy_run.run_id)
    assert cfg.sysid_run_id == model_run.run_id
    assert reloaded_sysid_cfg.model == "mlp"
    assert 0.0 <= policy_run.results.comfort_violation_rate <= 1.0
