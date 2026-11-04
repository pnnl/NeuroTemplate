"""The thermal-zone flow, end to end, on a plant the library could have made.

A two-node network behind a cooling tracker, its heat metered a step late,
goes through what the guide tells an agent to do: declare the columns, run
`nt data ladder` and `nt data thermostat`, paste their snippets into the
block, and train `rc_network`. The trained model must then track the plant
a day ahead from its setpoints alone.
"""

import json
import math
import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch

from neurotemplate.cli import main as nt
from neurotemplate.common.config import load_config
from neurotemplate.modeling import network as nw
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.ladder import Rung
from neurotemplate.modeling.train import train_model

ZONE, HEAT, SETPOINT = "Zone [°C]", "Heat [kW]", "Cooling SP [°C]"
OUTDOOR, SOLAR = "Outdoor [°C]", "Solar [W/m²]"
STEPS_PER_HOUR = 12

HOOKS = f'''
import json, os
from neurotemplate.modeling.heads import HeadStack, RegressionHead

def data_mapping(df):
    return {{"Y": [{ZONE!r}, {HEAT!r}], "U": [{SETPOINT!r}], "D": [{OUTDOOR!r}, {SOLAR!r}],
            "t": ["sec_elapsed"]}}

def data_head_config(nm_mapping, norm_stats):
    return HeadStack([RegressionHead() for _ in nm_mapping["Y"]], nm_mapping["Y"])

def column_norm_config(nm_mapping):
    return {{}}

def block_mapping(nm_mapping, cfg):
    zone = {{"Y": [{ZONE!r}], "U": [], "D": [{OUTDOOR!r}, {SOLAR!r}], "driven_by": [{HEAT!r}],
            "ambient": {OUTDOOR!r}, "solar": {SOLAR!r}, "scale": {{{HEAT!r}: 1.0}}}}
    hvac = {{"Y": [{HEAT!r}], "U": [{SETPOINT!r}], "D": [], "reads": [{ZONE!r}]}}
    if os.path.exists("ladder/zone.json"):
        zone["params"] = {{"rc_network": json.load(open("ladder/zone.json"))}}
    if os.path.exists("ladder/hvac.thermostat.json"):
        hvac["params"] = {{"thermostat": json.load(open("ladder/hvac.thermostat.json"))}}
        hvac["model"], zone["model"] = "thermostat", "rc_network"
    return {{"zone": zone, "hvac": hvac}}

def synth_cfg(nm_mapping, norm_stats):
    return None

def sub_plot(df_true, column_name):
    return None
'''


def _plant(days: float, seed: int, start: str) -> pd.DataFrame:
    """The network under a tracker, the setpoint stepped every two hours."""
    network = Rung(mass=True).network((HEAT,), SOLAR, (), {HEAT: 1.0})
    values = {"C_Ti": 1.5, "C_Tm": 20.0, "R_Ti_Tm": 0.4, "R_Ti_ambient": 3.0,
              f"gain_{HEAT}_Ti": 1.0, f"gain_{SOLAR}_Ti": 0.004,
              "sigma_Ti": 0.05, "sigma_Tm": 0.02, "sigma_y": 0.02}
    a, b, _, _ = nw.discrete_model(network, nw.as_tensors(values), 1 / STEPS_PER_HOUR)
    a, b = a.numpy(), b.numpy()
    rng = np.random.default_rng(seed)
    steps = int(days * 24 * STEPS_PER_HOUR)
    hours = np.arange(steps) / STEPS_PER_HOUR
    outdoor = 27 + 6 * np.sin(2 * np.pi * (hours - 9) / 24)
    solar = 800 * np.clip(np.sin(2 * np.pi * (hours - 6) / 24), 0, None)
    setpoint = rng.choice([21.0, 22.5, 24.0, 26.0, 28.0], steps // 24 + 1).repeat(24)[:steps]
    x, zone, heat = np.array([24.0, 24.0]), np.empty(steps), np.zeros(steps)
    for k in range(steps):
        zone[k] = x[0] + rng.normal(0, 0.02)
        delivered = -np.clip(15.0 * (x[0] - setpoint[k] - 0.3), 0.0, 8.0)
        if k + 1 < steps:
            heat[k + 1] = delivered          # the meter reports the step that ends here
        x = a @ x + b @ np.array([outdoor[k], delivered, solar[k]]) \
            + rng.normal(0, [0.01, 0.003])
    return pd.DataFrame({ZONE: zone, HEAT: heat, SETPOINT: setpoint, OUTDOOR: outdoor,
                         SOLAR: solar},
                        index=pd.date_range(start, periods=steps, freq="5min",
                                            name="timestamp"))


@pytest.fixture(autouse=True)
def _restore_tracking_uri():
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)
    for name in list(sys.modules):
        if name.startswith("nt_model"):
            del sys.modules[name]


def test_declare_ladder_thermostat_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    for split, days, seed, start in (("train", 8, 0, "2025-07-01"), ("dev", 3, 1, "2025-07-09"),
                                     ("test", 3, 2, "2025-07-12")):
        _plant(days, seed, start).to_csv(tmp_path / "data" / f"{split}.csv")
    (tmp_path / "nt_model.py").write_text(HOOKS, encoding="utf-8")
    window = 24 * STEPS_PER_HOUR
    history = {c: 0 for c in (ZONE, HEAT, OUTDOOR, SOLAR, SETPOINT)}
    config = tmp_path / "zone.yaml"
    config.write_text("\n".join([
        "train_data_path: data/train.csv", "dev_data_path: data/dev.csv",
        "test_data_path: data/test.csv", "model: rc_network",
        f"user_hooks_path: {tmp_path / 'nt_model.py'}",
        "history: " + json.dumps(history, ensure_ascii=False), f"context_steps: {window + 1}",
        "epochs: 3", "warmup: 0", "patience: 3", f"train_pred_steps: {window}",
        f"dev_pred_steps: {window}", f"test_window: {window + 24 * STEPS_PER_HOUR}",
        f"test_stride: {6 * STEPS_PER_HOUR}", "plot: false", "seed: 0", "progress: silent",
        f"savedir: {tmp_path / 'run'}", f"location: sqlite:///{tmp_path / 'mlflow.db'}",
    ]), encoding="utf-8")

    nt(["data", "ladder", "--config", str(config)])
    nt(["data", "thermostat", "--config", str(config), "--cooling", SETPOINT])
    network = json.loads((tmp_path / "ladder" / "zone.json").read_text(encoding="utf-8"))
    assert {"Ti", "Tm"} <= set(nw.parse_declaration(network)[0].nodes)

    cfg = load_config(str(config), cls=ModelingConfig)
    torch.manual_seed(0)
    result = train_model(cfg, config_path=str(config))
    zone_rmse = next(m.rmse for m in result.test_results.metrics if m.name == ZONE)
    assert math.isfinite(zone_rmse) and zone_rmse < 0.1, zone_rmse
