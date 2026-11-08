"""The thermal-zone flow, end to end, on a plant the library could have made.

The example's plant, a two-node network behind a cooling tracker with its heat metered a
step late, goes through what the guide tells an agent to do: declare the columns, run
`nt data ladder` and `nt data thermostat`, paste their snippets into the
block, and train `rc_network`. The trained model must then track the plant
a day ahead from its setpoints alone.
"""

import json
import math
import sys

import mlflow
import pytest
import torch
import yaml

from neurotemplate.cli import main as nt
from neurotemplate.common.config import load_config
from neurotemplate.modeling import network as nw
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.train import train_model

from neurotemplate.example import HEAT, OUTDOOR, SETPOINT, SOLAR, ZONE, thermal_plant

HOOKS = f'''
import json, os
from neurotemplate.modeling.heads import HeadStack, RegressionHead

def output_heads(nm_mapping, norm_stats):
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
    from neurotemplate.metrics.physics import SynthSpec
    return SynthSpec(states=[{ZONE!r}], ambient={OUTDOOR!r},
                     off={{{SETPOINT!r}: 40.0, {HEAT!r}: 0.0}},
                     responds={{{ZONE!r}: {{{SETPOINT!r}: +1}}}})

def sub_plot(df_true, column_name):
    return None
'''


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
        thermal_plant(days, seed, start).to_csv(tmp_path / "data" / f"{split}.csv")
    (tmp_path / "nt_model.py").write_text(HOOKS, encoding="utf-8")
    past = {c: 0 for c in (ZONE, HEAT, OUTDOOR, SOLAR, SETPOINT)}
    config = tmp_path / "zone.yaml"
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({
        "train": "data/train.csv", "dev": "data/dev.csv", "test": "data/test.csv",
        "columns": {"Y": [ZONE, HEAT], "U": [SETPOINT], "D": [OUTDOOR, SOLAR]},
    }, allow_unicode=True), encoding="utf-8")
    config.write_text("\n".join([
        "model: rc_network",
        f"user_hooks_path: {tmp_path / 'nt_model.py'}",
        "past: " + json.dumps(past, ensure_ascii=False), "context: 1445m",
        "iters: 6", "horizon: 24h",
        "test_horizon: 1435m", "test_stride: 6h", "plot: false", "seed: 0", "progress: silent",
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
    # A thermostat commands nothing on its inactive side, and is not failed for it.
    silent = [r.name for r in result.test_results.physics if "Did not respond" in r.headline]
    assert silent == []
