"""The thermal-zone flow, end to end, on the guides' practice record.

Zone 1 of `guides/make_data.py`, air and a floor slab behind a two-sided thermostat, goes
through what `4-thermostat.md` tells a person to do: line the heat up as a meter would,
declare the columns, run `nt data ladder` and `nt data thermostat`, and train. The fits
must land near the truth the script simulates, as `3-rc-ladder.md` and `4-thermostat.md`
say they do, and the trained model must track the zone a day ahead from its setpoints
alone.
"""

import json
import math
import sys

import mlflow
import pandas as pd
import pytest
import torch
import yaml

from neurotemplate.cli import main as nt
from neurotemplate.common.config import load_config
from neurotemplate.modeling import network as nw
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.train import train_model

from practice import make_data

ZONE, HEAT = "Zone 1 Temp", "Zone 1 Delivered"
COOL_SP, HEAT_SP = "Zone 1 Cooling SP", "Zone 1 Heating SP"
OUTDOOR, SOLAR = "Outdoor Temp", "Solar"

HOOKS = f'''
import os
from neurotemplate.modeling.heads import HeadStack, RegressionHead

def output_heads(nm_mapping, norm_stats):
    return HeadStack([RegressionHead() for _ in nm_mapping["Y"]], nm_mapping["Y"])

def column_norm_config(nm_mapping):
    return {{}}

def block_mapping(nm_mapping, cfg):
    zone = {{"Y": [{ZONE!r}], "U": [], "D": [{OUTDOOR!r}, {SOLAR!r}], "driven_by": [{HEAT!r}],
            "ambient": {OUTDOOR!r}, "solar": {SOLAR!r},
            "params": {{"rc_network": "ladder/zone.json"}}}}
    hvac = {{"Y": [{HEAT!r}], "U": [{COOL_SP!r}, {HEAT_SP!r}], "D": [], "reads": [{ZONE!r}],
            "params": {{"thermostat": "ladder/hvac.thermostat.json"}}}}
    if os.path.exists("ladder/hvac.thermostat.json"):
        hvac["model"], zone["model"] = "thermostat", "rc_network"
    return {{"zone": zone, "hvac": hvac}}

def synth_cfg(nm_mapping, norm_stats):
    from neurotemplate.metrics.physics import SynthSpec
    return SynthSpec(states=[{ZONE!r}], ambient={OUTDOOR!r},
                     off={{{COOL_SP!r}: 40.0, {HEAT_SP!r}: 0.0, {HEAT!r}: 0.0}},
                     responds={{{ZONE!r}: {{{COOL_SP!r}: +1, {HEAT_SP!r}: +1}}}})

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


def test_declare_ladder_thermostat_train(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for path in make_data.write_record(tmp_path / "data", {"train": 8, "dev": 3, "test": 3}):
        df = pd.read_csv(path)                       # 4-thermostat.md, step 1
        df[HEAT] = df["Zone 1 Heat"].shift(1).bfill()
        df.to_csv(path, index=False)
    (tmp_path / "nt_model.py").write_text(HOOKS, encoding="utf-8")
    past = {c: 0 for c in (ZONE, HEAT, OUTDOOR, SOLAR, COOL_SP, HEAT_SP)}
    config = tmp_path / "zone.yaml"
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({
        "train": "data/train.csv", "dev": "data/dev.csv", "test": "data/test.csv",
        "columns": {"Y": [ZONE, HEAT], "U": [COOL_SP, HEAT_SP], "D": [OUTDOOR, SOLAR]},
    }, allow_unicode=True), encoding="utf-8")
    config.write_text("\n".join([
        "model: rc_network",
        f"user_hooks_path: {tmp_path / 'nt_model.py'}",
        "past: " + json.dumps(past, ensure_ascii=False), "spinup: 1445m",
        "epochs: 3", "warmup: 0", "patience: 3", "horizon: 24h",
        "test_horizon: 1435m", "test_stride: 6h", "plot: false", "seed: 0", "progress: silent",
        f"savedir: {tmp_path / 'run'}", f"location: sqlite:///{tmp_path / 'mlflow.db'}",
    ]), encoding="utf-8")

    nt(["data", "ladder", "--config", str(config)])
    brief = capsys.readouterr().out
    assert "→" in brief and "params in ladder/*.json" in brief and "p vs parent" not in brief
    nt(["data", "ladder", "--config", str(config), "-v"])
    assert "p vs parent" in capsys.readouterr().out
    nt(["data", "thermostat", "--config", str(config), "--cooling", COOL_SP,
        "--heating", HEAT_SP])
    network = json.loads((tmp_path / "ladder" / "zone.json").read_text(encoding="utf-8"))
    assert set(nw.parse_declaration(network)[0].nodes) == {"Ti", "Tm"}
    # Near the truth, as 3-rc-ladder.md's table says: not on it, from a noisy closed loop.
    truth = make_data.ZONES[1]
    fitted = {"C_air": network["nodes"]["Ti"], "C_slab": network["nodes"]["Tm"],
              "R_slab": network["edges"]["Ti-Tm"], "R_out": network["edges"]["Ti-ambient"],
              "sun": network["inputs"][SOLAR]["Ti"]}
    for name, box in fitted.items():
        assert abs(sum(box) / 2 / truth[name] - 1) < 0.3, (name, box, truth[name])
    hvac = json.loads((tmp_path / "ladder" / "hvac.thermostat.json").read_text(encoding="utf-8"))
    for side in ("heating", "cooling"):
        for name, true in (("gain", truth["gain"]), ("capacity", truth[side]),
                           ("offset", make_data.OFFSET)):
            box = hvac[side][name]
            assert abs(sum(box) / 2 / true - 1) < 0.3, (side, name, box, true)

    cfg = load_config(str(config), cls=ModelingConfig)
    torch.manual_seed(0)
    result = train_model(cfg, config_path=str(config))
    zone_rmse = next(m.rmse for m in result.test_results.metrics if m.name == ZONE)
    assert math.isfinite(zone_rmse) and zone_rmse < 0.1, zone_rmse
    # A thermostat commands nothing on its inactive side, and is not failed for it.
    silent = [r.name for r in result.test_results.physics if "Did not respond" in r.headline]
    assert silent == []
