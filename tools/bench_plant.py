"""Closed-loop score of exported policies on the example's true plant. Temporary bench.
    bench_plant.py <export_dir> [<export_dir> ...]       (each from `nt export policy --format onnx`)

The plant is `example.thermal_plant`'s network and cooling tracker, without noise, so a policy
trained on the fitted surrogate is scored against the truth. Each scenario starts at an hour
of the day from a zone temperature (the mass at the same), then runs a day under the policy.
24 °C is where the records sit; 27 and 30 °C are off them (the plant only cools, so a cold start
scores every controller alike). `rule` holds the setpoint just
under the band's top. Prints degree-hours outside the band and the energy bill, per day."""
import json, sys
from pathlib import Path

import numpy as np

from neurotemplate.example import HEAT, SOLAR, STEPS_PER_HOUR
from neurotemplate.modeling import network as nw
from neurotemplate.modeling.ladder import Rung

START_HOURS = (0, 6, 12, 18)
START_ZONES = (24.0, 27.0, 30.0)
DAY = 24 * STEPS_PER_HOUR
DT_H = 1 / STEPS_PER_HOUR


def plant():
    network = Rung(mass=True).network((HEAT,), SOLAR, (), {HEAT: 1.0})
    values = {"C_Ti": 1.5, "C_Tm": 20.0, "R_Ti_Tm": 0.4, "R_Ti_ambient": 3.0,
              f"gain_{HEAT}_Ti": 1.0, f"gain_{SOLAR}_Ti": 0.004,
              "sigma_Ti": 0.05, "sigma_Tm": 0.02, "sigma_y": 0.02}
    a, b, _, _ = nw.discrete_model(network, nw.as_tensors(values), DT_H)
    return a.numpy(), b.numpy()


def signals(hours):
    """Every exogenous column at `hours` (any shape), as `example.thermal_plant` makes them."""
    hour_of_day = np.floor(hours % 24)
    occupied = (hour_of_day >= 8) & (hour_of_day < 18)
    return {"Outdoor [°C]": 27 + 6 * np.sin(2 * np.pi * (hours - 9) / 24),
            "Solar [W/m²]": 800 * np.clip(np.sin(2 * np.pi * (hours - 6) / 24), 0, None),
            "Band Low [°C]": np.where(occupied, 22.0, 20.0),
            "Band High [°C]": np.where(occupied, 25.0, 28.0),
            "Tariff [$/kWh]": np.where((hour_of_day >= 14) & (hour_of_day < 20), 0.48, 0.12)}


def window(spec, now_hours, history):
    """One input's `[batch, steps, n]` window at each scenario's current time."""
    rows = np.arange(spec["steps"]) - spec["current_row"]
    if spec["columns"] == ["Zone [°C]"]:
        return history[:, -spec["steps"]:, None]
    at = signals(now_hours[:, None] + rows[None, :] * DT_H)
    return np.stack([at[c] for c in spec["columns"]], axis=-1)


def run(act):
    """Degree-hours outside the band and the bill per day, per start zone, under `act`."""
    a, b = plant()
    starts = [(h, z) for z in START_ZONES for h in START_HOURS]
    hours = np.array([h for h, _ in starts], dtype=float)
    x = np.array([[z, z] for _, z in starts])
    history = np.repeat(x[:, :1], 12, axis=1)
    outside, bill = np.zeros(len(starts)), np.zeros(len(starts))
    for _ in range(DAY):
        setpoint = act(hours, history)
        now = signals(hours)
        delivered = -np.clip(15.0 * (x[:, 0] - setpoint - 0.3), 0.0, 8.0)
        u = np.stack([now["Outdoor [°C]"], delivered, now["Solar [W/m²]"]], axis=-1)
        x = x @ a.T + u @ b.T
        hours = hours + DT_H
        later = signals(hours)
        outside += DT_H * (np.clip(later["Band Low [°C]"] - x[:, 0], 0, None)
                           + np.clip(x[:, 0] - later["Band High [°C]"], 0, None))
        bill += DT_H * (np.abs(delivered) / 3.5 + 0.2) * now["Tariff [$/kWh]"]
        history = np.concatenate([history[:, 1:], x[:, :1]], axis=1)
    zones = np.array([z for _, z in starts])
    return {z: (outside[zones == z].mean(), bill[zones == z].mean()) for z in START_ZONES}


def policy_act(export_dir):
    sys.path.insert(0, str(export_dir))
    from runtime import Policy
    sys.path.pop(0)
    sys.modules.pop("runtime")
    policy = Policy(Path(export_dir) / "policy")
    inputs = json.loads((Path(export_dir) / "policy" / "policy.json").read_text())["inputs"]
    return lambda hours, history: policy.act_batch(
        {name: window(spec, hours, history) for name, spec in inputs.items()})[:, 0]


def rule(hours, history):
    return np.clip(signals(hours)["Band High [°C]"] - 0.5, 21.0, 28.0)


def report(name, scores):
    cells = "  ".join(f"{z:.0f}°C {k:6.2f} Kh ${c:5.2f}" for z, (k, c) in scores.items())
    print(f"{name:28s} {cells}", flush=True)


if __name__ == "__main__":
    report("rule", run(rule))
    for export_dir in sys.argv[1:]:
        report(Path(export_dir).name, run(policy_act(export_dir)))
