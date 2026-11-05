"""
`nt example thermal [dir]`: a working project on synthetic data, zero edits.

The plant is a two-node RC zone behind a cooling tracker, its heat metered a step late,
the same one the thermal end-to-end test fits, so no plant data ships. `examples/thermal`
holds the project: its hooks, configs and the network and thermostat `nt data ladder` and
`nt data thermostat` chose on this record, which you can re-run to see how.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ZONE, HEAT, POWER, SETPOINT = "Zone [°C]", "Heat [kW]", "Power [kW]", "Cooling SP [°C]"
OUTDOOR, SOLAR = "Outdoor [°C]", "Solar [W/m²]"
BAND_LOW, BAND_HIGH, TARIFF = "Band Low [°C]", "Band High [°C]", "Tariff [$/kWh]"
STEPS_PER_HOUR = 12

EXAMPLES = {"thermal": Path(__file__).with_name("examples") / "thermal"}

#: The splits: name, days, seed, first timestamp.
SPLITS = (("train", 14, 0, "2025-07-01"), ("dev", 4, 1, "2025-07-15"), ("test", 4, 2, "2025-07-19"))


def thermal_plant(days: float, seed: int, start: str) -> pd.DataFrame:
    """The network under a tracker, the setpoint stepped every two hours, in the record
    with a comfort band and a tariff no role claims."""
    from neurotemplate.modeling import network as nw
    from neurotemplate.modeling.ladder import Rung

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
        x = a @ x + b @ np.array([outdoor[k], delivered, solar[k]]) + rng.normal(0, [0.01, 0.003])
    power = np.abs(heat) / 3.5 + 0.2 + rng.normal(0, 0.01, steps)
    index = pd.date_range(start, periods=steps, freq="5min", name="timestamp")
    hour = index.hour.to_numpy()
    occupied = (hour >= 8) & (hour < 18)
    return pd.DataFrame({
        ZONE: zone, HEAT: heat, POWER: power, SETPOINT: setpoint, OUTDOOR: outdoor, SOLAR: solar,
        BAND_LOW: np.where(occupied, 22.0, 20.0), BAND_HIGH: np.where(occupied, 25.0, 28.0),
        TARIFF: np.where((hour >= 14) & (hour < 20), 0.48, 0.12),
    }, index=index)


def write_example(name: str, target: Path) -> list[Path]:
    """Copy the example's project into `target`, and generate its record."""
    source = EXAMPLES[name]
    files = sorted(p.relative_to(source) for p in source.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    existing = [str(f) for f in files if (target / f).exists()]
    if existing:
        raise FileExistsError(f"{target} already has {', '.join(existing)}; nothing written.")
    for f in files:
        (target / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / f, target / f)
    (target / "data").mkdir(exist_ok=True)
    for split, days, seed, start in SPLITS:
        path = target / "data" / f"{split}.csv"
        thermal_plant(days, seed, start).to_csv(path)
        files.append(path.relative_to(target))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(prog="nt example", description=__doc__.splitlines()[1])
    parser.add_argument("name", choices=list(EXAMPLES))
    parser.add_argument("dir", nargs="?", default=None,
                        help="Directory to write (default: nt-example-<name>).")
    args = parser.parse_args()
    target = Path(args.dir or f"nt-example-{args.name}")
    try:
        write_example(args.name, target)
    except FileExistsError as exc:
        raise SystemExit(f"nt example: {exc}") from exc
    print(f"Wrote {target}/: a project on synthetic data, ready to run:\n\n"
          f"  cd {target}\n"
          "  nt model train\n"
          "  nt control train --set sysid_run_id=<the run id it printed>\n"
          "  nt export policy --run_id <the control run id> --out delivery\n")
