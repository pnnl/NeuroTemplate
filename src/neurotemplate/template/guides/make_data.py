"""
A practice record for the guides: three zones of one building, in 5-minute
steps, written to data/train.csv, dev.csv and test.csv (28, 7 and 7 days).

    uv run python guides/make_data.py

Nothing in it is measured. This file simulates every column, so you can
open it and hold what the library finds against the truth: `ZONES` below.

Each zone is the room air (Ti), coupled to a heavy floor slab (Tm), losing
heat to outdoors through the air, with sun through the windows:

           sun   heat
             ╲   ╱
   slab ──R_slab── air ──R_out── outdoors
    C_slab        C_air

A thermostat holds each zone between a heating and a cooling setpoint, each
stepped every two hours to somewhere new, so the record shows the zone
both driven and drifting.

Columns, for each zone n of 1, 2, 3 (most guides use zone 1):

  Zone n Temp          the zone, °C, measured at the row
  Zone n Heat          the heat delivered over the step ahead, kW:
                       positive heating, negative cooling
  Zone n Heating SP    the setpoints, °C, held over the step ahead
  Zone n Cooling SP
  Zone n Power         electric power over the step just ended, kW: a meter

and, for the whole building, Outdoor Temp (°C), Solar (W/m²) and Price
($/kWh).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: The truth. Capacities in kWh/K, resistances in K/kW, `sun` in kW per W/m²,
#: the thermostat's `gain` in kW per K past its setpoint and its capacities in kW.
ZONES = {
    1: {"C_air": 1.5, "C_slab": 20.0, "R_slab": 0.4, "R_out": 3.0, "sun": 0.004,
        "gain": 15.0, "heating": 6.0, "cooling": 8.0},
    2: {"C_air": 2.5, "C_slab": 35.0, "R_slab": 0.25, "R_out": 2.0, "sun": 0.006,
        "gain": 20.0, "heating": 9.0, "cooling": 12.0},
    3: {"C_air": 1.0, "C_slab": 12.0, "R_slab": 0.6, "R_out": 4.0, "sun": 0.002,
        "gain": 10.0, "heating": 4.0, "cooling": 5.0},
}
OFFSET = 0.3            # K past its setpoint where each thermostat starts
COP = {"heating": 3.0, "cooling": 3.5}
STANDBY = 0.2           # kW each zone's unit draws when idle
HEATING_SP = [16.0, 17.0, 18.0, 19.0, 20.0]
COOLING_SP = [21.0, 22.0, 23.0, 24.0, 25.0]

STEP_H = 5 / 60
SPLITS = {"train": 28, "dev": 7, "test": 7}   # days, in time order
SETTLE_DAYS = 2         # simulated first and dropped, so the slab starts settled


def step_matrices(zone: dict) -> tuple[np.ndarray, np.ndarray]:
    """`x ← A x + B u` over one step, for x = (air, slab) and u = (outdoor,
    heat, sun) held over the step: Euler's method on

        C_air  · d air/dt  = (slab − air)/R_slab + (outdoor − air)/R_out
                             + heat + sun · Solar
        C_slab · d slab/dt = (air − slab)/R_slab

    in 4096 substeps, by squaring the one-substep matrix 12 times."""
    c_air, c_slab, r_slab, r_out = (zone[k] for k in ("C_air", "C_slab", "R_slab", "R_out"))
    m = np.zeros((5, 5))
    m[0, :] = [-(1 / r_slab + 1 / r_out), 1 / r_slab, 1 / r_out, 1.0, zone["sun"]]
    m[0, :] /= c_air
    m[1, :2] = [1 / r_slab, -1 / r_slab]
    m[1, :] /= c_slab
    e = np.eye(5) + m * STEP_H / 4096
    for _ in range(12):
        e = e @ e
    return e[:2, :2], e[:2, 2:]


def weather(steps: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Spring: a daily swing of ±6 K about a mean near 19 °C that wanders from day to
    day, and sun on a clear-sky curve dimmed by each day's cloud."""
    hours = np.arange(steps) * STEP_H
    days = int(hours[-1] // 24) + 2
    daily_mean = np.interp(hours, np.arange(days) * 24 + 12, 19 + 3 * rng.standard_normal(days))
    outdoor = daily_mean + 6 * np.sin(2 * np.pi * (hours - 9) / 24)
    cloud = rng.uniform(0.3, 1.0, days)[(hours // 24).astype(int)]
    solar = 800 * np.clip(np.sin(2 * np.pi * (hours - 6) / 24), 0, None) * cloud
    return outdoor, solar


def practice_record(days: float, seed: int = 0, start: str = "2025-04-01") -> pd.DataFrame:
    """`days` of the building, every zone under its thermostat."""
    rng = np.random.default_rng(seed)
    settle = SETTLE_DAYS * int(24 / STEP_H)
    steps = settle + int(days * 24 / STEP_H)
    outdoor, solar = weather(steps, rng)
    columns = {}
    for n, zone in ZONES.items():
        a, b = step_matrices(zone)
        every_2h = steps // 24 + 1
        heating_sp = rng.choice(HEATING_SP, every_2h).repeat(24)[:steps]
        cooling_sp = rng.choice(COOLING_SP, every_2h).repeat(24)[:steps]
        x = np.array([20.0, 20.0])
        temp, heat = np.empty(steps), np.empty(steps)
        for k in range(steps):
            temp[k] = x[0] + rng.normal(0, 0.02)                 # the sensor
            heat[k] = (np.clip(zone["gain"] * (heating_sp[k] - temp[k] - OFFSET),
                               0, zone["heating"])
                       - np.clip(zone["gain"] * (temp[k] - cooling_sp[k] - OFFSET),
                                 0, zone["cooling"]))
            x = a @ x + b @ [outdoor[k], heat[k], solar[k]] + rng.normal(0, [0.01, 0.003])
        # The meter at row k reports the step that ended there, driven by heat[k − 1].
        drawn = np.clip(heat, 0, None) / COP["heating"] + np.clip(-heat, 0, None) / COP["cooling"]
        power = STANDBY + np.concatenate([[0.0], drawn[:-1]]) + rng.normal(0, 0.01, steps)
        columns.update({f"Zone {n} Temp": temp, f"Zone {n} Heat": heat,
                        f"Zone {n} Heating SP": heating_sp, f"Zone {n} Cooling SP": cooling_sp,
                        f"Zone {n} Power": power})
    index = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=SETTLE_DAYS),
                          periods=steps, freq="5min", name="timestamp")
    hour = index.hour.to_numpy()
    columns.update({"Outdoor Temp": outdoor, "Solar": solar,
                    "Price": np.where((hour >= 14) & (hour < 20), 0.48, 0.12)})
    return pd.DataFrame(columns, index=index).iloc[settle:].round(4)


def write_record(out: Path, splits: dict = SPLITS, seed: int = 0) -> list[Path]:
    """One record, cut in time order into a file per split. Writes nothing if
    any of those files exists."""
    paths = {split: out / f"{split}.csv" for split in splits}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise FileExistsError(f"{', '.join(existing)} already exist; nothing written. "
                              "Move them aside, or pass --out another directory.")
    record = practice_record(sum(splits.values()), seed)
    out.mkdir(parents=True, exist_ok=True)
    first = record.index[0]
    for split, days in splits.items():
        last = first + pd.Timedelta(days=days)
        record[(record.index >= first) & (record.index < last)].to_csv(paths[split])
        first = last
    return list(paths.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", default="data", help="Directory to write (default: data).")
    parser.add_argument("--seed", type=int, default=0, help="Another seed, another record.")
    args = parser.parse_args()
    try:
        paths = write_record(Path(args.out), seed=args.seed)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
