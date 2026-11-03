"""
Write a reference file for one plant file: the comfort band (and, optionally,
a tariff) that the control objective names, on the plant file's timestamps.

    python make_references.py data/train.csv data/train_reference.csv \
        --band LO HI --width MIN MAX --segment-hours H --seed 0
    python make_references.py data/train.csv data/train_reference.csv \
        --band LO HI --width MIN MAX --segment-hours H --price data/price.csv

Example code, not library behaviour: what it means to randomise a band or a
tariff is a data decision, so it lives here where you can read and change it.
`nt control train` reads the file as data — it checks the timestamps match
the plant file's and nothing about how the values were made.

Several draws are several files. Run this once per draw with a new `--seed`,
then list the plant file once per draw in control.yaml:

    train_data_path:      [data/train.csv, data/train.csv, data/train.csv]
    train_reference_path: [data/train_ref_0.csv, data/train_ref_1.csv, data/train_ref_2.csv]

Train over the range of bands the policy will be handed in service, and make
the test file the reference you want it judged on. Sizing the band:

- `--width`: a band is the room a controller has to work in. Storing
  anything (pre-conditioning ahead of an expensive period) needs a band wide
  enough to store it in; a very narrow band leaves a policy nothing to do
  but track.
- `--segment-hours`: how long one band holds. A band that changes inside a
  training rollout teaches the policy to follow changes; one that holds for
  the whole rollout teaches it to work within a band. Choose by what the
  policy will face.

On the tariff: draw it from a real historical record, never from the clock —
a tariff that is a function of hour-of-day is one a policy can learn *as* the
clock. Train on `--price-align random` (the default): it slices the record at
a random offset, so the peak lands at a random hour of the plant's day and
the policy has to read the price. `day` keeps the time of day and draws only
the day, which lines the peaks up so a response is visible in a plot: use it
for the test reference you plot. To confirm a response is to the price and
not the clock, write a second test reference with the peak at another hour
(same plant file) and evaluate the trained policy on both.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from neurotemplate.common.data_utils import get_data
from neurotemplate.prep import expand


def sample_band(n_rows: int, lo: float, hi: float, width_min: float,
                width_max: float, segment: int, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray]:
    """A piecewise-constant band, a new one every `segment` rows: width
    ~ U[width_min, width_max], floor ~ U[lo, hi - width]."""
    if width_max > hi - lo:
        raise ValueError(f"width_max {width_max} is wider than the range "
                         f"[{lo}, {hi}] it is drawn inside.")
    n_segments = -(-n_rows // segment)
    widths = rng.uniform(width_min, width_max, n_segments)
    floors = lo + (hi - widths - lo) * rng.uniform(0.0, 1.0, n_segments)
    low = np.repeat(floors, segment)[:n_rows]
    return low, low + np.repeat(widths, segment)[:n_rows]


def slice_price(plant: pd.DataFrame, plant_step: float, price: pd.Series,
                align: str, rng: np.random.Generator) -> np.ndarray:
    """`len(plant)` rows of `price`, held onto the plant's timestep.

    `random` starts at any offset; `day` starts where the price's time of day
    matches the plant file's first row, on a random day.
    """
    series = expand(price, pd.Timedelta(seconds=plant_step), "hold").to_numpy()
    n = len(plant)
    if len(series) < n:
        raise ValueError(f"The price record covers {len(series)} plant steps; "
                         f"the plant file needs {n}.")
    if align == "random":
        start = int(rng.integers(0, len(series) - n + 1))
    else:
        per_day = round(86400 / plant_step)

        def step_of_day(t: pd.Timestamp) -> int:
            return round((t - t.normalize()).total_seconds() / plant_step)

        first = (step_of_day(plant.index[0]) - step_of_day(price.index[0])) % per_day
        days = (len(series) - n - first) // per_day + 1
        if days < 1:
            raise ValueError("The price record is too short to hold the plant "
                             "file at its own time of day; use --price-align random.")
        start = first + int(rng.integers(0, days)) * per_day
    return series[start:start + n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("plant", help="Plant CSV whose timestamps to follow.")
    parser.add_argument("out", help="Reference CSV to write.")
    parser.add_argument("--band", nargs=2, type=float, required=True,
                        metavar=("LO", "HI"), help="Range the band is drawn in.")
    parser.add_argument("--width", nargs=2, type=float, required=True,
                        metavar=("MIN", "MAX"),
                        help="Band width range (see the module docstring).")
    parser.add_argument("--segment-hours", type=float, required=True,
                        help="How long each band holds before the next draw.")
    parser.add_argument("--low-column", default="Comfort Low [°C]")
    parser.add_argument("--high-column", default="Comfort High [°C]")
    parser.add_argument("--price", help="Historical price CSV (timestamp + price).")
    parser.add_argument("--price-column", help="Its price column, if it has several.")
    parser.add_argument("--price-align", choices=("random", "day"), default="random")
    parser.add_argument("--tariff-column", default="Tariff [$/kWh]")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    plant, plant_step = get_data(args.plant)
    segment = max(1, round(args.segment_hours * 3600 / plant_step))
    low, high = sample_band(len(plant), *args.band, *args.width, segment, rng)
    reference = pd.DataFrame({args.low_column: low, args.high_column: high},
                             index=plant.index)

    if args.price:
        prices, _ = get_data(args.price)
        columns = [c for c in prices.columns if c != "sec_elapsed"]
        column = args.price_column or (columns[0] if len(columns) == 1 else None)
        if column is None:
            raise ValueError(f"{args.price} has several columns {columns}; "
                             "pass --price-column.")
        reference[args.tariff_column] = slice_price(
            plant, plant_step, prices[column], args.price_align, rng)

    reference.rename_axis("timestamp").to_csv(args.out)
    print(f"wrote {args.out}: {len(reference)} rows, columns {list(reference.columns)}")


if __name__ == "__main__":
    main()
