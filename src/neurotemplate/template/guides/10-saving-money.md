# A controller that saves money

Advanced. Your first controller holds a band (`7-first-controller.md`). This
one also pays for energy, and when the price changes over the day it learns
to pre-cool or pre-heat ahead of the expensive hours. It needs a model that
predicts power and passes its physics checks (`9-physics-checks.md`).

You need, beside your first controller's columns:

- a power meter, in kW, as a Y of the model,
- a price column in the data, in no role, like the band.

The practice record has both: `Zone 1 Power` and `Price`.

1. **Give the meter its own block**, one that sees the command driving it.
   The zone and the meter both need the heat, so neither lists it in `U`:
   a U no block claims goes to every block.

   ```python
   TEMP, HEAT, POWER = "Zone 1 Temp", "Zone 1 Heat", "Zone 1 Power"
   OUTDOOR, SUN = "Outdoor Temp", "Solar"


   def block_mapping(nm_mapping, cfg):
       return {
           "zone": {
               "Y": [TEMP], "D": [OUTDOOR, SUN],
               "model": "rc_network", "ambient": OUTDOOR, "solar": SUN,
               "params": {"rc_network": "ladder/zone.json"},
           },
           "meter": {
               "Y": [POWER], "D": [OUTDOOR],
               "model": "mlp", "predicts": "value",
           },
       }
   ```

   Add `Zone 1 Power` to `Y` in `data.yaml`, and retrain. Behind a
   thermostat, the meter takes `"driven_by": [HEAT]` instead: the heat the
   thermostat block predicts.

2. **Name the meters** in `nt_control.py`. The cost sums these:

   ```python
   def power_channels(nm_mapping):
       return ["Zone 1 Power"]
   ```

3. **Let it see ahead.** It can't move load before the expensive hours
   unless it sees them coming. In `control_configs/custom.yaml`:

   ```yaml
   policy_window:
     Zone 1 Temp: {past: 15m}
     Outdoor Temp: {past: 0, future: 6h}
     Band Low: {past: 0, future: 23h55m}
     Band High: {past: 0, future: 23h55m}
     Price: {past: 0, future: 23h55m}
   horizon: 24h
   ```

   It reads only what this lists. `horizon` is how long each practice run
   on the model lasts: long enough to hold the action and its payoff, a
   pre-cool at noon and the peak it saves at five.

4. **Add the cost** to the objective:

   ```yaml
   losses:
     too_cold: {column: Band Low, target: Zone 1 Temp, map: relu, norm: 1}
     too_hot: {column: Band High, target: Zone 1 Temp, map: nrelu, norm: 1}
     cost: {column: power, norm: 1, weight: Price}
     smooth: {column: Zone 1 Heat, map: first_difference, norm: 1}
   gains: {too_cold: 100.0, too_hot: 100.0, cost: 0.001, smooth: 0.1}
   ```

   `power` is the sum of the meters you named, and `weight` prices each step.

5. **Size the gains, then train.**

   ```
   nt control train --config custom
   ```

   Before training it prints each cost's share of the total. A term near 0%
   does nothing yet: raise its gain tenfold and look again. Your power and
   price units set the scale, so no number here fits every building. After
   training it prints the shares again, and a band term near 0% there means
   the band held, which is what you want.

6. **Compare it with the best possible plan.**

   ```
   nt control gap --run_id <control run id>
   ```

   `gap` plans the whole test period through the same model, knowing
   everything in advance, and sets your controller beside it. A big gap is
   the controller's to close: what it sees, or more training. Something the
   plan doesn't do either, the model or the objective doesn't reward.

Good to know:

- **Why `23h55m`, not `24h`.** A `future` counts the rows after now. A day
  of 5-minute forecast that includes the current row has 287 rows after it:
  23h55m. On hourly data the same day is `23h`.
- **Did the price buy anything?** Train the same config with the `cost`
  gain at 0. The difference between the two runs is what pricing did.
- **Train on several prices.** On one daily price, a controller learns the
  clock, not the price: "cool at 2 pm". Shift the price a few hours each
  way, save each shift as its own file, and list them:

  ```python
  import numpy as np
  import pandas as pd

  df = pd.read_csv("data/train.csv")
  for name, rows in (("later", 36), ("earlier", -36)):  # 3 h of 5-min rows
      draw = df[["timestamp", "Price"]].copy()
      draw["Price"] = np.roll(draw["Price"], rows)
      draw.to_csv(f"data/price_{name}.csv", index=False)
  ```

  ```yaml
  draws: [data/price_later.csv, data/price_earlier.csv]
  ```

- **Bound every unpriced command.** If only the cooling is metered, the
  controller can run the heating against it, and the cost sees half the
  bill. Keep the setpoints from crossing in `control_bounds`:

  ```yaml
  control_bounds:
    Zone 1 Cooling SP: [Band Low, [Band High, -0.5]]
    Zone 1 Heating SP: [15.0, Band Low]
  ```

  Each side is a number, a column, or a column plus an offset: the cooling
  setpoint stays between the band's floor and half a degree under its top.
- **Judge the winner on the real building.** The model's own comfort numbers
  have been wrong by several times in both directions. Use them to compare
  controllers, not to decide one is safe.
