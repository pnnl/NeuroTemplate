# Your first controller

A controller learns to set the U columns so the model's predictions do what
you ask: here, hold a zone inside a comfort band without jumpy commands. It
practises on the model, not on the building, so start from a trained model
you trust (`1-first-model.md`). It uses the latest one.

1. **Add the band to your data**, if it isn't there. The objective reads its
   limits from columns, so add them to all three files when you prepare the
   data:

   ```python
   import pandas as pd

   for split in ("train", "dev", "test"):
       df = pd.read_csv(f"data/{split}.csv")
       df["Band Low"], df["Band High"] = 21.0, 24.0
       df.to_csv(f"data/{split}.csv", index=False)
   ```

   Leave them out of `columns` in `data.yaml`: the model doesn't read them.
   A band that changes over the day, wider at night, is just a column that
   changes.

2. **Write the objective** in `control_configs/custom.yaml`, below the
   note at the top:

   ```yaml
   losses:
     too_cold: {column: Band Low, target: Zone 1 Temp, map: relu, norm: 1}
     too_hot: {column: Band High, target: Zone 1 Temp, map: nrelu, norm: 1}
     smooth: {column: Zone 1 Heat, map: first_difference, norm: 1}
   gains: {too_cold: 10.0, too_hot: 10.0, smooth: 0.1}
   ```

   Each line is a cost. `relu` costs the zone going below its column,
   `nrelu` going above it, and `first_difference` each change in a command.
   The gains weigh them against each other. If your U is a setpoint, put
   `smooth` on the setpoint.

3. **Train.**

   ```
   nt control train --config custom
   ```

4. **Read the results.** Before training it prints each cost's share of the
   total. At the end, how often the zone left the band on test, and the
   path to `control_reportcard.md`. `uv run mlflow ui` has the plots.

   Too many violations? Raise the band gains. Commands too jumpy? Raise
   `smooth`. Change one gain per run, so you know what did it.

Good to know:

- **Commands stay inside what the data recorded.** The controller can't
  ask for more heat than the record ever saw. To set your own limits, use
  `control_bounds` (`9-saving-money.md` has an example).
- **What it reads:** by default the zone, its last command, and the
  weather and band a few hours ahead. `policy_window` changes that
  (`9-saving-money.md`).

Next:

- ship it: `7-export.md`
- pay for energy, and shift load ahead of expensive hours:
  `9-saving-money.md`
