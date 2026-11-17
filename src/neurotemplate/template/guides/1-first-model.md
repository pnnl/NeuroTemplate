# Your first model

Train a model that forecasts your building from its own data, and see how
good it is. Run every command from this folder. If `nt` is not found, put
`uv run` in front of it.

1. **Add your data.** Copy your files into `data/` as `train.csv`, `dev.csv`
   and `test.csv`. The model learns from train, checks itself on dev as it
   goes, and gets one final score on test, which it never sees while
   learning. Split one record in time order: four weeks, one and one is a
   fine start.

   Each file has one row per timestep, evenly spaced, and a time column
   named `timestamp`, `time`, `date` or `datetime`. If your files have
   different names or paths, just change the paths in `data.yaml`.

   **To practise first**, make a record instead:

   ```
   uv run python guides/make_data.py
   ```

   It simulates three zones of a building for six weeks, and writes them
   split four weeks, one and one. Every guide's examples use its columns,
   zone 1 unless the guide is about several zones, so each step runs as
   written. With your own data, put your column names in their place.

2. **List your columns.**

   ```
   nt model train
   ```

   It stops and prints every column, then an empty `columns:` block.

3. **Assign the columns.** Paste that block over `columns: {}` in `data.yaml`
   and put each column you want in one list:

   ```yaml
   columns:
     Y: [Zone 1 Temp]
     U: [Zone 1 Heat]
     D: [Outdoor Temp, Solar]
   ```

   - `Y`: what the model predicts. Zone temperatures, meters.
   - `U`: what a controller will set. Setpoints, heat or valve commands.
   - `D`: what you know without the building. Weather, schedules.

   Leave a column out and the model ignores it. Put a name with brackets, a
   colon or a `#` in quotes: `"Zone Temp [°C]"`.

4. **Smoke test.** Two passes over the data: seconds, and it catches every
   misspelled name. Its numbers mean nothing yet.

   ```
   nt model train --config debug
   ```

5. **Train.**

   ```
   nt model train --set progress=log
   ```

   `--set progress=log` prints a line every 10 epochs (passes over the data)
   with the error on train and on dev. Both should fall. It stops on its own
   once dev stops improving. Leave the flag off for a quiet run.

   At the end it prints each output's error on test beside persistence, the
   forecast that nothing changes from the last reading. Skill is the share
   of that error the model removes: 0.3 is 30% less. Above 0 beats it.

6. **Look at the results.**

   ```
   uv run mlflow ui
   ```

   Open http://localhost:5000 in a browser, pick the run, and look under
   Artifacts for the plots and `reportcard.md`. The Verdict at the top of
   the reportcard sums it up.

7. **Change a setting.** `model_configs/modeling.yaml` lists every setting at
   its default, with a note on each. Copy the ones you want into
   `model_configs/custom.yaml`, edit them there, and run:

   ```
   nt model train --config custom --set progress=log
   ```

   Make as many config files as you like, and run any of them with
   `--config <its name>`. A file per experiment makes them easy to compare.

Where next:

- a zone as walls and air, with physical parameters: `2-rc-network.md`
- letting the data choose that zone's network: `3-rc-ladder.md`
- a zone behind a thermostat's setpoints: `4-thermostat.md`
- several zones, or a model in parts: `5-multi-block.md`. It needs only
  this guide: the model you just trained already predicts every zone you
  list as a Y.
- a controller: `6-first-controller.md`

Good to know:

- **Timing.** A Y at row t was measured at t. A U or D at row t holds for
  the step from t to t+1: it is what was applied next. A column logged the
  other way, like a meter that reports the step just ended, needs shifting
  one row when you prepare the data. The library never shifts it for you.
- **A meter is a Y, never a D.** D is what a forecast or a schedule could
  tell you in advance. Nothing forecasts a meter.
- **A price or a comfort band goes in no role.** As a D, the model would
  learn that the price drives the building. A controller's objective reads
  it straight from the file.
- **A counter that only grows** (total kWh) can't be learned from. Use what
  it adds each step instead.
- **Drop a column that never changes in `train.csv`.** The model has never
  seen it move, and goes wild the day it does.
- **The model only knows what the data did.** If a controller will push a
  setpoint somewhere the record never went, the model is guessing there.
- **The practice building is in plain sight.** `guides/make_data.py`
  simulates it, and its `ZONES` holds the truth: what each zone is made of
  and how its thermostat behaves. Later guides hold what the library finds
  against it.
