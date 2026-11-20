# A thermal zone behind a thermostat

When what you control is a setpoint, the model has two parts: a thermostat
that turns setpoints into heat, and the zone that heat warms or cools (an RC
network chosen by the ladder, `3-rc-ladder.md`). Each is fitted to your data
first, in seconds, and then trained together.

This guide is for one zone. For several, give each zone its own pair of
blocks: `5-multi-block.md`.

You need:

- the zone temperature,
- the cooling setpoint, the heating setpoint, or both,
- the heat the equipment delivers, in kW: positive heating, negative cooling,
- the outdoor temperature, and solar irradiance (optional).

1. **Line up the heat as a meter would.** Here the heat is a Y, predicted
   from the setpoints, and a Y at row t holds what was measured at t: the
   heat over the step that just ended. The practice record logs zone 1's
   heat over the step ahead, as a U wants it, so make a metered copy one
   row later when you prepare the data:

   ```python
   import pandas as pd

   for split in ("train", "dev", "test"):
       df = pd.read_csv(f"data/{split}.csv")
       df["Zone 1 Delivered"] = df["Zone 1 Heat"].shift(1).bfill()
       df.to_csv(f"data/{split}.csv", index=False)
   ```

   A heat meter's own reading is already this way round. No heat column at
   all? Make one from the supply air:

   ```python
   df["Delivered"] = 1.2 * df["Supply Flow"] * (df["Supply Temp"]
                                                - df["Zone Temp"])
   ```

   with the flow in m³/s and temperatures in °C. 1.2 is air's density times
   its heat capacity, in kJ/(m³·K), so the result is in kW.

2. **Assign the columns** in `data.yaml`. The heat is a Y: the model predicts
   it from the setpoints.

   ```yaml
   columns:
     Y: [Zone 1 Temp, Zone 1 Delivered]
     U: [Zone 1 Cooling SP, Zone 1 Heating SP]
     D: [Outdoor Temp, Solar]
   ```

3. **Declare both parts** in `nt_model.py`, with your column names in the
   first three lines:

   ```python
   TEMP, HEAT = "Zone 1 Temp", "Zone 1 Delivered"
   COOL_SP, HEAT_SP = "Zone 1 Cooling SP", "Zone 1 Heating SP"
   OUTDOOR, SUN = "Outdoor Temp", "Solar"


   def block_mapping(nm_mapping, cfg):
       return {
           "hvac": {
               "Y": [HEAT], "U": [COOL_SP, HEAT_SP], "D": [OUTDOOR],
               "model": "thermostat", "ambient": OUTDOOR,
               "reads": [TEMP],          # the zone it is holding
               "params": {"thermostat": "ladder/hvac.thermostat.json"},
           },
           "zone": {
               "Y": [TEMP], "U": [], "D": [OUTDOOR, SUN],
               "model": "rc_network", "ambient": OUTDOOR, "solar": SUN,
               "driven_by": [HEAT],      # the heat hvac delivers
               "params": {"rc_network": "ladder/zone.json"},
           },
       }
   ```

4. **Fit the zone**, then the thermostat through it:

   ```
   nt data ladder
   nt data thermostat --cooling "Zone 1 Cooling SP" \
       --heating "Zone 1 Heating SP"
   ```

   They write `ladder/zone.json` and `ladder/hvac.thermostat.json`. Each card
   has a line starting `→`: the form it chose.

5. **Train.**

   ```
   nt model train --set progress=log
   ```

   The run prints `block 'hvac': thermostat` and `block 'zone': rc_network`.

What the thermostat's file holds, per side (cooling, heating):

- `capacity`: the most it delivers, in kW. `capacity2` is a second stage.
- `gain`: kW per K of error, how hard it pushes as the zone drifts past the
  setpoint. Very large means plain on/off.
- `offset`: where it starts, in K past the setpoint.
- `capacity_per_K`: how capacity changes with outdoor temperature.

On the practice record, hold them against `ZONES[1]` and `OFFSET` in
`guides/make_data.py`: a gain of 15 kW/K, capacities of 6 kW heating and
8 kW cooling, an offset of 0.3 K, and nothing that changes with the
weather, so `capacity_per_K` near 0.

Good to know:

- **"Not placed by this record"** names a parameter your data never showed.
  The file leaves it `null` or very wide, and training refuses a `null`. Set
  it from what you know: `"offset": [-1, 1]` (starts within a degree of the
  setpoint), or `"capacity": [40, 60]` for a 50 kW furnace. A side that
  never ran, like heating in a summer record, is the usual case.
- **Only one setpoint?** Pass just `--cooling` or just `--heating`, and list
  only that one in the block's `U`.
- **Rerun both fits** whenever you change the columns or the data, the
  ladder first: the thermostat is fitted through the zone's network.
