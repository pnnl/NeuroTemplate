# A thermal zone as an RC network

An `rc_network` models a zone the way you'd sketch it: air and walls as
capacitors, the paths between them as resistors. You get physical
parameters you can sanity-check, and a model that stays sensible where the
data is thin. Start from a working first model (`1-first-model.md`).

This guide writes a network by hand, as an example: two nodes, and a range
for each value. It's a starting point, as the first model's size was, not
the right network for your zone. On the practice record it isn't the
building's network, on purpose. The next guide lets the data choose, and
you can see the difference.

This guide is for one zone. For several, give each zone its own block:
`5-multi-block.md`.

You need four columns (in the practice record, zone 1's):

- the zone temperature (`Zone 1 Temp`),
- the heat into the zone, in kW: positive heating, negative cooling
  (`Zone 1 Heat`),
- the outdoor temperature (`Outdoor Temp`),
- solar irradiance, in W/m², optional (`Solar`).

If what you control is a setpoint rather than the heat, follow
`4-thermostat.md` instead.

1. **Assign the columns** in `data.yaml`:

   ```yaml
   columns:
     Y: [Zone 1 Temp]
     U: [Zone 1 Heat]
     D: [Outdoor Temp, Solar]
   ```

2. **Declare the zone** in `nt_model.py`. Two nodes: the room air (`Ti`)
   and the envelope (`Te`), the walls and roof between it and outdoors:

   ```
     Heat, sun               sun
         ↓                    ↓
        Ti ────[Ti-Te]────── Te ────[Te-ambient]──── outdoors
     room air             envelope
   ```

   Each node holds heat (a capacity, in kWh/K), and each line between two
   resists its flow (a resistance, in K/kW). Each value is a `[low, high]`
   box, and training finds where in it your zone sits. Put your column
   names in the first two lines; the rest stays.

   ```python
   TEMP, HEAT = "Zone 1 Temp", "Zone 1 Heat"
   OUTDOOR, SUN = "Outdoor Temp", "Solar"

   ZONE = {
       "nodes": {                     # heat capacity, kWh/K
           "Ti": [0.5, 5],            # room air and furnishings
           "Te": [5, 50],             # envelope
       },
       "edges": {                     # thermal resistance, K/kW
           "Ti-Te": [0.1, 10],
           "Te-ambient": [0.1, 10],   # envelope to outdoors
       },
       "inputs": {                    # what heats each node
           HEAT: {"Ti": 1.0},         # all of the heat goes into the air
           SUN: {"Ti": [0.0001, 0.1],   # kW per W/m²: through windows
                 "Te": [0.0001, 0.1]},  # and onto the walls
       },
       "noise": {"Ti": 0.3, "Te": 0.3, "sensor": 0.1},
   }


   def block_mapping(nm_mapping, cfg):
       return {
           "zone": {
               "Y": [TEMP], "U": [HEAT], "D": [OUTDOOR, SUN],
               "model": "rc_network",
               "ambient": OUTDOOR,       # what the envelope conducts to
               "solar": SUN,
               "params": {"rc_network": ZONE},
           },
       }
   ```

3. **Train.**

   ```
   nt model train --set progress=log
   ```

   The run prints `block 'zone': rc_network`. The reportcard's Model
   section lists the zone's time constants: how fast the air and the
   envelope respond, in hours. On the practice record it does worse than
   your first model: its structure isn't the building's, and training only
   moves values inside it. The next guide's network beats both.

Good to know:

- **No solar column?** Drop `SUN` from `inputs`, from `D`, and the `solar`
  line.
- **The heat must be in kW**, the heat actually reaching the zone. That
  is what makes capacities come out in kWh/K and resistances in K/kW.
  Convert W to kW in your data prep.
- **Heat input is electric power?** Add `"scale": {HEAT: 3.0}` for a known
  COP of 3. Or, if you know the room's capacity instead, add
  `"scale": {"C_Ti": 2.0}`, and the fit finds the COP. Two heat inputs in
  one zone? Add `"scale"` naming the one in kW.
- **The first day of each file is spinup.** Nobody measures the walls'
  temperature, so the model watches a day of data to work out where it must
  be, and scores nothing there. Test on 15 days to score 14. The `spinup`
  setting changes how long.
- **`noise`** is how much each temperature wanders that the inputs don't
  explain, and the sensor's error, in K. The spinup uses it. Leave it be.
- **A value stopped at the edge of its box?** The reportcard flags it with
  "At a box edge". Widen that box and train again.
- **More inputs:** add a column that is never negative, such as a 0/1
  occupancy flag, to the zone's `D` and to `inputs`, with a box for the kW
  it adds: `"Occupied": {"Ti": [0.1, 10]}`.
- **Your heat is measured, not commanded?** Row t should hold the heat over
  the step from t to t+1. If your meter reports it at the end of the step,
  shift it back one row when you prepare the data. The practice record's
  heat is already logged this way.

## Next: let the data choose the network

This network is a guess at a structure, and training only searches inside
its boxes. On the practice record the guess is wrong: the building's zones
have a heavy floor slab inside the room, not an envelope between the room
and outdoors. Nothing in the reportcard names the wrong guess. The error is
just larger than it needs to be.

We have implemented the model selection protocol of Bacher & Madsen (2011),
*Identifying suitable models for the heat dynamics of buildings*. It fits a
ladder of networks to your data, from one node up, keeps the simplest one
the data supports, and writes it in this same form, with fitted values.
The next guide, `3-rc-ladder.md`, shows how.
