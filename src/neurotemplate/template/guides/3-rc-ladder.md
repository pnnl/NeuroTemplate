# Let the data choose the RC network

`nt data ladder` picks a zone's RC network for you, with the model selection
protocol of Bacher & Madsen (2011), *Identifying suitable models for the heat
dynamics of buildings*. It starts from one node and climbs:

- `Ti`: the room air, conducting to outdoors,
- `Tm`: a mass inside the room, such as a floor slab,
- `Te`: the envelope, between the room and outdoors,
- `Am`, `Ae`: the sun landing on the mass or the envelope too,
- `Ria`: a direct path from the room to outdoors beside the envelope.

At each rung it fits every one-step extension to your data, keeps the best,
and stops when the next one isn't a real improvement. It takes seconds.

Start from the zone you declared in `2-rc-network.md`.

1. **Fit the ladder.**

   ```
   nt data ladder
   ```

   It reads your `rc_network` blocks from `nt_model.py` (the zone, its
   inputs and the outdoor temperature) and needs nothing else. The network
   you wrote isn't used here: the ladder builds its own.

2. **Read the card.** It prints one line per zone: `→` and the network it
   chose, its time constants, and the file it wrote. On the practice
   record, `→ TiTm`: the air and a mass inside the room. Add `-v` for the
   full card, one row per network it tried. The columns worth a look:

   - `p vs parent`: under 0.05 means the extra node earned its place,
   - `RMSE 1 h / 24 h`: how far off it forecasts, an hour and a day ahead,
     on dev, which it wasn't fitted to.

   The line starting with `→` names the network it chose. Below it, notes
   on anything the data couldn't pin down.

3. **Hold it against yours.** Save the network you wrote as JSON, from
   Python in this folder:

   ```python
   import json, nt_model
   json.dump(nt_model.ZONE, open("mine.json", "w"))
   ```

   and fit it beside the ladder's, through the same tests:

   ```
   nt data ladder --candidate mine.json
   ```

   It shows up in the card as `mine (yours)`. On the practice record, its
   day-ahead RMSE is several times `TiTm`'s.

4. **Hold it against the truth**, on the practice record. The ladder wrote
   `ladder/zone.json`, one file per block, named after it. Open it beside
   `ZONES[1]` in `guides/make_data.py`:

   | `ladder/zone.json`        | `make_data.py` | truth            |
   |---------------------------|----------------|------------------|
   | `nodes` `Ti`              | `C_air`        | 1.5 kWh/K        |
   | `nodes` `Tm`              | `C_slab`       | 20 kWh/K         |
   | `edges` `Ti-Tm`           | `R_slab`       | 0.4 K/kW         |
   | `edges` `Ti-ambient`      | `R_out`        | 3.0 K/kW         |
   | `inputs` `Solar` `Ti`     | `sun`          | 0.004 kW per W/m² |

   Each value is a `[low, high]` box around the fit. They land near the
   truth, some just off it: four weeks of noisy record, a thermostat in the
   loop, fitted at 30-minute steps. Training refines them.

5. **Use it.** In `nt_model.py`, delete `ZONE` and point the block at the
   file:

   ```python
   "params": {"rc_network": "ladder/zone.json"},
   ```

   Same block as before; only `params` changed. It names the file instead
   of holding the network.

6. **Train.**

   ```
   nt model train --set progress=log
   ```

   Training starts from the ladder's fit instead of the middle of each box.
   The reportcard's Model section lists the time constants training found
   beside the ladder's.

Good to know:

- **Rerun the ladder** whenever you change the zone's columns or data. It
  overwrites `ladder/zone.json`, and the next training run reads the new
  one.
- **Why the file is a path, not loaded in Python:** the ladder reads
  `nt_model.py` too, before the file exists. The library opens it only when
  training builds the model. So a new zone can start here: declare its
  block with the path, then run the ladder.
- **An input fitted at zero is left out** of `ladder/zone.json`, and the
  `-v` card says so. Leave it in the block's `D`: training notes it and the
  zone ignores it.
- **"Not placed by this record"** means the data can't see one of the
  network's time constants: slower than the record is long, or faster than
  a step. Common when the HVAC holds the zone steady. The file keeps a wide
  box there; narrow it from design data if you have it.
- **The file is yours to edit.** Narrow a box you know from design data, or
  pin a value to a single number.
