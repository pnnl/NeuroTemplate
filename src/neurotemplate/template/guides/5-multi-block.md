# Several zones: a model in parts

Your first model already handles several zones: list every zone's
temperature as a Y, and one model predicts them all. This guide needs only
that one (`1-first-model.md`).

One model for everything can pick up false links between zones: the north
zone "responding" to the south zone's heat because they happened to move
together. With blocks, each part gets its own model and sees only what you
give it. Each can be a different kind of model.

The rules:

- every Y column belongs to exactly one block;
- a U column belongs to one block. A U no block lists goes to all of them,
  such as one air handler serving every zone;
- any block may list any D column.

1. **Assign the columns** in `data.yaml`, every zone's together. In the
   practice record:

   ```yaml
   columns:
     Y: [Zone 1 Temp, Zone 2 Temp, Zone 3 Temp]
     U: [Zone 1 Heat, Zone 2 Heat, Zone 3 Heat]
     D: [Outdoor Temp, Solar]
   ```

2. **Give each zone a block** in `nt_model.py`. Write the block once and
   loop:

   ```python
   OUTDOOR, SUN = "Outdoor Temp", "Solar"
   ZONES = [1, 2, 3]


   def block_mapping(nm_mapping, cfg):
       blocks = {}
       for n in ZONES:
           blocks[f"zone{n}"] = {
               "Y": [f"Zone {n} Temp"], "U": [f"Zone {n} Heat"],
               "D": [OUTDOOR, SUN],
               "model": "nssm",
           }
       return blocks
   ```

   `model` is the block's kind: a neural network, `nssm` (the default),
   `mlp`, `gru` or `lssm`, or `rc_network` or `thermostat` below.

3. **Train.**

   ```
   nt model train --set progress=log
   ```

   The run lists each block with its kind, and the test error per output.

## Mixing in physics

Any block can be another kind. Make a zone an RC network when you have its
heat in kW and want physical parameters you can check, or a model that
stays sensible outside the record (`2-rc-network.md` says what one is).
Leave the rest black box. Here zone 1 becomes one:

```python
OUTDOOR, SUN = "Outdoor Temp", "Solar"
ZONES = [1, 2, 3]
RC = [1]                  # the zones that are RC networks


def block_mapping(nm_mapping, cfg):
    blocks = {}
    for n in ZONES:
        block = {"Y": [f"Zone {n} Temp"], "U": [f"Zone {n} Heat"],
                 "D": [OUTDOOR, SUN], "model": "nssm"}
        if n in RC:
            block.update({
                "model": "rc_network", "ambient": OUTDOOR, "solar": SUN,
                "params": {"rc_network": f"ladder/zone{n}.json"},
            })
        blocks[f"zone{n}"] = block
    return blocks
```

Then fit the RC networks and train. One command fits every `rc_network`
block, each to its own `ladder/<block>.json` (`3-rc-ladder.md` shows how
to read what it wrote):

```
nt data ladder
nt model train --set progress=log
```

Every zone an RC network is `RC = ZONES`.

## Behind thermostats

When what you control is each zone's setpoints, give each zone the pair of
blocks from `4-thermostat.md`, `hvac{n}` and `zone{n}`, inside the same
loop, with each zone's heat lined up as that guide's step 1 shows. Fit
every zone's network, then each thermostat by its setpoints:

```
nt data ladder
nt data thermostat --cooling "Zone 1 Cooling SP" --heating "Zone 1 Heating SP"
nt data thermostat --cooling "Zone 2 Cooling SP" --heating "Zone 2 Heating SP"
```

and so on. It finds the zone from the setpoints.

## Connecting blocks

- `"reads": [column]` gives a block another block's Y as an input, as of the
  step before. Use it for a zone that feels its neighbor.
- `"driven_by": [column]` gives it another block's Y from this same step.
  Use it for a meter driven by the heat a thermostat block produces, or the
  zone in `4-thermostat.md`.
- **Two blocks that need the same U**, such as a zone and the power meter of
  the unit heating it: list the U in neither block's `U`. A U no block
  claims goes to every block.

`nt docs block_mapping` has every key.
