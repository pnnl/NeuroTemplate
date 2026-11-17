# A model a controller can trust

Advanced. Low test error says the model is good where your data went. A
controller goes elsewhere: it pushes setpoints to their limits, holds them
there, and finds every mistake in the model that pays. This guide checks
the model where the data never went, and closes the gaps it finds. Start
from a trained model.

1. **Add the physics checks** to `nt_model.py`. For a zone behind a
   thermostat (`4-thermostat.md`):

   ```python
   from neurotemplate.metrics.physics import SynthSpec


   def synth_cfg(nm_mapping, norm_stats):
       return SynthSpec(
           states=[TEMP],               # temperatures that settle at rest
           ambient=OUTDOOR,             # what they settle toward
           off={COOL_SP: 30.0,          # each U where it asks for nothing,
                HEAT_SP: 10.0,          # and each other Y at rest
                HEAT: 0.0},
           responds={TEMP: {COOL_SP: +1,    # raise either setpoint and
                            HEAT_SP: +1}},  # the zone ends up warmer
       )
   ```

   `off` is "the equipment is idle": a cooling setpoint above anything the
   zone reaches, a heating setpoint below it, a heat or a meter at 0. Every
   U and every Y that isn't in `states` needs one. `responds` is which way
   each state moves as a command goes up. For a heat command (`2-rc-network.md`)
   that's `{TEMP: {HEAT: +1}}`, with `off={HEAT: 0.0}`.

2. **Train**, and read the Physics section of `reportcard.md`:

   - **Step response**: each command moves each state the way you said.
     The wrong way means a controller would pull that lever backwards: don't
     control with this model until it passes. Too small a move is milder:
     note it and carry on.
   - **Equilibrium**: with nothing driving it, the zone holds still.
   - **Reachability**: how far all the commands together can push the zone
     each way. Your comfort band has to fit inside that span.
   - **Free decay**: with the equipment off, the zone drifts toward
     outdoors and doesn't overshoot it.

3. **If a step goes the wrong way**, the model has learned a coincidence
   instead of the physics. In order of what usually fixes it:

   - **Give each output only what drives it.** Split the model into blocks
     (`5-multi-block.md`) so a zone never sees another zone's heat, or a
     meter it doesn't depend on.
   - **Use a physical model** for the zone (`2-rc-network.md`). An RC
     network can't get the sign of heat wrong.
   - **Tell a neural model the signs** you are sure of. Write them to a
     file, `{output: {input: +1 or -1}}`:

     ```python
     import json

     signs = {"Zone 1 Temp": {"Zone 1 Heat": +1, "Outdoor Temp": +1}}
     json.dump(signs, open("data/signs.json", "w"), indent=2)
     ```

     and set, in `model_configs/custom.yaml`:

     ```yaml
     grad_path: data/signs.json
     grad_gain: 1.0
     ```

     List only signs that hold everywhere. They're checked at random
     states, not only the recorded ones, so "more cooling setpoint, warmer
     zone" fails wherever the cooling is off anyway.

4. **Keep meters at or above zero.** A model that lets a power meter dip
   below zero tells the controller's cost that running the equipment earns
   money, and the controller will run it. In `nt_model.py`:

   ```python
   from neurotemplate.modeling.heads import (BoundedHead, HeadStack,
                                             RegressionHead)


   def output_heads(nm_mapping, norm_stats):
       heads = [BoundedHead(0.0, float("inf")) if c == "Zone 1 Power"
                else RegressionHead() for c in nm_mapping["Y"]]
       return HeadStack(heads, nm_mapping["Y"])
   ```

Good to know:

- **Compare two models on several seeds.** Training starts from random
  numbers, so one run is one draw. `ensemble_seeds: 3` in the config trains
  three and reports their spread, before you call one change better.
- `nt docs synth_cfg`, `nt docs grad_path` and `nt docs output_heads` have
  every option.
