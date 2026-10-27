# NeuroTemplate: the guided flow

`nt guide` prints this overview; `nt guide <step>` prints one step. Each step is a
few decisions, each with the command whose output settles it. Read a step when you
reach it. The references in `docs/` (OVERVIEW, MODELING, CONTROL, EXPORT) and the hook
docstrings go deeper; open them when a step points you there, not before.

| step | decides | ends with |
|---|---|---|
| `nt guide setup` | the project, the column roles, the data | a smoke run that loads your data |
| `nt guide model` | the model's structure and priors | a reportcard whose physics checks pass |
| `nt guide check` | whether the model is fit to plan against | the reference rule scored on the model next to the data |
| `nt guide control` | the objective, what the policy sees, its bounds | a policy that beats the rule on the model, read from the plots |
| `nt guide export` | the package that runs without the library | a verified export and a passing check |
| `nt guide example` | (optional) the steps above as declarations, for one thermal zone | a starting point to adapt |

Three things decide most outcomes, in this order:

1. **The model must say what the actuator does where the policy will take it**, and
   what that costs. A policy exploits every error in the direction that pays.
2. **The objective must contain the trade you want** (comfort against cost, say),
   with the information to make it (a long enough preview of what is coming).
3. **Read every result against something**: the rule, the previous run, the data.
   A number alone says little.

Every training run ends by printing its run id and its reportcard's path. Read the
reportcard before deciding the next run.

## setup

1. **Project.** `cp -r <template> <project>` and work there; `nt` finds the hooks
   (`modeling_user_defined.py`, `control_user_defined.py`) and the config
   directories in the directory it runs from.
2. **Column roles**, in `modeling_user_defined.py`:
   - `Y`: what the model predicts. The states a controller acts on, plus any meter
     the objective prices.
   - `U`: what the controller writes.
   - `D`: what is known without the plant: weather, schedules, anything a forecast
     can supply at run time. Nothing forecasts a meter, so a meter is never `D`.
3. **Data paths** in `model_configs/*.yaml`, then
   `nt model train --config model_configs/debug.yaml`. It takes seconds and catches
   every wrong name.
4. **Coverage.** The model only knows commands the data issued, and the policy's
   bounds default to that range. If the controller must go somewhere the data never
   went, the model cannot tell it what happens there.

## model

Decide the structure in `block_mapping`; it matters more than any hyperparameter.
Every key below is in docs/MODELING.md ("Blocks", "Declaring what a column is").

1. **Family.** `2r2c` for a temperature you will plan against: a grey-box thermal
   network that is stable under any input, with storage you can read in hours. The
   black boxes (`mlp`, `nssm`, `gru`, `lssm`) fit freely; for planning, give them sign
   priors (step 5).
2. **What each input is.** In the block's `inputs`:
   - `demand` for a setpoint: the model sees the thermostat's error, not the raw
     setpoint.
   - `linear` for a delivered power or irradiance.
   - `ambient` for the outdoor temperature the zone conducts to.
3. **Meters the objective prices** go in their own block (`model: mlp`, `reads` the
   zone).
   - A meter responds to an actuator only if it sees it. A U column no block claims
     is shared, and a declared `demand` reaches every block that sees it. So leave
     the setpoint the meter draws from out of every block's `U`, and declare it in
     the zone block's `inputs`.
   - Claim every other actuator in the block it drives (a heating setpoint in the
     zone block's `U`). A meter that sees an actuator it does not draw from learns
     the data's correlations ("heating on, so cooling off"), and the policy exploits
     them: it runs the heating to make the modelled cooling power fall.
   - A meter cannot read below zero, so give it a head that cannot either:
     `BoundedHead(project_physical_to_norm(0.0, stats), float("inf"))` in
     `data_head_config`. A regression head that dips negative tells the cost term
     that running the equipment earns money.
4. **When the actuator's effect depends on conditions** (equipment that delivers
   more on a cool night than on a hot afternoon), name those columns in the demand
   entry's `modulated_by`. **When the equipment latches** (it starts above the
   setpoint and, once running, runs until the zone is some way below), set
   `hysteresis: true`. `nt data latch` checks: it prints how often the equipment runs
   at the same error, after a step running and after a step idle.
5. **Priors.** Time constants come from `nt data decay`; a `2r2c` block's
   `tau_fast_h` box cannot start below two data steps (0.17 h on 5-minute data), the
   floor its stability guarantee needs. Output heads come from
   `nt data heads`. For a black-box block, a sign the physics is sure of, such as
   "more demand never draws less power", goes in `grad_path` with `grad_gain` > 0.
   A `2r2c` zone's `history` lag is how much record seeds its hidden mass: an hour
   leaves the mass at about the zone's temperature and the fit shortens `tau_slow`
   to compensate; half a day seeds it better. A better-seeded mass makes storage look
   more valuable, which tends to buy a larger saving and more time outside the band:
   read it as a trade, not a fix.
6. **Physics checks**, `synth_cfg`: a few lines that test the model in states the
   data never visits. A failed step response means do not plan against this model.

Reading the reportcard: the physics checks first, then the time constants and each
declared branch in physical units, then test error. Test error is measured where the
data went, not where a controller will go.

## check

Before training a policy, score the rule you would otherwise run, on the model:
`reference_control` in `control_user_defined.py`, every `residual_delta` box `[0, 0]`,
the band terms only. Compare its band violation on the model with the rule's in the
data. If the model says the rule cannot hold a band the data shows it holding, the
model is wrong about the actuators, and every policy trained on it will optimise
against that error. Fix the model, not the controller.

## control

1. **Objective**, in `control_configs/control.yaml`: the band as two terms (`relu`
   below, `nrelu` above), the cost as `power` weighted by the tariff, and a small
   `first_difference` on each command. For a band you must hold, use `norm: 1` with a
   gain that makes a degree of mean excess outweigh any saving: a squared term makes a
   small overshoot nearly free, and the policy takes it. `power` is `power_metric`, by default the sum
   of `power_channels`, the meters the model predicts. Before training, the run prints
   the share each term has of the objective. A term near 0% is not doing anything;
   size the gains from that print.
2. **What the policy sees**, `policy_window`: every column with a `past`, and for
   forecasts a `future`. To move load in time, the policy must see the expensive
   period coming. The tariff's and the band's `future` must cover the stretch from
   when the action pays off to when it is taken, and `train_pred_steps` must hold
   both the action and its payoff in one rollout (a day, for daily cycles). Before
   training, the run notes any band or price it reads whose preview is shorter than
   the gap between its changes in your data, or longer than a rollout. No
   `future` can be longer than the forecast available at run time (a 288-row
   forecast that includes the current row supplies 287 ahead): the exported
   policy refuses to act without it.
3. **Bounds.** The policy writes each command inside `actuation_bounds` (by default
   the data's range). `command_bounds` narrows a command to a range that follows a
   reference at each step: a cooling setpoint inside the comfort band, a heating
   setpoint under its floor, which also keeps the two from crossing. Keep a margin
   inside an edge the equipment overshoots (`[column, offset]`): a latching thermostat
   starts only past its setpoint, so a policy pinned at the ceiling rides just over
   it. Where one
   actuator is unpriced (gas heating under an electricity tariff), bound it: with
   heating free, a policy can run the furnace against the compressor and let the cost
   term see only the electricity. A rule-anchored
   residual (`reference_control` plus a `residual_delta` box) is an option for a
   small, trusted correction. It is not the way to make a policy behave: a box narrow
   enough to keep a wrong model safe is too narrow to reach what pays.
4. **Time before you train.** After its first epoch a run prints the seconds per
   epoch and what the configured epochs will cost. On a fixed budget, size `epochs`,
   `train_pred_steps` and the previews from that line: a long preview and a day-long
   rollout are what pay, and they are also what costs. The run ends by saying whether
   patience stopped it or the epoch cap did with the dev loss still falling.
5. **Train on several draws of the price.** List the plant file once per reference
   file and give each a different tariff: the same one shifted a few hours each way,
   or with its contrast changed. A policy trained on one tariff learns its hours; one
   trained on several learns to read the price, and holds the band on a tariff it has
   not seen.
6. **Read the run**, plots first. Does the action come before the expensive period,
   with the relief through it? Is a command pinned at a bound, jittering, or crossing
   another? Then the reportcard's objective balance.
7. **Its blind twin.** Train the same config with the price gain at 0. The difference
   between the two is what the price bought; a cost number alone is mostly a fact
   about the test week.

## export

`nt export policy --run_id <control run> --out <dir>/nt_package` writes the policy as
a graph that runs with numpy and onnxruntime, and checks the package against the
trained policy before writing it. docs/EXPORT.md covers the package layout and the C and
torch targets.

## example

One thermal zone served by staged cooling and heating, with metered power, as the steps
above declare it. Syntax, not values: the column names, the time-constant boxes and the
gains belong to some other building, and every choice here is one the steps tell you how
to check on yours.

`modeling_user_defined.py`:

```python
def block_mapping(nm_mapping, cfg):
    return {
        # The zone claims heating only. Cooling stays shared, so the meter sees the
        # cooling demand too, and nothing it does not draw from.
        "zone": {"Y": ["Zone Temp"], "U": ["Heating Setpoint"], "model": "2r2c",
                 "ambient": "Outdoor Temp",
                 "params": {"tau_fast_h": [0.2, 1.5], "tau_ambient_h": [2.0, 20.0],
                            "tau_slow_h": [24.0, 150.0]},
                 "inputs": {
                     "Cooling Setpoint": {"kind": "demand", "sign": -1, "state": "Zone Temp",
                                          "hysteresis": True,            # nt data latch
                                          "modulated_by": ["Outdoor Temp"]},
                     "Heating Setpoint": {"kind": "demand", "sign": +1, "state": "Zone Temp"},
                     "Solar": {"kind": "linear", "sign": +1}}},
        "meter": {"Y": ["Compressor Power", "Fan Power"], "reads": ["Zone Temp"],
                  "model": "mlp"},
    }

def data_head_config(nm_mapping, norm_stats):     # meters cannot read below zero
    heads = [BoundedHead(project_physical_to_norm(0.0, norm_stats[c]), float("inf"))
             if c in ("Compressor Power", "Fan Power") else RegressionHead()
             for c in nm_mapping["Y"]]
    return HeadStack(heads, nm_mapping["Y"])
```

with sign priors on the meter (`grad_path` → `{"Compressor Power": {"Cooling Setpoint": 1},
"Fan Power": {"Cooling Setpoint": 1}}`, `grad_gain: 1.0`), and `power_channels` naming the
two meters. `control_configs/control.yaml`:

```yaml
policy_window:                  # every column; a day of band and price ahead
  "Zone Temp":        {past: 3}
  "Compressor Power": {past: 0}
  "Fan Power":        {past: 0}
  "Cooling Setpoint": {past: 1}
  "Heating Setpoint": {past: 1}
  "Outdoor Temp":     {past: 0, future: 72}
  "Solar":            {past: 0, future: 72}
  "Band Low":         {past: 0, future: 287}
  "Band High":        {past: 0, future: 287}
  "Tariff":           {past: 0, future: 287}
train_data_path: [data/train.csv, data/train.csv, data/train.csv]   # one per price draw
train_reference_path: [data/train_ref.csv, data/train_ref_later.csv, data/train_ref_earlier.csv]
train_pred_steps: 288           # one rollout holds the action and its payoff
losses:
  comfort_lower: {column: "Band Low",  target: "Zone Temp", map: relu,  norm: 1}
  comfort_upper: {column: "Band High", target: "Zone Temp", map: nrelu, norm: 1}
  cost: {column: power, norm: 1, weight: "Tariff"}
  du:   {column: "Cooling Setpoint", map: first_difference, norm: 1}
gains: {comfort_lower: 100.0, comfort_upper: 100.0, cost: 1.0e-3, du: 0.5}
command_bounds:                 # inside the band, a margin under its ceiling, heat under the floor
  "Cooling Setpoint": ["Band Low", ["Band High", -0.5]]
  "Heating Setpoint": [15.0, "Band Low"]
policy: mlp
```

Size `epochs` from the cost line the run prints before training.

