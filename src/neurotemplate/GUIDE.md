# NeuroTemplate: the guided flow

`nt guide` prints this overview; `nt guide <step>` prints one step or guide. Each
step is a few decisions, each with the command whose output settles it. Read a step
when you reach it. The references in `docs/` (OVERVIEW, MODELING, CONTROL, EXPORT) and
the hook docstrings go deeper; open them when a step points you there, not before.

| step | decides | ends with |
|---|---|---|
| `nt guide setup` | the project, the column roles, the data | a smoke run that loads your data |
| `nt guide model` | the model's structure and priors | a reportcard whose physics checks pass |
| `nt guide check` | whether the model is fit to plan against | the reference rule scored on the model next to the data |
| `nt guide control` | the objective, what the controller sees, its bounds | a controller that beats the rule on the model, read against the best plan |
| `nt guide export` | the package that runs without the library | a verified export and a passing check |

A guide is one kind of system or one way of working, with the structure and the
settings worked out for it. Start from the one that fits and adapt it:

| guide | for |
|---|---|
| `nt guide thermal-zone` | a zone served by staged cooling and heating, its delivered heat measured |
| `nt guide own-model` | a model of your own, given the library's training, checks, control and export |
| `nt guide mpc` | planning through the model at every step instead of training a policy |

Three things decide most outcomes, in this order:

1. **The model must say what the actuator does where the controller will take it**,
   and what that costs. A controller exploits every error in the direction that pays.
2. **The objective must contain the trade you want** (comfort against cost, say),
   with the information to make it (a long enough preview of what is coming).
3. **Read every result against something**: the rule, the best plan, the previous
   run, the data. A number alone says little.

Every training run ends by printing its run id and its reportcard's path. Read the
reportcard before deciding the next run.

## setup

1. **Project.** `nt init` in the application's directory writes the hooks
   (`nt_model.py`, `nt_control.py`), the config directories and an empty `data/`;
   `nt` finds them in the directory it runs from. Put the record in `data/`, split in
   time order into `train.csv`, `dev.csv` and `test.csv`.
2. **Column roles**, in `nt_model.py`:
   - `Y`: what the model predicts. The states a controller acts on, plus any meter
     the objective prices.
   - `U`: what the controller writes.
   - `D`: what is known without the plant: weather, schedules, a calendar, anything
     a forecast can supply at run time. Nothing forecasts a meter, so a meter is
     never `D`.
3. **Data paths** in `model_configs/*.yaml`, then
   `nt model train --config model_configs/debug.yaml`. It takes seconds and catches
   every wrong name.
4. **Coverage.** The model only knows commands the data issued, and the controller's
   bounds default to that range. If the controller must go somewhere the data never
   went, the model cannot tell it what happens there.

Everything a hook takes or returns is in your data's own units.

## model

Decide the structure first; it matters more than any hyperparameter.

1. **Family.** A grey box for an output you will plan against: structure the
   physics is sure of (signs, storage, saturation) that a controller cannot argue
   with. `rc_network` is the built-in one, for a zone whose delivered heat is
   measured (`nt guide thermal-zone`). A
   model of your own goes behind the same contract (`nt guide own-model`). The black
   boxes (`mlp`, `nssm`, `gru`, `lssm`) fit freely; for planning, give them sign
   priors (step 4).
2. **What reaches each output.** `block_mapping` splits the outputs into blocks, each
   seeing its own inputs (docs/MODELING.md, "Blocks"). Give an output only the inputs
   that drive it: a block that sees an input it does not depend on learns the data's
   correlations through it, and a controller exploits them.
3. **Heads**, in `data_head_config`, in physical units: a meter cannot read below
   zero (`BoundedHead(0.0, float("inf"))`), a staged output has levels
   (`OrdinalHead(n_levels=K, level_values=[...])`). `nt data heads` reads each output's
   distribution and suggests one. A head that lets a meter dip negative tells a cost
   term that running the equipment earns money.
4. **Priors.** Boxes on the grey box's parameters, from the data where a command can
   measure them (`nt data ladder`, `nt data thermostat`), wide where not. For a black box,
   a sign the physics is sure of, such as "more demand never draws less power", goes
   in `grad_path` with `grad_gain` > 0.
5. **Physics checks**, `synth_cfg`: a few lines that test the model in states the
   data never visits. A failed step response means do not plan against this model.

Reading the reportcard: the physics checks first, then each parameter in physical
units and where it sits in its box, then test error against persistence. Test error
is measured where the data went, not where a controller will go. Declaring a column
(`demand`, `heat`) changes the physics, not the shapes, so a wrong declaration shows
only on the reportcard. Before comparing two model configs, set `ensemble_seeds` to
a few: one seed is one draw.

## check

Before training a controller, score the rule you would otherwise run, on the model:
`reference_control` in `nt_control.py`, every `residual_delta` box `[0, 0]`,
the band terms only. Compare its band violation on the model with the rule's in the
data. If the model says the rule cannot hold a band the data shows it holding, the
model is wrong about the actuators, and every controller trained on it will optimise
against that error. Fix the model, not the controller.

## control

1. **Objective**, in `control_configs/control.yaml`: a band as two terms (`relu`
   below, `nrelu` above), a cost as `power` weighted by the tariff, and a small
   `first_difference` on each command. For a band you must hold, use `norm: 1` with a
   gain that makes a unit of mean excess outweigh any saving: a squared term makes a
   small overshoot nearly free, and the controller takes it. `power` is
   `power_metric`, by default the sum of `power_channels`, the meters the model
   predicts. Before training, the run prints the share each term has of the
   objective. A term near 0% is not doing anything; size the gains from that print.
   Build it up a run at a time: the band terms alone first, which should not come
   out much worse than the rule (if it does, a model iteration is worth more than
   any gain), then the full objective. The band and the tariff are reference
   columns, in a reference file per plant file: `python make_references.py --help`.
2. **What the controller sees**, `policy_window`: every column with a `past`, and for
   forecasts a `future`. To move load in time, the controller must see the expensive
   period coming. The tariff's and the band's `future` must cover the stretch from
   when the action pays off to when it is taken, and `train_pred_steps` must hold
   both the action and its payoff in one rollout (a day, for daily cycles); a
   `future` longer than `train_pred_steps` is refused. How far ahead is enough is
   set by how long the plant takes to respond: compare the preview with the
   surrogate's time constants on its reportcard. No `future` can be longer than the
   forecast available at run time: the exported controller refuses to act without it.
3. **Bounds.** Each command stays inside `actuation_bounds` (by default the data's
   range). `command_bounds` narrows a command to a range that follows a reference at
   each step, `[low, high]`, each side a number, a reference column, `[column,
   offset]`, or another command, which keeps two commands from crossing. Where one
   actuator is unpriced, bound it: a controller will run it against the priced one
   and let the cost term see only half the bill. A rule-anchored residual
   (`reference_control` plus a `residual_delta` box) is an option for a small,
   trusted correction, not the way to make a controller behave on a wrong model.
4. **Time before you train.** After its first epoch a run prints the seconds per
   epoch and what the configured epochs will cost. On a fixed budget, size `epochs`,
   `train_pred_steps` and the previews from that line. The run ends by saying whether
   patience stopped it or the epoch cap did with the dev loss still falling.
5. **Train on several draws of the price.** List the plant file once per reference
   file and give each a different tariff: the same one shifted a few hours each way,
   or with its contrast changed. A policy trained on one tariff learns its hours; one
   trained on several learns to read the price.
6. **Read the run against the best plan.** The reportcard's Controller gap sets the
   controller beside a free command per step, optimised through the same model,
   bounds and objective with the whole window known. The gap between them is the
   controller's: its inputs, structure or training. What the plan does not do either
   (move load before the expensive period, ride the band edge), the model or the
   objective does not reward, and no controller will find it.
7. **Read the plots.** Does the action come before the expensive period, with the
   relief through it? Is a command pinned at a bound, jittering, or crossing another?
8. **Its blind twin.** Train the same config with the price gain at 0. The difference
   between the two is what the price bought; a cost number alone is mostly a fact
   about the test week. Split a saving into offset and shape: offset is the mean
   difference from the rule (doing less overall), shape is doing it at different
   times (the difference in the expensive hours minus that in the cheap ones).

Never decide what to ship on the surrogate's band number: it is a fact about the
surrogate, and has been wrong by several times in both directions. Compare
controllers on the same surrogate against the rule; judge the winner on the plant.

## export

`nt export policy --run_id <control run> --out <dir>/nt_package` writes the
controller as a package that runs without the library, and checks it against the
trained one before writing it. docs/EXPORT.md covers the package layout, the C and
torch targets, and what an MPC package carries.

## thermal-zone

One zone served by staged cooling and heating, with its delivered heat measured and
its power metered. Syntax, not values: the column names, boxes and gains below
belong to some other building, and every choice is one this guide says how to check
on yours.

1. **Delivered heat is data prep, yours.** From supply flow and temperature at the
   unit or the VAV box: `1.2 · flow [m³/s] · (supply − zone) [K]` kW, positive
   heating. It is a meter: a row is the heat over the step that ends there, which is
   how a plant reports it. A zone without it takes a black box for now.
2. **Declare what each column is**, one block per zone, `model: rc_network`: `Y` the
   zone temperature and its `heat`; `U` the setpoints; `D` the outdoor temperature
   (`ambient`), the irradiance (`solar`), and any schedule that adds heat. An
   occupancy flag was worth a third of the day-ahead error on two plant buildings.
3. **`nt data ladder`** fits a ladder of RC networks to the record, 30 min steps,
   by maximum likelihood, and picks one by likelihood-ratio tests. Read its card:
   the 24 h error on dev, whiteness, BIC beside the tests, a parameter at a limit or
   barely placed. Put its snippet in the block's `params` as `network`, and edit it
   as you like: a node, an edge, an input into another node. `--candidate mine.json`
   runs your version through the same tests.
4. **`nt data thermostat --cooling COL --heating COL`** fits the map from setpoint
   to heat, one step at a time, and rolls each form out 24 h through your network,
   by regime. On cycling equipment the likelihood ratio rewards every extra
   parameter; the rollout error is the one to trust, and the card says when another
   form rolls out better. Put its snippet in `params` as `thermostat`. A side the
   record never exercised comes back unplaced (a null offset, a wide box): declare it
   from what you know of the equipment, or the build refuses.
5. **History.** A day for the zone, its heat and its `D` (288 steps at 5 min): the
   window the filter places the hidden nodes from. Setpoints at 0. `train_pred_steps`
   a day too: one rollout holds an action and its payoff. Training moves every
   parameter inside its box; the reportcard prints the time constants beside the
   ladder's and flags one that left its error bars.
6. **Meters in their own block** (`model: mlp`), reading the zone's heat: the
   compressor draws what the coil delivers. Claim every actuator in the block it
   drives: a meter that sees heating learns "heating on, so cooling off", and the
   controller runs the furnace to make the modelled cooling power fall.
7. **Bounds that suit a thermostat.** The cooling setpoint inside the band with a
   margin under its ceiling; heating under the floor, which keeps the two from
   crossing; gas heating bounded when the tariff prices electricity.

`nt_model.py`:

```python
def block_mapping(nm_mapping, cfg):
    return {
        "zone": {"Y": ["Zone Temp", "Delivered Heat"],
                 "U": ["Cooling Setpoint", "Heating Setpoint"],
                 "D": ["Outdoor Temp", "Solar", "Occupied"], "model": "rc_network",
                 "heat": "Delivered Heat", "ambient": "Outdoor Temp", "solar": "Solar",
                 "params": {"network": json.load(open("ladder/zone.json")),
                            "thermostat": json.load(open("ladder/zone.thermostat.json"))}},
        "meter": {"Y": ["Compressor Power", "Fan Power"], "U": [],
                  "reads": ["Delivered Heat"], "model": "mlp"},
    }

def data_head_config(nm_mapping, norm_stats):     # meters cannot read below zero
    heads = [BoundedHead(0.0, float("inf"))
             if c in ("Compressor Power", "Fan Power") else RegressionHead()
             for c in nm_mapping["Y"]]
    return HeadStack(heads, nm_mapping["Y"])
```

with `power_channels` naming the two meters. `control_configs/control.yaml`:

```yaml
policy_window:                  # every column; a day of band and price ahead
  "Zone Temp":        {past: 3}
  "Delivered Heat":   {past: 0}
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

Size `epochs` from the seconds-per-epoch line the run prints after its first epoch.

## own-model

Any model that fits one contract gets the rest of the library: training, the
reportcard and physics checks, reload, control (a policy, or MPC) and export.

```
model(y, u, d) -> y_next      y, u, d: [batch, history_steps, n]; y_next: [batch, ny]
```

Physical units throughout; row -1 of `u` and `d` drives this step. The full contract
is in docs/MODELING.md ("Your own model"). The decisions:

1. **Register it** in `nt_model.py`, `custom_models()` returning
   `{name: build(cfg, nm_mapping, norm_stats) -> nn.Module}`, and set `model` to the
   name. The run keeps a copy of that file, so the model's code and its starting
   values live in it, not beside it.
2. **State** past the window (a filtered mass, a latch): `rollout_state()` and
   `set_rollout_state(state)`, one `[batch, width]` tensor, `None` when cold. Cold, the
   model seeds it from its window, so `history` sets how much record seeds it. Seed in
   closed form (an exponential filter is a weighted sum over the window) rather than a
   loop over rows: it traces to a small graph.
3. **Signs and boxes**: `neurotemplate.modeling.bounded.BoundedParameter` holds a
   parameter inside `[low, high]` and reports where it landed. A coefficient that must
   not change sign gets a box with 0 at one end.
4. **What the data must carry**: calendar features are `D` columns (an hour, a
   weekday flag), so the model and the controller both see them. A model fitted to a
   quantity a controller does not command (delivered heat rather than a setpoint)
   makes a controller that commands that quantity.
5. **Fit or pin.** Start from your own fit; `lr: 0` keeps it, a small `lr` refines it
   against the rollout loss. Compare the reportcard's rollout error against
   persistence and against the starting point.
6. **Read its physics checks** like a built-in's: they test your model in states the
   data never visits, with no knowledge of its structure.

A worked case: a multi-timescale zone model (the zone, EWMAs of it over 0.25 to 168 h,
filtered weather, hour-of-day gains, delivered heat) ported from least squares: the
EWMAs seeded in closed form as state, each coefficient a `BoundedParameter` started at
its fitted value, the fitted values inline in the hooks file. At `lr: 0` it reproduced
its own fit; a few epochs of the rollout loss refined it; it exported to ONNX with its
state lifted, and a policy trained against it exported beside it.

## mpc

`policy: mpc` trains nothing: every step it plans the next `mpc_horizon` commands
through its own copy of the model, inside the same bounds, against the same objective,
with Adam from its previous plan until `mpc_patience` steps pass without the objective
improving by `mpc_rtol` of it (at most `mpc_max_iters`), and commands the first.

1. **Horizon.** Long enough to see the action and its payoff, like a policy's
   preview: every D and reference column's `future` must reach it.
2. **Effort.** `mpc_rtol` and `mpc_patience` say when a solve is done, `mpc_max_iters`
   caps it; the reportcard says how often the cap bound. The Controller gap sets MPC
   beside the best plan for the whole window: a gap that falls as `mpc_rtol` tightens
   is effort; one that does not is the horizon.
3. **Cost.** A solve per step, against a policy's one forward pass. The exported
   package's `example.py` times a command. If the plant's control period cannot
   hold a solve, train a policy and read its gap against MPC's on the same model.
4. **Export** with `--format torch`: the package plans through its own `model/`,
   with torch at run time. A model with a latch or a clamp is piecewise, so the
   export checks that nearly every window plans the trained command, not every one.
