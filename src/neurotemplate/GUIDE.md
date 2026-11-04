# NeuroTemplate: the guided flow

`nt guide` prints this overview; `nt guide <step>` prints one step or guide. Each
step is a few decisions and the command whose output settles each. Read a step when
you reach it; it points to a reference (`nt docs`) or a hook docstring when you
need more.

| step | decides | ends with |
|---|---|---|
| `nt guide setup` | the project, the column roles, the data | a smoke run that loads your data |
| `nt guide model` | the model's structure and priors | a reportcard whose physics checks pass |
| `nt guide check` | whether the model is fit to plan against | the reference rule scored on the model next to the data |
| `nt guide control` | the objective, what the controller sees, its bounds | a controller that beats the rule on the model, read against the best plan |
| `nt guide export` | the package that runs without the library | a verified export and a passing check |

A guide works out the structure and settings for one kind of system or way of
working. Start from the one that fits:

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

Every training run prints its run id and its reportcard's path. Read the reportcard
before deciding the next run.

## setup

1. **Project.** `nt init` writes the hooks (`nt_model.py`, `nt_control.py`), the
   configs and `data/`; run `nt` from that directory. Put the record in `data/` as
   `train.csv`, `dev.csv` and `test.csv`, split in time order.
2. **Column roles**, in `nt_model.py`:
   - `Y`: what the model predicts. The states a controller acts on, plus any meter
     the objective prices.
   - `U`: what the controller writes.
   - `D`: what is known without the plant: weather, schedules, a calendar, anything
     a forecast can supply at run time. Nothing forecasts a meter, so a meter is
     never `D`.

   Timing is by role, and assumed, never shifted for you: `Y` at row t is sampled at t, or measured over the step that ends at t; `U` and `D` at row t are held over the step that starts at t, `[t, t+1)`.
   A column recorded the other way is shifted in `data_mapping`.

   `nt data plot data/train.csv` then lists the record's channels under those
   roles; one left over is one the model never reads.
3. **Smoke run**: `nt model train --config model_configs/debug.yaml`. It takes
   seconds and catches every wrong name.
4. **Coverage.** The model only knows commands the data issued, and the controller's
   bounds default to that range. If the controller must go somewhere the data never
   went, the model cannot tell it what happens there.

Everything a hook takes or returns is in your data's own units.

## model

Decide the structure first; it matters more than any hyperparameter.

1. **Family.** For an output you will plan against, a grey box: structure the
   physics is sure of (signs, storage, saturation) that a controller cannot argue
   with. `rc_network` is built in, for a zone whose delivered heat is measured
   (`nt guide thermal-zone`); a model of your own fits the same contract
   (`nt guide own-model`). The black boxes (`mlp`, `nssm`, `gru`, `lssm`) fit
   freely; for planning, give them sign priors (step 4).
2. **What reaches each output.** `block_mapping` splits the outputs into blocks, each
   seeing its own inputs (`nt docs modeling`, "Blocks"). Give an output only the inputs
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
is measured where the data went, not where a controller will go. A wrong column
declaration (`demand`, `scale`) shows only here. To compare two configs, set
`ensemble_seeds` to a few: one seed is one draw.

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
   `power_metric`, by default the sum of `power_channels`. Bands and tariffs are
   reference columns, one reference file per plant file
   (`python make_references.py --help`).
2. **Build it up a run at a time.** The band terms alone first: if they come out
   much worse than the rule, a model iteration is worth more than any gain. Then the
   full objective. Before training, a run prints each term's share of the objective;
   a term near 0% does nothing, so size the gains from that print.
3. **What the controller sees**, `policy_window`: every column with a `past`, and
   forecasts with a `future`. To move load in time the controller must see the
   expensive period coming: the tariff's and the band's `future` must span from the
   action to its payoff, and `train_pred_steps` must hold both in one rollout (a day,
   for daily cycles); a `future` longer than it is refused. Size the preview against
   the surrogate's time constants on its reportcard. No `future` can exceed the
   forecast available at run time: the exported controller refuses to act without it.
4. **Bounds.** Each command stays inside `actuation_bounds` (by default the data's
   range). `command_bounds` narrows a command to a range that follows a reference at
   each step, `[low, high]`, each side a number, a reference column, `[column,
   offset]`, or another command, which keeps two commands from crossing. Where one
   actuator is unpriced, bound it: a controller will run it against the priced one
   and let the cost term see only half the bill. A rule-anchored residual
   (`reference_control` plus a `residual_delta` box) is an option for a small,
   trusted correction, not the way to make a controller behave on a wrong model.
5. **Time before you train.** After its first epoch a run prints the seconds per
   epoch and what the configured epochs will cost; size `epochs`, `train_pred_steps`
   and the previews from it. The run ends by saying whether patience stopped it or
   the epoch cap did, with the dev loss still falling.
6. **Train on several draws of the price.** List the plant file once per reference
   file and give each a different tariff: the same one shifted a few hours each way,
   or with its contrast changed. A policy trained on one tariff learns its hours; one
   trained on several learns to read the price.
7. **Read the run against the best plan.** The reportcard's Controller gap sets the
   controller beside a free command per step, optimised through the same model,
   bounds and objective with the whole window known. The gap is the controller's:
   its inputs, structure or training. What the plan does not do either (move load
   ahead of the expensive period, ride the band edge), the model or the objective
   does not reward.
8. **Read the plots.** Does the action come before the expensive period, with the
   relief through it? Is a command pinned at a bound, jittering, or crossing another?
9. **Its blind twin.** Train the same config with the price gain at 0; the
   difference is what the price bought. A cost number alone is mostly a fact about
   the test week. Split a saving into offset (the mean difference from the rule:
   doing less) and shape (the difference in the expensive hours minus that in the
   cheap ones: doing it at other times).

Do not choose what to ship by the surrogate's band violation; it has been wrong by
several times in both directions. Compare controllers on the same surrogate against
the rule, and judge the winner on the plant.

## export

`nt export policy --run_id <control run> --out <dir>/nt_package` writes the
controller as a package that runs without the library, and checks it against the
trained one before writing it. `nt docs export` covers the package layout, the C and
torch targets, and what an MPC package carries.

## thermal-zone

One zone served by staged cooling and heating, with its delivered heat measured and
its power metered. Syntax, not values: the column names, boxes and gains below
belong to some other building, and every choice is one this guide says how to check
on yours.

1. **What drives the zone is data prep, yours.** Delivered heat from supply flow and
   temperature at the unit or the VAV box: `1.2 · flow [m³/s] · (supply − zone) [K]` kW,
   positive heating; or the stage times the equipment's capacity. Metered electric power
   works too, negated for cooling, with a COP to fit (step 2). A zone with none of these
   takes a black box.
2. **One block per part, each its own signal.** A `thermostat` block produces the heat:
   `Y` the heat, `U` the setpoints, `reads` the zone. An `rc_network` block is the zone:
   `Y` the zone temperature, `driven_by` the heat (it runs after the thermostat and sees
   the heat of the step being taken), `D` the outdoor temperature (`ambient`), the
   irradiance (`solar`), and any schedule that adds heat. An occupancy flag was worth a
   third of the day-ahead error on two plant buildings. An input's gain is ≥ 0, so it must
   be non-negative: a 0/1 flag per part of the day, not sin/cos of the hour, which fits at
   zero. If the controller commands the heat rather than setpoints, there is no thermostat
   block (skip step 4): the heat is the zone's `U`. Rows mean what their role says (setup,
   step 2): a metered heat that becomes a `U` is shifted back one row.
   **`scale` says what sets the units**, and is required: temperature alone places only
   gain / C, so one thing has to be stated. `{"Delivered Heat": 1.0}` for a heat in kW
   makes C and R physical, and its error theirs. For a fitted COP, pin a capacitance
   instead, `{"C_Ti": ...}` from design data, and let the power's gain be the COP.
3. **`nt data ladder`** fits a ladder of RC networks to the record, 30 min steps,
   by maximum likelihood, and picks one by likelihood-ratio tests. Read its card:
   the 24 h error on dev, whiteness, BIC beside the tests, a parameter at a limit or
   barely placed. Put its snippet in the zone block's `params` as `network`, and edit it
   as you like: a node, an edge, an input into another node. `--candidate mine.json`
   runs your version through the same tests. The ladder needs no setpoints, only the
   zone, its inputs and the weather. It will not select a node the record cannot place (an infinite error bar, or slower than the
   record is long); on closed-loop data, where the HVAC holds the zone, that is often
   the envelope, and the card says so.
4. **`nt data thermostat --cooling COL --heating COL`** fits the map from setpoint
   to heat, one step at a time, and rolls each form out 24 h through your network,
   by regime. On cycling equipment the likelihood ratio rewards every extra
   parameter; the rollout error is the one to trust, and the card says when another
   form rolls out better. Put its snippet in the thermostat block's `params` as
   `thermostat`; it carries the gain limit the zone's capacitance sets. A side the
   record never exercised comes back unplaced (a null offset, a wide box): declare it
   from what you know of the equipment, or the build refuses.
5. **Windows.** `context_steps`, in rows, is the record the zone's filter places its
   hidden nodes from. There is no default for a network with hidden nodes: it is long
   enough when a longer one stops changing the day-ahead error, and a slow building
   needs more than a quick one. The thermal blocks are not in the `history` table, so
   list what only they read at 0. `train_pred_steps` long enough for one rollout to hold an
   action and its payoff. `train_stride` sets how far apart training windows start: left
   unset it is the whole window depth, which with a long context is few windows, and with
   `batch_size` above their number, one update per epoch. Training moves every parameter
   inside its box; the reportcard prints the time constants beside the ladder's and flags
   one that left its error bars.
6. **Meters in their own block** (`model: mlp`), `driven_by` the heat so they see the step
   being taken, `residual: false` so they say the value rather than integrate their errors:
   the compressor draws what the coil delivers. `history` lists what the meter reads,
   usually the current row. Claim every actuator in the block it drives: a meter that
   sees heating learns "heating on, so cooling off", and the controller runs the furnace
   to make the modelled cooling power fall.
7. **Bounds that suit a thermostat.** The cooling setpoint inside the band with a
   margin under its ceiling; heating under the floor, which keeps the two from
   crossing; gas heating bounded when the tariff prices electricity.

`nt_model.py`:

```python
def block_mapping(nm_mapping, cfg):
    return {
        "hvac": {"Y": ["Delivered Heat"], "U": ["Cooling Setpoint", "Heating Setpoint"],
                 "D": [], "reads": ["Zone Temp"], "model": "thermostat",
                 "params": {"thermostat": json.load(open("ladder/hvac.thermostat.json"))}},
        "zone": {"Y": ["Zone Temp"], "U": [], "D": ["Outdoor Temp", "Solar", "Occupied"],
                 "driven_by": ["Delivered Heat"], "model": "rc_network",
                 "ambient": "Outdoor Temp", "solar": "Solar", "scale": {"Delivered Heat": 1.0},
                 "params": {"network": json.load(open("ladder/zone.json"))}},
        "meter": {"Y": ["Compressor Power", "Fan Power"], "U": [], "D": ["Outdoor Temp"],
                  "driven_by": ["Delivered Heat"], "model": "mlp", "residual": False},
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

## own-model

Any model that fits one contract gets the rest of the library: training, the
reportcard and physics checks, reload, control (a policy, or MPC) and export.

```
model(y, u, d) -> y_next      y, u, d: [batch, history_steps, n]; y_next: [batch, ny]
```

Physical units throughout; row -1 of `u` and `d` drives this step. The full contract
is in `nt docs modeling` ("Your own model"). The decisions:

1. **Register it** in `nt_model.py`, `custom_models()` returning
   `{name: build(cfg, nm_mapping, norm_stats) -> nn.Module}`, and set `model` to the
   name. The run keeps a copy of that file, so the model's code and its starting
   values live in it, not beside it.
2. **State** past the window (a filtered mass, a latch): `rollout_state()` and
   `set_rollout_state(state)`, one `[batch, width]` tensor, `None` when cold. Cold, the
   model seeds it from its window, so `context_steps` sets how much record seeds it. Seed in
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

A multi-timescale zone model fitted by least squares ports this way: its EWMAs as
state seeded in closed form, each coefficient a `BoundedParameter` started at its
fitted value. At `lr: 0` it reproduces its fit, and it exports with a policy beside it.

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
