# Modeling

`nt model train` learns a surrogate of your system from its record: given a
window of recent outputs `y`, controls `u` and disturbances `d`, predict the
next output, then feed that prediction back to forecast as far as you need.
A controller is later trained against that rollout, so the surrogate is
judged on how it behaves over many steps, and in states the data never
visited, not only on its one-step error.

The column roles are yours to set. `Y` is what a controller or its objective
needs, `U` is what a controller can command, and `D` is what you could know
without the plant, such as weather or schedules, since a policy is later fed
`D` from a forecast.

## What you provide

The hooks in `nt_model.py`, scaffolded by `nt init` with a docstring each:

| Hook | Says | |
|---|---|---|
| `data_mapping` | which columns are `Y`, `U` and `D` | required |
| `data_head_config` | an output head per `Y` column: regression, binary, bounded or ordinal | required |
| `column_norm_config` | per-column normalization overrides | required |
| `synth_cfg` | which outputs are thermal states, for the physics checks (`None` skips them) | required |
| `sub_plot` | how output plots are grouped | required |
| `block_mapping` | blocks, a family per block, what each column is, a thermal zone's network and thermostat | optional |
| `custom_models` | models of your own, by name ("Your own model") | optional |
| `hyper_args` | the search space for `nt model optuna` | that command only |

"Required" means the function must exist, even if it returns the trivial
answer. Settings that are not about your data, such as the family default,
network size and training schedule, live in `model_configs/modeling.yaml`,
where every field is annotated. So does `history`, the one setting there
that names your columns.

```bash
nt model train --config model_configs/debug.yaml            # a smoke run first
nt model train --set epochs=200 --set ensemble_seeds=3      # one-off overrides
```

## What the model reads: `history`

`history` gives every `Y`, `U` and `D` column a lag, the steps before the
current one the model reads. Every column is listed, and there is no
default.

```yaml
history:
  "Tank Temperature [degC]": 11      # syntax, not values
  "Heater Command [-]": 0
  "Ambient Temperature [degC]": 3
```

The deepest lag plus the current step is `history_steps`: the window every
block shares, the seeded history before the first prediction, the stride
between training windows, and the rows of every column an exported model
takes. Each learned map then reads its columns at their own lags. The
family constrains the table. A `gru` reads one lag for every column it reads,
because its window is its warm-up. An `rc_network` block reads its `Y` and
`D` at one lag, the window its filter places the hidden nodes from (a day is
enough), and its setpoints at the current step only. The build refuses a
table that misses a column, a lag no block actually reads (a `U` column only
rc_network blocks read must be 0), and a `demand` setpoint lagged further
back than its state. The reportcard lists the lags
each block reads.

## Model families

| `model` | Kind | What it is |
|---|---|---|
| `mlp` | black box | one network over the flattened `[y, u, d]` window |
| `nssm` | black box | separable: `fy(y) + fu(u) + fd(d)`, each a network |
| `lssm` | black box | the same, linear |
| `gru` | black box | recurrent, with memory that outlives the input window |
| `rc_network` | grey box | a thermal zone: a declared RC network behind a thermostat, all physical |

The black boxes often fit best and are fine for prediction you won't steer
by. Nothing bounds their long rollouts, so read the stability and physics
sections of the reportcard before planning against one.

### rc_network: a thermal zone you can read

```
    setpoints ─[thermostat]→ heat delivered (kW) ─[RC network]→ zone temperature
```

For a zone whose delivered heat is measured (supply flow and temperature at
the unit or the VAV box). The block's `Y` are the zone temperature and that
heat (its `heat`, in kW, positive heating); its `U` the setpoints; its `D`
the outdoor temperature (`ambient`) and every other input the network takes
(`solar`, an occupancy schedule). Everything inside is physical:

- **The network** is capacitances (kWh/K) and resistances (K/kW) between the
  zone air, hidden nodes such as the building's mass or envelope, and the
  outdoors, with each input entering one node. Stable for any positive
  values, and its time constants read in hours.
- **The thermostat** maps each setpoint and the zone to the heat delivered:
  a tracker with a gain, an offset and a capacity per side, optionally
  capacity linear in outdoor temperature, a second stage, or a latch. Its
  gain is held under the bound past which the discrete loop would ring.
- **The hidden nodes are filtered, not guessed.** The first call of a
  rollout runs a Kalman filter over the history window, driven by the
  measured heat, and leaves every node where the record puts it.

Two commands write the declarations, so they come from the data rather than
from guesses:

- `nt data ladder` fits a ladder of networks (Bacher & Madsen 2011) to the
  record by maximum likelihood at a coarse step, 30 min by default, and
  picks one by likelihood-ratio tests. Its card prints every rung's
  likelihood, BIC, whiteness and 24 h error on dev, and the snippet it saves
  holds each C and R as a box from its error bar. An agent's own network
  (`--candidate file.json`) goes through the same tests.
- `nt data thermostat --cooling COL --heating COL` fits the thermostat's
  forms one step at a time and rolls each out 24 h on dev through the
  declared network, with the error split by regime (satisfied, tracking,
  saturated).

```python
"zone": {"Y": ["Zone Temp", "Delivered Heat [kW]"], "U": ["Cooling SP", "Heating SP"],
         "D": ["Outdoor Temp", "Solar", "Occupied"], "model": "rc_network",
         "heat": "Delivered Heat [kW]", "ambient": "Outdoor Temp", "solar": "Solar",
         "params": {"network": json.load(open("ladder/zone.json")),
                    "thermostat": json.load(open("ladder/zone.thermostat.json"))}},
```

Training then fits every parameter inside its box by the rollout loss. That
is the fine fit: the likelihood at 5 min chases sensor noise and cycling and
shortens the slow mode, the rollout loss seeded by the filter does not. The
reportcard prints the time constants beside the ladder's and flags one the
training moved outside three of its error bars.

### Blocks: one model, a family per group of outputs

A block is a group of `Y` columns with its own sub-model: its own
parameters, family and hidden state. Declaring blocks in `block_mapping` is
how one model holds, for example, an `rc_network` zone and an `mlp` power
meter that reads the zone's delivered heat:

```python
def block_mapping(nm_mapping, cfg):
    return {
        "zone":  {"Y": ["Zone Temperature", "Delivered Heat"], "U": ["Cooling Setpoint"],
                  "model": "rc_network", ...},
        "power": {"Y": ["Total Power"], "U": [], "reads": ["Delivered Heat"],
                  "model": "mlp"},
    }
```

- `Y` must partition exactly: every output in one block, none twice.
- A `U` column no block claims is shared by all of them, as one air handler
  serving several zones is. `D` has no partition rule.
- `reads` lets a block see other blocks' outputs from the history window,
  never their same-step predictions, so evaluation order never matters.
  `rc_network` refuses `reads`: its network and thermostat say where every
  column goes.

There is no conduction between blocks: they interact only through shared
inputs and `reads`.

### Declaring what a column is

A block can also say what its input columns are. The model then gets that
physics instead of having to learn it:

| Declaration | Acts in | Does |
|---|---|---|
| `demand` | every family | replaces a setpoint with the thermostat's clamped error against the block's own state, `relu(zone − setpoint)` for cooling, computed from the model's own prediction during a rollout. A setpoint above the zone is then free in the surrogate, as it is in the plant. |
| `ambient`, `heat`, `solar` | `rc_network` | the outdoor temperature the network conducts to, the delivered heat it is driven by, the irradiance the ladder may route into more than one node |

Nothing is inferred from column names. A declaration changes the
physics but not the shapes, so a wrong one shows only in the reportcard.

### Your own model

Any model that fits one contract gets the rest of the library: training,
the reportcard and physics checks, reload, control and export.

```
model(y, u, d) -> y_next
  y  [batch, history_steps, ny]   the last history_steps outputs, oldest first
  u  [batch, history_steps, nu]   the matching inputs; row -1 drives this step
  d  [batch, history_steps, nd]   the matching disturbances
  -> [batch, ny]                  the next outputs
```

Every value is in your data's own units. Register a builder by name in
`nt_model.py` and set `model` to that name:

```python
def custom_models():
    return {"my_zone": lambda cfg, nm_mapping, norm_stats: MyZone(...)}
```

- It is one `torch.nn.Module` over every column: `block_mapping` returns
  `None` and every head is a `RegressionHead`. A bound or a level belongs
  inside the model; `neurotemplate.modeling.heads` has the activations.
- State that outlives the window (a filtered mass, a latch) goes through two
  methods, `rollout_state()` and `set_rollout_state(state)`, the state one
  `[batch, width]` tensor. When the state is `None` the model is cold and
  seeds it from its window; the library resets it between rollouts and
  lifts it into the exported graphs.
- To fit it from data it must be differentiable: at `lr > 0` training
  refuses a trainable parameter with no gradient path to the loss (freeze
  one on purpose with `requires_grad_(False)`). A controller trains or plans
  through its gradient with respect to U, so `nt control train` refuses a
  model with no gradient path to U. For a fixed model, set `lr: 0`:
  nothing trains, the run is the model as built, and the reportcard says
  so. The training loss scores each output scaled by its
  `norm_stats` stretch. `contract.Normalized` puts a model that computes in
  normalised units behind the contract, as the built-in families are.
- The sampled gradient penalty is a built-in family's (`grad_gain: 0`).
  Export to ONNX or C needs the operators in `export.trace.ALLOWED_OPS`;
  `--format torch` takes any.
- The run keeps a copy of the hooks file, so the model's code travels with
  it. Keep it in that one file.

## What every run reports

Fit metrics cannot see the failures that cost the most, so every run also
reports these, in `reportcard.md` and in MLflow. Each is an instrument: read
it against the previous run or against persistence, not against a fixed
threshold.

- **Rollout stability**: how far a long rollout rises and drifts, against
  the ground truth's own. This catches a model that has stopped responding
  to its inputs and emits a ramp, which a short-window error will not.
- **Physics checks**, when `synth_cfg` describes the thermal states. They
  test four things in synthetic states the data never contains:
  equilibrium, step response, reachability and free decay, each answered in
  physical units. A failed step response means not to plan against this
  surrogate. A small equilibrium drift after an honest attempt is worth
  noting and moving past.
- **Readout channels**: order statistics for every output that is not a
  thermal state. There is no verdict, but a readout that is wrong at
  equilibrium feeds back into the states that are checked.
- **Level metrics** for binary and ordinal outputs, scored as a classifier
  per level. Staged equipment is a rare-event channel, where a low error
  can mean the model never predicts the event.
- **Persistence**: the same metrics for holding the last measured state,
  beneath the model's.

Staged outputs get two things set from the data, both on by default: each
level's loss is weighted by how rare it is, and an ordinal head's decision
thresholds are moved after training until the predicted level shares match
the true ones. The kept thresholds are saved with the run.

Test error is measured on the excitation in your data, which is rarely
where a controller will drive the plant. The stability and physics sections
are the better guide to a surrogate for control.

## Preparing the record

The loader refuses an unevenly spaced timestamp or a NaN; training never
invents data. `neurotemplate.prep` has small functions for cleaning a
record yourself. They work on a pandas Series or DataFrame with a
DatetimeIndex, and the library never calls them:

```python
from neurotemplate.prep import average, expand, fill_gaps, smooth

df = average(raw, "5min")                          # onto an even grid
df = fill_gaps(df, "15min")                        # interpolate short NaN runs
tariff = expand(hourly_tariff, "5min", "hold")     # a held quantity
hourly_oat = hourly_oat.fillna(smooth(hourly_oat, period="1D"))  # longer gaps
oat = expand(hourly_oat, "5min", "quadratic")      # a smooth physical one
```

`smooth` needs the `prep` extra. Before training, `nt data heads` reports
which output head a column's distribution suggests, and `nt data ladder`
and `nt data thermostat` a thermal zone's network and thermostat (see
"rc_network"). All are advice, not checks.

## Searching, rescoring, scripting

```bash
nt model optuna --set n_trials=50                         # search space: hyper_args
nt model load --run_id <run> --data data/next_week.csv    # score a run on other data
```

From Python, each command is one call, and a training call opens and closes
its own MLflow run:

```python
from neurotemplate.common.config import load_config
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.load import evaluate_run, load_trained_dynamics
from neurotemplate.modeling.train import train_model

cfg = load_config("model_configs/modeling.yaml", ModelingConfig, {"epochs": 50})
run = train_model(cfg)                           # run.run_id, run.test_results
results = evaluate_run(run.run_id, ["data/next_week.csv"])
dyn_sys, cfg, nm_mapping, norm_stats = load_trained_dynamics(run.run_id)
```

`load_config` is `--config` plus `--set`. Scoring always uses the run's own
normalization, never statistics fitted to the new data.

A training call sets two process-wide torch settings from its config, as the
command line does: the CPU thread count (`threads`, 1 unless set; `null`
leaves torch's own) and, off the CPU, the default device. A reload runs on
this machine, not wherever the run trained: `load_trained_dynamics(run_id,
device="cuda")`, or no `device` for CUDA if present and the CPU otherwise.

A script that retrains on a schedule, or across several sites, should know
three things:

- **A run records data paths, not data.** Write each pull to its own file;
  a file overwritten in place leaves earlier runs pointing at data they
  never saw.
- **A run copies its hooks file only.** Code the hooks import from an
  installed package is not copied, so a reload uses whatever version is
  installed then.
- **A hooks file is imported once per path per process.** Rewriting one
  file between sites keeps the first version; give each site's hooks their
  own path.
