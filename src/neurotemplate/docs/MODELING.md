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

**Timing, by role.** `Y` at row t is sampled at t, or measured over the step
that ends at t (a meter). `U` and `D` at row t are held over the step that
starts at t, `[t, t+1)`: the command applied, the weather in force. The
library assumes this and never shifts a column; a column recorded the other
way is shifted in `data_mapping`.

## What you provide

The hooks in `nt_model.py`, scaffolded by `nt init` with a docstring each:

| Hook | Says | |
|---|---|---|
| `data_mapping` | which columns are `Y`, `U` and `D`; config's `exclude_columns` then drops some | required |
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

The table lists what learned maps read: each column at its own lag. A `gru`
reads one lag for every column it reads, because its window is its warm-up. The
build refuses a table that misses a column, a lag no block actually reads, and a
`demand` setpoint lagged further back than its state. The reportcard lists the
lags each block reads.

`context_steps` is the record a model that keeps state starts from, beyond what
the table asks: an `rc_network`'s filter places its hidden nodes from it. The
thermal blocks are not in the table, so a column only they read is listed at 0.

The larger of the deepest lag plus one and `context_steps` is `history_steps`:
the rows of every column a model is handed, the seeded history before the first
prediction, and the rows an exported model takes. Training windows start
`train_stride` rows apart, by default `history_steps`; a long context then means
few windows, and with `batch_size` above their number, one update per epoch.

## Model families

| `model` | Kind | What it is |
|---|---|---|
| `mlp` | black box | one network over the flattened `[y, u, d]` window |
| `nssm` | black box | separable: `fy(y) + fu(u) + fd(d)`, each a network |
| `lssm` | black box | the same, linear |
| `gru` | black box | recurrent, with memory that outlives the input window |
| `rc_network` | grey box | a thermal zone: a declared RC network, driven by a commanded input or another block's output, all physical |
| `thermostat` | grey box | setpoints to heat: a tracker or latch with physical parameters, a block of its own |

The black boxes often fit best and are fine for prediction you won't steer
by. Nothing bounds their long rollouts, so read the stability and physics
sections of the reportcard before planning against one.

### rc_network and thermostat: a thermal zone you can read

```
    setpoints ─[thermostat block]→ heat ─[rc_network block]→ zone temperature
                    heat from a U ───────[rc_network block]→ zone temperature
```

Two blocks, each one signal, neither knowing the other exists. What the zone is
driven by is data prep: delivered heat (metered from supply flow and temperature,
or estimated from the stage and the equipment's capacity), in kW, positive heating,
or metered electric power through a gain that is its COP.

- **`rc_network`.** `Y` is the zone temperature alone. Its network's inputs are a
  `driven_by` column (the heat a thermostat block produced this step), a `U` (a
  commanded heat) or a `D`. `ambient` is the outdoor temperature it conducts to.
- **`thermostat`.** `Y` is the heat, `U` the setpoints it follows, `reads` the zone,
  `D` and `ambient` the outdoor temperature, which some of its forms read.

**`scale` says what sets the units, and is required.** Scaling every C and gain up
and every R down by the same factor leaves every prediction where it was, so
temperature alone places only gain / C. `{"Delivered Heat": 1.0}` pins a heat's
gain at 1, which makes C and R physical, so an estimated heat off by a fifth puts C
and R off by a fifth. `{"C_Ti": ...}` pins a capacitance from design data instead,
and leaves a power's gain to fit as a COP. A network that pins nothing is refused.
Everything inside is physical:

- **The network** is capacitances (kWh/K) and resistances (K/kW) between the
  zone air, hidden nodes such as the building's mass or envelope, and the
  outdoors, with each input entering one node. Stable for any positive
  values, and its time constants read in hours.
- **The thermostat** maps each setpoint and the zone to the heat delivered:
  a tracker with a gain, an offset and a capacity per side, optionally
  capacity linear in outdoor temperature, a second stage, or a latch. Its
  gain is held under its `gain_limit`, `C_zone / dt`, the bound past which the
  discrete loop would ring.
- **The hidden nodes are filtered, not guessed.** The first call of a rollout runs
  a Kalman filter over the `context_steps` window, driven by the recorded inputs,
  and leaves every node where the record puts it. After that the zone needs one
  row: the nodes carry over.

Two commands write the declarations, so they come from the data rather than
from guesses:

- `nt data ladder` fits a ladder of networks (Bacher & Madsen 2011) to the
  record by maximum likelihood at a coarse step, 30 min by default, and
  picks one by likelihood-ratio tests. Its card prints every rung's
  likelihood, BIC, whiteness and 24 h error on dev, and the snippet it saves
  holds each C and R as a box from its error bar. An agent's own network
  (`--candidate file.json`) goes through the same tests. It reads every block
  that declares `scale`, and needs no setpoints. It does not select a node the record cannot place: an infinite error bar,
  or a time constant longer than the record. Across several zones the
  summed BIC must fall as well as the ratio test pass, since a likelihood
  summed over many zones passes it for almost any parameter. Every gain
  input is fitted ≥ 0: give it as non-negative indicators or schedules.
- `nt data thermostat --cooling COL --heating COL` fits the thermostat's
  forms one step at a time and rolls each out 24 h on dev through the
  declared network, with the error split by regime (satisfied, tracking,
  saturated).

```python
"hvac": {"Y": ["Delivered Heat [kW]"], "U": ["Cooling SP", "Heating SP"],
         "D": ["Outdoor Temp"], "ambient": "Outdoor Temp", "reads": ["Zone Temp"],
         "model": "thermostat",
         "params": {"thermostat": json.load(open("ladder/hvac.thermostat.json"))}},
"zone": {"Y": ["Zone Temp"], "U": [], "D": ["Outdoor Temp", "Solar", "Occupied"],
         "driven_by": ["Delivered Heat [kW]"], "model": "rc_network",
         "ambient": "Outdoor Temp", "solar": "Solar", "scale": {"Delivered Heat [kW]": 1.0},
         "params": {"rc_network": json.load(open("ladder/zone.json"))}},
# commanded: the heat is the zone's U, and there is no thermostat block
"zone": {"Y": ["Zone Temp"], "U": ["Heat Command [kW]"],
         "D": ["Outdoor Temp", "Solar", "Occupied"], "model": "rc_network",
         "ambient": "Outdoor Temp", "solar": "Solar", "scale": {"Heat Command [kW]": 1.0},
         "params": {"rc_network": json.load(open("ladder/zone.json"))}},
```

Training then fits every parameter inside its box by the rollout loss. That
is the fine fit: the likelihood at 5 min chases sensor noise and cycling and
shortens the slow mode, the rollout loss seeded by the filter does not. The
reportcard prints the time constants beside the ladder's and flags one the
training moved outside three of its error bars.

### Blocks: one model, a family per group of outputs

A block is a group of `Y` columns with its own sub-model: its own
parameters, family and hidden state. Declaring blocks in `block_mapping` is
how one model holds, for example, a thermostat, an `rc_network` zone and an
`mlp` power meter on the heat the thermostat delivers:

```python
def block_mapping(nm_mapping, cfg):
    return {
        "hvac":  {"Y": ["Delivered Heat"], "U": ["Cooling Setpoint"], "reads": ["Zone Temperature"],
                  "model": "thermostat", ...},
        "zone":  {"Y": ["Zone Temperature"], "U": [], "driven_by": ["Delivered Heat"],
                  "model": "rc_network", ...},
        "power": {"Y": ["Total Power"], "U": [], "driven_by": ["Delivered Heat"],
                  "model": "mlp", "residual": False},
    }
```

- `Y` must partition exactly: every output in one block, none twice.
- A `U` column no block claims is shared by all of them, as one air handler
  serving several zones is. `D` has no partition rule.
- `reads` lets a block see another block's output as of the row before the step.
- `driven_by` lets it see that output as of this step: the block runs after its
  producer, and the column's rows mean what a `U`'s do, row j held over the step
  from j to j + 1, the last row what the producer just wrote. A heat is `driven_by`:
  a meter or a zone reading it through `reads` is a step late. A cycle of
  `driven_by` is refused; close a loop through `reads`, as a thermostat does.
- `residual` lets a block say whether it predicts a change on the last row or the
  value itself, whatever the run's `residual` says. A meter says the value: under a
  residual it integrates its own errors.

There is no conduction between blocks: they interact only through shared
inputs, `reads` and `driven_by`.

**Picking a family takes one line.** Without a hook, the model is one block over
every column. With one, a block takes its family from the first of these that
names one, and every run prints which it took and from where:

1. `block_models: {zone: nssm}` in the config;
2. the block's own `model` in `block_mapping`;
3. `model` in the config.

So a hook that names no families swaps every block by changing `model`, and any
one block by a line of `block_models`. Any family can sit in any block. `params`
is keyed by the family that reads it, `{"rc_network": ...}` or `{"thermostat": ...}`,
so a block keeps its priors while it is swapped out of that family. The columns a
family needs still have to be declared: a block swapped into `gru` needs one lag
for every column it reads, and the build says so.

### Declaring what a column is

A block can also say what its input columns are. The model then gets that
physics instead of having to learn it:

| Declaration | Acts in | Does |
|---|---|---|
| `demand` | every family | replaces a setpoint with the thermostat's clamped error against the block's own state, `relu(zone − setpoint)` for cooling, computed from the model's own prediction during a rollout. A setpoint above the zone is then free in the surrogate, as it is in the plant. |
| `ambient`, `scale`, `solar` | `rc_network`, `thermostat` | the outdoor temperature the network conducts to, what sets its units, the irradiance the ladder may route into more than one node |

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

It is the same contract the built-in families are behind: each is a
`stepper.Stepper` (normalise, derive inputs, core, residual, heads, physical
units) around a core, and the pieces are yours to use. A short `forward` over
them is the natural way to write what blocks cannot say, such as a signal no
column records:

```python
from neurotemplate.modeling.contract import named_columns
from neurotemplate.modeling.thermal import RCZone, thermostat

class Building(nn.Module):
    def __init__(self, nm_mapping, window, dt_h):
        super().__init__()
        self.columns = nm_mapping
        self.hvac = thermostat(json.load(open("ladder/hvac.thermostat.json")))
        self.zone = RCZone(json.load(open("ladder/zone.json")), ["Heat"], window, dt_h)

    def forward(self, y, u, d):
        c = named_columns(self.columns, y, u, d)
        heat = self.hvac(u[:, -1], c["Zone"][:, -1])
        # A meter reports a step late; `heat` is the step being taken.
        held = torch.cat([c["Heat"][:, 1:], heat[:, None]], dim=1)
        return torch.stack([self.zone(c["Zone"], c["Outdoor"], held), heat], dim=-1)
```

- It is one `torch.nn.Module` over every column, so `block_mapping` returns
  `None`: blocks are the built-in families' structure. A bound or a level goes
  inside the model. If it owns a `head_stack`, as a `Stepper` does when handed one,
  the library's level metrics and calibration use it; a head declared in
  `data_head_config` that the model does not own is refused.
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
  `norm_stats` stretch. `stepper.Stepper` puts a core that computes in
  normalised units behind the contract, as the built-in families are.
- The sampled gradient penalty differentiates its contract call, from cold.
  Export to ONNX or C needs the operators in `export.trace.ALLOWED_OPS`;
  `--format torch` takes any.
- The run keeps a copy of the hooks file, so the model's code travels with
  it. Keep it in that one file.

## What every run reports

`reportcard.md` opens with a verdict, one line per instrument, then reports
exceptions rather than inventories: healthy channels are counted, the worst
are named. Each instrument's full output is a JSON file in the run's
`results/` folder, and the headline numbers are MLflow metrics. Read each
against the previous run or against persistence, not against a fixed
threshold.

- **Accuracy**: rollout RMSE over `test_window` steps beside persistence
  (holding the last measured state), and skill, 1 − RMSE / persistence. At
  or below 0 the model adds nothing to a copy-forward.
- **Whiteness** of the one-step-ahead residuals, every test step predicted
  from the recorded history before it: the autocorrelation function and the
  cumulated periodogram (Bacher & Madsen 2011). A model that has a channel's
  dynamics leaves white noise; what it leaves instead is grouped by the
  period holding the most residual power. Under an hour is a missing fast
  state or lag, daily is a daily driver no input carries (occupancy,
  internal gains), the longest periods a slow drift. Continuous outputs
  only; staged ones have level metrics.
- **Physics checks**, when `synth_cfg` describes the thermal states. They
  test four things in synthetic states the data never contains:
  equilibrium, step response, reachability and free decay, each answered in
  physical units. A failed step response means not to plan against this
  surrogate. A small equilibrium drift after an honest attempt is worth
  noting and moving past. What the other outputs did in each scenario is in
  `results/readout_channels.json`.
- **Rollout**: channels whose rollout ramps (monotone, unlike the truth) or
  drifts more than twice the truth's own drift. A model that has stopped
  responding to its inputs emits a ramp that a short window's error hides.
- **Level metrics** for binary and ordinal outputs, scored as a classifier:
  balanced accuracy beside persistence's. Staged equipment is a rare-event
  channel, where a low error can mean the model never predicts the event.

Staged outputs get two things set from the data, both on by default: each
level's loss is weighted by how rare it is, and an ordinal head's decision
thresholds are moved after training until the predicted level shares match
the true ones. The kept thresholds are saved with the run.

Test error is measured on the excitation in your data, which is rarely
where a controller will drive the plant. The rollout and physics sections
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

`smooth` needs the `prep` extra. `nt data plot FILE` shows a record as
recorded, one channel at a time, gaps as breaks in the line; run from the
application directory, its channel list is grouped by `data_mapping`'s roles
(`--column C --output P.png` writes one channel to an image instead). Before
training, `nt data heads` reports
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
