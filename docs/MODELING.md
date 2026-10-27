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

The hooks in `modeling_user_defined.py`, scaffolded in
[template/](../template/) with a docstring each:

| Hook | Says | |
|---|---|---|
| `data_mapping` | which columns are `Y`, `U` and `D` | required |
| `data_head_config` | an output head per `Y` column: regression, binary, bounded or ordinal | required |
| `column_norm_config` | per-column normalization overrides | required |
| `synth_cfg` | which outputs are thermal states, for the physics checks (`None` skips them) | required |
| `sub_plot` | how output plots are grouped | required |
| `block_mapping` | blocks, a family per block, what each input column is, 2R2C priors | optional |
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
because its window is its warm-up. A `2r2c` block reads its own outputs at
one lag, which seeds its mass, and its `U` and `D` at the current step only.
The build refuses a table that misses a column, a lag no block actually reads
(a `U` or `D` column only 2r2c blocks read must be 0), and a `demand`
setpoint lagged further back than its state. The reportcard lists the lags
each block reads.

## Model families

| `model` | Kind | What it is |
|---|---|---|
| `mlp` | black box | one network over the flattened `[y, u, d]` window |
| `nssm` | black box | separable: `fy(y) + fu(u) + fd(d)`, each a network |
| `lssm` | black box | the same, linear |
| `gru` | black box | recurrent, with memory that outlives the input window |
| `2r2c` | grey box | a thermal network: zone air, a hidden mass, a learned heat input |

The black boxes often fit best and are fine for prediction you won't steer
by. Nothing bounds their long rollouts, so read the stability and physics
sections of the reportcard before planning against one.

### 2R2C: storage you can read

A black box that predicts the next temperature well can still be wrong
about the thing load shifting depends on: how long a temperature difference
survives. `2r2c` builds the storage in. Zone air and the building's mass are
two capacitances, joined to each other and to ambient by one resistance
each:

```
    ambient ──R_o──┬── zone air (C_z) ──R_m── mass (C_m)
                   │
               heat input  (learned, from the current u and d)
```

The state update is fixed physics, and only the heat input is learned. That
input reads `u` and `d`, never the state, so two things hold whatever the
network learns:

- **A rollout cannot diverge**, at any horizon and under any input,
  including the extreme inputs a policy will try.
- **Its time constants are readable in hours** on the reportcard, so you can
  check them against the building.

On one measured building, a 2 °C head start held with the plant off
survived about as long in `2r2c` as in the building (half-life 6.1 h
against 5.8 h), where the black boxes ranged from 1.2 h to 14.5 h. `2r2c`
paid for that in one-step error, which is the trade the family exists to
make.

**Its time constants are a prior you declare.** Each `2r2c` block states
three time constants in hours, in its `block_mapping` entry. A scalar pins
one; `[min, max]` lets training move it inside that box. There is no
default, and a block missing one is refused. They are boxed rather than
free because the slow mode cannot be identified from a training rollout:
the mass temperature is never measured, so every rollout seeds it from a
short window, and a free fit shortens the slow mode to absorb that wrong
seed. The reportcard prints where each one landed in its box, and says so
when one sits on an edge. `nt data decay` estimates the slow one from
free-float periods in your record, which is a reasonable first box.

### Blocks: one model, a family per group of outputs

A block is a group of `Y` columns with its own sub-model: its own
parameters, family and hidden state. Declaring blocks in `block_mapping` is
how one model holds, for example, a `2r2c` zone and an `mlp` power meter
that reads the zone's temperature:

```python
def block_mapping(nm_mapping, cfg):
    return {
        "zone":  {"Y": ["Zone Temperature"], "U": ["Cooling Setpoint"],
                  "model": "2r2c", "params": {...}},
        "power": {"Y": ["Total Power"], "reads": ["Zone Temperature"],
                  "model": "mlp"},
    }
```

- `Y` must partition exactly: every output in one block, none twice.
- A `U` column no block claims is shared by all of them, as one air handler
  serving several zones is. `D` has no partition rule.
- `reads` lets a block see other blocks' outputs from the history window,
  never their same-step predictions, so evaluation order never matters.
  `2r2c` refuses `reads`, since its stability guarantee depends on the
  learned part never seeing a state.

There is no conduction between blocks: they interact only through shared
inputs and `reads`.

### Declaring what a column is

A block can also say what its input columns are. The model then gets that
physics instead of having to learn it:

| Declaration | Acts in | Does |
|---|---|---|
| `demand` | every family | replaces a setpoint with the thermostat's clamped error against the block's own state, `relu(zone − setpoint)` for cooling, computed from the model's own prediction during a rollout. A setpoint above the zone is then free in the surrogate, as it is in the plant. |
| `linear` | `2r2c` | one sign-fixed coefficient times the physical value: delivered power, irradiance |
| `ambient` | `2r2c` | explicit conduction to that column at the block's ambient time constant |

Inside `2r2c`, a `demand` column becomes a saturating branch, bounded in
value and in slope, so the stability guarantee survives the one state read
it admits. Its `capacity`, how fast full stage moves the zone in °C per
hour, is declared like a time constant: a scalar pins it, `[min, max]` fits
it inside, and there is no default. When the equipment behind a setpoint
latches (a thermostat that starts a stage above the setpoint and, once
running, keeps it running until the zone is some way below), declare
`"hysteresis": {"on_at": [min, max], "off_at": [min, max]}` in °C of error:
the branch then carries its stage between steps and fits the two thresholds
inside those boxes, which may start at 0. The reportcard lists capacities
and thresholds with the time constants, and flags one pressed against an
edge.

```python
"Cooling Setpoint": {"kind": "demand", "sign": -1, "state": "Zone Temp",
                     "capacity": [lo, hi],                                # °C/h
                     "hysteresis": {"on_at": [lo, hi], "off_at": [lo, hi]}}   # °C
```

`nt data latch` estimates all three from the record. It prints how often the
equipment runs at the same zone-minus-setpoint after a step running and
after a step idle: equal columns mean a memoryless thermostat, columns that
split around zero mean a latch. Under the table it prints where each column
crosses one half, the error at which the stage starts and stops, and the
zone's rate running against idle at the same error, a capacity. These are
measurements of memory at a fixed measured error, which a deadband, a
minimum run time or a thermostat that reads another temperature than the
logged zone all produce, so use them to centre a box, not to pin a value.
Nothing is inferred from column names. A declaration changes the
physics but not the shapes, so a wrong one shows only in the reportcard's
branch lines.

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
which output head a column's distribution suggests, and `nt data decay
--state <column> --ref <ambient column>` fits a state's free-decay
envelope. Both are advice, not checks.

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
