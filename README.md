# NeuroTemplate

A library of generic, time-series, data-driven black-box modeling and control building blocks, layered on top of [Neuromancer](https://github.com/pnnl/neuromancer).

The repo contains two sister subsystems:

- **Modeling** — train a system-identification surrogate `y_{k+1} = f(y_k, u_k, d_k) [+ y_k]` from data, with an optional gradient-sign penalty, post-training physics assertions (equilibrium / step response / reachability / free decay), Optuna search, MLflow tracking. Model families: `mlp`, `nssm`, `lssm`, `gru` (black box) and `2r2c` (grey box — a thermal network with a hidden mass, for buildings whose storage matters). Families can be mixed in one graph, one per block of signals.
- **Control** — train a differentiable predictive control (DPC) policy against a frozen surrogate. Closed-loop rollouts, a declared objective over named columns (comfort bands, tracked schedules, energy, cost and peak-demand terms), reference signals read from their own files. Policy families: `mlp`, `linear` (a saturated affine feedback law). Optionally anchors chosen U columns to a rule, each inside its own box.

Both sides share a thin `common/` layer (YAML/CLI loader, CSV + sliding-window data utilities, per-channel metrics).

NeuroTemplate is meant to be installed as a dependency in an application repo. You bring the data and the two user-defined hook files; the library brings everything else. New framework features land here once, get tagged, and downstream apps pick them up via a version bump.

Defects are tracked in [audit.md](audit.md), unfinished work in [issues.md](issues.md), and design direction in [generalization_report.md](generalization_report.md).

## Installing as a dependency

Add NeuroTemplate to an application's `pyproject.toml`:

```toml
[project]
dependencies = [
    "NeuroTemplate @ git+ssh://git@github.com/<owner>/sys_id_framework.git@v0.1.0",
]

# Until upstream Neuromancer ships a macOS arm64 wheel for torch-harmonics,
# every downstream app needs this same override. The framework's own
# pyproject has the same block, but uv overrides don't propagate to consumers.
[tool.uv]
override-dependencies = [
    "torch-harmonics; sys_platform != 'darwin'",
]
```

For local development against an in-progress framework checkout, swap the dep source to an editable local install:

```toml
[tool.uv.sources]
NeuroTemplate = { path = "../sys_id_framework", editable = true }
```

Then `uv sync` and a single `nt` console script is on your `PATH`. It dispatches
`nt <group> <verb>` to the subcommands below.

## What lands on PATH

| Command | Purpose |
|---|---|
| `nt model train` | Train a system-ID surrogate |
| `nt model optuna` | Optuna hyperparameter search + retrain |
| `nt model load` | Re-evaluate a saved MLflow run on new data |
| `nt model debug-heads` | Diagnostic for output-head behaviour |
| `nt data heads` | Which head does this Y column want? (pre-training advice) |
| `nt data decay` | What is this column's envelope time constant? (pre-training advice) |
| `nt control train` | Train a DPC policy against a frozen surrogate |
| `nt export model` | Ship a trained surrogate as ONNX graphs + a torch-free runtime |
| `nt export policy` | Ship a trained DPC policy the same way (same package, if you like) |

`nt` with no arguments (or `nt -h`) prints the command list. The `train`
commands take the same standard flags (parsed by [common/config.py](src/neurotemplate/common/config.py)):

```
--config PATH         YAML config path or short name resolved against the command's config directory
--set key=value       Override any YAML field (repeatable)
--print-config        Render the resolved config to stdout and exit
--dump-config PATH    Write resolved config to YAML and exit
--list-configs        List available configs in the config directory and exit
```

## What an application repo looks like

```
my_app/
├── pyproject.toml              # depends on NeuroTemplate + the torch-harmonics override
├── modeling_user_defined.py    # the modeling hooks
├── control_user_defined.py     # the control hooks (only if you train policies)
├── model_configs/
│   └── modeling.yaml
├── control_configs/
│   └── control.yaml
└── data/                       # whatever your YAML paths point at
```

`nt model train` looks for configs in `model_configs/`; `nt control train` looks in `control_configs/`. Keep them separate so each folder only contains the configs relevant to that subsystem.

The library finds the hook files automatically: by default it looks for `modeling_user_defined.py` / `control_user_defined.py` in the current working directory. If you'd rather keep them elsewhere, set `user_hooks_path` in your YAML (a file path or dotted module name). A hooks file named by path is imported once per run under its own file name, and can import the modules beside it — including the canonical `modeling_user_defined`, so a variant hooks file can reuse most of the main one.

[template/](template/) *is* that tree, ready to copy: `cp -r template/. my_app/`. Both hook files carry every hook with the right signature — dataset-specific ones stubbed with a comment on what belongs there, the rest with working defaults — and both config directories carry every field annotated with its default, plus a `debug.yaml` smoke run per side. See [template/README.md](template/README.md) for the fill-in order.

## Hook contracts

### Modeling (`modeling_user_defined.py`)

| Function | Required | Used by |
|---|---|---|
| `data_mapping(df, cfg) -> {Y, U, D, t: [col]}` | yes | pipeline at data-prep time |
| `data_head_config(nm_mapping, norm_stats) -> HeadStack` | must be defined; return `HeadStack([RegressionHead() for _ in Y], Y)` for plain regression | output-head construction |
| `column_norm_config(nm_mapping) -> {col: norm_type \| None}` | must be defined; return `{}` for no overrides | per-column norm override |
| `synth_cfg(nm_mapping, norm_stats) -> SynthSpec \| None` | must be defined; return `None` to disable | physics checks |
| `sub_plot(df_true, col_name) -> tuple \| None` | must be defined; return `None` for default | per-column subplot guidance |
| `hyper_args(trial, cfg) -> Config` | required only for `nt model optuna` | Optuna trial setup |
| `block_mapping(nm_mapping, cfg) -> {block: {Y, U, D, reads, model, inputs, ambient}} \| None` | optional | split Y into independent sub-models, each with its own family, and declare what each input column is (see [Blocks](#blocks)) |

The library imports each hook by attribute (no defaulting), so every row above except `hyper_args` and `block_mapping` must exist in `modeling_user_defined.py`, even when the body just returns a trivial value. `hyper_args` is only looked up on the code path that needs it; `block_mapping` is `hasattr`-gated and skipped when absent.

### Control (`control_user_defined.py`)

| Function | Required | Used by |
|---|---|---|
| `actuation_bounds(nm_mapping, norm_stats) -> (umin, umax)` | yes | policy bounds |
| `power_channels(nm_mapping) -> [col]` | must be defined; return `[]` when nothing is metered | closed-loop plot power row |
| `plot_context_channels(nm_mapping) -> [col]` | must be defined; return `[]` for none | D traces overlaid on the closed-loop plot |
| `power_metric(yn, U, D, nm_mapping, norm_stats) -> Tensor` + `POWER_METRIC_IMPLEMENTED: bool` | required only when a loss term names `power` | power terms (energy, cost, peak) + energy reportcard |
| `reference_control(y_now, r_now, d_now, nm_mapping, norm_stats) -> dict[str, Tensor]` | required with `residual_delta` | the residual policy's anchor, one command per anchored U column |
| `preview_config(nm_mapping) -> {D column: n_steps}` | optional | per-column disturbance preview (a reference column's preview is `reference_preview` in the config) |

The objective is not a hook: it is the `losses` table in `control.yaml`, below.

`power_metric` returns whatever units the use case meters; the library never converts, so the reportcard labels energy as `power·h`. Metered channels the surrogate already predicts are the best thing to return, because the energy numbers then mean something outside the run. An actuation proxy such as `(setpoint - ambient)²` is a fine alternative when nothing meters power — just read the energy section as relative rather than physical. What has actually cost campaigns here is the term's *scale*, not its shape: a proxy carried over from another dataset came out at 1.7× the entire comfort objective while every pooled metric looked fine. The objective-balance report and the shape sweep both print before training so that magnitude is on screen from the start.

`reference_control` is memoryless and works on the current step: `y_now` `[b, ny]` and `d_now` `[b, nd]` normalized, and `r_now`, each reference column the objective names by name, `[b]`, normalized — `norm_stats` carries an entry for each, so a rule denormalizes a band edge exactly as it does a zone temperature. It returns `{U column: [b]}`, normalized, for exactly the columns `residual_delta` names (anything else is refused, naming both sets). Each of those columns commands `reference_control(...)[column] + δ` with `δ` inside that column's box instead of a setpoint outright: the rule keeps comfort, δ answers whatever else is being optimised, and the box is a trust region on surrogate error rather than a claim about the network. The reportcard reports each box and the fraction of steps δ spent pinned at each edge — an edge that binds most of the time means either the rule is wrong for this plant or the optimizer is riding a surrogate artifact into the wall. Note a box bounds the *action*, not the outcome: the commanded setpoint stays within its box around the rule's, the zone temperature does not, because the plant is in between. The library does not score the rule against the learned policy; to score the rule itself, train with every box `[0, 0]`.

### The control objective: a declared table over named columns

```yaml
losses:
  comfort_lower: {column: "Comfort Low [°C]",  target: "Zone Air Temperature [°C]", map: relu,  norm: 2}
  comfort_upper: {column: "Comfort High [°C]", target: "Zone Air Temperature [°C]", map: nrelu, norm: 2}
  du:            {column: "Zone Cooling Setpoint [°C]", map: first_difference, norm: 1}
  cost:          {column: power, norm: 1, weight: "Tariff [$/kWh]"}
gains: {comfort_lower: 5.0, comfort_upper: 5.0, du: 1.0, cost: 1.0e-3}
```

Each entry is read in one order: the operand `column − target` (`target` optional), then the `map`, then the `norm`, then `gains[name]`. Nothing else is in the objective.

| axis | values | notes |
|---|---|---|
| operand | `column`, optionally `− target` | `column` names a Y, U or D column of the surrogate, a reference column, or `power`; `target` a Y column or `power` |
| `map` | *(none)*, `relu`, `nrelu`, `first_difference` | `relu` on `column − target` is a lower bound, `nrelu` an upper one, so the direction is in the map, not the name |
| `norm` | `q ≥ 1` | the power mean `(mean_t w·|h|^q)^(1/q)` per window, averaged over the batch: 1 = mean, 2 = RMS, larger approaches the max |
| `weight` | a reference or D column | per-step weighting in physical units; a tariff is the `weight` of a power term, not a kind of reference |
| `units` | `physical` (default), `normalized` | `physical` denormalises `column` and `target` before the map, so a gain is per °C, per W; `power` is Watts either way |

Every reduction is horizon-free and homogeneous of degree 1, so a gain tuned at `train_pred_steps: 48` means the same at 288, and switching a power metric from kW to W moves every power gain by the same factor. Two actuators that want different smoothing are two `first_difference` rows, each named on the reportcard. The terms appear in MLflow and on the objective-balance report as `<name>_loss`.

**The regulated set is derived**: the Y targets of the `relu`/`nrelu` rows, each with at most one lower and one upper bound — a band, a one-sided limit, or (the same column on both sides) a tracked setpoint schedule. The reportcard scores comfort and band-jump response against exactly those bands.

**Reference columns are data, in their own files.** A name that is not a surrogate column or `power` is a reference column, read from `train_reference_path` / `dev_reference_path` / `test_reference_path` — one list per split, parallel to its plant files, each file row-aligned with its plant file and carrying exactly its timestamps (a mismatch is refused, naming both ranges). They reach the policy and the loss as one `R` key and never the surrogate, whose Node reads only `yn`, `U` and `D`. Several draws are several files: list the plant file once per draw. [template/make_references.py](template/make_references.py) samples a band and slices a tariff into such a file; it is example code to edit, not library behaviour.

A reference column bounding a Y column is normalised with that column's stats; any other (a tariff, a cap on power) needs a `reference_scale` entry — a divisor, no shift, so a zero price stays a zero cost. `reference_preview` gives each reference column its own preview depth: a cost term needs one covering the price cycle before it can *shift* load rather than react to it. The policy's reference input is ordered by sorted column name, so reordering the YAML never moves an input.

**What the library does not check about your references.** How a band or a tariff was prepared is the data's business. Two things are worth checking yourself:

- **Draw the tariff from a real historical record, never from the clock.** A tariff that is a deterministic function of hour-of-day is one the policy can learn *as* hour-of-day — D is strongly clock-correlated, so "read the price" and "read the time" collapse into the same task and you get a convincing load-shifting trace with no price responsiveness in it. Real day-to-day variation is what forces the policy to read the column.
- **Confirm a price response before believing it.** A reference file aligns the tariff with the weather row for row, so the price↔weather correlation is a second route to the same shortcut. `make_references.py` slices the tariff at a random offset by default, putting its peak at a random hour of the plant's day; `--price-align day` keeps the time of day, which is what makes a pre-cool visible in a plot and partly re-opens the shortcut. Either way, re-score the trained policy against a test reference with the peak moved to a different hour (same weather, same surrogate) and check the load moves with it.

The tables above are the canonical hook contract. To start writing hooks, copy [template/](template/) — it has all of them stubbed with the right signatures. For worked-through implementations (including an ordinal/bounded `data_head_config`), read [tests/fixtures/modeling_user_defined.py](tests/fixtures/modeling_user_defined.py) and [tests/fixtures/control_user_defined.py](tests/fixtures/control_user_defined.py).

## Typical session

```bash
cd my_app/

# 1. Train a system-ID surrogate. (--config defaults to model_configs/modeling.yaml.)
nt model train

# Watch for `Finished run: <run_id>` — that's the MLflow id of the surrogate.

# 2. Train a control policy against it.
nt control train --config control_configs/control.yaml \
    --set sysid_run_id=<run_id>

# 3. Re-evaluate the surrogate on different data later.
nt model load --run_id <run_id> --data path/to/test.csv

# 4. Ship them: one package, no torch needed to run it.
nt export model  --run_id <run_id> --data data/test.csv --out delivery/site --package site
nt export policy --run_id <control_run_id> --out delivery/site --package site

# Hyperparameter search instead of a single train:
nt model optuna --config model_configs/modeling.yaml \
    --set n_trials=50 --set trial_epochs=30
```

One-off overrides without editing YAML:

```bash
nt model train --config model_configs/modeling.yaml \
    --set epochs=1 \
    --set log2_hidden_size=4 \
    --set grad_gain=0.5
```

The `--set` parser handles `true`/`false`, `null`/`none`, quoted strings, JSON lists/dicts, floats (anything with `.`/`e`/`E`), and ints. Unknown keys fail fast with `ConfigError: Override target does not exist: <key>`.

Inspect a fully-resolved config without running anything:

```bash
nt model train --config model_configs/modeling.yaml \
    --set epochs=10 --print-config
```

## Model families

| `model` | Kind | What it is |
|---|---|---|
| `mlp` | black box | one MLP over the flattened `[y \| u \| d]` window |
| `nssm` | black box | separable: `fy(y) + fu(u) + fd(d)`, each an MLP |
| `lssm` | black box | the same, linear |
| `gru` | black box | stateful — its memory outlives `in_steps` |
| `2r2c` | grey box | a thermal network: zone air, a hidden mass, and a learned heat input |

### 2R2C: when the storage is what you need

The black-box families predict the next zone temperature from a window of
recent observations. That is enough to be accurate one step ahead, and it is
why they can still be wrong about the thing load shifting depends on: **how
long a temperature difference survives.** A model scored over a four-hour
rollout can re-derive the zone from the last few setpoints and never learn
to integrate. Coolth stored at midday is then gone by the evening peak, and
pre-cooling cannot pay inside the model whatever the real building does.

`2r2c` builds the storage in instead of hoping for it. Zone air is one
capacitance, the building's mass — walls, slab, furniture — is a second,
hidden one, and heat moves between zone and ambient and between zone and
mass through one resistance each:

```
    ambient ──R_o──┬── zone air (C_z) ──R_m── mass (C_m)
                   │
                heat input
```

The state update is fixed physics; only the heat input `g` is learned, by an
MLP over the current `u` and `d`. We have no plug-load channel and no
zone-level heat metering, so `g` is where all of that goes — along with the
solar gain and the conduction to outdoors, which is why nothing has to tell
the block which `D` column is the weather.

Because `g` reads inputs and never the state, two things hold whatever the
MLP learns:

- **A rollout cannot diverge**, at any horizon, under any input — including
  the adversarial inputs a DPC policy will go looking for.
- **The two time constants are readable in hours** on the reportcard, so you
  can check them against the building rather than taking the surrogate's word.

Measured on `data/excited` in the RTU-mall app, where the envelope decays at
6.4–10.9 h — a 5.8 h half-life at the mid-range estimate — this is how long a
2 °C head start survives with the compressor held off:

| | building | `2r2c` | `nssm` | `mlp` | `gru` | `lssm` |
|---|---|---|---|---|---|---|
| half-life | 5.80 h | **6.08 h** | 3.92 h | 8.00 h | 14.50 h | 1.17 h |
| left after 12 h | 23.8% | **32.0%** | 4.0% | 7.4% | 66.3% | ~0% |

`2r2c` is the closest of any family, at a third of `nssm`'s parameters. It
pays for that in test RMSE (0.54 against 0.41–0.44), which is the trade the
family exists to make: rollout RMSE cannot see the storage, and load
shifting cannot live without it.

**The time constants are a prior, held in a box.** Each `2r2c` block declares
three, in hours, in its `block_mapping` entry's `params` — a scalar pins one,
`[min, max]` lets training move it inside that box:

```python
"zone": {"Y": ["Zone Air Temperature [°C]"], "model": "2r2c",
         "params": {"tau_fast_h": [0.3, 1.0],      # the air settling after an HVAC step
                    "tau_ambient_h": [2.0, 8.0],   # the initial slope of an HVAC-off drift
                    "tau_slow_h": [20.0, 60.0]}},  # that drift after its first hour
```

These are the two modes and the ambient rate — what one measures, so where
priors are actually held — rather than the network's component rates.
`tau_slow` is what `nt data decay` fits, and its range is the natural
`tau_slow_h` box. The boxes must be disjoint in the order fast < ambient <
slow; every point inside them is then a valid building, so training cannot
leave the feasible set, and `tau_fast` may not start below two timesteps of
the data, the stability guarantee's floor. There is no default: a block
missing one is refused, and so is `model: 2r2c` with no `block_mapping` hook,
because there is nowhere to declare them.

A box, not a free fit, because the slow mode is not identifiable from a
training rollout. The mass temperature is unobserved, so every rollout has to
seed it from an `in_steps` window an hour long at most, against a state that
lags by many hours. Left free, the optimizer discovers that shortening the
slow mode is a cheaper way to absorb that wrong seed than fixing the forcing —
on synthetic data from a known 31.6 h building, seeded truthfully training
moves `tau_slow` 13.4 → 16.5 h, and seeded the way a real rollout must, the
same training drives it to 8.7 h. A box stops that rather than curing it: the
seed still pulls `tau_slow`, now to its lower bound, and the reportcard says
so — *"tau_slow at 97% of the way to its lower bound — the data (or the seed)
disagrees with the prior"* — per parameter, with the seed ensemble showing
whether every seed does it.

The reportcard prints the modes per block and per channel, each bounded
parameter's box and where its fit landed, and the block's family — the
interesting fact once families are mixed:

```
- **fy:** `BlockStack`, params=3,493, in=60, out=2
  - **block `zone` (2r2c):** `RCBlock`, params=1,223, in=48, out=1
    - Zone Air Temperature [°C]: tau_fast=0.55 h, tau_slow=20.66 h
    - tau_fast (Zone Air Temperature [°C]): 0.548 in [0.3, 1], 50% of the way up
    - tau_ambient: pinned at 2
    - tau_slow (Zone Air Temperature [°C]): 20.7 in [20, 60], 3% of the way up — ⚠️ tau_slow at 97% of the way to its lower bound — the data (or the seed) disagrees with the prior
  - **block `power` (mlp):** `MLP`, params=2,273, in=36, out=1
```

A block that declares its columns (below) prints one line per branch under
its time constants, each in physical units, plus the derived inputs the
pre-block computed:

```
  - **block `zone` (2r2c):** `RCBlock`, params=486, in=48, out=1
    - Zone Air Temperature [°C]: tau_fast=0.78 h, tau_slow=104.00 h
    - ambient: Ambient Temperature [°C]
    - linear: GHI [W/m2], k=0.000105 °C/step per unit
    - saturating: Cooling Setpoint [°C], cap=0.77 °C/step, w=8.6 °C, slope_share=0.10
    - mlp: Occupancy

Derived inputs (replace the raw column before every family):
- Cooling Setpoint [°C]: demand of Zone Air Temperature [°C], sign=-1, scale=15.2
```

What `g` cannot do on its own is see the state, so an undeclared block
cannot represent a thermostat deadband: "the compressor is off whenever the
setpoint is above the zone" is a function of the *difference*, and the block
renders it as a monotone response to setpoint with no cut-off. Declaring the
setpoint as `demand` (below) is the answer — the difference is computed
outside `g` and handed to a bounded branch that keeps the stability
guarantee.

### Blocks

A *block* is a group of Y channels that gets its own sub-model: its own
parameters, its own **architecture**, and its own hidden state. Declare them
with the optional `block_mapping` hook:

```python
def block_mapping(nm_mapping, cfg):
    return {
        "zone":  {"Y": ["Zone Air Temperature [°C]"],
                  "U": ["Cooling Setpoint [°C]"],
                  "model": "2r2c"},
        "power": {"Y": ["Total Power [W]"],
                  "reads": ["Zone Air Temperature [°C]"],
                  "D": ["Ambient Temperature [°C]"],
                  "model": "mlp"},
    }
```

That is the point of the split: **one graph, different families per signal
group.** Zone air as a grey-box thermal network whose storage you can read
off the reportcard, power as an MLP that sees the zone it is cooling.

What a block owns:

| | |
|---|---|
| `Y` | its own channels, and only its own. **Y must partition exactly** — a state claimed twice, left out, or naming a column not in the mapping is reported before the first epoch. |
| `U` | its own actuation, plus any U channel **no block claims** — unclaimed controls are shared with every block, which is what you want for one air handler serving several. |
| `D` | all of it by default, or a named subset. D is exogenous, so it has **no partition rule**: two blocks may claim the same column. |
| `reads` | other blocks' Y channels this block may also see, as ordinary inputs. |
| `model` | the family this block builds under. Omit it to fall back to `cfg.model`. |

Cross-block `reads` are safe because every block input comes from the shared
**history window**, never from another block's same-step output — so there is
no evaluation-order dependency between blocks and the gather stays a fixed
permutation. **`2r2c` refuses `reads`**, and that refusal is load-bearing: its
unconditional-stability guarantee holds only because its learned forcing map
sees `u` and `d` and never state. (It still reads its own Y, in the fixed
linear term — what it cannot do is let a *learned* map see state.)

`2r2c` is per-block whether or not the hook exists: with no hook, each Y
channel still gets its own RC network driven by every U channel. Declaring
blocks is how you say *which setpoint drives which block*. For `mlp` and `gru`
a single-family split is opt-in with `per_block: true`; a block naming its
own `model` needs no flag.

### Declaring what a column is

A block may also say what each of its U and D columns *is*, and each family
decides what that buys it:

```python
"zone": {
    "Y": ["Zone Air Temperature [°C]"],
    "U": ["Cooling Setpoint [°C]", "Fan Power [kW]"],
    "D": ["Ambient Temperature [°C]", "GHI [W/m2]", "Occupancy"],
    "model": "2r2c",
    "ambient": "Ambient Temperature [°C]",
    "inputs": {
        "Cooling Setpoint [°C]": {"kind": "demand", "sign": -1,
                                  "state": "Zone Air Temperature [°C]"},
        "Fan Power [kW]": {"kind": "linear", "sign": +1},
        "GHI [W/m2]":     {"kind": "linear", "sign": +1},
        # Occupancy is undeclared: it feeds the learned residual.
    },
}
```

The three declarations, and what each family does with them (`ambient`
is the block-level key shown above, not an `inputs` kind):

| declaration | who acts on it | what it does |
|---|---|---|
| `demand` | every family | replaces the setpoint with the thermostat's clamped error against the block's own state, `relu(zone − setpoint)` for cooling (`sign = -1`) and the mirror for heating, computed in physical units from the model's *own* predicted zone during a rollout. Raising the setpoint above the zone is then free on the surrogate the way it is on the plant. Pairing with another block's state is refused — that is `reads`. |
| `linear` | `2r2c` | one sign-fixed coefficient times the physical value: measured power removes heat, irradiance adds it. |
| `ambient` | `2r2c` | the explicit conduction `a_o (T_ambient − zone)` at the block's `tau_ambient_h` rate, and the column leaves the residual MLP's input. Give `tau_ambient_h` a box wide enough to hold the truth: with the column declared, a wrong rate would otherwise be cancelled by a fictitious constant heat gain. |

Inside a `2r2c` block a `demand` column becomes a *saturating* branch,
`cap · tanh(x / w)` — a staged coil's response, bounded in value and in
slope. The parameterization keeps that slope inside the stability headroom
`1 − a_m − a_o`, so the unconditional-stability guarantee survives the one
state read it admits. Add `"branch": "mlp"` to the entry to feed the demand
feature into the learned residual instead; that gives up the bound, and the
reportcard says so. The reportcard prints each branch beside the time
constants (`cap` in °C/step, `w` in °C of demand, and the share of the
headroom in use). Other families ignore `linear` and `ambient` — their
networks already see every column — and a warning says so at build time.
The declaration is persisted with the run in `block_spec.json`.

Nothing is inferred from column names: a dataset whose actuator is metered
power rather than a setpoint declares it `linear` (or leaves it undeclared)
and writes no `demand` entry at all.

A `demand` declaration also makes the gradient-sign penalty usable on that
column for the first time. The Jacobian is taken downstream of the
pre-block, so `grad_path` entries for a declared column constrain
`∂state/∂demand` — whose sign is the declared `sign` — rather than
`∂state/∂setpoint`, which is zero on the inactive side and has no universal
sign anywhere.

**The physical scalars train at their own learning rate**, `rc_branch_lr`
(default 0.05). These have units of travel to cover — an ambient time
constant moving 2 h to 20 h, a coil capacity crossing an order of magnitude —
where a network weight moves in thousandths, and Adam steps a raw parameter
by about `lr` per update. At `lr` they stall within a few percent of where
they started, which reads on the reportcard as a branch that "learned" its
initial value. Every 2r2c block's bounded time constants, `k`, `cap` and `w`
are in that group, and every time constant moves in log-space inside its box,
so hours-to-days is a few units of raw travel rather than hundreds.

The one-step test NRMSE is not expected to improve with a declaration; the
closed-loop and physics readouts are what it is for.

Inter-block conduction is not modeled. Blocks are independent by construction
apart from `reads`, and a plant whose groups exchange meaningful energy
through their *states* is outside what this buys you.

## Two optional control capabilities

Both are off until asked for and both fail fast rather than degrade, so a run
that does not ask for them is unchanged.

### Grid response: cost and peak-demand objectives

Energy, cost and peak are three rows over one signal — `power`, the
`power_metric` hook's output — reduced three ways: a mean, a tariff-weighted
mean, and a power mean with `q > 1`. All three read the same hook on the same
steps, so they cannot end up pricing different quantities.

To turn on price-driven control:

1. **Implement `power_metric` and set `POWER_METRIC_IMPLEMENTED = True`.**
   Nothing here works without it.
2. **Get a real price series** (timestamp + one price column in $/kWh).
   Sample: `grid_response.ipynb` in Neuromancer's
   `examples/domain_examples/` downloads PG&E Hourly Flex Pricing from
   `https://raw.githubusercontent.com/EladMichael/small_example_datasets/main/hourly-flex-prices-historical-sample.xlsx`.
3. **Write it into the reference files** with
   `make_references.py --price price.csv` — it resamples a coarser price
   record onto the plant's timestep and re-stamps it with the plant's
   timestamps. *Do not build a time-of-use table keyed on hour-of-day* — see
   the objective section above for why the policy will learn the clock
   instead of the price.
4. **Declare the terms, the tariff's scale and its preview** — pre-cooling
   happens *before* the expensive block, so a policy that only sees the
   current price can react but never shift:

   ```yaml
   losses:
     # ...the comfort rows...
     energy: {column: power, norm: 1}
     cost:   {column: power, norm: 1, weight: "Tariff [$/kWh]"}
     peak:   {column: power, norm: 4}      # 1 = mean, 2 = RMS, larger -> hard max
   gains: {comfort_lower: 5.0, comfort_upper: 5.0, energy: 1.0e-4, cost: 1.0e-3, peak: 1.0e-4}
   reference_scale: {"Tariff [$/kWh]": 0.30}
   reference_preview: {"Tariff [$/kWh]": 48}   # 48 steps = 4 h at 5-minute data
   train_pred_steps: 288                       # 48 is 4 hours: too short to precool and coast
   ```

`train_pred_steps` is the real cost — rollout memory grows roughly linearly.

#### If what you want is pre-cooling

Pre-cooling is load shifting you can point at: drive the zone down through the
cheap block, then coast up through the expensive one on the building's own
thermal mass. Four things have to be true at once or it will not appear, and
three of them are not the gains:

- **`reference_preview` covers the run-up to the peak.** Pre-cooling is by
  definition an action taken before the expensive block. A policy whose only
  price input is the current one can be cheap, but it cannot pre-cool.
- **The comfort band is wide enough to store anything.** Load shifting *is*
  thermal storage, and a 0.5 °C band leaves nowhere to put it — reporting "no
  price response" out of a narrow band is measuring the band. 1.5–2.5 °C is a
  reasonable place to start.
- **The rollout spans the cycle.** 288 steps (24 h at 5-minute data) covers a
  precool and the coast that pays for it; the default 48 covers neither.
- **The cost term carries the power side, not an energy term.** They pull
  different ways: energy rewards using less power at *every* hour, so the
  cheapest way to satisfy it is to stop cooling — load *shedding*, which shows
  up as a big power cut with comfort going backwards. Only cost rewards moving
  power to the cheap hours. If pre-cooling is the goal, start with cost alone
  and a comfort gain high enough that the savings have to be found rather than
  bought.

Write the references with `--price-align day` while you are looking for it, or
the tariff's evening peak lands at a different hour in every window and the
precool averages out of every plot. Then confirm it is the price and not the
clock, as above. Pre-heating against an evening peak is the same recipe with
the sign flipped.

### Residual control: bound the policy to a rule

An anchored U column commands `reference_control(...)[column] + δ`, with
`-below ≤ δ ≤ above`, instead of commanding a setpoint. This is a trust region
on surrogate error, not a claim about what networks find easy to learn:
training's gradient *and* its model-selection criterion both come from a
surrogate whose held-out score is not the plant's, and a box caps what that
error can cost you.

1. **Implement `reference_control`** (see the template; its `bracket_band`
   does the hard part for a heating/cooling pair). It returns one command per
   anchored column.
2. **Set `residual_delta`**, a box per anchored U column, in that column's
   physical units:

   ```yaml
   residual_delta:
     Heating Setpoint [°C]: [0.0, 0.5]   # up off the band floor, never down
     Cooling Setpoint [°C]: [0.5, 0.0]   # down off the ceiling, never up
   ```

   A heating/cooling pair is held inward in opposite directions, which is why
   the box is per column. A U column not named is a free policy output over
   its actuation bounds. Training starts exactly at the rule, and a zero side
   does not trap it there: the box clamp is exact going forward and the
   identity going backward, so the gradient keeps the net's own slope.

Nothing is rounded to an actuator's resolution — a BMS with half-degree
setpoints implements that, and the controller commands through it. Smoothness
is the `first_difference` loss term's job.

### Together

This is the case the residual design is really for — the rule keeps comfort,
δ answers the incentive, and Δ is an operator-legible knob ("how far from the
comfort rule am I willing to move for money"): a `residual_delta` box of a
degree or so on each setpoint, beside the cost and peak rows above, with `train_pred_steps: 288`.

## Diagnostics that run whether you ask or not

Fit metrics and pooled comfort rates cannot see the failures that cost the
most, because those failures are silent: nothing crashes and the headline
number looks reasonable. Every run therefore also produces these, in the
reportcard and in MLflow.

**After `nt model train`:**

- **Rollout stability** — rise fraction and drift against the ground truth's
  own, which catches a model that has stopped responding to its inputs and is
  emitting a ramp. RMSE on a short window will not.
- **Physics checks** — four assertions per state channel, on synthetic states
  the recorded data never contains, each answered in physical units. The states
  must hold at **equilibrium**, move the declared way under a **step response**,
  be drivable across a usable span (**reachability**), and relax toward ambient
  in **free decay**. *"`Zone Air Temperature` closed 0.4% of an 8.3 °C gap to
  ambient in 24 steps"* is a reason to reject a surrogate; a test-split RMSE
  cannot produce one. Reachability also answers "can any policy hold the band I
  am about to configure" before a controller is built. Needs
  `modeling_user_defined.synth_cfg` to return a `SynthSpec`; skipped when it
  returns `None`.
- **Readout channels** — order statistics for every Y channel that is *not* a
  declared thermal state. No verdict: what a readout should do is
  scenario-dependent. But a readout emitting 4000 W at equilibrium feeds back
  into the state predictions that *are* asserted on, so the spread is worth
  seeing.
- **Level metrics** — every binary / ordinal channel scored as a classifier:
  balanced accuracy, per-level recall, and predicted-vs-true share of each
  level. Staged equipment is usually a rare-event channel, and NRMSE is
  anti-correlated with catching the event — the best-NRMSE furnace surrogate
  in the head bake-off never fired once. A constant scores 1/K here, not
  near the top.
- **Persistence reference** — the same regression and level metrics for
  "hold the last seeded state for the whole window", beneath the model's.
  A regression head on a staged channel learns exactly that and then
  collapses in the rollout; this row is what makes the collapse readable.

Two things are set from the data for staged channels rather than left to
the optimiser, both on by default (`balance_levels`, `calibrate_thresholds`):
each levelled head's loss is weighted by inverse square-root training
frequency, and after training each ordinal head's decision thresholds are
share-matched — moved, over a handful of dev rollouts, to where the
predicted level shares equal the true ones. The reportcard's "Threshold
calibration" section shows the full search, not just where it landed; the
kept thresholds are saved in `head_config.json`, so `nt model load` decides
the way the run was evaluated. `nt data heads` reports where a channel's
mass sits and flags a level under 5%.

**Before `nt control train` trains anything:**

- **Objective balance** — every term's value and share of the objective on one
  real batch, warning when one term outweighs all the others combined. This is
  the one that asserts something: a term carried over from another dataset once
  came out at 1.7× the whole comfort objective.
- **Power term shape** — the power signal swept across each actuation box, so
  its magnitude and shape sit next to the gains you are choosing. One sweep
  serves every term on `power`, since each is a reduction of the same signal. Reported without a verdict: an honest power metric is
  monotone in actuator effort, and the interior optimum belongs to the summed
  objective, not to this term alone.
- **Power clamped to zero** — the share of scored steps where the power signal
  sits at exactly zero and so contributes no gradient. A rounding detail for a
  mean energy term; not one for a cost term, because the near-zero steps are
  disproportionately the cheap hours load is supposed to move *into*. A large
  figure is a surrogate problem, not a weight.
- **Objective notes** — configurations that are legal but cannot do what they
  were switched on for: a term weighted by a reference column the policy does
  not preview (reactive, not shifting), and power terms that are all peaks
  (minimisable by *adding* off-peak load).

**After the closed-loop evaluation:**

- **Actuation** — per-control mean |Δu| and the fraction of the evaluation
  spent against each bound. Reported, not judged: a setpoint parked at its
  least-aggressive limit while nothing is demanded of it looks identical to one
  that is stuck. The band-jump audit below is what tells them apart.
- **Band-jump response** — every step change in a band that left a
  regulated channel out of band, scored on whether any control moved, whether
  the channel followed, and whether it got back in band. Two controllers
  measured at 16.6% and 17.9% pooled violation reacted to 0% and 75% of their
  cooling-demand jumps respectively; the pooled metric ranked them backwards.
- **Residual saturation** — for a residual policy, each anchored column's box
  and the fraction of steps δ spent pinned at each edge, measured *after* the
  actuator clamp so it is the deviation actually deployed. A cap that binds
  most of the time means either the rule is wrong for this plant or the
  optimizer is riding a surrogate artifact into the wall. Read it before the
  comfort number.

## Shipping a model: ONNX export

`nt export model` and `nt export policy` serialise the *trained* graph —
they never reimplement it — into a package that needs only `numpy` and
`onnxruntime`, or, from C++, only `onnxruntime`:

```
delivery/site/
  site/runtime.py        Model (warmup / step / rollout / predict), Policy (act)
  cpp/                   the same runtime in C++17 (runtime.hpp), validate.cpp,
                         example.cpp, CMakeLists.txt
  model/                 warm_start.onnx, step.onnx, model.json, reference/
  policy/                policy.onnx, policy.json, reference/
  README.md  pyproject.toml  validate.py  example.py
```

Normalisation, window layout and the hidden state all live inside the
graphs, so the signature is physical units in and physical units out. Every
family, head, block split and prestack exports. Before anything is written,
the export rolls the trained system and the package out over the whole
held-out split (for a policy: the trained closed loop over its test split)
and refuses to write unless they agree. The shipped `reference/` fixtures come
from the trained torch model, so `python validate.py` on any machine checks
the whole chain. `validate.py` and `example.py` are written once and then
belong to the project; re-exports never overwrite them. Linux (aarch64,
x86-64) and macOS arm64 have been verified; other platforms have not.

A model whose own rollout blows up on `--data` (non-finite, or more than
100 stretches from its columns' means) cannot be verified: float round-off
grows with it. The exporter then asks whether to export anyway, and with no
terminal to ask on the answer is no. On yes, `model.json` says
`verified: false`, no `model/reference/` ships, `Model(...)` warns on load,
and both `validate`s report the model as unverified instead of checking it.
A model that diverges on held-out data is the thing to be worried about, so
the export does not loosen its check to fit one.

From C++, `cmake -S cpp -B build && cmake --build build && build/validate`
in the package directory checks the same fixtures through `cpp/runtime.hpp`.
Configuring downloads a pinned onnxruntime release unless
`-DONNXRUNTIME_ROOT` names one. The C++ runtime is header-only and
bit-identical to the Python one. The package's README shows the API.

A new stateful family exports for free if it implements
`rollout_state()` / `set_rollout_state(state)` — its hidden state as one
`[batch, width]` tensor — the way `RCBlock` and `GRUBlock` do.

## Programmatic use

If you'd rather drive things from your own Python instead of the CLIs:

```python
from neurotemplate.common.config import load_config_from_cli, MODEL_CONFIG_DIR
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.pipeline import train_and_evaluate

cfg, _ = load_config_from_cli(
    ["--config", "model_configs/modeling.yaml", "--set", "epochs=1"],
    cls=ModelingConfig,
    config_dir=MODEL_CONFIG_DIR,
)
# ...then call train_and_evaluate(cfg, ...) etc.
```

The CLI wrappers ([modeling/cli/train.py](src/neurotemplate/modeling/cli/train.py), [control/cli/train.py](src/neurotemplate/control/cli/train.py)) are short enough — ~30 lines each — that they're worth reading as a template if you want to write your own driver.

## Package layout

```
src/neurotemplate/
├── hooks.py             # load_modeling_hooks(cfg), load_control_hooks(cfg)
├── common/
│   ├── config.py        # YAML, --set, dict_to_dataclass, CLI parser
│   ├── data_utils.py    # CSV load, sliding windows, norm/denorm
│   ├── metrics.py       # ChannelMetrics / LevelMetrics, per_channel_metrics, level_metrics
│   └── reportcard.py    # _section, write_reportcard (shared by both sides)
├── modeling/
│   ├── config.py        # ModelingConfig
│   ├── dynamics.py      # SSMGrad, GRUBlock, BlockStack, get_dynamics
│   ├── rc.py            # RCBlock — the 2R2C thermal network
│   ├── blocks.py        # build_blocks — the Y/U/D partition behind per-block models
│   ├── diagnostics.py   # nt data heads / nt data decay — pre-training advice
│   ├── windows.py       # positions inside the flattened history window
│   ├── heads.py         # HeadStack (regression / binary / bounded / ordinal)
│   ├── calibration.py   # balanced level weights, calibrated decision thresholds
│   ├── loss.py          # build_loss
│   ├── gradient_sampling.py  # SampledGradientPenalty
│   ├── scenarios.py     # SynthSpec + physics checks
│   ├── experiment.py    # train_with_curriculum, build_trainer
│   ├── pipeline.py      # train_and_evaluate, evaluate_and_plot
│   ├── plotting.py
│   ├── reportcard.py
│   └── cli/             # train, optuna, load, debug_heads
└── control/
    ├── config.py        # ControlConfig
    ├── policy.py        # policy_catalog (MLP/linear), build_policy, describe_policy, residual policy
    ├── losses.py        # the declared objective: parse, resolve names, power_mean
    ├── closed_loop.py   # build_closed_loop, build_control_loss, power_term
    ├── objective.py     # term balance + power-signal shape sweep
    ├── scenarios.py     # load_control_splits (plant + reference files), build_window_dict, build_loaders
    ├── metrics.py       # ComfortMetrics, EnergyMetrics, ActuationMetrics, JumpResponse, ResidualMetrics
    ├── pipeline.py      # train_policy, evaluate_closed_loop, log_control_results
    ├── reportcard.py    # render_control_reportcard
    ├── load.py          # load_frozen_dynamics, load_trained_policy
    └── cli/             # train
└── export/              # nt export model / nt export policy
    ├── state.py         # lift hidden state via the rollout-state protocol
    ├── graphs.py        # WarmStart / Step: a trained SSMGrad in physical units
    ├── policy.py        # PolicyGraph, closed-loop verification, export_policy
    ├── trace.py         # torch -> ONNX, checker, operator allowlist
    ├── verify.py        # trained-system vs package rollout, tolerances
    ├── dynamics.py      # export_model
    ├── package.py       # staging, install, scaffold
    ├── spec.py          # model.json
    ├── runtime.py       # the shipped runtime (numpy + onnxruntime only)
    ├── cpp/             # the shipped C++ runtime, validate / example, CMakeLists
    └── templates/       # README / validate / example / pyproject
```

Module names inside `modeling/` and `control/` are unprefixed (the directory disambiguates). Anything *outside* those subpackages that is specific to one side carries an explicit `modeling_*` / `control_*` prefix — the two hook files at the app root.

## Relationship to Neuromancer

NeuroTemplate is a layer **on top of** Neuromancer, not a replacement. We reuse `Node`, `System` / `SystemPreview`, `Problem`, `Trainer`, `MLP` / `MLP_bounds`, `variable` / `Loss` / `PenaltyLoss`, `DictDataset`, and `MLFlowLogger` directly. What NeuroTemplate adds that Neuromancer does not ship:

- `SSMGrad` plus a vmap-compatible `GRUBlock` (manual GRU unroll for `jacrev` / `vmap`)
- `SampledGradientPenalty` — off-trajectory Jacobian sign sampling
- Synthetic physics assertions (equilibrium / step response / reachability / free decay)
- `HeadStack` per-channel output heads (regression / binary / bounded / ordinal)
- Per-channel metrics + markdown reportcards
- Control-side: a declared objective over named columns, closed-loop assembly

If you find yourself shadowing a Neuromancer class, prefer the import.

## Tests

```bash
uv run pytest tests/
```

Neuromancer is pinned to a git SHA in `pyproject.toml`; bump it deliberately. The suite is deliberately small. It keeps the tests whose failure would mean a run silently trains on wrong numbers — normalization and windowing, the RC and GRU updates, loss alignment, gradient plumbing (what the optimizer can and cannot reach), head round-trips, calibration — plus the physics/objective/stability diagnostics that a reader acts on, and one end-to-end wiring pass per CLI sequence. It does not test report rendering (nothing downstream consumes a particular format yet), validation error messages beyond the checks that would otherwise let a run train silently on the wrong objective, or plots.

Cases that differ only in their input live in `pytest.mark.parametrize` tables rather than in one function each. Before adding a test, ask what silently goes wrong if the logic it covers breaks; if the answer is "a worse error message" or "a differently-worded report", leave it out.
