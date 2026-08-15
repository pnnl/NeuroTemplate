# NeuroTemplate

A library of generic, time-series, data-driven black-box modeling and control building blocks, layered on top of [Neuromancer](https://github.com/pnnl/neuromancer).

The repo contains two sister subsystems:

- **Modeling** — train a system-identification surrogate `y_{k+1} = f(y_k, u_k, d_k) [+ y_k]` from data, with an optional gradient-sign penalty, post-training physics assertions (equilibrium / step response / reachability / free decay), Optuna search, MLflow tracking. Model families: `mlp`, `nssm`, `lssm`, `gru` (black box) and `2r2c` (grey box — a thermal network with a hidden mass, for buildings whose storage matters). Any of them can be split one-per-zone.
- **Control** — train a differentiable predictive control (DPC) policy against a frozen surrogate. Closed-loop rollouts, comfort-band penalties, optional energy, cost and peak-demand objectives, setpoint quantization. Policy families: `mlp`, `linear` (a saturated affine feedback law). Optionally scores a rule-based baseline alongside the learned policy.

Both sides share a thin `common/` layer (YAML/CLI loader, CSV + sliding-window data utilities, per-channel metrics).

NeuroTemplate is meant to be installed as a dependency in an application repo. You bring the data and the two user-defined hook files; the library brings everything else. New framework features land here once, get tagged, and downstream apps pick them up via a version bump.

## Status

Restructured from a flat script collection into an installable library. Phase A complete.

- ✅ Package + install + console script wired (`uv add "NeuroTemplate @ file://..."` works)
- ✅ YAML + CLI override machinery, multi-path data fields (`Union[str, list[str]]`)
- ✅ Hook discovery via `neurotemplate.hooks` (CWD + `cfg.user_hooks_path` override)
- ✅ Single `nt <group> <verb>` dispatcher (replaces the five flat console scripts)
- ✅ 474 / 474 tests passing
- ✅ Control-side parity with sysid: policy family registry (`mlp` / `linear`), comfort + energy metrics, sectioned reportcard with a golden test, MLflow metric logging, optional rule-based baseline comparison
- ☐ Phase B (scrub remaining HVAC vocabulary from library code; relocate runtime artifacts off CWD)
  - ⚠️ **Breaking hook change:** `plot_closed_loop` no longer greps D columns for `"outdoor"`/`"dry-bulb"` and no longer reaches into a private `_power_channel_indices`. Apps must add `power_channels` and `plot_context_channels` to `control_user_defined.py`; `has_power_channels` is removed (it is now the truthiness of `power_channels`).
- ☐ Phase C (ship an in-repo `examples/rtu_mall/`, README templates, end-to-end smoke test)
  - ✅ [template/](template/) — copyable app skeleton: both hook files with every hook stubbed, both config dirs with every field annotated
- ✅ `2r2c` grey-box thermal family and the per-zone model split (see [Model families](#model-families))

### Upgrading: breaking changes from the v6 follow-up

Four signatures moved. All four fail loudly — none of them degrade silently.

| Was | Now | Why |
|---|---|---|
| `preview_config()` returning a `"price"` key | `price_preview` in the control config | the hook cannot see the config, so one config wanting a price preview forced price paths onto every config sharing the hook file |
| `plot_closed_loop(..., plot_dir)` | `plot_closed_loop(..., plot_dir, origins=[df.index[0] for df in test_dfs])` | `t` is elapsed seconds, so without the files' wall-clock origins every plot was dated 1970 |
| `objective.frac_power_zero(...) -> float` | `objective.power_diagnostics(...) -> PowerDiagnostics` | the forward zero count alone reports a correct "compressor off" prediction as a dead gradient |
| `render_control_reportcard(..., frac_power_zero=)` | `render_control_reportcard(..., power=)` | carries the above |

`ResidualMetrics.frac_at_cap` is now `Optional[float]` and is `None` at Δ = 0, where a zero-width box made it vacuously 1.0.

Open work tracked in [audit.md](audit.md).

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
| `nt control train` | Train a DPC policy against a frozen surrogate |

`nt` with no arguments (or `nt -h`) prints the command list. Each command takes
the same standard flags (parsed by [common/config.py](src/neurotemplate/common/config.py)):

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
├── modeling_user_defined.py    # six modeling hooks
├── control_user_defined.py     # the control hooks (only if you train policies)
├── model_configs/
│   └── modeling.yaml
├── control_configs/
│   └── control.yaml
└── data/                       # whatever your YAML paths point at
```

`nt model train` looks for configs in `model_configs/`; `nt control train` looks in `control_configs/`. Keep them separate so each folder only contains the configs relevant to that subsystem.

The library finds the hook files automatically: by default it looks for `modeling_user_defined.py` / `control_user_defined.py` in the current working directory. If you'd rather keep them elsewhere, set `user_hooks_path` in your YAML (a file path or dotted module name).

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
| `hyper_args(trial, base_cfg) -> Config` | required only for `nt model optuna` | Optuna trial setup |
| `tie_norm_stats(nm_mapping, norm_stats, cfg) -> norm_stats` | optional | post-process norm stats before they are frozen (e.g. force two columns onto a shared `shift`/`stretch`) |
| `zone_mapping(nm_mapping, cfg) -> {zone: {Y: [col], U: [col]}} \| None` | optional | split Y and U into independent per-zone sub-models (see [Zones](#zones)) |

The library imports each hook by attribute (no defaulting), so every row above except `hyper_args`, `tie_norm_stats` and `zone_mapping` must exist in `modeling_user_defined.py`, even when the body just returns a trivial value. `hyper_args` is only looked up on the code path that needs it; `tie_norm_stats` and `zone_mapping` are `hasattr`-gated and skipped when absent.

### Control (`control_user_defined.py`)

| Function | Required | Used by |
|---|---|---|
| `controlled_channels(nm_mapping) -> ([col], [idx])` | yes | comfort objective |
| `comfort_band_signal(nm_mapping, norm_stats, n_samples, nsteps, band, generator) -> (ymin, ymax)` | yes | scenarios |
| `actuation_bounds(nm_mapping, norm_stats) -> (umin, umax)` | yes | policy bounds |
| `power_channels(nm_mapping) -> [col]` | must be defined; return `[]` when nothing is metered | closed-loop plot power row + energy metrics |
| `plot_context_channels(nm_mapping) -> [col]` | must be defined; return `[]` for none | D traces overlaid on the closed-loop temperature axis |
| `power_metric(yn, U, D, nm_mapping, norm_stats) -> Tensor` + `POWER_METRIC_IMPLEMENTED: bool` | required only when `energy_weight > 0` | energy / cost / peak objectives + energy reportcard |
| `reference_control(yn_now, ymin_now, ymax_now, d_now, nm_mapping, norm_stats) -> Tensor` | optional | rule-based baseline **and** the residual policy's anchor |
| `build_baseline_policy(cfg, sysid_cfg, nm_mapping, norm_stats) -> Node \| None` | optional; mutually exclusive with `reference_control` | escape hatch for a baseline a memoryless rule cannot express |
| `preview_config(nm_mapping) -> {signal: n_steps}` | optional | per-column disturbance and comfort-band preview widths (the tariff preview is `price_preview` in the config) |

The energy tariff is config, not a hook: `train_price_path` / `dev_price_path` / `test_price_path` (one CSV per split) + `price_scale` + `price_preview` in `control.yaml`, required together only when `cost_weight > 0`.

`power_metric` returns whatever units the use case meters; the library never converts, so the reportcard labels energy as `power·h`. Metered channels the surrogate already predicts are the best thing to return, because the energy numbers then mean something outside the run. An actuation proxy such as `(setpoint - ambient)²` is a fine alternative when nothing meters power — just read the energy section as relative rather than physical. What has actually cost campaigns here is the term's *scale*, not its shape: a proxy carried over from another dataset came out at 1.7× the entire comfort objective while every pooled metric looked fine. The objective-balance report and the shape sweep both print before training so that magnitude is on screen from the start.

If `reference_control` is defined, `nt control train` rolls that rule out on the same test windows against the same fixed-seed comfort band, adds a learned-vs-baseline table to the reportcard, and states plainly whether the learned policy cleared its own reference on mean exceedance. **A policy that does not clear its reference should ship the reference** — training is scored against a surrogate whose held-out error is not the plant's, and the rule is the one controller here whose behaviour does not depend on that. Without a rule the comparison section just says so.

The same hook is what `residual_delta` anchors to. Set it and the policy commands `reference_control(...) + δ` with `|δ| ≤ residual_delta` instead of commanding a setpoint outright: the rule keeps comfort, δ answers whatever else is being optimised, and the ±Δ box is a trust region on surrogate error rather than a claim about the network. The reportcard reports the box and the fraction of steps δ spent pinned at it — a cap that binds most of the time means either the rule is wrong for this plant or the optimizer is riding a surrogate artifact into the wall. Note Δ bounds the *action*, not the outcome: the commanded setpoint stays within Δ of the rule's, the zone temperature does not, because the plant is in between.

`train_price_path`/`dev_price_path`/`test_price_path` are the switch for price-driven control — set all three (one CSV per split: a timestamp column plus one price column, in physical $/kWh) plus `price_scale`, and the library divides by `price_scale` so the policy sees an O(1) input and multiplies it back to report dollars. Price rides in the rollout dict rather than as a D column: a D column would need the frozen surrogate retrained, and would tell the dynamics model that electricity price drives zone temperature.

**Point it at a real historical series, and do not derive it from the clock.** A tariff that is a deterministic function of hour-of-day is one the policy can learn *as* hour-of-day — D is strongly clock-correlated, so "read the price" and "read the time" collapse into the same task and you get a convincing load-shifting trace with no price responsiveness in it. Real day-to-day variation is what forces the policy to read the channel. (`t` would not help anyway: `sec_elapsed` counts from the first row of the file, not from midnight.) A per-split file also means the test file is where to put whatever price swings you want the eval judged on.

The library draws its own window offsets out of each split's file, so price windows are deliberately independent of the weather windows they are scored beside — that removes the residual price↔weather correlation as a second route to the same shortcut, and it means the price record need not span the plant record. Reproducibility rides on the `generator` `build_price` is handed: `evaluate_closed_loop` reseeds it from `EVAL_BAND_SEED`, so the learned policy and the baseline are scored against a byte-identical tariff. Price is drawn after the band, so enabling it leaves band realizations unchanged.

`price_align` chooses how those offsets are drawn, and the choice is a real trade-off:

| | `random` (default) | `day` |
|---|---|---|
| offset | any step | chosen so the tariff's time of day matches the window's |
| the tariff's 8 pm peak | lands at a random hour of the plant's day | lands at the plant's 8 pm, in every window and every plot |
| clock shortcut | closed | partly re-opened |

**Use `day` when the point is to see pre-cooling.** Under `random` a real evening peak is drawn at a different hour in every window, so the pre-cool it should provoke averages away and no plot shows it — v6 had to work around this by cutting the eval price file down to exactly one window long, leaving `randint` a single legal offset. `day` keeps the day random (so day-to-day price variation still forces the policy to read the channel) and pins only the time of day, read off the plant frame's own `DatetimeIndex`. The cost is that a series peaking at hour 19 on most days becomes that much more learnable *as* hour 19, so under `day` a load-shifting trace is not by itself evidence of price response: re-score the trained policy with the peak moved to a different hour and check the load moves with it.

`cost_weight` then weights `mean_t(price · power)`, and `peak_weight` weights the per-window power mean `M_q(power)` — all three power terms are reductions of one shared signal. Neither is solvable at the default `train_pred_steps: 48`; expect 144–288 steps, and expect `cost_weight` to need a price preview (`price_preview`) before it can shift load rather than merely react to it.

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
nt model load --run_id <run_id> --id_data path/to/test.csv

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
| `dummy` | neither | a passthrough that mints a checkpoint without identifying anything |

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

**The time constants are a prior, not a fit**, and they default to frozen.
The mass temperature is unobserved, so every rollout has to seed it from an
`in_steps` window an hour long at most, against a state that lags by many
hours. Left trainable, the optimizer discovers that shortening the slow mode
is a cheaper way to absorb that wrong seed than fixing the forcing — on
synthetic data from a known 31.6 h building, seeded truthfully training moves
`tau_slow` 13.4 → 16.5 h, and seeded the way a real rollout must, the same
training drives it to 8.7 h. Letting the window choose its own seed does not
help; an hour of zone temperature carries no information about a ten-hour
state. So set the prior from what you know — a free-decay episode gives it
directly — and read back what you built:

```yaml
model: 2r2c
rc_tau_ambient_h: 2.0     # zone <-> ambient
rc_tau_zone_mass_h: 1.0   # zone <-> mass, from the zone's side
rc_tau_mass_zone_h: 4.0   # mass <-> zone, from the mass's side
# rc_train_taus: false    # only with a rollout long enough to identify them
```

Those three are components, not modes. The modes they produce on 5-minute
data — `tau_fast = 0.60 h`, `tau_slow = 13.40 h` — are what the reportcard
prints, per zone and per channel:

```
- **fy:** `PerZoneBlock`, params=2,504, in=42, out=2
  - **zone `north`:** `RCBlock`, params=1,252, in=30, out=1
    - North Zone Air Temperature [°C]: tau_fast=0.60 h, tau_slow=13.40 h
```

What it gives up: `g` cannot see the state, so it cannot represent a
thermostat deadband. "The compressor is off whenever the setpoint is above
the zone" is a function of the *difference*, and `2r2c` renders it as a
monotone response to setpoint with no cut-off — smooth cooling where the
real plant's is staged. That is the price of the stability guarantee, and it
is why this is one family among several rather than the only one.

### Zones

A *zone* is a group of Y channels that gets its own sub-model: its own
parameters, its own hidden state, and no term reading any other zone's
state. Declare them with the optional `zone_mapping` hook:

```python
def zone_mapping(nm_mapping, cfg):
    return {
        "north": {"Y": ["North Zone Air Temperature [°C]"],
                  "U": ["North Cooling Setpoint [°C]"]},
        "south": {"Y": ["South Zone Air Temperature [°C]"],
                  "U": ["South Cooling Setpoint [°C]"]},
    }
```

Each zone owns its own Y and U channels and sees every D channel. **U need
not partition:** any control no zone claims is shared with all of them,
which is what you want for one air handler serving several zones. Y must
partition exactly, and anything else — a state claimed twice, a state left
out, a column that is not in the mapping — is named before the first epoch
runs.

`2r2c` is per-zone whether or not the hook exists: with no hook, each Y
channel still gets its own RC network, driven by every U channel. Declaring
zones is how you say *which setpoint drives which zone*. For `mlp` and `gru`
the split is opt-in with `per_zone: true`, so you can run 2R2C-per-zone and
MLP-per-zone against each other on the same mapping.

Inter-zone conduction is not modeled. Zones are independent by construction,
and a building whose zones exchange meaningful heat is outside what this
buys you.

## Two optional control capabilities

Both are off by default and both fail fast rather than degrade, so a run that
does not ask for them is unchanged.

### Grid response: cost and peak-demand objectives

`energy`, `cost` and `peak` are three *reductions of one power signal* —
`mean_t(p)`, `mean_t(price·p)`, and the per-window power mean
`M_q(p) = (mean_t p^q)^(1/q)` averaged over the batch. All three read the same
`power_metric` output, so they cannot end up pricing different quantities.

To turn on price-driven control:

1. **Implement `power_metric` and set `POWER_METRIC_IMPLEMENTED = True`.**
   Nothing else here works without it.
2. **Get a real price series, as a CSV per split** (timestamp + one price
   column in $/kWh). Sample: `grid_response.ipynb` in Neuromancer's
   `examples/domain_examples/` downloads PG&E Hourly Flex Pricing from
   `https://raw.githubusercontent.com/EladMichael/small_example_datasets/main/hourly-flex-prices-historical-sample.xlsx`.
   Convert it to CSV; no need to upsample it yourself.
3. **Set `train_price_path`/`dev_price_path`/`test_price_path` + `price_scale`
   in control.yaml**, and `price_repeat` if the file is coarser than your
   plant data's timestep (12 for hourly price over 5-minute data — the
   library upsamples it, so it needs a whole-number ratio, not the finer
   resolution itself).
   *Do not build a time-of-use table keyed on hour-of-day* — see the hook
   contract above for why the policy will learn the clock instead of the price.
4. **Give the policy a price preview**, or the cost term is not solvable —
   pre-cooling happens *before* the expensive block, so a policy that only
   sees the current price can react but never shift:

   ```yaml
   price_preview: 48       # 48 steps = 4 h at 5-minute data
   ```

5. **Set the weights and lengthen the rollout:**

   ```yaml
   energy_weight: 0.1      # keep one of these non-zero alongside peak_weight
   cost_weight: 0.5
   peak_weight: 0.1
   peak_q: 4               # 1 = plain mean, 2 = RMS, larger -> hard max
   train_pred_steps: 288   # 48 is 4 hours: too short to precool and coast
   ```

`train_pred_steps` is the real cost — rollout memory grows roughly linearly.

#### If what you want is pre-cooling

Pre-cooling is load shifting you can point at: drive the zone down through the
cheap block, then coast up through the expensive one on the building's own
thermal mass. Four things have to be true at once or it will not appear, and
three of them are not the weights:

- **`price_preview` covers the run-up to the peak.** Pre-cooling is by
  definition an action taken before the expensive block. A policy whose only
  price input is the current one can be cheap, but it cannot pre-cool.
- **The comfort band is wide enough to store anything.** Load shifting *is*
  thermal storage, and a 0.5 °C band leaves nowhere to put it — reporting "no
  price response" out of a narrow band is measuring the band. 1.5–2.5 °C is a
  reasonable place to start.
- **The rollout spans the cycle.** 288 steps (24 h at 5-minute data) covers a
  precool and the coast that pays for it; the default 48 covers neither.
- **`cost_weight` carries the power side, not `energy_weight`.** They pull
  different ways: `energy_weight` rewards using less power at *every* hour, so
  the cheapest way to satisfy it is to stop cooling — load *shedding*, which
  shows up as a big power cut with comfort going backwards. Only `cost_weight`
  rewards moving power to the cheap hours. If pre-cooling is the goal, start
  with `cost_weight` alone and a comfort weight high enough that the savings
  have to be found rather than bought.

Set `price_align: day` while you are looking for it, or the tariff's evening
peak lands at a different hour in every window and the precool averages out of
every plot. Then confirm it is the price and not the clock: re-score the
trained policy with the peak moved to a different hour (holding the weather,
the band and the surrogate fixed) and check the load follows. Pre-heating
against an evening peak is the same recipe with the sign flipped.

### Residual control: bound the policy to a rule

The policy commands `reference_control(...) + δ` with `|δ| ≤ residual_delta`
instead of commanding a setpoint. This is a trust region on surrogate error,
not a claim about what networks find easy to learn: training's gradient *and*
its model-selection criterion both come from a surrogate whose held-out score
is not the plant's, and a ±Δ box caps what that error can cost you.

1. **Implement `reference_control`** (see the template; `control.policy.bracket_band`
   does the hard part for a heating/cooling pair). Delete
   `build_baseline_policy` if you have one — defining both raises.
2. **Set `residual_delta`**, in physical units:

   ```yaml
   residual_delta: 1.0     # +/- 1 degC around the rule
   setpoint_step: 0.5      # Delta below one grid step is rejected: every
                           # residual would quantize back to zero
   ```

3. **Read the verdict.** The reportcard scores the rule on the same windows
   against the same fixed-seed band and states whether the learned policy
   cleared it on mean exceedance. It does not always. **A policy that does not
   clear its reference should ship the reference** — that call is the reason
   δ = 0 is worth scoring at all, because the Trainer checkpoints after each
   epoch's optimizer steps and never considers δ = 0 itself.

Δ bounds the **action**, not the outcome: the commanded setpoint stays within Δ
of the rule's, the zone temperature does not, because the plant is in between.

### Together

This is the case the residual design is really for — the rule keeps comfort,
δ answers the incentive, and Δ is an operator-legible knob ("how far from the
comfort rule am I willing to move for money"):

```yaml
residual_delta: 1.0
cost_weight: 0.5
peak_weight: 0.1
energy_weight: 0.1
train_pred_steps: 288
```

One caveat if you deploy: a policy that reads price gains a fifth key
(`"price"`) on its Node's `input_keys`. The residual alone leaves the Node
signature untouched — that is why the rule is composed *inside* the policy's
callable — but residual-plus-price needs the evaluator rebuilt.

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

**Before `nt control train` trains anything:**

- **Objective balance** — every term's value and share of the objective on one
  real batch, warning when one term outweighs all the others combined. This is
  the one that asserts something: a term carried over from another dataset once
  came out at 1.7× the whole comfort objective.
- **Power term shape** — the power signal swept across each actuation box, so
  its magnitude and shape sit next to the weights you are choosing. One sweep
  serves all three power-derived terms, since cost and peak are reductions of
  the same signal. Reported without a verdict: an honest power metric is
  monotone in actuator effort, and the interior optimum belongs to the summed
  objective, not to this term alone.
- **Power clamped to zero** — the share of scored steps where the power signal
  sits at exactly zero and so contributes no gradient. A rounding detail for a
  mean energy term; not one for a cost term, because the near-zero steps are
  disproportionately the cheap hours load is supposed to move *into*. A large
  figure is a surrogate problem, not a weight.
- **Objective notes** — configurations that are legal but cannot do what they
  were switched on for: `cost_weight` with no price preview (reactive, not
  shifting), and `peak_weight` as the only power term (minimisable by *adding*
  off-peak load).

**After the closed-loop evaluation:**

- **Actuation** — per-control mean |Δu| and the fraction of the evaluation
  spent against each bound. Reported, not judged: a setpoint parked at its
  least-aggressive limit while nothing is demanded of it looks identical to one
  that is stuck. The band-jump audit below is what tells them apart.
- **Band-jump response** — every step change in the reference band that left a
  controlled channel out of band, scored on whether any control moved, whether
  the channel followed, and whether it got back in band. Two controllers
  measured at 16.6% and 17.9% pooled violation reacted to 0% and 75% of their
  cooling-demand jumps respectively; the pooled metric ranked them backwards.
- **Residual saturation** — for a residual policy, Δ and the fraction of steps
  δ spent pinned at ±Δ, measured *after* quantization and the actuator clamp so
  it is the deviation actually deployed. A cap that binds most of the time means
  either the rule is wrong for this plant or the optimizer is riding a surrogate
  artifact into the wall. Read it before the comfort number.
- **Baseline comparison and the ship/don't-ship call** — the same headline
  numbers for the learned policy and the rule, on identical windows and an
  identical band, plus an explicit verdict. For a residual policy this is also
  the δ = 0 score, which is what turns its floor from rhetorical into a
  decision: the Trainer checkpoints after each epoch's optimizer steps, so
  δ = 0 is never itself a candidate and has to be scored on purpose.

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
│   ├── metrics.py       # ChannelMetrics, per_channel_metrics
│   └── reportcard.py    # _section, write_reportcard (shared by both sides)
├── modeling/
│   ├── config.py        # ModelingConfig
│   ├── dynamics.py      # SSMGrad, GRUBlock, PerZoneBlock, get_dynamics
│   ├── rc.py            # RCBlock — the 2R2C thermal network
│   ├── zones.py         # build_zones — the Y/U partition behind per-zone models
│   ├── windows.py       # positions inside the flattened history window
│   ├── heads.py         # HeadStack (regression / binary / bounded / ordinal)
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
    ├── policy.py        # policy_catalog (MLP/linear), build_policy, describe_policy, STEQuantize, residual policy
    ├── closed_loop.py   # build_closed_loop, build_control_loss, power_term
    ├── objective.py     # term balance + power-signal shape sweep
    ├── scenarios.py     # BandSpec, build_window_dict, build_loaders, build_price
    ├── metrics.py       # ComfortMetrics, EnergyMetrics, ActuationMetrics, JumpResponse, ResidualMetrics
    ├── pipeline.py      # train_policy, evaluate_closed_loop, evaluate_baseline, clears_reference
    ├── reportcard.py    # render_control_reportcard
    ├── load.py          # load_frozen_dynamics
    └── cli/             # train
```

Module names inside `modeling/` and `control/` are unprefixed (the directory disambiguates). Anything *outside* those subpackages that is specific to one side carries an explicit `modeling_*` / `control_*` prefix — the two hook files at the app root, and the five console scripts.

## Relationship to Neuromancer

NeuroTemplate is a layer **on top of** Neuromancer, not a replacement. We reuse `Node`, `SystemPreview`, `SSM`, `Block` / `MLP` / `MLP_bounds`, `LitTrainer`, `LitProblem`, `PenaltyLoss`, `DictDataset`, and `MLFlowLogger` directly. What NeuroTemplate adds that Neuromancer does not ship:

- `SSMGrad` plus a vmap-compatible `GRUBlock` (manual GRU unroll for `jacrev` / `vmap`)
- `SampledGradientPenalty` — off-trajectory Jacobian sign sampling
- Synthetic physics assertions (equilibrium / step response / reachability / free decay)
- `HeadStack` per-channel output heads (regression / binary / bounded / ordinal)
- Per-channel metrics + markdown reportcards
- Control-side: `STEQuantize`, `BandSpec` + comfort-band penalties, closed-loop assembly

If you find yourself shadowing a Neuromancer class, prefer the import.

## Tests

```bash
uv run pytest tests/
```

All 170 pass today, against the current local Neuromancer checkout. The suite covers config, data utilities, metrics, heads, reportcards (both sides, golden-file pinned), hook contracts, and the Neuromancer-touching dynamics/loss/closed-loop paths.
