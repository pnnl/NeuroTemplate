# NeuroTemplate

A library of generic, time-series, data-driven black-box modeling and control building blocks, layered on top of [Neuromancer](https://github.com/pnnl/neuromancer).

The repo contains two sister subsystems:

- **Modeling** — train a system-identification surrogate `y_{k+1} = f(y_k, u_k, d_k) [+ y_k]` from data, with an optional gradient-sign penalty, post-training physics assertions (equilibrium / step response / reachability / free decay), Optuna search, MLflow tracking. Model families: `mlp`, `nssm`, `lssm`, `gru`.
- **Control** — train a differentiable predictive control (DPC) policy against a frozen surrogate. Closed-loop rollouts, comfort-band penalties, optional energy/effort objectives, setpoint quantization. Policy families: `mlp`, `linear` (a saturated affine feedback law). Optionally scores a rule-based baseline alongside the learned policy.

Both sides share a thin `common/` layer (YAML/CLI loader, CSV + sliding-window data utilities, per-channel metrics).

NeuroTemplate is meant to be installed as a dependency in an application repo. You bring the data and the two user-defined hook files; the library brings everything else. New framework features land here once, get tagged, and downstream apps pick them up via a version bump.

## Status

Restructured from a flat script collection into an installable library. Phase A complete.

- ✅ Package + install + console script wired (`uv add "NeuroTemplate @ file://..."` works)
- ✅ YAML + CLI override machinery, multi-path data fields (`Union[str, list[str]]`)
- ✅ Hook discovery via `neurotemplate.hooks` (CWD + `cfg.user_hooks_path` override)
- ✅ Single `nt <group> <verb>` dispatcher (replaces the five flat console scripts)
- ✅ 170 / 170 tests passing
- ✅ Control-side parity with sysid: policy family registry (`mlp` / `linear`), comfort + energy metrics, sectioned reportcard with a golden test, MLflow metric logging, optional rule-based baseline comparison
- ☐ Phase B (scrub remaining HVAC vocabulary from library code; relocate runtime artifacts off CWD)
  - ⚠️ **Breaking hook change:** `plot_closed_loop` no longer greps D columns for `"outdoor"`/`"dry-bulb"` and no longer reaches into a private `_power_channel_indices`. Apps must add `power_channels` and `plot_context_channels` to `control_user_defined.py`; `has_power_channels` is removed (it is now the truthiness of `power_channels`).
- ☐ Phase C (ship an in-repo `examples/rtu_mall/`, README templates, end-to-end smoke test)
  - ✅ [template/](template/) — copyable app skeleton: both hook files with every hook stubbed, both config dirs with every field annotated

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
├── control_user_defined.py     # four control hooks (only if you train policies)
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

The library imports each hook by attribute (no defaulting), so every row above except `hyper_args` and `tie_norm_stats` must exist in `modeling_user_defined.py`, even when the body just returns a trivial value. `hyper_args` is only looked up on the code path that needs it; `tie_norm_stats` is `hasattr`-gated and skipped when absent.

### Control (`control_user_defined.py`)

| Function | Required | Used by |
|---|---|---|
| `controlled_channels(nm_mapping) -> ([col], [idx])` | yes | comfort objective |
| `comfort_band_signal(nm_mapping, norm_stats, n_samples, nsteps, band, generator) -> (ymin, ymax)` | yes | scenarios |
| `actuation_bounds(nm_mapping, norm_stats) -> (umin, umax)` | yes | policy bounds |
| `power_channels(nm_mapping) -> [col]` | must be defined; return `[]` when nothing is metered | closed-loop plot power row + energy metrics |
| `plot_context_channels(nm_mapping) -> [col]` | must be defined; return `[]` for none | D traces overlaid on the closed-loop temperature axis |
| `power_metric(yn, U, D, nm_mapping, norm_stats) -> Tensor` + `POWER_METRIC_IMPLEMENTED: bool` | required only when `effort_weight > 0` | effort objective + energy reportcard |
| `build_baseline_policy(cfg, sysid_cfg, nm_mapping, norm_stats) -> Node \| None` | optional | rule-based controller scored alongside the learned policy |
| `preview_config(nm_mapping) -> {signal: n_steps}` | optional | per-column disturbance + comfort-band preview widths |

`power_metric` returns whatever units the use case meters; the library never converts, so the reportcard labels energy as `power·h`. Metered channels the surrogate already predicts are the best thing to return, because the energy numbers then mean something outside the run. An actuation proxy such as `(setpoint - ambient)²` is a fine alternative when nothing meters power — just read the energy section as relative rather than physical. What has actually cost campaigns here is the term's *scale*, not its shape: a proxy carried over from another dataset came out at 1.7× the entire comfort objective while every pooled metric looked fine. The objective-balance report and the shape sweep both print before training so that magnitude is on screen from the start.

If `build_baseline_policy` is defined, `nt control train` rolls that controller out on the same test windows against the same fixed-seed comfort band and adds a learned-vs-baseline table to the reportcard. Without it the comparison section just says so. The returned `Node` must write `["U"]` and have no trainable parameters — it is only ever evaluated, never trained.

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
- **Effort term shape** — the effort term swept across each actuation box, so
  its magnitude and shape sit next to the weights you are choosing. Reported
  without a verdict: an honest power metric is monotone in actuator effort, and
  the interior optimum belongs to the summed objective, not to this term alone.

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
│   ├── dynamics.py      # SSMGrad, GRUBlock, get_dynamics  (NSSM/LSSM/MLP/GRU)
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
    ├── policy.py        # policy_catalog (MLP/linear), build_policy, describe_policy, STEQuantize
    ├── closed_loop.py   # build_closed_loop, build_control_loss, power_term
    ├── objective.py     # term balance + effort-term shape sweep
    ├── scenarios.py     # BandSpec, build_window_dict, build_loaders
    ├── metrics.py       # ComfortMetrics, EnergyMetrics, ActuationMetrics, JumpResponse
    ├── pipeline.py      # train_policy, evaluate_closed_loop, evaluate_baseline
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
