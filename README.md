# NeuroTemplate

A library of generic, time-series, data-driven black-box modeling and control building blocks, layered on top of [Neuromancer](https://github.com/pnnl/neuromancer).

The repo contains two sister subsystems:

- **Modeling** — train a system-identification surrogate `y_{k+1} = f(y_k, u_k, d_k) [+ y_k]` from data, with optional physics-informed losses (gradient sign penalties, equilibrium / step-response / free-decay scenarios), Optuna search, MLflow tracking.
- **Control** — train a differentiable predictive control (DPC) policy against a frozen surrogate. Closed-loop rollouts, comfort-band penalties, optional energy/effort objectives, setpoint quantization.

Both sides share a thin `common/` layer (YAML/CLI loader, CSV + sliding-window data utilities, per-channel metrics).

NeuroTemplate is meant to be installed as a dependency in an application repo. You bring the data and the two user-defined hook files; the library brings everything else. New framework features land here once, get tagged, and downstream apps pick them up via a version bump.

## Status

Restructured from a flat script collection into an installable library. Phase A complete.

- ✅ Package + install + console script wired (`uv add "NeuroTemplate @ file://..."` works)
- ✅ YAML + CLI override machinery, multi-path data fields (`Union[str, list[str]]`)
- ✅ Hook discovery via `neurotemplate.hooks` (CWD + `cfg.user_hooks_path` override)
- ✅ Single `nt <group> <verb>` dispatcher (replaces the five flat console scripts)
- ✅ 102 / 102 tests passing
- ☐ Phase B (scrub remaining HVAC vocabulary from library code; relocate runtime artifacts off CWD)
- ☐ Phase C (ship `examples/rtu_mall/`, README templates, end-to-end smoke test)

Open work tracked in [audit.md](audit.md). The restructure plan is in `.claude/plans/toasty-toasting-glade.md`.

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

Until Phase C ships `examples/rtu_mall/`, the fastest way to bootstrap an app is to copy [tests/fixtures/modeling_user_defined.py](tests/fixtures/modeling_user_defined.py) + [tests/fixtures/control_user_defined.py](tests/fixtures/control_user_defined.py) into a new directory and create `model_configs/modeling.yaml` and `control_configs/control.yaml`.

## Hook contracts

### Modeling (`modeling_user_defined.py`)

| Function | Required | Used by |
|---|---|---|
| `data_mapping(df, cfg) -> {Y, U, D, t: [col]}` | yes | pipeline at data-prep time |
| `data_head_config(nm_mapping) -> {col: HeadSpec}` | optional, default `{}` | output-head construction |
| `column_norm_config(nm_mapping) -> {col: norm_type \| None}` | optional, default `{}` | per-column norm override |
| `synth_cfg(nm_mapping, norm_stats) -> dict \| None` | required, return `None` to disable | synthetic-scenario penalties |
| `hyper_args(trial, base_cfg) -> Config` | required only for `*-modeling-optuna` | Optuna trial setup |
| `sub_plot(df_true, col_name) -> tuple \| None` | optional, default `None` | per-column subplot guidance |

### Control (`control_user_defined.py`)

| Function | Required | Used by |
|---|---|---|
| `controlled_channels(nm_mapping) -> ([col], [idx])` | yes | comfort objective |
| `comfort_band_signal(nm_mapping, norm_stats, n_samples, nsteps, band, generator) -> (ymin, ymax)` | yes | scenarios |
| `actuation_bounds(nm_mapping, norm_stats) -> (umin, umax)` | yes | policy bounds |
| `power_metric(yn, U, nm_mapping, norm_stats) -> Tensor` + `POWER_METRIC_IMPLEMENTED: bool` | optional, gated | effort objective |

Once Phase C lands, `docs/HOOKS.md` will hold the canonical contract reference with signatures + examples.

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
│   └── metrics.py       # ChannelMetrics, per_channel_metrics
├── modeling/
│   ├── config.py        # ModelingConfig
│   ├── dynamics.py      # SSMGrad, GRUBlock, get_dynamics  (NSSM/LSSM/MLP/TCN/GRU)
│   ├── heads.py         # HeadStack (regression / binary / bounded / ordinal)
│   ├── loss.py          # build_loss
│   ├── gradient_sampling.py  # SampledGradientPenalty
│   ├── scenarios.py     # synthetic-physics penalties (equilibrium / step / free-decay)
│   ├── experiment.py    # train_with_curriculum, build_trainer
│   ├── pipeline.py      # train_and_evaluate, evaluate_and_plot
│   ├── plotting.py
│   ├── reportcard.py
│   └── cli/             # train, optuna, load, debug_heads
└── control/
    ├── config.py        # ControlConfig
    ├── policy.py        # build_policy, STEQuantize
    ├── closed_loop.py   # build_closed_loop, build_control_loss
    ├── scenarios.py     # BandSpec, build_window_dict, build_loaders
    ├── pipeline.py      # train_policy, evaluate_closed_loop, render_control_reportcard
    ├── load.py          # load_frozen_dynamics
    └── cli/             # train
```

Module names inside `modeling/` and `control/` are unprefixed (the directory disambiguates). Anything *outside* those subpackages that is specific to one side carries an explicit `modeling_*` / `control_*` prefix — the two hook files at the app root, and the five console scripts.

## Relationship to Neuromancer

NeuroTemplate is a layer **on top of** Neuromancer, not a replacement. We reuse `Node`, `SystemPreview`, `SSM`, `Block` / `MLP` / `MLP_bounds`, `LitTrainer`, `LitProblem`, `PenaltyLoss`, `DictDataset`, and `MLFlowLogger` directly. What NeuroTemplate adds that Neuromancer does not ship:

- `SSMGrad` plus a vmap-compatible `GRUBlock` (manual GRU unroll for `jacrev` / `vmap`)
- `SampledGradientPenalty` — off-trajectory Jacobian sign sampling
- Synthetic physics-scenario penalties (equilibrium / step response / free decay)
- `HeadStack` per-channel output heads (regression / binary / bounded / ordinal)
- Per-channel metrics + markdown reportcards
- Control-side: `STEQuantize`, `BandSpec` + comfort-band penalties, closed-loop assembly

If you find yourself shadowing a Neuromancer class, prefer the import.

## Tests

```bash
uv run pytest tests/
```

All 102 pass today, against the current local Neuromancer checkout. The suite covers config, data utilities, metrics, heads, reportcard, hook contracts, and the Neuromancer-touching dynamics/loss/closed-loop paths.
