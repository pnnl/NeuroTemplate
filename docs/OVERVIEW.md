# NeuroTemplate: an overview

What the library does, a session end to end, and every command. The guided flow
(`nt guide`, or [GUIDE.md](../GUIDE.md)) is where to start; this page and the others
in `docs/` are references it points into.

| To | Read |
|---|---|
| install the library and set up an application | [INSTALL.md](INSTALL.md) |
| train a dynamics model (system identification) | [MODELING.md](MODELING.md) |
| train a controller against it | [CONTROL.md](CONTROL.md) |
| ship a trained model or controller | [EXPORT.md](EXPORT.md) |

## What it does

- **Modeling.** Trains a surrogate `y[k+1] = f(y, u, d)` over the named
  columns of a CSV: outputs `Y`, controls `U` and disturbances `D`. You can
  choose black-box families or a grey-box thermal network, and mix them in
  one model, one family per group of outputs. The thermal network can say
  what a setpoint is (a thermostat's demand, optionally latched), what
  conditions change its effect, and what its equipment draws. Every run is also scored on
  how it behaves in a long rollout and, if you describe the physics, in
  states the data never visited.
- **Control.** Trains a differentiable predictive control (DPC) policy
  through the frozen surrogate. The objective is a table over named
  columns: bands to stay in, schedules to track, energy, cost, peak demand.
  Each command stays inside bounds that may follow a forecast at every step
  (a setpoint inside the comfort band), or, as an option, within a box
  around a rule you already trust.
- **Export.** Ships a model or a policy as ONNX graphs run by onnxruntime,
  as torch programs run by torch, or as plain C with no runtime library at
  all. Before anything is written, the export checks that the package
  reproduces the trained model.

You bring the data, two hook files of Python and a YAML config per side,
all scaffolded by `nt init`. Each run is tracked in MLflow and
leaves a markdown reportcard.

## A session

```bash
nt init my_app && cd my_app/
# data/: train.csv, dev.csv, test.csv; fill in nt_model.py and model_configs/modeling.yaml
nt model train                                   # ... Finished run: <model_run>
nt control train --set sysid_run_id=<model_run>  # ... Finished control run: <policy_run>
nt export model  --run_id <model_run> --data data/test.csv --out delivery
nt export policy --run_id <policy_run> --out delivery
```

Every `nt` command is a thin wrapper over a Python function, so the same
runs can be driven from a notebook or a script. MODELING.md and CONTROL.md
show the calls.

## Commands

| Command | Purpose |
|---|---|
| `nt init [dir]` | start an application: the hooks, the configs, `data/` and `AGENTS.md` |
| `nt guide [step]` | the guided flow, one step at a time, and the guides beside it |
| `nt model train` | train a surrogate |
| `nt model optuna` | hyperparameter search, then retrain the best |
| `nt model load` | score a trained surrogate on other data |
| `nt model debug-heads` | inspect how the output heads behave |
| `nt data heads` | before training: which output head a column wants |
| `nt data ladder` | before training: which RC network a thermal zone's record supports |
| `nt data thermostat` | before training: the heat a zone's setpoints deliver |
| `nt control train` | train a policy against a trained surrogate |
| `nt export model` | ship a surrogate |
| `nt export policy` | ship a policy |

`nt` alone lists them, and `nt <group> <verb> --help` gives a command's
flags. The `train` and `optuna` commands read a YAML config and share these
flags:

```
--config PATH        a YAML file, or a name in the side's config directory
--set key=value      override any field (repeatable)
--print-config       print the resolved config and exit
--dump-config PATH   write the resolved config to a file and exit
--list-configs       list the side's config directory and exit
```

## Development

```bash
uv sync --extra dev
uv run pytest tests/
```

The suite is kept small on purpose. It covers what would let a run train
silently on wrong numbers (normalization and windowing, the recurrent and
RC updates, loss alignment, gradient plumbing, head round trips,
calibration, export agreement), plus the diagnostics a reader acts on and
one end-to-end pass per command sequence. It does not test report
formatting, error-message wording or plots.

NeuroTemplate sits on top of Neuromancer and reuses its `Node`, `System`,
`Problem`, `Trainer`, blocks and losses directly. Where a Neuromancer class
does the job, import it rather than shadowing it. Neuromancer is pinned to
one git commit in `pyproject.toml`; bump it deliberately.
