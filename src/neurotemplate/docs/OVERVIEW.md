# NeuroTemplate: an overview

What the library does, a session end to end, and every command. The guided flow
(`nt guide`, or [GUIDE.md](../GUIDE.md)) is where to start; this page and the other
references (`nt docs`) are what it points into.

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
  what a setpoint does (a thermostat, optionally latched) and what its equipment
  draws. Every run is also scored on
  how it behaves in a long rollout and, if you describe the physics, in
  states the data never visited.
- **Control.** Trains a differentiable predictive control (DPC) policy
  through the frozen surrogate. The objective is a table over named
  columns: bands to stay in, schedules to track, energy, cost, peak demand.
  Each command stays inside bounds that may follow a forecast at every step
  (a setpoint inside the comfort band).
- **Export.** Ships a model or a policy as ONNX graphs run by onnxruntime,
  as torch programs run by torch, or as plain C with no runtime library at
  all. Before anything is written, the export checks that the package
  reproduces the trained model.

You bring the data and its column roles; a controller adds an objective. Everything else
is a default each run prints, and hooks in Python are where you change the model, all
scaffolded by `nt init`. Each run is tracked in MLflow and leaves a markdown reportcard.

## A session

```bash
nt init my_app && cd my_app/
# data/: train.csv, dev.csv, test.csv
nt model train                     # lists the record's columns for data.yaml, then, with roles:
nt model train                     # ... Finished run: <model_run>
# control_configs/control.yaml: your objective
nt control train                   # the latest model run. ... Finished control run: <policy_run>
nt export model  --run_id <model_run> --data data/test.csv --out delivery
nt export policy --run_id <policy_run> --out delivery
```

Every `nt` command is a thin wrapper over a Python function, so the same
runs can be driven from a notebook or a script. MODELING.md and CONTROL.md
show the calls.

## Commands

| Command | Purpose |
|---|---|
| `nt example thermal [dir]` | a working project on synthetic data, zero edits |
| `nt init [dir]` | start an application: `data.yaml`, the hook menus, the configs, `data/` and `AGENTS.md` |
| `nt guide [step]` | the guided flow, one step at a time, and the guides beside it |
| `nt docs [name]` | these references, each hook (`nt docs hooks`), and every config field (`nt docs config`) |
| `nt model train` | train a surrogate |
| `nt model load` | score a trained surrogate on other data |
| `nt data plot` | look at a record one channel at a time, grouped by role |
| `nt data ladder` | before training: which RC network a thermal zone's record supports |
| `nt data thermostat` | before training: the heat a zone's setpoints deliver |
| `nt control train` | train a policy against a trained surrogate |
| `nt control gap` | the best open-loop plan, beside a trained controller |
| `nt export model` | ship a surrogate |
| `nt export policy` | ship a policy |

`nt` alone lists them, and `nt <group> <verb> --help` gives a command's
flags. The `train` commands read a YAML config and share these flags:

```
--config PATH        a YAML file, or a name in the side's config directory
--set key=value      override any field (repeatable)
--print-config       print the resolved config and exit
--dump-config PATH   write the resolved config to a file and exit
--list-configs       list the side's config directory and exit
```

## Development

```bash
git clone https://github.com/pnnl/NeuroTemplate neurotemplate && cd neurotemplate
uv sync --extra dev
uv run pytest tests/
```

The suite is kept small on purpose. It covers what would let a run train
silently on wrong numbers (normalization and windowing, the recurrent and
RC updates, loss alignment, gradient plumbing, head round trips,
export agreement), plus the diagnostics a reader acts on and
one end-to-end pass per command sequence. It does not test report
formatting, error-message wording or plots.

NeuroTemplate runs on PyTorch alone. Its `Node`, `System`, `Problem`,
`Trainer`, blocks and losses are in `neurotemplate/nm`: a trimmed copy of
Neuromancer's, under Neuromancer's BSD-2 license, kept until the library's
own rollout and training loop replace it. Don't grow it.
