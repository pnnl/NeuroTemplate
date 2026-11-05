# A NeuroTemplate application

`nt init` wrote this directory. The library (`nt`) learns a dynamics model of a
plant from its time-series record, trains a controller against that model, and
exports either one as a package that runs without the library.

**Start with `nt guide`** (`uv run nt guide` if the environment is not active). It
is the flow in four steps; do them in order and read each when you reach it.
`nt docs` lists everything the steps point into.

## What is here

| file | holds |
|---|---|
| `data.yaml` | where the record is, and which column plays which role |
| `nt_model.py` | a menu of the modeling hooks, all optional (`nt docs hooks`) |
| `nt_control.py` | a menu of the control hooks, all optional |
| `model_configs/modeling.yaml` | settings for a model run; none are needed |
| `control_configs/control.yaml` | the objective, yours to edit |
| `*/debug.yaml` | a two-epoch smoke run per side |
| `data/` | the record, split in time order into `train.csv`, `dev.csv`, `test.csv` |

The hook file names and the two config directory names are fixed: `nt` imports and
resolves them by name from the directory it runs in.

## Ask the human

The data cannot say these; ask before the first real run:

- which columns a controller will write (`U`), and which are known ahead of time
  without the plant, from a schedule or a forecast (`D`);
- what the controller is for: the band to hold, what costs money, at what tariff.

## Working rules

- **`nt docs <hook>` is the specification** for the installed version. Read it before
  writing a hook; do not write hooks from memory of another version.
- **The examples are syntax.** Column names and numbers in them come from some
  other plant, and gains do not port.
- **Smoke-run first**, with each side's `debug.yaml`. `nt model train --print-config`
  checks a config without running anything.
- **Each run prints what you set apart from the defaults it used.** Read it.
- **One change per run, a numbered config per run** (`model_configs/m01.yaml`,
  `m02.yaml`, ...). Two to four model iterations, each decided from the
  reportcard, is usually the useful budget. Change structure and priors before
  width.
- **Read every result against something**: the previous run, the best plan,
  persistence. Expect a couple of attempts before a number means what you think.
- **A control run takes the latest model run** unless `sysid_run_id` names another.
  Keep each run id.
- **A trained run reloads with the hooks file it trained with.** Editing a hook
  changes the next run, never a finished one.
- **Data prep is yours.** The loader refuses a gap or a ragged timestamp rather
  than repairing it; `neurotemplate.prep` has tools to fix the record. Deriving or
  shifting a column belongs in your prep script.
- **If the library raises an error you cannot trace to the hooks or a config,
  report it** to the human. Do not patch the installed package.
