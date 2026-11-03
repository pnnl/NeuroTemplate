# A NeuroTemplate application

`nt init` wrote this directory. The library (`nt`) learns a dynamics model of a
plant from its time-series record, trains a controller against that model, and
exports either one as a package that runs without the library.

**Start with `nt guide`** (`uv run nt guide` if the environment is not active). It
is the flow in five steps; do them in order and read each when you reach it.
`nt docs` lists the references the steps point into.

## What is here

| file | holds |
|---|---|
| `nt_model.py` | the modeling hooks: column roles, blocks, heads, physics checks |
| `nt_control.py` | the control hooks; only for training a controller |
| `model_configs/modeling.yaml` | every modeling field, annotated |
| `model_configs/debug.yaml` | a two-epoch smoke run |
| `control_configs/` | the same pair for control |
| `make_references.py` | writes a reference file (band, tariff) per plant file |
| `data/` | the record, split in time order into `train.csv`, `dev.csv`, `test.csv` |

The hook file names and the two config directory names are fixed: `nt` imports and
resolves them by name from the directory it runs in.

## Ask the human

The data cannot say these; ask before the first real run:

- which columns a controller will write (`U`), and which are known ahead of time
  without the plant, from a schedule or a forecast (`D`);
- what the controller is for: the band to hold, what costs money, at what tariff.

## Working rules

- **The hook docstrings in this copy are the specification** for the installed
  version. Read each before filling it in; do not write hooks from memory of
  another version.
- **The examples are syntax.** Column names and numbers in comments come from some
  other plant, and gains do not port. Where a value has to be sized, the comment
  beside it says how.
- **Smoke-run first**, with each side's `debug.yaml`. `nt model train --config
  <file> --print-config` checks a config without running anything.
- **One change per run, a numbered config per run** (`model_configs/m01.yaml`,
  `m02.yaml`, ...). Two to four model iterations, each decided from the
  reportcard, is usually the useful budget. Change structure and priors before
  width.
- **Read every result against something**: the previous run, the rule,
  persistence. Expect a couple of attempts before a number means what you think.
- **Keep each run id.** A training run ends with `Finished run: <run_id>`; the
  control side finds its model by it (`--set sysid_run_id=<run_id>`).
- **A trained run reloads with the hooks file it trained with.** Editing a hook
  changes the next run, never a finished one.
- **Data prep is yours.** The loader refuses a gap or a ragged timestamp rather
  than repairing it; `neurotemplate.prep` has tools to fix the record.
- **If the library raises an error you cannot trace to the hooks or a config,
  report it** to the human. Do not patch the installed package.
