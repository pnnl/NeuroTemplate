# Installing NeuroTemplate

NeuroTemplate is installed as a dependency of an application repo, which
holds your data, hook files and configs. New library features arrive by
bumping the version the application pins.

It needs Python 3.10 or newer. These instructions use
[uv](https://docs.astral.sh/uv/); plain pip works too.

## As a dependency

From an empty directory:

```bash
uv init my_app && cd my_app
uv add "NeuroTemplate @ git+ssh://git@<host>/<owner>/neurotemplate.git@<tag>"   # a release tag, e.g. v0.2.0
uv run nt init
```

That puts one command, `nt`, on the environment's `PATH`, and `nt init` writes
the application (below).

**Neuromancer is pinned to one git commit.** If the application lists its
own Neuromancer source, such as a local checkout, point it at the same
commit or drop it. Otherwise `uv sync` refuses with conflicting URLs for
`neuromancer`.

**Numba's JIT is off in any process that imports the library.** Neuromancer
imports pyts, which compiles numba kernels at import (20-60 s) for a plot this
library never draws, so `neurotemplate` sets `NUMBA_DISABLE_JIT=1` unless the
environment already sets it. Your own numba code in the same process then runs
uncompiled; set `NUMBA_DISABLE_JIT=0` to keep the JIT and pay the import.

### Optional extras

| Extra | Adds | For |
|---|---|---|
| `prep` | statsmodels | `neurotemplate.prep.smooth`, the gap smoother ([MODELING.md](MODELING.md#preparing-the-record)) |
| `dev` | pytest, `prep` | working on the library itself |

Ask for one as `"NeuroTemplate[prep] @ git+ssh://..."`.

### Against a local checkout

To develop the library and an application together, point the dependency at
an editable checkout:

```toml
[tool.uv.sources]
NeuroTemplate = { path = "../neurotemplate", editable = true }
```

## Setting up an application

```bash
nt init            # or: nt init my_app
```

This writes both hook files, both config directories with every field
annotated, a smoke-run config per side, `make_references.py`, an empty
`data/`, and `AGENTS.md` (with a `CLAUDE.md` pointing at it) for an agent
working in the directory. It writes nothing if any of those files already
exists. Put the record in `data/`, split in time order into `train.csv`,
`dev.csv` and `test.csv`, the paths the configs name; then `nt guide setup`.

Run `nt` from the application's root. The library finds
`nt_model.py` and `nt_control.py` there, or wherever
a config's `user_hooks_path` points.

**A trained run keeps a copy of its hooks file.** Reloading a run, to score
it, to train a policy on it or to export it, imports that copy, never the
file in your working directory. A hook such as a reference rule or an
actuation bound has no weights, so editing it after training would
otherwise silently change what the trained run is. The copy is a single
file, so training refuses a hooks file that imports a module sitting next
to it. Keep shared code inside the hooks file, or install it as a package.

**Runs go to MLflow**, in the store a config's `location` names:
`sqlite:///mlflow.db` in the working directory by default, with artifacts
under `mlruns/`. Commands that load a run by its id read that same default
store, unless `MLFLOW_TRACKING_URI` (or, from Python,
`mlflow.set_tracking_uri`) points elsewhere.

## Running an exported package

An exported model or policy does not need this library at all. Depending
on how it was exported, it needs numpy and onnxruntime, numpy and torch, or
only a C compiler. See [EXPORT.md](EXPORT.md). The C format also needs
[onnx2c](https://github.com/kraiskil/onnx2c) on the machine that exports.
