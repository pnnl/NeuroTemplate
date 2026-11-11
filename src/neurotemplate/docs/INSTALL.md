# Installing NeuroTemplate

NeuroTemplate is installed as a dependency of an application repo, which
holds your data, hook files and configs. The library is installed from the
tip of `main` on [GitHub](https://github.com/pnnl/NeuroTemplate), which is
kept stable to develop against.

It needs Python 3.10 or newer. These instructions use
[uv](https://docs.astral.sh/uv/); plain pip works too.

## As a dependency

From an empty directory:

```bash
uv init my_app && cd my_app
uv add "NeuroTemplate @ git+https://github.com/pnnl/NeuroTemplate"
uv run nt init
```

`uv add` puts one command, `nt`, on the environment's `PATH`; `nt init` writes
the application (below).

**The application stays on the commit it installed.** `uv.lock` records the
commit of `main` that `uv add` resolved, so every `uv sync` installs the same
library until you move it. To take the current `main`:

```bash
uv lock --upgrade-package neurotemplate && uv sync
```

A trained run keeps a copy of its hooks file but not of the library:
reloading it after an upgrade runs the new library, so upgrade between
comparisons, not in the middle of one.

**Neuromancer is pinned to one git commit.** If the application lists its
own Neuromancer source, such as a local checkout, point it at the same
commit or drop it. Otherwise `uv sync` refuses with conflicting URLs for
`neuromancer`.

**Importing the library sets `NUMBA_DISABLE_JIT=1`** unless the environment
sets it. Neuromancer imports pyts, which compiles numba kernels at import
(20-60 s) for a plot this library never draws. Your own numba code in the same
process then runs uncompiled; `NUMBA_DISABLE_JIT=0` keeps the JIT.

### Optional extras

| Extra | Adds | For |
|---|---|---|
| `prep` | statsmodels | `neurotemplate.prep.smooth`, the gap smoother ([MODELING.md](MODELING.md#preparing-the-record)) |
| `dev` | pytest, `prep` | working on the library itself |

Ask for one as `"NeuroTemplate[prep] @ git+https://github.com/pnnl/NeuroTemplate"`.

### Against a local checkout

To develop the library and an application together, clone it beside the
application:

```bash
git clone https://github.com/pnnl/NeuroTemplate neurotemplate
```

and point the application's dependency at the editable checkout:

```toml
[tool.uv.sources]
NeuroTemplate = { path = "../neurotemplate", editable = true }
```

## Setting up an application

```bash
nt init            # or: nt init my_app
```

This reads nothing and writes `data.yaml` with its paths and no columns, both hook menus,
both config directories, a smoke-run config per side, an empty `data/`, and `AGENTS.md` (with a
`CLAUDE.md` pointing at it) for an agent working in the directory. It writes nothing if any of
those files exists. Put the record in `data/` as `train.csv`, `dev.csv` and `test.csv`, split in
time order; `nt model train` then lists its columns, ready to paste into `data.yaml`. `nt guide setup`
has the rest. To see it all work first, `nt example thermal` writes a project on synthetic data.

Run `nt` from the application's root. The library finds
`nt_model.py` and `nt_control.py` there, or wherever
a config's `user_hooks_path` points.

**A trained run keeps a copy of its hooks file.** Reloading a run, to score
it, to train a policy on it or to export it, imports that copy, never the
file in your working directory. A hook such as `power_metric` has no
weights, so editing it after training would otherwise silently change what the
trained run is. The copy is a single
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
onnx2c, built from upstream, on the machine that exports;
[EXPORT.md](EXPORT.md#the-c-format-needs-onnx2c) has the steps.
