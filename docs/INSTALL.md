# Installing NeuroTemplate

NeuroTemplate is installed as a dependency of an application repo, which
holds your data, hook files and configs. New library features arrive by
bumping the version the application pins.

It needs Python 3.10 or newer. These instructions use
[uv](https://docs.astral.sh/uv/); plain pip works too.

## As a dependency

In the application's `pyproject.toml`:

```toml
[project]
dependencies = [
    "NeuroTemplate @ git+ssh://git@<host>/<owner>/neurotemplate.git@<tag>",
]
```

Then `uv sync`. That puts one command, `nt`, on the environment's `PATH`.

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
cp -r /path/to/neurotemplate/template/. my_app/
```

This gives you both hook files, both config directories with every field
annotated, and a smoke-run config per side.
[template/README.md](../template/README.md) is the fill-in order.

Run `nt` from the application's root. The library finds
`modeling_user_defined.py` and `control_user_defined.py` there, or wherever
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
