# NeuroTemplate

Learn a dynamics model of a system from its time-series record, train a
controller against that model, and ship either one as a small package that
runs without this library. Built on
[Neuromancer](https://github.com/pnnl/neuromancer).

```bash
uv init my_app && cd my_app
uv add "NeuroTemplate @ git+https://github.com/pnnl/NeuroTemplate"
uv run nt init                     # data.yaml, hook menus, configs, data/, guides/ (a minute)
uv run python guides/make_data.py  # a practice record, to try the guides on first
# then open guides/1-first-model.md
```

That installs the tip of `main`, which is kept stable to develop against; `uv.lock`
keeps the application on that commit until `uv lock --upgrade-package neurotemplate`.
[INSTALL.md](src/neurotemplate/docs/INSTALL.md) has the details. The command is `nt`; the
Python package is `neurotemplate`, so `import neurotemplate`, not `import nt`.

**Start with `guides/1-first-model.md`**, which `nt init` writes into your
project ([the same files](src/neurotemplate/template/guides/)). Each guide is a
few minutes of steps for one thing: a first model, an RC network for a zone, a
thermostat, several zones, a first controller, an export, physics checks, a
controller that pays for energy. The references are `nt docs`, the same text as
[src/neurotemplate/docs/](src/neurotemplate/docs/).
