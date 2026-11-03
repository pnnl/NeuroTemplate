# NeuroTemplate

Learn a dynamics model of a system from its time-series record, train a
controller against that model, and ship either one as a small package that
runs without this library. Built on
[Neuromancer](https://github.com/pnnl/neuromancer).

```bash
uv init my_app && cd my_app
uv add "NeuroTemplate @ git+ssh://git@tanuki-data.pnnl.gov/buildings-technology/neurotemplate.git@v0.2"
uv run nt init          # hooks, configs, data/, AGENTS.md
# put train.csv, dev.csv, test.csv in data/
uv run nt guide
```

[INSTALL.md](src/neurotemplate/docs/INSTALL.md) has the details.

**Start with `nt guide`** (the same text is [GUIDE.md](GUIDE.md)). It is the whole
flow in five short steps, each a few decisions and the command whose output settles
them, and each step names the reference or the hook to open when you need more.
Guides for one kind of system or one way of working sit beside it: a thermal zone,
a model of your own, MPC. The references are `nt docs`, the same text as
[src/neurotemplate/docs/](src/neurotemplate/docs/).
