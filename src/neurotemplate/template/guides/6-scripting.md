# From a script

Every command in these guides is also a Python call, so a whole session can
live in one file. That pays once you repeat yourself: the same steps every
week, a setting tried at several values, a record per site. This guide
needs only a trained model (`1-first-model.md`).

Run the script from this folder, as you run the commands: the paths in
`data.yaml` and the configs are read from here, and the runs land in the
same `mlflow ui`.

1. **The commands, as they are.** `nt` is a function. Give it what you would
   type after `nt`, one word per item:

   ```python
   from neurotemplate.cli import main as nt

   nt(["model", "train", "--set", "progress=log"])
   nt(["model", "train", "--config", "custom"])
   ```

   Save it as `run.py` and run it:

   ```
   uv run python run.py
   ```

   Each line does what the command does and prints the same. When something
   is wrong it stops the script with the same message.

2. **Use the results.** To act on a run in the script, call training
   directly. It hands back the run. Here, three learning rates, each scored
   on test:

   ```python
   from neurotemplate.common.config import load_config
   from neurotemplate.metrics.rollout import skill
   from neurotemplate.modeling.config import ModelingConfig
   from neurotemplate.modeling.train import train_model

   for lr in (0.001, 0.003, 0.01):
       cfg = load_config("model_configs/custom.yaml", ModelingConfig,
                         {"lr": lr})
       run = train_model(cfg)
       res = run.test_results
       for m, p in zip(res.metrics, res.persistence):
           print(f"lr {lr}: {m.name} rmse {m.rmse:.3f}, "
                 f"skill {skill(m.rmse, p.rmse):.2f}  ({run.run_id})")
   ```

   `load_config` is `--config`, given as a path, and its dict is `--set`. A
   setting in neither takes its default, as on the command line. Each
   `train_model` is one run in `mlflow ui`, plots and reportcard included.
   `run.run_id` names it for anything that takes `--run_id`.

3. **Mix the two.** A command with no call of its own, such as
   `nt data ladder` or `nt export`, goes through `nt` with the run id the
   call returned:

   ```python
   nt(["export", "model", "--run_id", run.run_id,
       "--data", "data/test.csv", "--out", "delivery"])
   ```

The controller works the same way. When you reach `7-first-controller.md`,
its training is one call too, and a script can name the model it trains
on instead of taking the latest:

```python
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.train import train_control

cfg = load_config("control_configs/custom.yaml", ControlConfig,
                  {"sysid_run_id": run.run_id})
policy = train_control(cfg)
print(policy.results.comfort_violation_rate)
```

Good to know:

- **A run records data paths, not data.** A script that pulls a fresh
  record each week should write each pull to its own file. Overwrite one
  in place and older runs point at data they never saw.
- **`nt_model.py` is read once per script.** Edit it while a script runs
  and that script keeps the old version. Run it again.
- **Reloading and rescoring a run**, and the rest of the calls:
  `nt docs modeling` and `nt docs control`, under "From Python".
