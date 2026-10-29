# Application template

Copy this directory's contents to the root of your application repo:

```bash
cp -r /path/to/neurotemplate/template/. my_app/
```

You get:

```
my_app/
├── modeling_user_defined.py    # modeling hooks — fill these in
├── control_user_defined.py     # control hooks — only if you train policies
├── make_references.py          # writes a reference file (band, tariff) per plant file
├── model_configs/
│   ├── modeling.yaml           # every ModelingConfig field, annotated
│   └── debug.yaml              # 2-epoch smoke run
├── control_configs/
│   ├── control.yaml            # every ControlConfig field, annotated
│   └── debug.yaml
└── data/                       # your CSVs; whatever the YAML paths point at
```

The two hook filenames and the two config directory names are load-bearing —
the library imports and resolves them by name, relative to the directory you
run `nt` from. (`user_hooks_path` in the YAML relocates the hook files if you
want them elsewhere; the config directories are fixed.)

Add the dependency to your `pyproject.toml` and `uv sync` — see
[INSTALL.md](../docs/INSTALL.md) — and `nt` lands on your PATH.

## Working with it

- **Fill the hooks in place, and read each docstring in your copy.** Each
  docstring is that hook's specification for the version you have
  installed; a hook file rewritten from memory of another version gets
  import paths and signatures wrong.
- **The examples are syntax.** Column names and numbers in comments come
  from some plant that is not yours, and gains in particular do not port.
  Where a value has to be sized, the comment beside it says how.
- **Smoke-run first** with each side's `debug.yaml`. A wrong column name or
  unset path costs seconds there and an hour in a real run.
- **One change per run, a numbered config per run.** Two to four model
  iterations, each decided from the reportcard, is usually the useful
  budget. Change structure and priors before width.
- **Expect some trial and error.** The readouts below are instruments:
  read each against something — the previous run, the rule, persistence —
  and expect a couple of attempts before a number means what you think.
- **If the library itself raises an error you cannot trace to your hooks
  or config, report it** rather than patching the installed package.

## Fill-in order

1. **Column roles** — `Y_COLUMNS` / `U_COLUMNS` / `D_COLUMNS` in
   `modeling_user_defined.py`. `D` is only what you could know without the
   plant (weather, schedules): a policy is fed `D` from a forecast, and
   nothing forecasts a meter. Put in `Y` only what a controller or its
   objective needs. Drop any column constant in the training split
   (training warns about each).
2. **Data paths** in `model_configs/*.yaml`, then a smoke run:
   `nt model train --config model_configs/debug.yaml`. If the record has
   gaps or a ragged timestamp, the loader refuses it; `neurotemplate.prep`
   has tools to fix that yourself.
3. **Check the data covers where the controller will go.** A surrogate
   only knows the inputs its data visited, and the policy's actuation box
   defaults to that range. If the targets you will hold need commands the
   data never issued, collect data that does before anything else.
4. **Model structure** — the decisions that matter more than any
   hyperparameter, all in `block_mapping`:
   - *Family.* `2r2c` is a grey-box thermal network, stable under any input
     and with storage you can read off; try it for a temperature a
     controller will plan against. The black boxes (`nssm`, `mlp`, `gru`,
     `lssm`) often fit better and are fine for prediction you won't steer
     by.
   - *Blocks.* If the plant is several parts that interact mainly through
     shared inputs, give each its own sub-model.
   - *Declared columns.* Say which column is a setpoint (`demand`), a
     delivered power or irradiance (`linear`), the outdoor temperature
     (`ambient`), and the model gets that physics instead of having to
     learn it.
5. **Physics checks** — `synth_cfg`. A few lines; the reportcard then tests
   the model in states the data never visits, which is where a controller
   will take it.
6. **`nt model train`** for real. Note the `Finished run: <run_id>` line —
   that id is how the control side finds the surrogate. Before comparing
   two model configs, set `ensemble_seeds` to a few: one seed is one draw.
7. **Control** — a reference file per plant file (`python
   make_references.py --help`), the `losses` and `gains` in
   `control_configs/control.yaml` naming your columns, and the meters the
   objective prices in `power_channels`. Then
   `nt control train --set sysid_run_id=<run_id>`.

Check any config without running anything:

```bash
nt model train --config model_configs/modeling.yaml --print-config
```

## A controller, one step at a time

Each run is read against the one before it.

1. **The rule, scored on the model** — `reference_control` with every
   `residual_delta` box `[0, 0]`, the band terms only. It is the cheapest
   check of the model there is: if the model says a rule cannot hold a band
   the plant holds easily, the model is wrong about the actuators, and every
   policy trained on it will optimise against that error.
2. **The band terms only**, the policy free inside `actuation_bounds`. This
   should not come out much worse than the rule. If it does, a model
   iteration is worth more than any gain.
3. **The full objective** — cost, peak, whatever you are after.
4. **Its blind twin** — the same config with the price gain at 0 and
   nothing else changed (keep the price references so the draws match).
   A cost number alone is mostly a fact about the test week; the
   difference from the twin is the result.

## Optional control capabilities

Each is off until you ask for it — see [CONTROL.md](../docs/CONTROL.md).

| Want | Implement | Then declare |
|---|---|---|
| an energy objective, and the energy section of the reportcard | `power_channels` (or a proxy in `power_metric`) | a `{column: power, norm: 1}` row |
| price-driven load shifting | `power_channels`, and a tariff column in the reference files (`make_references.py --price`) | a `{column: power, norm: 1, weight: <tariff>}` row, `reference_scale`, a `future` on the tariff in `policy_window` |
| peak shaving | `power_channels` | a `{column: power, norm: 4}` row |
| a small correction to a trusted rule, per U column | `reference_control` | `residual_delta` |
| the rule itself, scored | `reference_control` | `residual_delta` with every box `[0, 0]` |

## Reading a run

- **Sys-id:** `reportcard.md` in the run's artifacts — the physics checks,
  rollout stability, and (under `2r2c`) the time constants and declared
  branches in physical units. The `Finished run:` lines and
  `test_metrics.json` give per-channel test errors in physical units;
  `test_rmse` in `run_summary.json` is one number pooled over the
  channels, each normalised by its training stats. Test error is measured on your data's excitation, not where a
  controller will drive the plant; the stability and physics sections are
  the better guide to a surrogate for control.
- **Control:** look at the closed-loop plots before any number. Does the
  action come *before* the expensive period and the relief through it?
  Does the policy sit still when nothing is asked of it? One value for the
  whole rollout is saturation, usually the surrogate's doing; a command
  pinned at one edge all day, jitter, or two actuators crossing are each a
  finding. Then `control_reportcard.md`: the objective balance (a term
  near 0% of the trained objective is not doing anything), time pinned at
  each actuator bound, and how much of each residual box was used.
- **Never decide what to ship on the surrogate's band number.** The
  surrogate's violation rate is a fact about the surrogate, and it has been
  wrong by several times in both directions. Compare controllers on the
  same surrogate, against the rule; judge the winner on the plant.
- **Split a saving into offset and shape.** Offset is the mean difference
  from the rule — doing less overall, legitimate but unremarkable. Shape is
  doing it at different times: the difference in the expensive hours minus
  the difference in the cheap ones.

## Things that bite silently

- **Declaring a column changes the physics, not the shapes.** `nu` and every
  column name stay the same, so a wrong declaration shows up only in the
  reportcard's branch lines.
- **The actuation box is the training data's range** by default
  (`actuation_bounds`). If the data never drove an actuator somewhere, the
  policy cannot go there either, and the plots look plausible regardless.
- **`grad_path` and `demand`:** the gradient penalty differentiates a
  `demand` column as the demand feature, so its sign in the mapping is the
  declared `sign`, not the raw setpoint's.
- **A trained run reloads with the hooks file it trained with**, not the
  one in your working directory. Editing a hook changes the next run, never
  a finished one.

## Worked examples

For non-trivial hook bodies — ordinal and bounded output heads, a real
`power_metric`, a rule-based `reference_control` — see
[tests/fixtures/modeling_user_defined.py](../tests/fixtures/modeling_user_defined.py)
and [tests/fixtures/control_user_defined.py](../tests/fixtures/control_user_defined.py).
What each feature does and why is in [MODELING.md](../docs/MODELING.md) and
[CONTROL.md](../docs/CONTROL.md); shipping a trained run is in
[EXPORT.md](../docs/EXPORT.md).
