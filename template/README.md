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

Add the dependency to your `pyproject.toml` and `uv sync` — see the install
section of the [framework README](../README.md) — and `nt` lands on your PATH.

Each hook's docstring is its specification. Read them in your copy rather
than from memory of another version.

## Fill-in order

1. **Column roles** — `Y_COLUMNS` / `U_COLUMNS` / `D_COLUMNS` in
   `modeling_user_defined.py`. `D` is only what you could know without the
   plant (weather, schedules): a policy is fed `D` from a forecast, and
   nothing forecasts a meter. Put in `Y` only what you need predicted.
2. **Data paths** in `model_configs/*.yaml`, then a smoke run:
   `nt model train --config model_configs/debug.yaml`.
3. **Model structure** — the decisions that matter more than any
   hyperparameter:
   - *Family* (`model` in the YAML). `2r2c` is a grey-box thermal network,
     stable under any input and with storage you can read off; it is the
     family for a temperature a controller will plan against. The black
     boxes (`nssm`, `mlp`, `gru`, `lssm`) often fit better and are fine for
     prediction you won't steer by.
   - *Blocks* (`block_mapping`). If the plant is several parts that
     interact mainly through shared inputs — several zones, a zone and a
     meter — give each its own sub-model, and its own family if it wants
     one.
   - *Declared columns* (`block_mapping` again). Say which column is a
     setpoint (`demand`), a delivered power or irradiance (`linear`), the
     outdoor temperature (`ambient`), and the model gets that physics
     instead of having to learn it.
4. **Physics checks** — `synth_cfg`. A few lines; the reportcard then tests
   the model in states the data never visits, which is where a controller
   will take it.
5. **`nt model train`** for real. Note the `Finished run: <run_id>` line —
   that id is how the control side finds the surrogate.
6. **Control** — `REGULATED_COLUMNS` in `control_user_defined.py`, the band
   in `control_configs/control.yaml` (physical units), and ideally
   `reference_control`: a simple rule the reportcard scores the learned
   policy against. Then
   `nt control train --set sysid_run_id=<run_id>`.

Check any config without running anything:

```bash
nt model train --config model_configs/modeling.yaml --print-config
```

## Optional control capabilities

Each is gated on a hook and off until you implement it — see
[the framework README](../README.md#two-optional-control-capabilities).

| Want | Implement | Then set |
|---|---|---|
| an energy objective, and the energy section of the reportcard | `power_metric` + `POWER_METRIC_IMPLEMENTED` | `energy_weight` |
| price-driven load shifting | `power_metric`, plus the TARIFF fields in control.yaml | `cost_weight`, `price_preview` |
| peak shaving | `power_metric` | `peak_weight`, `peak_q` |
| a rule-based baseline on the reportcard | `reference_control` | *(nothing)* |
| a policy bounded to that rule | `reference_control` | `residual_delta` |

## Reading a run

- **Sys-id:** `reportcard.md` in the run's artifacts — the physics checks,
  rollout stability, and (under `2r2c`) the time constants and declared
  branches in physical units. `test_metrics.json` has per-channel errors in
  physical units; the headline `test_rmse` is one number in normalized space.
- **Control:** `control_reportcard.md` — the objective balance (a term near
  0% of the trained objective is not doing anything), time pinned at each
  actuator bound, and the learned-vs-rule comparison if you defined a rule.
  Look at the closed-loop plots before the numbers.

## Things that bite silently

- **Declaring a column changes the physics, not the shapes.** `nu` and every
  column name stay the same, so a wrong declaration shows up only in the
  reportcard's branch lines.
- **The actuation box is the training data's range** by default
  (`actuation_bounds`). If the data never drove an actuator somewhere, the
  policy cannot go there either.
- **`preview_config` replaces `d_preview` if it merely exists**, so it is
  commented out in the template. Define it fully or not at all.
- **`grad_path` and `demand`:** the gradient penalty differentiates a
  `demand` column as the demand feature, so its sign in the mapping is the
  declared `sign`, not the raw setpoint's.

## Worked examples

For non-trivial hook bodies — ordinal and bounded output heads, a real
`power_metric`, a rule-based `reference_control` — see
[tests/fixtures/modeling_user_defined.py](../tests/fixtures/modeling_user_defined.py)
and [tests/fixtures/control_user_defined.py](../tests/fixtures/control_user_defined.py).
The hook contract tables in the [framework README](../README.md#hook-contracts)
are canonical.
