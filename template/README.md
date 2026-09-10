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

## Fill-in order

1. **`modeling_user_defined.py` → `Y_COLUMNS` / `U_COLUMNS` / `D_COLUMNS`.**
   Nothing runs until every column has a role. The other modeling hooks have
   working defaults; come back to them once training works.
2. **`model_configs/modeling.yaml` → the three data paths.** Then
   `nt model train --config model_configs/debug.yaml` for a 2-epoch smoke run.
3. **`nt model train`** for real. Note the `Finished run: <run_id>` line — that
   id is how the control side finds the surrogate.
4. **`control_user_defined.py` → `REGULATED_COLUMNS`**, and
   **`control_configs/control.yaml` → the four `band_*` fields**, which are in
   the physical units of the controlled channel.
5. **`nt control train --set sysid_run_id=<run_id>`.**

Everything above gets you a comfort-band controller. Three optional
capabilities sit on top, each gated on one hook, each off until you implement
it — see [the framework README](../README.md#two-optional-control-capabilities)
for the step-by-step:

| Want | Implement | Then set |
|---|---|---|
| an energy objective, and the energy section of the reportcard | `power_metric` + `POWER_METRIC_IMPLEMENTED` | `energy_weight` |
| price-driven load shifting | `power_metric`, plus `train/dev/test_price_path` + `price_scale` in control.yaml | `cost_weight`, `price_preview` |
| peak shaving | `power_metric` | `peak_weight`, `peak_q` |
| a rule-based baseline, and the learned-vs-rule verdict | `reference_control` | *(nothing)* |
| a policy bounded to that rule | `reference_control` | `residual_delta` |

Sanity-check any config without running anything:

```bash
nt model train --config model_configs/modeling.yaml --print-config
```

## Things that bite

- **`energy_weight` stays `0.0`** until you implement `power_metric` and set
  `POWER_METRIC_IMPLEMENTED = True`. A positive weight without it fails at
  problem build, by design. Same for `cost_weight` and `peak_weight`, which
  are two more reductions of that same power signal, and `cost_weight`
  additionally needs `train/dev/test_price_path` + `price_scale` set.
- **`preview_config` is commented out** in `control_user_defined.py` on
  purpose. If the function merely exists it replaces `cfg.d_preview`
  entirely, so an empty stub would silently disable a preview set in YAML.
  Leave it commented, or define it fully.
- **The price paths are unset by default for the same reason** — setting them
  adds a tariff to every rollout and a price input to the policy. Point them
  at a real historical price series rather than a time-of-use table keyed on
  hour-of-day: a tariff that is a function of the clock is one the policy will
  learn *as* the clock, since the disturbances already carry that signal.
- **`cost_weight` without `price_preview` cannot pre-cool.** Pre-cooling is an
  action taken *before* the expensive block, so a policy whose only price
  input is the current one can react but never shift. The reportcard flags
  this combination, but it is easier to just set it: 48 steps = 4 h on
  5-minute data.
- **`residual_delta` must be at least one `setpoint_step`** if quantization is
  on. Below that every residual snaps back to zero and you train for an hour
  to reproduce the rule; the library rejects it rather than let that happen.
- **Define `reference_control` or `build_baseline_policy`, never both.** Two
  copies of one rule is two places for a cooling inset to drift apart, and the
  inset is what decides whether the rule works at all.

## Worked examples

The template bodies are deliberately minimal. For non-trivial hook
implementations — ordinal and bounded output heads, a real `power_metric`, a
rule-based `reference_control` — read
[tests/fixtures/modeling_user_defined.py](../tests/fixtures/modeling_user_defined.py)
and [tests/fixtures/control_user_defined.py](../tests/fixtures/control_user_defined.py).
The hook contract tables in the [framework README](../README.md#hook-contracts)
are canonical.
