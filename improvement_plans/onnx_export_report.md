# ONNX export: what shipped, how to use it, what changed

**Branch:** `onnx_export`, off `main` @ `1c4bbac`, not pushed.
**Date:** 2026-09-22
**Plan:** [onnx_export_plan.md](onnx_export_plan.md)
**Stack:** `torch 2.9.0`, `onnx 1.23.0`, `onnxruntime 1.23.2`, `onnxscript 0.7.2`

## 1. Summary

`nt export model` and `nt export policy` turn a trained surrogate and a
trained DPC policy into one package that runs on `numpy` + `onnxruntime`
alone. The package contains no torch, neuromancer or neurotemplate, and no
hand-written model code. The trained graph is serialised, not reimplemented.

The plan's exit criterion is met, and on real data. A `2r2c + mlp` model with
a demand prestack, the site's binary furnace head and a lifted RC mass was
trained on `modelica_rtu_mall` spring data and exported. It agreed with the
trained `SystemPreview` over the whole 2,004-step test split. A DPC policy
trained against it was exported into the same package and agreed on all
1,925 closed-loop windows it saw. The generated `validate.py` passes both.

| phase | status |
|---|---|
| 1 — core: protocol, state lifting, graphs, trace + audit, verify, tests | done |
| 2 — delivery: `model.json`, runtime, `validate.py`, CLI, no pandas | done |
| 3 — migration | partly done: the NSSM is re-exported and verified against its shipped reference; the 2R2C can't be loaded by `main` at all (§6.1); `elexity_v2` itself is untouched |
| 4 — trained policy export | done (`nt export policy`) |
| 5 — C++ single binary | not started (optional; the plan says to prototype first) |

Two design points changed during the work:
- **The runtime API.** You lifted the plan's "keep the public API unchanged"
  constraint, so the runtime is family-neutral and numpy-only.
- **The package layout.** Each artifact now has its own directory, so one
  package holds a model, a policy, or both.

§4 lists every deviation from the plan.

## 2. How to use it

### Export

```bash
# In the application repo, with its hooks and MLflow store:
nt export model  --run_id <sysid run> --data data/test.csv \
                 --out delivery/site --package site --description "Site B1 zones"
nt export policy --run_id <control run> --out delivery/site --package site
```

| flag | meaning |
|---|---|
| `--data` (model only) | a held-out CSV. Verification covers **every row** of it, and it also becomes the shipped fixture |
| `--fixture_steps N` (model) | ship only the first N steps as the fixture. Verification still covers every row |
| `--fixture_windows N` (policy, default 256) | how many of the verified closed-loop windows to ship, evenly spaced |
| `--exporter dynamo\|legacy` | `dynamo` (torch.export, the default) or the deprecated TorchScript tracer, which writes smaller files (§3.3) |
| `--version` | written into the package's `pyproject.toml` |

`nt export policy` needs only the control run id. It reads the control run's
logged config, reloads the frozen surrogate from `sysid_run_id`, rebuilds the
closed loop with the training builders, and loads the checkpoint strictly.
It then verifies on the control config's own test split, using the same
fixed-seed evaluation band that `nt control train` scores against.

The export **refuses to write** in two cases:
- the graphs disagree with the trained model by more than 1e-4 of a
  column's normalisation stretch, anywhere on the split;
- the staged package fails its own `validate` check.

A refused export leaves `--out` untouched. The refusal also reports how far
the trained model's own rollout ran, because a model that diverges on the
split grows float32 round-off with it (§3.4).

### The package

```
delivery/site/
  README.md  pyproject.toml         generated on every export
  validate.py  example.py           written once, then yours: re-exports never overwrite them
  site/runtime.py, __init__.py      the runtime, copied verbatim
  model/   warm_start.onnx  step.onnx  model.json
           reference/ example_input.csv  expected_output.csv  expected_state.csv  REFERENCE.json
  policy/  policy.onnx  policy.json
           reference/ inputs.npz  expected_actions.csv  REFERENCE.json
```

Each export replaces only its own directory (`model/` or `policy/`). Anything
else the project keeps at the top level, such as a `MODEL_CARD.md`, is never
touched.

```python
from site import Model, Policy, check_reference, read_csv

model = Model("model")
state = model.warmup(history)            # {column: array}, or a DataFrame; last history_steps rows
state = model.step(state, inputs)        # {u and d column: value} -> new State (pure)
state.outputs                            # {y column: value}
forecast = model.predict(frame)          # {y column: array}, open loop after the warm-up
outputs, hidden = model.rollout(frame)   # arrays, plus the hidden-state trajectory

policy = Policy("policy")
u = policy.act(y, ymin, ymax, d)         # each a [steps, n] window; price= if it reads a tariff
U = policy.act_batch({"y": ..., ...})    # [batch, steps, n] each -> [batch, n_u]
```

Everything is in physical units on both sides. `policy.json` states each
input's columns, window length and `current_row`. Rows after the current row
are preview: forecast disturbances, the comfort schedule, the tariff.

`python validate.py` reproduces each `reference/` fixture with the local
onnxruntime. It reports output and hidden-state mismatches separately, so a
failure points at warm-up/step or at the output layer.

### Adding a stateful family

Implement two methods on the block, as `RCBlock` and `GRUBlock` now do:

```python
def rollout_state(self) -> Tensor | None: ...          # [batch, width], None when cold
def set_rollout_state(self, state: Tensor | None): ... # None forces a cold start
```

That is the whole contract. `make_reset_init` and the exporter both key off
it (`dynamics.stateful_blocks`).

## 3. Measurements

All measured on macOS arm64, single-threaded.

### 3.1 Agreement: exported package vs the trained model

| case | data | steps / windows | worst output error | hidden state |
|---|---|---|---|---|
| mixed 2r2c + mlp, demand prestack, binary furnace head (rtu_mall `ps_m02` recipe, retrained) | real, spring test | 2,004 steps | 6.4e-6 °C zone; 4.8e-3 W furnace | 4.3e-6 (width 1) |
| DPC policy against it (`ps_c02` recipe, retrained) | real, closed loop | 1,925 windows | 1.9e-6 °C setpoint | — |
| Elexity NSSM, run `56cdba1d` (the shipped delivery's run) | real, `test_300.csv` | 1,428 steps | 2.0e-5 °C | — |
| same NSSM package vs the **old delivery's** `expected_output.csv` | the old 24 h window | 287 × 18 zones | 1.5e-5 °C (tolerance 1e-3) | — |
| 18-zone 2R2C, Elexity geometry (random weights) | synthetic | 288 steps | 8.4e-6 °C | — |
| every family, head type and split in `tests/test_export.py` | synthetic | 56 steps each | within 1e-4 of stretch (asserted) | asserted |

The results are not bit-exact, unlike the plan's §3.1. Its 0.000 was
measured in normalised space. The shipped graphs take physical units, so
every step round-trips the output through `x·stretch + shift` and back in
float32. That costs about 1e-6 of a stretch per step, and it does not grow
for a stable model. The policy residual-and-quantise case, compared on
identical inputs, is exactly 0.

### 3.2 Latency (batch 1 unless stated)

| model | eager torch (same physical wrapper) | ORT, dynamo graph | ORT, legacy graph | `runtime.step` incl. Python |
|---|---|---|---|---|
| 18-zone 2R2C (1.5k params) | 725 µs | 70 µs (10×) | 74 µs | 79 µs |
| Elexity NSSM (30k params) | 288 µs | 65 µs (4.4×) | 120 µs (2.4×) | — |
| Elexity NSSM, batch 64 | 325 µs | 125 µs (2.6×) | 207 µs | — |

The plan found NSSM at batch 64 to be *slower* under ORT (0.5×). That was
the plan's larger 163k-param NSSM, graphed by the legacy exporter. The real,
smaller NSSM graphed by dynamo is faster at batch 64 too. The per-call
Python overhead of the runtime is about 9 µs.

### 3.3 File sizes (`step.onnx`)

| model | dynamo | legacy |
|---|---|---|
| 18-zone 2R2C | 532 KiB | 154 KiB |
| Elexity NSSM | 469 KiB | 279 KiB |

Dynamo stays the default. It is the supported path, it is up to 1.8× faster
here, and a few hundred KiB does not matter for these deliveries.
`--exporter legacy` stays for when size does matter.

### 3.4 A diverging model is refused, and the refusal says why

An untrained 18-zone NSSM rolls out from 23 to 7×10⁵ over 288 steps. The
first step agrees to 1.5e-6, and the error stays at about 1e-6 to 1e-5 of
the rollout's magnitude throughout. So the export is right and the model is
exploding. The absolute check still fails, which is the right outcome for a
model like that. The refusal now names the rollout's reach ("reached N
stretches from its columns' means") so that nobody goes hunting for an
export bug. The real Elexity NSSM, which also grows over its window, passes
comfortably.

### 3.5 Operators

These are the operators observed across both exporters, every family, every
head, the prestack, and every policy family × nonlinearity × wrapper. All
are core ONNX, with no custom ops:

`Add Cast Clip Concat Constant ConstantOfShape Div Equal Erf Exp Expand GRU
Gather GatherND Gemm Greater Identity Max Min Mul Neg Range Reciprocal
ReduceMean ReduceSum Relu Reshape Round ScatterND Shape Sigmoid Slice
Softplus Split Squeeze Sub Tanh Transpose Unsqueeze Where`

`trace.ALLOWED_OPS` is exactly this set. Any other operator fails at export
with its name. The audit caught three real ones during the work: `Erf` from
the default GELU policy nonlinearity, `Round` from setpoint quantisation,
and `Max`/`Min` from the legacy tracer's version of the residual clamp.

## 4. Where the implementation departs from the plan, and why

1. **Window layout: the prototype was wrong for multi-column keys.** The
   plan's §3.5 physical-units wrapper flattened each window with
   `transpose(1, 2)`, which is channel-major. `SystemPreview` (and
   `modeling/windows.py`) lays each key out time-major, with every column of
   a step together. The two agree only when a key has a single column. The
   prototype never saw the bug because it compared the wrapper with itself,
   and its System comparison used 1-column blocks. It fell out immediately
   here because verification runs the real System. `tests/test_export.py`
   uses multi-column keys throughout for this reason.
2. **Dynamo needs `Dim.AUTO`, not a named batch dim.** `nn.GRU` guards on
   `batch != 1`, which a strict named `Dim` refuses. `AUTO` lets the guard
   through, and the graph is still correct at batch 1. That is tested for
   every family.
3. **Runtime API redesigned (your call).**
   - `ZoneModel` / `ZoneState.temperatures` became `Model` / `State.outputs`.
   - `predict` returns `{column: array}` instead of a DataFrame, and pandas
     is gone from the runtime.
   - `State` carries the output window, the input history and the hidden
     state explicitly.
   - Stateless models ship both graphs too, so the runtime never branches
     on model family.
4. **Package layout: one directory per artifact** (`model/`, `policy/`, each
   with its own `reference/`). This replaces the plan's single `data/`, so a
   model and its controller ship together and re-export independently.
   `README.md` and `pyproject.toml` are regenerated from whatever is present.
   `validate.py` and `example.py` are written once. The plan's per-project
   `validate.py` assertions now live below a generic reference check, which
   calls `check_reference()` inside the runtime, so the generic part never
   goes stale.
5. **Tolerances are fractions of each column's normalisation stretch.** That
   gives 1e-4 at export and 1e-3 on the consumer's machine. The plan wrote
   them in °C. Stretch-relative is the same idea for a template whose
   outputs include W, kW and fractions, and for temperatures (stretch ≈
   1–2 °C) it lands where the plan's numbers did. `REFERENCE.json` stores the
   resulting per-column absolute tolerances, so a consumer never needs the
   stretch.
6. **The model fixture is the whole verified split by default.** The plan
   left the fixture window open. Shipping the rollout we verified means one
   rollout covers both graphs, with no second case. `--fixture_steps`
   trims it.
7. **The policy reference is the torch policy on the shipped physical
   inputs.** A quantised policy can round the other way at a grid midpoint
   under float noise. So the check that must be exact (ORT vs torch) runs
   on identical inputs. The test separately checks that the wrapper's
   window layout matches the closed loop's own flattened windows, on >99%
   of commands.
8. **Graphs are stamped with an `export_id`**, and the runtime refuses a
   graph whose id does not match the json beside it. This is also where the
   opset is asserted on load (plan §8), since onnxruntime does not expose
   the opset itself.
9. **Nothing in `elexity_v2` was changed.** `campaigns/export_delivery.py`
   and the two hand-written `model.py` files are not retired (plan step 14).
   That repo is outside this branch, and its 2R2C run can't be re-exported
   anyway (§6.1).

## 5. What changed in the repo

| where | change |
|---|---|
| `modeling/rc.py`, `modeling/dynamics.py` | `rollout_state` / `set_rollout_state` on `RCBlock` and `GRUBlock` (GRU owns its layer-first ↔ batch-first reshape). `reset()` is now `set_rollout_state(None)`. New `stateful_blocks(model)`; `make_reset_init` keys off it instead of probing `fy.reset` |
| `modeling/cli/load.py` | `_config_from_run_params` → public `config_from_run_params(run_id, cls=ModelingConfig)`, so control runs reload too |
| `control/load.py` | `load_trained_policy(run_id)`: rebuilds the closed loop with the training builders and strict-loads `best_model_state_dict.pth` |
| `export/` (new) | `state` `graphs` `trace` `verify` `spec` `dynamics` `policy` `package` `runtime` `cli`, plus `templates/`; about 1,760 lines including templates and docstrings |
| `cli.py` | `nt export model`, `nt export policy` |
| `pyproject.toml` / `uv.lock` | + `onnx`, `onnxruntime<1.24` (1.24 has no cp310 wheels), `onnxscript` (dynamo). Additions only; no existing pin moved |
| `tests/test_export.py` (new, 13 tests) | every family, per-block split, head types, the mixed prestack + ordinal case under dynamo, the baked-state regression (§3.4 of the plan), refuse-to-write, and two policy cases (plain; residual + quantised + priced) |
| `README.md` | command table, typical session, a "Shipping a model" section, package layout, test count |

Suite: 371 passed (was 358), about 15–20 s warm. The first cold run after
install took 75 s, mostly one-off torch.export/onnxscript start-up.

## 6. Open items

### 6.1 Every 2R2C run from before `1c4bbac` is unloadable on `main`

This morning's reparameterisation renamed `RCBlock._theta` to `_log_tau`
(and changed the rate map), and nothing converts old checkpoints. Affected
runs include the delivered Elexity 2R2C (`4ceb297f`, which also predates the
`block_spec.json` artifact) and the rtu_mall prestack runs from 2026-09-17
(`116e4fe9` and its DPC runs). To ship the Elexity 2R2C through the new path
it has to be retrained. I left this alone, per the no-compatibility stance.

### 6.2 Other open items

- **Cross-platform reproducibility is still unmeasured** (plan §8). Every
  number above is macOS arm64. The consumer tolerance keeps its 10×
  headroom until someone runs `validate.py` on x86 Linux.
- **Phase 5 (C++ single binary)** is not started. The runtime contains no
  numeric logic, so a C++ driver only has to move windows, but the
  minimal-build size is still unverified.
- **`elexity_v2`'s MLflow store was written by a newer mlflow (schema
  `6f8d9c3b2a1e`) than this venv's 3.9.0.** `nt export` can't open it
  in-place. I re-exported the NSSM through a scratch driver that reads the
  params from sqlite directly.
- **A diverging model can't be exported over a long split.** That is
  correct, but a user with a legitimately growing model could hit it. The
  refusal says so; `--fixture_steps` doesn't help, because verification
  always covers the whole `--data`. Pass a shorter `--data` if that is the
  real intent.
