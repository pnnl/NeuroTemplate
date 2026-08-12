# Plan: let the dynamics model consume derived control features

## Context

The sys-id surrogate consumes raw control values — a cooling setpoint at 24.5 °C arrives
as 24.5 °C. But a thermostat acts on the *error*, not the level. That mismatch costs
three things:

1. **Wasted capacity.** The model must learn a 2-D `f(zone, setpoint)` when the true
   dependence is on the 1-D quantity `zone − setpoint`, gated at zero.
2. **A shortcut.** Setpoint *schedules* correlate with occupancy and time of day, so the
   model can fit the schedule instead of the physics and still score well.
3. **An unusable gradient prior.** `∂zone/∂setpoint` is 0 wherever the setpoint sits on
   the inactive side, so it has no universal sign — which is why `u.json` in
   `modelica_rtu_mall` is a 0-byte file and `yud.json` carries no setpoint edges. The U
   half of the gradient-sign penalty has never been usable. `∂zone/∂relu(zone − setpoint)`
   **does** have a universal sign.

**The controller keeps commanding raw setpoints.** Only the model's view changes.

**Scope decision:** the library ships the *mechanism*, not an HVAC abstraction. An
`ErrorFeature(state, control, active_when=...)` spec would bake in "one control, one
state, one-sided clamp" and could not express a two-sided deadband, an unclamped error, or
a feature over two states — and the README lists "scrub remaining HVAC vocabulary from
library code" as open Phase B work. The app supplies the transform; the library owns the
plumbing.

---

## First: why the derived value cannot be a precomputed dataframe column

This is the one option that is simply wrong, and the reason is autoregression, not
Neuromancer. During a sys-id rollout `yn` is the model's *own prediction*, which drifts
from the recorded `Y`. A column computed from recorded `Y` therefore leaks ground truth
into the rollout:

```
sys-id rollout, in_steps = 3

index            0      1      2  |   3      4      5
                 ---- seeded ---- | ---- predicted ----
U  (setpoint)  24.0   24.0   24.0 |  24.0   24.0   24.0
Y  (recorded)  23.5   23.7   23.9 |  24.1   24.4   24.8
yn (predicted) 23.5   23.7   23.9 |  23.8   23.6   23.2   <-- model drifting cold

PRECOMPUTED   e = relu(Y_recorded - U)
                  0      0      0 |   0.1    0.4    0.8   <-- "cooling is running"

CORRECT       e = relu(yn_predicted - U)
                  0      0      0 |     0      0      0   <-- its own state says otherwise
```

The precomputed column tells the model cooling is active while its own state says it is
not — so the model can partially invert the demand signal to recover the true zone.
Training loss drops, the test rollout looks excellent, and closed-loop performance
collapses, because at control time there is no recorded zone and the demand *must* come
from the prediction. Classic train/serve skew.

The seeded history window (indices 0–2 above) is the exception: there `yn` **is** the
recorded data, so seeding those entries from the dataframe is correct. That matters for
Option A below.

---

## The real choice: transform Node vs. in-model transform

Both are correct. I previously said a Node "cannot work"; that was wrong. The accurate
constraint is that a Node's output key must be **seeded in the data dict**, exactly as
`U` already is — an unseeded key's buffer starts at rollout step 0 and the dynamics'
`pad_mode="nearest"` silently collapses the whole history window to one repeated value.

### Option A — a transform Node in the graph

```
closed loop, hmax = 3, start_iter = 2

buffer index      0     1     2  |  3     4
                  --seeded--
yn              [ y0    y1    y2 ] ->  y3    y4      dynamics appends
U               [ u0    u1    u2 ] ->  u3    u4      policy appends
U_eff           [ e0    e1 ]       ->  e2    e3   e4  transform appends
                  ^^^^^^^^
                  seeded with start_iter = 2 entries, NOT hmax = 3

iteration i = 2:
   policy      reads yn[..2]        -> appends U     at index 3
   transform   reads yn[2], U[2]    -> appends U_eff at index 2   (len was 2)
   dynamics    reads U_eff[0..2]    <- includes the value just written
```

The asymmetry is the catch: `U_eff` must be seeded with exactly `start_iter` entries while
`U` is seeded with `hmax`. One more and every derived value is one step stale; the rollout
still runs and the numbers still look plausible.

**For:** idiomatic Neuromancer; the demand signal becomes a real trajectory in the output
dict, so it can be plotted and logged — genuinely useful, and in the spirit of the rest of
this branch; the model stays a pure `f(y, u_eff, d)`.

**Against:** seven `nodes[0]` sites assume the dynamics System has exactly one node
(`control/load.py:38` raises outright; `experiment.py:104`, `dynamics.py:453`,
`cli/load.py:186`, `cli/debug_heads.py:191` index it); `U_eff` must be seeded in
`data_utils.get_sw_dict`, `control/scenarios.build_window_dict`, **and** every physics
scenario in `modeling/scenarios.py` that builds its own data dict; `nm_mapping` needs a
new key for the derived columns; and the off-by-one above is a silent-wrong-answer
generator.

### Option B — the transform inside `SSMGrad.forward`

```
Node gathers windows -> yn [b, in_steps*ny], U [b, in_steps*nu_raw], D [...]
        |
        v
SSMGrad.forward
        u_eff = transform( yn.view(b, in_steps, ny),
                           U .view(b, in_steps, nu_raw) )   <- pairing is co-temporal
        compute_output(y, u_eff, d)

no new buffer key, no seeding, no node-order question
```

Works identically in sys-id (`System=[dyn]`) and closed loop (`System=[policy, dyn]`) with
zero extra plumbing. The policy, `STEQuantize`, `actuation_bounds`, the du penalty,
`power_metric`, `SynthSpec.off`/`responds` and every physics scenario keep passing raw U
and are untouched.

**Against:** the transform is invisible in the graph — no `U_eff` trajectory to plot
unless one is added deliberately; `forward` and `compute_output` end up with different
input semantics (raw vs. derived), which needs a loud docstring.

**A useful accident of Option B:** `SampledGradientPenalty` enters at `compute_output`, so
with the transform in `forward` the penalty differentiates w.r.t. the *derived feature* —
which is what makes a U-half mask valid — and the app's module is never `vmap`-traced, so
it carries none of the vmap-safety constraints that forced `GRUBlock._manual_gru` to
exist. Under Option A the same holds for a different reason (the transform is a separate
Node entirely).

**This is the open question.** The rest of the plan is written against Option B and is
mostly shared; the Option A deltas are listed at the end.

---

## Design (Option B)

### Hook contract

```python
# modeling_user_defined.py — required by attribute, stub returns None
def control_transform(nm_mapping, norm_stats) -> Optional[nn.Module]: ...
```

The library enforces three things and inspects nothing else:

- it is an `nn.Module` (so `.to(device)` and `state_dict` recurse)
- `forward(y: [b, in_steps, ny], u: [b, in_steps, nu_raw]) -> [b, in_steps, nu_eff]`
- `feature_names: list[str]`, length `nu_eff`

The time-major/feature-minor flatten is a library implementation detail; the app only ever
sees unflattened tensors.

### Getting the effective width to the factories

Every factory sources `nu` from one expression, `len(nm_mapping['U'])`. So replace the
list, not the signature:

```python
# modeling/dynamics.py
def effective_mapping(nm_mapping, control_transform) -> dict[str, list[str]]
def validate_control_transform(transform, nm_mapping, in_steps) -> list[str]
```

`validate_*` collects every problem and raises once (the `SynthSpec.validate` pattern):
not an `nn.Module`; missing/short/duplicate `feature_names`; a name colliding with a Y or D
column (`get_grad_masks` resolves Y first, so a collision would silently land in the Y
mask); then a **smoke forward on zeros** asserting the output shape — which also catches a
device-stranded constant, since `resolve_device` has already set the default device.

`get_dynamics` gains `control_transform=None`, builds the effective mapping, calls the
factory with it, then `model.attach_control_transform(...)` post-construction, mirroring
how `head_stack` is attached. The GRU works for free: `GRUBlock.__init__` receives
`nu_eff` and builds `_seq_perm` / `_last_step_slices` from `(ny, nu_eff, nd)`.

`model: dummy` must **raise**, not silently disable — `_SetpointPassthrough` slices the U
block out as the literal next state and requires `ny == nu`.

The effective mapping must not leak anywhere else. Its only other consumer is grad masks.

### The one reusable primitive

In `modeling/heads.py`, where the STE idiom already lives and which the template already
imports from. No domain vocabulary:

```python
def straight_through(y_soft, y_hard)      # y_soft + (y_hard - y_soft).detach()
def soft_clamp_min(x, sharpness=10.0)     # relu forward, softplus derivative backward
```

Then have `HeadStack.activate` call `straight_through` — a one-line dedup already pinned by
the head tests.

The backward matters: with a plain `relu`, a policy commanding a setpoint outside the
zone's reachable range gets exactly zero gradient forever. `d/dx softplus(βx) = σ(βx) > 0`
everywhere — the same reason `BoundedHead` rejects a bare ReLU.

### The example transform (app code)

Canonical in `template/modeling_user_defined.py`, working copy in
`tests/fixtures/modeling_user_defined.py`. Three decisions belong in its docstring because
app authors will copy them verbatim:

- **Subtract in physical units.** Under the default `z_score` the zone and setpoint have
  different means and stds, so `y_norm − u_norm` is meaningless and the zero-crossing is
  not at zero. Self-contained; no dependency on `tie_norm_stats`.
- **Normalize the result by scale only, never shift.** A clamped feature has a large atom
  at exactly 0; a z-score shift moves that atom off zero and puts the gate at a threshold.
- **Scale by the widest error the recorded ranges permit**, direction-aware
  (`state_max − control_min`, or the mirror). Maps any in-range pair to `[0, 1]`, so
  `sharpness` is dimensionless and one value works at any plant scale.

Register constants as **persistent buffers**; keep the module parameter-free.

---

## Persistence — the one place this cannot follow the heads precedent

`HeadStack` round-trips **by value**: every head is a library class with a JSON-safe
config, so the library owns the taxonomy. An app module has arbitrary code and constructor
arguments; pickling would tie the checkpoint to an importable module path and a torch
version. So this round-trips **by contract surface**:

- `pipeline.py` logs `control_features.json` = `{"feature_names": [...], "class": "..."}`.
- `cli/load.py` gains `_resolve_control_transform` with **no silent fallback** — a
  deliberate divergence from `_resolve_head_stack`, which warns and falls back. That is a
  guess about an output activation; this would be a guess about the model's *inputs*.

| artifact | hook returns | action |
|---|---|---|
| absent | `None` | `None` — ordinary pre-existing run |
| absent | a module | **raise** — trained on raw values; would change inputs under trained weights |
| present | `None` | **raise** — trained with a transform, hook no longer supplies one |
| present | a module | rebuild, assert `feature_names` matches exactly |

**Residual risk, stated plainly:** a hook whose feature *names* are stable but whose *math*
changed reloads silently wrong. Partial mitigation: if the app registers its derived
constants as **persistent buffers**, they live in the checkpoint under
`control_transform.*` and `load_state_dict` overwrites whatever the rebuilt module
computed — so the checkpoint's numbers win, reload survives `norm_stats` drift, and any
structural change fails strict-load loudly. Only pure formula drift with identical buffer
shapes still slips through.

---

## Gradient penalty (`modeling/gradient_sampling.py`)

Two changes, both required in the same commit as the width change:

- **Sample on-manifold.** Draw raw `u` at `model.nu_raw`, then push it through the
  transform before the Jacobian. Drawing N(0,1) directly for a clamped non-negative feature
  would enforce the sign at points that cannot occur — the same "fires on the wrong half of
  the random-sample space" failure the fixture's own grad docstring warns about. The
  transform runs *outside* `jac_fn`, preserving vmap safety.
- **Assert `mask.shape[-1]` against `model.{ny,nu,nd}`.** Nothing checks this today; the
  penalty slices `j_mat[:, :, -nx:]` with `nx` from the mask, so a width drift silently
  mis-slices.

Pass `effective_mapping(...)` to `get_grad_masks` at all three call sites, so a stale grad
JSON naming the raw setpoint raises `Grad key ... not in Y/U/D` — loud and correct.

**Worth knowing:** the Y half changes meaning. `full_step` differentiates `compute_output`,
downstream of the transform, so the penalty measures `∂y_{k+1}/∂y_k` *at fixed derived
feature* rather than the total derivative. For a thermal plant that is the better claim —
at fixed cooling demand it is +1 (thermal mass), whereas the total derivative mixes in the
cooling response and can go negative. Document it or someone will file it as a bug.

---

## Reporting

`describe_dynamics` adds `nu_raw` and a `control_features` block (names, class,
param_count). The reportcard renders `nu=2 (derived from 1 control column)` on **one
line** — `arch["nu"]` and `len(nm_mapping["U"])` currently agree by construction and would
otherwise silently diverge ~30 lines apart.

`param_count` matters: a transform with trainable parameters **will** be optimised by
`Adam(problem.parameters())`. Visible beats surprising.

All `arch.get(...)`-guarded, emitting nothing when absent —
`tests/golden/reportcard_nssm_baseline.md` is byte-pinned and must not need regenerating.

---

## Files

| File | Change |
|---|---|
| `modeling/heads.py` | `straight_through`, `soft_clamp_min`; `HeadStack.activate` uses the former |
| `modeling/dynamics.py` | `effective_mapping`, `validate_control_transform`, 3 `SSMGrad` members + `forward` hook-in, `get_dynamics` kwarg, dummy guard, `describe_dynamics` |
| `modeling/gradient_sampling.py` | on-manifold `u`, mask-width assert |
| `modeling/reportcard.py` | architecture + data sections, guarded |
| `modeling/experiment.py`, `modeling/pipeline.py` | thread `control_transform` mirroring `head_stack`; log `control_features.json` |
| `modeling/cli/{train,optuna}.py` | call hook after `tie_norm_stats`; effective mapping for grad masks |
| `modeling/cli/load.py` | `_resolve_control_transform` + the four-case table |
| `template/modeling_user_defined.py`, `README.md` | stub + worked example + hook-contract row |
| `tests/fixtures/modeling_user_defined.py` | real implementation for tests to import |

Order: heads → dynamics → reportcard are independently mergeable. experiment/pipeline/CLIs/
grad-penalty must land **together** with dynamics or the widths and masks disagree.

---

## Verification

**Tests** — library plumbing only; the HVAC transform is the fixture, never the subject.

- `test_heads.py`: `soft_clamp_min` forward bit-exact `relu`; backward strictly positive for
  `x < 0` (the dead-zone property the design rests on) and ≈1 for `x ≫ 0`.
- `test_control_transform.py` (new):
  - **layout/pairing** — a recording transform driven through a real `SystemPreview` with
    distinguishable values (`y[b,t,i] = 100*t + i`), asserting time-major recovery and
    co-temporal pairing. Catches a `view`/`reshape` transposition.
  - **width propagation** parametrized over `mlp`/`nssm`/`lssm`/`gru` with `nu_eff ≠
    nu_raw`; for GRU assert `_seq_perm` length against the effective width.
  - **passthrough identity** — a transform returning `u` unchanged reproduces the
    no-transform model at the same seed.
  - `dummy` rejected; every contract violation raises with all problems listed at once.
  - reload: names match ⇒ rebuild + `load_state_dict`; any artifact/hook mismatch ⇒ raise
    with both lists in the message.
- `test_closed_loop.py`: policy + dynamics with a transform — `out["U"].shape[-1]` still raw
  width, **and** a policy initialised entirely on the inactive side of the clamp still
  receives non-zero gradient. The end-to-end version of "no dead zone".
- `test_reportcard.py`: golden unchanged (regression guard on the guarding).
- `test_modeling_diagnostics_e2e.py`: `evaluate_and_plot` end-to-end with the fixture
  transform; both name lists render.

**End-to-end run** — the smoke config from this session's verification:

```bash
cd <scratch>/smoke        # model: mlp, 1 epoch, excited/train.csv
uv run nt model train --config model_configs/modeling.yaml
uv run nt model load --run_id <id>
```

Confirm four physics verdicts still print, both `nu` numbers render on one line, and the
reload rebuilds without a width error.

**Empirical A/B before believing any of it.** The premise is a hypothesis about inductive
bias, and this branch's standard is that unvalidated machinery is worse than none. Train
two surrogates on `modelica_rtu_mall/data/excited` differing *only* in whether
`control_transform` returns `None`, then compare:

- test RMSE / NRMSE on the zone channel (does fit hold up?)
- the four physics verdicts, especially **step response** and **reachability** — the
  hypothesis predicts these improve most, being the out-of-distribution ones
- rollout stability over a long horizon
- closed-loop violation rate from `nt control train` against each

If the transformed model does not win on the physics checks, drop the idea rather than
tune it.

---

## Option A deltas

If the Node is preferred, the hook, the STE primitive, the example transform, persistence
and the A/B are all unchanged. What changes:

- `get_dynamics` returns a **two-node** System; fix the seven `nodes[0]` sites (index by
  name or `nodes[-1]`), including the explicit raise in `control/load.py:38`.
- Seed `U_eff` with exactly `start_iter` entries in `data_utils.get_sw_dict`,
  `control/scenarios.build_window_dict`, and each `_rollout_data` in `modeling/scenarios.py`
  — and pin that off-by-one with a dedicated test, because it fails silently.
- `nm_mapping` gains a derived-columns key; decide whether it is persisted in
  `nm_mapping.json` (checkpoint shape contract) or derived.
- Gain: `U_eff` lands in the problem output dict, so add it to the closed-loop plots and the
  reportcard — the demand signal becomes inspectable, which is worth real money on a plant.

---

## Open questions

1. **Option A (Node) or Option B (in-model)?** Node is idiomatic and makes the demand
   signal inspectable; in-model has near-zero blast radius and no off-by-one. I lean B for
   the first cut and adding an explicit demand-logging path later if the signal proves worth
   watching — but the inspectability argument for A is real and this branch has consistently
   valued surfacing things.
2. **Persistence.** Recommendation above (contract surface + persistent buffers). The
   alternative is requiring app modules to implement `to_config`/`from_config` like heads,
   making checkpoints fully self-describing at the cost of per-app boilerplate and a class
   registry.
3. **Should the A/B run before the full implementation?** A throwaway branch hacking the
   transform straight into `SSMGrad.forward` with hard-coded column indices would answer
   "does this help on our data?" in an afternoon, before committing to ten files of
   plumbing. I lean yes.
