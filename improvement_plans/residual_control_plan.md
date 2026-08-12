# Plan: subsume the rule-based controller — residual DPC

## Status: implemented on `curated_v2`

Steps 1 and 2 of the recommended sequence landed. Step 3 (wiring
`evaluations/rbc.py` into `dynamic.py`, plant-side site energy) is in the
evaluator repo and is untouched here. Step 4 is the
[grid-response plan](grid_responsive_objective_plan.md), which landed
alongside.

| Commit | Content |
|---|---|
| `6abfbde` | `reference_control` hook, `_ReferenceRule`, `_ResidualPolicy`, zero-init, `residual_delta` + its two validations, `ResidualMetrics`, the reference-derived baseline, `clears_reference` and the reportcard verdict |
| `bc2ceeb` | integration: a residual policy still inside its box after real training and after quantization |

Shape B as recommended, and the claim that made it the recommendation is now
asserted rather than argued: `test_residual_keeps_the_node_signature_the_deploy_path_calls`
checks a residual policy's Node has the same `input_keys`, `input_map` and
`output_keys` as a plain one. The evaluator needs no edits.

### Six decisions that changed under implementation

**1. δ-saturation is measured as the *realized* residual, not the network's
output.** The plan asked for "the fraction of steps where δ sits at ±Δ". Read
off `MLP_bounds` that would be the pre-quantization, pre-clamp deviation —
which is not what was deployed. Instead `reference_over_horizon` replays the
(memoryless) rule on the recorded trajectory and the metric is `U − u_ref`,
after the grid and after the actuator clamp.

**2. That replay exposed a genuine off-by-one.** The rollout starts at
`start_iter = hmax − 1`, so the command appended at `U[hmax + k]` was chosen
from the signals at index `hmax − 1 + k`. Slicing the inputs at `hmax + k` —
the obvious thing, and what every objective term correctly does for *its* own
quantities — would have compared each command against the *next* step's rule.
On a band that moves, that is a residual that never was. Pinned by a test that
reconstructs the baseline Node's own output from the recorded band.

**3. `build_baseline_policy` survives as a documented escape hatch, and
defining both raises.** The plan said keep it for a rule a memoryless function
cannot express, and delete it if that case never shows up. It is kept and
narrowed to exactly that case (a rule with its own integrator or timer). What
the plan did not specify is what happens when both hooks exist; the answer is a
`ValueError` naming both, because silently preferring one is precisely how two
copies of a cooling inset drift apart. The test fixture migrated to
`reference_control` and no longer defines the other.

**4. The template ships `reference_control` commented out, not working.**
`comfort_band_signal` and `actuation_bounds` ship generic-and-working, so a
working default rule was the obvious symmetry. It is the wrong symmetry: a
band is plant-agnostic and a control rule is not, the template's
`CONTROLLED_COLUMNS` is a stub anyway, and a rule that silently works is a rule
nobody measured. The template carries a complete `bracket_band` implementation
in comments with the two documented ways to get it backwards; the fixture
carries a complete midpoint rule as the worked example.

**5. The decision rule is in-library and stops at the verdict.**
`pipeline.clears_reference` compares mean exceedance, the reportcard prints
either "clears its reference" or "**Ship the reference.**", the CLI prints it,
and MLflow gets `control_clears_reference`. Actually *shipping* the reference
is a submission-side action in another repo — this makes the call, it cannot
enforce it.

**6. `describe_policy` reports the clamp's bounds, not the network's.** For a
residual policy `net.min`/`net.max` are ±Δ, so the actuation-bounds block on
the reportcard would have read as a ±1 °C actuator. It now reads the
`_ResidualPolicy`'s own clamp buffers, and the ±Δ box gets its own line.

### The composition, since both plans landed together

This is the case the plan called "the strongest argument for the design", and
it is now buildable in one config:

```yaml
residual_delta: 1.0      # the rule keeps comfort, within a box you can point at
cost_weight: 0.5         # δ answers the incentive
peak_weight: 0.1
energy_weight: 0.1       # peak alone can be minimised by ADDING off-peak load
train_pred_steps: 288    # a precool-plus-coast cycle does not fit in 48
```

with `preview_config()` returning a `price` width, `reference_control`
implemented, and `price_signal` + `PRICE_SCALE` implemented. `_ReferenceRule`
accepts and ignores inputs beyond its four, so the price preview reaches the
network without touching the rule — which is exactly the division of labour the
plan described. `test_a_residual_policy_trains_and_stays_in_its_box` runs that
combination end to end.

One caveat the plan did not anticipate: adding a price input changes the
policy Node's `input_keys`, so the "zero evaluator changes" guarantee holds for
the residual *alone*, not for residual-plus-price. Price is opt-in and off by
default, so nothing existing is affected — but a price-driven submission needs
the evaluator image rebuilt, on top of the version-pinning risk already noted
below.

### What is deliberately still missing

Risk 5 stands undiminished and unaddressed: once every controller is anchored
to the same rule, an arm comparison measures δ, whose spread will be far
smaller than today's 0.032–0.162 — and that spread already fits inside a 0.243
noise floor. Nothing in this branch improves the readout. That decision was
supposed to be made *before* the residual shipped; it has not been, and any
arm comparison run after this merges will produce null results that are
indistinguishable from noise by construction.

---

## Verdict

**Build it.** It is feasible, it is small (~120 lines of library code), and — the finding that
decides the whole design — **it needs zero changes on the evaluator side**, because the deploy
path rebuilds the controller by calling `build_policy` and then invokes the policy `Node` with
exactly the four signals a memoryless rule needs.

But be precise about *why* it works, because the stated reason ("it is easier for the network
to learn a zero output than an identity output") is only half of it, and the weaker half. The
load-bearing property is that a bounded residual is a **trust region on surrogate error**. The
reports do not show controllers that failed to *represent* the rule; they show controllers that
were optimized and selected against a surrogate whose own held-out score is anti-correlated
with plant fidelity. A ±Δ box caps what that error can cost you. The zero-init is free and
worth having, but it is the garnish.

The honest expectation to set: **δ = 0 (pure rule) will often be the best cell, and that is the
point.** This design converts a 35-out-of-36 loss into a bounded bet with a scoreable floor.
It does not, on its own, make DPC beat RBC.

---

## What the measurements actually say

From `~/container-fleet/reports/narrative-1-surrogate-metrics` and
`.../narrative-2-self-improvement`:

| | |
|---|---|
| bracket RBC vs learned | beats **35 of 36** core cells outright, and the *best* learned policy on 3 of 4 buildings (N1/03 §3) |
| mean comfort violation | RBC **0.050** vs 0.152 for the best learned policy per building; 0.397–0.429 pooled across arms |
| actuation | RBC `mean_abs_du` 0.0007–0.0009 vs a 36-cell mean of **0.147** — ~180× |
| energy | **63 of 63** learned policies use more site energy than the native thermostat (+7% to +8536%) |
| compute | ~20 s per RBC run vs a 456 s mean per agent container |
| the one exception | `n2-curated/books_NYC`, 0.074 vs the rule's 0.120 — bought with 36% more energy |

Three further facts explain *why* the learned side loses, and all three point the same way:

1. **The surrogate is wrong on the plant, and the run cannot tell.** The agent's own held-out
   score over-states plant fidelity in **39/39** cells and is anti-correlated with it (−0.238,
   N1/05). The optimizer's gradient and the model-selection criterion both come from that
   surrogate.
2. **The objective cannot price energy.** `energy_proxy` is the agent's own `power_metric`
   hook: 9 of 36 cells report exactly 0.0 and the other 27 span a factor of 1.5 × 10⁵ (N1/03
   §2). Nothing in the loop penalizes running the equipment flat out.
3. **The measurement floor is 0.243** in the arm mean (N1/03 §1) — larger than every arm effect
   either narrative reports. Nothing in the current loop can *detect* a 0.05-scale improvement
   even if one were produced.

N2 is the clean demonstration that (1) is not fixable from the model side: the instrument
improved in three verified steps, adoption went 0/4 → 4/4 → 4/4, and the controller did not
move. Better model diagnostics do not reach the controller. What *can* reach it is shortening
the distance between the deployed action and a known-good action. That is the residual.

---

## Why the "zero vs identity" argument is the weaker half

The current policy is `MLP_bounds` over a flattened `[yn | ymin | ymax | D]` window
([policy.py:264](../src/neurotemplate/control/policy.py#L264)). The bracket rule is roughly
affine in `ymin`/`ymax` followed by a quantizer — and the catalog already ships a `linear`
family that expresses exactly `u = sat(K x + b)`
([policy.py:204](../src/neurotemplate/control/policy.py#L204)). Representation is not the
binding constraint. The training signal is: it points at whatever the surrogate is wrong about,
and dev loss — the selection criterion — is computed against the same surrogate.

So the residual buys three things, in descending order of importance:

1. **A hard cap on deployed deviation.** Whatever the surrogate hallucinates, the commanded
   setpoint stays within Δ of the rule's. That bounds the cost of (1) above without needing to
   fix it.
2. **An exactly-recoverable, scoreable δ = 0 point** inside the same run, on the same band
   realization and the same windows.
3. **A better starting point.** With `sigmoid_scale`, the bound is
   `(max−min)·σ(x)+min` ([neuromancer/modules/functions.py:18](file:///Users/mich604/neuromancer/src/neuromancer/modules/functions.py#L18)),
   so a **symmetric** box `[−Δ, +Δ]` with a **zero-initialized final layer** gives δ ≡ 0
   *exactly* at initialization — training starts at the RBC. Three lines. The gradient at that
   point is also maximal (σ′(0) = ¼), so it is a responsive start, not a dead one.

---

## Where it goes — the integration shape is forced

`evaluations/submission.py:225` rebuilds the controller with the same
`build_policy(control_cfg, sysid_cfg, nm_mapping, norm_stats)` call training used and loads
weights `strict=True`. `evaluations/dynamic.py:418` then calls the bare policy `Node` with
exactly four pre-windowed keys: `yn`, `ymin`, `ymax`, `D`.

That forces the choice:

**Shape A — the rule as its own Node in the closed loop** (`[ref_node, policy_node, dyn_node]`,
ref writes `U_ref`, policy reads it). Clean in Neuromancer, and it is what the existing
`build_baseline_policy` hook already produces. **Rejected:** the deploy path calls
`policy_node` directly, so this breaks the evaluator and requires a cross-repo change; and
`U_ref` would have to be seeded in the data dict or `pad_mode="nearest"` silently collapses its
history window — the exact trap documented in
[control_transform_plan.md](control_transform_plan.md).

**Shape B — the rule composed inside the policy Node's callable.** A `_ResidualPolicy(base,
reference)` wrapper, mirroring the existing `_QuantizedPolicy` wrapper
([policy.py:145](../src/neurotemplate/control/policy.py#L145)) exactly. The Node's input keys,
`input_map`, and output are unchanged, so **submission.py and dynamic.py need no edits at
all**. This is the recommendation.

### The hook contract

The user's question was whether the rule should be a hook, a reference signal, or a range. The
answer is: a hook that returns the reference, with the range as config, and the library owning
the window convention.

```python
# control_user_defined.py
@beartype
def reference_control(
    yn_now: torch.Tensor,      # [b, ny]     current state, normalized
    ymin_now: torch.Tensor,    # [b, n_ctrl] current band, normalized
    ymax_now: torch.Tensor,    # [b, n_ctrl]
    d_now: torch.Tensor,       # [b, nd]     current disturbances, normalized
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> torch.Tensor:             # [b, nu]     normalized U — the same space the policy emits
```

Three deliberate restrictions:

- **Current step only, unflattened.** The library slices the current step out of the flattened
  windows and hands the hook plain `[b, n]` tensors. The flattening convention
  (`(batch, window, dim).reshape(batch, -1)`, oldest-first — `system.py:468`) is library
  business and must not leak into a use case's rule.
- **Memoryless.** Every RBC in the reports is a memoryless function of band and state, and it
  is what the deploy-side callback can supply without new plumbing. A rule that needs its own
  integrator is out of scope; say so and fail fast rather than adding a state-carrying path.
- **Normalized in, normalized out**, matching every other control hook.

The template implementation is then ~15 lines over the library's existing `bracket_band`
([policy.py:50](../src/neurotemplate/control/policy.py#L50)): denormalize the band, call
`bracket_band(lo, hi, step=cfg.setpoint_step, cool_inset=…, heat_inset=…)`, renormalize. The
two easy mistakes (round toward the band interior; the cooling inset is a gain correction, not
a safety margin) are already documented there and in the template.

### Composition order inside the wrapper

```
δ      = MLP_bounds(window)                  # bounded to [-Δ_n, +Δ_n], last layer zero-init
u      = clamp(quantize(u_ref + δ), umin, umax)
```

with `Δ_n = residual_delta / stretch` per U column, same conversion `setpoint_step` already
uses ([policy.py:280](../src/neurotemplate/control/policy.py#L280)).

Order matters and is easy to get backwards: the reference is *already* grid-snapped and δ is
not, so the **sum** must be snapped, and the clamp must come **last** — quantizing after
clamping can push a value back outside the actuator box.

### Config

One field, in `ControlConfig`:

```python
# Half-width of the residual box, in physical units (°C). None disables the
# residual entirely (today's behaviour). 0.0 emits the reference rule exactly,
# which is a useful floor to train nothing and score.
residual_delta: Optional[float] = None
```

Validation, in `validate()`:

- `residual_delta is not None` requires `control_user_defined.reference_control` — fail fast at
  build time with the hook name, the same shape as the `POWER_METRIC_IMPLEMENTED` check.
- `0 < residual_delta < setpoint_step` while quantization is on is a **silent no-op**: every δ
  snaps back to zero and you get a rule that trained for an hour. Raise, naming both values.
  This is the single most likely way to waste a campaign on this feature.

### DRY: one rule, two consumers

`build_baseline_policy` already exists, already returns a Node emitting normalized U, and is
already scored on the identical band and windows by `evaluate_baseline`
([pipeline.py:306](../src/neurotemplate/control/pipeline.py#L306)). Do **not** let a use case
define the bracket rule twice — two copies of `cool_inset` is two places to get it wrong, and
the reports show that inset is the parameter that decides whether the rule works at all.

Make `reference_control` the single authoritative rule and have the library wrap it into the
baseline Node. Keep `build_baseline_policy` only for the case a memoryless function cannot
express, and if that case never shows up, delete it. The migration is one function each in
`template/control_user_defined.py` and `tests/fixtures/control_user_defined.py`.

### Touch list

| file | change |
|---|---|
| `control/policy.py` | `_ResidualPolicy` wrapper + zero-init helper + reference resolution (~60 lines) |
| `control/config.py` | `residual_delta` + the two validations (~15) |
| `control/policy.py:337` | `describe_policy` unwraps `_QuantizedPolicy` only; teach it the new layer or the reportcard reads the wrong `net` |
| `control/reportcard.py` | report Δ, and **the fraction of steps where δ sits at ±Δ** — see below |
| `template/`, `tests/fixtures/` | `reference_control` over `bracket_band` (~15 each) |
| `control/pipeline.py` | derive the baseline Node from `reference_control` |

The δ-saturation fraction is the diagnostic that earns its line: if the cap is binding most of
the time, either the reference is wrong for this plant or the optimizer is riding a surrogate
artifact into the wall. Both are things you want to know before reading the comfort number.

---

## What it buys, and what it does not

**The floor is not automatic.** Neuromancer's `Trainer` checkpoints the best model after each
epoch's optimizer steps (`trainer.py:289-294`), so the δ = 0 point is never itself a candidate —
by the first evaluation δ is already nonzero. To actually get the guarantee you must score
δ = 0 explicitly and choose. `evaluate_baseline` already does exactly that scoring, on a
byte-identical band; what is missing is a **decision rule**: a submission that does not clear
its own reference ships the reference. Add that, or the floor is rhetorical.

**Δ bounds the action, not the outcome.** The commanded setpoint stays within Δ of the rule's;
the zone temperature does not, because the plant is in between. The useful quantitative form:
the plant's setpoint→zone gain is well under 1 °C/°C (measured by the staircase probe cited in
`evaluations/rbc.py`), so Δ = 1 °C of setpoint deviation is comfortably under 1 °C of zone
deviation. State it that way rather than as a comfort guarantee.

**Where the win is most likely real: actuation and energy.** The residual inherits the rule's
~0.0008 `mean_abs_du` floor instead of the learned 0.147, and with it most of the energy gap —
36 of 36 cells burn more site energy than the thermostat and `n2p-main` burns 31× `n2p-curated`
on the same building. That is a large, measurable, already-instrumented effect (plant counters
in `controlled_fmu.csv`, differenced by `reports/plant_energy.py`), and it does not depend on δ
being clever. Comfort is the coin-flip; actuation is the near-certainty.

---

## The price-driven case — the strongest argument for the design

This is where the residual stops being a safety net and starts being the right architecture,
and the plan should be built with it in view:

- Price enters as a **D column with preview**. `preview_config` already supports per-column
  disturbance preview ([policy.py:158](../src/neurotemplate/control/policy.py#L158)), so the
  policy can see a price forecast with no library change.
- **The rule keeps comfort; δ answers the incentive.** That is exactly the division of labour
  that is annoying to program into an RBC and natural for a network — pre-cool into a price
  spike, coast through it, all inside a box you can point at.
- **Δ becomes an operator-legible knob:** "how far from the comfort rule am I willing to move
  for money." One number, in °C, with a hard bound. That is a far easier thing to sell than a
  learned policy's word.
- **Prerequisite:** the objective must be able to price anything at all, and today it cannot —
  `power_metric` is agent-defined, incomparable across submissions, and returns 0.0 in a
  quarter of cells. A price-weighted power term is a ~5-line hook (`power_metric` already
  receives `D`), but plant-side energy has to be in the scored loop first (N1 conclusion 8,
  already an open recommendation).

---

## Risks, in the order they will actually bite

1. **Δ smaller than the setpoint grid → silent no-op.** Validated against above. Most likely
   failure; cheapest to prevent.
2. **The reference is the agent's to write, so agent quality re-enters** — now bounded, but a
   residual anchored to a bad rule is a bad controller with extra steps. Mitigate by requiring
   the hook (no silent "no reference → plain policy" path) and by keeping `bracket_band` in the
   library where its two documented mistakes are already written down.
3. **Selection is still on the surrogate.** A δ that helps in-surrogate can hurt on the plant;
   nothing here changes that. It changes how much it can hurt.
4. **Version pinning across repos.** `load_policy` reconstructs `ControlConfig` from the run's
   MLflow params (`submission.py:201`), so a residual submission scored by an evaluator image
   built on an older neurotemplate SHA fails at `strict=True` key mismatch. Loud, not silent —
   but plan the image rebuild alongside the merge.
5. **It makes the current experiment design blind.** Once every controller is anchored to the
   same rule, an arm comparison measures δ, whose spread will be *far* smaller than today's
   0.032–0.162 — and today's spread already fits inside a 0.243 noise floor. Anything comparing
   arms after this lands needs a lower-variance readout: paired seeds, more buildings, and mean
   exceedance rather than violation rate (`pipeline.py:186` already argues the threshold count
   saturates and can select a worse controller). Worth deciding *before* the residual ships,
   because afterwards the null results will be indistinguishable from noise by construction.

---

## Recommended sequence

1. `reference_control` hook + `_ResidualPolicy` + `residual_delta` + validation, with the
   template rule over `bracket_band`. Tests: δ-box of zero reproduces the reference exactly;
   the in-rollout current-step slice equals the deploy-side layout; `Δ < step` raises.
2. The decision rule in the reportcard and in `evaluate_baseline`: learned vs δ = 0 on the same
   band, and a submission that does not clear its reference ships the reference.
3. Wire `evaluations/rbc.py` into `dynamic.py` as a second baseline (already recommended
   independently by N1/03 §3) and add the plant-side site-energy delta to the dynamic block
   (N1 conclusion 8). Without these two, a residual's actuation and energy win is invisible to
   the scorecard that is supposed to report it.
4. Only then the price work: price into D with preview, a price-weighted `power_metric`, and a
   Δ sweep as the headline experiment.

Steps 1–2 are the plan. Step 3 is the reason the plan will be believed.
