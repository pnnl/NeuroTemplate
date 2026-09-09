# How the 2R2C pipeline identifies the parameters it holds fixed

Written against `f4cbd52`. Companion to
[rc_model_report.md](rc_model_report.md), which argues *why* the storage is
asserted rather than fitted. This one answers the next question: **given that
it is asserted, what is the procedure for arriving at the number, and what does
the library do to help?**

The short answer, stated up front because it is the finding:

> There is **no identification step inside this library.** Stage one happens
> outside it, by hand, with tools that are not in this repo. The pipeline's
> contribution is to make the assumption *explicit*, *typed*, and *read back in
> physical units* — and that is genuinely valuable — but it does not estimate,
> and its physics checks **cannot detect a wrong prior**. §4 shows the
> arithmetic on that last claim, because it is the part most likely to be
> assumed away.

---

## 1. Inventory: everything `2r2c` holds fixed

"The three time constants" is the headline, but seven things are frozen, and
they are not the same kind of thing. Distinguishing them is most of the work,
because only two of the seven are identifiable from data at all.

| # | What | Where | Kind | Identifiable from data? |
|---|---|---|---|---|
| 1 | **Topology**: two states, one chain `ambient—zone—mass` | [rc.py](../src/neurotemplate/modeling/rc.py) | structural choice | No — a modeling commitment, not a parameter |
| 2 | `rc_tau_ambient_h` = 2.0 — zone↔ambient | [config.py:77](../src/neurotemplate/modeling/config.py#L77) | **prior** | **Yes**, from free decay |
| 3 | `rc_tau_zone_mass_h` = 1.0 — zone↔mass, zone's side | [config.py:78](../src/neurotemplate/modeling/config.py#L78) | **prior** | Weakly — sets the fast mode |
| 4 | `rc_tau_mass_zone_h` = 4.0 — mass↔zone, mass's side | [config.py:79](../src/neurotemplate/modeling/config.py#L79) | **prior** | No, except through #3 as a ratio |
| 5 | **Capacitance ratio** `C_m/C_z = a_m/a_c` = 4.0 | implied by #3 ÷ #4 | derived prior | No directly |
| 6 | `MIN_TAU_STEPS` = 2.0 | [rc.py:MIN_TAU_STEPS](../src/neurotemplate/modeling/rc.py) | stability constant | Not a parameter — a guard rail |
| 7 | **The mass seed rule**: window mean, then integrate | `RCBlock._seed_mass` | initialization policy | No — the experiment in the RC report is precisely this |

Two more that are fixed but not usually thought of as parameters:

- **The zero-ambient convention.** `-a_ambient * zone` is conduction toward
  zero, correct only because `g` is affine in D and D carries the relaxation
  target. This is an *assumption about your data*, and it fails silently if D
  has no ambient-like channel. It is fixed the moment you pick the family.
- **`g` reads `(u_k, d_k)` only, never the state.** Fixed by design; it is the
  stability proof. Not a knob.

### The asymmetry that matters

Of the three config fields, only **#2 has an independent measurement**. #3 and
#4 do not have separate experiments behind them — they are two knobs on one
physical quantity pair (the fast mode and the capacitance ratio), and the
defaults 1 h / 4 h are a *convention* ("mass is 4× the air"), not a finding.

So when the RC report says "set these from what you know about the building",
it means, honestly:

- **#2**: measure it. There is a real procedure. §3.
- **#3**: pick something well under an hour, or search it. It sets `tau_fast`,
  which the reportcard prints and which you can sanity-check against how fast
  the zone responds to a setpoint change.
- **#4**: leave the 4× convention alone unless you have a reason. It moves
  `tau_slow` too, so if you tune it, tune it *instead of* #2, not alongside —
  otherwise you have two knobs on one number and nothing to pin either.

That last point is the practical identifiability statement for this
parameterization: **three fields, two observable modes.** The map from
(#2, #3, #4) to (`tau_fast`, `tau_slow`) is 3 → 2 and therefore not injective.
Fixing #4 at the convention is what makes the remaining two-to-two map usable.
This is not written down anywhere in the code, and it should be.

---

## 2. What the library does and does not provide

An honest map, because the RC report's "three good ways to get stage one" reads
like three features, and only one and a half of them are in the repo.

| Route | Status in this repo | What actually exists |
|---|---|---|
| **1. Measure it** (free decay in the data) | **Not implemented.** | Nothing. `data/excited` and `v7_precool_probe.py` are in the RTU-mall repo. This library has no decay estimator, no autocorrelation tool, no step-response fit. `common/metrics.py` is RMSE/R²/stability only. |
| **2. Search it** (Optuna) | **Works, unaided.** | `rc_tau_ambient_h` is an ordinary config field; put `cfg.rc_tau_ambient_h = trial.suggest_float(...)` in [`hyper_args`](../template/modeling_user_defined.py) and [optuna.py](../src/neurotemplate/modeling/cli/optuna.py) sweeps it like any other. No RC-specific support, and none needed. |
| **3. Sweep and read** (three runs, three priors) | **Works, unaided.** | `--set rc_tau_ambient_h=...`, then read `tau_slow` off the reportcard. |
| **4. Read it back** | **Implemented.** | `RCBlock.describe()` → `mode_taus` → [reportcard.py:110](../src/neurotemplate/modeling/reportcard.py#L110), per zone, per channel, in hours. |

So the pipeline supports **assert → train → read back**, and supports
*searching* the assertion against dev loss. It does not support *measuring*
it. Route 1 — the one the report calls the preferred option — is the one with
no library support at all.

That is a defensible place to be (a decay estimator is a data-analysis tool,
not a training-pipeline concern) but it should be said out loud, because the
current framing — "a free-decay episode gives it directly" — reads as though
the pipeline will do it for you.

### Why route 4 is load-bearing and why it is trustworthy

The reportcard number is the only feedback loop in the whole procedure, so it
is worth being precise that it is not circular. `tau_slow` is not a re-print of
what you typed:

- You set three **component** time constants; the reportcard prints the two
  **modes**, which are eigenvalues of the assembled matrix and are none of the
  three (2/1/4 h → 0.60 h and 13.40 h).
- The number survives normalization. Both states share the zone's affine
  frame by definition (`m = (T_m − s)/g`), so the state matrix is unchanged
  under z-scoring, so the eigenvalues are unchanged, so `steps / steps_per_hour`
  is real hours. That is why it can be compared against a building.
- If `rc_train_taus: true`, it prints what training *ended at*, not what you
  set — which is exactly how you would catch the drift the RC report documents.

There is one silent transform to be aware of: `_init_theta` clamps a requested
time constant up to `MIN_TAU_STEPS + 1 = 3` steps. On hourly data,
`rc_tau_zone_mass_h: 1.0` is one step and is silently raised to three hours.
The reportcard shows the consequence, but nothing announces the clamp. On
5-minute data it never fires; on coarse data it fires and changes your
assertion without saying so.

---

## 3. The procedure, end to end

This is what "identifying the fixed parameters" concretely means today.

### Stage 0 — check the family's precondition

`2r2c` is correct only if **D carries the channel the states relax toward**
(or that target is zero in normalized space). This is the same assumption
`SynthSpec.ambient` encodes. If your D has no such channel, stop — the block
is quietly wrong, not merely inaccurate, because `g` has nothing to form
`+a_o·T_target` from.

### Stage 1 — get `tau_slow` from the data (outside this library)

You need an episode where **the plant is uncontrolled and the states are away
from their target**: HVAC off, doors shut, a temperature difference decaying.
Then:

1. Find those windows. In practice: filter to rows where the actuators are at
   their `off` values (the same values `SynthSpec.off` declares) for a run of
   at least ~2× the expected `tau_slow`.
2. Fit `T_zone(t) − T_ambient(t) = A·exp(−t/τ)` over each window. If ambient is
   moving, fit against the *gap*, not the raw temperature.
3. You will get a spread, not a number — the RTU-mall data gave 6.4–10.9 h
   depending on the specification. That spread is real and is the honest
   uncertainty on the assertion. Take a value, and note the range.

The half-life is often the easier thing to read and to check against intuition:
`t½ = τ·ln2`. A 10 h τ is a 6.9 h half-life.

**What to do when there is no such episode.** This is the common case — a
building under continuous supervisory control never decays freely. Then route 1
is unavailable and you fall back to route 2 (Optuna over
`rc_tau_ambient_h`), which does not have the wrong-seed pathology because each
trial's prior is fixed throughout that trial. Dev loss is a different criterion
from identification and you should say so on the reportcard rather than
claiming a measurement. Route 2 picks *the prior the data prefers*; it does
not measure the building.

### Stage 2 — invert the modes to get the config fields

You measured a mode; the config takes components. The map is `mode_taus`,
and there is no inverse helper, so in practice:

1. Fix `rc_tau_mass_zone_h` at the 4× convention (or at your own capacitance
   ratio if you know it).
2. Pick `rc_tau_zone_mass_h` from how fast the zone responds to a setpoint
   change — sub-hour for air.
3. Sweep `rc_tau_ambient_h` over a handful of values, running `mode_taus` (or
   just a short training run) until the printed `tau_slow` matches what you
   measured in stage 1.

Three values takes minutes. It is a two-parameter inversion done by hand, and
that is fine at this scale — but it is a real step that no document currently
spells out, and it is why users end up thinking the config fields *are* the
time constants they measured. **They are not.** `rc_tau_ambient_h: 2.0` gives
`tau_slow: 13.4 h`, not 2 h.

### Stage 3 — train, and read it back

`nt model train`. Then confirm on the reportcard that `tau_slow` reads back at
what you asserted, per zone:

```
- **fy:** `PerZoneBlock`, params=2,504, in=42, out=2
  - **zone `north`:** `RCBlock`, params=1,252, in=30, out=1
    - North Zone Air Temperature [°C]: tau_fast=0.60 h, tau_slow=13.40 h
```

With `rc_train_taus: false` this is a *check that the freeze held*, not a
result. It is worth doing anyway — it catches the `_init_theta` clamp, and it
catches a `steps_per_hour` resolved from the wrong sampling frequency, which
would place every time constant against the wrong timestep.

### Stage 4 — decide whether the prior was any good

The only quantitative signal the pipeline gives you is **test RMSE across
priors**, and it is a real one (from the RC report's own sweep):

| prior | `tau_slow` | test RMSE |
|---|---|---|
| 0.85 | 7.8 h | 0.579 |
| **2.0** | **13.4 h** | **0.536** |
| 4.2 | 24.3 h | 0.649 |

The data has an opinion. It is a *weak* opinion — an 8% RMSE spread over a 3×
range of storage — but it is not nothing, and it is the empirical basis for
route 2 working at all.

---

## 4. The gap: the physics checks cannot falsify the prior

This is the part worth being blunt about, because the pipeline *looks* like it
validates the storage and does not.

[`FreeDecayScenario`](../src/neurotemplate/modeling/scenarios.py) teleports the
model to a new climate with actuators off and asserts the states relax toward
ambient. It reports "closed X% of the gap in N steps" and passes when

```
DECAY_MIN_CLOSED = 0.02  <  closed fraction  <  DECAY_MAX_CLOSED = 1.10
```

with the horizon set by `synth_hours`, default **4.0 hours**.

Run the arithmetic. For a single exponential, the fraction closed over 4 hours
is `1 − exp(−4/τ)`:

| τ | closed in 4 h | verdict |
|---|---|---|
| 6.4 h | 46.5% | pass |
| 8 h | 39.3% | pass |
| 13.4 h (default) | 25.8% | pass |
| 24.3 h | 15.2% | pass |
| 31.6 h | 11.9% | pass |
| 198 h | 2.0% | **the first failure** |

To fail `DECAY_MIN_CLOSED` at the default horizon a model would need a slow
mode above roughly **198 hours** — eight days. And the real 2R2C response is a
two-mode sum whose fast mode closes a chunk of the gap immediately, so the true
closed fraction is *higher* than these figures, which makes the check less
discriminating still.

**Conclusion: the free-decay check tests the sign of the envelope coupling, not
its rate.** It will pass an 8-hour building modeled as a 24-hour one without a
murmur, which is exactly the error the whole "assert it" design is exposed to.
That is not a bug — the scenario's docstring says it is asserting relaxation,
not measuring it — but it means:

> The reportcard shows you `tau_slow` and, separately, a free-decay check that
> passes. It is very easy to read those two together as corroboration. They are
> not. The check does not know what the building's decay rate is, because
> nothing in this library ever looked at the building's decay rate.

The prior is falsifiable only by stage 1, which is outside the library, or
weakly by test RMSE, which is the table in §3.

---

## 5. What would close the gap

Smallest thing first. These are proposals, not commitments.

**A. A data-side decay estimator (~half a day).** One function, in
`common/metrics.py` or a small `modeling/identify.py`:

```
decay_time_constant(df, state_col, ambient_col, off_mask) -> (tau_h, n_windows, spread)
```

Find runs where `off_mask` holds, regress `log|T_state − T_ambient|` on time,
report τ with a spread across windows. It needs the same `off` values
`SynthSpec` already declares, so the declaration exists — this would be the
second consumer of it.

This is the single highest-value addition to the 2R2C story, because it turns
the design from *"assert a number and read it back"* into an actual two-stage
grey-box fit — which is what the user expected the pipeline to be doing in the
first place, and the expectation was reasonable. Stage one just does not have
to share an optimizer with stage two; it does have to exist.

**B. Print the measurement next to the assertion.** Once A exists, the
reportcard line becomes checkable rather than merely informative:

```
North Zone Air Temperature [°C]: tau_fast=0.60 h, tau_slow=13.40 h
  (data: free decay over 7 windows gives 6.4–10.9 h — prior is 1.2–2.1x slow)
```

That single line does more for "make the sci-ml choices easier" than any amount
of further modeling. It is also the only way a user finds out their prior is
wrong without running a controlled experiment on the real building.

**C. Say the 3 → 2 identifiability thing in the config comments.** Three fields,
two observable modes; fix `rc_tau_mass_zone_h` at the convention and move one
knob. Costs a comment, prevents a common way of wasting an afternoon.

**D. Announce the `_init_theta` clamp.** A warning when a requested time
constant is raised to the floor. One line; it silently changes an assertion on
coarse data.

Explicitly **not** recommended: an automatic stage-one fit wired into
`nt model train`. Estimating τ from data requires judgment about which windows
count as free decay, and a pipeline that guessed would produce a number that
looks measured and is not — the exact failure the `rc_train_taus: false`
default exists to avoid, moved one stage earlier.

---

## 6. Summary

- **Seven things are fixed**, of which three are config fields, one is a
  convention (the 4× capacitance ratio), one is an initialization policy (the
  mass seed), one is a stability guard, and one is the topology. Only
  `rc_tau_ambient_h` has an independent measurement behind it.
- **Three fields map onto two observable modes**, so one of them must be pinned
  by convention for the other two to be meaningful. Nothing says this today.
- **The config fields are not the time constants you measure.** You measure a
  mode; you set components; `mode_taus` is the map and it has no inverse
  helper. Expect a short manual sweep.
- **The library supports assert → train → read back, and Optuna search.** It
  does not support measuring, and the preferred route in the RC report is the
  one with no library support.
- **The free-decay physics check cannot falsify the prior** — at the default
  4-hour horizon it only fails a slow mode above ~198 h. Do not read a passing
  check as corroboration of `tau_slow`.
- **The fix is a data-side decay estimator** whose output is printed beside the
  asserted `tau_slow` on the reportcard. That is the missing stage one, and it
  is small.
