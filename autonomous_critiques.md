# Autonomous Upgrades / Synthetic Scenarios — Critiques

A running log of issues found while walking through the `autonomous_upgrades` and
`synthetic_scenarios` branches commit by commit. Each entry names the file, states
the problem, and records a verdict/recommendation.

---

## `modeling/probe.py` — actuator authority probe holds co-actuators at a physically ungrounded midpoint

**What it does:** `probe_authority` measures how much each control input moves each
predicted channel. It sweeps one U column at a time to its min and max, while every
*other* U column is held fixed at `0.5 * (umin + umax)` — the arithmetic midpoint of
its actuation range — for the whole rollout. D (disturbances, e.g. outdoor temp) is
left at its real recorded values from the test window.

**The problem:** the midpoint has no physical meaning. It is not the actuator's "off"
state, not a typical operating value, just the middle of its configured range. For
independent actuators (each driving its own separate channel) this is harmless. For
actuators that interact on the same channel — the heating/cooling setpoint pair this
library is explicitly built around — it can silently construct a physically incoherent
combination, e.g. sweeping the heating setpoint up past a cooling setpoint parked at
its own midpoint. No real controller would ever command that combination, so the
surrogate has no real training basis for its prediction there. The authority number
that comes out could be measuring how the model extrapolates into a nonsense state
rather than measuring a real physical lever.

**Untested:** `tests/test_probe.py` only exercises independent setpoint/zone pairs
(`setpoint [°C]` → `zone [°C]`, `setpoint 2 [°C]` → `zone 2 [°C]`), never two
actuators sharing a channel. There is no test — and apparently no consideration in the
design — of the interacting-actuator case, which is the case that breaks.

**Verdict (mine, on request):** partially defensible on one axis — the probe is meant
to run unconditionally with zero extra config, before an app has necessarily written
`modeling_user_defined.synth_cfg` at all, so it can't lean on `SynthSpec.off` (which is
optional, per-app work) without giving up that guarantee. But the midpoint choice
itself is arbitrary and unvalidated, and the failure mode is exactly the kind of silent,
plausible-looking-but-wrong number this whole branch otherwise exists to catch.

**User's verdict:** the probe as structured is extremely problematic — a
half-justified check produces a signal that shouldn't be trusted at all. The
underlying idea (does this actuator move anything) is good, but it should be folded
into the synthetic-scenario framework (which already has a physically grounded
per-actuator `off` value and validates its spec up front) rather than shipped as a
separate always-on probe that fabricates its own "everything else is normal" state
with no physical justification.

**Recommendation:** don't run this as an independent, unconditional dummy probe.
Fold the authority/envelope question into the synthetic scenarios subsystem, where
"what are the other actuators doing" is already an explicit, validated,
per-application decision (`SynthSpec.off`) instead of an arithmetic default.

---

## `common/data_utils.py` — constant-column guard is silent until the reportcard, which is silent until training ends

**What it does:** `degenerate_columns(norm_stats)` is the only place that surfaces a
column left in raw units because it was constant in training. It's called from
exactly one place — `_data_section` in `modeling/reportcard.py` — which only renders
after the full run (training + eval) has finished.

**The problem:** the condition is knowable the moment `norm_stats` is computed, before
the first epoch even starts — it's a property of the training data alone. But nothing
prints it there. Someone watching a training run live — the common case, since that's
when the loss curve is on screen and interesting — has no way to see it until they go
read the reportcard file afterward, if they read it at all. A run can complete, look
fine from the printed loss, and only reveal a channel silently sitting in raw physical
units (with the divergence risk that implies) to someone who opens the markdown file.

**Recommendation:** print a plain terminal warning for degenerate columns at the point
`norm_stats` is computed / frozen, in addition to the reportcard section — so it's
visible to someone who only ever watches stdout during training, not just to someone
who reads the reportcard afterward.

---

## `control/objective.py` + `control/metrics.py` — monotonicity and saturation are treated as bugs when they can be correct control

Three related spots all make the same mistake: they treat a *shape* (monotone term,
actuator pinned at a bound) as evidence of a defective objective, when the shape can
just as easily be exactly what good control looks like.

**1. `term_shapes` / `minimum_on_boundary` (`control/objective.py`)**

What it does: sweeps the raw `power_metric` alone — no comfort term, nothing else —
across one actuator's range, holding the rest of the rollout fixed. If the minimum
lands on an endpoint, it's flagged: reportcard says *"No value of `effort_weight`
fixes this... the term needs re-deriving so it has an interior optimum,"* stdout says
*"Re-derive the term, do not re-tune effort_weight."*

The problem: this is generally false. A real power meter — more setpoint depth costs
more watts — has no reason to have an interior minimum on its own; it's supposed to be
monotone. The interior optimum belongs to the *combined* objective (effort summed with
comfort, which is flat inside the band and steep outside it), not to the effort term in
isolation. Confirmed by the test suite itself: the one "good, not flagged" example in
`test_objective.py` is `(u - target)²` — a quadratic tracking cost, not a power term at
all. There is no test of an honestly-monotone real power metric, because the check has
no way to accept one; it would fire on every real power meter this hook could ever
return.

**2. Actuator saturation warning (`control/metrics.py`, `control/reportcard.py`)**

`SATURATION_WARN` flags any control sitting at a bound >50% of the evaluation, and the
reportcard states it as near-certain user error: *"A control held against its limit is
not being modulated... This is usually an objective term whose gradient points one way
across the whole feasible set, not a policy that chose to saturate."*

The problem: pinning a setpoint at a bound can be the correct, conservative answer, not
a sign of a broken objective or an ignored actuator. If cooling isn't needed, driving
the cooling setpoint to its maximum (least aggressive) value and leaving it there is
good control, not a stuck actuator.

**3. README's blanket warning against actuation proxies**

*"do **not** substitute an actuation proxy such as `(setpoint - ambient)²`... it has
broken three separate campaigns."* This overreaches in the same direction: a proxy term
is a legitimate choice to ship, even one whose literal analytic minimum sits outside the
feasible actuation box, as long as it drives good behavior in practice. Whether a term
is "sufficient" is an empirical question about the behavior it produces, not something
decidable by inspecting its shape in isolation. A real `power_metric` is not required to
make `effort_weight` usable, and the library shouldn't imply it is.

**User's position:** pull all three. Keep reporting the actual thing that's useful —
penalty magnitudes (`objective_balance`/dominance), and factual per-actuator numbers
(saturation fraction, term values across the sweep) — so a human has the information to
balance the objectives themselves. Stop asserting a verdict ("bad," "re-derive," "not
being modulated") from shape or bound-sitting alone; neither is sufficient evidence on
its own, and both are common in perfectly good controllers.

**Recommendation:** keep `objective_balance` (dominance/magnitude) and the raw
`term_shapes` sweep and saturation-fraction numbers as *reported information*. Remove
the pass/fail verdict language built on top of them — `minimum_on_boundary`'s "no
weight fixes this," the actuation section's "not being modulated," and the README's
blanket proxy warning — and let the user judge from the resulting closed-loop behavior
instead.

---

## `control/cli/train.py` + `control/reportcard.py` — envelope-vs-band check inherits the probe's flaws, plus its own

**What it does:** `ff98ca4` runs `probe_authority` before policy training starts and
compares the resulting achievable envelope of each controlled channel to the
configured comfort band. If the band falls outside the envelope, it prints/reports
*"the band ... lies outside what this surrogate can hold. No policy can satisfy it."*

**The problem:** this is the same `probe_authority` from the first entry, called a
second time, so it inherits that critique in full (arbitrary midpoint holding for
co-actuators). It also adds a new one: the "achievable envelope" is built entirely from
`test_dfs` — real recorded test-file data, subsampled to at most 32 windows — so D
(outdoor temperature, etc.) is never swept, only whatever weather actually occurred in
that file, weighted by how often it occurred. If the test file doesn't contain a heat
wave or cold snap, the envelope has no way to know what happens on one, and the check
will bless a band that's only actually achievable in the weather the test file happened
to record.

The language overclaims exactly that gap: *"No policy can satisfy it"* is a statement
about the physical plant, but what's measured is "no policy satisfied it given the
weather in this one file" — not the same claim.

This is doubly frustrating because the fix already exists elsewhere in the same branch:
`FreeDecayScenario` samples ambient *uniformly* across the D column's full recorded
range rather than frequency-weighted by how often it occurred — the right instinct for
a reachability question, where the tail matters more than the typical case. This check
doesn't use it.

**Recommendation:** same as the probe entry — fold this into the synthetic-scenario
subsystem and drive the disturbance sweep the way `FreeDecayScenario` already does
(uniform over the recorded range), instead of replaying one test file. Keep the
underlying idea (check the band against reality before training) — it's cheap and
worth doing — just build it on a sampling foundation that can actually answer the
question being asked.

---

## Emerging theme

Entries 1, 3, and 4 are converging on the same root cause: several of these
diagnostics (actuator authority, envelope vs. band, effort-term shape) are informal,
ad hoc attempts at real control-theory questions — reachability, controllability,
saturation — built on unprincipled foundations (arbitrary midpoints, whatever's in one
test file, shape checks with no notion of the paired objective) instead of a real
analysis. The synthetic-scenario subsystem already has the right sampling primitives
(uniform sampling across the full recorded range, explicit validated per-actuator
`off`/rest states) to be that foundation. The likely right move is not four separate
patches but one consolidation: fold authority, envelope, and reachability questions
into the synthetic-scenario framework, and let it be the one place that does this
properly.

---
---

# Branch: `synthetic_scenarios` (on top of everything above)

Critiques below are specific to the physics-scenario rewrite (`SynthSpec`,
`EquilibriumScenario`, `StepResponseScenario`, `FreeDecayScenario`) and its follow-up
commits, layered on top of the `autonomous_upgrades` findings above — several of
which (entries 1 and 4) already point at this subsystem as the right home for a fix.

---

## What the rewrite got right — keep all of this

The `fbde01a` rewrite fixed real problems and the following should survive any
restructuring:

- **`states` existing at all.** The old code had no notion of which Y channels are
  temperatures: it took min/max across *all* of Y to decide what an "equilibrium
  temperature" was and wrote it into every Y column. On a plant with power channels
  that meant sampling temperatures out of a 0–61728 W range. Real bug, correctly fixed.
- **`validate()` up front**, reporting every problem at once against the column
  mapping before anything runs.
- **Verdicts in physical units** instead of a scalar loss.
- **Step response seeded from equilibrium** (`StepResponseScenario._data` calls
  `_equilibrium_inputs`, ambient pinned equal to indoors), so the actuator's effect is
  isolated from the envelope gradient. This is exactly right and should not change.
- **Free decay sampling ambient from its own full recorded range**, independent of the
  indoor temperature — so a winter dataset correctly tests decay toward a cold outdoors.
- **Per-channel tolerances** as a share of each channel's own recorded range.

---

## `modeling/scenarios.py` + `modeling/loss.py` — remove the training penalty

**What it does:** every scenario carries a `compute()` returning a differentiable
scalar, summed into a `synth_penalty` loss term gated on `cfg.synth_gain`.
`ScenarioResult` carries a `penalty` field alongside the verdict.

**The problem:** this is the framing the rewrite itself half-disowned ("the optional
extra it should always have been"). No campaign in the record ever set
`synth_gain > 0`. It costs a config field, a fail-fast branch in `build_loss`, a
`compute()` on all three scenario classes, and a `penalty` field on every result — which
carried a live bug until `f1d9a2f` (`check()` called `compute()`, drawing a *different*
random sample, so the printed penalty described different scenarios than the verdict).
Keeping it also constrains the scenarios to stay differentiable, which limits how the
checks can be written.

**Recommendation:** delete the penalty path entirely; keep the scenarios as assertions
only. **The gradient penalty (`grad_gain` / `SampledGradientPenalty`) stays** — enforcing
physicality through gradient signs during training is a genuinely interesting mechanism
and is not what this critique is about.

---

## `modeling/scenarios.py` — equilibrium seeds from the *union* of thermal-state ranges

**What it does:**

```python
def _state_range(spec, norm_stats):
    return (min(norm_stats[c]["min"] for c in spec.states),
            max(norm_stats[c]["max"] for c in spec.states))
```

Min-of-mins to max-of-maxes — the widest span across all state channels. The sampled
equilibrium temperature is drawn uniformly from it, and `_equilibrium_inputs` writes
that same temperature into ambient.

**The problem:** measured on `modelica_rtu_mall/data/{excited,spring}/train.csv`
(6048 rows each):

| Channel | excited (summer) | spring |
|---|---|---|
| Outdoor Dry-Bulb | 19.37 .. 33.91 | 1.69 .. 26.72 |
| Zone Air | 22.22 .. 28.61 | 18.89 .. 27.29 |
| Supply Air | 8.71 .. 31.41 | 8.56 .. 54.52 |
| Return Air | 20.05 .. 28.61 | — |
| Mixed Air | 20.93 .. 30.93 | — |

Supply air spans 46 °C against the zone's 8.4 °C. Adding it as a thermal state makes
the union range on `spring` **8.56 .. 54.52**, so the scenario would assert the building
holds steady at 51 °C indoors *and outdoors* — 28 °C past ambient's recorded maximum of
26.72. Every number from that is extrapolation noise.

**Recommendation:** seed from the **intersection** (`max` of mins to `min` of maxes) —
the window where every thermal state has actually been observed. Measured:

- excited: **22.22 .. 28.61**, spring: **18.89 .. 27.29** — both non-empty, both landing
  on the zone range (the narrowest channel wins, which is the right answer).
- **Backward compatible:** with one state, intersection ≡ union, so the existing
  single-state fixture is unchanged.
- **Fixes free decay's minimum gap too.** `MIN_GAP_FRAC` is 10% of the state span:
  4.6 °C under union on spring, 0.84 °C under intersection. The union version demanded a
  gap half the zone's entire operating range.
- **Tolerances are unaffected** — `_range(c)` is already per-channel, so supply air would
  get 0.02 × 46 = 0.92 °C of slack while the zone gets 0.17 °C. "Care less about the wide
  channel" emerges on its own.

**Ambient must not participate in the intersection**, and today cannot: `_state_range`
reads only `spec.states`, and `validate()` forces states ⊆ Y while ambient ∈ D. This
matters — in winter, ambient (−10..10) and the zone (18..27) do not overlap at all, and
requiring it would make the tests unusable for half the year.

**On an empty intersection:** error loudly, but word it carefully. An empty intersection
means *the data never shows these channels overlapping*, not that they are physically
incapable of reaching a common temperature (a chilled-water loop and a zone really would
converge if the plant sat off for a week).

**Known residual, not worth a fallback:** in winter, equilibrium seeds *ambient* at ~22 °C,
a value that dataset never recorded, so the test is a pure extrapolation with no nearby
data support. Handle it the way `_stability_section` handles the rollout horizon — state
the seeded range next to ambient's recorded range in the report and let the reader weigh
it. Do **not** sample differently by season; that would make the test mean different
things on different datasets and destroy comparability.

---

## `modeling/scenarios.py` — every scenario collapses to a single worst channel

**What it does:**

```python
worst = max(range(len(tol)), key=lambda i: float(drift[i]) / tol[i])
value, limit = float(drift[worst]), tol[worst]
channel = self.spec.states[worst]
```

Each scenario returns exactly one `ScenarioResult`, naming only the worst-scoring
channel.

**The problem:** this makes the thermal-state classification a hard gate. The moment a
second state is added, a failure on the wide-ranging one produces
`Equilibrium: FAIL — supply air drifted 2.0` and the zone's result **vanishes from the
report entirely**. There is no way to look at a channel you care about, and no way to
shrug off one you don't.

**Recommendation:** report per channel. This is the highest-leverage change in this
section: it converts the thermal-state/readout partition from a hard gate (a wrong call
silently kills a check or fabricates a failure) into a soft label that only affects
*seeding*. A user who thinks supply air temperature should behave like a thermometer in
the room can declare it a thermal state, see its row, and decide for themselves whether
its dynamics matter — rather than having that judgement bundled into a pass/fail.

Note the knock-on: `physics_checks_passed` is logged as a single leaderboard scalar. It
should become per-channel too, or an ignorable channel will pollute run comparisons.

---

## `modeling/scenarios.py` + `common/data_utils.py` — direction is declared twice, and actuator coverage is never enforced

**What it does:** `Actuation` is a hand-written bundle — a label, a dict of arbitrary
U columns → values, and one `direction` int. `SynthSpec.actuations` is whatever list the
app author chose to write; the RTU fixture defines exactly one (`cooling`, setpoint at
its minimum, direction −1).

**The problems, three of them:**

1. **The same physics is declared twice, in two formats, with no cross-check.**
   `get_grad_masks` already builds `grad_mask["U"]: (ny, nu)` and
   `grad_mask["D"]: (ny, nd)` sign matrices from a JSON file — that *is* "per state
   channel, per input, which way does it push." `Actuation.direction` is a lossy
   projection of it: one sign for a bundle of columns, applied to every state at once.
   An app can declare `+1` in the JSON and `direction=-1` in the actuation and nothing
   notices.
2. **One sign cannot cover multiple states.** `act.direction * move[i]` scores every
   state channel against the same sign, so a multi-zone plant where one actuator warms
   zone A and cools zone B cannot be expressed at all.
3. **Coverage is unenforced.** `validate()` checks only that named columns are in U and
   that direction is ±1. Nothing requires every U column to appear in an actuation, or
   both extremes to be tested. An empty list is legal and silently skips the check
   (`test_no_actuations_means_no_step_checks` pins this). On the RTU fixture the step
   check therefore tests one column, one extreme, one direction.

**Recommendation:** make the gradient file a **required** input to `SynthSpec` and derive
directions from it. `off` is still needed (an actuator's off value is not derivable), but
mins/maxes come from `norm_stats` and directions come from the mask — so per-column,
both-extreme coverage can be generated rather than hand-written. Cross-validate the two
declarations against each other. The D→Y edges also supply the adversarial ambient
direction the reachability test needs, with no new declaration.

Precedent already exists in the same dataclass: `off` is required per U column and
enforced (`missing = [c for c in u if c not in self.off]`). `direction` should be held to
the same standard.

---

## `modeling/config.py` — `synth_n_steps` is unitless, while the sampling frequency is right there

**What it does:** `synth_n_steps: int = 24`, a single global step count shared by all
scenarios.

**The problem:** nothing relates it to physical time or to the plant's time constant.
For a reachability claim ("this is the maximum swing achievable") the horizon is
load-bearing, and an arbitrary shared step count is weak evidence. Meanwhile `freq_s` is
already threaded into `_run_physics_checks(...)` and passed to `plot_synthetic` for axis
labels — the sampling frequency is available at the exact point the tests run and is
simply not used to size them.

**Recommendation:** specify scenario horizons in physical time (e.g. 4–8 h) and convert
via `freq_s`. No new plumbing required.

---

## Readouts: seeded, then free — described, not regulated

**What happens today:** `_rollout_data` writes the resting row into `yn` for `in_steps`
steps; `Y` is `torch.zeros(...)` and is never read ("only sets the rollout length"). After
the seed, `yn` is both input and output of the Node, so the model's own readout
predictions feed back as inputs at every subsequent step. Only U and D stay pinned for
the whole rollout, since the model cannot write those.

So a non-state Y channel is pinned during the seed, then free, then fed back into the
state predictions, and never scored. If the surrogate predicts 4000 W of DX power at
equilibrium with the compressor off, that fabricated value becomes an input to the next
zone-temperature prediction — an equilibrium drift you cannot attribute.

**Decision: no assertions, but report order statistics.** Only thermal states are
regulated. Asserting on readouts was considered and rejected — a readout's correct
behaviour is scenario-dependent (hold at off during equilibrium and free decay, but
*move* under a step response) and encoding that is more machinery than it is worth.

Instead, for each readout column in each scenario, report the **0 / 25 / 50 / 75 / 100
percentiles** of what the surrogate emitted. No pass/fail, no tolerance — just the range.
That is enough to tell `0 / 0 / 0 / 0 / 0` from `0 / 1200 / 3000 / 4200 / 5800` at
equilibrium, which is the whole question a reader has.

This substantially retires the blind spot described above. A drifting equilibrium whose
cause is a readout gone haywire is now visible in the same table, without the library
having to take a position on what the readout *should* have done.

Three details to settle when implementing:

- **Quantile over the free-running steps only.** Index 0 of the rollout is the last
  seeded step, which for a readout is its `off` value by construction — including it puts
  a known constant into the min or max. Report the off value alongside as context
  instead.
- **Pool across scenarios and time** for a channel within a scenario. These are meant to
  be cheap descriptive statistics, not a per-rollout breakdown.
- **Use a separate result type.** `ScenarioResult` is assertion-shaped (`passed`,
  `tolerance`, `channel`, `value`); order statistics do not fit it, and forcing them in
  with `passed=True` would be worse than a small dedicated dataclass rendered into its
  own table.

Cost is a tensor slice and a `torch.quantile` on rollouts that already ran — no extra
forward passes.

This also strengthens the soft-label property from the worst-channel entry above:
misclassifying a channel now moves it between *asserted* and *described*, and both
produce output. Nothing disappears from the report either way.

---

## Target design

Two classes, differing **only in how they are seeded**:

- **Thermal states** — commensurate with ambient, have thermal mass. Seeded at the
  sampled equilibrium temperature drawn from the *intersection* of their ranges. These
  are the only channels scored.
- **Readouts** — everything else in Y. Seeded at their off value, then unregulated, but
  reported: 0 / 25 / 50 / 75 / 100 percentiles per channel per scenario.

Four scenarios, results reported **per channel**:

1. **Equilibrium** — everything off, ambient equal to indoors. States must hold.
2. **Free decay** — from equilibrium, ambient jumps to a new value sampled from its own
   recorded range, controls still off. States must move toward it and must not sail past.
3. **Actuator step response** — from equilibrium, push one actuator at a time. Each state
   must move in the direction the gradient mask specifies for that `(state, control)`
   pair. Every U column, both extremes.
4. **State reachability** — from equilibrium, push *every* actuator that affects a given
   state in the same direction, and every exogenous input to the extreme the gradient
   sign says reinforces it. Run for a physically meaningful horizon; repeat in both
   directions. The resulting span is that state's reachable envelope under this
   surrogate.

Tests 3 and 4 are deliberately different questions and stay separate: step response
**isolates** one actuator's authority (ambient pinned to indoors, no envelope gradient),
while reachability **deliberately conflates** actuator and weather effects because it is
asking for a bound, not an attribution.

Test 4 is also the honest version of what `probe.py`'s envelope and the control-side
envelope-vs-band check were reaching for in entries 1 and 4 — grounded in declared `off`
values and gradient-derived directions rather than arithmetic midpoints and whatever
weather happened to land in one test file. That is the consolidation the "Emerging
theme" note above was pointing at.
