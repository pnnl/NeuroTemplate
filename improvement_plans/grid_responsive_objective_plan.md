# Plan: price-driven and peak-shaving control objectives

## Status: implemented on `curated_v2`

All six phases landed. Commits, oldest first:

| Commit | Phase |
|---|---|
| `f24e20f` | 0 — `price_signal` + `PRICE_SCALE` hooks, `price` in the window dict, price on the plots and in the trajectory CSV |
| `ce5453b` | 1 — `"price"` in `preview_config`, the `insize` term, `price_preview` on the reportcard |
| `1dd8beb` | 2 & 3 — `cost_weight`, `peak_weight`, `peak_q`, the shared power variable, fail-fast checks, objective notes |
| `bdd6ad2` | 4 — `EnergyMetrics.energy_cost`, the MLflow keys, the baseline cost column |
| `c2dee7a` | 5 — `effort_weight` → `energy_weight` |
| `bc2ceeb` | integration: a training run with all three power terms weighted |
| `03530e2` | price re-sourced: sampled from a real historical series, not derived from `t` |

Phases 2 and 3 shipped together: the shared power variable is the same edit
for both terms, and splitting it would have meant writing `_power_objectives`
twice. Everything else went as planned.

### Eight decisions that changed under implementation

**1. The `t`-derived tariff was wrong, and got replaced.** This is the plan's
one substantive error, and it took two forms.

The visible one: `sec_elapsed` counts from the *first row of the file*
(`data_utils._engineer_features`), not from midnight, so `(t / 3600) % 24` is
hour-of-day only if the file starts there. That was survivable — document the
offset and move on — and it is what shipped first.

The real one, which the plan never considered: **a tariff that is a
deterministic function of the clock is a tariff the policy can learn *as* the
clock.** D is strongly hour-of-day correlated, so "read the price" and "read
the time" become the same task; the policy posts a convincing load-shifting
trace with no price responsiveness in it, and falls over the moment the tariff
changes — which is the one thing about a tariff that does change. This is
exactly the shortcut the [control transform plan](control_transform_plan.md)
documents for setpoint schedules, reintroduced in a new place by a plan that
cites it.

The fix reverses the plan's Part 1 recommendation: **price is sampled per
window from a real historical series, exactly like the comfort band.** The
hook is now
`price_signal(n_samples, nsteps, generator) -> [n, W, 1]` returning *physical*
$/kWh, with the library owning the `/ PRICE_SCALE` normalization — one clear
contract instead of asking the hook to pre-divide. The template shows it over
PG&E Hourly Flex Pricing (`grid_response.ipynb` hosts a sample), upsampled
hourly→5-minute and sliced at drawn offsets.

Three consequences worth recording:

- **Window offsets are drawn independently of the weather windows**, which is
  deliberate: it removes the residual price↔weather correlation as a second
  route to the same shortcut, and it decouples how much price data is needed
  from the length of the plant record. The notebook makes the same choice.
- **Reproducibility moves from "free" to "seeded".** The plan's argument that a
  `t`-derived price needs no seed discipline was true and is now moot: price
  rides the same `generator` the band does, `evaluate_closed_loop` reseeds from
  `EVAL_BAND_SEED`, and the learned/baseline comparison stays byte-identical.
  Price is drawn *after* the band so enabling it does not re-roll every band.
- **Negative prices are allowed.** The first implementation rejected them to
  protect the cost gradient's sign. With real market data a negative price
  means what it says — a market paying to absorb load — and a cost term that
  then wants more load is correct. Only non-finite values are rejected, since a
  gap in the source series would propagate silently.

The real prices also make `energy_cost` a real bill rather than an index, which
is what the baseline comparison column was for.

**2. `frac_power_clamped` is not observable; `frac_power_zero` is.** The clamp
lives *inside* `power_metric`, so the library never sees the pre-clamp value
and cannot report what fraction of steps it zeroed. It can see exact zeros,
which is the same signal under the recommended `.clamp_min(0.0)` shape. Named
for what it measures.

**3. "Build the power variable once" saves code, not compute.** `PenaltyLoss`
evaluates each `Constraint`'s graph separately, so the shared `power` node is
recomputed once per term regardless. Kept anyway, for the reason that actually
matters: three terms that could price *different* power signals would be
indistinguishable on every report.

**4. The `p ≥ 0` guard is `torch.relu`, not `clamp_min`.** Neuromancer's
`Variable.__torch_function__` reorders arguments — it collects the non-Variable
args and puts them *first* — so `torch.clamp_min(power, 0.0)` evaluates as
`torch.clamp_min(0.0, tensor)`. Any torch call on a Variable with a scalar
argument has this problem; single-argument calls and keyword arguments are
safe. `torch.relu` is `clamp_min(0)` exactly and takes one argument.

**5. No epsilon under the M_q root.** `x^(1/q)` has an infinite derivative at
`x = 0`, and `power_metric`'s clamp makes all-zero windows plausible, so this
looked like it needed a floor. It does not: PyTorch's `pow` backward returns 0
rather than NaN there. Verified and pinned with a test rather than papered over
with a magic constant.

**6. The "no price preview" diagnostic grew a sibling.** Both are the same
failure — a term switched on that cannot do the thing it was switched on for —
so they live together in `objective.objective_notes(cfg, arch)`: cost without
preview (reactive, not shifting), and peak as the only power term (minimisable
by *adding* off-peak load). Reported as prose, not raised, because both
configurations are legal and occasionally deliberate. This needed `arch`, so
`describe_policy` now runs right after `build_policy` instead of at reportcard
time.

**7. The sweep is bound to the power signal, as the plan asked — and the JSON
key says so.** `objective_diagnostics.json`'s `effort_shape` became
`power_shape`. One sweep serves all three terms; sweeping `cost` would be the
same curve times a constant.

**8. The baseline comparison table grew a seventh column.** Cost per window
comes through `_comparison_values` for free, which is the headline number the
feature exists to produce: dollars per window, learned vs. rule, identical
windows, identical tariff.

### What is deliberately still missing

The non-goals held. No user-supplied loss terms, no cross-window demand charge
(the running-peak rollout state), no nested `objective:` config block, no
tariff realism beyond a per-step price. One new caveat worth recording: the
policy reading `price` adds a fifth key to its Node's `input_keys`, which
*does* change the deploy-side call. That is the one place this plan collides
with the [residual plan](residual_control_plan.md)'s "zero evaluator changes"
guarantee, and it is opt-in — a run with no `price_signal` hook is byte-for-byte
the run it was before.

---

## What we want

Two new closed-loop objectives, on top of the comfort band the controller already holds:

1. **Minimum energy cost** — minimize `Σ_t price_t · power_t` over the rollout, with a
   time-varying tariff. This is load *shifting*: precool while energy is cheap, coast
   through the expensive block.
2. **Minimum peak demand** — penalize the *shape* of the load rather than its integral, so
   the controller prefers a flat profile over a spiky one of the same total energy.

Both are power-derived. The library already computes a per-step power signal
(`control_user_defined.power_metric`) and already reduces it to a scalar
(`closed_loop.py:179`, `mean_power`). What it cannot express is **a different reduction of
that signal**, and it has **no path for an exogenous economic signal** to reach either the
loss or the policy. Those two gaps are the whole job.

---

## What actually blocks this today

**1. There is exactly one power term, and its reduction is hardcoded.**
`build_control_loss` ([closed_loop.py:163-181](../src/neurotemplate/control/closed_loop.py#L163-L181))
builds `power → torch.mean(power) → × effort_weight`. Cost needs an elementwise weighting
*before* the reduction; peak needs a *different* reduction. Both are the same signal.

**2. No exogenous signal path.** The window dict
([scenarios.py:106-113](../src/neurotemplate/control/scenarios.py#L106-L113)) carries
`yn, U, D, t, ymin, ymax` and nothing else. A tariff is neither a state, a control, a
disturbance the surrogate consumes, nor a comfort band.

**3. The policy cannot see the future price**, so it can only react to the current one —
which is not load shifting, it is a slightly more expensive thermostat. The preview
mechanism exists (`_preview_steps`, [policy.py:158](../src/neurotemplate/control/policy.py#L158))
but its allowed keys are D columns and `"band"`.

**4. Config has three fixed weight fields** (`comfort_weight`, `du_weight`,
`effort_weight`, [config.py:76-86](../src/neurotemplate/control/config.py#L76-L86)) and
`validate()` checks them by name.

Everything else already generalizes: `objective_balance`
([objective.py:126](../src/neurotemplate/control/objective.py#L126)) reads term names off
`loss.objectives`, so new terms appear on the pre-training balance report and in the
reportcard with no changes at all. That is the piece worth *not* breaking.

---

## The shape of the refactor: signals, reductions, weights

Three layers, and the middle one is new:

```
  LAYER 1 — per-step signals over the scored horizon [batch, P]

     power_t = power_metric(yn, U, D)      differentiable, flows to the policy
     price_t = price_signal(t)             data; no gradient

  LAYER 2 — reductions to a scalar          <-- the new degree of freedom

     energy  = mean_t( power )                    $-free, units of power
     cost    = mean_t( price * power )            units of power·$/kWh = $/h
     peak    = mean_batch( softmax_t( power ) )   units of power

  LAYER 3 — weights, summed                 unchanged in kind, wider in number

     comfort_weight · comfort + du_weight · du
       + energy_weight · energy + cost_weight · cost + peak_weight · peak
```

The only structural claim here is that **the power signal and its reduction are separate
concerns**. Everything below follows from that.

---

## Part 1 — Getting the price into the data stream

### Why price must not be a D column

Tempting, because D already flows everywhere the policy and the loss can see it. It is
wrong for two reasons, and the first is fatal:

- `nm_mapping["D"]` is **fixed by the frozen sys-id run**. The surrogate's input width is
  baked into its weights. Adding a column means retraining the surrogate.
- Even if you did retrain it, you would be telling the dynamics model that electricity
  price is a physical driver of zone temperature. It will happily fit the correlation
  between price schedules and time of day, which is the same shortcut the
  [control transform plan](control_transform_plan.md) documents for setpoint schedules.

Price belongs on the same footing as `ymin`/`ymax`: an exogenous signal the *policy and the
objective* read, and the *dynamics never sees*.

### Where it enters

A new key `price` of shape `[n, W, 1]` in the window dict, built in `build_window_dict`
alongside the comfort band. Verified against `SystemPreview.forward`: `data = input_dict.copy()`
and only keys a node *writes* get restacked, so an extra key passes through the rollout
untouched and is available in the loss as `variable("price")` and in `out["price"]` at eval.
`build_window_dict_multi` concatenates whatever keys `build_window_dict` produced, so
multi-file pooling comes for free.

```
window dict, hmax = 12, P = 288

index          0 ............ 11 | 12 .................. 299
               ---- warmup ----- | ------ scored horizon ------
yn             seeded (real Y)   | predicted by the surrogate
U              seeded (real U)   | written by the policy
D    ------------- real disturbances, full W -----------------
t    ------------- POSIX seconds, full W ---------------------
ymin/ymax ------- sampled band, full W -----------------------
price ----------- NEW: tariff, full W ------------------------
                                 ^
                        every objective term slices [:, hmax:, :]
```

### Derive it from `t`, not from a per-window sample

> **Superseded — see the status block above.** This section is the plan's one
> substantive error, kept here because the argument it makes is the one that
> had to be answered. `t`-derived alignment is real, but it buys that property
> by making the tariff a function of the clock, which the policy can learn
> instead of the price. Price is now sampled per window from a real historical
> series, with drawn offsets and the same seed discipline as the band.

The comfort band is *sampled* — it is a synthetic scenario, so it needs a `torch.Generator`
and a fixed `EVAL_BAND_SEED` to stay reproducible. A tariff is not synthetic: it is a real
function of wall-clock time, and `t` (POSIX seconds) is already sliced per window and
already correct across multi-file pooling. So the hook is:

```python
def price_signal(t: torch.Tensor) -> torch.Tensor:
    """[n, W, 1] tariff aligned to the timestamps in t ([n, W, 1] POSIX seconds)."""
```

with a default template implementation that is a time-of-use table on hour-of-day, and a
documented alternative that reads a `price` column out of the dataframe. Two consequences
worth having:

- **Alignment is not a thing you can get wrong.** The price at index `i` is by construction
  the price at the timestamp of `yn[i]`. No "step 0 is midnight" assumption anywhere.
- **The baseline comparison stays apples-to-apples for free.** `evaluate_baseline`
  re-enters `evaluate_closed_loop`, which rebuilds the window dict; a `t`-derived price is
  byte-identical across those two calls with no seed discipline required.

### Units: keep price dimensionless in the dict, one constant in the hooks file

Everything in the window dict is normalized; a raw `$0.34/kWh` sitting next to normalized
states would be both an inconsistency and badly conditioned as a network input. Give the
hooks file a single declared constant:

```python
PRICE_SCALE = 0.30    # $/kWh; the dict carries price / PRICE_SCALE
```

The dict holds `price / PRICE_SCALE` (O(1), and **shift = 0**, deliberately — a zero price
must stay a zero cost, and price must stay non-negative so the cost gradient keeps its
sign). The cost term multiplies `PRICE_SCALE` back in when it wants dollars. This is the
existing `shift`/`stretch` convention with `shift = 0`, so it needs no new machinery.

---

## Part 2 — The policy must preview the price

**A cost objective with no price preview is not solvable.** Precooling is an action taken
*before* the expensive block; a policy whose only price input is `price_t` learns "reduce
load when it is expensive right now", which is the reactive strategy, and it will look like
it partly works because peak price correlates with peak ambient.

The mechanism exists — extend `_preview_steps`'s allowed keys from
`set(nm_mapping["D"]) | {"band"}` to also include `"price"`, return a third preview width,
and add `n_price * (1 + price_future)` to the `insize` sum at
[policy.py:264](../src/neurotemplate/control/policy.py#L264). Add `"price"` to the Node's
input keys and its `input_map` with `past=0`. The `describe_policy` window-width report
gains a `price_preview` field, which is what lets a reader re-check the `insize` arithmetic
by eye — a wrong window is otherwise an invisible failure.

**Preview width should cover the price cycle, not a few steps.** A 24-hour-ahead tariff is
what utilities actually publish, and the grid_response notebook previews 4 hours at
5-minute resolution (48 steps) for a single-zone building. At 48 steps × nd this starts to
dominate the input width; a downsampled preview (every 12th step over 24 h) is the obvious
next move if the flat window gets unwieldy, but do not build it until the flat one is
measured.

**Diagnostic, not an error:** when `cost_weight > 0` and the resolved price preview is 0,
print a line in the objective report saying the policy can react but not shift. That is
consistent with how this module already treats `term_shapes` — report, do not judge.

---

## Part 3 — The cost term

```
cost = mean_t( price_norm · PRICE_SCALE · power )        [$/h]
```

**Reduce with a mean, not a sum.** A sum over the horizon makes the term's magnitude
proportional to `train_pred_steps`, so a weight tuned at P=48 silently means something else
at P=288. Every existing term is a mean over the scored horizon; keep that invariant so
weights are horizon-independent, and let the reportcard do the integration into `$/window`.

**Gradient.** `∂cost/∂power_t = price_t / P`. Price carries no gradient (it is data), and it
is non-negative, so the cost term is a *reweighting* of the existing energy gradient — the
direction at each timestep is unchanged, only the magnitude varies over the horizon. That is
exactly the property that makes it safe to add next to `energy`: it cannot flip the sign of
anything the effort term was already doing.

**One real gradient hazard, inherited.** The recommended `power_metric` shape ends in
`.clamp_min(0.0)` ([control_user_defined.py:210](../template/control_user_defined.py#L210)).
Wherever the surrogate predicts negative power, that clamp zeroes the gradient. For a mean
energy term this is a rounding detail; for a cost term it is not, because the near-zero-power
steps are disproportionately the cheap hours the controller is supposed to be *moving load
into*. Add `frac_power_clamped` to the objective diagnostics dict. If it is large, that is a
surrogate problem (a power head that goes negative), not a weighting problem.

---

## Part 4 — The peak term

This is the harder one, and the instinct in the request — "softmax, or L2, or L4, the
softmax is kind of the limit" — is exactly right. Make it precise.

### Which family

Let `p_t ≥ 0` be power over the scored horizon, length P.

| Form | Definition | Limits | Homogeneity |
|---|---|---|---|
| mean square | `mean(p²)` | — | degree **2** |
| power mean | `M_q(p) = ( mean(p^q) )^{1/q}` | q=1 → mean, q=2 → RMS, q→∞ → max | degree **1** |
| smooth max | `S_β(p) = (1/β)[ logΣexp(β p) − log P ]` | β→0 → mean, β→∞ → max | **none** |

Two observations decide it.

**Mean-square is variance in disguise, and its weight is unit-dependent.**
`mean(p²) = mean(p)² + var(p)` — so an L2 term is "total energy squared, plus load
variability". It does flatten the profile, but it is homogeneous of degree 2: switch the
power metric from kW to W and the correct weight moves by 10⁶, while every other term's
weight moves by 10³. Given that mis-scaled weights are this library's most frequent recorded
failure (four campaigns out of four, per `objective.py`'s own docstring), a term whose
weight scales *differently from its neighbours* is a trap worth refusing.

**The power mean `M_q` is the same idea without the trap.** It is homogeneous of degree 1,
so it has the same units and the same weight-scaling behaviour as the existing energy term —
a reader comparing `energy_weight` and `peak_weight` on the balance report is comparing like
with like. And it interpolates continuously from mean (q=1) to max (q→∞) with a single
dimensionless knob.

**The smooth max is the one to avoid as a default**, precisely because `β` carries units of
1/power. Choosing it requires knowing the plant's power scale, and choosing it wrong gives
either the mean (β too small) or a one-hot gradient (β too large) — with no visible symptom
either way.

**Recommendation: `peak = M_q(p)` with `peak_q` defaulting to 4.** Fail fast if `peak_q < 1`.
Document `β`-LSE in the docstring as the alternative for anyone who wants a true smooth max
and is willing to own the scale.

### Gradient

```
∂M_q/∂p_t = (1/P) · (p_t / M_q)^{q-1}
```

Dense (every timestep gets some gradient, unlike a hard `max`, which is one-hot per window),
and concentrated on the high-power steps by a factor that grows with `q`. This is the whole
reason not to use `torch.max`: with a hard max, a 288-step rollout produces one non-zero
gradient entry per window, and the policy learns to shave exactly one timestep.

`p ≥ 0` is required — `p^q` for even `q` would otherwise make large negative predictions
look like a peak. The clamp in `power_metric` covers this; assert it rather than assume it.

### Reduce per window, then average over the batch — never pool

The batch axis holds independent rollout windows, typically different days and possibly
different source files. A peak pooled across the batch is the worst day's peak, which gives
gradient to one window and none to the rest. Per-window peak, then batch mean:

```python
peak = (power.clamp_min(0.0).pow(q).mean(dim=1)).pow(1.0 / q).mean()
#                                      ^^^^^^^^^ time            ^^^^^^ batch
```

This also lines up with `EnergyMetrics.mean_window_peak`
([metrics.py:239](../src/neurotemplate/control/metrics.py#L239)), which is already the
demand-charge-shaped metric on the reportcard — so the thing being trained and the thing
being reported are the same quantity, hard vs. soft.

### The honest limitation: the window is not the billing period

A demand charge bills the maximum 15-minute average over a *month*. The training rollout is
`train_pred_steps` long. Within-window soft-peak minimization is a **proxy** for that, and
it is only a good proxy if the window spans the period where the peak actually occurs.

Two practical consequences, both worth stating in the config comments:

- **The default `train_pred_steps: 48` is too short for either new objective.** At 5-minute
  data that is 4 hours — shorter than the precool-plus-coast cycle a price signal is asking
  for, and it cannot see a daily peak. Expect to run 144–288 steps (12–24 h) for grid
  response, which raises rollout memory roughly linearly. This is the main compute cost of
  the whole feature.
- **Peak alone will happily raise total energy.** Flattening a profile by adding off-peak
  load is a valid solution to `min M_q(p)` and a bad outcome. Always carry a non-zero
  `energy_weight` or `cost_weight` alongside `peak_weight`; the balance report will show
  whether the peak term is running the show.

If true cross-window demand charges are needed later, the extension is a running peak
carried as a rollout signal (`peak_t = softmax(peak_{t-1}, p_t)`, penalized at the final
step, with the running peak fed to the policy as an input). That is a genuinely different
piece of work — a new stateful node, a new seeded buffer, and the `start_iter` off-by-one
hazard documented in the control transform plan. **Do not build it in this pass.**

---

## Part 5 — Assembling the loss

### Keep flat weights; kill the boilerplate, not the branching

The tempting move is a term registry — a dict of builders keyed by name, driven by an
`objective: {comfort: 50, cost: 1.0, ...}` config block. Recommend **against** it:

- Five terms is five `if weight > 0:` blocks, ~140 lines, fully greppable. A registry
  converts that into a dict of closures plus a config schema, which is harder to read and
  buys nothing until terms are user-supplied.
- Flat `*_weight` fields keep `validate()`, `--set cost_weight=0.5`, and the MLflow
  hyperparameter log working unchanged. A nested dict breaks all three.
- No config migration, so every existing YAML and every test keeps running.

What *is* worth extracting is the four-line `expr == 0; update_name(...)` incantation
repeated per term, into one small helper. That is the actual duplication.

The one genuinely shared piece is the power signal: `energy`, `cost` and `peak` must all
read the *same* `power_metric` output, evaluated once. Build the `variable([yn, u, d], _power)`
once when any of the three weights is non-zero, and reduce it three ways.

### New config fields

```yaml
cost_weight: 0.0      # weights mean_t(price · power); requires the price hook
peak_weight: 0.0      # weights the per-window power mean M_q; requires power_metric
peak_q: 4             # 1 = mean, 2 = RMS, larger = closer to a hard max
```

Rename note: `effort_weight` now sits next to two other power-derived weights, and "effort"
does not distinguish it. Renaming it to `energy_weight` is the honest fix; it touches the
config, the YAML template, the loss-name strings in
[cli/train.py:126-134](../src/neurotemplate/control/cli/train.py#L126-L134), the golden
reportcard, and several tests. Worth doing in the same pass — a config where three
power-derived weights are named `effort`, `cost` and `peak` invites exactly the kind of
mis-set weight this module was written to catch — but it is a mechanical rename, so do it
*last*, after the behaviour lands.

### Fail-fast rules, in the existing style

`POWER_METRIC_IMPLEMENTED` already gates `effort_weight > 0` at problem build. Mirror it:

- `cost_weight > 0` requires **both** `POWER_METRIC_IMPLEMENTED` and a `price_signal` hook →
  `NotImplementedError` naming which one is missing and how to turn the term off.
- `peak_weight > 0` requires `POWER_METRIC_IMPLEMENTED`.
- `peak_q >= 1`, all new weights `>= 0`, in `validate()`.

The one thing *not* to do is fall back to a proxy when a hook is missing. The template's own
docstring records that quietly substituting an actuation proxy for a real power metric cost
three of four campaigns; a silently-substituted flat tariff would be the same mistake in a
new place.

---

## Part 6 — Gradient audit: what silently breaks

Things to verify explicitly, because each fails quietly and produces plausible numbers:

1. **Horizon slice.** Every new term slices `[:, hmax:, :]`, matching the existing ones. A
   term that scores the warmup is scoring seeded ground truth and gets no gradient from it.
2. **Price/power co-indexing.** `price[:, hmax:]` must line up with `power(yn[:, hmax:], ...)`.
   Deriving price from `t` makes this automatic; a test that shifts the tariff by one step
   and asserts the cost changes is the cheap guard.
3. **Price carries no gradient.** It is data, so `requires_grad` is False and
   `price · power` differentiates only through power. Assert it once.
4. **Power reaches the policy at all.** The chain is
   `θ → U → (STE quantize) → surrogate → yn → power_metric → term`. It only closes if the
   power channels are things the *surrogate predicts*, or a differentiable function of `U`
   and `D`. A power metric that reads only recorded `D` has a zero gradient and will train
   to nothing while looking fine. Test: `torch.autograd.grad(term, policy_params)` is
   non-zero for each new term.
5. **The clamp.** See Part 3 — report `frac_power_clamped`.
6. **Setpoint quantization interacts.** `setpoint_step: 0.5` with a straight-through
   estimator means the policy can only shape load in half-degree jumps. Fine for comfort,
   coarse for price modulation. Not a bug, but the first knob to loosen if cost barely moves.
7. **`du_weight` fights the price signal.** A tariff step at an hour boundary asks for a
   setpoint step; `du_weight` penalizes exactly that. Expect to re-tune it, and read the
   balance report rather than guessing.

---

## Part 7 — The surrogate is the real limit

Worth saying plainly before anyone spends a week on weights: **price-driven control asks the
surrogate questions it may never have been asked.** Precooling means driving the setpoint
several degrees below anything in the identification data and trusting the predicted thermal
storage. If the training data came from a fixed schedule, that response is extrapolation and
the resulting policy is optimizing a fiction.

The reachability scenario on the sys-id reportcard is already the place this gets checked
(see [cli/train.py:96-98](../src/neurotemplate/control/cli/train.py#L96-L98)). Read it for
the *precool* direction specifically — not just "can the band be held", but "does the
surrogate show real storage when driven hard and then released". If it does not, the fix is
excitation in the sys-id campaign, not a weight.

---

## Part 8 — Metrics, reportcard, plots

Mostly additive, because the diagnostics are already generic:

- **Free:** `objective_balance` reads term names off `loss.objectives`, so `cost_loss` and
  `peak_loss` appear on the pre-training balance report, in `objective_diagnostics.json`,
  and in the reportcard's objective section without any change.
- **`EnergyMetrics`** gains `energy_cost` (per-window `Σ π·p·Δt`, in dollars). `energy_metrics`
  takes an optional price tensor; `None` price keeps today's behaviour exactly.
- **`evaluate_closed_loop`** already denormalizes and calls `power_metric`
  ([pipeline.py:210-216](../src/neurotemplate/control/pipeline.py#L210-L216)); the price is
  in `out["price"]` by passthrough, so this is a two-line change.
- **Baseline comparison** picks up cost automatically via `_comparison_values`, which gives
  the headline number this whole feature exists to produce: dollars per window, learned
  policy vs. the rule-based controller, on identical windows and an identical tariff.
- **Plots:** price belongs on the closed-loop figure — as a shaded background or a third
  axis against the power row. Add it to the trajectory CSV too; a price trace next to the
  power trace is how you see load shifting at a glance, and it is what makes a failed
  precool obvious.
- **`term_shapes`** sweeps the power signal across each actuation box. Keep it bound to
  `power`, not to the individual terms — one sweep serves all three power-derived terms,
  and sweeping `cost` would just be the power sweep scaled by a constant.

---

## Phasing

Each phase ends somewhere runnable, and none of the early ones can change existing results.

**Phase 0 — plumbing, weights at zero.** `price_signal` hook + `PRICE_SCALE`, `price` in the
window dict, price on the plots and in the CSV. Weights stay 0, so every existing run is
byte-identical. Test: shapes, `t`-alignment, multi-file pooling, and that a run with
`cost_weight: 0` matches the golden reportcard.

**Phase 1 — price preview.** Extend `_preview_steps` and `insize`; `describe_policy` reports
`price_preview`. Test: insize arithmetic for each preview combination, and that the policy
output actually changes when the previewed price changes.

**Phase 2 — the cost term.** `cost_weight`, the shared power variable, the fail-fast checks,
the "no preview" diagnostic line. Test: cost is zero when price is zero; cost is
price-weighted mean power on a hand-computed batch; gradient reaches the policy parameters.

**Phase 3 — the peak term.** `peak_weight`, `peak_q`. Test: `M_1` equals the mean and `M_q`
is monotone in `q` on a fixed signal; a spiky and a flat profile of equal total energy score
differently and the flat one scores lower; per-window-then-batch reduction verified against a
hand-computed two-window case; gradient is dense over time (not one-hot).

**Phase 4 — reporting.** `energy_cost` in `EnergyMetrics`, the MLflow flat keys, the baseline
cost comparison, `frac_power_clamped`. Regenerate the golden reportcard.

**Phase 5 — the `effort_weight` → `energy_weight` rename.** Mechanical, and last.

---

## Non-goals

- **User-supplied loss terms.** Cost and peak are plant-agnostic and belong in the library;
  a hook for arbitrary terms is speculative extensibility until something needs it. If it
  ever does, it slots in at Layer 2 as one more reduction of the same power signal.
- **True cross-window demand charges** (running peak as rollout state) — see Part 4.
- **A nested `objective:` config block** — see Part 5.
- **Tariff realism**: demand ratchets, TOU-plus-demand two-part tariffs, 15-minute rolling
  averages. The hook returns a per-step price; anything more elaborate is a tariff detail and
  belongs in the use case, which is the same line `energy_metrics` already draws.
