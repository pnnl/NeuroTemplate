# Plan: what v6 says to change about the price and residual features

## Status: Parts 2–8 implemented on `curated_v2`; Parts 1 and 9 not started

Parts 2, 3, 4, 5, 6, 7 and 8 landed as eight commits (`71f28ce`..`4b21724`),
with 439 tests green. Part 1 (the counterfactual-tariff audit) and Part 9
(multi-seed) were deliberately left out of that pass.

**Two things came out different from the plan below, both found by running it:**

1. **Part 8 needed to align to the *plant's* clock, not the price file's.**
   The plan said "snap the offset to a whole-day boundary in the price file",
   which pins the within-window phase but not the hour — an end-to-end run put
   a 16:00–21:00 tariff at 22:35–03:30, because the plant window started at
   06:55. The shipped version solves the offset against each window's own time
   of day, read from the plant frame's `DatetimeIndex`. Same run now reports
   the expensive block at 16:00–20:55 in every window. It requires a plant
   frame with a real `DatetimeIndex` and says so.
2. **Part 6's inert-term warning had to be gated on the sweep as well.** A
   term computed only from the commands has no gradient into `yn` either, so
   `reads_outputs` alone would have called it inert. `is_inert` requires both:
   flat in every control *and* no gradient into the rollout.

`price_align` defaults to `random`, so no existing config changes behaviour.
Four signatures moved and all four fail loudly — see the upgrade table in the
main README.

Source: `SANITY_CHECK_V6.md` (modelica_rtu_mall), the first campaign to run
`cost_weight`/`peak_weight` and `residual_delta` end to end on a real plant and
a real PG&E tariff. Both features worked — the campaign's headline is that the
price response is genuine and the ±Δ box was not binding — so nothing here is a
rescue. This is the list of things the campaign had to work around, plus the two
places a diagnostic told the operator something that was not true.

The ordering below is by evidence strength, not by size. Part 9 is the third
campaign in a row asking for the same thing and is included on that basis
alone. Sections marked ✅ are implemented; the text is left as written so the
plan and the outcome can be compared.

| # | Change | Why it is on the list | Size |
|---|---|---|---|
| 1 | Counterfactual-tariff audit + price-tercile power table | The only evidence separating price response from weather correlation; v6 had to write it app-side | ~200 lines + tests |
| ✅ 2 | Hourly price CSVs cannot be loaded | The README's own canonical recipe raises | 3 lines |
| ✅ 3 | Closed-loop plots dated 1970 | Undercuts every price figure the feature exists to produce | ~10 lines |
| ✅ 4 | Gradient-measured dead-power diagnostic | Current `frac_power_zero` warning is wrong for ordinal heads | ~15 lines |
| ✅ 5 | `frac_at_cap` is meaningless at Δ = 0 | The library recommends running Δ = 0 and then misreports it | ~5 lines |
| ✅ 6 | `term_shapes` says "no direct effect" about a term that has one | v5 finding 3, reproduced unchanged in v6 | ~15 lines |
| ✅ 7 | `price_preview` belongs in the config, not the hook | One config wanting price preview forces price paths on every config sharing the hook file | ~15 lines |
| ✅ 8 | `price_align: random \| day` | Eval tariffs cannot be put on the plant's clock; workaround is a degenerate one-window price file | ~25 lines |
| 9 | Multi-seed run mode | Third campaign asking; v6 has the sharpest evidence yet | ~150 lines |

---

## Part 1 — Move the counterfactual-tariff check into the library

**The finding.** v6's c03 posts a textbook load-shifting trace: zone driven to
0.77 of its comfort band through the cheap midday hours, coasting to 0.88
through the expensive evening, DX power falling 2650 W → 900 W across the same
span. That trace is *not evidence*. On this plant the cooling load peaks
mid-afternoon and the tariff peaks right after it, so "reads the price channel"
and "reads the weather and happens to line up" produce the same picture. The
pooled cost metric cannot separate them either, because it is lower for a policy
that simply uses less energy.

What settles it is re-scoring the *trained* policy with everything held fixed —
same weights, same weather, same fixed-seed band, same surrogate — and only the
tariff rewritten. v6 measured the peak-window power ratio inverting from 0.71
(real 8 pm peak) to 1.22 (same prices rolled to a 7 am peak), with total
electricity nearly unchanged. That is load moving rather than load disappearing,
and no other number in the reportcard says it.

This is ~170 lines living in the app at
`modelica_rtu_mall/experiments/analysis/v6_price_counterfactual.py`. Every app
that turns on `cost_weight` needs it, and most will not write it.

### The design, and where it improves on the app version

New module `src/neurotemplate/control/price_audit.py`, called from
`cli/train.py` after `evaluate_closed_loop` whenever `cfg.cost_weight > 0` and a
tariff is configured.

**Generate the counterfactual tariffs in memory, not from extra CSVs.** The app
had to write `test_price_flat.csv` and `test_price_shifted.csv` by hand. Both
are pure functions of the real test tariff tensor:

```python
def flat_price(price):    # same mean, no schedule
    return price.mean(dim=1, keepdim=True).expand_as(price)

def shift_price(price, steps):  # same values, same spikiness, moved
    return price.roll(steps, dims=1)
```

Roll by half a window (`W // 2`), which on a 24 h evaluation window is v6's
12 h shift without hard-coding a day length.

**Feed them in through one new optional argument.** `evaluate_closed_loop`
(`control/pipeline.py:140`) gains `price_transform: Optional[Callable] = None`,
applied to `data["price"]` immediately after `build_window_dict_multi`
(`pipeline.py:171-175`) and before the rollout. One parameter, one line in the
body, and the band generator is already reseeded from `EVAL_BAND_SEED` inside
the function, so all three scorings get a byte-identical band and identical
weather for free. Nothing else in the eval path changes.

**Score by price tercile, not by hour of day.** The app version defined "peak
hours" as the hours where the real tariff exceeded $1/kWh, which needs
hour-of-day — and hour-of-day is exactly what Part 3 shows is broken. Instead,
split each window's steps into terciles of the *real* tariff and hold that step
partition fixed across all three rows. It is clock-free, correct for any window
length, and it is simultaneously the answer to a second finding (below).

```python
@dataclass
class PriceResponse:
    label: str              # "real" | "flat" | "shifted"
    violation_rate: float
    mean_power: float
    power_cheap: float      # mean power over the cheapest third of steps
    power_expensive: float  # ... the most expensive third, per the REAL tariff
    ratio: float            # expensive / cheap
    cost_on_real: float     # billed on the real tariff whatever was previewed
```

`cost_on_real` matters: billing every row on the real tariff is what turns the
audit into a number an operator cares about (v6: $13.83/day reading the tariff,
$15.88 blind, $17.33 mis-timed).

**Report, do not judge — with one note.** Consistent with `objective_notes`
(`control/objective.py:177`), emit a single line when the real row's ratio is
not below both counterfactual rows: *"the load profile did not move when the
tariff moved; this run's cost saving is energy reduction, not load shifting."*
That is the whole verdict, and it is a comparison of three floats.

### The same table answers the shift-vs-shed finding

v6's c02 followed the library's own suggested weights, passed the pre-training
balance check (cost 16%, energy 8%, peak 2%, comfort 69%), and produced the
wrong behaviour: mean power fell 78% with violations rising 7.8% → 29.6% and the
cooling setpoint pinned at its *upper* bound 20.3% of the evaluation. It stopped
cooling. `energy_weight` rewards using less power at every hour; `cost_weight`
rewards using less power at the expensive hours. Objective balance measures
magnitude and is structurally unable to tell those apart.

The tercile columns flag it in one line — a shedding policy cuts power in every
tercile, so `ratio ≈ 1` while `mean_power` collapses and comfort degrades. No
second diagnostic needed. Emit the tercile table whenever a tariff is
configured, even at `cost_weight = 0`, so the price-blind anchor run has a
comparable row.

### The reload gap this exposes

The app script had to glob `mlruns/*/{run}/artifacts/best_model_state_dict.pth`
and re-assemble the problem by hand to get a trained policy back. `control/load.py`
loads a frozen *surrogate* and nothing loads a trained *policy*. If the audit is
to be re-runnable outside the training process — and it should be, since it is
an eval-only question asked about a run that already finished — add
`load_trained_policy(control_run_id, ...)` next to `load_frozen_dynamics`,
mirroring its signature. ~40 lines, and it is the difference between the audit
being a library feature and being a thing you can only get if you thought of it
before training.

**Tests.** Two, both synthetic and fast: a policy whose commands are a fixed
function of the price input must produce `ratio(real) < ratio(shifted)`; a
policy ignoring the price key must produce three near-identical rows.

---

## ✅ Part 2 — An hourly price CSV cannot be loaded at all

`common/data_utils.py:66-69`:

```python
if df_freq == 'min':
    freq_seconds = 60
else:
    freq_seconds = pd.to_timedelta(df_freq).total_seconds()
```

`pd.infer_freq` returns a bare `"h"` for an hourly file and `pd.to_timedelta("h")`
raises `ValueError: unit abbreviation w/o a number`. The `min` special case is
there because the same thing happens for `"min"`. So the bug hits every
frequency string without a numeric multiplier — `h`, `D`, `W` — which includes
the README's own canonical price recipe (`price_repeat: 12`, "hourly price over
5-minute plant data"). v6 worked around it by pre-upsampling every price file to
5 minutes and setting `price_repeat: 1`, which means `price_repeat` has never
been exercised as documented.

Fix, verified against the installed pandas — it also deletes the `min` special
case rather than adding a second one:

```python
freq_seconds = pd.Timedelta(pd.tseries.frequencies.to_offset(df_freq)).total_seconds()
```

`h → 3600`, `D → 86400`, `min → 60`, `5min → 300`, `15min → 900`, `2h → 7200`.
Non-fixed offsets (`W`, `M`) still raise, now with a message that names the
problem. The existing `assert freq_seconds > 0` stays.

**Test.** Parametrize the existing `get_data` test over `["5min", "h", "D"]` and
assert the returned `freq_seconds`.

---

## ✅ Part 3 — Every closed-loop plot and trajectory CSV is dated 1970

`control/pipeline.py:499` says "sec_elapsed is POSIX seconds". It is not:
`data_utils.py:76` rebases it with `df['sec_elapsed'] -= df['sec_elapsed'].iloc[0]`.
So `pd.to_datetime(t_unix[w], unit="s")` at `pipeline.py:513` lands on the epoch,
and v6's August test week produced day folders named `Jan_01 … Jan_07` and figure
titles reading `1970-01-01`. Hour-of-day survives only by luck — every split in
that campaign happens to start at midnight.

For a feature whose stated goal is prices that make sense to buildings people,
figure titles that disagree with the data are worse than no titles.

**The origin is not lost, it is just not plumbed.** `get_data` sets a
`DatetimeIndex` (`data_utils.py:63`) and `normalize_df` copies the frame, so
`test_dfs[i].index[0]` is the real start timestamp of file `i`. `plot_closed_loop`
already carries `out["source_idx"]` (`pipeline.py:186`) to route windows back to
their file.

Note that the modeling side already treats `sec_elapsed` correctly as *elapsed*
(`modeling/plotting.py:75` reconstructs `df_true.index[0] + to_timedelta(...)`).
Control is the half that is wrong, so the fix goes in control — do **not** stop
rebasing in `data_utils`, which would double-count the origin in modeling's plots.

**Change.** `plot_closed_loop` takes `origins: list[pd.Timestamp]` (built in
`cli/train.py` as `[df.index[0] for df in test_dfs]`), and line 513 becomes:

```python
ts = origins[int(source_idx[w])] + pd.to_timedelta(t_unix[w], unit="s")
```

Day folder names, the figure title, and the CSV index all follow. Fix the
misleading comment at `pipeline.py:499` in the same edit.

**Test.** Build a two-file test split with known non-midnight, non-1970 start
timestamps and assert the emitted CSV index matches.

This is also a prerequisite for Part 8: aligning a tariff to the plant's clock
is not meaningful while the plant's clock reads 1970.

---

## ✅ Part 4 — Measure the dead gradient instead of inferring it

`objective.py:157` reports the share of scored steps where the power signal is
exactly zero, and `format_objective_report` (`objective.py:335`) prints it as
"clamped => no gradient there".

That warning did real work in v6: 18.4% of scored steps were clamped under plain
regression heads (the surrogate was predicting a median of −101 W of DX cooling
at equilibrium), versus 0.0% under a lower-bounded head — and switching heads
also produced the best summer zone NRMSE in the repo's record. First time a
pre-training diagnostic changed a modeling decision.

**Then the same diagnostic was wrong.** Against an `OrdinalHead` it reported
9.0% and the warning is false: an ordinal head snaps through a straight-through
estimator, so the backward path is the smooth sum-of-sigmoids and gradient still
flows at a predicted zero. It was counting correct compressor-off predictions —
79% of the truth on that plant — as a defect.

The report's suggested fix is to teach the diagnostic about head kinds. Don't:
that is a table of head types to keep in sync, and it will be wrong again the
next time someone adds a head. **Measure the quantity the warning actually
claims.** `clamp_min` passes gradient at exactly zero and blocks it below
(verified: input −1 → grad 0, input 0 → grad 1), which is precisely the
distinction, so one autograd pass separates the two cases without knowing
anything about the head:

```python
yn = out["yn"][:, hmax:, :].detach().requires_grad_(True)
power = term_fn(yn, U, D)
power.sum().backward()
dead = (power == 0) & (yn.grad.abs().sum(-1) == 0)
```

Report `frac_power_zero` (forward value, unchanged) alongside `frac_power_dead`
(zero *and* no gradient), and attach the "no gradient there" wording only to the
second. On the v6 runs this reads 18.4% / 18.4% for the regression head and
9.0% / 0.0% for the ordinal one — the ordinal head stops looking broken, and the
regression head still looks exactly as broken as it was.

**Test.** Two synthetic term functions — one ending in `clamp_min(0)` over
negative inputs, one returning hard zeros through a straight-through
`x + (x.round() - x).detach()` — must give the same `frac_power_zero` and
different `frac_power_dead`.

---

## ✅ Part 5 — `frac_at_cap` at Δ = 0

`metrics.py:366`:

```python
frac_at_cap=float((delta >= delta_cap - tol).float().mean()),
```

At `delta_cap = 0` every deviation is trivially at the bound of a zero-width
box, so the reportcard prints "Pinned at ±Δ on 100.0% of steps"
(`reportcard.py:271`) followed by the standing advice that a cap binding most of
the time means the reference is wrong or the optimizer is riding a surrogate
artifact into the wall (`reportcard.py:272-274`). Neither sentence means anything
about δ ≡ 0 — which is the configuration the library itself recommends running
first, to score the bare rule.

Make `frac_at_cap: Optional[float]`, `None` when `delta_cap == 0`, and have the
reportcard render `n/a (Δ = 0: the policy is the reference rule)` and suppress
the advice paragraph. `mean_abs_delta` and `max_abs_delta` stay — at Δ = 0 they
are a useful assertion that the residual path really is inert.

**Test.** `residual_metrics(u, u, delta_cap=0.0, tol=...)` returns
`frac_at_cap is None`.

### While here: one sentence next to "clears its reference"

v6's sharpest incidental finding: the identical fixed rule scores 36.59%
violation against the bounded-head surrogate and 64.23% against the ordinal one,
from surrogates whose zone NRMSE differs by 3%. A smooth power head sells the
rule fractional cooling that a three-stage plant cannot deliver, so the smooth
surrogate was flattering the rule — and `clears_reference`, which is the
ship/don't-ship verdict, is scored entirely inside that surrogate.

There is no cheap measurement of "actuation realism" to add here, but the
verdict should not be printed bare. One line in the reportcard next to it: *the
reference is scored against the same surrogate as the policy, so its bar moves
with the surrogate's actuation realism; a smooth power head can hand the rule an
actuator the plant does not have.* Free, and it is the caveat that a reader of
that verdict most needs.

---

## ✅ Part 6 — `term_shapes` reporting "no direct effect" about a term that has one

v5 finding 3, reproduced unchanged in v6. `term_shapes` (`objective.py:203`)
sweeps each control across its actuation box and evaluates `term_fn(yn, swept, d)`.
A metered `power_metric` reads power channels out of `yn` and ignores `U`
entirely, so the sweep is flat and `shapes_to_markdown` (`objective.py:274`)
renders "flat (no direct effect)". The term has a very direct effect; it just
reaches the policy through the dynamics rather than through the command.

The diagnostic exists to catch a power term that cannot influence anything, and
against metered power it is structurally blind to that risk while sounding
confident.

**Fix, sharing Part 4's machinery.** When a sweep is flat in `U`, measure
whether the term is flat in `yn` too, and say which:

- flat in `U`, non-zero gradient w.r.t. `yn` → *"reads model outputs, not
  commands; gradient reaches the policy through the dynamics"*
- flat in both → *"this term is inert"*, which is the real alarm and is
  currently indistinguishable from the first case.

One shared helper `_term_gradients(term_fn, yn, U, D) -> (du, dyn)` serves this
and Part 4.

**Test.** A `term_fn` reading only `yn` and one returning a constant must
produce different messages.

---

## ✅ Part 7 — `price_preview` belongs in the config

`policy._preview_steps` (`policy.py:288-313`) reads `preview_config(nm_mapping)`
and raises when a `"price"` entry is requested with no `train_price_path`. The
hook cannot see the config, so it cannot ask whether a tariff exists. Consequence
in v6: once one config in the app wanted a price preview, **every** config
sharing that hook file had to carry price paths — the v5 configs stopped loading
for that reason alone.

Don't widen the hook signature to take `cfg` (it breaks every app's hook file,
loudly but pointlessly). The observation that resolves this: how many D columns
to preview is *structural* — a property of the plant — while how far ahead to
look at the tariff is a *per-run tuning knob*, which is why it collided with
per-run configs. `cfg.d_preview` already exists as the config-side preview
width, so there is precedent and one obvious home.

- Add `price_preview: int = 0` to `ControlConfig`, beside the other price fields
  (`config.py:52-66`).
- Reject `"price"` as a `preview_config` key with a message naming the new field
  (`policy.py:293`).
- Move the consistency check into `ControlConfig.validate()`, next to the price
  validation that already lives at `config.py:187-215`: `price_preview > 0`
  requires the price paths. It fails at config load rather than at policy build,
  which is the earlier and better boundary.

`"band"` stays in the hook — it is structural and nobody has reported pain.

**Test.** A config with `price_preview: 48` and no price paths fails
`validate()`; a `preview_config` returning `{"price": 48}` raises a message
naming `price_preview`.

---

## ✅ Part 8 — `price_align: random | day`

`build_price` (`scenarios.py:170`) draws a uniformly random window offset into
the price file, independent of the weather window it is scored beside. The
docstring's argument for that is correct and should survive: it removes the
residual price↔weather correlation as a route to learning the tariff as the
clock.

But there is no opt-out, so a multi-day price file lands its 8 pm peak at a
random hour of the plant's day, and nothing in the config can pin it. v6's
workaround was to make the test price file *exactly one evaluation window long*,
leaving `randint` a single legal offset. It works and it is a hack: the eval
tariff is reproducible and clock-correct only because the file is degenerate.

**Add `price_align: str = "random"` with one alternative, `"day"`:** snap the
random offset to a whole number of days, so the day is still drawn at random but
the tariff's midnight lands on the window's midnight. Day-to-day price variation
— the thing the anti-shortcut argument actually rests on — is fully preserved;
only the intra-day phase is pinned. `steps_per_day` comes from the `t` tensor,
which `build_window_dict` already has in hand at `scenarios.py:227` before it
calls `build_price`.

Two things to state plainly in the docstring and in the field comment:

1. `"day"` partially re-opens the clock shortcut — on v6's series the price
   peaks at hour 19–20 on 151 of 184 days, so an hour-aligned tariff is that
   much more learnable as the hour.
2. Which is exactly why this is safe to offer now and would not have been
   before: **Part 1 detects the shortcut it re-opens.** A policy that learned
   the clock instead of the price produces three identical rows in the
   counterfactual table. Land Part 1 first.

**Not doing: `"clock"`** (slice the price by absolute timestamp against the
plant record). It requires the two records to overlap in wall-clock time, and
v6's train prices are deliberately from a different year than the plant record.
Real need, no demonstrated demand, skip until there is one.

**Test.** With `price_align: "day"` and a synthetic ramp price file, every
sampled window must start at the same within-day phase; with `"random"`, not.

---

## Part 9 — Multi-seed as a run mode

Not a v6 feature, and included only because v6 supplies the strongest evidence
yet after two campaigns of asking.

v5 localized seed volatility to one recipe and reported mlp as flat (0.9% swing
across seeds). v6 ran the same architecture on the summer split and got
**0.0913 / 0.1133 / 0.1276** zone NRMSE across seeds 1234 / 5 / 42 — a 40%
relative spread — and, worse, **the heating step-response sign check passes at
seed 1234 and fails at both other seeds.** Two of three seeds produce a surrogate
that will pull the heating setpoint backwards, and single-seed selection picks
one of the three blind. "mlp is seed-stable" was a statement about the spring
plant, not about mlps.

**Minimum viable version, and I would not build more than this.** A `--seeds
1234,5,42` flag on `nt model train` that loops the existing single-seed pipeline
as nested MLflow runs and writes one `seed_spread.md`: per-seed headline metrics
plus, in its own column, every physics check that *disagrees* across seeds. No
change to the training path, no aggregation, no automatic "best" pick — the
disagreement column is the whole product, because a check that flips across
seeds is a different and much louder signal than a metric that wobbles.

---

## Deliberately not on this list

- **Rescaling `energy_cost` into real dollars.** `power_metric` returns Watts,
  `price_scale` is $/kWh, and the reported cost is off by 1000 — which
  `metrics.py:84` and `reportcard.py:220-222` both say out loud. A
  `cost_unit_scale` field would fix the headline number and add a config knob
  whose only job is a constant an app can already apply in `power_metric`. Worth
  it only if the dollar figure is going to be quoted outside the reportcard.
- **A fuel-blended `power_metric`.** Open since v4 and the reason v6 had to
  abandon `data/spring` (the plant's real energy user there is a 61.7 kW gas
  furnace that `power_metric` excludes). It needs a gas price series and an
  explicit stance on the electricity/fuel trade — an app decision the library
  should not make.
- **Anything about the ordinal head.** v6 used it well and its findings
  (`POWER_LEVELS` must be re-measured per split; R² and MAE point opposite ways
  on a staged channel) are usage guidance, not library defects. Part 4 covers
  the one place the library was actively misleading about it.

## Phasing

1. **Parts 2, 3, 5, 6** — four small, independent corrections, all with the same
   character: a number or a label that says something untrue. Half a day
   together, and Part 3 unblocks Part 8.
2. **Part 1** — the headline. Land the audit and the tercile table before
   anything that makes the clock shortcut easier to learn.
3. **Parts 4, 7** — small, and both want Part 1's shared gradient helper and
   config-side precedent respectively.
4. **Part 8** — only after Part 1 and Part 3.
5. **Part 9** — separate, modeling-side, and independent of everything above.
