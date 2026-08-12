# Report: v6 follow-up, Parts 2–8

Branch `curated_v2`, nine commits on top of `a19616f`. **439 tests pass** (411
before; 28 added). Parts 1 (counterfactual-tariff audit) and 9 (multi-seed)
were excluded by request and are untouched — the plan sections for both are
left in place, unmarked.

Everything below is aimed at the same thing: making pre-cooling something the
library can produce, show, and be trusted about. Two of the changes were
blocking that outright (the tariff could not be loaded in its documented
format; every plot was dated 1970), one enables it (`price_align: day`), one
removes the friction that cost v6 its whole config set (`price_preview`), and
three stop diagnostics from saying things that are not true.

---

## What changed

### 1. An hourly tariff CSV now loads — `71f28ce`

`pd.infer_freq` returns bare `"h"` for an hourly file and `pd.to_timedelta("h")`
raises `unit abbreviation w/o a number`. Only `"min"` was special-cased, so
every frequency without a numeric multiplier (`h`, `D`) failed — including the
README's own canonical price recipe, "hourly price over 5-minute plant data
with `price_repeat: 12`". That recipe had therefore never worked; v6 pre-upsampled
every price file by hand to get around it.

Going through the frequency offset instead resolves the unit and multiplied
forms alike, so [data_utils.py:66](src/neurotemplate/common/data_utils.py#L66)
got shorter rather than gaining a second special case. Verified end to end: 24
hourly rows × `price_repeat: 12` → 288 five-minute steps, each hour held for 12.
New [tests/test_get_data.py](tests/test_get_data.py) covers `5min`/`15min`/`min`/`h`/`2h`/`D`
— `get_data` had no test at all before.

### 2. Plots carry the plant's clock — `59bf646`

`plot_closed_loop` read `t` as POSIX seconds; `get_data` rebases it to zero at
each file's first row. So every figure title, day folder and trajectory CSV was
stamped 1970-01-0X, and v6's August test week filed itself under `Jan_01..Jan_07`.
Hour-of-day survived only because those splits happened to start at midnight —
which is exactly the case where the bug is invisible.

The origin was never lost, only unplumbed: `get_data` leaves a `DatetimeIndex`,
and `out["source_idx"]` already said which file each window came from (it had
been plumbed for this purpose and never used). `plot_closed_loop` now takes one
origin per test file.

The fix went in control rather than in `data_utils` on purpose:
[modeling/plotting.py:75](src/neurotemplate/modeling/plotting.py#L75) already
treats `sec_elapsed` correctly as *elapsed*, so un-rebasing it there would have
double-counted the origin in the modeling plots.

### 3. `price_preview` moved from the hooks file to the config — `80ed1a9`

`preview_config` cannot see the config, so it could not ask whether a tariff was
configured — and requesting a price preview without one raised. The consequence
in v6: once one config wanted a price preview, *every* config sharing that hook
file had to carry price paths, and the entire v5 config set stopped loading for
that reason alone.

The split that resolves it: how far ahead to look at the *weather* is structural
(a property of the plant, shared by every run) and stays in the hook; how far
ahead to look at the *tariff* is a per-run knob and is now `price_preview` in the
control config, validated beside the other price fields. That also moves the
failure from policy build to config load. A leftover `"price"` key in
`preview_config` raises and names the field that replaced it.

### 4. `price_align: day` — the tariff on the plant's clock — `d265daf`, `87bbfab`

**This is the one that matters for pre-cooling.** `build_price` drew a uniformly
random offset into the price record with no opt-out, so a real 8 pm peak landed
at a random hour of the plant's day. The default is right — it is what stops
"read the price" collapsing into "read the clock" — but it means the pre-cool an
evening peak should provoke averages away and appears in no plot. v6's workaround
was to cut the eval price file down to exactly one window long, leaving `randint`
a single legal offset.

`price_align: day` chooses the offset so the tariff's time of day matches each
window's own, with the **day still drawn at random** — so day-to-day price
variation still forces the policy to read the channel, and only the time of day
is pinned. Default stays `random`, so no existing config changes behaviour.

You asked not to worry about time-of-day correlation for now, and this respects
that while stating the trade rather than hiding it: a series peaking at hour 19
on most days becomes that much more learnable *as* hour 19, so under `day` a
load-shifting trace is not by itself evidence of price response. That caveat is
in the config comment, the docstring and the README, and it points at the check
that would settle it (which is Part 1, not built).

The README also gained a **"If what you want is pre-cooling"** section, because
three of the four things that have to be true are not weights: the preview has to
cover the run-up to the peak; the comfort band has to be wide enough to store
anything (load shifting *is* thermal storage — a 0.5 °C band measures the band,
not the plant); the rollout has to span the cycle; and `cost_weight` has to carry
the power side, because `energy_weight` pulls the opposite way and buys shedding
instead of shifting. That last one is v6's c02 written down.

### 5. The dead-power gradient is measured, not inferred — `65510fb`

`frac_power_zero` reported the share of scored steps where power read zero and
printed "clamped ⇒ no gradient there". That warning did real work against a
regression head predicting −101 W of DX cooling at equilibrium. Against an
ordinal head it was simply false: a straight-through snap to a real compressor-off
stage reports zero and *still passes gradient*, because `clamp_min` is inactive at
exactly zero (verified: input −1 → grad 0, input 0 → grad 1). The diagnostic was
counting correct predictions as a defect on 79% of the truth.

The plan proposed teaching it about head kinds. I measured instead — one backward
pass separates the two cases without a head-type table to keep in sync, and it
cannot go stale when someone adds a head. `power_diagnostics` now reports
`frac_zero` (unchanged), `frac_dead` (zero *and* no gradient into the rollout —
the number that carries the warning) and `reads_outputs`.

### 6. A flat power sweep says which kind of flat it is — `65510fb`

v5 finding 3, reproduced unchanged in v6: `term_shapes` rendered a flat sweep as
"flat (no direct effect)", which is wrong for metered power — it reads the
surrogate's outputs and is very much in effect, it just reaches the policy through
the dynamics. The diagnostic exists to catch a term that cannot influence
anything, and it was structurally blind to that while sounding confident.

`reads_outputs` (same backward pass, no extra cost) tells them apart. A flat sweep
now reads "flat in U — reads model outputs", and "INERT" is reported only when the
term reads *neither* the commands nor the outputs. **One thing the plan got wrong
here**: gating on `reads_outputs` alone would have called a commands-only term
inert, since it has no gradient into `yn` either. `is_inert` requires both — flat
in every control *and* no gradient into the rollout.

### 7. A zero-width residual box no longer reports 100% saturation — `60bbc31`

At Δ = 0 every deviation is trivially at the bound of a box with no interior, so
`frac_at_cap` came out 1.0 and the reportcard printed "Pinned at ±Δ on 100.0% of
steps" followed by its warning that a binding cap means the reference is wrong or
the optimizer is riding a surrogate artifact into the wall. Neither applies to the
run the library itself recommends doing first — Δ = 0 is how the bare rule gets
scored. `frac_at_cap` is now `None` there and renders as n/a with the warning
suppressed; mean |δ| keeps its place as the assertion that the residual path is
inert.

The binding-cap advice also gained the third case v6 demonstrated (c07): an
incentive that has outgrown what ±Δ can express, which is the cap working rather
than a warning sign.

### 8. The reference's bar moves with the surrogate — `f61384a`

`clears_reference` is the ship/don't-ship verdict and it is scored entirely inside
the frozen surrogate — for the rule as well as the policy. v6 measured the
identical fixed rule at 36.59% violation against a bounded-head surrogate and
64.23% against an ordinal one, from a 3% difference in zone NRMSE, because a
smooth power head hands the rule a fractional actuator the three-stage plant does
not have. There is no cheap measurement of actuation realism to add, but the
verdict should not be printed bare, so it now carries that caveat.

---

## Verification

Beyond the unit tests, I ran the CLI's whole sequence (minus MLflow/argparse) on
a config that exercises every change at once: an **hourly** tariff CSV with
`price_repeat: 12`, `price_align: day`, `price_preview: 48`, a cost objective, a
24 h rollout, and plots — over four days of 5-minute plant data starting at a
deliberately awkward **2024-08-02 06:35**.

```
price file : 480 HOURLY rows, $0.05-$0.55/kWh, peak 16:00-21:00
plant file : 1152 5-minute rows from 2024-08-02 06:35:00

Power term across each actuation box:
  Zone Cooling Setpoint [°C]   ...   flat in U — reads model outputs
  power exactly zero on 0.0% of scored steps
  ...of which 0.0% of all steps also pass no gradient back

policy reads price   : True, preview=48
plot day folders     : ['Aug_02', 'Aug_03', 'Aug_04']
  Aug_02   starts 2024-08-02 06:55   expensive block 16:00-20:55
  Aug_03   starts 2024-08-03 06:55   expensive block 16:00-20:55
  Aug_04   starts 2024-08-04 06:55   expensive block 16:00-20:55
```

Every line there was broken or misleading before: the price file would not load,
the folders would have read `Jan_01`, the sweep would have said "no direct
effect", and the expensive block would have landed at a different hour in each
window.

**This run is what caught the one real design error.** The first cut of
`price_align: day` snapped the offset to a whole-day boundary in the *price*
file, which pins the within-window phase but not the hour — that same 16:00–21:00
tariff came out at **22:35–03:30**, because the plant window starts at 06:55. The
docs I had written promised "an 8 pm peak is drawn at 8 pm", so the implementation
was the half that was wrong. `87bbfab` solves the offset against each window's own
time of day instead, read from the plant frame's `DatetimeIndex`.

The test fixture grew a `DatetimeIndex` at 06:35 as part of this — deliberately
not midnight, because a midnight frame hides every off-by-an-origin bug, which is
precisely how the 1970 bug survived five campaigns.

---

## Breaking changes

Four signatures moved. All four fail loudly; none degrade silently. The upgrade
table is in the main [README](README.md).

| Was | Now |
|---|---|
| `preview_config()` returning a `"price"` key | `price_preview` in the control config |
| `plot_closed_loop(..., plot_dir)` | `plot_closed_loop(..., plot_dir, origins=[...])` |
| `objective.frac_power_zero(...) -> float` | `objective.power_diagnostics(...) -> PowerDiagnostics` |
| `render_control_reportcard(..., frac_power_zero=)` | `render_control_reportcard(..., power=)` |

`ResidualMetrics.frac_at_cap` is now `Optional[float]`, `None` at Δ = 0.

For the app repo, the v6 configs need `price_preview: 48` moved out of
`preview_config` and into the config, and `price_align: day` added wherever the
one-window price file was being used as a workaround — that file can go back to
being multi-day.

## Documentation updated

- [README.md](README.md) — upgrade/breaking-changes table, `price_align` trade-off
  table, the "If what you want is pre-cooling" recipe, hook contract row for
  `preview_config`, price-preview step of the walkthrough, test count.
- [template/README.md](template/README.md) — capability table row for price-driven
  shifting, and a "things that bite" entry on `cost_weight` without `price_preview`.
- [template/control_configs/control.yaml](template/control_configs/control.yaml) —
  `price_preview` and `price_align` documented with their trade-offs.
- [template/control_user_defined.py](template/control_user_defined.py) — notes that
  the tariff preview is not in `preview_config` and why.
- [improvement_plans/v6_followup_plan.md](improvement_plans/v6_followup_plan.md) —
  status header, implemented parts marked, both design changes recorded against
  the original text.

## Not done, and worth knowing

**Part 1, the counterfactual-tariff audit, is the one thing that would make a
`price_align: day` result trustworthy.** Under `day` the clock shortcut is
partly open by construction, so a convincing load-shifting trace is exactly what
both a real price-responsive policy and a clock-learner produce. The check that
separates them — hold the weather, the band and the surrogate, move the tariff's
peak, see whether the load follows — is designed in Part 1 of the plan and is
about 200 lines. Until it exists, treat pre-cooling under `day` as promising
rather than demonstrated.

Part 9 (multi-seed) is also untouched; v6's evidence for it is the strongest yet
(mlp zone NRMSE swinging 40% across three seeds, with the heating step-response
sign check passing at one seed and failing at the other two).
