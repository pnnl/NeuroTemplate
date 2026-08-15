# Report: the 2R2C grey-box family and the per-zone model split

Branch `resistor_capacitor`, eleven commits on top of `5425c95`. **465 tests
pass** (439 before; 36 added for the new code, 10 removed from the old by
consolidation).

Two things landed. `model: 2r2c` is a thermal network — zone air, a hidden
building mass, a learned heat input — that *asserts* the storage the black-box
families only ever hope for. The per-zone split is the machinery underneath it,
and it generalizes: `per_zone: true` gives `mlp` and `gru` the same treatment.

---

## First, a correction

> *I suspect you run a fit for the tau parameters, fix R and C to mimic that,
> then fit the MLP?*

No — and the difference is the whole point of the design, so it is worth being
precise about.

**There is no fit for the time constants. There is one training run, and only
the MLP is in it.** The three time constants come out of the config, are
converted from hours into timesteps, and are frozen (`requires_grad=False`)
before the first epoch. Adam never touches them. What you describe — identify
R and C in stage one, freeze them, fit the input map in stage two — is the
classical two-stage grey-box recipe, and it is a good one. What I built is
**stage two only, with stage one replaced by an assertion you make.**

That is not laziness, and it is not an oversight I would fix given more time.
It is the conclusion of an experiment that says stage one cannot be run against
this data at all — a training rollout does not identify these parameters, it
*corrupts* them. [Part 2](#part-2--the-philosophy) is that experiment. If you
do want stage one, [there is a way to get it](#the-two-stage-fit-you-described-and-how-to-get-it),
and it is not gradient descent.

---

## Part 1 — How it works

### The network

Zone air is one capacitance. The building's thermal mass — walls, slab,
furniture — is a second, hidden one. Heat moves zone ↔ ambient through one
resistance and zone ↔ mass through the other:

```
    ambient ──R_o──┬── zone air (C_z) ──R_m── mass (C_m)
                   │
                heat input
```

Discretized, that is two lines, and they are the entire state update:

```
    z_{k+1} = z_k + a_m (m_k - z_k) - a_o z_k + g(u_k, d_k)
    m_{k+1} = m_k + a_c (z_k - m_k)
```

`z` is the zone temperature (in `Y`, observed). `m` is the mass temperature —
**not in the data**, carried by the block. The three rates are dimensionless
per-step exchange coefficients:

| rate | is | config field |
|---|---|---|
| `a_o` | `dt / (R_o · C_z)` — zone ↔ ambient | `rc_tau_ambient_h` |
| `a_m` | `dt / (R_m · C_z)` — zone ↔ mass, from the zone's side | `rc_tau_zone_mass_h` |
| `a_c` | `dt / (R_m · C_m)` — mass ↔ zone, from the mass's side | `rc_tau_mass_zone_h` |

`a_m / a_c = C_m / C_z`, so the last two together are the capacitance ratio:
the defaults of 1 h against 4 h make the mass four times heavier than the air.

`g` is the only learned thing in the update: an MLP over the **current** `u`
and `d`, sized by the existing `num_layers` / `log2_hidden_size` knobs. We have
no plug-load channel, no internal-gains channel and no zone-level heat
metering, so `g` is where all of that goes.

### Why `-a_o z` and not `a_o (T_ambient - z)`

The conduction term reads as loss to a *zero* ambient, which looks wrong until
you notice that `g` is affine in `d` and `d` carries the outdoor temperature.
The real `+ a_o · T_ambient` is one of the terms `g` can form, and forming it
is a linear map the MLP represents exactly.

The payoff is that **nothing has to tell this block which `D` column is the
weather.** No hook, no config field, no substring match on `"outdoor"`. A block
that had to be told would be one more thing for an application to get wrong,
and the library has been trying to delete exactly that kind of coupling.

### The parameterization, and why stability is structural

Rates are not stored directly. Each is

```
    a = 1 / (MIN_TAU_STEPS + softplus(θ))          MIN_TAU_STEPS = 2.0
```

so every rate is positive and strictly below `1 / MIN_TAU_STEPS = 0.5`, and
the three of them sum below 1.5. That bound is the whole stability argument.
Write the update as `I - L` with

```
    L = [[a_m + a_o,  -a_m],
         [-a_c,        a_c]]
```

`L` has real positive eigenvalues (positive trace, positive determinant
`a_o·a_c`, and the discriminant is a sum of squares — see
[mode_taus](src/neurotemplate/modeling/rc.py#L144)). The update's eigenvalues
are `1 - λ`, and `λ ∈ (0, a_m + a_o + a_c) ⊂ (0, 2)` puts every one of them
inside the unit circle.

That is a *structural* guarantee, not an empirical one. It holds for any
parameter values, any horizon, any input — including the adversarial inputs a
DPC policy goes looking for — and it holds **only because `g` never reads the
state**. Let the gain see `z` and the state matrix stops being fixed and the
argument evaporates. That restriction is load-bearing and is commented as such
at the site.

`MIN_TAU_STEPS = 2.0` is the one arbitrary number here, and it costs nothing
real: a two-step time constant is a zone with essentially no thermal inertia.

### The two numbers you actually read

The three component time constants are *not* what the model does. The two
eigenvalues are, and they are neither of the three. At the defaults on
5-minute data:

```
    components:  tau_ambient = 2 h,  tau_zone_mass = 1 h,  tau_mass_zone = 4 h
    modes:       tau_fast    = 0.60 h,   tau_slow = 13.40 h
```

`tau_slow` is the storage — how long a temperature difference put into this
zone survives. It is the number pre-cooling lives on, and it is the number the
reportcard prints, per zone and per channel:

```
- **fy:** `PerZoneBlock`, params=2,504, in=42, out=2
  - **zone `north`:** `RCBlock`, params=1,252, in=30, out=1
    - North Zone Air Temperature [°C]: tau_fast=0.60 h, tau_slow=13.40 h
```

### Why those hours are real hours, on z-scored data

This is the part that could easily have been quietly wrong, so: the block runs
in normalized coordinates, and the time constants are still physical.

Normalization is per column, `z = (T_z - s) / g`. The mass is unobserved, so we
are free to *define* which frame it lives in, and it is defined in the zone's:
`m = (T_m - s) / g`, same shift, same stretch. Both states therefore undergo
one shared affine change of coordinates, under which the state matrix is
unchanged — the stretch cancels in every difference (`(T_m - T_z)/g = m - z`)
and the shift contributes a constant that the gain's bias absorbs.

Same matrix ⇒ same eigenvalues ⇒ **the time constants in steps are exactly the
physical ones**, and `steps / steps_per_hour` is exactly physical hours. That
property is why the reportcard can print a number you can check against the
building instead of a number in arbitrary units. It is a consequence of the
definition, not an accident, and it would break the moment the mass got its own
independent scale.

### The hidden mass: seed, warm, reset

`m` is state, and state has a lifecycle. It follows `GRUBlock`'s exactly, which
is why the machinery around it already existed.

**Cold** (no state, i.e. the first call of a rollout) — the block has only the
observed `in_steps` window of `z` to go on. It seeds `m` at the window mean and
then integrates the mass equation across the window, all but the final step:

```python
mass = torch.stack(steps, dim=0).mean(dim=0)
for zone in steps[:-1]:
    mass = mass + a_mass_zone * (zone - mass)
```

The last step is held back deliberately: it is the one the first prediction
consumes, so it drives the update rather than being folded in ahead of it.
The rule is right in the two limits that matter — a zone sitting at steady
state hands the mass that steady value, and a zone that has been driven hard
hands it something lagging behind.

**Warm** — every later call reads the stored `m_k`, uses it for `z_{k+1}`, and
*then* advances it. Both lines of the update read `(z_k, m_k)`, so the ordering
matters and is what the code does.

**Reset** — at a sequence boundary the state must be cleared or it leaks from
one batch into the next, dragging the autograd graph with it.
[make_reset_init](src/neurotemplate/modeling/dynamics.py#L624) wires this into
the `SystemPreview` on both the modeling and the control side. It detects
statefulness by the presence of `reset()`, which was an open audit item filed
as fragile duck-typing — two blocks now have one and both want exactly that
behaviour, so it turned out to be the contract rather than a proxy for it. The
item is closed in [audit.md](audit.md).

**Under `vmap`** — the sampled gradient penalty runs `jacrev` under `vmap`,
where there is no batch dimension and no state can be stored. The block detects
this from `x.ndim` and runs cold without storing, matching `GRUBlock`. This is
why the gradient-sign penalty still works on a 2R2C.

### The residual contract

`RCBlock.forward` returns the **increment** `z_{k+1} - z_k`, not `z_{k+1}`.
`SSMGrad` adds `z_k` back when `residual=True`, which is the existing
mechanism, so nothing new was needed — but it means `2r2c` with
`residual: false` would silently read an increment as an absolute temperature.
It raises instead. Same for a head that is not `residual_compatible`: the
`BinaryHead` on a 2R2C output would never restore the state, so
[get_dynamics](src/neurotemplate/modeling/dynamics.py#L657) names the offending
channels and refuses.

### Zones

A *zone* is a group of `Y` channels that gets its own sub-model: its own
parameters, its own hidden state, and no term reading any other zone's state.
The split comes from an optional `zone_mapping` hook:

```python
def zone_mapping(nm_mapping, cfg):
    return {
        "north": {"Y": ["North Zone Air Temperature [°C]"],
                  "U": ["North Cooling Setpoint [°C]"]},
        "south": {"Y": ["South Zone Air Temperature [°C]"],
                  "U": ["South Cooling Setpoint [°C]"]},
    }
```

[build_zones](src/neurotemplate/modeling/zones.py#L81) validates it into index
slices and is the only place the rules live:

- **Y partitions exactly.** Every state in one zone, none left out, none
  claimed twice. Anything else is named before the first epoch.
- **U need not partition.** Any control no zone claims is shared with *all* of
  them. That is one line of code and it is what makes a single air handler
  serving several zones expressible without a second concept.
- **D is shared by definition.** Weather is not per zone.

[PerZoneBlock](src/neurotemplate/modeling/dynamics.py#L189) does two things and
nothing else: gather each zone's columns out of the one shared flattened
window, and put the per-zone outputs back in `nm_mapping["Y"]` order (zones may
be declared in any order and hold their states in any order; the loss and the
feedback loop both assume the mapping's). Both are fixed permutations of
positions computed once at construction, so both are `index_select` against a
buffer — which is what keeps the wrapper `vmap`-safe and the gradient penalty
alive.

It takes a `make_block(zone, nd)` factory, so the same wrapper serves every
family. `get_2r2c` passes `RCBlock`; `get_mlp` and `get_gru` pass theirs when
`per_zone: true`. `2r2c` is per-zone unconditionally — with no hook there is
one default zone holding everything, which still gives *each Y channel* its own
RC network, just driven by every U channel. Declaring zones is how you say
which setpoint drives which zone.

Inter-zone conduction is not modeled, per the brief.

### The two new things the model builder needs

Both follow paths that already existed, rather than inventing plumbing:

- **`cfg.steps_per_hour`** is resolved from the data's sampling frequency at
  the CLI edge — `cfg.steps_per_hour = 3600.0 / freq_s`, one line beside
  `cfg.device = resolve_device(...)` and `cfg.seed = set_seeds(...)`, which are
  resolved runtime facts on the config for exactly the same reason. It rides
  into MLflow with the rest of the config, so `nt model load` gets it back free.
- **`zone_spec`** is threaded from the hook down through
  `train_and_evaluate → train_with_curriculum → build_trainer → get_dynamics`,
  which is precisely the path `head_stack` already takes from
  `data_head_config`. It is logged as `zone_spec.json` so a reload rebuilds the
  same topology even if the hooks file has moved on.

---

## Part 2 — The philosophy

### The problem this family exists for

`v7_precool_probe.py` in the RTU-mall repo made the case already: a surrogate
can post a respectable RMSE and still be wrong about **how long a temperature
difference survives**, because a model scored over a four-hour rollout can
re-derive the zone from the last few setpoints and never learn to integrate.
Coolth stored at midday is then gone by the evening peak and pre-cooling cannot
pay inside the model whatever the real building does.

That is not a bug in any of `mlp`/`nssm`/`lssm`/`gru`. It is what a windowed
regressor trained on a short horizon is *for*. No amount of window is a
substitute for a slow state, because the slow state is not in the window.

So the family's job is not to be more accurate. It is to **be right about one
specific thing, on purpose, and say so in a number you can check.**

### Why the time constants are asserted rather than fitted

I did not start here. The first version had all three rates trainable, and on
the RTU-mall data they moved almost not at all (13.40 h → 13.32 h) — which
raised the obvious worry that they were decorative. So I built a controlled
identifiability experiment: generate data from a **known** 2R2C building, train
an `RCBlock` on it, and see whether the learned `tau_slow` moves toward the
truth.

Truth 31.62 h. Initialization 13.40 h. Learning rate on the taus boosted 20× so
they could actually move:

| mass seed at each window start | horizon 48 (4 h) | horizon 144 (12 h) |
|---|---|---|
| the truth's own mass | 16.45 h | 14.41 h |
| window mean (what a real rollout must do) | **8.67 h** | **7.70 h** |

Given the true mass, training moves `tau_slow` **toward** the truth. Given the
seed a real rollout is forced to use, the same training drives it **away** —
past the initialization, in the wrong direction, by roughly as much.

The mechanism is not subtle once you see it. The mass is unobserved, so every
rollout has to guess it from an `in_steps` window an hour long at most, against
a state that lags by many hours. That guess is wrong. The optimizer then has
two ways to reduce the resulting error: fix the forcing, or **shorten the slow
mode so a wrong mass matters less**. The second is cheaper, so it takes it. A
shorter `tau_slow` is not a discovery about the building; it is a hedge against
the block's own ignorance of its state.

I tried to fix the seed rather than surrender it. Replacing the fixed rule with
a *learned* one — softmax weights over the window plus an offset, initialized
to exactly the old rule, so it could only improve — changed nothing:

| seed rule | `in_steps` 6 | `in_steps` 12 |
|---|---|---|
| window mean | 8.67 h | 12.36 h |
| learned filter | 9.46 h | 12.48 h |

Which is the right answer, and in hindsight the obvious one: **an hour of zone
temperature carries no information about a ten-hour state.** There is nothing
for a cleverer seed to extract. (The `in_steps` 6 → 12 column is the real
lever, and it is capped at `MAX_IN_STEPS = 12`.)

So the conclusion is not "the optimizer needs help." It is that this parameter
is **not identifiable from this data by this procedure**, and a parameter that
is not identifiable should not be presented as fitted. Making it trainable
offers a knob that reliably produces a wrong answer and then prints that answer
on the reportcard as though it were measured.

Hence: three config fields, frozen by default, and a reportcard that reads back
what you asserted. `rc_train_taus: true` is still there for the case where the
rollout *is* long relative to the slow mode and the data excites it — but it is
off, and the config comment says why.

This is the same instinct as CLAUDE.md's rule 5, one level up. "Fail fast rather
than fall back" applied to a *number*: if the procedure cannot produce a
trustworthy value, do not produce a value and dress it up — make the user state
their assumption, and then show them what it implies.

### The two-stage fit you described, and how to get it

Your instinct is right that a grey box wants its R and C from somewhere. The
report's claim is only that gradient descent through a training rollout is the
wrong somewhere. Three good ones, in order of preference:

1. **Measure it.** A free-decay episode gives `tau_slow` directly — the probe
   did exactly this on `data/excited` and got 6.4–10.9 h depending on
   specification. That is stage one, run properly, with a method suited to the
   quantity. Then set `rc_tau_ambient_h` and let the MLP fit around it. This is
   your two-stage recipe; the stages just do not share an optimizer.
2. **Search it.** `nt model optuna` treats `rc_tau_ambient_h` as an ordinary
   hyperparameter — `cfg.rc_tau_ambient_h = trial.suggest_float(...)` in
   `hyper_args`, verified working. Held-out dev loss is a *different* criterion
   from identification (dev loss is not the storage either), but it picks the
   prior the data prefers rather than the one we shipped, and it does not have
   the wrong-seed pathology because each trial's prior is fixed throughout.
3. **Sweep it and read the reportcard.** Three runs at three priors takes
   minutes and the storage number is printed each time.

What I would *not* add is a stage-one gradient fit that looks like
identification and is not.

### Why one mechanism instead of two

The first design had `RCBlock` handle all zones at once, vectorized with masked
input couplings, and a separate `PerZoneBlock` wrapper for MLPs. Two mechanisms
for one idea. RC is linear so the vectorized form was tempting and would have
been marginally faster.

It went in the bin. One wrapper that holds *N* sub-blocks explains itself in a
sentence, makes "2R2C per zone" and "MLP per zone" the same feature rather than
two features that resemble each other, and costs nothing measurable at the
handful of zones a real building has. `make_block` is the only seam, and adding
a per-zone family is now one line in a factory.

### Refuse rather than degrade

Four things fail loudly instead of producing a plausible-looking wrong model:

| condition | why silence would be worse |
|---|---|
| `2r2c` with `residual: false` | an increment read as an absolute temperature — wrong by however far the zone is from zero, and only that |
| `2r2c` with no `steps_per_hour` | time constants placed against an unknown timestep |
| `2r2c` under a non-residual head | same as the first, per channel, and invisible on any report |
| `per_zone: true` on a family that ignores it | a flag that silently does nothing is worse than one that is rejected |

And `build_zones` names every malformed split — a state claimed twice, a state
left out, a column that is not in the mapping — before a single epoch runs.

---

## Part 3 — Evidence

### Storage: the thing it was built for

`data/excited`, where the envelope decays at 6.4–10.9 h (a 5.80 h half-life at
the mid-range estimate). Two rollouts, identical setpoints and weather,
differing only in initial zone temperature, compressor held off — so what
decays is the envelope's memory and not a thermostat closing an error:

| | building | **2r2c** | nssm | mlp | gru | lssm |
|---|---|---|---|---|---|---|
| half-life | 5.80 h | **6.08 h** | 3.92 h | 8.00 h | 14.50 h | 1.17 h |
| left after 6 h | 48.8% | **50.2%** | 21.1% | 67.1% | 93.9% | 0.1% |
| left after 12 h | 23.8% | **32.0%** | 4.0% | 7.4% | 66.3% | ~0% |
| left after 24 h | 5.7% | **13.0%** | 1.8% | 0.6% | 36.9% | 0% |

`2r2c` is the closest of any family at every horizon. `nssm` forgets roughly
twice too fast — the failure the probe identified. `gru` over-remembers by
nearly 3×. `mlp`'s half-life looks fine and its shape is not exponential at
all: 67% left at 6 h and 7% at 12 h is not a decaying building.

### Accuracy: what it costs

Same split, same channels, same in_steps — only the family differs:

| family | test RMSE | params |
|---|---|---|
| gru | 0.411 | 151,041 |
| nssm | 0.424 | 57,987 |
| mlp | 0.436 | 24,449 |
| **2r2c** | **0.536** | **17,284** |
| lssm | 0.712 | 63 |

About 25% worse than the black boxes at a third of `nssm`'s parameters. That is
the trade, stated plainly: rollout RMSE cannot see storage, and load shifting
cannot live without it. `2r2c` is a family to *reach for* when the storage is
the point, not a replacement for the four that exist.

### The prior is load-bearing

Which is the honest flip side of asserting it — a badly chosen prior costs
accuracy, visibly:

| prior | resulting `tau_slow` | test RMSE |
|---|---|---|
| `rc_tau_ambient_h: 0.85` | 7.8 h | 0.579 |
| `rc_tau_ambient_h: 2.0` (default) | 13.4 h | **0.536** |
| `rc_tau_ambient_h: 4.2` | 24.3 h | 0.649 |
| default, `rc_train_taus: true` | 13.3 h | 0.536 |

Two things to read here. The prior matters and the data has an opinion about
it — which is what makes option 2 above (search it) work. And the trainable row
is a near-no-op at the default learning rate, which is why the earlier "they
barely move" observation was not evidence of safety: it took the 20× boost to
reveal that when they *do* move, they move the wrong way.

### End to end

Every CLI path was run, not just imported:

- `nt model train` with `2r2c`, single and two-zone, plus `per_zone` `mlp`/`gru`
- `nt model load` — per-zone topology rebuilt from `zone_spec.json`, weights
  loaded without a remap, OOD RMSE matching the training run's test RMSE
- `nt model optuna` — 3 trials searching `rc_tau_ambient_h`, then the phase-2
  retrain
- `nt control train` — a full DPC policy against a frozen 2R2C surrogate: loss
  31.9 → 12.6 over 15 epochs, policy clears its reference, closed-loop
  evaluation completes. This is the path that would have broken if the mass
  state leaked across batches, and it does not.

On a two-zone synthetic building where the zones are genuinely independent,
`mlp` with `per_zone: true` scored 0.124 against 0.158 jointly — the split earns
its keep on its own, separately from 2R2C.

---

## Part 4 — What it gives up

**No thermostat deadband.** `g` cannot see the state, so it cannot represent
"the compressor is off whenever the setpoint is above the zone" — that is a
function of the *difference*. The family renders staged cooling as a smooth
monotone response to setpoint with no cut-off. The sign is right, so DPC
gradients point the right way, but a policy will read cooling as available in
proportion where the real plant stages it.

This is the direct price of the stability guarantee and I would not trade it
back cheaply: letting the gain see `z` buys the deadband and costs
unconditional stability, which is the property that makes the surrogate safe to
put a DPC policy against.

**A prior that can be wrong.** See the table above. The mitigation is that it
is wrong *visibly* — on the reportcard, in hours, next to a building you can
compare it to.

**No inter-zone coupling.** By instruction, and the code says so rather than
implying otherwise.

Both limits are recorded in [audit.md](audit.md) under "Known limits,
documented rather than open", so they are not rediscovered as bugs.

---

## Part 5 — Where things live

| file | what |
|---|---|
| [modeling/rc.py](src/neurotemplate/modeling/rc.py) | `RCBlock`, `mode_taus`, the constants. The physics, and the long-form version of Part 2 |
| [modeling/zones.py](src/neurotemplate/modeling/zones.py) | `Zone`, `build_zones` — the partition and all its rules |
| [modeling/windows.py](src/neurotemplate/modeling/windows.py) | positions inside the flattened key-major history window; `GRUBlock` now shares it |
| [modeling/dynamics.py](src/neurotemplate/modeling/dynamics.py) | `PerZoneBlock`, `get_2r2c`, `PER_ZONE_FAMILIES`, the catalog |
| [modeling/config.py](src/neurotemplate/modeling/config.py#L77) | `rc_tau_*_h`, `rc_train_taus`, `per_zone`, `steps_per_hour` |
| [template/](template/) | `zone_mapping` stubbed, every new config field annotated |

Three new modules rather than one, because they are three different things: the
physics, the partition, and window arithmetic. `windows.py` also removed
duplicated index math from `GRUBlock`, which is the DRY win that made a third
module worth having rather than a cost of it.

### Using it

```yaml
model: 2r2c
in_steps: 12
residual: true            # required; 2r2c predicts an increment

rc_tau_ambient_h: 2.0     # zone <-> ambient
rc_tau_zone_mass_h: 1.0   # zone <-> mass, from the zone's side
rc_tau_mass_zone_h: 4.0   # mass <-> zone, from the mass's side
# rc_train_taus: false    # read Part 2 before turning this on
```

Then read `tau_slow` off the reportcard and compare it to the building. If it
is wrong, the fix is a config field, not a training run.

---

## Appendix: the test suite

36 cases added — 35 in [tests/test_rc_dynamics.py](tests/test_rc_dynamics.py)
and [tests/test_zones.py](tests/test_zones.py), one pinning the per-zone
reportcard rendering. They are written as the list of
claims the family makes, because that is what a family which *asserts* things
has to be tested on: the update is the RC equation arithmetic and all;
`mode_taus` matches the eigenvalues of the assembled matrix; no rate can leave
the stable range at any `θ`; a week of adversarial input does not diverge; a
head start outlives the history window; the prior stays frozen; one zone's
input cannot move another's state; and each of the four refusals above raises.

On the second half of the brief — reducing the test count — the honest finding
is that **there was not much to cut.** The suite is not padded; it is ~350
distinct behavioural assertions over 11.6k lines of library. What *was* bloat
was validation boilerplate: two dozen near-identical "set a field, expect a
`ValueError`" functions per file, which are `parametrize` tables now.

| file | before | after |
|---|---|---|
| `test_control_config.py` | 21 functions, 224 lines | 5 functions, 120 lines |
| `test_config.py` | 16 functions, 146 lines | 9 functions, 101 lines |
| `test_heads.py` | 26 functions, 372 lines | 12 functions, 222 lines |
| `test_control_metrics.py` | 22 functions, 301 lines | 14 functions, 264 lines |

`test_heads.py` also lost seven hand-rolled `try/except/raise AssertionError`
blocks in favour of `pytest.raises` asserting the actual message — so a head
that starts raising the *wrong* error is now caught, which it was not before.
No case was dropped except exact duplicates (`peak_q` tested three values of
one boundary; the all-or-nothing tariff check tested three fields of one loop).

Net: 439 → 465 collected, of which 36 are new, so the pre-existing suite
shrank by 10 cases and 49 functions (395 → 346). If a bigger number matters, the
remaining lever is merging genuinely distinct behavioural cases, and I would
advise against it — the count is high because the coverage is real.
