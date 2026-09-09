# Implementation report: what shipped, and every interface it added

Implements [generalization_report.md](../generalization_report.md) Tiers 0–2.
Tier 3 is untouched by design — those items are gated on a real second dataset.

**465 → 581 tests, all passing.** 116 added, none removed. Verified end to end
with real CLI runs against a synthetic plant, not just unit tests (§7).

Two plan items changed on contact with the code and are argued in §6: the `R`
role was withdrawn, and per-block hyperparameter overrides were deliberately
not built.

---

## 1. Blocks — mixed model families in one graph

The headline capability: **zone air as a grey-box `2r2c`, power as an `mlp`,
in one model.** `cfg.model` used to be one string for every predicted channel.

### The hook

`block_mapping(nm_mapping, cfg) -> {block: spec} | None`, in
`modeling_user_defined.py`. `zone_mapping` still works as a deprecated alias.

```python
def block_mapping(nm_mapping, cfg):
    return {
        "zone":  {"Y": ["Zone Air Temperature [°C]"],
                  "U": ["Cooling Setpoint [°C]"],
                  "model": "2r2c"},
        "power": {"Y": ["Total Power [W]"],
                  "reads": ["Zone Air Temperature [°C]"],
                  "D": ["Ambient Temperature [°C]"],
                  "model": "mlp"},
    }
```

| key | meaning | rule |
|---|---|---|
| `Y` | channels this block predicts | **partitions exactly** — claimed twice, left out, or unknown is named before epoch 1 |
| `U` | its actuation | need not partition; **unclaimed U is shared with every block** |
| `D` | its disturbance subset | absent = all of D; **no partition rule**, D is exogenous |
| `reads` | other blocks' Y it may see | must exist in Y, must not be self-owned |
| `model` | family for this block | absent = `cfg.model` |

### New API

```python
# modeling/blocks.py   (was zones.py; zones.py is now a deprecation shim)
@dataclass(frozen=True)
class Block:
    name: str
    y_idx: tuple[int, ...]          # predicts
    u_idx: tuple[int, ...]
    y_names: tuple[str, ...]
    d_idx: tuple[int, ...] = ()
    reads: tuple[int, ...] = ()
    model: Optional[str] = None
    # properties: ny, nu, nd, ny_in
    #   y_in_idx = y_idx + sorted(reads)   ← reads, with own channels first

def build_blocks(nm_mapping, block_spec=None) -> list[Block]

# modeling/dynamics.py
class BlockStack(nn.Module)                     # was PerZoneBlock
BLOCK_BUILDERS: dict[str, (cfg, Block) -> nn.Module]   # mlp, gru, 2r2c
PER_BLOCK_FAMILIES: tuple[str, ...]             # derived from BLOCK_BUILDERS
VALID_MODEL_NAMES: tuple[str, ...]              # model_catalog | BLOCK_BUILDERS
def family_by_channel(cfg, nm_mapping, block_spec=None) -> dict[str, str]

# modeling/pipeline.py
def resolve_block_spec(hooks, nm_mapping, cfg)  # block_mapping → zone_mapping → None
```

### Why `reads` is safe, and where it isn't

Every block input comes from the **shared history window**, never from another
block's same-step output. So there is no evaluation-order dependency between
blocks, the gather stays a fixed permutation, and the whole stack remains
`vmap`-safe — which is what keeps the sampled gradient penalty alive.

**`2r2c` refuses `reads`.** Its unconditional-stability guarantee holds only
because the learned forcing map `g` reads `u` and `d` and never state. Note the
precision: the block still reads its *own* Y in the fixed linear term; what it
cannot do is let a *learned* map see state, and a `reads` entry would have to
enter through `g`. The refusal:

> `2r2c block 'power' cannot declare 'reads': its unconditional-stability
> guarantee holds only because the learned forcing map g reads u and d and
> never state. A 'reads' entry would let g see another block's Y — exactly the
> state-dependence the guarantee forbids. Use a different family for this block
> if it needs to see another block's state.`

### Reportcard

Per block, with its family — the interesting fact once families mix:

```
- **fy:** `BlockStack`, params=3,493, in=60, out=2
  - **block `zone` (2r2c):** `RCBlock`, params=1,220, in=48, out=1
    - Zone Air Temperature [°C]: tau_fast=0.60 h, tau_slow=13.40 h
  - **block `power` (mlp):** `MLP`, params=2,273, in=36, out=1
```

The input widths are the proof both features work: `zone` reads `12 × (1 Y +
1 U + 2 D) = 48`; `power` reads `12 × (2 Y_in + 1 U + 1 D) = 36` — its own
channel **plus one read**, and a **D subset of one**.

### Backward compatibility

`Zone`, `build_zones`, `PerZoneBlock`, `PER_ZONE_FAMILIES`, the `zone_mapping`
hook, and a logged `zone_spec.json` all still work. Saved runs keep loading:
`cli/load.py` prefers `block_spec.json` and falls back to `zone_spec.json`.

---

## 2. Diagnostics — `nt data`

A new CLI group. `nt data *` reads the **dataset**; `nt model debug-heads`
(pre-existing) reads a **trained run**. Prior question vs. posterior question.

```
nt data heads [--column NAME] [--k-max K]
nt data decay --state COL --ref COL [--min-hours H]
```

### `nt data heads`

Recommends an output head per Y column from the **shape** of its distribution,
not its name — the template previously suggested `if "power" in col.lower()`.

The test is *not* distinct-value counting. A staged compressor at 0/4.2/8.1 kW
carries noise on every sample, so it has thousands of distinct floats and three
modes: continuous by cardinality, discrete by shape. The statistic is 1-D
k-means over K = 2..k_max, reporting **variance explained by the cluster means**
and **gap occupancy** (share of samples not near any mode).

```python
# modeling/diagnostics.py — pure numpy in, dataclasses out
@dataclass(frozen=True)
class LevelFit:  n_levels, centers, variance_explained, gap_occupancy
@dataclass(frozen=True)
class HeadRecommendation:  column, kind, evidence, fits, snippet

def fit_levels(x, k_max=6) -> tuple[LevelFit, ...]
def recommend_head(x, column, family=None, k_max=6) -> HeadRecommendation
```

Constants are scale-free fractions, following `scenarios.py`:
`GAP_TOL_FRAC=0.25`, `HIGH_VARIANCE_EXPLAINED=0.90`, `LOW_GAP_OCCUPANCY=0.05`,
`BOUNDARY_MASS_FRAC=0.05`, `K_MAX_DEFAULT=6`.

Real output, on a plant whose power channel is genuinely staged:

```
Total Power [W]  —  3 modes at 5.11082, 4201.13, 8100.47
  100.0% of samples close to a mode; gaps hold 0.0%
  K=2: 77.1% | K=3: 99.9% | K=4: 99.9% | K=5: 99.9% | K=6: 100.0%  (elbow at 3)
  Scope: this only reads 'Total Power [W]'s marginal distribution. ...

  OrdinalHead(n_levels=3, level_values=torch.tensor(
    [project_physical_to_norm(v, norm_stats["Total Power [W]"])
     for v in (5.11082, 4201.13, 8100.47)], dtype=torch.float32))
```

Two honesty features are unconditional, not optional:

- **The scope note** ships with every recommendation. It reads the marginal
  distribution and nothing else; a channel driven into three continuous regimes
  by an exogenous variable looks identical to staged equipment here.
- **The `2r2c` conflict**, resolved **per channel** via `family_by_channel`.
  `OrdinalHead.residual_compatible` is `False` and a 2r2c block needs
  residual-compatible heads, so `get_dynamics` would raise. A recommendation the
  user cannot take is worse than none — but so is a warning that does not apply,
  which is why this reads the block spec and not `cfg.model` (§7.2).

### `nt data decay`

The missing "stage one" of the 2R2C grey-box fit, argued in
[rc_identification.md](rc_identification.md). `2r2c` asks you to assert
`tau_slow` and, until now, gave you nothing to check it against — the
free-decay physics check cannot do it, since at the default 4-hour horizon it
only fails a slow mode above ~198 h.

```python
@dataclass(frozen=True)
class DecayWindow:  start, n_steps, tau_h, half_life_h, r_squared

def actuator_off_mask(values, off, tol_frac=0.02) -> np.ndarray
def find_runs(mask, min_len) -> list[tuple[int, int]]
def estimate_decay(t_hours, state, ref, off, min_run_steps) -> tuple[DecayWindow, ...]
def summarize_taus(windows) -> tuple[float, float, float]   # median, min, max
```

Finds runs where every U column sits within 2% of its `SynthSpec.off` value,
regresses `log|state − ref|` on time, reports τ and half-life per window plus
the spread — because real data gives a *range*, and that range is the honest
uncertainty on the assertion.

When no window qualifies it says so and names the next move, rather than
inventing a number:

```
  actuators off (every U column within 2% of its off value) for >= 4h: 0 qualifying window(s)
  No free-decay episode found in this data -- the plant is never uncoupled for
  long enough at these off values. The rc_tau_ambient_h prior then has to come
  from an Optuna sweep instead: it is an ordinary hyperparameter there, not
  something this script can measure for you.
```

Deliberately **not** wired into `nt model train`: deciding which windows count
as free decay takes judgement, and a pipeline that guessed would print a number
that looks measured and is not — the failure `rc_train_taus: false` already
exists to avoid, moved one stage earlier.

---

## 3. Seed-ensemble spread — error bars on everything

Every number the library printed was a single draw. `optuna.py` trained
`trial_seeds` models per trial and threw them away.

```python
# modeling/config.py
ensemble_seeds: int = 1        # 1 == today's behaviour, byte-identical

# modeling/reportcard.py
@dataclass(frozen=True)
class SeedSpread:      min, median, max
@dataclass(frozen=True)
class EnsembleSpread:  n_seeds, rmse, physics, tau_slow
def spread_stats(values: list[float]) -> SeedSpread
def render_reportcard(..., ensemble: Optional[EnsembleSpread] = None)
```

It lives in `pipeline.train_and_evaluate`, so **both** `nt model train` and
Optuna's phase-2 retrain get it. The primary seed's weights are what gets
logged — the ensemble adds error bars, it does not change the artifact.
Min/median/max rather than mean ± std, because a handful of seeds does not
justify a Gaussian. Seeds stay a property of the *campaign*: no block, family,
or hook ever sees one.

**It earned its keep immediately.** On the smoke-test plant:

| Scenario | Channel | Min | Median | Max |
|---|---|---:|---:|---:|
| Step response: Cooling Setpoint to 24 | Zone Air Temperature | **−0.4857** | −0.4283 | **+0.7841** |
| Free decay | Zone Air Temperature | −0.02209 | 0.09628 | 0.178 |

The step response **changes sign across seeds**. The single-seed run reported a
confident `[ok]` for that check. That is exactly the "nobody can currently
tell" problem — and it is now visible on the page. `tau_slow` came back 13.4 h
on all three seeds, confirming it is frozen by design rather than fitted.

`objective()` also prints the per-trial dev-loss spread, which was free.

---

## 4. Roles and observation

### `regulated` (was `controlled_channels`)

The OCP **regulated output**: the Y channels the objective bands and the
reportcard scores. Renamed to stop it being confused with U, which is what is
actually *controlled*.

```python
# control/policy.py
def regulated_channels(hooks, nm_mapping) -> tuple[list[str], list[int]]
```

One documented fallback to `controlled_channels`, defined **once** here rather
than at each of its seven call sites — which was also the DRY fix for repeating
`hooks.controlled_channels(nm_mapping)` everywhere.

### Per-column disturbance preview — a real defect fixed

`preview_config(nm_mapping) -> {signal: n_steps}` accepted **per-column**
preview depths and the implementation collapsed them:

```python
d_future = max([spec.get(col, 0) for col in nm_mapping["D"]], default=0)
```

Declaring 2-step preview on ambient silently gave the policy 2-step preview on
*every* disturbance. Now each column is previewed only as far as declared:
`SystemPreview` still hands over the max window (it cannot window columns
differently), and a fixed `index_select` permutation — `_PreviewGather`, same
discipline as `BlockStack` — narrows it to the declared (column, step) pairs.
`insize`'s D term became `h_ctrl*nd + sum(d_depths)`, which reduces exactly to
the old `nd*(h_ctrl + d_future)` when depths agree. `describe_policy` now
reports `d_preview_by_column`.

### The observation assumption, stated

Every DPC policy in this library reads every modeled output, and nothing said
so. The control reportcard now does:

> **Observation:** Policy observes all N Y channels — no measurement restriction.

Not a feature — the statement of an assumption the library already made
silently. `Y_obs`/`U_obs`/`D_obs` stay Tier 3, gated on a system that actually
withholds a sensor, and must ship with the full-vs-restricted objective
comparison rather than the knob alone.

---

## 5. Units, de-HVAC-ing, and cleanups

### Units are per-column metadata

They live where per-column metadata already lives — `norm_stats` — so they are
persisted and reloaded with the model for free. No new hook, no new config
field, no new persistence path.

```python
# common/data_utils.py
def parse_column_unit(col) -> Optional[str]   # trailing "[...]" only
def unit_of(norm_stats, col) -> Optional[str] # None for old runs, never raises
def axis_label(norm_stats, cols, fallback) -> str
```

The value is parsed from the column's own name (`Total DX Power [W]` → `W`),
extractively — whatever is between the brackets is returned verbatim, since
what counts as a unit is the dataset owner's business. `axis_label` renders
`"<fallback> [unit]"` only when every column on that axis agrees, which matters
now that one Y can mix °C and W.

The escape hatch is the **existing** `tie_norm_stats` hook, which already
post-processes `norm_stats` — no second mechanism was added. `cfg.synth_y_unit`
survives as the one documented fallback for unit-less column names.

`plot_closed_loop`'s three hardcoded labels (`"Temp [°C]"`, `"Setpoint [°C]"`,
`"Power"`) now derive from the columns each axis actually plots, with
role-descriptive fallbacks (`"Regulated"`, `"Control input"`, `"Power"`).

### Physical constants lost their °C defaults

`band_lo`/`band_hi` have **no default** — required when the comfort band is
used, failing with:

> `band_lo and band_hi must both be set — the comfort band's edges are in the
> plant's physical units (°C for HVAC, volts for a battery, ...), so there is no
> library-wide default for them.`

`setpoint_step` defaults to `0.0` (no quantization) rather than a half-degree
grid. `residual_delta`'s comment was de-HVAC'd.

### `bracket_band` left the library

An HVAC thermostat algorithm that the library never called, reachable only if
an application's `reference_control` hook imported it. It now lives in
`template/control_user_defined.py` as example code, with its 15 tests, and no
re-export shim — the point is that it is not library API.

### `MAX_IN_STEPS` and the factory table

`in_steps` above 12 now **warns** with the actual cost (memory; context past
the training horizon) instead of raising — on **both** sides. The modeling side
was changed first, which left `control/config.py` still raising on the same
value; that inconsistency was caught and fixed, and the now-unused
`MAX_IN_STEPS` alias and the dead `check_max` validator were removed.

`model_catalog` was doing double duty as the factory table *and* the name
whitelist, which had left `get_2r2c` as an unreachable stub raising
`RuntimeError` purely to keep a key present. Split: `model_catalog` holds only
monolithic factories, `VALID_MODEL_NAMES` is the derived union, and the stub is
gone.

---

## 6. Two plan items that changed on contact

### `R` as a fourth role — withdrawn

The plan had `ymin`/`ymax`/`price` becoming `nm_mapping["R"]`. **They are not
dataframe columns.** `scenarios.py` builds the control window dict:

```python
D = _sliding(df, nm_mapping["D"], W, stride, device)            # from the dataframe
ymin, ymax = load_control_hooks(cfg).comfort_band_signal(...)   # GENERATED
price = build_price(cfg, n, W, price_path, generator, ...)      # from a tariff file
```

`df["ymin"]` does not exist. Every `nm_mapping` consumer assumes columns
(`df[nm_mapping[key]]`, per-column `norm_stats`, `warn_degenerate_columns`), so
an `R` key would make the mapping mean two different things depending on which
key you read.

Worse, it would **weaken** the invariant it was meant to state. `build_price`'s
own docstring already makes the argument: a D column *"would be baked into the
frozen surrogate's input width, and would tell the dynamics model that
electricity price drives zone temperature."* "Enters the cost, never the
dynamics" is guaranteed *because* these signals are not columns.

What survives: three signals hand-threaded through four modules with two ad-hoc
preview paths. A registry would collapse that, but it would have one consumer
and three permanently fixed members. **The trigger is a fourth reference
signal** — a DR event schedule, a carbon-intensity series — which would also
settle the width question the current three disagree on (the band is indexed by
the regulated subset; price is global). Recorded in
[generalization_report.md](../generalization_report.md) §3.2.

### Per-block hyperparameter overrides — not built

The plan mentioned `model` as "a family name plus per-block overrides". Only
the family name shipped. There is no proven need for per-block widths or depths,
and CLAUDE.md rule 4 forbids the extension point until there is. Documented as
such in `blocks.py` rather than left as a silent omission.

---

## 7. Verification

`uv run pytest -q` → **581 passed** (from 465). Unit tests alone would not have
caught either defect in §7.2, so the new capabilities were also run for real: a
synthetic 5-minute plant with a staged power channel, a mixed-family block spec,
and every new CLI path.

### 7.1 What the end-to-end runs proved

- `nt model train` builds a mixed `2r2c` + `mlp` graph; the reportcard shows
  per-block families, and the input widths confirm `reads` and the per-block D
  subset are actually applied.
- `tau_slow` reads back at the configured prior (13.40 h) through the new path.
- `nt data heads` recovers the true stages **0 / 4200 / 8100 W** as
  `5.1 / 4201 / 8100` from data with thousands of distinct floats — the exact
  case the feature exists for — while the continuous zone channel stays
  regression.
- `nt data decay` recovers a known τ = 9.0 h exactly (r² = 1.0), and reports the
  honest "no free-decay episode" outcome on a plant that never idles.
- `ensemble_seeds=3` produced the sign-flipping step response in §3.

### 7.2 Two defects the smoke test found

Both were fixed, and both were invisible to the test suite:

1. **The `2r2c` head-conflict warning was over-broad.** It read `cfg.model`, so
   with `cfg.model: 2r2c` and a mixed spec it warned that an ordinal head on the
   *mlp-built* power channel would be refused. It would not. The library was
   right; the diagnostic was steering users off the correct head. Fixed by
   resolving the family per channel through `family_by_channel`, which reuses
   `build_blocks` + the trainer's own family resolution so the two cannot
   disagree. Pinned by `test_family_by_channel_is_per_block_not_per_run`.
2. **Stale `zone` vocabulary** in the shipped config template and the README —
   including the hook table, the file tree, and the whole "Zones" section, which
   is what a new user reads first. Rewritten as "Blocks", with the deprecated
   aliases still documented *as* aliases.

### 7.3 Known-good and known-limited

- The golden reportcard `reportcard_nssm_baseline.md` is **unchanged** — nssm is
  not per-block, so the rendering rename never reaches it.
  `control_reportcard_baseline.md` gained exactly one line (the observation
  statement), diffed by hand.
- `nt data decay` reads only the **train** split, matching how `train.py` pools
  data. A free-decay episode living only in dev/test needs `train_data_path`
  pointed at it.
- The reportcard's `Factory:` line still prints `cfg.model`, which is only the
  *default* family under a mixed spec. Cosmetic; the per-block lines below it
  are authoritative.
- `regulated_channels` lives in `control/policy.py` and is imported by
  `closed_loop.py` and `pipeline.py`. It is a hook accessor, so `hooks.py` is
  arguably its home — a tidy-up, not a defect.
