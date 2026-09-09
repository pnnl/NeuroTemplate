# Generalization report: blocks, roles, and what to build

Written against the tree at `f4cbd52`. Companion to [audit.md](audit.md), which
tracks defects; this one is about direction. See also
[rc_identification.md](improvement_plans/rc_identification.md) for how the 2R2C
family's fixed parameters are meant to be arrived at.

## Context

Two goals: the library usable on more systems than an RTU mall, and the sci-ml
choices easier to make. Two changes get most of the way there, and they are
independent of each other.

**Blocks.** Today one `cfg.model` string picks one architecture for every
predicted signal. It should be a per-block choice: *this* group of signals is
2R2C, *that* one is an MLP, in one graph. The machinery for this is largely
built — `PerZoneBlock` is already a heterogeneous-block executor — and the
missing piece is a richer spec, not new plumbing.

**Roles.** `Y`/`U`/`D` stay as they are and grow per-role observation masks for
the controller — `Y_obs`/`U_obs`/`D_obs` — making explicit an assumption every
DPC run in this library already makes silently: that the policy sees every
modeled signal. An earlier draft added a fourth role `R` for the reference and
tariff signals; §3.2 records why that was wrong, and what would have to be true
to revisit it. Nothing here is renamed and no saved run stops loading.

**Diagnostics.** Two of the library's most consequential choices — which head
each output channel gets, and what the 2R2C storage prior should be — are made
by hand, before training, with nothing to inform them but the column name. A
small family of offline scripts that read the dataset and print a recommendation
would fix both, and neither needs any pipeline wiring. §4.

Everything else on the list is smaller, and §6 has it.

---

## 1. What exists

One `SystemPreview`, one `Node`, one autoregressive variable:

```
data dict:  {Y, U, D, t, yn}            get_sw_dict, data_utils.py:351-357
                │
Node(['yn','U','D'] -> ['yn'])          dynamics.py:727-733
                │
        SSMGrad.forward(y, u, d)        dynamics.py:317
                │
   ┌────────────┴────────────┐
   fy (+ optional fu, fd)    head_stack.activate()
   │
   └── PerZoneBlock          dynamics.py:189-265
         gather -> [block_0 … block_N] -> scatter back to Y order
```

Five facts that decide most of what follows:

| | |
|---|---|
| `nm_mapping` | `dict[str, list[str]]` — already a dict of named signal groups. The keys happen to be `Y`/`U`/`D`/`t`. |
| `norm_stats` | `{col: {shift, stretch, min, max, mean, stddev, median}}` — keyed by *column name*, flat across roles, persisted with the model. |
| [windows.py](src/neurotemplate/modeling/windows.py) | generic over `widths: tuple[int, ...]` — any number of groups. |
| `zones.Zone` | `(name, y_idx, u_idx, y_names)` — a per-signal-group block spec that already exists. |
| `PerZoneBlock` | the heterogeneous-block executor: an `nn.ModuleList`, already `vmap`-safe, already recursed into by `describe_dynamics`. |

The fixed-arity parts are narrower than they look — `SSMGrad.forward(y, u, d)`,
`_widths() -> (ny, nu, nd)`, `Node(['yn','U','D'], ['yn'])`, `get_sw_dict`'s
literal `['Y','U','D','t']`, and
[gradient_sampling.py:66-70](src/neurotemplate/modeling/gradient_sampling.py#L66-L70)
mapping role name to a `jacrev` argnum. Roughly 112 `"Y"`/`"U"`/`"D"` literals
across 16 files.

One structural property governs §3, so it is worth stating here.
[windows.py:21-27](src/neurotemplate/modeling/windows.py#L21-L27):

```python
def key_blocks(widths, in_steps):
    slices, offset = [], 0
    for width in widths:
        slices.append(slice(offset, offset + in_steps * width))
        offset += in_steps * width      # running sum over roles
```

Every gather index in the library — `last_step`, `per_step`, `time_major`,
`gather_columns`, `RCBlock`'s `_y_steps` / `_u_last` / `_d_last` — is a
cumulative sum of role widths, and `build_sliding_windows`
([data_utils.py:324](src/neurotemplate/common/data_utils.py#L324)) builds one
tensor per key from `df[nm_mapping[key]]`. **`nm_mapping` must partition the
columns.** A key that is a subset of another key either duplicates those columns
into the flattened window or shifts every offset after it.

---

## 2. Blocks

### 2.1 The spec

A *block* produces a declared subset of the predicted signals, using a declared
architecture, from declared inputs. The spec comes from a hook, extending the
existing `zone_mapping`:

```python
def block_mapping(nm_mapping, cfg):
    return {
        "north": {"Y": ["North Zone Temp [°C]"],
                  "U": ["North Cooling SP [°C]"],
                  "model": "2r2c"},
        "power": {"Y": ["Total Power [W]"],
                  "reads": ["North Zone Temp [°C]"],
                  "D": ["Ambient Temperature [degC]"],
                  "model": "mlp"},
    }
```

- **`Y` partitions.** Every predicted signal is produced by exactly one block.
  Nothing left out, nothing claimed twice, named before the first epoch.
- **`U` and `D` are subsets**, defaulting to all. Any U channel no block claims
  is shared with every block — one line of code, and it is what makes a single
  air handler serving several blocks expressible without a second concept.
- **`reads`** names other blocks' Y channels this block may see.
- **`model`** is a family name plus per-block overrides; `cfg.model` remains the
  default for blocks that do not name one.

Almost none of this needs new machinery. `PerZoneBlock` holds an `nn.ModuleList`
and never assumes its children share a type — `reset()` and `reg_error()`
already `getattr`/`hasattr` across heterogeneous children
([dynamics.py:247-256](src/neurotemplate/modeling/dynamics.py#L247-L256)). The
scatter step already guarantees the merged output is in `nm_mapping["Y"]` order,
which is what the loss, the heads, and the autoregressive feedback assume.
`windows.gather_columns(widths, in_steps, keep)` already takes arbitrary
per-key column tuples, so a real D subset replaces `tuple(range(nd))` with no
change to the helper. `describe_dynamics` already recurses and prints per-block
detail on the reportcard.

**[loss.py](src/neurotemplate/modeling/loss.py) and
[control/objective.py](src/neurotemplate/control/objective.py) have zero
awareness of blocks.** They see one flat `yn`. Heterogeneous blocks require no
loss changes at all, and the control half is untouched.

The work is: a richer `Block` dataclass, a `make_block` that dispatches per
block through the existing `model_catalog`, the `reads` gather, and validation.

**Rename on the way through.** A "zone" is a sub-model over a group of signals.
`zones.py` → `blocks.py`, `Zone` → `Block`, `PerZoneBlock` → `BlockStack`,
`zone_mapping` kept as a deprecated alias. That rename does more for "usable on
more systems" than any amount of role work, because `zone` is the most
misleading word in the API.

### 2.2 Cross-block reads, and the one refusal

Reads of another block's Y are ordinary inputs, handled like D reads, with one
exception.

They are ordinary because **all block inputs come from the shared history
window**, never from another block's same-step output. Gather and scatter stay
fixed permutations computed once at construction, there is no evaluation-order
dependency between blocks, and the whole thing stays `vmap`-safe — which is what
keeps the sampled gradient penalty alive. Same-step coupling would require a
topological sort and would cost all of that; it is not proposed.

**The exception is `2r2c`, and it is not bureaucratic — it is the family's
advertised guarantee.** `RCBlock` claims unconditional stability: bounded at any
horizon under any input, including the adversarial inputs a DPC policy goes
looking for. That claim rests on the state matrix being fixed by three positive
rates whatever the learned map does, which holds only because the forcing map
`g` reads `u` and `d` only
([rc.py](src/neurotemplate/modeling/rc.py)). Note the precise statement: the
block *does* read its own Y, in the fixed linear term `-a_o·z + a_m(m - z)`.
What it cannot do is let a *learned* map see state. A `reads` entry would have
to enter through `g`, and the eigenvalue argument evaporates.

So `RCBlock` rejects `reads`, the way it already rejects `residual: false` and a
non-residual head — a refusal at the block, with the reason at the site. Every
other family already reads its own Y through the window and makes no stability
claim, so for `mlp`/`nssm`/`lssm`/`gru` a cross-block read is exactly a D read,
as one would expect.

### 2.3 Heads

Already built and already orthogonal. `HeadStack` is per-channel, so "power is
bounded ≥ 0, temperature is regression" works today and composes with the block
split without either knowing about the other. What is missing is not the
mechanism but any help *choosing* — see §4.2. The one interaction is the
existing one: a non-`residual_compatible` head on a `2r2c` channel would never
restore the state, so
[get_dynamics](src/neurotemplate/modeling/dynamics.py#L657) names the offending
channels and refuses.

### 2.4 What it costs

Validation, mostly: Y partitions exactly, `2r2c` blocks reject `reads`, every
declared column exists, and the family whitelist stops being duplicated
(`model_catalog` at [dynamics.py:533](src/neurotemplate/modeling/dynamics.py#L533),
`check_one_of` at [config.py:168](src/neurotemplate/modeling/config.py#L168), and
`PER_ZONE_FAMILIES` are three lists of the same thing — derive the whitelist
from the registry).

The golden reportcard (`tests/golden/reportcard_nssm_baseline.md`) needs
regenerating if block labels change. Diff it by hand.

---

## 3. Roles

### 3.1 The partition

```
nm_mapping — every DATAFRAME COLUMN in exactly one role
  Y   the signals the model predicts and feeds back — all of them measured
  U   the control inputs: policy output at control time, recorded data at sys-id time
  D   disturbances — exogenous, enter the DYNAMICS
  t   clock

reference signals — exogenous, enter the COST, NOT columns (§3.2)
  ymin / ymax / price
```

Two invariants make this more than a relabelling:

- **D may be empty. Y and U may not.**
- **Y is the model's output, not its state.** §3.5.

The reference signals were going to be a fourth role, `R`. They cannot be, and
§3.2 explains why — the reason is worth reading, because it is the same
category error §3.4 catches for the regulated and observed sets, arriving from
a different direction.

### 3.2 R was wrong, and the reason is instructive

An earlier draft of this report proposed `R` as a fourth `nm_mapping` role,
giving `ymin`, `ymax` and `price` a home instead of being hand-plumbed through
four control modules. **That proposal rests on a false premise and is
withdrawn.** Recorded here rather than deleted, because the mistake is the
useful part.

**The premise was that these are columns.** They are not.
[scenarios.py](src/neurotemplate/control/scenarios.py) builds the control
window dict, and the split is stark:

```python
Y = _sliding(df, nm_mapping["Y"], W, stride, device)     # from the dataframe
U = _sliding(df, nm_mapping["U"], W, stride, device)     # from the dataframe
D = _sliding(df, nm_mapping["D"], W, stride, device)     # from the dataframe

ymin, ymax = load_control_hooks(cfg).comfort_band_signal(...)   # GENERATED
price = build_price(cfg, n, W, price_path, generator, ...)      # from a tariff file
```

`nm_mapping` maps a role to **dataframe column names**, and every consumer
depends on that: `get_sw_dict` does `df[nm_mapping[key]]`, `norm_stats` is
computed per column, `warn_degenerate_columns` and `data_dict_to_df` both
iterate columns. `df["ymin"]` does not exist and never will — the band is
sampled per window from a `BandSpec`, and the tariff is drawn from a per-split
CSV. Adding `nm_mapping["R"] = ["ymin", "ymax", "price"]` would make the
mapping mean two different things depending on the key, which is precisely the
category error §3.4 rejects for the regulated and observed sets.

**The invariant R was supposed to introduce is already enforced, structurally,
by the current design.** `build_price`'s docstring makes the argument
unprompted:

> *Price is exogenous data the policy and the objective read and the dynamics
> never sees, so it travels in the window dict rather than as a D column: a D
> column would be baked into the frozen surrogate's input width, and would tell
> the dynamics model that electricity price drives zone temperature.*

So "R enters the cost, never the dynamics" is not a property a new role would
add. It is the property that **keeping these signals out of `nm_mapping`
already guarantees** — a reference signal cannot leak into the surrogate's
input width because it was never a column to begin with. Promoting them to a
role would weaken the very invariant it was meant to state.

**What is left of the complaint.** The three signals really are threaded by
hand through `policy.py`, `closed_loop.py`, `control/pipeline.py` and
`control/reportcard.py`, and their preview depths travel two ad-hoc paths
(`band_future`, `price_future`). A declaration that named the reference
signals once — name, width rule, generator, preview depth — would collapse
that. But it would be a registry with one consumer and three permanently fixed
members, and [CLAUDE.md](CLAUDE.md) rule 4 is explicit that an extension point
needs a proven need. **The trigger is a fourth reference signal** — a
demand-response event schedule, an occupancy-driven band, a carbon-intensity
series. Build it then, shaped by that signal's actual requirements, and note
that the widths already disagree (`comfort_band_signal` returns
`[n_samples, nsteps, n_regulated]` while price is global), so the registry has
real design content that a fourth member would settle and abstraction in
advance would guess at.

### 3.3 The observation masks

```
Y_obs ⊆ Y    U_obs ⊆ U    D_obs ⊆ D        — what the controller sees
```

There is no `R_obs`. The controller is R's only consumer and consumes all of it,
so a mask over it would have exactly one setting.

**What this makes explicit.** [policy.py:499](src/neurotemplate/control/policy.py#L499)
sizes the policy net with `ny = len(nm_mapping["Y"])`, and
[policy.py:550](src/neurotemplate/control/policy.py#L550) reads all of `yn`.
Every DPC policy in this library reads every modeled output — full measurement,
in output-feedback terms — and nothing anywhere says so. The masks are how that stops being silent.

**Why per-role subsets are the right shape**, three ways:

1. They match how the policy input is built. `insize` is a sum of per-role
   terms, each with its own window depth; a flat mask over the union would have
   to be decomposed back into per-role gathers to construct anything.
2. The data structure exists. `windows.gather_columns(widths, in_steps, keep)`
   takes `keep` as per-key column tuples — which is exactly
   `(Y_obs, U_obs, D_obs)`. `PerZoneBlock` already uses it to give each block its
   slice. So a policy's observation spec and a block's input spec become one
   mechanism with two consumers.
3. Half of it is already in the API and being discarded. An observation spec has
   two axes, *which columns* and *how far ahead*. The temporal axis is already
   declarative and already per-column: `preview_config(nm_mapping) -> {signal:
   n_steps}`, whose template example names individual columns
   (`{"band": 2, "Ambient Temperature [degC]": 2}`). Then
   [policy.py:316](src/neurotemplate/control/policy.py#L316) does
   `d_future = max([spec.get(col, 0) for col in nm_mapping["D"]], default=0)` —
   the maximum, applied uniformly to every D column. Per-column intent is
   accepted by the signature and thrown away by the implementation.

**`U_obs` is a new capability, not a mask over something existing.** The policy
writes U and never reads it — `input_keys` is `["yn", "ymin", "ymax", "D"]` plus
price. A policy that sees its own previous action is standard and useful (rate
limiting, actuator state), and there is little reason to hide it, so identity is
the sensible end state. But it should **default to empty**, matching today:
switching it on changes `insize` for every existing policy and invalidates saved
control runs.

**The invariant.** An observation spec narrows a **consumer's read**; it never
narrows the **rollout dict**. `build_closed_loop` sets `nstep_key="D"`
([closed_loop.py:43](src/neurotemplate/control/closed_loop.py#L43)), so D
determines rollout length and the frozen surrogate needs every D column
regardless of what the policy sees. Filtering D when the batch is built — the
natural implementation — would silently break the dynamics model. Same for Y:
autoregressive feedback needs every Y channel even when the policy reads three
of them. It is a gather at the consumer, exactly as `PerZoneBlock`
gathers its slice from the one shared window.

**What still needs designing when this is built:**

- `_ReferenceRule` ([policy.py:182-203](src/neurotemplate/control/policy.py#L182-L203))
  is constructed with `ny=len(nm_mapping["Y"])`, and a residual policy is a rule
  plus a bounded correction. Restrict the net's observations and the baseline it
  corrects still reads everything. Either the restriction applies to both, or
  the reportcard says it applies to one.
- Regulating a channel that is not observed is legal, and is sometimes the point
  (inferential control) and sometimes a mistake. Permit it; name it.
- Restricting observations degrades closed-loop performance in a way no current
  metric isolates. The deliverable here is a reportcard comparison of achieved
  objective under full versus restricted observation. The knob is the easy half,
  and shipping it alone would add a quiet way to make control worse.

### 3.4 The regulated set stays a hook

`hooks.controlled_channels(nm_mapping)`
([template/control_user_defined.py:41](template/control_user_defined.py#L41))
returns `(names, indices)` into Y and is consumed at
[pipeline.py:156](src/neurotemplate/control/pipeline.py#L156),
[closed_loop.py:145](src/neurotemplate/control/closed_loop.py#L145),
[policy.py:483](src/neurotemplate/control/policy.py#L483) and the control
reportcard. It is the OCP regulated output and it needs a better name —
`regulated` — not a promotion to a role.

It cannot be a role for the reason in §1: it is a subset of Y, and `nm_mapping`
partitions. The same applies to `Y_obs` and friends, which belong in one
`observation_spec` hook, not in `nm_mapping`.

The underlying reason all of these want to be letters and cannot: OCP
conventions name *signals in a block diagram*, where `y = Cx` and `z = Hx` are
computed outputs and a signal may be in both or neither. `nm_mapping` names
*columns in a dataframe*. For Y, U, D and R the two coincide — each is a
genuinely distinct set of columns. For the regulated and observed sets they do
not, because those are projections of Y.

### 3.5 Why Y, and not X

`Y` is not a legacy name kept to avoid a refactor. It is the correct one, and
the reason matters enough to state.

In state-space notation `x` is the internal state and `y = Cx` the measured
output. This library models **only measured channels**: every column in
`nm_mapping["Y"]` is a dataframe column, and the update is
`y_{k+1} = f(y_k, u_k, d_k)` — autoregression over observations. That is an
input-output (NARX / output-error) structure, not a state-space one. Naming the
rollout variable `X` would claim an internal state the library does not
represent.

The qualification, because it is not quite "all black box": `RCBlock._mass` and
`GRUBlock._h` are genuinely hidden state. But they are *block-internal* — module
attributes cleared at sequence boundaries by `make_reset_init`
([dynamics.py:624-653](src/neurotemplate/modeling/dynamics.py#L624-L653)), never
in the rollout dict, never scored, never observed. So the structure is
`y_{k+1} = f(y_k, u_k, d_k; ξ_k)` with ξ private to a block. The dict names the
outputs; the hidden parts are correctly unnamed because they are not in it.

Three consequences:

1. **`Y_obs` is the most idiomatic notation available.** In the standard
   formulation you never mask `x` — you select through `C`. `Y_obs ⊆ Y` *is* a
   row-selection `C`: which measurements the controller has. And `e = r - y` is
   the textbook tracking error, which reads correctly with R and would not with
   an X.
2. **Renaming Y → X would spend the letter needed later.** If a dataset ever
   carries a latent proxy channel and hidden state is promoted to the rollout
   dict, the promoted variable wants to be `X` and the measured channels are
   still `Y`. Renaming now would take that letter and force a second rename then.
   Keeping Y preserves the option.
3. **Every role change in this document is therefore additive.** R is a new key.
   The masks are a new hook. `regulated` is a hook rename. Nothing touches the
   ~112 `"Y"`/`"U"`/`"D"` literals, the Neuromancer string contract
   (`Node(input_keys=[...])`, `variable("yn")`), `zone_spec.json`, the golden
   reportcards, or `nstep_key`. In particular `nm_mapping.json` is an MLflow
   artifact reloaded by `nt model load`
   ([load.py:164](src/neurotemplate/modeling/cli/load.py#L164)) — a renamed key
   would stop every saved run loading, and no such rename is proposed.

The vocabulary this lands on — `y`, `u`, `d`, `r` — is standard OCP notation for
an input-output plant, which is what a data-driven surrogate is.

## 4. Diagnostics

### 4.1 The pattern

Three of the library's decisions are made by hand, before training, with no
tooling behind them:

| decision | where it is declared | informed by |
|---|---|---|
| which head each Y channel gets | `data_head_config` hook | the column's name |
| the 2R2C storage prior | `rc_tau_*_h` config fields | a number you assert |
| the gradient-sign mapping | `grad/mapping.json` | domain knowledge |

The third is legitimately domain knowledge and already has a helper
(`write_gradient_mapping` in the template). The first two are **questions about
the data**, and the data can answer them.

So: a small family of offline scripts that read a dataset, compute one
statistic, and print a recommendation the user pastes into their hooks file.
They train nothing, are wired into no pipeline, and add no extension point —
which is exactly why they are cheap. Two rules keep them honest, both borrowed
from the physics checks:

- **Report the evidence, not just the verdict.** `scenarios.py` prints "closed
  4% of an 8.3 °C gap in 24 steps", not `free_decay_loss: 0.0271`, because the
  first is checkable by someone who knows the plant. A head recommendation must
  do the same.
- **Say what the statistic cannot see.** Every one of these reads a narrow slice
  of the data and a user will over-trust it otherwise.

They belong under a new CLI group, because they read the *dataset* where the
existing `nt model debug-heads` reads a *trained run*:

```
nt data heads   [--column NAME]     which head does this channel want?
nt data decay   --state Y --ref D   what is the envelope time constant?
nt model debug-heads --run_id ...   (exists) is the head I chose working?
```

That `debug-heads` already exists is the argument for the group, not against it.
It answers the posterior question — have these thresholds collapsed, is this β
so loose the bound never binds — and has no way to answer the prior one.

### 4.2 `nt data heads`

`data_head_config` is a required hook returning one head per Y column, and the
template's own worked example guesses from the name:

```python
# elif "power" in col.lower():
#     heads.append(BoundedHead(project_physical_to_norm(0.0, stats),
#                              float("inf")))
```

That is the failure this diagnostic exists for. A DX compressor's power reads
like a continuous non-negative quantity and is usually a **staged** one — off,
stage 1, stage 2 — where an `OrdinalHead` will beat a regression outright.

**The test is not "count the distinct values."** A staged compressor at
0 / 4.2 / 8.1 kW carries sensor noise and part-load variation, so it has
thousands of distinct floats and three modes. The channel looks continuous by
cardinality and is discrete by shape. What distinguishes them is **modality with
unoccupied gaps**: values cluster tightly and the space between clusters is
empty.

The statistic that captures this and stays explainable in two sentences: 1-D
k-means over `K = 2 .. K_max`, reporting for each K the share of variance
explained by the cluster means and the share of samples falling in the gaps
between clusters. A channel where `K = 3` explains 97% of the variance with
0.6% of samples in the gaps is staged. One where it explains 40% is continuous.
Recommend the smallest K past the elbow, and print both numbers so the verdict
can be overruled.

The same pass gives the other heads for free:

| shape in the data | head |
|---|---|
| multimodal, gaps unoccupied | `OrdinalHead(n_levels=K, level_values=...)` |
| exactly two clusters | `BinaryHead()` |
| mass piled at one or both range ends, continuous interior | `BoundedHead(lo, hi)` |
| unimodal, gaps occupied, no boundary mass | `RegressionHead()` |

**Output format is the deliverable.** Heads live in head-space — after
normalization — and the hook docstring already shows the projection idiom, so
the recommendation should be pasteable:

```
Total DX Power [W]  —  3 modes at 0, 4210, 8070 W
  99.4% of samples within 0.15 kW of a mode; gaps hold 0.6%
  K=2: 71% var explained | K=3: 97% | K=4: 97.4%  (elbow at 3)

  OrdinalHead(n_levels=3, level_values=torch.tensor(
      [project_physical_to_norm(v, norm_stats["Total DX Power [W]"])
       for v in (0.0, 4210.0, 8070.0)], dtype=torch.float32))
```

**One interaction it must know about.** `OrdinalHead.residual_compatible` is
`False`, and `2r2c` requires `residual=True` — so
[get_dynamics](src/neurotemplate/modeling/dynamics.py#L657) already refuses an
ordinal head on a channel produced by an RC block. A recommendation the user
cannot take is worse than none, so when the block spec is available the
diagnostic should say *"ordinal recommended, but this channel is produced by a
`2r2c` block — use a different family for it or accept `BoundedHead`."* This is
the kind of check that makes a diagnostic worth having rather than decorative,
and it is the same class of refusal `RCBlock` already carries four of.

**What it cannot see, and must say so:** the marginal distribution only. A
channel can be trimodal because three operating regimes are driven by an
exogenous variable while being perfectly continuous within each — ordinal would
be wrong there. For staged equipment this rarely bites, but the scope statement
belongs in the output, next to the numbers.

### 4.3 `nt data decay`

The 2R2C counterpart, argued in full in
[rc_identification.md](improvement_plans/rc_identification.md). `2r2c` asks the
user to assert `tau_slow` and gives them nothing to check it against — and the
free-decay physics check cannot do the checking, since at the default 4-hour
horizon it only fails a slow mode above ~198 h. It tests the sign of the
envelope coupling, not its rate.

Given a state column, a reference column, and the actuator `off` values
`SynthSpec` already declares, the script finds runs where the plant is
uncoupled, regresses `log|y - y_ref|` on time, and reports τ with the spread
across windows. Printing that beside the asserted `tau_slow` on the reportcard
is the missing stage one of the two-stage grey-box fit — the stages simply do
not share an optimizer.

It should **not** be wired into `nt model train`. Deciding which windows count
as free decay takes judgment, and a pipeline that guessed would produce a number
that looks measured and is not — the exact failure `rc_train_taus: false` exists
to avoid, moved one stage earlier.

## 5. Units, and domain specificity

Roughly 80–85% of the library is already domain-general: all of `common/`,
`hooks.py`, `windows.py`, `heads.py`, `loss.py`, `gradient_sampling.py`, the
`mlp`/`nssm`/`lssm`/`gru` families, `SSMGrad`, `PerZoneBlock`, the config/CLI
plumbing, and most of `control/`. The physics checks are general dynamical-systems
assertions wearing thermal vocabulary — hold at equilibrium, respond with the
declared sign, span a reachable set, relax toward a reference. `RCBlock` itself
is a two-state linear reservoir with learned state-independent forcing: a
battery with a diffusion layer, a jacketed tank, a soil column.

**The genuinely HVAC-shaped part is `SynthSpec`**, which requires exactly one
`ambient` D column and asserts the states relax toward it. Generalizing that to
an optional, possibly per-state *reference channel* is the single change that
would let someone model a battery or a reactor without lying to the spec — and
it is also the change most likely to be designed wrong in the abstract. Leave it
until a non-HVAC application shapes it. Vocabulary (`zone`, `comfort`,
`degree_hours`) is cheap to scrub whenever.

Two things are not domain specificity but misplacement, and are worth fixing on
their own:

- **`bracket_band`** ([control/policy.py:49-104](src/neurotemplate/control/policy.py#L49-L104))
  is a thermostat algorithm in library code that the library never calls — it is
  reachable only if an app's `reference_control` hook imports it. It belongs in
  [template/control_user_defined.py](template/control_user_defined.py) as example
  code.
- **Hardcoded units.** Below.

### Units

Units appear in two different guises and want two different fixes.

**Labels.** [control/pipeline.py:570-581](src/neurotemplate/control/pipeline.py#L570-L581)
hardcodes `"Temp [°C]"`, `"Setpoint [°C]"` and `"Power"` with no override — the
one place HVAC is an un-parameterized *behavior* rather than a name. The
modeling side is better but still wrong in shape:
`cfg.synth_y_unit: str = "°C"` ([config.py:142](src/neurotemplate/modeling/config.py#L142))
is *one* unit for *all* of Y, which is already incorrect for a mixed Y of
temperature and power — exactly the case §2 exists to enable.

A unit is per-column metadata, so it should live where per-column metadata
already lives: **`norm_stats`**. It is keyed by column name, flat across roles,
computed once, and persisted and reloaded with the model, so a `unit` field
travels for free and needs no new structure, no new hook and no new config field.
One accessor, `unit_of(norm_stats, col)`, returning `None` when absent; plots and
reportcards call it and fall back to the bare column name — which for this
dataset already reads `North Zone Air Temperature [°C]`, so the degraded output
is fine. `cfg.synth_y_unit` becomes a deprecated global default.

**Physical constants with unit-bearing defaults.** A different problem:
`band_lo: 23.0` / `band_hi: 27.0`
([control/config.py:110-111](src/neurotemplate/control/config.py#L110-L111)),
`setpoint_step: 0.5` "(°C)", `residual_delta` "(°C)". These are not labels; they
are °C baked into the defaults of a library that claims to be generic, and no
label mechanism fixes them. They should have **no defaults** — required when the
feature is used — or be expressed as fractions of an observed range, which is
the pattern [scenarios.py:67-70](src/neurotemplate/modeling/scenarios.py#L67-L70)
already uses for its pass thresholds (`EQUILIBRIUM_DRIFT_FRAC`,
`RESPONSE_MOVE_FRAC`, `DECAY_MIN_CLOSED`). Reusing that convention keeps one idea
in the codebase instead of two.

---

## 6. What else is missing

Ordered by how much each blocks "more systems" / "easier choices".

1. **The seed ensemble is built and thrown away.**
   [optuna.py:66-86](src/neurotemplate/modeling/cli/optuna.py#L66-L86) trains
   `trial_seeds` models per trial and returns the mean dev loss, then discards
   them. There is no uncertainty anywhere in the library — the readout quantiles
   at [scenarios.py:200](src/neurotemplate/modeling/scenarios.py#L200) are
   within-rollout order statistics, not predictive spread. So every number on the
   reportcard is a single draw: `tau_slow`, the physics-check verdicts, the
   family RMSE comparison, the prior sweep in
   [rc_identification.md](improvement_plans/rc_identification.md). It is entirely
   possible some published family comparisons are within seed noise, and nobody
   can currently tell. Keeping the models and reporting the spread is the
   cheapest real win in this document. Seeds are a property of a *campaign*, not
   a model, so the loop stays in `optuna.py` and no block, family or hook ever
   sees one.
2. **The pre-training choices have no tooling.** Which head each channel gets
   and what the 2R2C storage prior should be are both decided by hand from the
   column name. Both are answerable from the data. §4.
3. **NaN handling does not exist.** `_load_split` raises on any NaN
   ([data_utils.py:158-159](src/neurotemplate/common/data_utils.py#L158-L159)),
   and [cli/load.py:231](src/neurotemplate/modeling/cli/load.py#L231) does it
   again. Every real dataset outside a simulator has gaps. This is the largest
   barrier to onboarding a new system and it is invisible until someone tries.
   Needs gap-aware windowing or a mask channel.
4. **`preview_config`'s per-column depth is discarded.** §3.3, item 3. A defect
   on its own, and half of `D_obs` delivered early.
5. **The modeling side cannot see forecasts.** `get_dynamics` hardcodes
   `"future": 0` ([dynamics.py:719](src/neurotemplate/modeling/dynamics.py#L719))
   although `SystemPreview` supports it and the control side uses previews
   heavily. That physical systems are not future-informed is true of the *plant*
   and is an argument for the *default*, not for hardcoding: a surrogate of a
   plant with a known schedule — a tariff, an occupancy calendar, a committed
   dispatch — can legitimately read ahead.
6. **`MAX_IN_STEPS = 12`** ([config.py:18](src/neurotemplate/modeling/config.py#L18))
   is a hard library-wide ceiling on context length, defensible for buildings on
   5-minute data and arbitrary otherwise. A ceiling on a value the user sets
   deliberately is the wrong shape of guard; the real risks (memory, a window
   longer than the training horizon) are better served by a warning that states
   the cost. It also sits directly on top of the one lever that moved `tau_slow`
   at all in the RC experiments.
7. **Column-name → tensor-index is re-derived at ~a dozen sites** via
   `list.index(...)` — in `data_utils`, `zones`, `scenarios`, `policy`, and in the
   user hook template
   ([control_user_defined.py:245](template/control_user_defined.py#L245)). One
   `resolve(nm_mapping, role, col)` helper would DRY this, and it is a
   prerequisite for anything that reshuffles groups.
8. **No Δt awareness outside 2R2C.** Only `RCBlock` takes `steps_per_hour`; every
   other family bakes the sampling rate into its weights, and nothing says so.
   Making a discrete-time family genuinely Δt-aware means making it
   continuous-time, which is a neural-ODE question and a different project. The
   tractable middle is a guard that refuses evaluation at a sampling rate other
   than the training one.
9. **`t` is only `sec_elapsed`, counted from file start** — already flagged in the
   README as a footgun for price alignment. Calendar features have to be
   user-supplied D columns.

---

## 7. Plan

The ordering principle: **things that make an existing claim checkable come
before things that add capability.** The library's value is that it is
opinionated — the physics checks, `objective_balance`, the residual-vs-reference
discipline are good *because* they assume something — and every number it prints
without an error bar spends a little of that.

### Tier 0 — make the existing claims checkable (~2 days)

None of these generalizes anything, and none waits on a second system. Each
attaches evidence to a number the library currently prints — or a choice it
currently asks for — without any.

- **Seed-ensemble spread** (§6.1), reported on the physics-check verdicts, test
  RMSE, and `tau_slow`.
- **`nt data decay`** (§4.3), printed beside the asserted `tau_slow`.
- **`nt data heads`** (§4.2), recommending a head per column from the shape of
  its distribution rather than its name. Independent of the other two and of
  Tier 1; the ordinal-vs-regression case is the one with a real accuracy
  payoff, and the `2r2c`-plus-ordinal conflict check is what stops it producing
  advice that cannot be taken.

Plus three one-liners that state an assumption or honor a declaration already
accepted:

- Reportcard: *"Policy observes all N Y channels — no measurement restriction."*
- Honor `preview_config`'s per-column depth instead of maxing it across D.
- Rename `controlled_channels` → `regulated`, documented as the OCP regulated
  output.

### Tier 1 — blocks (~2-3 days)

§2, end to end.

- `zones.py` → `blocks.py`; `Zone` → `Block` gaining `d_idx`, `reads` and
  `model`; `zone_mapping` and `Zone` kept as deprecated aliases.
- `PerZoneBlock` → `BlockStack`; `make_block` dispatches per block through
  `model_catalog`; `block.d_idx` replaces `tuple(range(nd))` in the
  `gather_columns` call.
- Validation: Y partitions exactly, `2r2c` rejects `reads`, declared columns
  exist. Derive the family whitelist from `model_catalog` rather than keeping
  three lists.
- Reportcard: `_describe_block` already recurses; relabel.
- Tests: a mixed-family case (2R2C on one channel, MLP on another) asserting the
  scatter restores Y order and that `reset()` reaches only the stateful child;
  a `reads`-on-2R2C refusal; a cross-block read that changes the reading block's
  output and not the read block's.

### Tier 2 — units and the misplacements

- **Units** (§5): a `unit` field on `norm_stats` with one accessor;
  `plot_closed_loop`'s three labels parameterized; `cfg.synth_y_unit` deprecated.
- **Unit-bearing config defaults** (§5): `band_lo`/`band_hi`/`setpoint_step`/
  `residual_delta` lose their °C defaults or become fractions of observed range.
- **Move `bracket_band`** into the template.
- **`MAX_IN_STEPS`** becomes a warning (§6.6).

### Tier 3 — paid for by a real dataset

- **NaN handling** (§6.3) — first among these the moment a real dataset arrives.
- **The `observation_spec` hook** (§3.3) — `Y_obs`/`U_obs`/`D_obs`, gated on a
  system that actually withholds a sensor, and shipped *with* the full-versus-
  restricted objective comparison, not without it.
- **Forecast previews on the modeling side** (§6.5), defaulting to `future: 0`.
- **`SynthSpec.ambient` → an optional reference channel** (§5), when a non-HVAC
  application shapes it.
- **Δt guard** (§6.8).

### Explicitly not recommended

- Dissolving the roles into generic named groups. U is what the policy writes
  and what needs bounds; D is what must be supplied exogenously and is why
  `nstep_key="D"` works at all. Replace them with generic groups and the first
  thing DPC does is re-derive which is which.
- A regulated-output or observed-set *role* in `nm_mapping`, `Y_obs` included
  (§1, §3.4). They are subsets of Y; roles partition.
- Filtering the rollout dict to a consumer's observed columns (§3.3).
- Renaming `Y` to `X` (§3.5). The library models measured outputs, not internal
  state; `X` would be the wrong name and would spend the letter a future latent
  variable needs.
- **`R` as a fourth `nm_mapping` role (§3.2).** `ymin`/`ymax`/`price` are
  generated signals, not dataframe columns, and keeping them out of the mapping
  is exactly what guarantees they cannot reach the surrogate's input width. A
  reference-signal registry waits on a fourth reference signal to shape it.
- Promoting the RC mass — or any block's hidden state — to a rollout variable,
  until a dataset carries a latent proxy channel. `RCBlock._mass` and
  `GRUBlock._h` are module attributes reset at sequence boundaries by
  `make_reset_init`, which is a good design. Promoting them costs an init policy,
  a windowing decision, carriage through `build_closed_loop`, and a loss taught
  not to score them. The trigger is a wall or slab sensor that would turn
  `tau_slow` from a frozen prior into a fitted parameter — and note the mass is
  *structurally* unobserved, so no framing recovers information the data does not
  contain.
- Same-step coupling between blocks (§2.2), which would cost `vmap`-safety and
  the gradient penalty with it.
- Any general "graph DSL" for wiring nodes. Two nodes exist today and there are
  three `Node()` construction sites in the repo — an abstraction with no repeated
  pattern to earn it.

---

## 8. Verification

- `uv run pytest` — 465 tests. Tier 1 moves the zone, RC, GRU, heads and
  reportcard suites; regenerate `tests/golden/reportcard_nssm_baseline.md` by
  hand rather than blind.
- `nt model train` against
  [tests/fixtures/modeling_user_defined.py](tests/fixtures/modeling_user_defined.py)
  with a mixed-family block spec; confirm the reportcard prints per-block
  families and that `2r2c`'s `tau_slow` still reads back at the configured prior.
- `nt model load` on that run — per-block topology rebuilt from the logged spec,
  weights loaded without a remap.
- `nt control train --set sysid_run_id=<id>` — the control half should be
  genuinely untouched by Tier 1, since it only ever sees flat `yn`/`U`/`D`.
- Stability regression: roll a mixed-block model out for 10× the training horizon
  under adversarial inputs and assert the `2r2c` channels stay bounded.
  `test_rollout_stability.py` is the place.
