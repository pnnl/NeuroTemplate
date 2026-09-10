# Report: tuning the binary and ordinal heads for rare-event channels

Branch `head-tuning`, eight commits on top of `758647e`. **636 tests pass**
(600 before; 36 added). Implements the six-item "What to change, in order"
list from `modelica_rtu_mall/head_bakeoff/README.md`. Nothing in it turned out
to be at odds with the library's ethos; one recommendation (the calibration
regime) was deliberately done differently from the bake-off, and one aside
(a decision threshold on `BinaryHead`) was deliberately not done. Both are
explained below.

The bake-off's root cause was that nothing in the pipeline optimises a
criterion that values a rare level. After this branch, three things do: the
loss (class weights), the decision rule (calibrated thresholds), and the
report (balanced accuracy beside NRMSE, with persistence as the bar). The
System's output contract is unchanged — no `z` in the output dict, no hook
signature change, no new head kind.

---

## What changed, against the README's list

| # | Recommendation | Done | Where |
|---|---|---|---|
| 1 | Classification metrics for non-regression heads | ✅ | `common/metrics.py`, reportcard §"Level metrics", MLflow, `run_summary.json` |
| 2 | Make the decision rule tunable, calibrate it | ✅ | `OrdinalHead.decision_thresholds`; `modeling/calibration.py`; `cfg.calibrate_thresholds` |
| 3 | Class weights on the existing MSE / BCE loss | ✅ | `OrdinalHead.level_weights`, `BinaryHead.level_weights`; `cfg.balance_levels` |
| 4 | `recommend_head` reports mode occupancy, warns under 5 % | ✅ | `diagnostics.py` (`RARE_MODE_FRAC`) |
| 5 | Persistence as a reportcard reference row | ✅ | `pipeline.evaluate_and_plot`, reportcard §"Test Metrics" |
| 6 | Leave β at 15 and activation thresholds at midpoints | ✅ (untouched) | — |

### 1. Heads: `decision_thresholds`, `level_weights`, `levels()` — `c7eef52`

[heads.py](src/neurotemplate/modeling/heads.py) is where both fixes land, as
the README said they could.

- `OrdinalHead` gains `decision_thresholds` (K−1; where `decode` snaps;
  default the activation midpoints) and `level_weights` (K; default ones).
  `BinaryHead` gains `level_weights` (2). Both are constructor arguments that
  round-trip through `config()`.
- The weighted loss is `Σ w·err / Σ w`, so all-ones reproduces the unweighted
  mean to floating point and the loss scale does not move with the weights.
  Pinned by test: every existing run's loss means what it meant.
- `Head.levels()` / `Head.level_index(y)` — one nearest-level rule serves the
  weighted loss (target → level) and the classification metrics (emitted value
  → level). Regression and bounded heads return `None` and are never scored as
  classifiers.
- `inverse_frequency_weights(y, levels)` builds balanced weights from a column,
  averaging to 1; a level the data never visits counts as one sample, not
  infinity.

**A serialization subtlety worth knowing.** The two new buffers are
`persistent=False`. Training saves `best_model_state_dict.pth` *before*
calibration runs, so if `decision_thresholds` lived in the state dict,
`nt model load` would rebuild the head from `head_config.json` (calibrated)
and then `load_state_dict` would overwrite it with the midpoints. Keeping them
out of the state dict makes `head_config.json` their single source of truth.
Test `test_calibrated_thresholds_survive_load_state_dict` pins this.

`HeadStack.forward` now wraps `activate`, and `SSMGrad` calls the stack through
`__call__`. That is the entire mechanism by which calibration reads `z`: a
standard `register_forward_hook`, no output-contract change.

### 2. Reportcard: level metrics and persistence — `37f81e2`

[common/metrics.py](src/neurotemplate/common/metrics.py) gains `LevelMetrics`
(per-level recall, true share, predicted share; `balanced_accuracy` = mean
recall over the levels the truth visits, so a constant scores 1/K) and
`level_metrics(pred_idx, true_idx, levels, name)`.

`evaluate_and_plot` computes, for every levelled channel, the level metrics of
the rollout **and** of persistence (hold the last seeded state for the whole
window — the same information the rollout starts from), plus persistence's
RMSE/MAE/R²/NRMSE for every channel. All of it reaches:

- the reportcard: a persistence table under the test metrics; a new
  "Level metrics (staged channels)" section with balanced accuracy beside
  persistence's, one row per level, and the ordinal decision thresholds in
  physical units;
- MLflow: `test_<ch>_balanced_acc`, `persistence_<ch>_rmse`,
  `persistence_<ch>_balanced_acc`; `level_metrics.json`,
  `persistence_metrics.json`;
- `run_summary.json`: `worst_level_balanced_acc` (only when there are levelled
  channels, so the golden file is untouched);
- the seed-ensemble spread (`EnsembleSpread.balanced_acc`) — the bake-off's
  DX ROC-AUC of 0.744 ± 0.271 is exactly the kind of seed variance RMSE hides;
- `nt model load`'s stdout.

Runs without levelled heads render byte-identically (golden test unchanged).

### 3. Calibration and balancing in the pipeline — `161e1a8`, `c7c1f95`

New [modeling/calibration.py](src/neurotemplate/modeling/calibration.py), two
config flags, both **default on**:

- `balance_levels` — `pipeline.build_head_stack` (shared by the `train` and
  `optuna` CLIs, so a search trial trains against the same loss as the final
  retrain) sets every levelled head's `level_weights` from the training pool.
- `calibrate_thresholds` — after `train_with_curriculum`, for every ordinal
  head: roll the model out on dev at `dev_pred_steps`, read `z` through the
  hook, and move the thresholds to the balanced-accuracy optimum by coordinate
  ascent over 201 quantile candidates (exact for K=2; a local optimum for K>2,
  which is what the bake-off's 2-D grid found too). A channel with a level dev
  never visits is skipped with a warning — there is no recall to maximise.

**The regime is deliberately not the bake-off's.** The README calibrated on
teacher-forced one-step `z` from a tabular model and flagged, under *Untested*,
that "the tuned θ is calibrated on teacher-forced z, and in the rollout the z
distribution shifts … a calibration applied across a distribution shift is
exactly the kind of thing that fails quietly." This branch calibrates on the
dev *rollout* — the regime the dev loss is measured in and the surrogate is
judged in — which removes one distribution shift. It does not remove the
other: thresholds are tuned on `z` produced under the *old* rule, and the new
rule changes what is fed back. So `c7c1f95` adds the one guard the risk calls
for: a second dev rollout under the tuned rule, and any channel whose
*realized* balanced accuracy did not improve is reverted with a warning that
says so. The trainer prints `dev balanced acc before -> after` per channel.
This guard fired on the 2-epoch test model and did not fire on the 40-epoch
smoke run below — both are the right answer.

`get_dynamics` now attaches a **deep copy** of the head stack to the model.
Calibration writes into the stack the System trained with; without the copy,
ensemble member 2 (and every optuna trial) would train with member 1's
calibrated thresholds. `head_config.json` is logged from the System's own
stack for the same reason. A caller who wants the trained/calibrated stack
reads `dyn_sys.nodes[0].callable.head_stack`, which `cli/load.py` and
`cli/debug_heads.py` already did.

### 4. `nt data heads` reports occupancy — `363fa7f`

`LevelFit.mode_shares` is recorded in the K sweep; the evidence prints "mode
shares: 0 -> 97.7%, 61728.4 -> 2.3%" and, for a mode under `RARE_MODE_FRAC =
0.05`, a "RARE LEVEL" line saying what will happen (unweighted loss and midpoint
snap never emit it; NRMSE rewards that) and what to do (keep both flags on;
judge by level metrics). Same for a 0/1 flag whose ones are rare. On
`data/spring/train.csv` the output reproduces the README's table exactly:
DX 2.5 %, fan 4.5 %, furnace 2.3 %.

### 5. Docs and tooling — `b868f1d`, `a90a4c9`

README (diagnostics section, layout, test count), the template hook docstring
and `modeling.yaml`, `debug-heads` (prints decision thresholds and level
weights; draws the snap points on the ordinal plot), `nt model load` (prints
balanced accuracy vs persistence).

---

## Smoke run on the real data

Not an experiment — one seed, 40 epochs, patience 8, `hb_gru_advice.yaml`,
`HEAD_MODE=advice`, both flags on. Run and reloaded in an isolated MLflow store.
Test split, **287-step rollout** (the regime the README measured its
catastrophic failure in):

| Channel | Balanced acc | Persistence | Rare-level recall | Rare-level pred share (true) | NRMSE (persistence) |
|---|---:|---:|---:|---:|---:|
| DX Cooling Power | **0.855** | 0.500 | 0.919 | 23.7 % (3.8 %) | 0.468 (0.179) |
| Fan Power (K=3) | 0.471 | 0.336 | 0.699 | 27.5 % (5.9 %) | 0.446 (0.417) |
| Furnace Gas Power | **0.920** | 0.461 | **1.000** | 17.9 % (2.3 %) | 0.395 (0.404) |

For comparison, the README's h=287 numbers: every configuration at or below
persistence on the furnace (0.744 at h=1; recall **0.000** at h=287); DX
0.676 best case; fan top-level share **0.0000 at every β**. Calibration moved
the thresholds only a little here (furnace 0.500 → 0.495 in head-space,
dev 0.746 → 0.748); the class weights did most of the work at this training
length, which is consistent with §8 of the README.

Three honest readings of that table:

- **NRMSE is worse than persistence on all three channels** while balanced
  accuracy is far above it. That is the anti-correlation the README documented,
  now visible on one page.
- **The precision cost is real and unmeasured in control terms.** The furnace
  fires on 17.9 % of steps against a 2.3 % base rate. The README's *Untested*
  section says where that trade belongs (what does a spurious furnace fire
  cost the policy?) and nothing here answers it. Balanced accuracy weights both
  errors equally; if the costs are asymmetric the criterion in
  `calibration._balanced_accuracy` is the one function to change.
- **`nt model load` reproduces the training-time level metrics exactly**, so
  the calibrated rule really does travel with the run.

The reproduction commands in the README (`HEAD_MODE=advice uv run nt model
train --config head_bakeoff/configs/hb_gru_advice.yaml`, then
`head_bakeoff/score_nt.py`) still work unchanged and are the way to get a
seeded, apples-to-apples number.

---

## Decisions that differ from the README, and why

**No decision threshold on `BinaryHead`.** The README's §9 says "BinaryHead is
simpler still … its decision boundary is the same single scalar." In this
library `BinaryHead` has no `decode`: the loop feeds back σ(z), a probability,
by design, and BCE on a snapped 0/1 value is degenerate (−log 0). A threshold
that only affected the *metric* would report a decision the surrogate never
emits. So binary gets class weights only, and its level metrics read the
probability at p = 0.5. Documented in the class docstring. None of the bake-off
channels use `BinaryHead` (all three are `OrdinalHead` at K=2–3).

**Calibrated on dev rollouts, not teacher-forced one-step.** See §3 above. The
+0.108 "training data alone" variant was not implemented: the pipeline already
selects on dev, the README itself calls the dev variant the same category of
choice, and calibrating on the training rollout would tune the rule against
windows the weights were fitted on.

**Both flags default on.** The README's finding is that the defaults were
ignoring the rare level; keeping the fix opt-in would leave them doing so.
Every existing config keeps loading (`_config_from_run_params` tolerates new
fields); a run without levelled heads is unaffected by construction; a run
with them changes behaviour, visibly, on the reportcard.

**One guard, not a fallback chain.** The revert-if-worse check is the single
place calibration can decline, and it warns when it does. It is there because
the README named the failure mode; it is not there "just in case".

---

## Untested, still

Carried over from the README's own list, unchanged by this branch:

- Whether a better one-step / dev-rollout classifier makes a better
  **closed-loop** surrogate. The control side never sees these metrics.
- The **precision cost** in control terms.
- **Transfer across periods** — thresholds tuned on spring dev applied to a
  different season. The revert guard checks the dev rollout only.
- Optuna still selects on `dev_loss`. The README's warning that a tuning loop
  pointed at NRMSE selects for a model that predicts nothing applies to a
  loop pointed at MSE too; `test_<ch>_balanced_acc` is now logged per trial
  retrain, but nothing uses it as an objective.
