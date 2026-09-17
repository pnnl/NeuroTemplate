# PreStack MVP: what a real run shows

Written 2026-09-17 against `prestack` branch `e75c658`. Implements and measures
[prestack_demand_plan.md](prestack_demand_plan.md) Part VII on
`modelica_rtu_mall/data/spring` (shoulder season, both actuators live), the
dataset and recipe of the `generalization_push` campaign.

## The result in one table

Every policy is scored through its own frozen surrogate, as in every campaign on
record. The number that says whether the *surrogate* got more honest is the
bracket-band rule's violation on it: on the real plant that rule holds comfort
almost perfectly, so its surrogate score is a measure of the surrogate's
actuation realism, not of the rule.

| surrogate | rule violation on it | rule mean exceedance | learned policy violation | learned − rule | learned mean power vs rule |
|---|---:|---:|---:|---:|---|
| m01 `mixed`, no declaration (the campaign's c08) | **54.6 %** | 0.351 | 13.5 % | −41 pts | 716 vs 562 W (+27 %) |
| m02 `mixed` + declaration | **15.6 %** | 0.045 | 9.6 % | −6 pts | 811 vs 851 W (−5 %) |
| m03 `mixed` + declaration, demand into the MLP | 25.4 % | 0.047 | 7.4 % | −18 pts | 1031 vs 836 W (+23 %) |

The declaration cuts the rule's violation on the surrogate from 54.6 % to 15.6 %
and its mean exceedance 7.8×. The learned policy's margin over the rule drops
from 41 points to 6 — the honest margin; the baseline surrogate was mostly
rewarding the policy for beating a fiction. On m01 the learned policy spends 27 %
more power than the rule to hold the band (the "cool harder" bias the plan
describes); on m02 it holds the band better than the rule on 5 % *less* power.
Power comes from the plant MLP block, which also changed (it sees the demand
feature now), so read the energy column as suggestive.

## Setup

| | |
|---|---|
| data | `data/spring` train/val/test, 5-minute, z-scored; Y = zone temp + DX, fan, furnace power (furnace on a `BinaryHead`); U = cooling and heating setpoints; D = outdoor dry-bulb, GHI |
| topology | `BLOCK_SPEC=mixed`: thermal block `2r2c` (zone), plant block `mlp` (three meters). The campaign's best. |
| declaration (`mixed_demand`) | thermal block: `ambient` = outdoor dry-bulb; `inputs`: cooling setpoint `demand` sign −1, heating setpoint `demand` sign +1 (both against the zone), GHI `linear` +1. Every thermal column declared, so the block has no MLP. The plant block is undeclared but, like every family, receives the demand features in place of the raw setpoints. |
| `mixed_demand_mlp` | as above with `"branch": "mlp"` on both demands |
| `demand_zone` | the same declaration on a zone-only Y with no `model` key, so a monolithic `nssm` takes it through the pre-stack alone |
| model recipe | `gp_m13`: `in_steps 12`, `2**7 × 2` MLP, 288-step rollouts, `ensemble_seeds 3`, seed 1234; tau prior 2 / 1 / 4 h |
| control recipe | `gp_c08`: band [20, 24], `comfort_weight 50`, `du_weight 50`, `energy_weight 0`, 4 h horizon, seed 1234 |

Run ids (`modelica_rtu_mall/mlruns`): m01 `4ee66a99`, m02 `dc613b70`, m03
`116e4fe9`, m04 `59a74d44`, m05 `245e2d03`; c01 `390e6dfd`, c02 `5bd32026`,
c03 `ec563771`. m01/c01 reproduce the campaign's m13/c08 (zone NRMSE 0.1433,
violation 13.51 % vs 13.56 %).

## Surrogates

Median of 3 seeds for the physics values; zone NRMSE is the logged seed's, on
287-step test windows (persistence: 0.2063).

| | m01 mixed | m02 + declaration | m03 demand→mlp | m04 nssm zone | m05 nssm + demand |
|---|---:|---:|---:|---:|---:|
| best dev loss | 0.225 | 0.197 | **0.194** | 0.390 | **0.218** |
| zone NRMSE / R² | 0.143 / 0.55 | **0.120 / 0.68** | 0.134 / 0.61 | 0.119 / 0.69 | **0.106 / 0.75** |
| 7-day open-loop RMSE / bias (°C) | 0.595 / +0.16 | **0.544 / +0.06** | 0.663 / +0.41 | 0.605 / +0.09 | 0.519 / +0.13 |
| equilibrium drift (limit 0.168) | FAIL 0.67 | **pass 0.038** | FAIL 0.61 | FAIL 1.11 | FAIL 0.64 |
| cooling step (−, ≥ 0.084) | pass −1.16 | pass −1.97 | pass −1.22 | pass −1.58 | pass −2.38 |
| heating step (+, ≥ 0.084) | **FAIL** on 2/3 seeds | pass +1.24 (all seeds) | pass +1.58 | pass +1.79 | pass +0.71 |
| reachability span (°C / 4 h) | 1.36 | 3.03 | 2.86 | 2.91 | 2.67 |
| free decay, gap closed in 4 h | 21.7 % | 8.8 % | 3.2 % | 12.6 % | 10.5 % |
| furnace balanced accuracy | 0.50 | **0.80** | 0.68 | — | — |
| DX / fan power RMSE (W) | 692 / 402 | 805 / 362 | 808 / 374 | — | — |

The plan expected the one-step NRMSE to be no better and possibly worse. It is
better for both families, and the black-box pair (m04 → m05) answers the plan's
separate question: the feature alone, with no structural storage behind it,
takes nssm from 0.119 to 0.106 and halves its dev loss and its equilibrium drift.

**Free decay is the one row that reads like a regression and is not.** The
spring data has no coasting episode long enough for `nt data decay`, but a
regression of the zone's step change on outdoor gap and GHI over the 3 183
coasting rows (both demands zero, both meters at zero) gives `a_o = 0.0019`/step
(`τ_ambient ≈ 44 h`), `k_GHI = 3.3e-5`, gains +0.07 °C/h, and a mean drift of
−0.23 °C/h at a 12–20 °C gap — about 8 % of an 11.6 °C gap in 4 h. m02's 8.8 % is
the plant; m01's 21.7 % was the overstatement. This building is much better
insulated than the shipped 2 h ambient prior says, and than the summer decay
episodes in the README (6–11 h) suggested.

### What m02 learned, in physical units

| term | learned | empirical (coasting regression) / plant |
|---|---|---|
| `τ_ambient` | 20.1 h (from the 2 h prior) | ≈ 44 h, correlated with solar |
| solar `k` | 1.05e-4 °C/step per W/m² (+0.02 °C/step at 200 W/m²) | 3.3e-5 |
| cooling | `cap` 0.77 °C/step over `w` 8.6 °C → 0.09 °C/step per °C of demand, `slope_share` 0.10 | DX coil −0.09 °C/step at full stage |
| heating | `cap` 0.20 °C/step over `w` 0.23 °C, `slope_share` **0.97** | furnace +0.48 °C/step at full stage, 2.3 % duty |
| modes | `τ_fast` 0.78 h, `τ_slow` 104 h | — |

The cooling branch is in its proportional regime — the time-averaged response of
a cycling single-stage coil, which is what a 5-minute model of a thermostat can
be — and its coefficient matches the plant's coil to the second digit. The
heating branch is a near-step at the deadband and sits at the stability bound:
the parameterization is doing exactly what it was built to do, and the
reportcard says so.

## What broke, and what was fixed

The first declared run was worse than persistence (zone NRMSE 0.28, R² −0.76),
drifted 2.8 °C at equilibrium, and moved the zone the *wrong* way under a cooling
step. Three fixes, each traceable to one reportcard number; all in the library,
committed as `e75c658`.

1. **The frozen ambient prior became load-bearing.** With `a_o (T_oa − z)`
   explicit, `τ_ambient = 2 h` asserts a 5.5 °C/h loss to a 10.9 °C-colder
   outdoors. The undeclared block never had to form `+a_o·T_oa` at full strength
   — its MLP learned whatever coupling fit — but the declared block has no such
   freedom, and the one free term left (an intercept) learned +4.5 °C/h of
   fictitious internal gain to cancel it. That balance holds on the training
   distribution and nowhere else, which is why every scenario (ambient = zone,
   HVAC off) drove the zone up until the cooling `off` setpoint engaged.
   *Fix:* declaring `ambient` unfreezes the ambient rate, starting from
   `rc_tau_ambient_h`. The plan anticipated this ("once the reportcard shows
   the prior is wrong"): the term has no hidden state behind it, so the
   argument that freezes the mass rates does not apply.

2. **The branch scalars could not move.** Unfrozen, `τ_ambient` went 2.00 →
   2.04 h. The rates' `MIN_TAU_STEPS + softplus(θ)` form is linear in the time
   constant, so 2 h → 8 h is 72 units of raw travel, and this run makes ~600
   Adam updates at `lr = 1e-3`. The same arithmetic capped every branch: the
   heating `cap` needed θ to move 3.5 and got 0.5. *Fix:* the declared
   `AmbientTerm` owns its rate in log-space (`τ = MIN + exp(θ)`, 1.4 units for
   the same trip), and every declared branch's parameters go in their own Adam
   group at `cfg.rc_branch_lr` (default 0.05). The storage priors stay at
   `cfg.lr`, so `rc_train_taus` behaves as before.

3. **A fully declared block had a free intercept.** My addition, not the plan's
   ("a constant offset belongs in the MLP's intercept" — and with everything
   declared there is no MLP). It is unidentifiable against `a_o`: the two trade
   off along a ridge that only the zero-gap equilibrium scenario can see, and
   the fit landed 3× / 8× too high on both, with a +0.44 °C warm bias over a
   7-day rollout. *Fix:* no free constant in a fully declared block. That one
   change turned all five physics checks green (equilibrium 0.47 → 0.04) and
   took the zone NRMSE from 0.133 to 0.120. Unmetered gains now have to enter
   through a declared column (occupancy, say) or go unmodelled — the grey-box
   stance, stated in the `rc.py` docstring.

Also added on request: `"branch": "mlp"` on a `demand` entry routes the feature
into the RC block's residual MLP instead of the bounded branch. m03 measures it:
the fit is a little worse, the equilibrium check fails again (the MLP's bias is
the intercept of fix 3), and the reportcard names the block as unbounded. The
learned policy scores lowest on it (7.4 %) while the rule scores worse than on
m02 (25.4 %) — a surrogate that is easier to control and less faithful, which is
the pattern the campaigns keep finding. Prefer the saturating branch.

## Everything changed

### `neurotemplate`, branch `prestack` (two commits, 358 tests pass)

`8c5bcbb` — the plan as written, plus:
- `heads.py`: `soft_clamp_min`, `straight_through` (HeadStack now uses it),
  `PreTransform` / `Demand` / `PreStack` with `to_config` / `from_config` /
  `describe`.
- `blocks.py`: `inputs` and `ambient` on `Block`, plus `u_names` / `d_names`;
  validation collects every problem per block.
- `rc.py`: `AmbientTerm`, `LinearTerm`, `SaturatingTerm`; MLP over the
  undeclared columns; `describe()` reports each branch; undeclared block
  byte-identical, state_dict included (pinned by a test).
- `dynamics.py`: pre-stack applied in `SSMGrad.forward`, built from the block
  spec for every family; `norm_stats` threaded through `get_dynamics` and the
  per-block builders; `dummy` deleted; a build-time warning for `linear` /
  `ambient` declarations a non-2r2c family ignores (the plan said reportcard
  note; the monolithic nssm path — the one the A/B uses — has no per-block
  descriptor to hang one on).
- `gradient_sampling.py`: sampled `u` goes through the pre-stack outside
  `jac_fn`; mask shapes checked.
- `reportcard.py`: branch lines beside the time constants; derived-inputs list.
- `tie_norm_stats` removed from the CLIs, template and README. Deviations from
  the plan: no `test_reportcard.py` / NSSM golden exists in the repo, so that
  item was skipped rather than invented.

`e75c658` — the three fixes above, `rc_branch_lr`, `"branch": "mlp"`.

### `modelica_rtu_mall` (uncommitted; the app repo is the user's)

- `modeling_user_defined.py`: `BLOCK_SPEC` gains `mixed_demand`,
  `mixed_demand_mlp`, `demand_zone`, built by a `_declared_columns()` helper.
  Untouched but now dead: `tie_norm_stats` (no caller) and the `dummy` paths
  in `data_mapping` / `column_norm_config` (`model: dummy` no longer validates);
  `model_configs/dummy.yaml` and `control_configs/dummy.yaml` will fail to load.
- `model_configs/ps_m01..m05_*.yaml`, `control_configs/ps_c01..c03_*.yaml`.

## Answers to the two questions asked mid-way

- **Demand into the MLP instead of the saturating branch:** yes, now —
  `{"kind": "demand", ..., "branch": "mlp"}`. Measured above as m03.
- **No demand at all, power as the input:** yes. Nothing is inferred from column
  names; a metered-power actuator is declared `linear` with `sign: -1` (or left
  undeclared, to the MLP) and no `demand` entry is written. The fixture in
  `test_a_linear_sign_is_a_constraint_not_a_hint` is exactly that shape.

## Open, out of scope here

- **Plant-in-the-loop scoring.** The "rule on the plant" side of the plan's
  headline metric needs the FMU container (`container-fleet/evaluations`). The
  surrogate-side numbers above are the half that can be run on this machine.
- **`a_o` against solar.** Empirically `τ_ambient ≈ 44 h`; the fit gives 20 h
  with a 3× larger solar coefficient. The two covary on this data (clear cold
  days). A measured `τ_ambient` from a free-decay episode should be *freezable*
  when declared; there is no flag for that yet, and until there is, the
  learned value is what you get.
- **Heating at the slope bound.** `slope_share` 0.97 says the furnace wants a
  sharper step than `1 − a_m − a_o` allows per °C of demand. The plan's
  deferred deadband offset `δ` is the natural next term; the step-response
  readout passes as is, so nothing asked for it yet.
- **DX power fit** regressed (692 → 805 W RMSE) while the fan improved and the
  furnace head started working. The plant block is undeclared; whether it wants
  the raw setpoint back alongside the demand (the plan's deferred "in
  addition" mode) is a question for the power-usage line of work.
- **The equilibrium scenario assumes zero internal gains.** With a free
  constant in the model it fails by construction, which is why fix 3 matters
  for the check as much as for the fit. A dataset with real metered gains
  should declare them.
- **App cleanup.** The dead `dummy` / `tie_norm_stats` code and configs in
  `modelica_rtu_mall`, listed above.
