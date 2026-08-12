# Control Reportcard: snapshot-control

## Summary

- **Run:** `snapshot-control`
- **Timestamp:** 2026-05-08T12:00:00
- **Sys-id run:** `test-run` (model: `mlp`)
- **Best dev loss:** 0.012346
- **Train time:** 42.0 s
- **Eval rollout time:** 0.50 s

## Policy architecture

- **Family:** `mlp`
- **Nonlinearity:** gelu
- **Bound method:** `sigmoid_scale`
- **Hidden sizes:** [64, 64]
- **Trainable params:** 6,017
- **Input width:** 27 (H_ctrl=12, D preview=0, band preview=0)
- **Output width:** 1
- **Setpoint quantization:** 0.5 (straight-through)
- **Actuation bounds (physical units):**
  - `setpoint [°C]`: [22, 28]

## Scenario

- **Horizons:** H_sys=4, H_ctrl=3, train_pred_steps=6, test_window=12
- **Controlled channels (1):** `zone [°C]`
- **Control U (1):** `setpoint [°C]`
- **Train band:** lo ~ U[23.0, 27.0], width ~ U[0.5, 2.0], segments=1
- **Test band:** lo ~ U[23.0, 27.0], width ~ U[0.5, 2.0], segments=4
- **Objective weights:** comfort=50.0, du=10, effort=0.1, cost=0.0, peak=0.0 (q=4)
- **Eval band seed:** 12345 *(fixed, so runs are scored against the same band realization)*

## Objective balance

Each objective term on a single training batch, taken before the first optimizer step. The share column is what the policy is really being asked to minimise, whatever the weights suggest.

| Term | Value | Share | vs. the rest | |
|---|---:|---:|---:|---|
| comfort_lower_loss | 1 | 10% | 0.11× |  |
| comfort_upper_loss | 1.5 | 15% | 0.18× |  |
| du_loss | 0.5 | 5% | 0.05× |  |
| effort_loss | 7 | 70% | 2.33× | ⚠️ dominates |

**⚠️ One term is running the objective:** `effort_loss` at 2.3× everything else combined. Re-derive its weight against the magnitude of the quantity it multiplies. A weight that was right on the last dataset is not evidence about this one.

The power signal swept across each actuation box, with the rest of the rollout held at its recorded values. One sweep serves all three power-derived terms — cost and peak are reductions of this same signal — so this is the magnitude and shape they all multiply, reported so it can be weighed against the terms above. A minimum on an endpoint is not a defect: an honest power metric is monotone in actuator effort, and the trade-off that produces an interior optimum belongs to the summed objective, not to this term alone.

| Control | Term at lo | at mid | at hi | Minimised at |
|---|---:|---:|---:|---:|
| setpoint [°C] | 4 | 2 | 0.5 | 28 |

## Closed-loop performance

Comfort, per controlled channel. Excursions and degree-hours are in the physical units of each channel.

| Channel | Mean exceedance | RMS exceedance | Violation rate | Below | Above | Mean excursion (violating only) | Max excursion | Degree-hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zone [°C] | 0.5000 | 0.7071 | 50.00% | 0.00% | 50.00% | 1.0000 | 1.0000 | 1.0000 |

- **Pooled mean exceedance:** **0.5000** *(mean over all steps — tune on this)*
- **Pooled violation rate:** 50.00% *(acceptance criterion — report it, don't tune on it)*
- **Mean |Δsetpoint|:** 0.07143
- **Energy** *(native units of `power_metric`; energy is power·h)*:
  - Mean power: 1500
  - Energy per window: 3000
  - Peak power (worst instant): 1500
  - Mean per-window peak: 1500

## Actuation

What each control input actually did, in its own physical units. "At bound" counts commands within 1% of the actuation range of a limit, because a smooth bound method never reaches one exactly. A high fraction is a fact about the run, not a fault — a setpoint held at its least-aggressive limit while nothing is demanded of it reads the same way here as one that is stuck.

| Control | Bounds | Mean | Mean \|Δu\| | At lower bound | At upper bound |
|---|---|---:|---:|---:|---:|
| setpoint [°C] | [22, 28] | 22.25 | 0.07143 | 75.0% | 0.0% |

## Band-jump response

Every step change in the reference band that left a controlled channel outside the new band, scored over the segment until the next jump. **Actuated** is whether any control moved on the same step; **moved right way** and **back in band** are whether the channel then followed. The per-control Δ columns name the actuator that did (or did not) do the work.

| Channel | Demand | Jumps | Steps scored | Actuated | Moved right way | Back in band | Mean move | Δ`setpoint [°C]` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| zone [°C] | fall | 2 | 3 | 50% | 0% | 0% | +0 | +0.5 |

## Baseline comparison

Both controllers were rolled out on the same test windows against the same fixed-seed comfort band, so these numbers are directly comparable. Degree-hours is summed across controlled channels. Lower is better in every column, so a negative Δ favours the learned policy.

| Controller | Mean exceedance | Violation rate (%) | Degree-hours | Mean \|Δu\| | Mean power | Peak power |
|---|---:|---:|---:|---:|---:|---:|
| Learned policy | 0.5 | 50 | 1 | 0.07143 | 1500 | 1500 |
| Baseline | 1 | 50 | 2 | 0.07143 | 1000 | 1000 |
| Δ (learned − baseline) | -0.5 | +0 | -1 | +0 | +500 | +500 |

## Tracking error vs. comfort-band midpoint

Error of each controlled channel against the midpoint of its comfort band, in physical units. Tracking the midpoint is not the objective — staying inside the band is — so treat this as a shape diagnostic.

| Channel | RMSE | MAE | R² | NRMSE |
|---|---:|---:|---:|---:|
| zone [°C] | 2.4749 | 1.7500 | -97.0000 | 4.9497 |

## Hyperparameters

- **lr:** 0.001
- **batch_size:** 4
- **epochs:** 1
- **warmup:** 5, patience=25
- **test_stride:** 5
- **seed:** 0
