# Reportcard: snapshot-baseline

## Summary

- **Run:** `snapshot-baseline`
- **Timestamp:** 2026-05-08T12:00:00
- **Best dev loss:** 0.012346
- **Train time:** 42.0 s
- **Test inference time:** 0.50 s

## Architecture

- **Factory:** `nssm`
- **Separable:** True
- **Residual:** True
- **Shapes:** in_steps=12, ny=2, nu=1, nd=1
- **Total trainable params:** 1,234

Blocks:
- **fy:** `MLP`, params=500, hsizes=[32, 32]
- **fu:** `MLP`, params=500, hsizes=[32, 32]
- **fd:** `MLP`, params=234, hsizes=[32, 32]

## Data

- **Split sizes (rows):** train=1,000, dev=200, test=200
- **Y (2 states):**
  - `zone_a [°C]`
  - `zone_b [°C]`
- **U (1 controls):**
  - `setpoint`
- **D (1 disturbances):**
  - `outdoor`

## Test Metrics (per channel)

All values are in the physical units of each channel. NRMSE is normalized by the channel range (max − min) of the test set.

| Channel | RMSE | MAE | R² | NRMSE |
|---|---:|---:|---:|---:|
| zone_a [°C] | 0.1000 | 0.1000 | 0.9988 | 0.0100 |
| zone_b [°C] | 0.1000 | 0.1000 | 0.9988 | 0.0100 |

## Hyperparameters

- **lr:** 0.001
- **batch_size:** 256
- **epochs:** 100
- **train_pred_steps:** 24
- **dev_pred_steps:** 48
- **test_window:** 287
- **curriculum_steps:** 1, ratio=4.0
- **norm_type (default):** min_max *(per-column overrides reported in the Data section)*
- **seed:** 42
- **Loss gains:** ref=1.0
