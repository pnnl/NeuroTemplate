"""
Per-run DPC control reportcard.

The control-side counterpart of modeling/reportcard.py, with the same shape:
one private `_*_section` per block, assembled by a single pure entry point
that returns markdown. The caller writes it to disk and logs it.

Sections (in order):

  1. Summary               — run, timestamp, surrogate, dev loss, timings
  2. Policy architecture   — from control.policy.describe_policy
  3. Scenario              — horizons, controlled channels, bands, weights
  4. Closed-loop performance — comfort table, actuation, energy
  5. Actuation             — per-control effort and saturation fraction
  6. Band-jump response    — did the loop react when the reference moved
  7. Baseline comparison   — learned vs. rule-based, or why there is none
  8. Tracking error        — per-channel error vs. the comfort-band midpoint
  9. Hyperparameters       — the knobs that produced this run
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from neurotemplate.common.metrics import metrics_to_markdown
from neurotemplate.common.reportcard import _fmt_float, _section
from neurotemplate.control.metrics import (
    SATURATION_WARN,
    actuation_to_markdown,
    comfort_to_markdown,
    response_to_markdown,
)
from neurotemplate.control.scenarios import horizons

from neurotemplate._typing import beartype


def _summary_section(
    run_name: str,
    timestamp: str,
    cfg,
    sysid_cfg,
    eval_time_s: float,
    best_dev_loss: Optional[float],
    train_time_s: Optional[float],
) -> str:
    rows = [
        f"- **Run:** `{run_name}`",
        f"- **Timestamp:** {timestamp}",
        f"- **Sys-id run:** `{cfg.sysid_run_id}` (model: `{sysid_cfg.model}`)",
        f"- **Best dev loss:** {_fmt_float(best_dev_loss)}",
    ]
    if train_time_s is not None:
        rows.append(f"- **Train time:** {train_time_s:.1f} s")
    rows.append(f"- **Eval rollout time:** {eval_time_s:.2f} s")
    return _section("Summary", "\n".join(rows))


def _policy_section(arch: dict) -> str:
    hsizes = arch["hsizes"] if arch["hsizes"] else "*(none — single affine map)*"
    bounds = "\n".join(
        f"  - `{col}`: [{lo:.4g}, {hi:.4g}]"
        for col, lo, hi in zip(
            arch["u_cols"], arch["umin_phys"], arch["umax_phys"]
        )
    )
    quant = (
        f"{arch['setpoint_step']:g} (straight-through)" if arch["quantized"]
        else "off"
    )
    rows = [
        f"- **Family:** `{arch['family']}`",
        f"- **Nonlinearity:** {arch['nonlin']}",
        f"- **Bound method:** `{arch['bound_method']}`",
        f"- **Hidden sizes:** {hsizes}",
        f"- **Trainable params:** {arch['param_count']:,}",
        f"- **Input width:** {arch['insize']} "
        f"(H_ctrl={arch['h_ctrl']}, D preview={arch['d_preview']}, "
        f"band preview={arch['band_preview']})",
        f"- **Output width:** {arch['outsize']}",
        f"- **Setpoint quantization:** {quant}",
        "- **Actuation bounds (physical units):**",
        bounds,
    ]
    return _section("Policy architecture", "\n".join(rows))


def _band_line(label: str, band) -> str:
    return (
        f"- **{label}:** lo ~ U[{band.lo}, {band.hi}], "
        f"width ~ U[{band.width_min}, {band.width_max}], "
        f"segments={band.sub_samples}"
    )


def _scenario_section(cfg, sysid_cfg, nm_mapping: dict, controlled: list[str],
                      eval_band_seed: int) -> str:
    h_sys, h_ctrl, _ = horizons(cfg, sysid_cfg)
    rows = [
        f"- **Horizons:** H_sys={h_sys}, H_ctrl={h_ctrl}, "
        f"train_pred_steps={cfg.train_pred_steps}, "
        f"test_window={cfg.test_window}",
        f"- **Controlled channels ({len(controlled)}):** "
        + ", ".join(f"`{c}`" for c in controlled),
        f"- **Control U ({len(nm_mapping['U'])}):** "
        + ", ".join(f"`{c}`" for c in nm_mapping["U"]),
        _band_line("Train band", cfg.train_band()),
        _band_line("Test band", cfg.test_band()),
        f"- **Objective weights:** comfort={cfg.comfort_weight}, "
        f"du={cfg.du_weight}, effort={cfg.effort_weight}",
        f"- **Eval band seed:** {eval_band_seed} "
        "*(fixed, so runs are scored against the same band realization)*",
    ]
    return _section("Scenario", "\n".join(rows))


def _energy_lines(energy) -> list[str]:
    if energy is None:
        return ["- **Energy:** n/a *(power_metric not implemented)*"]
    return [
        "- **Energy** *(native units of `power_metric`; energy is power·h)*:",
        f"  - Mean power: {energy.mean_power:.4g}",
        f"  - Energy per window: {energy.total_energy:.4g}",
        f"  - Peak power (worst instant): {energy.peak_power:.4g}",
        f"  - Mean per-window peak: {energy.mean_window_peak:.4g}",
    ]


def _performance_section(results) -> str:
    body = [
        "Comfort, per controlled channel. Excursions and degree-hours are in "
        "the physical units of each channel.\n",
        comfort_to_markdown(results.comfort),
        "",
        f"- **Pooled mean exceedance:** "
        f"**{results.comfort_mean_exceedance:.4f}** *(mean over all steps — "
        "tune on this)*",
        f"- **Pooled violation rate:** "
        f"{results.comfort_violation_rate * 100:.2f}% "
        "*(acceptance criterion — report it, don't tune on it)*",
        f"- **Mean |Δsetpoint|:** {results.mean_abs_du:.4g}",
    ]
    body += _energy_lines(results.energy)
    return _section("Closed-loop performance", "\n".join(body))


def _actuation_section(results) -> str:
    """Per-control effort and saturation — the cheapest silent-failure check.

    A pooled mean |Δu| hides an actuator that never moves, and a comfort score
    hides one that sits on a bound. Both are one column each here.
    """
    pinned = [m for m in results.actuation
              if m.frac_saturated > SATURATION_WARN]
    body = [
        "What each control input actually did, in its own physical units. "
        "\"At bound\" counts commands within 1% of the actuation range of a "
        "limit, because a smooth bound method never reaches one exactly.\n",
        actuation_to_markdown(results.actuation),
    ]
    if pinned:
        body += [
            "",
            "**⚠️ Pinned at a bound:** "
            + ", ".join(
                f"`{m.name}` ({m.frac_saturated * 100:.0f}% of the evaluation)"
                for m in pinned
            )
            + ". A control held against its limit is not being modulated — the "
            "closed loop is running on whatever is left. This is usually an "
            "objective term whose gradient points one way across the whole "
            "feasible set, not a policy that chose to saturate.",
        ]
    return _section("Actuation", "\n".join(body))


def _response_section(results, u_names: list[str]) -> str:
    """Did the loop answer when the reference band moved?

    The comparison that motivates this section: two controllers, 16.6% vs
    17.9% pooled violation, and the *better*-scoring one reacted to 0% of the
    band jumps that demanded cooling while the worse one reacted to 75%. The
    pooled metric ranked them backwards.
    """
    if not results.response:
        return _section(
            "Band-jump response",
            "*(no band jumps left a controlled channel outside its band, so "
            "there was nothing to audit — raise `test_band_sub_samples` or "
            "widen the band range to exercise the controller.)*",
        )
    body = [
        "Every step change in the reference band that left a controlled "
        "channel outside the new band, scored over the segment until the next "
        "jump. **Actuated** is whether any control moved on the same step; "
        "**moved right way** and **back in band** are whether the channel "
        "then followed. The per-control Δ columns name the actuator that did "
        "(or did not) do the work.\n",
        response_to_markdown(results.response, u_names),
    ]
    ignored = [r for r in results.response if r.frac_actuated < 0.5]
    if ignored:
        body += [
            "",
            "**⚠️ Requests largely ignored:** "
            + ", ".join(f"`{r.name}` / {r.direction} "
                        f"({r.frac_actuated * 100:.0f}% actuated)"
                        for r in ignored)
            + ". The band asked the loop to move the channel and the controls "
            "mostly stayed put. Check the Δ columns for which actuator is "
            "inert, then the Actuation section for whether it is pinned.",
        ]
    return _section("Band-jump response", "\n".join(body))


def _comparison_values(results) -> list[Optional[float]]:
    """The six headline numbers, in table order. None renders as n/a."""
    return [
        results.comfort_mean_exceedance,
        results.comfort_violation_rate * 100.0,
        sum(m.degree_hours for m in results.comfort),
        results.mean_abs_du,
        None if results.energy is None else results.energy.mean_power,
        None if results.energy is None else results.energy.peak_power,
    ]


def _comparison_row(label: str, values: list[Optional[float]],
                    signed: bool = False) -> str:
    fmt = "+.4g" if signed else ".4g"
    cells = ["n/a" if v is None else f"{v:{fmt}}" for v in values]
    return f"| {label} | " + " | ".join(cells) + " |"


def _baseline_section(results, baseline) -> str:
    """Learned vs. rule-based, scored on the identical band realization."""
    if baseline is None:
        return _section(
            "Baseline comparison",
            "*(none — `control_user_defined.build_baseline_policy` is not "
            "defined, so there is no rule-based controller to compare "
            "against.)*",
        )

    learned = _comparison_values(results)
    rule = _comparison_values(baseline)
    delta = [
        None if (a is None or b is None) else a - b
        for a, b in zip(learned, rule)
    ]

    body = [
        "Both controllers were rolled out on the same test windows against "
        "the same fixed-seed comfort band, so these numbers are directly "
        "comparable. Degree-hours is summed across controlled channels. "
        "Lower is better in every column, so a negative Δ favours the "
        "learned policy.\n",
        r"| Controller | Mean exceedance | Violation rate (%) | Degree-hours "
        r"| Mean \|Δu\| | Mean power | Peak power |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _comparison_row("Learned policy", learned),
        _comparison_row("Baseline", rule),
        _comparison_row("Δ (learned − baseline)", delta, signed=True),
    ]
    return _section("Baseline comparison", "\n".join(body))


def _tracking_section(results) -> str:
    body = (
        "Error of each controlled channel against the midpoint of its comfort "
        "band, in physical units. Tracking the midpoint is not the objective — "
        "staying inside the band is — so treat this as a shape diagnostic.\n\n"
        + metrics_to_markdown(results.tracking)
    )
    return _section("Tracking error vs. comfort-band midpoint", body)


def _hyperparameters_section(cfg) -> str:
    rows = [
        f"- **lr:** {cfg.lr}",
        f"- **batch_size:** {cfg.batch_size}",
        f"- **epochs:** {cfg.epochs}",
        f"- **warmup:** {cfg.warmup}, patience={cfg.patience}",
        f"- **test_stride:** {cfg.test_stride}",
        f"- **seed:** {cfg.seed}",
    ]
    return _section("Hyperparameters", "\n".join(rows))


@beartype
def render_control_reportcard(
    cfg,
    sysid_cfg,
    arch: dict,
    nm_mapping: dict[str, list[str]],
    controlled: list[str],
    results,
    *,
    eval_band_seed: int,
    baseline: Optional[object] = None,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    best_dev_loss: Optional[float] = None,
    train_time_s: Optional[float] = None,
) -> str:
    """
    Render the markdown reportcard for one DPC run.

    Args:
        arch:        Output of control.policy.describe_policy(...).
        controlled:  Controlled channel names (hooks.controlled_channels).
        results:     ControlResults for the learned policy.
        eval_band_seed: The fixed seed the evaluation band was drawn with.
        baseline:    Optional ControlResults for the rule-based controller.

    Returns:
        Markdown string. The caller writes it to disk and logs it.
    """
    name = run_name if run_name is not None else cfg.run
    ts = timestamp if timestamp is not None else datetime.now().isoformat(
        timespec="seconds"
    )

    parts = [
        f"# Control Reportcard: {name}\n",
        _summary_section(name, ts, cfg, sysid_cfg, results.eval_time_s,
                         best_dev_loss, train_time_s),
        _policy_section(arch),
        _scenario_section(cfg, sysid_cfg, nm_mapping, controlled,
                          eval_band_seed),
        _performance_section(results),
        _actuation_section(results),
        _response_section(results, nm_mapping["U"]),
        _baseline_section(results, baseline),
        _tracking_section(results),
        _hyperparameters_section(cfg),
    ]
    return "\n".join(parts)
