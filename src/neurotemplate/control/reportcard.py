"""
Per-run DPC control reportcard.

The control-side counterpart of modeling/reportcard.py, with the same shape:
one private `_*_section` per block, assembled by a single pure entry point
that returns markdown. The caller writes it to disk and logs it.

Sections (in order):

   1. Summary              — run, timestamp, surrogate, dev loss, timings
   2. Policy architecture  — from control.policy.describe_policy
   3. Scenario             — horizons, controlled channels, bands, weights
   4. Objective balance    — which term was actually being optimised, and the
                             magnitude and shape of the power signal
   5. Closed-loop performance — comfort table, actuation, energy
   6. Actuation            — per-control effort and time at each bound
   7. Band-jump response   — did the loop react when the reference moved
   8. Baseline comparison  — learned vs. rule-based, or why there is none
   9. Tracking error       — per-channel error vs. the comfort-band midpoint
  10. Hyperparameters      — the knobs that produced this run

Whether the configured band is reachable at all is answered on the sys-id side,
by the reachability scenario in modeling/scenarios.py, against a declared
equilibrium rather than whatever weather a test file happened to contain.

Sections 4, 6 and 7 are the behavioural ones. They exist because the comfort
table above them is silent about dominated objectives, what the actuators did,
and ignored setpoint changes — every one of which has been shipped in a run
whose pooled numbers looked fine.

Only two of them assert anything: a dominated objective term and an ignored
band jump. Actuation and power-term shape are reported without a verdict,
because bound-sitting and monotone power terms are both normal in a correct
controller.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from neurotemplate.common.metrics import metrics_to_markdown
from neurotemplate.common.reportcard import _fmt_float, _section
from neurotemplate.control.metrics import (
    actuation_to_markdown,
    comfort_to_markdown,
    response_to_markdown,
)
from neurotemplate.control.objective import (
    ObjectiveBalance,
    PowerDiagnostics,
    TermShape,
    balance_to_markdown,
    is_inert,
    shapes_to_markdown,
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
        f"band preview={arch['band_preview']}"
        + (f", price preview={arch['price_preview']}" if arch["reads_price"]
           else "")
        + ")",
        f"- **Output width:** {arch['outsize']}",
        f"- **Observation:** Policy observes all {arch['ny']} Y channels — "
        "no measurement restriction.",
        f"- **Setpoint quantization:** {quant}",
    ]
    if arch["residual_delta"] is not None:
        rows.append(
            f"- **Residual box:** ±{arch['residual_delta']:g} around "
            "`reference_control` *(the network commands the deviation, not "
            "the setpoint; Δ bounds the action, not the zone temperature)*"
        )
    rows += [
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
        f"du={cfg.du_weight}, energy={cfg.energy_weight}, "
        f"cost={cfg.cost_weight}, peak={cfg.peak_weight} (q={cfg.peak_q:g})",
        f"- **Eval band seed:** {eval_band_seed} "
        "*(fixed, so runs are scored against the same band realization)*",
    ]
    return _section("Scenario", "\n".join(rows))


def _objective_section(balance: ObjectiveBalance,
                       shapes: list[TermShape],
                       power: Optional[PowerDiagnostics] = None,
                       notes: Optional[list[str]] = None) -> str:
    """What the weights actually bought, measured on one batch before training.

    Objective weights do not port across datasets, and the library's job is
    not a better default — it is making a bad one visible on iteration 1.
    """
    body = [
        "Each objective term on a single training batch, taken before the "
        "first optimizer step. The share column is what the policy is really "
        "being asked to minimise, whatever the weights suggest.\n",
        balance_to_markdown(balance),
    ]
    dominant = balance.dominant()
    if dominant:
        body += [
            "",
            "**⚠️ One term is running the objective:** "
            + ", ".join(
                f"`{t.name}` at {t.ratio_to_rest:.1f}× everything else combined"
                for t in dominant
            )
            + ". Re-derive its weight against the magnitude of the quantity "
            "it multiplies. A weight that was right on the last dataset is "
            "not evidence about this one.",
        ]
    if shapes:
        body += [
            "",
            "The power signal swept across each actuation box, with the rest "
            "of the rollout held at its recorded values. One sweep serves all "
            "three power-derived terms — cost and peak are reductions of this "
            "same signal — so this is the magnitude and shape they all "
            "multiply, reported so it can be weighed against the terms above. "
            "A minimum on an endpoint is not a defect: an honest power metric "
            "is monotone in actuator effort, and the trade-off that produces "
            "an interior optimum belongs to the summed objective, not to this "
            "term alone.\n",
            shapes_to_markdown(shapes,
                               None if power is None else power.reads_outputs),
        ]
    if is_inert(shapes, power):
        body += [
            "",
            "**⚠️ The power term reads neither the commands nor the "
            "surrogate's outputs.** Its sweep is flat because nothing it "
            "depends on is being varied, and its gradient w.r.t. the rollout "
            "is identically zero — so whatever weight it carries, it cannot "
            "move the policy. Check `power_metric` against `nm_mapping`.",
        ]
    if power is not None:
        body += [
            "",
            f"- **Power exactly zero on {power.frac_zero * 100:.1f}% of "
            f"scored steps, of which the clamp kills the gradient on "
            f"{power.frac_dead * 100:.1f}%** (of all scored steps). The "
            "second number is the one to act on. A zero power reading is not "
            "itself a defect — an ordinal or staged head predicting "
            "\"compressor off\" is being right, and a straight-through "
            "estimator still passes gradient there. A *dead* step is a "
            "surrogate predicting negative power and getting clamped, which "
            "is harmless for a mean energy term and not harmless for a cost "
            "term, because those steps are disproportionately the cheap hours "
            "load is supposed to move *into*. A large dead fraction is a "
            "surrogate problem, not a weight.",
        ]
    if notes:
        body += [""] + [f"- {n}" for n in notes]
    return _section("Objective balance", "\n".join(body))


def _energy_lines(energy) -> list[str]:
    if energy is None:
        return ["- **Energy:** n/a *(power_metric not implemented)*"]
    lines = [
        "- **Energy** *(native units of `power_metric`; energy is power·h)*:",
        f"  - Mean power: {energy.mean_power:.4g}",
        f"  - Energy per window: {energy.total_energy:.4g}",
        f"  - Peak power (worst instant): {energy.peak_power:.4g}",
        f"  - Mean per-window peak: {energy.mean_window_peak:.4g}",
    ]
    if energy.energy_cost is not None:
        lines.append(
            f"  - Cost per window: {energy.energy_cost:.4g} "
            "*(∫price·power dt; price·power·h, unconverted — Watts against "
            "$/kWh is off by 1000)*"
        )
    return lines


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
    """Per-control effort and time-at-bound, as measured.

    A pooled mean |Δu| hides an actuator that never moves, so these are broken
    out per control. No verdict is attached to a high bound fraction: sitting a
    cooling setpoint at its least-aggressive limit through a period that needs
    no cooling is correct, conservative control, not a stuck actuator. Read it
    together with the band-jump response below, which is what distinguishes
    "parked because nothing was asked of it" from "did not answer".
    """
    body = [
        "What each control input actually did, in its own physical units. "
        "\"At bound\" counts commands within 1% of the actuation range of a "
        "limit, because a smooth bound method never reaches one exactly. A "
        "high fraction is a fact about the run, not a fault — a setpoint held "
        "at its least-aggressive limit while nothing is demanded of it reads "
        "the same way here as one that is stuck.\n",
        actuation_to_markdown(results.actuation),
    ]
    r = results.residual
    if r is not None:
        body += [
            "",
            f"- **Residual vs. the reference rule** (Δ = ±{r.delta_cap:g}):",
            f"  - Mean |δ|: {r.mean_abs_delta:.4g}   max |δ|: "
            f"{r.max_abs_delta:.4g}",
        ]
        if r.frac_at_cap is None:
            body += [
                "  - Pinned at ±Δ: n/a — Δ = 0, so the policy *is* the "
                "reference rule and the box has no interior to bind. Mean |δ| "
                "above is the check that the residual path is inert; anything "
                "but zero there is a bug.",
            ]
        else:
            body += [
                f"  - Pinned at ±Δ on {r.frac_at_cap * 100:.1f}% of steps",
                "  - A cap that binds most of the time means one of three "
                "things: the reference is wrong for this plant, the optimizer "
                "is riding a surrogate artifact into the wall, or the "
                "incentive has simply outgrown what ±Δ can express. The third "
                "is the cap doing its job. Read this before the comfort "
                "number either way.",
            ]
        body += [
            "  - This is the *realized* deviation, measured after quantization "
            "and the actuator clamp. With `setpoint_step` on and a reference "
            "that is not itself grid-aligned, mean |δ| has a floor of half a "
            "grid step that is snapping, not steering.",
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
    """The seven headline numbers, in table order. None renders as n/a."""
    energy = results.energy
    return [
        results.comfort_mean_exceedance,
        results.comfort_violation_rate * 100.0,
        sum(m.degree_hours for m in results.comfort),
        results.mean_abs_du,
        None if energy is None else energy.mean_power,
        None if energy is None else energy.peak_power,
        None if energy is None else energy.energy_cost,
    ]


def _comparison_row(label: str, values: list[Optional[float]],
                    signed: bool = False) -> str:
    fmt = "+.4g" if signed else ".4g"
    cells = ["n/a" if v is None else f"{v:{fmt}}" for v in values]
    return f"| {label} | " + " | ".join(cells) + " |"


def _baseline_section(results, baseline, clears: Optional[bool] = None) -> str:
    """Learned vs. rule-based, scored on the identical band realization."""
    if baseline is None:
        return _section(
            "Baseline comparison",
            "*(none — neither `control_user_defined.reference_control` nor "
            "`build_baseline_policy` is defined, so there is no rule-based "
            "controller to compare against.)*",
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
        r"| Mean \|Δu\| | Mean power | Peak power | Cost per window |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _comparison_row("Learned policy", learned),
        _comparison_row("Baseline", rule),
        _comparison_row("Δ (learned − baseline)", delta, signed=True),
    ]
    if clears is not None:
        verdict = (
            "**✅ The learned policy clears its reference** on mean "
            "exceedance."
            if clears else
            "**⚠️ The learned policy does not clear its reference** on mean "
            "exceedance. **Ship the reference.** Training is scored against a "
            "surrogate whose held-out error is not the plant's, and the rule "
            "is the one controller here whose behaviour does not depend on "
            "that. Nothing about a longer run changes that arithmetic."
        )
        body += [
            "",
            verdict + " Judged on mean exceedance rather than violation rate: "
            "a threshold count saturates, and measured against a surrogate its "
            "argmin can select a substantially worse controller.",
            "",
            "**The bar this verdict clears moves with the surrogate.** The "
            "reference is rolled out against the same frozen model as the "
            "policy, so the rule's score is a property of that model's "
            "actuation realism and not of the rule. A smooth power head sells "
            "the rule fractional cooling that a staged plant cannot deliver, "
            "and flatters it; a staged head does not. The same fixed rule has "
            "been measured 28 points apart on two surrogates whose zone NRMSE "
            "differed by 3%. Re-read this section after any change to the "
            "surrogate's readout heads.",
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
    clears_reference: Optional[bool] = None,
    balance: Optional[ObjectiveBalance] = None,
    shapes: Optional[list[TermShape]] = None,
    power: Optional[PowerDiagnostics] = None,
    objective_notes: Optional[list[str]] = None,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    best_dev_loss: Optional[float] = None,
    train_time_s: Optional[float] = None,
) -> str:
    """
    Render the markdown reportcard for one DPC run.

    Args:
        arch:        Output of control.policy.describe_policy(...).
        controlled:  Regulated channel names (the `regulated` hook, or the
                     older `controlled_channels` name it falls back to).
        results:     ControlResults for the learned policy.
        eval_band_seed: The fixed seed the evaluation band was drawn with.
        baseline:    Optional ControlResults for the rule-based controller.
        clears_reference: Optional pipeline.clears_reference(...) verdict; the
                     baseline section renders it as a ship/don't-ship call.
        balance:     Optional objective.ObjectiveBalance from the pre-training
                     forward pass; the section is skipped without it.
        shapes:      Optional objective.term_shapes(...) for the power signal.
        power:       Optional objective.power_diagnostics(...): how often the
                     power signal reads zero, how much of that is a dead
                     gradient, and whether the term reads the rollout at all.
        objective_notes: Optional objective.objective_notes(...) prose.

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
    ]
    if balance is not None:
        parts.append(_objective_section(balance, shapes or [],
                                        power, objective_notes))
    parts += [
        _performance_section(results),
        _actuation_section(results),
        _response_section(results, nm_mapping["U"]),
        _baseline_section(results, baseline, clears_reference),
        _tracking_section(results),
        _hyperparameters_section(cfg),
    ]
    return "\n".join(parts)
