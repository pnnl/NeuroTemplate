"""
Per-run DPC control reportcard.

The control-side counterpart of modeling/reportcard.py, with the same shape:
one private `_*_section` per block, assembled by a single pure entry point
that returns markdown. The caller writes it to disk and logs it.

Sections (in order):

   1. Summary              — run, timestamp, surrogate, dev loss, timings
   2. Surrogate architecture — the frozen sys-id model's own describe_dynamics
                             output (num_layers, hidden size, RC forcing
                             mappings, ...); reuses modeling/reportcard.py's
                             renderer so the two cards never drift apart
   3. Policy architecture  — from control.policy.describe_policy
   4. Scenario             — horizons, regulated channels, the objective table
   5. Objective balance    — which term was actually being optimised, and the
                             magnitude and shape of the power signal
   6. Closed-loop performance — comfort table, actuation, energy
   7. Actuation            — per-control effort and time at each bound
   8. Band-jump response   — did the loop react when the reference moved
   9. Tracking error       — per-channel error vs. the comfort-band midpoint
  10. Hyperparameters      — the knobs that produced this run

Whether a band is reachable at all is answered on the sys-id side,
by the reachability scenario in modeling/scenarios.py, against a declared
equilibrium rather than whatever weather a test file happened to contain.

Sections 5, 7 and 8 are the behavioural ones. They exist because the comfort
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
from neurotemplate.modeling.reportcard import _architecture_section
from neurotemplate.control.metrics import (
    actuation_to_markdown,
    comfort_to_markdown,
    response_to_markdown,
)
from neurotemplate.control.objective import (
    ObjectiveBalance,
    PowerDiagnostics,
    balance_to_markdown,
)
from neurotemplate.control.losses import Objective, resolve_objective
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


def _box(below: float, above: float) -> str:
    if below == above:
        return f"±{above:g}"
    lo = f"−{below:g}" if below else "0"
    hi = f"+{above:g}" if above else "0"
    return f"[{lo}, {hi}]"


def _mpc_section(arch: dict) -> str:
    rows = [
        "- **Family:** `mpc`: at every step, the plan over the next steps "
        "through a copy of the surrogate, inside the bounds, against the "
        "objective; the first step is commanded",
        f"- **Horizon:** {arch['horizon']} steps, {arch['iters']} Adam steps of "
        f"{arch['lr']:g} per control step, from the last plan shifted",
        "- **Inputs:** " + ", ".join(f"`{key}`" for key in arch["inputs"]),
    ]
    if arch.get("command_bounds"):
        rows.append("- **Per-step bounds:**")
        rows += [f"  - `{col}`: [{low}, {high}]"
                 for col, (low, high) in arch["command_bounds"].items()]
    rows.append("- **Actuation bounds (physical units):**")
    rows += [f"  - `{col}`: [{lo:.4g}, {hi:.4g}]"
             for col, lo, hi in zip(arch["u_cols"], arch["umin_phys"], arch["umax_phys"])]
    return _section("Controller: MPC", "\n".join(rows))


def _policy_section(arch: dict) -> str:
    if arch["family"] == "mpc":
        return _mpc_section(arch)
    hsizes = arch["hsizes"] if arch["hsizes"] else "*(none — single affine map)*"
    bounds = "\n".join(
        f"  - `{col}`: [{lo:.4g}, {hi:.4g}]"
        for col, lo, hi in zip(
            arch["u_cols"], arch["umin_phys"], arch["umax_phys"]
        )
    )
    rows = [
        f"- **Family:** `{arch['family']}`",
        f"- **Nonlinearity:** {arch['nonlin']}",
        f"- **Bound method:** `{arch['bound_method']}`",
        f"- **Hidden sizes:** {hsizes}",
        f"- **Trainable params:** {arch['param_count']:,}",
        f"- **Input width:** {arch['insize']}, over "
        + ", ".join(f"`{key}`" for key in arch["inputs"]),
        "- **Window (steps before / after the current one; a U column's "
        "window ends at the last command applied):**",
        *(f"  - `{col}`: past {entry['past']}"
          + (f", future {entry['future']}" if "future" in entry else "")
          for col, entry in arch["window"].items()),
        f"- **Output width:** {arch['outsize']}",
        f"- **Observation:** Policy observes all {arch['ny']} Y channels — "
        "no measurement restriction.",
    ]
    if arch["residual_box"] is not None:
        rows.append(
            "- **Residual boxes** around `reference_control` *(an anchored "
            "column commands the deviation, not the setpoint; the box bounds "
            "the action, not the zone temperature; a column not listed is a "
            "free output over its actuation bounds)*:"
        )
        rows += [f"  - `{col}`: {_box(*box)}"
                 for col, box in arch["residual_box"].items()]
    if arch.get("command_bounds"):
        rows.append("- **Per-step bounds** *(the actuation box mapped onto [low, high] "
                    "at each step; a name is a reference column)*:")
        rows += [f"  - `{col}`: [{low}, {high}]"
                 for col, (low, high) in arch["command_bounds"].items()]
    rows += [
        "- **Actuation bounds (physical units):**",
        bounds,
    ]
    return _section("Policy architecture", "\n".join(rows))


def _objective_table(objective: Objective) -> str:
    rows = ["| Term | Operand | Map | Norm | Weight | Gain |",
            "|---|---|---|---:|---|---:|"]
    for t in objective.terms:
        operand = f"`{t.column}`" + ("" if t.target is None else f" − `{t.target}`")
        rows.append(
            f"| {t.name} | {operand} | {t.map or '—'} | {t.norm:g} | "
            f"{'—' if t.weight is None else f'`{t.weight}`'} | {t.gain:g} |"
        )
    return "\n".join(rows)


def _scenario_section(cfg, sysid_cfg, nm_mapping: dict,
                      objective: Objective) -> str:
    h_sys, h_ctrl, _ = horizons(cfg, sysid_cfg)
    rows = [
        f"- **Horizons:** H_sys={h_sys}, H_ctrl={h_ctrl}, "
        f"train_pred_steps={cfg.train_pred_steps}, "
        f"test_window={cfg.test_window}",
        f"- **Regulated channels ({len(objective.regulated)}):** "
        + ", ".join(f"`{c}`" for c in objective.regulated)
        + " *(the Y targets of the bound terms)*",
        f"- **Control U ({len(nm_mapping['U'])}):** "
        + ", ".join(f"`{c}`" for c in nm_mapping["U"]),
        "- **Objective:** each term is `gain · power-mean(map(operand))` per "
        "window, averaged over the batch, the operand in physical units.\n",
        _objective_table(objective),
    ]
    return _section("Scenario", "\n".join(rows))


def _objective_section(balance: ObjectiveBalance,
                       power: Optional[PowerDiagnostics] = None,
                       notes: Optional[list[str]] = None,
                       trained: Optional[ObjectiveBalance] = None) -> str:
    """What the weights actually bought, measured on one batch before training
    and again through the trained policy.

    Objective weights do not port across datasets, and the library's job is
    not a better default — it is making a bad one visible on iteration 1.
    The trained shares are what a weight can be tuned against: a residual
    policy starts at δ ≡ 0, so its Δu term is exactly zero before training.
    """
    body = [
        "Each objective term on a single training batch, taken before the "
        "first optimizer step"
        + (" and again through the trained policy" if trained is not None else "")
        + ". The share column is what the policy is really "
        "being asked to minimise, whatever the weights suggest.\n",
        balance_to_markdown(balance, trained),
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
            "clamp on the power signal's path passing no gradient: a power "
            "head predicting negative power, or a `relu`/`clamp_min` in "
            "`power_metric` itself. That is harmless for a mean energy term "
            "and not harmless for a cost term, because those steps are "
            "disproportionately the cheap hours load is supposed to move "
            "*into*. A large dead fraction is fixed where the clamp is, not "
            "with a weight; a proxy's clamp keeps its zero and passes "
            "gradient with `soft_clamp_min` (`neurotemplate.modeling.prestack`).",
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
    for name, value in energy.weighted_energy.items():
        lines.append(
            f"  - ∫`{name}`·power dt per window: {value:.4g} *(unconverted — "
            "Watts against $/kWh is off by 1000)*"
        )
    return lines


def _performance_section(results) -> str:
    body = [
        "Comfort, per regulated channel. Excursions and degree-hours are in "
        "the physical units of each channel.\n",
        comfort_to_markdown(results.comfort),
        "",
        f"- **Pooled mean exceedance:** "
        f"**{results.comfort_mean_exceedance:.4f}** *(mean over all steps — "
        "tune on this)*",
        f"- **Pooled violation rate:** "
        f"{results.comfort_violation_rate * 100:.2f}% "
        "*(acceptance criterion — report it, don't tune on it)*",
        f"- **Mean |ΔU|** (all U columns, physical units): {results.mean_abs_du:.4g}",
    ]
    body += _energy_lines(results.energy)
    return _section("Closed-loop performance", "\n".join(body))


def _gap_section(plan, results) -> str:
    """The policy beside the best open-loop plan on the same windows
    (`control.plan`), term by term."""
    names = [term.name for term in plan.policy.terms]
    by_name = [{term.name: term.value for term in balance.terms}
               for balance in (plan.policy, plan.plan)]
    cost = {} if results.energy is None else results.energy.weighted_energy
    plan_cost = {} if plan.results.energy is None else plan.results.energy.weighted_energy
    head = (["", "Objective"] + [n.removesuffix("_loss") for n in names]
            + ["Violation", "Mean exceedance"]
            + [f"∫`{w}`·power" for w in cost])
    rows = []
    for label, values, scored, weighted in (("Controller", by_name[0], results, cost),
                                            ("Best plan", by_name[1], plan.results, plan_cost)):
        total = sum(values.values())
        rows.append([label, f"{total:.4g}"] + [f"{values[n]:.4g}" for n in names]
                    + [f"{scored.comfort_violation_rate * 100:.2f}%",
                       f"{scored.comfort_mean_exceedance:.4g}"]
                    + [f"{weighted[w]:.4g}" for w in cost])
    body = [
        f"The best plan is a free command per step through the same surrogate, "
        f"bounds and objective, over these test windows, with every step ahead "
        f"known ({plan.iterations} Adam steps, {plan.time_s:.0f} s): a bar for "
        f"any controller on this surrogate, not a controller. Its objective is "
        f"**{plan.gap * 100:.0f}%** below the controller's. That gap is the "
        "controller's; what the plan does not do either, the surrogate does not "
        "make pay.\n",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|",
    ] + ["| " + " | ".join(row) + " |" for row in rows]
    return _section("Controller gap", "\n".join(body))


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
    for r in results.residual:
        body += [
            "",
            f"- **Residual of `{r.name}` vs. the reference rule** (box "
            f"{_box(r.below_cap, r.above_cap)}):",
            f"  - Mean |δ|: {r.mean_abs_delta:.4g}   max |δ|: "
            f"{r.max_abs_delta:.4g}",
        ]
        if r.frac_at_below is None and r.frac_at_above is None:
            body += [
                "  - Pinned at an edge: n/a — the box is zero-width, so this "
                "column *is* the reference rule and the box has no interior "
                "to bind. Mean |δ| above is the check that the residual path "
                "is inert; anything but zero there is a bug.",
            ]
        else:
            body += [
                f"  - Pinned at the {name} edge on {frac * 100:.1f}% of steps"
                for name, frac in (("lower", r.frac_at_below),
                                   ("upper", r.frac_at_above))
                if frac is not None
            ]
    if any(r.frac_at_below is not None or r.frac_at_above is not None
           for r in results.residual):
        body += [
            "",
            "An edge that binds most of the time means one of three things: "
            "the reference is wrong for this plant, the optimizer is riding a "
            "surrogate artifact into the wall, or the incentive has simply "
            "outgrown what the box can express. The third is the box doing its "
            "job. Read this before the comfort number either way. Each δ is "
            "the *realized* deviation, measured after the actuator clamp.",
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
            "*(no band jumps left a regulated channel outside its band, so "
            "there was nothing to audit — a test reference file whose bounds "
            "step is what exercises the controller.)*",
        )
    body = [
        "Every step change in a band that left a regulated "
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


def _tracking_section(results) -> str:
    body = (
        "Error of each regulated channel with a two-sided band against the "
        "band's midpoint, in physical units. Tracking the midpoint is not the "
        "objective — staying inside the band is — so treat this as a shape "
        "diagnostic.\n\n"
        + metrics_to_markdown(results.tracking)
    )
    return _section("Tracking error vs. band midpoint", body)


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
    results,
    *,
    surrogate_arch: Optional[dict] = None,
    balance: Optional[ObjectiveBalance] = None,
    trained_balance: Optional[ObjectiveBalance] = None,
    power: Optional[PowerDiagnostics] = None,
    objective_notes: Optional[list[str]] = None,
    plan=None,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    best_dev_loss: Optional[float] = None,
    train_time_s: Optional[float] = None,
) -> str:
    """
    Render the markdown reportcard for one DPC run.

    Args:
        arch:        Output of control.policy.describe_policy(...).
        results:     ControlResults for the learned policy.
        surrogate_arch: Optional output of modeling.dynamics.describe_dynamics
                     for the frozen sys-id model this policy was trained
                     against (num_layers, hidden size, RC forcing maps, ...).
                     None renders no extra section.
        balance:     Optional objective.ObjectiveBalance from the pre-training
                     forward pass; the section is skipped without it.
        trained_balance: Optional ObjectiveBalance of the same batch through
                     the trained policy, shown beside `balance`.
        power:       Optional objective.power_diagnostics(...): how often the
                     power signal reads zero, and how much of that is a dead
                     gradient.
        objective_notes: Optional objective.objective_notes(...) prose.
        plan:        Optional plan.PlanResult, the best open-loop plan on the
                     same windows; None renders no gap section.

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
    ]
    if surrogate_arch is not None:
        parts.append(_architecture_section(surrogate_arch, "Surrogate architecture"))
    parts += [
        _policy_section(arch),
        _scenario_section(cfg, sysid_cfg, nm_mapping,
                          resolve_objective(cfg, nm_mapping)),
    ]
    if balance is not None:
        parts.append(_objective_section(balance, power, objective_notes, trained_balance))
    parts.append(_performance_section(results))
    if plan is not None:
        parts.append(_gap_section(plan, results))
    parts += [
        _actuation_section(results),
        _response_section(results, nm_mapping["U"]),
        _tracking_section(results),
        _hyperparameters_section(cfg),
    ]
    return "\n".join(parts)
