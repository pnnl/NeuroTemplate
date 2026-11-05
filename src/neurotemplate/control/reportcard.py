"""
Per-run control reportcard: the verdicts on one controller, short enough to read.

The control-side counterpart of modeling/reportcard.py, on the same rules:
exceptions rather than inventories, each check explained once, the full
output of every instrument in the run's `results/` folder.

Sections (in order):

  Verdict      — comfort, band-jump response, and a warning for each check
                 that fired (a dominated objective, a dead power gradient, a
                 command the surrogate ignores, MPC solves at the cap)
  Controller   — family, size, window and bounds, identical entries grouped;
                 the surrogate is described on its own run's card
  Objective    — the loss table with each term's share before and after
                 training, the power diagnostic, and the objective's notes
  Comfort      — per regulated channel, worst first when there are many;
                 energy and effort on one line each
  Actuation    — per command, or only the ones parked or idle when there are
                 many
  Band jumps   — did the loop react when the reference moved
  Hyperparameters — one line

Only the dominated objective and the ignored band jump are verdicts on the
controller. Time at a bound is reported, not judged: a setpoint parked at its
least-aggressive limit while nothing is asked of it is correct control.

`render_gap_report` is the card `nt control gap` writes: the controller beside the
best open-loop plan.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from neurotemplate.common.reportcard import RESULTS_DIR, _names, _section
from neurotemplate.control.losses import Objective, resolve_objective
from neurotemplate.control.metrics import SATURATION_TOL
from neurotemplate.control.objective import ObjectiveBalance, PowerDiagnostics
from neurotemplate.control.scenarios import horizons
from neurotemplate.control.sensitivity import CommandSensitivity
from neurotemplate.modeling.reportcard import MAX_ROWS, WORST_ROWS

from neurotemplate._typing import beartype

#: A band-jump request actuated less often than this is "largely ignored".
IGNORED_BELOW = 0.5
#: A command whose surrogate gradient is zero this often cannot be learned.
INSENSITIVE_AT = 0.5
#: Below this share of power steps passing no gradient, the clamp is not worth a line.
DEAD_POWER_AT = 0.01


def _header(name: str, timestamp: str, cfg, sysid_cfg, eval_time_s: float,
            best_dev_loss: Optional[float], train_time_s: Optional[float]) -> str:
    parts = [timestamp, f"sys-id run `{cfg.sysid_run_id}` ({sysid_cfg.model}; "
             "its reportcard describes the surrogate)"]
    if cfg.policy == "mpc":
        parts.append("MPC: nothing trained")
    if best_dev_loss is not None:
        parts.append(f"dev loss {best_dev_loss:.4g}")
    if train_time_s is not None:
        parts.append(f"trained {train_time_s:.0f} s")
    parts.append(f"evaluation {eval_time_s:.2f} s")
    return f"# Control reportcard: {name}\n\n" + " · ".join(parts) + "\n"


# ── Verdict ───────────────────────────────────────────────────────


def _verdict_section(results, balance: Optional[ObjectiveBalance],
                     trained: Optional[ObjectiveBalance],
                     power: Optional[PowerDiagnostics],
                     sensitivity: Optional[list[CommandSensitivity]], arch: dict) -> str:
    worst = max(results.comfort, key=lambda m: m.mean_exceedance)
    lines = [f"- **Comfort:** {results.comfort_violation_rate * 100:.1f}% of steps outside "
             f"the band, mean exceedance {results.comfort_mean_exceedance:.4g} over "
             f"{len(results.comfort)} regulated channel(s); worst `{worst.name}` "
             f"{worst.mean_exceedance:.4g}."]
    if results.response:
        jumps = sum(r.n_jumps for r in results.response)

        def share(attr: str) -> float:
            return sum(getattr(r, attr) * r.n_jumps for r in results.response) / jumps
        lines.append(f"- **Band jumps:** {jumps} requests; actuated on "
                     f"{share('frac_actuated') * 100:.0f}%, moved the right way on "
                     f"{share('frac_closed') * 100:.0f}%, back in band on "
                     f"{share('frac_resolved') * 100:.0f}%.")
        ignored = [r for r in results.response if r.frac_actuated < IGNORED_BELOW]
        if ignored:
            lines.append("- ⚠️ Requests largely ignored: " + _names(
                [f"{r.name} / {r.direction} ({r.frac_actuated * 100:.0f}% actuated)"
                 for r in ignored], quote=False) + " (Band jumps).")
    else:
        lines.append("- **Band jumps:** none left a channel outside its band, so the "
                     "response is untested; a test reference whose band steps tests it.")
    for b, when in ((balance, "before training"), (trained, "after training")):
        if b is not None:
            for t in b.dominant():
                lines.append(f"- ⚠️ `{t.name}` runs the objective {when}: "
                             f"{t.ratio_to_rest:.1f}× every other term combined (Objective).")
    if power is not None and power.frac_dead > DEAD_POWER_AT:
        lines.append(f"- ⚠️ No gradient through power on {power.frac_dead * 100:.0f}% of "
                     "scored steps: a clamp on the power path (Objective).")
    for s in sensitivity or []:
        if s.zero_share >= INSENSITIVE_AT:
            lines.append(f"- ⚠️ The surrogate ignores `{s.column}` on "
                         f"{s.zero_share * 100:.0f}% of recorded windows: its gradient is "
                         "zero there, so no controller can learn or plan it.")
    if arch.get("solves"):
        capped = sum(n >= arch["max_iters"] for n in arch["solves"]) / len(arch["solves"])
        if capped > 0:
            lines.append(f"- ⚠️ {capped * 100:.0f}% of MPC solves stopped at "
                         f"the solver's `max_iters` ({arch['max_iters']}) rather than converging.")
    lines.append(f"\nEvery instrument's full output is in `{RESULTS_DIR}/` beside this card.")
    return _section("Verdict", "\n".join(lines))


# ── Controller ────────────────────────────────────────────────────


def _grouped(entries: dict[str, str]) -> list[str]:
    """{column: description} as one line per distinct description."""
    groups: dict[str, list[str]] = {}
    for column, text in entries.items():
        groups.setdefault(text, []).append(column)
    return [f"  - {text}: {_names(cols)}" for text, cols in groups.items()]


def _bounds_lines(arch: dict) -> list[str]:
    lines = ["- Command range:"] + _grouped({
        col: f"[{lo:.4g}, {hi:.4g}]"
        for col, lo, hi in zip(arch["u_cols"], arch["umin_phys"], arch["umax_phys"])})
    if arch.get("control_bounds"):
        lines += ["- `control_bounds` (a name moves per step):"] + _grouped({
            col: f"[{low}, {high}]" for col, (low, high) in arch["control_bounds"].items()})
    return lines


def _controller_section(arch: dict) -> str:
    inputs = ", ".join(f"`{key}`" for key in arch["inputs"])
    if arch["family"] == "mpc":
        lines = [
            f"- `mpc`: at every step, Adam plans {arch['horizon']} steps through the "
            f"surrogate (lr {arch['lr']:g} of each range, patience {arch['patience']}, "
            f"rtol {arch['rtol']:g}, cap {arch['max_iters']}) and commands the first; "
            f"reads {inputs}",
        ]
        if arch.get("solves"):
            steps = sorted(arch["solves"])
            lines.append(f"- {len(steps)} solves on the test windows; Adam steps median "
                         f"{steps[len(steps) // 2]}, max {steps[-1]}")
        return _section("Controller", "\n".join(lines + _bounds_lines(arch)))
    hsizes = arch["hsizes"] or "none (one affine map)"
    lines = [
        f"- `{arch['family']}` ({arch['nonlin']}, `{arch['bound_method']}`), hidden "
        f"{hsizes}, {arch['param_count']:,} params; {arch['insize']} inputs over {inputs}, "
        f"{arch['outsize']} outputs",
        "- Window, steps before / after now (a U column's ends at the last command):",
        *_grouped({col: f"past {e['past']}" + (f", future {e['future']}" if "future" in e else "")
                   for col, e in arch["window"].items()}),
    ]
    return _section("Controller", "\n".join(lines + _bounds_lines(arch)))


# ── Objective ─────────────────────────────────────────────────────


def _objective_section(cfg, sysid_cfg, objective: Objective,
                       balance: Optional[ObjectiveBalance],
                       trained: Optional[ObjectiveBalance],
                       power: Optional[PowerDiagnostics],
                       notes: Optional[list[str]], trains: bool) -> str:
    """Objective weights do not port across datasets; the shares say what the
    controller was really asked to minimise, whatever the weights suggest."""
    h_sys, h_ctrl, _ = horizons(cfg, sysid_cfg)
    before = {} if balance is None else {t.name.removesuffix("_loss"): t for t in balance.terms}
    after = {} if trained is None else {t.name.removesuffix("_loss"): t for t in trained.terms}
    when = "before training" if trains else "through the controller"
    body = [
        f"History H_sys={h_sys}, H_ctrl={h_ctrl} steps; horizon {cfg.horizon}, "
        f"test horizon {cfg.test_horizon}. Regulated: {_names(objective.regulated)}. "
        "Each term is `gain · power-mean(map(operand))` per window, in physical units; "
        f"its share of the objective is measured on one batch {when}"
        + (" and after." if trained is not None else ".") + "\n",
        "| Term | Operand | Map | Norm | Weight | Gain | Share"
        + (" | Trained |" if trained is not None else " |"),
        "|---|---|---|---:|---|---:|---:|" + ("---:|" if trained is not None else ""),
    ]
    for t in objective.terms:
        operand = f"`{t.column}`" + ("" if t.target is None else f" − `{t.target}`")
        row = (f"| {t.name} | {operand} | {t.map or '—'} | {t.norm:g} | "
               f"{'—' if t.weight is None else f'`{t.weight}`'} | {t.gain:g} | ")
        row += f"{before[t.name].share * 100:.0f}% |" if t.name in before else "— |"
        if trained is not None:
            row += f" {after[t.name].share * 100:.0f}% |" if t.name in after else " — |"
        body.append(row)
    if power is not None:
        body.append(f"\n- Power reads exactly zero on {power.frac_zero * 100:.1f}% of scored "
                    f"steps and passes no gradient on {power.frac_dead * 100:.1f}%. Zero is "
                    "fine (equipment off); no gradient is a clamp on the power path — a "
                    "negative power head, or `relu`/`clamp_min` in `power_metric` — and "
                    "starves a cost term of exactly the cheap hours. `soft_clamp_min` "
                    "keeps the zero and passes gradient.")
    body += [f"- {n}" for n in notes or []]
    return _section("Objective", "\n".join(body))


# ── Comfort ───────────────────────────────────────────────────────


def _comfort_section(results) -> str:
    rows = sorted(results.comfort, key=lambda m: -m.mean_exceedance)
    shown = rows if len(rows) <= MAX_ROWS else rows[:WORST_ROWS]
    body = ["Mean exceedance is over all steps (tune on it); violation is the share "
            "of steps outside the band (accept on it). Physical units.\n"]
    if len(shown) < len(rows):
        body[0] += (f"The {len(shown)} worst of {len(rows)} channels; all are in "
                    f"`{RESULTS_DIR}/control_metrics.json`.\n")
    body += ["| Channel | Mean exceedance | Violation | Below | Above | Max excursion | "
             "Degree-hours |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m in shown:
        body.append(f"| `{m.name}` | {m.mean_exceedance:.4g} | {m.violation_rate * 100:.1f}% | "
                    f"{m.frac_below * 100:.1f}% | {m.frac_above * 100:.1f}% | "
                    f"{m.max_violation:.4g} | {m.degree_hours:.4g} |")
    body.append(f"\n- Effort: mean |ΔU| {results.mean_abs_du:.4g} over every command.")
    e = results.energy
    if e is None:
        body.append("- Energy: n/a (no `power_metric`).")
    else:
        weighted = "".join(f" · ∫`{w}`·power {v:.4g}" for w, v in e.weighted_energy.items())
        body.append(f"- Energy, in `power_metric`'s units (energy is power·h; a weighted "
                    f"integral is not unit-converted): mean power {e.mean_power:.4g} · per "
                    f"window {e.total_energy:.4g} · peak {e.peak_power:.4g} · mean window "
                    f"peak {e.mean_window_peak:.4g}{weighted}")
    return _section("Comfort", "\n".join(body))


def render_gap_report(plan, results, run_id: str) -> str:
    """The controller beside the best open-loop plan on the same windows
    (`control.plan`): a bar for any controller on this surrogate."""
    names = [term.name for term in plan.policy.terms]
    by_name = [{term.name: term.value for term in balance.terms}
               for balance in (plan.policy, plan.plan)]
    cost = {} if results.energy is None else results.energy.weighted_energy
    plan_cost = {} if plan.results.energy is None else plan.results.energy.weighted_energy
    head = (["", "Objective"] + [n.removesuffix("_loss") for n in names]
            + ["Violation", "Mean exceedance"] + [f"∫`{w}`·power" for w in cost])
    rows = []
    for label, values, scored, weighted in (("Controller", by_name[0], results, cost),
                                            ("Best plan", by_name[1], plan.results, plan_cost)):
        rows.append([label, f"{sum(values.values()):.4g}"]
                    + [f"{values[n]:.4g}" for n in names]
                    + [f"{scored.comfort_violation_rate * 100:.1f}%",
                       f"{scored.comfort_mean_exceedance:.4g}"]
                    + [f"{weighted[w]:.4g}" for w in cost])
    converged = ("every window stopped on patience" if plan.converged else
                 f"**some windows were still improving at the solver's `max_iters` "
                 f"({plan.steps})**, so the bar and the gap are understated")
    body = [
        f"The best plan is a free command per step through the same surrogate, bounds "
        f"and objective, knowing every step ahead ({plan.time_s:.0f} s, {plan.starts} "
        f"start(s); {converged}). What it cannot do either, the surrogate does not make "
        "pay.\n",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|",
    ] + ["| " + " | ".join(row) + " |" for row in rows]
    return (f"# Controller gap: {run_id}\n\nThe best plan on the surrogate scores "
            f"{plan.gap * 100:.0f}% below the controller.\n\n" + "\n".join(body) + "\n")


# ── Actuation ─────────────────────────────────────────────────────


def _actuation_section(results) -> str:
    """Per-command effort and time at a bound, without a verdict: read it
    with the band jumps, which tell "parked because nothing was asked" from
    "did not answer"."""
    body = [f"\"At a bound\" is within {SATURATION_TOL:.0%} of the range of a limit. "
            "Time there is a fact, not a fault.\n"]
    acts = results.actuation
    if len(acts) <= MAX_ROWS:
        body += ["| Command | Bounds | Mean | Mean \\|Δu\\| | At lower | At upper |",
                 "|---|---|---:|---:|---:|---:|"]
        body += [f"| `{a.name}` | [{a.u_min:.4g}, {a.u_max:.4g}] | {a.mean_value:.4g} | "
                 f"{a.mean_abs_du:.4g} | {a.frac_at_min * 100:.0f}% | "
                 f"{a.frac_at_max * 100:.0f}% |" for a in acts]
    else:
        parked = [f"{a.name} {a.frac_saturated * 100:.0f}%" for a in acts
                  if a.frac_saturated > 0.5]
        idle = [a.name for a in acts
                if a.mean_abs_du < SATURATION_TOL * (a.u_max - a.u_min)]
        body.append(f"- {len(acts)} commands; per command in "
                    f"`{RESULTS_DIR}/control_metrics.json`.")
        body.append("- At a bound more than half the time: "
                    + (_names(parked, quote=False) if parked else "none"))
        body.append(f"- Moving less than {SATURATION_TOL:.0%} of their range per step: "
                    + (_names(idle) if idle else "none"))
    return _section("Actuation", "\n".join(body))


# ── Band jumps ────────────────────────────────────────────────────


def _response_section(results) -> Optional[str]:
    """Pooled violation ranked two real controllers backwards once: 16.6% vs
    17.9%, and the better-scoring one reacted to none of the band jumps that
    demanded cooling. This is the audit that tells them apart."""
    if not results.response:
        return None
    rows = sorted(results.response, key=lambda r: r.frac_actuated)
    shown = rows if len(rows) <= MAX_ROWS else rows[:WORST_ROWS]
    body = ["Each band step that left a channel outside the new band, scored until the "
            "next one: did a command move on that step, did the channel then move the "
            "demanded way, and was it back in band by the end.\n"]
    if len(shown) < len(rows):
        body[0] += f"The {len(shown)} least actuated of {len(rows)}.\n"
    body += ["| Channel | Demand | Jumps | Actuated | Right way | Back in band | "
             "Largest Δ command |", "|---|---|---:|---:|---:|---:|---|"]
    for r in shown:
        name, du = max(r.mean_du.items(), key=lambda kv: abs(kv[1]))
        body.append(f"| `{r.name}` | {r.direction} | {r.n_jumps} | "
                    f"{r.frac_actuated * 100:.0f}% | {r.frac_closed * 100:.0f}% | "
                    f"{r.frac_resolved * 100:.0f}% | `{name}` {du:+.3g} |")
    return _section("Band jumps", "\n".join(body))


def _hyperparameters_line(cfg) -> str:
    return _section("Hyperparameters", (
        f"lr {cfg.lr} · batch {cfg.batch_size} · epochs {cfg.epochs} · warmup "
        f"{cfg.warmup} · patience {cfg.patience} · test stride {cfg.test_stride} · "
        f"seed {cfg.seed}"))


@beartype
def render_control_reportcard(
    cfg,
    sysid_cfg,
    arch: dict,
    nm_mapping: dict[str, list[str]],
    results,
    *,
    balance: Optional[ObjectiveBalance] = None,
    trained_balance: Optional[ObjectiveBalance] = None,
    power: Optional[PowerDiagnostics] = None,
    objective_notes: Optional[list[str]] = None,
    sensitivity: Optional[list[CommandSensitivity]] = None,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    best_dev_loss: Optional[float] = None,
    train_time_s: Optional[float] = None,
) -> str:
    """
    Render the markdown reportcard for one control run.

    Args:
        arch:        Output of control.policy.describe_policy(...).
        results:     ControlResults for the controller.
        balance:     Optional objective.ObjectiveBalance from the pre-training
                     forward pass; `trained_balance` the same batch after.
        power:       Optional objective.power_diagnostics(...).
        objective_notes: Optional objective.objective_notes(...) prose.
        sensitivity: Optional sensitivity.command_sensitivity(...): per
                     command, how often the surrogate does not move with it.

    Returns:
        Markdown string. The caller writes it to disk and logs it.
    """
    name = run_name if run_name is not None else cfg.run
    ts = timestamp if timestamp is not None else datetime.now().isoformat(
        timespec="seconds"
    )
    trains = cfg.policy != "mpc"
    parts = [
        _header(name, ts, cfg, sysid_cfg, results.eval_time_s, best_dev_loss, train_time_s),
        _verdict_section(results, balance, trained_balance, power, sensitivity, arch),
        _controller_section(arch),
        _objective_section(cfg, sysid_cfg, resolve_objective(cfg, nm_mapping), balance,
                           trained_balance, power, objective_notes, trains),
        _comfort_section(results),
    ]
    parts.append(_actuation_section(results))
    response = _response_section(results)
    if response is not None:
        parts.append(response)
    parts.append(_hyperparameters_line(cfg))
    return "\n".join(parts)
