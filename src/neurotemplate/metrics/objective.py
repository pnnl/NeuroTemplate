"""
Objective diagnostics: how big is each term?

`build_control_loss` sums the declared loss terms, each scaled by its gain,
and nothing checks that a gain suits the magnitude of what it multiplies.
That is the highest-frequency failure in this library's record — four campaigns
out of four — and it is a *scale* problem: the term has the right shape but the
wrong magnitude. `objective_balance` shows it in one forward pass by printing
every term's contribution and share.

The shares are reported, not judged. A band term is a hinge and reads zero while the
band holds, so a controller keeping comfort shows cost near 100%, which is the good
case. A share cannot tell you the trade; the blind twin in `nt guide control` does.

The worked example: a power proxy of `(cooling_setpoint - outdoor)^2` that
came out at 1.7x the entire comfort objective. The numbers were on screen from
the first evaluation; nobody was comparing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from neurotemplate.control.losses import POWER, Objective
from neurotemplate.control.windows import scored_steps
from neurotemplate._typing import beartype


@dataclass
class TermBalance:
    """One objective term's contribution on a single batch."""
    name: str
    value: float
    share: float           # of the summed objective
    ratio_to_rest: float   # value / sum of the other terms; inf when alone


@dataclass
class ObjectiveBalance:
    terms: list[TermBalance]
    total: float


@dataclass
class PowerDiagnostics:
    """What one backward pass through the power term says about it.

    `frac_dead` is the number that carries a warning: power reading zero *and*
    no gradient reaching the surrogate, which is the clamp having cut the
    chain. `frac_zero` alone says nothing — see `power_diagnostics`.
    """
    frac_zero: float
    frac_dead: float


@beartype
def run_objective_pass(problem, batch: dict) -> dict:
    """One no-grad forward pass of the whole DPC problem on one batch.

    Returns the problem's output dict, which carries both the loss terms and
    the closed-loop trajectory the terms were computed from — so the balance
    report and the shape sweep share a single rollout. `Problem` prefixes
    every key with the batch's dataset name; that prefix is stripped here so
    callers index by the plain key.
    """
    was_training = problem.training
    problem.eval()
    try:
        with torch.no_grad():
            out = problem(dict(batch))
    finally:
        problem.train(was_training)

    prefix = f"{batch['name']}_"
    return {k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in out.items()}


@beartype
def objective_balance(loss, out: dict) -> ObjectiveBalance:
    """Each objective term's value and share of the total, from one pass.

    Term names come from the loss's own objectives (`Constraint.update_name`),
    so a new term shows up here without this function being told about it.
    """
    values = {o.name: float(out[o.name]) for o in loss.objectives}
    total = sum(values.values())

    terms = []
    for name, value in values.items():
        rest = total - value
        if rest > 0:
            ratio = value / rest
        elif value > 0:
            ratio = float("inf")   # the only live term, or the only positive one
        else:
            ratio = 0.0            # nothing to dominate: this term is zero too
        terms.append(TermBalance(
            name=name,
            value=value,
            share=value / total if total != 0 else 0.0,
            ratio_to_rest=ratio,
        ))
    return ObjectiveBalance(terms=terms, total=total)


@beartype
def power_gradient(term_fn, out: dict, hmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(power, |∂power/∂yn| summed over channels) over the scored horizon.

    Both are [batch, time]. The gradient is taken w.r.t. a fresh leaf copy of
    `yn`, so this measures what the power term itself passes back to the
    surrogate's outputs — the only place a clamp can silently cut the chain.

    A term that does not read `yn` at all gets an all-zero gradient.
    """
    with torch.enable_grad():
        yn = (out["yn"][:, scored_steps("yn", hmax), :]
              .detach().clone().requires_grad_(True))
        power = term_fn(yn, out["U"][:, scored_steps("U", hmax), :],
                        out["D"][:, scored_steps("D", hmax), :])
        if not power.requires_grad:
            return power.detach(), torch.zeros_like(power)
        grad, = torch.autograd.grad(power.sum(), yn, allow_unused=True)
        if grad is None:
            return power.detach(), torch.zeros_like(power)
        return power.detach(), grad.abs().sum(-1)


@beartype
def power_diagnostics(term_fn, out: dict, hmax: int) -> PowerDiagnostics:
    """How often the power signal reads zero, and how often that costs a gradient.

    A metered `power_metric` ends in `.clamp_min(0.0)`, and wherever the
    surrogate predicts negative power that clamp zeroes the gradient; a
    proxy written with a `relu` zeroes it wherever the equipment coasts. For
    a mean energy term that is a rounding detail. For a cost term it is not:
    the near-zero-power steps are disproportionately the cheap hours the
    controller is supposed to be moving load *into*, so a large dead fraction
    means the cost gradient is missing exactly where it was needed.

    The two numbers are separate because a zero power reading is not by itself
    a defect. A regression head that predicts −101 W is clamped to zero and stops
    passing gradient, a surrogate problem worth acting on; a bounded head resting
    on its floor reports zero and keeps passing gradient, the model being right.
    Measuring the gradient tells them apart without the diagnostic needing to know
    what heads exist.

    Reported, not judged — and a large `frac_dead` is a clamp problem (a power
    head that goes negative, or a hard clamp in a proxy), not a weighting
    problem.
    """
    power, grad = power_gradient(term_fn, out, hmax)
    is_zero = power == 0
    return PowerDiagnostics(
        frac_zero=float(is_zero.float().mean()),
        frac_dead=float((is_zero & (grad == 0)).float().mean()),
    )


@beartype
def objective_notes(objective: Objective, cfg) -> list[str]:
    """Objective diagnostics that need the config, not a forward pass.

    All the same kind of failure: a term that is switched on but cannot do
    the thing it was switched on for. Reported as prose rather than raised,
    because every one of these configurations is legal and occasionally
    deliberate.
    """
    notes = []
    for term in objective.terms:
        if (term.gain > 0 and term.weight is not None
                and objective.roles[term.weight] == "R"
                and cfg.window_steps.get(term.weight, {}).get("future", 0) == 0):
            notes.append(
                f"`{term.name}` is weighted by `{term.weight}`, but the policy "
                "previews 0 future steps of it, so it can react to the current "
                "value and cannot act ahead of it — shifting load into a "
                "cheaper hour needs its `future` in `policy_window` to cover the cycle "
                "(`future: 4h` to see four hours ahead)."
            )
    power_terms = [t for t in objective.terms
                   if t.gain > 0 and t.column == POWER and t.target is None]
    if power_terms and all(t.norm > 1 for t in power_terms):
        notes.append(
            "Every term on `power` has a norm above 1 (a peak). Flattening a "
            "profile by *adding* off-peak load is a valid solution to "
            "min M_q(p) and a bad outcome — carry a `norm: 1` power term "
            "(energy or cost) alongside it."
        )
    return notes


# ── Rendering ────────────────────────────────────────────────────


@beartype
def objective_to_dict(balance: ObjectiveBalance,
                      power: Optional[PowerDiagnostics] = None,
                      notes: Optional[list[str]] = None,
                      trained: Optional[ObjectiveBalance] = None) -> dict:
    """JSON-serializable objective diagnostics (for MLflow / disk)."""
    def terms(b: ObjectiveBalance) -> dict:
        return {t.name: {"value": t.value, "share": t.share,
                         "ratio_to_rest": t.ratio_to_rest}
                for t in b.terms}

    return {
        "total": balance.total,
        "terms": terms(balance),
        "trained_total": None if trained is None else trained.total,
        "trained_terms": None if trained is None else terms(trained),
        "frac_power_zero": None if power is None else power.frac_zero,
        "frac_power_dead": None if power is None else power.frac_dead,
        "notes": list(notes or []),
    }


@beartype
def format_objective_report(balance: ObjectiveBalance,
                            power: Optional[PowerDiagnostics] = None,
                            notes: Optional[list[str]] = None,
                            trains: bool = True) -> str:
    """Plain-text version for stdout, printed before training starts.
    `trains` is False for a controller that trains nothing (MPC)."""
    when = "before training" if trains else "through the controller"
    lines = [f"Objective term balance (one batch, {when}):"]
    for t in balance.terms:
        ratio = "inf" if t.ratio_to_rest == float("inf") else f"{t.ratio_to_rest:.2f}x"
        lines.append(f"  {t.name:<22s} {t.value:>12.4g}  "
                     f"{t.share * 100:>3.0f}%  {ratio:>8s} the rest")
    if power is not None:
        lines.append(f"  power exactly zero on {power.frac_zero * 100:.1f}% of "
                     "scored steps")
        lines.append(f"  ...of which {power.frac_dead * 100:.1f}% of all steps "
                     "also pass no gradient back (a clamp on the power path: "
                     "a negative power head, or relu/clamp_min in power_metric)")
    for note in notes or []:
        lines.append(f"  NOTE: {note}")
    return "\n".join(lines)
