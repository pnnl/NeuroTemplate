"""
Objective diagnostics: is each term the right size, and does it point the
right way?

`build_control_loss` sums comfort, du and effort with user-supplied weights,
and until now nothing checked that a weight suited the magnitude of what it
multiplies. That is the highest-frequency failure in this library's record —
four campaigns out of four — and it comes in two flavours that need different
answers:

  mis-scaled  the term has the right shape but the wrong magnitude, so it
              dominates or vanishes. Re-derive the weight. `objective_balance`
              finds this: one forward pass, print every term's contribution.

  mis-shaped  the term's gradient points the same way across the entire
              feasible set, so its minimum sits on an actuation bound. **No
              value of the weight fixes this** — the objective is instructing
              the policy to saturate an actuator. `term_shapes` finds this by
              sweeping each control across its box and asking where the
              minimum lies.

The worked example both were written for: an effort proxy of
`(cooling_setpoint - outdoor)^2` on a spring plant whose outdoor range
(9-18 °C) does not overlap its setpoint range (20-26 °C). The term is monotone
decreasing over the whole box, it was 1.7x the entire comfort objective, and
the policy did the rational thing — it pinned the cooling setpoint at its
floor for 87% of the evaluation and ran the zone on heating alone. Ten runs
scored a respectable violation rate before anyone noticed. Both numbers here
were on screen from epoch 1; nobody was comparing them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurotemplate._typing import beartype


# A term worth more than this multiple of everything else put together is
# running the show, whatever the other weights say.
DOMINANCE_FACTOR = 2.0

# Sweep resolution per control for the shape check. Odd, so the midpoint of
# the actuation box is always one of the samples.
SHAPE_GRID = 9

# A sweep whose spread is below this share of its own magnitude is flat: the
# term does not read that control directly, which is the normal case for
# metered power (the control reaches it through the dynamics instead).
FLAT_TOL = 1e-6


@dataclass
class TermBalance:
    """One objective term's contribution on a single batch."""
    name: str
    value: float
    share: float           # of the summed objective
    ratio_to_rest: float   # value / sum of the other terms; inf when alone

    def dominates(self, factor: float = DOMINANCE_FACTOR) -> bool:
        return self.ratio_to_rest > factor


@dataclass
class ObjectiveBalance:
    terms: list[TermBalance]
    total: float

    def dominant(self, factor: float = DOMINANCE_FACTOR) -> list[TermBalance]:
        return [t for t in self.terms if t.dominates(factor)]


@dataclass
class TermShape:
    """How an objective term varies as one control sweeps its actuation box."""
    u_name: str
    grid: list[float]     # commanded values swept, in physical units
    values: list[float]   # mean term value at each grid point
    argmin: int

    @property
    def spread(self) -> float:
        return max(self.values) - min(self.values)

    @property
    def is_flat(self) -> bool:
        scale = max(abs(sum(self.values) / len(self.values)), 1.0)
        return self.spread <= FLAT_TOL * scale

    @property
    def minimum_on_boundary(self) -> bool:
        """True when the term is minimised by pinning this control at a limit.

        A boundary minimum means the term is monotone across the feasible set,
        i.e. it is an instruction to saturate the actuator rather than a cost
        to trade off. Flat sweeps are excluded — they carry no instruction.
        """
        return (not self.is_flat
                and self.argmin in (0, len(self.values) - 1))

    @property
    def pinned_at(self) -> float:
        return self.grid[self.argmin]


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
def term_shapes(
    term_fn,
    out: dict,
    hmax: int,
    u_names: list[str],
    u_min: torch.Tensor,
    u_max: torch.Tensor,
    norm_stats: dict[str, dict[str, float]],
    n_grid: int = SHAPE_GRID,
) -> list[TermShape]:
    """Sweep each control across its actuation box and record the term's shape.

    Args:
        term_fn: `(yn, U, D) -> [batch, time]`, already bound to nm_mapping and
            norm_stats — the same callable the loss evaluates.
        out:     a closed-loop problem output (see `run_objective_pass`).
        hmax:    warmup steps to skip, so the sweep covers the scored horizon.
        u_min/u_max: actuation box in NORMALIZED space, one entry per control.

    Every other control keeps its recorded trajectory, so this is the same
    one-at-a-time reading as the sys-id authority probe: what does this term
    ask *this* actuator to do, holding the rest of the rollout fixed.
    """
    yn = out["yn"][:, hmax:, :]
    u = out["U"][:, hmax:, :]
    d = out["D"][:, hmax:, :]
    device = u.device
    u_min, u_max = u_min.to(device), u_max.to(device)

    shapes = []
    with torch.no_grad():
        for j, name in enumerate(u_names):
            s = norm_stats[name]
            grid_n = torch.linspace(float(u_min[j]), float(u_max[j]), n_grid,
                                    device=device)
            values = []
            for point in grid_n:
                swept = u.clone()
                swept[:, :, j] = point
                values.append(float(term_fn(yn, swept, d).mean()))
            shapes.append(TermShape(
                u_name=name,
                grid=[float(p) * s["stretch"] + s["shift"] for p in grid_n],
                values=values,
                argmin=min(range(n_grid), key=values.__getitem__),
            ))
    return shapes


# ── Rendering ────────────────────────────────────────────────────


@beartype
def balance_to_markdown(balance: ObjectiveBalance) -> str:
    rows = ["| Term | Value | Share | vs. the rest | |",
            "|---|---:|---:|---:|---|"]
    for t in balance.terms:
        ratio = "∞" if t.ratio_to_rest == float("inf") else f"{t.ratio_to_rest:.2f}×"
        flag = "⚠️ dominates" if t.dominates() else ""
        rows.append(
            f"| {t.name} | {t.value:.4g} | {t.share * 100:.0f}% | {ratio} | {flag} |"
        )
    return "\n".join(rows)


@beartype
def shapes_to_markdown(shapes: list[TermShape]) -> str:
    rows = ["| Control | Term at lo | at mid | at hi | Minimised at | |",
            "|---|---:|---:|---:|---:|---|"]
    for s in shapes:
        mid = len(s.values) // 2
        if s.is_flat:
            where, flag = "flat (no direct effect)", ""
        else:
            where = f"{s.pinned_at:.4g}"
            flag = "⚠️ minimum on the bound" if s.minimum_on_boundary else ""
        rows.append(
            f"| {s.u_name} | {s.values[0]:.4g} | {s.values[mid]:.4g} | "
            f"{s.values[-1]:.4g} | {where} | {flag} |"
        )
    return "\n".join(rows)


@beartype
def objective_to_dict(balance: ObjectiveBalance,
                      shapes: list[TermShape]) -> dict:
    """JSON-serializable objective diagnostics (for MLflow / disk)."""
    return {
        "total": balance.total,
        "terms": {
            t.name: {"value": t.value, "share": t.share,
                     "ratio_to_rest": t.ratio_to_rest,
                     "dominates": t.dominates()}
            for t in balance.terms
        },
        "effort_shape": {
            s.u_name: {"grid": s.grid, "values": s.values,
                       "flat": s.is_flat,
                       "minimum_on_boundary": s.minimum_on_boundary,
                       "pinned_at": s.pinned_at}
            for s in shapes
        },
    }


@beartype
def format_objective_report(balance: ObjectiveBalance,
                            shapes: list[TermShape]) -> str:
    """Plain-text version for stdout, printed before training starts."""
    lines = ["Objective term balance (one batch, before training):"]
    for t in balance.terms:
        ratio = "inf" if t.ratio_to_rest == float("inf") else f"{t.ratio_to_rest:.2f}x"
        note = "  <-- dominates" if t.dominates() else ""
        lines.append(f"  {t.name:<22s} {t.value:>12.4g}  "
                     f"{t.share * 100:>3.0f}%  {ratio:>8s} the rest{note}")
    for t in balance.dominant():
        lines.append(
            f"  WARNING: {t.name} is {t.ratio_to_rest:.1f}x every other term "
            "combined. Whatever the other weights say, this is the objective "
            "being optimised."
        )
    for s in shapes:
        if s.minimum_on_boundary:
            lines.append(
                f"  WARNING: the effort term is minimised by holding "
                f"{s.u_name} at {s.pinned_at:.4g}, the edge of its actuation "
                "range — it is monotone across the whole feasible set, so no "
                "weight fixes it. Re-derive the term, do not re-tune "
                "effort_weight."
            )
    return "\n".join(lines)
