"""
Objective diagnostics: how big is each term, and what does it look like?

`build_control_loss` sums comfort, du and effort with user-supplied weights,
and nothing checks that a weight suits the magnitude of what it multiplies.
That is the highest-frequency failure in this library's record — four campaigns
out of four — and it is a *scale* problem: the term has the right shape but the
wrong magnitude, so it dominates the objective or vanishes from it.
`objective_balance` answers it in one forward pass by printing every term's
contribution and share.

The worked example: an effort proxy of `(cooling_setpoint - outdoor)^2` that
came out at 1.7x the entire comfort objective. The numbers were on screen from
epoch 1; nobody was comparing them.

`term_shapes` sweeps the effort term across each actuation box and reports what
it looks like there — the value at each grid point and where the minimum lands.
This is **reported, not judged**. A monotone effort term is normal and usually
correct: a real power meter costs more as an actuator works harder, so it has
no interior optimum of its own. The trade-off that produces an interior optimum
belongs to the *summed* objective, where effort meets comfort — not to the
effort term in isolation. The sweep is here so a reader can see the shape and
the scale of the term they are weighting, and decide for themselves.
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
        """The term does not read this control directly.

        Normal for metered power, which the control reaches only through the
        dynamics. Reported so a flat row is not misread as a missing sweep.
        """
        scale = max(abs(sum(self.values) / len(self.values)), 1.0)
        return self.spread <= FLAT_TOL * scale

    @property
    def minimised_at(self) -> float:
        """Where over the swept box this term is smallest, in physical units.

        Carries no verdict. A minimum on an endpoint means the term is monotone
        over the actuation range, which is what an honest power metric looks
        like — the interior optimum belongs to the summed objective.
        """
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
    one-at-a-time reading the step-response scenario uses: what does this term
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
    rows = ["| Control | Term at lo | at mid | at hi | Minimised at |",
            "|---|---:|---:|---:|---:|"]
    for s in shapes:
        mid = len(s.values) // 2
        where = ("flat (no direct effect)" if s.is_flat
                 else f"{s.minimised_at:.4g}")
        rows.append(
            f"| {s.u_name} | {s.values[0]:.4g} | {s.values[mid]:.4g} | "
            f"{s.values[-1]:.4g} | {where} |"
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
                       "minimised_at": s.minimised_at}
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
    if shapes:
        lines.append("Effort term across each actuation box (no verdict "
                     "attached — read it against the weights above):")
        for s in shapes:
            where = ("flat, no direct effect" if s.is_flat
                     else f"minimised at {s.minimised_at:.4g}")
            lines.append(f"  {s.u_name:<22s} {s.values[0]:>12.4g} .. "
                         f"{s.values[-1]:<12.4g} {where}")
    return "\n".join(lines)
