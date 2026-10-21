"""
Objective diagnostics: how big is each term, and what does it look like?

`build_control_loss` sums the declared loss terms, each scaled by its gain,
and nothing checks that a gain suits the magnitude of what it multiplies.
That is the highest-frequency failure in this library's record — four campaigns
out of four — and it is a *scale* problem: the term has the right shape but the
wrong magnitude, so it dominates the objective or vanishes from it.
`objective_balance` answers it in one forward pass by printing every term's
contribution and share.

The worked example: a power proxy of `(cooling_setpoint - outdoor)^2` that
came out at 1.7x the entire comfort objective. The numbers were on screen from
epoch 1; nobody was comparing them.

`term_shapes` sweeps the power signal across each actuation box, through the
model: the rollout is re-run with one control held at each grid value, so a
metered power the model predicts moves as the model says it would. It reports
the value at each grid point and where the minimum lands.
This is **reported, not judged**. A monotone power term is normal and usually
correct: a real power meter costs more as an actuator works harder, so it has
no interior optimum of its own. The trade-off that produces an interior optimum
belongs to the *summed* objective, where energy meets comfort — not to the
power term in isolation. The sweep is here so a reader can see the shape and
the scale of the term they are weighting, and decide for themselves.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

from neurotemplate.common.data_utils import NormStats
from neurotemplate.control.losses import POWER, Objective
from neurotemplate.control.scenarios import scored_steps
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
class PowerDiagnostics:
    """What one backward pass through the power term says about it.

    `frac_dead` is the number that carries a warning: power reading zero *and*
    no gradient reaching the surrogate, which is the clamp having cut the
    chain. `frac_zero` alone says nothing — see `power_diagnostics`.

    `reads_outputs` is what makes a flat sweep readable. A term that is flat
    across every actuation box is either metered power (which reads the
    surrogate's outputs, and reaches the controls only through the dynamics)
    or genuinely inert. Those look identical in the sweep and are opposite
    findings.
    """
    frac_zero: float
    frac_dead: float
    reads_outputs: bool


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
        """Holding this control anywhere in its box does not move the term,
        through the model. For metered power that is a model whose meters do
        not see this control; `PowerDiagnostics.reads_outputs` separates it
        from a term that reads nothing at all."""
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
def power_gradient(term_fn, out: dict, hmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(power, |∂power/∂yn| summed over channels) over the scored horizon.

    Both are [batch, time]. The gradient is taken w.r.t. a fresh leaf copy of
    `yn`, so this measures what the power term itself passes back to the
    surrogate's outputs — the only place a clamp can silently cut the chain.

    A term that does not read `yn` at all has no gradient to take, and gets an
    all-zero one. That is the honest answer rather than an error: whether the
    term reads the rollout is exactly what the caller is asking.
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
    """How often the power signal reads zero, how often that costs a gradient,
    and whether the term reads the surrogate's outputs at all.

    A metered `power_metric` ends in `.clamp_min(0.0)`, and wherever the
    surrogate predicts negative power that clamp zeroes the gradient; a
    proxy written with a `relu` zeroes it wherever the equipment coasts. For
    a mean energy term that is a rounding detail. For a cost term it is not:
    the near-zero-power steps are disproportionately the cheap hours the
    controller is supposed to be moving load *into*, so a large dead fraction
    means the cost gradient is missing exactly where it was needed.

    The two numbers are separate because a zero power reading is not by itself
    a defect, and reporting only the forward value gets one whole class of head
    exactly backwards. A regression head that predicts −101 W is clamped to
    zero and stops passing gradient; an ordinal head that snaps to a real
    compressor-off stage reports zero and *keeps* passing gradient, because
    the clamp is inactive at exactly zero and the snap is a straight-through
    estimator. The first is a surrogate problem worth acting on; the second is
    the model being right. Measuring the gradient tells them apart without the
    diagnostic needing to know what heads exist.

    Reported, not judged — and a large `frac_dead` is a clamp problem (a power
    head that goes negative, or a hard clamp in a proxy), not a weighting
    problem.
    """
    power, grad = power_gradient(term_fn, out, hmax)
    is_zero = power == 0
    return PowerDiagnostics(
        frac_zero=float(is_zero.float().mean()),
        frac_dead=float((is_zero & (grad == 0)).float().mean()),
        reads_outputs=bool((grad != 0).any()),
    )


def median_change_gap(frames: list, column: str) -> Optional[float]:
    """The median number of steps between changes of `column`, over `frames`
    (each contiguous), or None if it never changes."""
    gaps = []
    for frame in frames:
        values = frame[column].to_numpy()
        changes = [i for i in range(1, len(values)) if values[i] != values[i - 1]]
        gaps += [b - a for a, b in zip(changes, changes[1:])]
    if not gaps:
        return None
    gaps.sort()
    return float(gaps[len(gaps) // 2])


def _preview_notes(objective: Objective, cfg, frames: Optional[list]) -> list[str]:
    """A forecast the objective reads, previewed too short to act on.

    Two ways, both measured rather than assumed: a preview shorter than the
    typical gap between the column's changes in the data, so the next change
    is usually out of sight; and a training rollout shorter than the preview,
    so what the policy sees past the rollout's end is never scored and never
    gives it a gradient.
    """
    read = []
    for term in objective.terms:
        if term.gain <= 0:
            continue
        for name in (term.column, term.target, term.weight):
            if name and objective.roles.get(name) in ("R", "D") and name not in read:
                read.append(name)
    notes = []
    longest = 0
    for name in read:
        future = cfg.policy_window.get(name, {}).get("future", 0)
        longest = max(longest, future)
        gap = median_change_gap(frames, name) if frames else None
        if gap is not None and future < gap:
            notes.append(
                f"`{name}` changes every {gap:.0f} steps (median, training data) and the "
                f"policy previews {future}, so it usually cannot see the next change "
                "coming. To prepare for a change, preview it: a `future` in "
                "`policy_window` at least that long.")
    if longest > cfg.train_pred_steps:
        notes.append(
            f"The policy previews up to {longest} steps but trains on rollouts of "
            f"{cfg.train_pred_steps}: nothing past a rollout's end is scored, so what it "
            "sees beyond that never gives it a gradient. Make `train_pred_steps` at "
            "least the preview, so one rollout holds an action and its payoff.")
    return notes


@beartype
def objective_notes(objective: Objective, cfg, frames: Optional[list] = None) -> list[str]:
    """Objective diagnostics that need the config (and, for the preview, the
    training frames), not a forward pass.

    All the same kind of failure: a term that is switched on but cannot do
    the thing it was switched on for. Reported as prose rather than raised,
    because every one of these configurations is legal and occasionally
    deliberate.
    """
    notes = _preview_notes(objective, cfg, frames)
    for term in objective.terms:
        if (term.gain > 0 and term.weight is not None
                and objective.roles[term.weight] == "R"
                and cfg.policy_window.get(term.weight, {}).get("future", 0) == 0):
            notes.append(
                f"`{term.name}` is weighted by `{term.weight}`, but the policy "
                "previews 0 future steps of it, so it can react to the current "
                "value and cannot act ahead of it — shifting load into a "
                "cheaper hour needs its `future` in `policy_window` to cover the cycle "
                "(48 steps = 4 h on 5-minute data)."
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


@contextmanager
def _pinned(module: torch.nn.Module, column: int, value: float):
    """While open, every command `module` emits has `column` held at `value`
    (normalized), after anything the module does to its output."""
    def hook(_module, _inputs, output):
        pinned = output.clone()
        pinned[..., column] = value
        return pinned

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@beartype
def term_shapes(
    term_fn,
    problem,
    policy_node,
    batch: dict,
    hmax: int,
    u_names: list[str],
    u_min: torch.Tensor,
    u_max: torch.Tensor,
    norm_stats: NormStats,
    n_grid: int = SHAPE_GRID,
) -> list[TermShape]:
    """Sweep each control across its actuation box, through the model.

    Args:
        term_fn: `(yn, U, D) -> [batch, time]`, already bound to nm_mapping and
            norm_stats — the same callable the loss evaluates.
        problem: the DPC problem; its closed loop is re-run per grid point.
        policy_node: the policy Node inside it, whose command is pinned.
        batch:   the batch the balance report used.
        hmax:    warmup steps to skip, so the sweep covers the scored horizon.
        u_min/u_max: actuation box in NORMALIZED space, one entry per control.

    At each grid point the rollout runs with this control held at that value
    for every step and the others as the (untrained) policy commands them, so
    the term is read where the model takes the state, not with the state held
    at a rollout that never saw the command. A metered power the model
    predicts then moves with the command, as it will in training.
    """
    device = u_min.device
    shapes = []
    for j, name in enumerate(u_names):
        s = norm_stats[name]
        grid_n = torch.linspace(float(u_min[j]), float(u_max[j]), n_grid, device=device)
        values = []
        for point in grid_n:
            with _pinned(policy_node.callable, j, float(point)):
                out = run_objective_pass(problem, batch)
            with torch.no_grad():
                values.append(float(term_fn(*(out[k][:, scored_steps(k, hmax), :]
                                              for k in ("yn", "U", "D"))).mean()))
        shapes.append(TermShape(
            u_name=name,
            grid=[float(p) * s["stretch"] + s["shift"] for p in grid_n],
            values=values,
            argmin=min(range(n_grid), key=values.__getitem__),
        ))
    return shapes


# ── Rendering ────────────────────────────────────────────────────


@beartype
def balance_to_markdown(balance: ObjectiveBalance,
                        trained: Optional[ObjectiveBalance] = None) -> str:
    """One row per term. With `trained` (the same batch through the trained
    policy), its value and share sit beside the pre-training ones."""
    after = {} if trained is None else {t.name: t for t in trained.terms}
    head = "| Term | Value | Share | vs. the rest | |"
    rule = "|---|---:|---:|---:|---|"
    if trained is not None:
        head += " Value, trained | Share, trained |"
        rule += "---:|---:|"
    rows = [head, rule]
    for t in balance.terms:
        ratio = "∞" if t.ratio_to_rest == float("inf") else f"{t.ratio_to_rest:.2f}×"
        flag = "⚠️ dominates" if t.dominates() else ""
        row = f"| {t.name} | {t.value:.4g} | {t.share * 100:.0f}% | {ratio} | {flag} |"
        if trained is not None:
            a = after.get(t.name)
            row += (" n/a | n/a |" if a is None
                    else f" {a.value:.4g} | {a.share * 100:.0f}% |")
        rows.append(row)
    return "\n".join(rows)


@beartype
def flat_sweep_reading(reads_outputs: Optional[bool]) -> str:
    """What a flat row means, given whether the term reads the surrogate.

    Shared by both renderers so the markdown and the stdout report cannot
    drift into saying different things about the same measurement.
    """
    if reads_outputs is None:
        return "flat in U"
    if reads_outputs:
        return "flat in U — the model's outputs it reads do not respond to this control"
    return "INERT — reads neither U nor model outputs"


@beartype
def is_inert(shapes: list[TermShape],
             power: Optional[PowerDiagnostics]) -> bool:
    """The power term cannot move the policy: flat in every control *and* no
    gradient into the surrogate's outputs.

    Both halves are needed. Flat-in-U alone is the normal reading for metered
    power, and no-gradient-into-yn alone is normal for a term computed from
    the commands.
    """
    if power is None or not shapes:
        return False
    return not power.reads_outputs and all(s.is_flat for s in shapes)


@beartype
def shapes_to_markdown(shapes: list[TermShape],
                       reads_outputs: Optional[bool] = None) -> str:
    rows = ["| Control | Term at lo | at mid | at hi | Minimised at |",
            "|---|---:|---:|---:|---:|"]
    for s in shapes:
        mid = len(s.values) // 2
        where = (flat_sweep_reading(reads_outputs) if s.is_flat
                 else f"{s.minimised_at:.4g}")
        rows.append(
            f"| {s.u_name} | {s.values[0]:.4g} | {s.values[mid]:.4g} | "
            f"{s.values[-1]:.4g} | {where} |"
        )
    return "\n".join(rows)


@beartype
def objective_to_dict(balance: ObjectiveBalance,
                      shapes: list[TermShape],
                      power: Optional[PowerDiagnostics] = None,
                      notes: Optional[list[str]] = None,
                      trained: Optional[ObjectiveBalance] = None) -> dict:
    """JSON-serializable objective diagnostics (for MLflow / disk)."""
    def terms(b: ObjectiveBalance) -> dict:
        return {t.name: {"value": t.value, "share": t.share,
                         "ratio_to_rest": t.ratio_to_rest,
                         "dominates": t.dominates()}
                for t in b.terms}

    return {
        "total": balance.total,
        "terms": terms(balance),
        "trained_total": None if trained is None else trained.total,
        "trained_terms": None if trained is None else terms(trained),
        "power_shape": {
            s.u_name: {"grid": s.grid, "values": s.values,
                       "flat": s.is_flat,
                       "minimised_at": s.minimised_at}
            for s in shapes
        },
        "frac_power_zero": None if power is None else power.frac_zero,
        "frac_power_dead": None if power is None else power.frac_dead,
        "power_reads_outputs": None if power is None else power.reads_outputs,
        "notes": list(notes or []),
    }


@beartype
def format_objective_report(balance: ObjectiveBalance,
                            shapes: list[TermShape],
                            power: Optional[PowerDiagnostics] = None,
                            notes: Optional[list[str]] = None) -> str:
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
        reads = None if power is None else power.reads_outputs
        lines.append("Power term across each actuation box (no verdict "
                     "attached — read it against the weights above):")
        for s in shapes:
            where = (flat_sweep_reading(reads) if s.is_flat
                     else f"minimised at {s.minimised_at:.4g}")
            lines.append(f"  {s.u_name:<22s} {s.values[0]:>12.4g} .. "
                         f"{s.values[-1]:<12.4g} {where}")
        if is_inert(shapes, power):
            lines.append(
                "  WARNING: the power term reads neither the commands nor the "
                "surrogate's outputs. Whatever its weight, it cannot move the "
                "policy — check `power_metric` against nm_mapping."
            )
    if power is not None:
        lines.append(f"  power exactly zero on {power.frac_zero * 100:.1f}% of "
                     "scored steps")
        lines.append(f"  ...of which {power.frac_dead * 100:.1f}% of all steps "
                     "also pass no gradient back (a clamp on the power path: "
                     "a negative power head, or relu/clamp_min in power_metric)")
    for note in notes or []:
        lines.append(f"  NOTE: {note}")
    return "\n".join(lines)
