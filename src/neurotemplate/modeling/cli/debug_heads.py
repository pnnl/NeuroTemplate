"""
Head diagnostics: plot activation and gradient curves of a trained model's
output heads.

Loads a run via modeling/load.load_trained_dynamics, then probes the
head_stack directly — no dataset, no rollout, just the heads themselves.

Useful for catching:
  - Ordinal thresholds that have collapsed onto each other (the head can no
    longer resolve the corresponding levels).
  - Ordinal thresholds that drifted to one extreme (head effectively
    behaves as fewer than K levels).
  - Bounded heads whose β is so loose the activation is near-linear over
    the visible z range (no real bound enforcement), or so tight that the
    gradient saturates almost everywhere.

Usage:
    nt model debug-heads --run_id <RUN_ID>
    nt model debug-heads --run_id <RUN_ID> --output /tmp/heads.png
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from neurotemplate.modeling.contract import contract_model
from neurotemplate.modeling.load import load_trained_dynamics
from neurotemplate.modeling.plotting import PLOT_STYLE

# Force-disable TeX rendering: titles contain Unicode (β, ∂, σ) and channel
# names contain underscores/brackets that LaTeX would have to be escaped to
# accept. mathtext handles all of it natively.
plt.rcParams["text.usetex"] = False


def _grad_curve(fn, zs: torch.Tensor) -> torch.Tensor:
    """∂fn/∂z evaluated over a 1-D z grid using autograd."""
    z = zs.clone().detach().requires_grad_(True).unsqueeze(-1)
    y = fn(z)
    return torch.autograd.grad(y.sum(), z)[0].squeeze(-1).detach()


def _bounded_z_range(head) -> torch.Tensor:
    """z range that exposes the full sigmoid/softplus transition with margin."""
    if math.isfinite(head.lo) and math.isfinite(head.hi):
        # σ(βz/(hi-lo)) saturates by |βz/(hi-lo)| ≈ 6.
        z_max = 4 * (head.hi - head.lo) / head.beta
        return torch.linspace(-z_max, z_max, 300)
    if math.isfinite(head.lo):
        # softplus(βz): near-zero by βz ≈ -6, near-linear by βz ≈ 6.
        return torch.linspace(-10 / head.beta, 30 / head.beta, 400)
    return torch.linspace(-30 / head.beta, 10 / head.beta, 400)


def _plot_bounded(head, axes, name):
    ax_act, ax_grad = axes
    zs = _bounded_z_range(head)
    with torch.no_grad():
        ys = head.activate(zs.unsqueeze(-1))
    gs = _grad_curve(head.activate, zs)

    ax_act.plot(zs.numpy(), ys.numpy(), label="activate(z)")
    if math.isfinite(head.lo):
        ax_act.axhline(head.lo, color="red", linestyle="--", alpha=0.5,
                       label=f"lo={head.lo:g}")
    if math.isfinite(head.hi):
        ax_act.axhline(head.hi, color="red", linestyle="--", alpha=0.5,
                       label=f"hi={head.hi:g}")
    ax_act.set_title(f"{name}\n{head.kind} β={head.beta:g}", fontsize=9)
    ax_act.set_xlabel("z (head-space)")
    ax_act.set_ylabel("y")
    ax_act.legend(loc="best", fontsize=8)
    ax_act.grid(alpha=0.3)

    ax_grad.plot(zs.numpy(), gs.numpy())
    ax_grad.set_title(f"{name}: ∂activate/∂z (max={gs.abs().max().item():.4g})",
                      fontsize=9)
    ax_grad.set_xlabel("z (head-space)")
    ax_grad.set_ylabel("∂y/∂z")
    ax_grad.grid(alpha=0.3)


def _plot_ordinal(head, axes, name):
    ax_act, ax_grad = axes
    thr = head.thresholds
    thr_min, thr_max = thr.min().item(), thr.max().item()
    zs = torch.linspace(thr_min - 5.0, thr_max + 5.0, 600)
    with torch.no_grad():
        ys = head.activate(zs.unsqueeze(-1))
    gs = _grad_curve(head.activate, zs)

    ax_act.plot(zs.numpy(), ys.numpy(), label="activate(z)")
    for v in head.level_values.tolist():
        ax_act.axhline(v, color="green", linestyle=":", alpha=0.4)
    for t in thr.tolist():
        ax_act.axvline(t, color="red", linestyle="--", alpha=0.4)
    # Where decode actually snaps — after calibration these sit away from
    # the activation thresholds, usually toward the common level.
    for t in head.decision_thresholds.tolist():
        ax_act.axvline(t, color="purple", linestyle="-.", alpha=0.6)
    levels_s = ", ".join(f"{v:g}" for v in head.level_values.tolist())
    thr_s = ", ".join(f"{t:.3f}" for t in thr.tolist())
    dec_s = ", ".join(f"{t:.3f}" for t in head.decision_thresholds.tolist())
    ax_act.set_title(
        f"{name}\nordinal K={head.n_levels} β={head.beta:g}\n"
        f"levels=[{levels_s}]\nthresholds=[{thr_s}]  decide at=[{dec_s}]",
        fontsize=8,
    )
    ax_act.set_xlabel("z (head-space)")
    ax_act.set_ylabel("y")
    ax_act.legend(loc="best", fontsize=8)
    ax_act.grid(alpha=0.3)

    ax_grad.plot(zs.numpy(), gs.numpy())
    for t in thr.tolist():
        ax_grad.axvline(t, color="red", linestyle="--", alpha=0.4)
    ax_grad.set_title(f"{name}: ∂activate/∂z (max={gs.abs().max().item():.4g})",
                      fontsize=9)
    ax_grad.set_xlabel("z (head-space)")
    ax_grad.set_ylabel("∂y/∂z")
    ax_grad.grid(alpha=0.3)


def _plot_binary(head, axes, name):
    ax_act, ax_grad = axes
    zs = torch.linspace(-3.0, 3.0, 300)
    with torch.no_grad():
        ys = head.activate(zs.unsqueeze(-1))
    gs = _grad_curve(head.activate, zs)

    ax_act.plot(zs.numpy(), ys.numpy(), label="activate(z) = σ(z)")
    ax_act.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax_act.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax_act.set_title(f"{name}\nbinary (sigmoid)", fontsize=9)
    ax_act.set_xlabel("z")
    ax_act.set_ylabel("p(y=1)")
    ax_act.legend(loc="best", fontsize=8)
    ax_act.grid(alpha=0.3)

    ax_grad.plot(zs.numpy(), gs.numpy())
    ax_grad.set_title(f"{name}: ∂activate/∂z", fontsize=9)
    ax_grad.set_xlabel("z")
    ax_grad.set_ylabel("∂y/∂z")
    ax_grad.grid(alpha=0.3)


_PLOTTERS = {
    "binary": _plot_binary,
    "bounded": _plot_bounded,
    "ordinal": _plot_ordinal,
}


def _print_summary(head_stack):
    print(f"\n{'channel':50s} {'kind':18s} details")
    print("-" * 120)
    for name, head in zip(head_stack.channel_names, head_stack.heads):
        if head.kind == "regression":
            details = "identity"
        elif head.kind == "binary":
            w_s = ", ".join(f"{w:.2f}" for w in head.level_weights.tolist())
            details = f"sigmoid level_weights=[{w_s}]"
        elif head.kind == "bounded":
            details = f"lo={head.lo:g} hi={head.hi:g} β={head.beta:g}"
        elif head.kind == "ordinal":
            thr = head.thresholds.tolist()
            gaps = [thr[i + 1] - thr[i] for i in range(len(thr) - 1)]
            gap_s = f"min_gap={min(gaps):.3f}" if gaps else "single-threshold"
            thr_s = ", ".join(f"{t:.3f}" for t in thr)
            dec_s = ", ".join(f"{t:.3f}" for t in head.decision_thresholds.tolist())
            w_s = ", ".join(f"{w:.2f}" for w in head.level_weights.tolist())
            details = (
                f"K={head.n_levels} β={head.beta:g} "
                f"thresholds=[{thr_s}] {gap_s} "
                f"decide_at=[{dec_s}] level_weights=[{w_s}]"
            )
        else:
            details = "?"
        print(f"{name:50s} {head.kind:18s} {details}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot activation/gradient curves of a trained model's heads."
    )
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Optional override for the artifacts directory.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Optional explicit weights path.")
    parser.add_argument("--output", type=str, default="head_debug.png",
                        help="Output PNG path.")
    args = parser.parse_args()

    dyn_sys, _cfg, _nm_mapping, _norm_stats = load_trained_dynamics(
        run_id=args.run_id, run_dir=args.run_dir, weights=args.weights,
    )
    # In the model's own (normalised) units: the activation curves are drawn
    # against the raw output z the family emits.
    head_stack = getattr(contract_model(dyn_sys), "head_stack", None)
    if head_stack is None:
        print("Model has no head_stack — nothing to plot.")
        return

    _print_summary(head_stack)

    plottable = [
        (name, head)
        for name, head in zip(head_stack.channel_names, head_stack.heads)
        if head.kind != "regression"
    ]
    if not plottable:
        print("No non-regression heads to plot.")
        return

    n = len(plottable)
    out = Path(args.output)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(n, 2, figsize=(13, 3.6 * n), squeeze=False)
        for row, (name, head) in enumerate(plottable):
            _PLOTTERS[head.kind](head, axes[row], name)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
