"""
Load a trained model from an MLflow run, evaluate it on a dataset the caller
points at, and optionally plot. Python callers use modeling.load.evaluate_run.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter as tic

from neurotemplate.common.data_utils import get_grad_masks
from neurotemplate.modeling.experiment import set_seeds
from neurotemplate.modeling.gradient_sampling import SampledGradientPenalty
from neurotemplate.modeling.load import evaluate_trained, load_trained_dynamics
from neurotemplate.modeling.pipeline import render_reportcard, write_reportcard
from neurotemplate.hooks import load_modeling_hooks

from neurotemplate._typing import beartype


def _eval_grad_penalty(dyn_sys, cfg, grad_masks):
    """Compute the raw (gain=1) gradient sign penalty."""
    raw_model = dyn_sys.nodes[0].callable
    penalty = SampledGradientPenalty(
        model=raw_model,
        grad_mapping=grad_masks,
        n_samples=cfg.grad_n_samples,
        device=cfg.device,
        cfg=cfg,
    )
    return penalty.compute().item()


def _parse_data_arg(raw: str) -> list[str]:
    """Accept a single path, comma-separated paths, or a JSON list.

    The MLflow-stored config round-trips lists already (via
    `_parse_value`'s ast.literal_eval fallback), so this only matters for
    CLI-typed values where shell quoting makes JSON awkward.
    """
    text = raw.strip()
    if text.startswith("["):
        paths = json.loads(text)
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError(f"--data argument must be a list of strings, got {raw!r}")
        return paths
    return [p.strip() for p in text.split(",") if p.strip()]


def get_cli_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load trained model and run it on the pointed-at data."
    )
    parser.add_argument("--run_id", type=str, required=True,
                        help="Run ID to load (not full path).")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Optional override for the artifacts directory.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Optional explicit path to weights.")
    parser.add_argument("--data", type=str,
                        required=True,
                        help="Test CSV(s) to evaluate the model on. Single "
                             "path, comma-separated list, or JSON array.")
    parser.add_argument("--grad_path", type=str, default=None,
                        help="Override grad_path for gradient penalty evaluation.")
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Output directory for plots (errors if it exists).")
    parser.add_argument("--no_synth", action="store_true", default=False,
                        help="Skip synthetic scenario evaluation/plot.")
    return parser.parse_args(argv)


@beartype
def main() -> None:
    args = get_cli_args(sys.argv[1:])

    if args.weights is None and args.run_dir is None:
        print("Loading dynamics & weights from MLflow run...")

    dyn_sys, cfg, nm_mapping, norm_stats = load_trained_dynamics(
        run_id=args.run_id, run_dir=args.run_dir, weights=args.weights,
    )

    set_seeds(cfg.seed)

    if cfg.verbose:
        print(f"Using device: {cfg.device}")

    grad_path_for_eval = args.grad_path or cfg.grad_path
    grad_masks = get_grad_masks(grad_path=grad_path_for_eval, nm_mapping=nm_mapping)

    # --no_synth → opt out by passing synth_cfg=None;
    # otherwise compute from modeling_user_defined.
    synth_cfg = (
        None if args.no_synth
        else load_modeling_hooks(cfg).synth_cfg(nm_mapping, norm_stats)
    )

    # Plot dir
    if args.plot_dir is not None:
        plot_run_dir = Path(args.plot_dir).expanduser()
        if plot_run_dir.exists():
            raise FileExistsError(f"Plot directory already exists: {plot_run_dir}")
    else:
        plot_run_dir = Path(cfg.plots_dir) / datetime.now().strftime("%b%d_%H_%M")
    plot_run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.verbose:
        print("Evaluating...")

    data_paths = _parse_data_arg(args.data)

    eval_t0 = tic()
    test_results, raw_dfs = evaluate_trained(
        cfg, dyn_sys, data_paths, norm_stats, nm_mapping,
        synth_cfg, plot_run_dir,
    )
    eval_time_s = tic() - eval_t0

    print(f"Normalized RMSE: {test_results.normalized_rmse:.4f}")
    if cfg.verbose:
        print(f"Evaluation time: {eval_time_s:.2f} sec")

    print("\nPer-channel metrics (physical units):")
    for m in test_results.metrics:
        print(f"  {m.name}: RMSE={m.rmse:.4f} R²={m.r2:.4f} NRMSE={m.nrmse:.4f}")
    if test_results.levels:
        persistence = {lm.name: lm for lm in test_results.persistence_levels}
        print("\nStaged channels (balanced accuracy, model vs persistence):")
        for lm in test_results.levels:
            print(f"  {lm.name}: {lm.balanced_accuracy:.3f} vs "
                  f"{persistence[lm.name].balanced_accuracy:.3f}")

    if grad_masks:
        grad_penalty = _eval_grad_penalty(dyn_sys, cfg, grad_masks)
        print(f"\nGrad Penalty (raw): {grad_penalty:.6f}")
    else:
        grad_penalty = None
        print("\nGrad Penalty (raw): N/A (no grad_path)")

    with open(plot_run_dir / "stats.dat", "w", encoding="utf-8") as f:
        f.write(f"rmse {test_results.normalized_rmse:.6f}\n")
        if grad_penalty is not None:
            f.write(f"grad_penalty_raw {grad_penalty:.6f}\n")

    md = render_reportcard(
        cfg=cfg, dyn_sys=dyn_sys, test_results=test_results,
        norm_stats=norm_stats, nm_mapping=nm_mapping,
        head_stack=dyn_sys.nodes[0].callable.head_stack,
        run_name=args.run_id,
        split_sizes={"test": {
            "rows": sum(len(d) for d in raw_dfs), "files": len(raw_dfs),
        }},
    )
    write_reportcard(md, plot_run_dir)


if __name__ == "__main__":
    main()
