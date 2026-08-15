"""
Load a trained model from an MLflow run, evaluate it on chosen splits,
and optionally plot. Wrapper over pipeline.evaluate_and_plot.
"""

import sys
import json
import warnings
from pathlib import Path
from time import perf_counter as tic
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
import argparse

import mlflow
import numpy as np
import torch

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.common.config import dict_to_dataclass, _parse_value
from neurotemplate.common.data_utils import (
    normalize_df,
    get_data,
    get_grad_masks,
)
from neurotemplate.modeling.experiment import resolve_device
from neurotemplate.modeling.gradient_sampling import SampledGradientPenalty
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.pipeline import evaluate_and_plot, render_reportcard, write_reportcard
from neurotemplate.hooks import load_modeling_hooks

from neurotemplate._typing import beartype


@beartype
def _load_run_params(run_id: str) -> dict[str, object]:
    run = mlflow.get_run(run_id)
    return {k: _parse_value(v) for k, v in run.data.params.items()}


@beartype
def _resolve_artifacts_dir(run_dir: Optional[str], run_id: str) -> Path:
    if run_dir is not None:
        p = Path(run_dir).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Run dir {run_dir} not found.")
        return p / "artifacts"
    uri = mlflow.get_run(run_id).info.artifact_uri
    return Path(urlparse(uri).path)


@beartype
def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


@beartype
def _remap_state_dict_keys(
    state_dict: dict, target_keys: set[str]
) -> Optional[dict]:
    """Strip a Neuromancer System wrapper from a saved state_dict.

    When a SystemPreview is saved, its inner module's parameter keys are
    prefixed with `nodes.<i>.` (we only ever use one node, so `nodes.0.`).
    Loading those weights back into the bare inner module fails unless the
    prefix is stripped — which is what this function does.
    """
    NODE_PREFIX = "nodes.0."
    if all(k.startswith(NODE_PREFIX) for k in state_dict.keys()):
        remapped = {k[len(NODE_PREFIX):]: v for k, v in state_dict.items()}
        if set(remapped.keys()) == target_keys:
            return remapped
    return None


def _resolve_head_stack(
    cfg,
    artifacts_dir: Path,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> HeadStack:
    """Prefer the head_config.json artifact saved at train time.

    Older runs (trained before head config was persisted) don't have the
    artifact; fall back to the live modeling_user_defined.data_head_config
    with a loud warning, since drift there silently changes the loaded
    model's output behaviour.
    """
    head_config_path = artifacts_dir / "head_config.json"
    if head_config_path.exists():
        return HeadStack.from_config(_load_json(head_config_path), nm_mapping["Y"])
    warnings.warn(
        f"No head_config.json in {artifacts_dir}; falling back to "
        "modeling_user_defined.data_head_config(). This is correct only "
        "if modeling_user_defined.py has not changed since the run was trained.",
        stacklevel=2,
    )
    return load_modeling_hooks(cfg).data_head_config(nm_mapping, norm_stats)


def _load_dyn_sys(cfg, nm_mapping, head_stack, zone_spec, weights_path: Path):
    """Build the dynamics system and load weights from disk."""
    dyn_sys = get_dynamics(
        cfg, nm_mapping, head_stack=head_stack, zone_spec=zone_spec,
    ).to(cfg.device)
    state = torch.load(weights_path, map_location=cfg.device)

    if isinstance(state, dict):
        state_dict = (
            state["state_dict"]
            if "state_dict" in state and isinstance(state["state_dict"], dict)
            else state
        )
        try:
            dyn_sys.load_state_dict(state_dict)
        except RuntimeError:
            remapped = _remap_state_dict_keys(
                state_dict, set(dyn_sys.state_dict().keys())
            )
            if remapped is None:
                raise
            dyn_sys.load_state_dict(remapped)
    elif hasattr(state, "nodes"):
        dyn_sys = (
            state.nodes[0].to(cfg.device)
            if hasattr(state.nodes[0], "nodes")
            else state.to(cfg.device)
        )
    elif hasattr(state, "state_dict"):
        dyn_sys = state.to(cfg.device)
    else:
        raise ValueError(
            "Unrecognized weights format; expected state_dict or model object."
        )

    dyn_sys.eval()
    return dyn_sys


def load_trained_dynamics(
    run_id: str,
    run_dir: Optional[str] = None,
    weights: Optional[str] = None,
):
    """
    Load a trained dynamics SystemPreview from an MLflow run.

    Public entry point reused by control_src to obtain a frozen surrogate.
    Returns (dyn_sys, cfg, nm_mapping, norm_stats). The caller is responsible
    for freezing / eval-mode if it wants the dynamics held fixed.

    cfg.device is resolved here so the returned system is on a valid device.
    """
    artifacts_dir = _resolve_artifacts_dir(run_dir, run_id)
    cfg = dict_to_dataclass(_load_run_params(run_id), ModelingConfig)
    
    cfg.validate()
    cfg.device = resolve_device(cfg.device)
    
    norm_stats_path = artifacts_dir / "norm_stats.json"
    nm_mapping_path = artifacts_dir / "nm_mapping.json"
    if not norm_stats_path.exists() or not nm_mapping_path.exists():
        raise FileNotFoundError(
            "Missing norm_stats.json or nm_mapping.json in run artifacts."
        )
    norm_stats = _load_json(norm_stats_path)
    nm_mapping = _load_json(nm_mapping_path)

    if weights is not None:
        weights_path = Path(weights).expanduser()
    else:
        weights_path = artifacts_dir / "best_model_state_dict.pth"
        if not weights_path.exists():
            weights_path = artifacts_dir / "best_model.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    head_stack = _resolve_head_stack(cfg, artifacts_dir, nm_mapping, norm_stats)
    # Absent for the single-zone runs that are the common case; a per-zone run
    # always writes it, so its absence is an answer rather than a gap.
    zone_spec_path = artifacts_dir / "zone_spec.json"
    zone_spec = _load_json(zone_spec_path) if zone_spec_path.exists() else None
    dyn_sys = _load_dyn_sys(cfg, nm_mapping, head_stack, zone_spec, weights_path)
    return dyn_sys, cfg, nm_mapping, norm_stats


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


def _eval_one_split(cfg, dyn_sys, data_paths, norm_stats, nm_mapping,
                    synth_cfg, plot_run_dir):
    """Evaluate + plot one logical split, which may span multiple CSVs.

    Returns (test_results, raw_dfs, freq_s). `raw_dfs` is always a list
    (length >= 1) of unnormalized dataframes — callers needing a row
    count sum lengths across the list.
    """
    raw_dfs: list = []
    freq_s: Optional[float] = None
    for p in data_paths:
        df, f = get_data(p, verbose=False)
        if df.isna().any().any():
            raise ValueError(f"NaNs found in {p}")
        if freq_s is None:
            freq_s = f
        elif f != freq_s:
            raise ValueError(f"Sampling frequency mismatch across split files: {p}")
        raw_dfs.append(df)

    norm_dfs = [normalize_df(df, norm_stats) for df in raw_dfs]
    test_results = evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys,
        test_dfs=norm_dfs,
        nm_mapping=nm_mapping, norm_stats=norm_stats,
        synth_cfg=synth_cfg, freq_s=freq_s,
        plot_run_dir=plot_run_dir,
    )
    return test_results, raw_dfs, freq_s


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
    parser.add_argument("--id_data", type=str,
                        default="",
                        required=True,
                        help="In-distribution test CSV(s). Single path, "
                             "comma-separated list, or JSON array.")
    parser.add_argument("--ood_data", type=str,
                        default="",
                        required=True,
                        help="Out-of-distribution test CSV(s). Single path, "
                             "comma-separated list, or JSON array.")
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

    if args.id_data=="" and args.ood_data=="":
        raise ValueError("Must specify at least one of ID or OOD data.")

    if args.weights is None and args.run_dir is None:
        print("Loading dynamics & weights from MLflow run...")

    dyn_sys, cfg, nm_mapping, norm_stats = load_trained_dynamics(
        run_id=args.run_id, run_dir=args.run_dir, weights=args.weights,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

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

    id_paths = _parse_data_arg(args.id_data)
    ood_paths = _parse_data_arg(args.ood_data)

    eval_t0 = tic()
    id_results, id_raws, id_freq_s = _eval_one_split(
        cfg, dyn_sys, id_paths, norm_stats, nm_mapping,
        synth_cfg, plot_run_dir / "id",
    )
    ood_results, ood_raws, _ = _eval_one_split(
        cfg, dyn_sys, ood_paths, norm_stats, nm_mapping,
        synth_cfg, plot_run_dir / "ood",
    )
    eval_time_s = tic() - eval_t0

    print(f"ID  aggregate RMSE: {id_results.aggregate_rmse:.4f} (normalized space)")
    print(f"OOD aggregate RMSE: {ood_results.aggregate_rmse:.4f} (normalized space)")
    if cfg.verbose:
        print(f"Evaluation time: {eval_time_s:.2f} sec")

    print("\nID per-channel metrics (physical units):")
    for m in id_results.metrics:
        print(f"  {m.name}: RMSE={m.rmse:.4f} R²={m.r2:.4f} NRMSE={m.nrmse:.4f}")
    print("\nOOD per-channel metrics (physical units):")
    for m in ood_results.metrics:
        print(f"  {m.name}: RMSE={m.rmse:.4f} R²={m.r2:.4f} NRMSE={m.nrmse:.4f}")

    if grad_masks:
        grad_penalty = _eval_grad_penalty(dyn_sys, cfg, grad_masks)
        print(f"\nGrad Penalty (raw): {grad_penalty:.6f}")
    else:
        grad_penalty = None
        print("\nGrad Penalty (raw): N/A (no grad_path)")

    with open(plot_run_dir / "stats.dat", "w") as f:
        f.write(f"id_rmse  {id_results.aggregate_rmse:.6f}\n")
        f.write(f"ood_rmse {ood_results.aggregate_rmse:.6f}\n")
        if grad_penalty is not None:
            f.write(f"grad_penalty_raw {grad_penalty:.6f}\n")

    # Reportcards (one per split)
    for split_name, results, raws in [
        ("id", id_results, id_raws),
        ("ood", ood_results, ood_raws),
    ]:
        md = render_reportcard(
            cfg=cfg, dyn_sys=dyn_sys, test_results=results,
            norm_stats=norm_stats, nm_mapping=nm_mapping,
            head_stack=dyn_sys.nodes[0].callable.head_stack,
            run_name=f"{args.run_id} [{split_name}]",
            split_sizes={split_name: {
                "rows": sum(len(d) for d in raws), "files": len(raws),
            }},
        )
        write_reportcard(md, plot_run_dir / split_name)


if __name__ == "__main__":
    main()
