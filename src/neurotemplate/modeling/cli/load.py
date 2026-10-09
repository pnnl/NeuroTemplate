"""
Load a trained model from an MLflow run, evaluate it on a dataset the caller
points at, and optionally plot. Wrapper over pipeline.evaluate_and_plot.
"""

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from time import perf_counter as tic
from typing import Optional
from urllib.parse import urlparse

import mlflow
import torch

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.common.config import ConfigError, dict_to_dataclass, _parse_value
from neurotemplate.common.tracking import CHECKPOINT
from neurotemplate.common.data_utils import get_grad_masks, load_split, normalize_df
from neurotemplate.modeling.experiment import resolve_device, set_seeds
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


def config_from_run_params(run_id: str, cls: type = ModelingConfig):
    """This run's params as a `cls` config.

    `cls` is ModelingConfig for a sys-id run and ControlConfig for a control
    run; both are logged the same way, as the config's fields. Strict: a run
    trained by a version whose fields this one has dropped fails here, naming
    them, rather than reloading without settings it was trained with.
    """
    params = _load_run_params(run_id)
    stale = sorted(set(params) - {f.name for f in fields(cls)})
    if stale:
        raise ConfigError(
            f"Run {run_id} logged param(s) this version of neurotemplate no "
            f"longer defines: {', '.join(stale)}. It was trained by an older "
            "version; retrain it."
        )
    return dict_to_dataclass(params, cls)


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
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing from the run's artifacts ({path.parent}); "
            "every run `nt model train` finishes writes it."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_block_spec_artifact(artifacts_dir: Path) -> Optional[dict]:
    """The run's `block_spec.json` artifact, or None.

    Absent entirely for the single-block runs that are the common case; a
    per-block run always writes one, so absence is an answer rather than a gap.
    """
    block_spec_path = artifacts_dir / "block_spec.json"
    return _load_json(block_spec_path) if block_spec_path.exists() else None


#: Training saves the whole Problem's state_dict, whose first node is the
#: dynamics System; this prefix is that node's place in it.
_PROBLEM_NODE_PREFIX = "nodes.0."


def _load_dyn_sys(cfg, nm_mapping, head_stack, block_spec, norm_stats,
                  weights_path: Path):
    """Build the dynamics system and load its weights from a training checkpoint.

    The checkpoint is `best_model_state_dict.pth` as training writes it: the
    Problem's state_dict. The System's own keys are those under the Problem's
    first node, and they are loaded strictly, so a rebuilt topology that does
    not match the trained one fails here, naming the keys.
    """
    dyn_sys = get_dynamics(
        cfg, nm_mapping, head_stack=head_stack, block_spec=block_spec,
        norm_stats=norm_stats,
    ).to(cfg.device)
    problem_state = torch.load(weights_path, map_location=cfg.device)
    if not all(k.startswith(_PROBLEM_NODE_PREFIX) for k in problem_state):
        raise ValueError(
            f"{weights_path} is not a training checkpoint: expected the "
            f"Problem state_dict `nt model train` saves, with every key under "
            f"{_PROBLEM_NODE_PREFIX!r}."
        )
    dyn_sys.load_state_dict({k[len(_PROBLEM_NODE_PREFIX):]: v
                             for k, v in problem_state.items()})
    dyn_sys.eval()
    return dyn_sys


def load_trained_dynamics(
    run_id: str,
    run_dir: Optional[str] = None,
    weights: Optional[str] = None,
):
    """
    Load a trained dynamics SystemPreview from an MLflow run.

    Public entry point, reused by the control side and the exporter.
    Returns (dyn_sys, cfg, nm_mapping, norm_stats). The caller is responsible
    for freezing / eval-mode if it wants the dynamics held fixed.

    `weights` overrides the run's own `best_model_state_dict.pth`; it must be
    a checkpoint in the same format.

    cfg.device is resolved here so the returned system is on a valid device.
    """
    artifacts_dir = _resolve_artifacts_dir(run_dir, run_id)
    cfg = config_from_run_params(run_id)
    cfg.validate()
    cfg.device = resolve_device(cfg.device)

    norm_stats = _load_json(artifacts_dir / "norm_stats.json")
    nm_mapping = _load_json(artifacts_dir / "nm_mapping.json")

    weights_path = (Path(weights).expanduser() if weights is not None
                    else artifacts_dir / CHECKPOINT)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    # The head topology and calibrated thresholds as trained, independent of
    # whatever the hooks file says now.
    head_stack = HeadStack.from_config(
        _load_json(artifacts_dir / "head_config.json"), nm_mapping["Y"])
    block_spec = _resolve_block_spec_artifact(artifacts_dir)
    dyn_sys = _load_dyn_sys(cfg, nm_mapping, head_stack, block_spec, norm_stats,
                            weights_path)
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
    raw_dfs, freq_s = load_split(data_paths, "--data")
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
    test_results, raw_dfs, _freq_s = _eval_one_split(
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
