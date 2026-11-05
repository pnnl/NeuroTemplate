"""
Rebuild a trained dynamics model from its MLflow run, and score it on data.

`load_trained_dynamics` is the public entry point: `nt model load`, the
control side and the exporter all load a surrogate through it.
`evaluate_run` scores a run on new data — `nt model load` without the
command line.
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import mlflow
import torch

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.common.config import ConfigError, dict_to_dataclass, _parse_value
from neurotemplate.common.data_utils import check_complete, load_split
from neurotemplate.common.tracking import CHECKPOINT
from neurotemplate.modeling.experiment import resolve_device, set_seeds
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.pipeline import TestResults, evaluate_and_plot
from neurotemplate.metrics.physics import SynthSpec
from neurotemplate.hooks import MODELING_SNAPSHOT, load_modeling_hooks, run_hooks_path

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
def resolve_artifacts_dir(run_dir: Optional[str], run_id: str) -> Path:
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
        cfg, nm_mapping, norm_stats, head_stack=head_stack, block_spec=block_spec,
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
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Load a trained dynamics SystemPreview from an MLflow run.

    Public entry point, reused by the control side and the exporter.
    Returns (dyn_sys, cfg, nm_mapping, norm_stats). The caller is responsible
    for freezing / eval-mode if it wants the dynamics held fixed.

    `weights` overrides the run's own `best_model_state_dict.pth`; it must be
    a checkpoint in the same format.

    `device` is where this process runs the model: `resolve_device`'s
    choice when None, as for training. The device the run logged is where it
    trained, which says nothing about this machine, so it is never read —
    and resolving it would also reset torch's default device under a caller
    that had already chosen one. cfg.device is set to the resolved device,
    and cfg.user_hooks_path points at the hooks file the run trained with.
    """
    artifacts_dir = resolve_artifacts_dir(run_dir, run_id)
    cfg = config_from_run_params(run_id)
    cfg.user_hooks_path = run_hooks_path(artifacts_dir, MODELING_SNAPSHOT)
    cfg.validate()
    cfg.device = resolve_device(device)

    norm_stats = _load_json(artifacts_dir / "norm_stats.json")
    nm_mapping = _load_json(artifacts_dir / "nm_mapping.json")

    weights_path = (Path(weights).expanduser() if weights is not None
                    else artifacts_dir / CHECKPOINT)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    # The head topology as trained, independent of whatever the hooks file says now.
    head_stack = HeadStack.from_config(
        _load_json(artifacts_dir / "head_config.json"), nm_mapping["Y"])
    block_spec = _resolve_block_spec_artifact(artifacts_dir)
    dyn_sys = _load_dyn_sys(cfg, nm_mapping, head_stack, block_spec, norm_stats,
                            weights_path)
    return dyn_sys, cfg, nm_mapping, norm_stats


def evaluate_trained(cfg, dyn_sys, data_paths: list[str], norm_stats: dict,
                     nm_mapping: dict[str, list[str]],
                     synth_cfg: Optional[SynthSpec],
                     plot_dir: Optional[Path]):
    """Evaluate a loaded model on `data_paths`: one split, one or more CSVs.

    The model normalises with the run's own `norm_stats`, never stats fitted
    to this data: it only reads inputs in the scaling it trained under.

    Returns (test_results, raw_dfs). `raw_dfs` is always a list (length >= 1)
    of dataframes — callers needing a row count sum lengths across the list.
    """
    raw_dfs, freq_s = load_split(data_paths, "--data")
    check_complete(raw_dfs, [c for role in ("Y", "U", "D") for c in nm_mapping[role]], "--data")
    test_results = evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys,
        test_dfs=raw_dfs,
        nm_mapping=nm_mapping, norm_stats=norm_stats,
        synth_cfg=synth_cfg, freq_s=freq_s,
        plot_run_dir=plot_dir,
    )
    return test_results, raw_dfs


@beartype
def evaluate_run(
    run_id: str,
    data_paths: list[str],
    *,
    run_dir: Optional[str] = None,
    weights: Optional[str] = None,
    physics_checks: bool = True,
    plot_dir: Optional[Path] = None,
) -> TestResults:
    """Score a trained run on `data_paths` — e.g. yesterday's model on
    today's test split — with the metrics its training run reported.

    `physics_checks=False` skips the hooks' synthetic scenarios. Plots are
    written to `plot_dir` when one is given, and nothing is written when not.
    `run_dir` and `weights` are `load_trained_dynamics`'s.
    """
    dyn_sys, cfg, nm_mapping, norm_stats = load_trained_dynamics(
        run_id, run_dir=run_dir, weights=weights)
    set_seeds(cfg.seed)
    hooks = load_modeling_hooks(cfg)
    synth_cfg = (hooks.synth_cfg(nm_mapping, norm_stats)
                 if physics_checks and hasattr(hooks, "synth_cfg") else None)
    cfg.plot = plot_dir is not None
    test_results, _ = evaluate_trained(cfg, dyn_sys, data_paths, norm_stats,
                                       nm_mapping, synth_cfg, plot_dir)
    return test_results
