"""
Rebuild a trained dynamics model from its MLflow run.

`load_trained_dynamics` is the public entry point: `nt model load`, the
control side and the exporter all load a surrogate through it.
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import mlflow
import torch

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.common.config import ConfigError, dict_to_dataclass, _parse_value
from neurotemplate.common.tracking import CHECKPOINT
from neurotemplate.modeling.experiment import resolve_device
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.hooks import MODELING_SNAPSHOT, run_hooks_path

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

    cfg.device is resolved here so the returned system is on a valid device,
    and cfg.user_hooks_path points at the hooks file the run trained with.
    """
    artifacts_dir = resolve_artifacts_dir(run_dir, run_id)
    cfg = config_from_run_params(run_id)
    cfg.user_hooks_path = run_hooks_path(artifacts_dir, MODELING_SNAPSHOT)
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
