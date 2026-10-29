"""
Load a frozen sys-id surrogate for closed-loop control, or a trained policy.

Thin wrapper over modeling/load.load_trained_dynamics.
Freezing is explicit here: the dynamics is held fixed while the policy trains
(the canonical Neuromancer "freeze identified dynamics, train policy" pattern).
"""

from __future__ import annotations


import torch

from neurotemplate.common.tracking import CHECKPOINT
from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem,
)
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.losses import extend_norm_stats, resolve_objective
from neurotemplate.control.controller import build_controller
from neurotemplate.hooks import CONTROL_SNAPSHOT, run_hooks_path
from neurotemplate.modeling.load import (
    config_from_run_params, load_trained_dynamics, resolve_artifacts_dir,
)
from neurotemplate.modeling.experiment import resolve_device

from neurotemplate._typing import beartype


@beartype
def load_frozen_dynamics(
    run_id: str,
    run_dir=None,
    weights=None,
    device=None,
):
    """
    Returns (dyn_sys, dyn_node, sysid_cfg, nm_mapping, norm_stats), on
    `device` (see `load_trained_dynamics`).

    dyn_sys  : the loaded SystemPreview (frozen, eval mode).
    dyn_node : its single inner Node — reused inside the closed-loop System.
    sysid_cfg: the sys-id run's Config (history_steps, model, device, ...).
    """
    dyn_sys, sysid_cfg, nm_mapping, norm_stats = load_trained_dynamics(
        run_id=run_id, run_dir=run_dir, weights=weights, device=device,
    )

    for p in dyn_sys.parameters():
        p.requires_grad_(False)
    dyn_sys.eval()

    if len(dyn_sys.nodes) != 1:
        raise ValueError(
            f"Expected a single-node dynamics System, got {len(dyn_sys.nodes)} "
            "nodes; closed-loop assembly assumes one dynamics Node."
        )
    dyn_node = dyn_sys.nodes[0]

    return dyn_sys, dyn_node, sysid_cfg, nm_mapping, norm_stats


def load_trained_policy(run_id: str, run_dir=None, sysid_run_dir=None, device=None):
    """Rebuild a trained control run's policy and closed loop, weights loaded.

    Returns (policy_node, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats).

    `run_dir` is the control run's directory and `sysid_run_dir` its
    surrogate's, each overriding where MLflow says that run's artifacts are —
    for a run tree that has moved (a container, another machine). Without
    `sysid_run_dir` the surrogate is found through the `run_dir` the control
    run logged. `device` is as for `load_trained_dynamics`: this process's,
    never the one the run logged.

    The closed loop is assembled exactly as `nt control train` assembled it —
    same builders, from the run's own logged config and the copy of the
    hooks file it trained with — and the checkpoint the trainer saved
    (`best_model_state_dict.pth`, the whole Problem) is loaded strictly, so a
    policy that does not rebuild to the trained shape fails here, loudly.
    `norm_stats` carries the reference columns, rebuilt from the logged
    config exactly as training built them.
    """
    cfg = config_from_run_params(run_id, ControlConfig)
    artifacts_dir = resolve_artifacts_dir(run_dir, run_id)
    cfg.user_hooks_path = run_hooks_path(artifacts_dir, CONTROL_SNAPSHOT)
    cfg.validate()
    cfg.device = resolve_device(device)
    _, dyn_node, sysid_cfg, nm_mapping, sysid_stats = load_frozen_dynamics(
        cfg.sysid_run_id, cfg.run_dir if sysid_run_dir is None else sysid_run_dir,
        cfg.weights, device=cfg.device)
    norm_stats = extend_norm_stats(resolve_objective(cfg, nm_mapping), cfg,
                                   sysid_stats)

    policy_node = build_controller(cfg, sysid_cfg, nm_mapping, norm_stats, dyn_node)
    cl_system = build_closed_loop(dyn_node, policy_node, cfg, sysid_cfg)
    problem = build_problem(
        cl_system, build_control_loss(cfg, sysid_cfg, nm_mapping),
    ).to(cfg.device)
    weights = artifacts_dir / CHECKPOINT
    problem.load_state_dict(torch.load(weights, map_location=cfg.device))
    problem.eval()
    return policy_node, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats
