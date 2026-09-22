"""
Load a frozen sys-id surrogate for closed-loop control, or a trained policy.

Thin wrapper over the public src/load_model.load_trained_dynamics loader.
Freezing is explicit here: the dynamics is held fixed while the policy trains
(the canonical Neuromancer "freeze identified dynamics, train policy" pattern).
"""

from __future__ import annotations


import torch

from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem,
)
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.policy import build_policy
from neurotemplate.modeling.cli.load import (
    _resolve_artifacts_dir, config_from_run_params, load_trained_dynamics,
)
from neurotemplate.modeling.experiment import resolve_device

from neurotemplate._typing import beartype


@beartype
def load_frozen_dynamics(
    run_id: str,
    run_dir=None,
    weights=None,
):
    """
    Returns (dyn_sys, dyn_node, sysid_cfg, nm_mapping, norm_stats).

    dyn_sys  : the loaded SystemPreview (frozen, eval mode).
    dyn_node : its single inner Node — reused inside the closed-loop System.
    sysid_cfg: the sys-id run's Config (in_steps, model, device, ...).
    """
    dyn_sys, sysid_cfg, nm_mapping, norm_stats = load_trained_dynamics(
        run_id=run_id, run_dir=run_dir, weights=weights,
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


def load_trained_policy(run_id: str, run_dir=None):
    """Rebuild a trained control run's policy and closed loop, weights loaded.

    Returns (policy_node, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats).

    The closed loop is assembled exactly as `nt control train` assembled it —
    same builders, from the run's own logged config and the use case's
    `control_user_defined` hooks — and the checkpoint the trainer saved
    (`best_model_state_dict.pth`, the whole Problem) is loaded strictly, so a
    policy that does not rebuild to the trained shape fails here, loudly.
    """
    cfg = config_from_run_params(run_id, ControlConfig)
    cfg.validate()
    cfg.device = resolve_device(cfg.device)
    _, dyn_node, sysid_cfg, nm_mapping, norm_stats = load_frozen_dynamics(
        cfg.sysid_run_id, cfg.run_dir, cfg.weights)

    policy_node = build_policy(cfg, sysid_cfg, nm_mapping, norm_stats)
    cl_system = build_closed_loop(dyn_node, policy_node, cfg, sysid_cfg)
    problem = build_problem(
        cl_system, build_control_loss(cfg, sysid_cfg, nm_mapping, norm_stats),
    ).to(cfg.device)
    weights = _resolve_artifacts_dir(run_dir, run_id) / "best_model_state_dict.pth"
    problem.load_state_dict(torch.load(weights, map_location=cfg.device))
    problem.eval()
    return policy_node, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats
