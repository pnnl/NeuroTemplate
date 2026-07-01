"""
Load a frozen sys-id surrogate for closed-loop control.

Thin wrapper over the public src/load_model.load_trained_dynamics loader.
Freezing is explicit here: the dynamics is held fixed while the policy trains
(the canonical Neuromancer "freeze identified dynamics, train policy" pattern).
"""

from __future__ import annotations


from neurotemplate.modeling.cli.load import load_trained_dynamics

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
