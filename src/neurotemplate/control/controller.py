"""
What a control run commands with: a trained policy (`control.policy`) or MPC
through the surrogate (`control.mpc`).

Either is a node of the closed loop that reads the policy's windows of `yn`,
`U`, `D` and `R`, physical units, and writes `U`, physical units. Anything
honouring that can stand in the loop, be scored, and be read against the best
plan (`control.plan`).
"""

from __future__ import annotations

from neuromancer.system import Node

from neurotemplate._typing import beartype
from neurotemplate.common.data_utils import NormStats
from neurotemplate.control.mpc import MPC, build_mpc
from neurotemplate.control.policy import build_policy, describe_policy


@beartype
def build_controller(cfg, sysid_cfg, nm_mapping: dict[str, list[str]],
                     norm_stats: NormStats, dyn_node: Node) -> Node:
    if cfg.policy == "mpc":
        return build_mpc(cfg, sysid_cfg, nm_mapping, norm_stats, dyn_node)
    return build_policy(cfg, sysid_cfg, nm_mapping, norm_stats)


@beartype
def describe_controller(node: Node, cfg, sysid_cfg, nm_mapping: dict[str, list[str]]) -> dict:
    """`describe_policy`, or what MPC is, for the reportcard."""
    if not isinstance(node.callable, MPC):
        return describe_policy(node, cfg, sysid_cfg, nm_mapping)
    mpc = node.callable
    return {
        "family": "mpc",
        "horizon": mpc.horizon,
        "lr": mpc.lr,
        "rtol": mpc.rtol,
        "patience": mpc.patience,
        "max_iters": mpc.max_iters,
        "inputs": list(node.input_keys),
        "u_cols": list(nm_mapping["U"]),
        "umin_phys": mpc.umin.tolist(),
        "umax_phys": mpc.umax.tolist(),
        "command_bounds": dict(cfg.command_bounds) or None,
    }
