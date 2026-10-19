"""
`model.json`: what a consumer needs to drive the two graphs, and what a
human needs to trust them.

The runtime reads only the contract fields: `format`, `graph_format`,
`export_id`, `opset`, `verified`, the column lists, `history_steps`,
`timestep_seconds` and `state.width`.
Everything under `architecture` is documentation the runtime never reads —
above all the 2R2C time constants from `RCBlock.describe()`, which are the
reason a grey-box model is worth trusting and which a graph hides. Do not
"clean up" that duplication: it is the only place a consumer can read the
physics without the training environment.
"""

from __future__ import annotations

from typing import Optional

from neurotemplate.export.state import StateSlot
from neurotemplate.export.trace import OPSET
from neurotemplate.modeling.dynamics import describe_dynamics

#: Must match `runtime.FORMAT`; the runtime refuses any other value.
FORMAT = 3


def build_spec(*, export_id: str, run_id: str, cfg, dyn_sys,
               nm_mapping: dict[str, list[str]], norm_stats: dict[str, dict],
               timestep_seconds: float, slots: list[StateSlot],
               description: Optional[str], verified: bool,
               graph_format: str) -> dict:
    """The `model.json` document for one export."""
    columns = nm_mapping["Y"] + nm_mapping["U"] + nm_mapping["D"]
    return {
        "format": FORMAT,
        "graph_format": graph_format,
        "export_id": export_id,
        "opset": opset(graph_format),
        # False only when the trained model blew up on the verification data
        # and the user chose to export anyway: the graphs were never checked
        # against it, and no reference/ ships.
        "verified": verified,
        "description": description,
        "y_columns": nm_mapping["Y"],
        "u_columns": nm_mapping["U"],
        "d_columns": nm_mapping["D"],
        "units": {c: norm_stats[c].get("unit") for c in columns},
        "history_steps": int(cfg.history_steps),
        "timestep_seconds": float(timestep_seconds),
        # Frozen at trace time: only the batch axis of the graphs is dynamic.
        "state": {
            "width": sum(slot.width for slot in slots),
            "slots": [{"block": slot.path, "width": slot.width} for slot in slots],
        },
        "norm_stats": {
            c: {k: float(norm_stats[c][k]) for k in ("shift", "stretch", "min", "max")}
            for c in columns
        },
        "architecture": {
            "note": "documentation only; the runtime never reads this section",
            **describe_dynamics(dyn_sys, cfg),
        },
        "trained_from": {"run_id": run_id, "model": cfg.model},
    }


def opset(graph_format: str) -> Optional[int]:
    """The ONNX opset the graphs target; a torch program has none."""
    return OPSET if graph_format == "onnx" else None
