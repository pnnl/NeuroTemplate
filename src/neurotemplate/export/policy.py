"""
Export a trained DPC policy into `<out>/policy/`.

    policy/
      policy.onnx|.pt2|.c   (y, d[, r]) -> u, physical units
      policy.h              C only: the contract as C constants
      policy.json           each input's columns, window length and alignment
      reference/
        inputs.csv          real closed-loop windows the trained policy saw,
                            one per row (see `runtime.window_columns`)
        expected_actions.csv  the TRAINED policy's commands on them
        REFERENCE.json      provenance and tolerances

A trained policy is an explicit network, fixed after training, with no
state: one graph, not two, and none of the state lifting the dynamics
export needs. Its windows are read off the policy Node's own `input_map`
(past / future per key), so whatever preview the policy was built with —
per-column disturbance, per-column reference — is exactly what the graph
expects.

Verification runs the trained closed loop over the control run's test split
(the same `evaluate_closed_loop` the run was scored with) and collects every
window the policy was handed along the way. The torch policy's commands on
those windows are the reference; ONNX Runtime must match them.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neurotemplate.control.losses import resolve_objective
from neurotemplate.control.pipeline import evaluate_closed_loop
from neurotemplate.control.policy import describe_policy
from neurotemplate.export import ccode
from neurotemplate.export.package import (
    FLOAT_FORMAT, check_graph_format, install, provenance, write_graph, write_json,
)
from neurotemplate.export.runtime import Policy, window_columns
from neurotemplate.export.spec import FORMAT, opset
from neurotemplate.export.verify import AGREEMENT_TOL, REPRODUCE_TOL

#: The policy Node's inputs, in the order `build_policy` gives them; a policy
#: whose objective names reference columns appends "R".
POLICY_KEYS = ["yn", "D"]
#: Policy Node input key -> graph input name.
GRAPH_NAMES = {"yn": "y", "D": "d", "R": "r"}


@dataclass(frozen=True)
class PolicyInput:
    """One policy input: its window, and how to put it in physical units."""
    key: str                # the Node's input key
    columns: list[str]
    steps: int              # window length
    current_row: int        # where the present step sits; later rows are preview
    shift: list[float]
    stretch: list[float]


def policy_inputs(policy_node, cfg, nm_mapping: dict[str, list[str]],
                  norm_stats: dict[str, dict]) -> list[PolicyInput]:
    """Every input the policy reads, in the order its Node passes them.

    `norm_stats` carries the reference columns (`losses.extend_norm_stats`),
    so every input denormalises the same way.
    """
    columns = {"yn": nm_mapping["Y"], "D": nm_mapping["D"],
               "R": resolve_objective(cfg, nm_mapping).r_columns}
    inputs = []
    for key in policy_node.input_keys:
        window = policy_node.input_map.get(key, {"past": 0, "future": 0})
        cols = columns[key]
        inputs.append(PolicyInput(
            key, cols, window["past"] + 1 + window["future"], window["past"],
            [float(norm_stats[c]["shift"]) for c in cols],
            [float(norm_stats[c]["stretch"]) for c in cols]))
    return inputs


class PolicyGraph(nn.Module):
    """A trained policy with physical units on both sides.

    Each input arrives as `[batch, steps, n]` and is normalised and flattened
    time-major — the layout `SystemPreview.get_mapped_data` hands the Node —
    before the policy sees it; its command leaves in physical units.
    """

    def __init__(self, policy: nn.Module, inputs: list[PolicyInput],
                 nm_mapping: dict[str, list[str]], norm_stats: dict[str, dict]):
        super().__init__()
        self.policy = policy
        self.keys = [spec.key for spec in inputs]
        # build_policy fixes this order; forward's signature spells it out
        # because torch.export cannot map dynamic shapes onto `*args`. The
        # reference columns are one tensor for the same reason: their count is
        # config-dependent, an argument count cannot be.
        if self.keys not in (POLICY_KEYS, POLICY_KEYS + ["R"]):
            raise ValueError(f"unexpected policy inputs {self.keys}; expected "
                             f"{POLICY_KEYS} plus optionally 'R'.")
        for spec in inputs:
            self.register_buffer(f"{spec.key}_shift", torch.tensor(spec.shift))
            self.register_buffer(f"{spec.key}_stretch", torch.tensor(spec.stretch))
        u = nm_mapping["U"]
        self.register_buffer("u_shift", torch.tensor([norm_stats[c]["shift"] for c in u]))
        self.register_buffer("u_stretch", torch.tensor([norm_stats[c]["stretch"] for c in u]))

    def forward(self, y: torch.Tensor, d: torch.Tensor,
                r: torch.Tensor | None = None) -> torch.Tensor:
        windows = [y, d] + ([r] if r is not None else [])
        flat = [((x - getattr(self, f"{key}_shift")) / getattr(self, f"{key}_stretch"))
                .reshape(x.shape[0], -1) for key, x in zip(self.keys, windows)]
        return self.policy(*flat) * self.u_stretch + self.u_shift


def closed_loop_windows(cl_system, policy_node, cfg, sysid_cfg, test_dfs: list,
                        nm_mapping, norm_stats) -> dict[str, torch.Tensor]:
    """Every window the policy is handed in the trained closed loop's
    evaluation rollout: `{key: [windows, steps * n]}`, normalised, flat."""
    _, out = evaluate_closed_loop(cfg, sysid_cfg, cl_system, test_dfs,
                                  nm_mapping, norm_stats)
    windows: dict[str, list[torch.Tensor]] = {key: [] for key in policy_node.input_keys}
    first = cl_system.start_iter
    for step in range(first, first + cl_system.nsteps):
        for key in policy_node.input_keys:
            if key in policy_node.input_map:
                window = cl_system.get_mapped_data(out[key], step, policy_node.input_map[key])
            else:
                window = out[key][:, step]
            windows[key].append(window)
    return {key: torch.cat(parts) for key, parts in windows.items()}


@dataclass(frozen=True)
class PolicyExport:
    out_dir: Path
    error: dict[str, float]     # worst |ORT - torch| per command, physical units
    windows_verified: int
    windows_shipped: int


def export_policy(*, policy_node, cl_system, cfg, sysid_cfg,
                  nm_mapping: dict[str, list[str]], norm_stats: dict[str, dict],
                  test_dfs: list, run_id: str, out_dir: Path,
                  description: Optional[str] = None, graph_format: str = "onnx",
                  exporter: str = "dynamo", onnx2c: str = "onnx2c",
                  fixture_windows: int = 256) -> PolicyExport:
    """Export, verify on every closed-loop window of the test split, install.

    `graph_format`, `exporter` and `onnx2c` are as for `dynamics.export_model`.
    `test_dfs` are the control run's normalised test frames. The shipped
    fixture is `fixture_windows` of the verified windows, evenly spaced.
    """
    check_graph_format(graph_format, exporter, onnx2c)
    # Export and verification run on the CPU. The evaluation rollout builds
    # its data on `sysid_cfg.device`, so the configs move with the modules —
    # as copies, leaving the caller's configs as they were.
    cl_system.cpu().eval()
    cpu = torch.device("cpu")
    cfg, sysid_cfg = replace(cfg, device=cpu), replace(sysid_cfg, device=cpu)
    inputs = policy_inputs(policy_node, cfg, nm_mapping, norm_stats)
    graph = PolicyGraph(policy_node.callable, inputs, nm_mapping, norm_stats).eval()

    flat = closed_loop_windows(cl_system, policy_node, cfg, sysid_cfg, test_dfs,
                               nm_mapping, norm_stats)
    physical = {spec.key: (flat[spec.key].reshape(-1, spec.steps, len(spec.columns))
                           * torch.tensor(spec.stretch) + torch.tensor(spec.shift))
                for spec in inputs}
    with torch.no_grad():
        reference = graph(*physical.values()).double().numpy()
    count = len(reference)

    u_stretch = {c: norm_stats[c]["stretch"] for c in nm_mapping["U"]}
    export_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        policy_dir = Path(tmp) / "policy"
        policy_dir.mkdir()
        # Trace on two real windows: batch >= 2, see trace.export_graph.
        write_graph(graph, tuple(x[:2] for x in physical.values()),
                    [GRAPH_NAMES[s.key] for s in inputs], ["u"],
                    policy_dir / "policy", graph_format, exporter, export_id, run_id,
                    onnx2c)
        spec = _spec(export_id, run_id, cfg, inputs, nm_mapping, description,
                     describe_policy(policy_node, cfg, sysid_cfg, nm_mapping, norm_stats),
                     graph_format)
        write_json(policy_dir / "policy.json", spec)
        u_tolerance = {c: REPRODUCE_TOL * s for c, s in u_stretch.items()}
        if graph_format == "c":
            typical = {GRAPH_NAMES[i.key]: i.shift for i in inputs}
            (policy_dir / "policy.h").write_text(
                ccode.policy_header(spec, typical, u_tolerance), encoding="utf-8")
            ccode.compile_library(Path(tmp), "policy")

        named = {GRAPH_NAMES[key]: x.numpy() for key, x in physical.items()}
        actions = Policy(policy_dir).act_batch(named)
        error = {c: float(np.abs(actions[:, i] - reference[:, i]).max())
                 for i, c in enumerate(nm_mapping["U"])}
        worst = max(error[c] / u_stretch[c] for c in error)
        if worst > AGREEMENT_TOL:
            raise RuntimeError(
                "REFUSING TO WRITE: the exported policy does not reproduce the "
                f"trained one (worst command error {worst:.2e} of its column's "
                f"stretch; tolerance {AGREEMENT_TOL:.0e}). Per column: {error}")

        shipped = np.unique(np.linspace(0, count - 1, min(fixture_windows, count)).astype(int))
        _write_reference(policy_dir / "reference", spec["inputs"], named, reference,
                         shipped, nm_mapping["U"], {
                             "export_id": export_id, "run_id": run_id,
                             "tolerance": {"u": u_tolerance},
                             "agreement_at_export": {
                                 "windows": count,
                                 "tolerance_fraction_of_stretch": AGREEMENT_TOL,
                                 "u_max_abs": error,
                             },
                             "exported_on": provenance(graph_format, exporter),
                         })
        install(Path(tmp), Path(out_dir), "policy")

    return PolicyExport(out_dir=Path(out_dir), error=error, windows_verified=count,
                        windows_shipped=len(shipped))


def _spec(export_id: str, run_id: str, cfg, inputs: list[PolicyInput],
          nm_mapping, description: Optional[str], architecture: dict,
          graph_format: str) -> dict:
    """The `policy.json` document. The runtime reads `graph_format`, `inputs`
    and `u_columns`."""
    return {
        "format": FORMAT,
        "graph_format": graph_format,
        "export_id": export_id,
        "opset": opset(graph_format),
        "description": description,
        "inputs": {
            GRAPH_NAMES[spec.key]: {"columns": spec.columns, "steps": spec.steps,
                                    "current_row": spec.current_row}
            for spec in inputs
        },
        "u_columns": nm_mapping["U"],
        "architecture": {
            "note": "documentation only; the runtime never reads this section",
            **architecture,
        },
        "trained_from": {"run_id": run_id, "policy": cfg.policy,
                         "sysid_run_id": cfg.sysid_run_id},
    }


def _write_reference(reference_dir: Path, inputs: dict[str, dict],
                     named: dict[str, np.ndarray], reference: np.ndarray,
                     shipped: np.ndarray, u_columns: list[str], document: dict) -> None:
    """CSV throughout, so the C++ runner reads the same fixture as Python."""
    reference_dir.mkdir()
    windows = pd.DataFrame(
        np.concatenate([named[name][shipped].reshape(len(shipped), -1) for name in inputs],
                       axis=1),
        columns=[c for name, entry in inputs.items() for c in window_columns(name, entry)])
    windows.to_csv(reference_dir / "inputs.csv", float_format=FLOAT_FORMAT,
                   index_label="window")
    pd.DataFrame(reference[shipped], columns=u_columns).to_csv(
        reference_dir / "expected_actions.csv", float_format=FLOAT_FORMAT,
        index_label="window")
    write_json(reference_dir / "REFERENCE.json", {
        "source": "the trained torch policy, not the exported graph",
        "windows": len(shipped),
        **document,
    })
