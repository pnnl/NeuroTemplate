"""
Export a trained DPC policy into `<out>/policy/`.

    policy/
      policy.onnx           (y, ymin, ymax, d[, price]) -> u, physical units
      policy.json           each input's columns, window length and alignment
      reference/
        inputs.npz          real closed-loop windows the trained policy saw
        expected_actions.csv  the TRAINED policy's commands on them
        REFERENCE.json      provenance and tolerances

A trained policy is an explicit network, fixed after training, with no
state: one graph, not two, and none of the state lifting the dynamics
export needs. Its windows are read off the policy Node's own `input_map`
(past / future per key), so whatever preview the policy was built with —
band, per-column disturbance, tariff — is exactly what the graph expects.

Verification runs the trained closed loop over the control run's test split
(the same `evaluate_closed_loop` the run was scored with) and collects every
window the policy was handed along the way. The torch policy's commands on
those windows are the reference; ONNX Runtime must match them.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neurotemplate.control.pipeline import evaluate_closed_loop
from neurotemplate.control.policy import describe_policy, regulated_channels
from neurotemplate.control.scenarios import price_scale
from neurotemplate.export.package import (
    FLOAT_FORMAT, check_package_name, install, provenance, stamp, write_json,
)
from neurotemplate.export.runtime import Policy
from neurotemplate.export.spec import FORMAT
from neurotemplate.export.trace import OPSET, export_graph
from neurotemplate.export.verify import AGREEMENT_TOL, REPRODUCE_TOL
from neurotemplate.hooks import load_control_hooks

#: The policy Node's inputs, in the order `build_policy` gives them; a
#: tariff-reading policy appends "price".
POLICY_KEYS = ["yn", "ymin", "ymax", "D"]
#: Policy Node input key -> graph input name.
GRAPH_NAMES = {"yn": "y", "ymin": "ymin", "ymax": "ymax", "D": "d", "price": "price"}


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
    """Every input the policy reads, in the order its Node passes them."""
    regulated, _ = regulated_channels(load_control_hooks(cfg), nm_mapping)
    scale = price_scale(cfg)
    columns = {"yn": nm_mapping["Y"], "ymin": regulated, "ymax": regulated,
               "D": nm_mapping["D"], "price": ["price"]}
    inputs = []
    for key in policy_node.input_keys:
        window = policy_node.input_map.get(key, {"past": 0, "future": 0})
        cols = columns[key]
        if key == "price":
            # The window dict carries price / price_scale, with no shift.
            shift, stretch = [0.0], [float(scale)]
        else:
            shift = [float(norm_stats[c]["shift"]) for c in cols]
            stretch = [float(norm_stats[c]["stretch"]) for c in cols]
        inputs.append(PolicyInput(key, cols, window["past"] + 1 + window["future"],
                                  window["past"], shift, stretch))
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
        # because torch.export cannot map dynamic shapes onto `*args`.
        if self.keys not in (POLICY_KEYS, POLICY_KEYS + ["price"]):
            raise ValueError(f"unexpected policy inputs {self.keys}; expected "
                             f"{POLICY_KEYS} plus optionally 'price'.")
        for spec in inputs:
            self.register_buffer(f"{spec.key}_shift", torch.tensor(spec.shift))
            self.register_buffer(f"{spec.key}_stretch", torch.tensor(spec.stretch))
        u = nm_mapping["U"]
        self.register_buffer("u_shift", torch.tensor([norm_stats[c]["shift"] for c in u]))
        self.register_buffer("u_stretch", torch.tensor([norm_stats[c]["stretch"] for c in u]))

    def forward(self, y: torch.Tensor, ymin: torch.Tensor, ymax: torch.Tensor,
                d: torch.Tensor, price: torch.Tensor | None = None) -> torch.Tensor:
        windows = [y, ymin, ymax, d] + ([price] if price is not None else [])
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
    kept: list[str]


def export_policy(*, policy_node, cl_system, cfg, sysid_cfg,
                  nm_mapping: dict[str, list[str]], norm_stats: dict[str, dict],
                  test_dfs: list, run_id: str, out_dir: Path, package: str,
                  description: Optional[str] = None, exporter: str = "dynamo",
                  fixture_windows: int = 256, version: str = "0.1.0") -> PolicyExport:
    """Export, verify on every closed-loop window of the test split, install.

    `test_dfs` are the control run's normalised test frames. The shipped
    fixture is `fixture_windows` of the verified windows, evenly spaced.
    """
    check_package_name(package)
    # Export and verification run on the CPU. The evaluation rollout builds
    # its data on `sysid_cfg.device`, so the configs move with the modules.
    cl_system.cpu().eval()
    cfg.device = sysid_cfg.device = torch.device("cpu")
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
        path = policy_dir / "policy.onnx"
        # Trace on two real windows: batch >= 2, see trace.export_graph.
        stamp(export_graph(graph, tuple(x[:2] for x in physical.values()),
                           [GRAPH_NAMES[s.key] for s in inputs], ["u"], path, exporter),
              path, export_id, run_id)
        write_json(policy_dir / "policy.json", _spec(
            export_id, run_id, cfg, inputs, nm_mapping, description,
            describe_policy(policy_node, cfg, sysid_cfg, nm_mapping, norm_stats)))

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
        _write_reference(policy_dir / "reference", named, reference, shipped,
                         nm_mapping["U"], {
                             "export_id": export_id, "run_id": run_id,
                             "tolerance": {"u": {c: REPRODUCE_TOL * s
                                                 for c, s in u_stretch.items()}},
                             "agreement_at_export": {
                                 "windows": count,
                                 "tolerance_fraction_of_stretch": AGREEMENT_TOL,
                                 "u_max_abs": error,
                             },
                             "exported_on": provenance(exporter),
                         })
        kept = install(Path(tmp), Path(out_dir), "policy", package, version)

    return PolicyExport(out_dir=Path(out_dir), error=error, windows_verified=count,
                        windows_shipped=len(shipped), kept=kept)


def _spec(export_id: str, run_id: str, cfg, inputs: list[PolicyInput],
          nm_mapping, description: Optional[str], architecture: dict) -> dict:
    """The `policy.json` document. The runtime reads `inputs` and `u_columns`."""
    return {
        "format": FORMAT,
        "export_id": export_id,
        "opset": OPSET,
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


def _write_reference(reference_dir: Path, named: dict[str, np.ndarray],
                     reference: np.ndarray, shipped: np.ndarray,
                     u_columns: list[str], document: dict) -> None:
    reference_dir.mkdir()
    np.savez(reference_dir / "inputs.npz",
             **{name: windows[shipped] for name, windows in named.items()})
    pd.DataFrame(reference[shipped], columns=u_columns).to_csv(
        reference_dir / "expected_actions.csv", float_format=FLOAT_FORMAT,
        index_label="window")
    write_json(reference_dir / "REFERENCE.json", {
        "source": "the trained torch policy, not the exported graph",
        "windows": len(shipped),
        **document,
    })
