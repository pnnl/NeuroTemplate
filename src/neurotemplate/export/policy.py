"""
Export a trained DPC policy into `<out>/policy/`.

    policy/
      policy.onnx|.pt2|.c   (y[, u_history], d[, r]) -> u, physical units
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
(past / future per key), so whatever window the policy was built with —
per-column history and preview — is exactly what the graph expects. A
policy reading past commands takes them as `u_history`, the last `steps`
commands applied, kept by the recipient as the model runtime keeps its own.

Verification runs the trained closed loop over the control run's test split
(the same `evaluate_closed_loop` the run was scored with) and collects every
window the policy was handed along the way. The torch policy's commands on
those windows are the reference; ONNX Runtime must match them.
"""

from __future__ import annotations

import json
import shutil
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
from neurotemplate.control.controller import describe_controller
from neurotemplate.control.mpc import MPC, _inner_keys
from neurotemplate.control.policy import _bound_side, _ReferenceBounds, with_command_bounds
from neurotemplate.hooks import load_control_hooks
from neurotemplate.export import ccode
from neurotemplate.export.package import (
    FLOAT_FORMAT, check_graph_format, install, provenance, write_graph, write_json,
)
from neurotemplate.export.runtime import Policy, window_columns
from neurotemplate.export.spec import FORMAT, opset
from neurotemplate.export.verify import AGREEMENT_TOL, REPRODUCE_TOL

#: The share of windows on which an exported MPC must plan what the trained
#: one plans. A surrogate with a latch or a clamp is piecewise, so a one-ulp
#: difference can send a few windows' Adam steps down another branch; an
#: export that is wrong disagrees on most of them.
MPC_AGREEMENT_SHARE = 0.95

#: Every input a policy Node may read, in the order `build_policy` gives
#: them; "U" only when it reads past commands, "R" only when its objective
#: names reference columns.
POLICY_KEYS = ["yn", "U", "D", "R"]
#: Policy Node input key -> graph input name.
GRAPH_NAMES = {"yn": "y", "U": "u_history", "D": "d", "R": "r"}


@dataclass(frozen=True)
class PolicyInput:
    """One policy input: its window, and its columns' typical value."""
    key: str                # the Node's input key
    columns: list[str]
    steps: int              # window length the graph takes
    current_row: int        # where the present step sits; later rows are preview.
                            # For u_history, `steps`: it ends the step before.
    shift: list[float]      # each column's typical value, for the C example


def policy_inputs(policy_node, cfg, nm_mapping: dict[str, list[str]],
                  norm_stats: dict[str, dict]) -> list[PolicyInput]:
    """Every input the policy reads, in the order its Node passes them.

    `norm_stats` carries the reference columns (`losses.extend_norm_stats`).
    """
    columns = {"yn": nm_mapping["Y"], "U": nm_mapping["U"], "D": nm_mapping["D"],
               "R": resolve_objective(cfg, nm_mapping).r_columns}
    inputs = []
    for key in policy_node.input_keys:
        window = policy_node.input_map[key]
        cols = columns[key]
        inputs.append(PolicyInput(
            key, cols, _graph_rows(key, window), window["past"],
            [float(norm_stats[c]["shift"]) for c in cols]))
    return inputs


def _graph_rows(key: str, window: dict) -> int:
    """Rows of one input the graph takes. The Node's U window also holds the
    current row, a padded copy the policy never reads (see
    `control.policy._input_window`), so the graph takes U without it."""
    rows = window["past"] + 1 + window["future"]
    return rows - 1 if key == "U" else rows


class PolicyGraph(nn.Module):
    """A trained policy, physical units on both sides as the policy already is.

    Each input arrives as `[batch, steps, n]` and is flattened time-major —
    the layout `SystemPreview.get_mapped_data` hands the Node — before the
    policy sees it.
    `u_history` gets its newest row repeated as the current one, as SystemPreview pads
    it in the closed loop; the policy drops that row, so the recipient
    never supplies it.

    This class is the graph of a policy that reads no past commands;
    `_PolicyGraphWithU` takes `u_history` too. `policy_graph` picks.
    """

    def __init__(self, policy: nn.Module, inputs: list[PolicyInput]):
        super().__init__()
        self.policy = policy
        self.keys = [spec.key for spec in inputs]
        # build_policy fixes this order; forward's signature spells it out
        # because torch.export cannot map dynamic shapes onto `*args`. The
        # reference columns are one tensor for the same reason: their count is
        # config-dependent, an argument count cannot be.
        if [key for key in POLICY_KEYS if key in self.keys] != self.keys or \
                not {"yn", "D"} <= set(self.keys):
            raise ValueError(f"unexpected policy inputs {self.keys}; expected "
                             f"{POLICY_KEYS}, U and R optional.")

    def forward(self, y: torch.Tensor, d: torch.Tensor,
                r: torch.Tensor | None = None) -> torch.Tensor:
        return self._command({"yn": y, "D": d, "R": r})

    def _command(self, windows: dict[str, torch.Tensor | None]) -> torch.Tensor:
        flat = []
        for key in self.keys:
            x = windows[key]
            if key == "U":
                x = torch.cat([x, x[:, -1:]], dim=1)
            flat.append(x.reshape(x.shape[0], -1))
        return self.policy(*flat)


class _PolicyGraphWithU(PolicyGraph):
    """`PolicyGraph` for a policy that reads past commands:
    `(y, u_history, d[, r])`."""

    def forward(self, y: torch.Tensor, u_history: torch.Tensor, d: torch.Tensor,
                r: torch.Tensor | None = None) -> torch.Tensor:
        return self._command({"yn": y, "U": u_history, "D": d, "R": r})


def policy_graph(policy: nn.Module, inputs: list[PolicyInput]) -> PolicyGraph:
    """The graph for `policy`: torch.export and the ONNX exporter bind the
    inputs positionally to `forward`, so its signature lists exactly the
    windows the policy reads."""
    reads_u = any(spec.key == "U" for spec in inputs)
    graph = _PolicyGraphWithU if reads_u else PolicyGraph
    return graph(policy, inputs)


def physical_windows(flat: dict[str, torch.Tensor],
                     inputs: list[PolicyInput]) -> dict[str, torch.Tensor]:
    """Node windows (`closed_loop_windows`) as the graph takes them:
    `{key: [windows, steps, n]}`, U without its current row."""
    out = {}
    for spec in inputs:
        n = len(spec.columns)
        rows = flat[spec.key].shape[-1] // n
        out[spec.key] = flat[spec.key].reshape(-1, rows, n)[:, :spec.steps]
    return out


def closed_loop_windows(cl_system, policy_node, cfg, sysid_cfg, test_dfs: list,
                        nm_mapping, norm_stats) -> dict[str, torch.Tensor]:
    """Every window the policy is handed in the trained closed loop's
    evaluation rollout: `{key: [windows, steps * n]}`, flat.

    Read off the finished rollout, so a U window's current row is the
    command chosen at that step rather than the padded copy the policy was
    handed. The policy drops that row either way."""
    _, out = evaluate_closed_loop(cfg, sysid_cfg, cl_system, test_dfs,
                                  nm_mapping, norm_stats)
    windows: dict[str, list[torch.Tensor]] = {key: [] for key in policy_node.input_keys}
    first = cl_system.start_iter
    for step in range(first, first + cl_system.nsteps):
        for key in policy_node.input_keys:
            windows[key].append(
                cl_system.get_mapped_data(out[key], step, policy_node.input_map[key]))
    return {key: torch.cat(parts) for key, parts in windows.items()}


@dataclass(frozen=True)
class PolicyExport:
    out_dir: Path
    error: dict[str, float]     # worst |ORT - torch| per command, physical units
    windows_verified: int
    windows_shipped: int
    share: float                # of the windows, every command within tolerance


def export_policy(*, policy_node, cl_system, cfg, sysid_cfg,
                  nm_mapping: dict[str, list[str]], norm_stats: dict[str, dict],
                  test_dfs: list, run_id: str, out_dir: Path,
                  description: Optional[str] = None, graph_format: str = "onnx",
                  exporter: str = "dynamo", onnx2c: str = "onnx2c",
                  fixture_windows: int = 256) -> PolicyExport:
    """Export, verify on every closed-loop window of the test split, install.

    `graph_format`, `exporter` and `onnx2c` are as for `dynamics.export_model`.
    `test_dfs` are the control run's test frames. The shipped
    fixture is `fixture_windows` of the verified windows, evenly spaced.
    """
    check_graph_format(graph_format, exporter, onnx2c)
    mpc = policy_node.callable if isinstance(policy_node.callable, MPC) else None
    if mpc is not None:
        _check_mpc_package(graph_format, Path(out_dir))
    # Export and verification run on the CPU. The evaluation rollout builds
    # its data on `sysid_cfg.device`, so the configs move with the modules —
    # as copies, leaving the caller's configs as they were.
    cl_system.cpu().eval()
    cpu = torch.device("cpu")
    cfg, sysid_cfg = replace(cfg, device=cpu), replace(sysid_cfg, device=cpu)
    inputs = policy_inputs(policy_node, cfg, nm_mapping, norm_stats)
    graph = None if mpc is not None else policy_graph(policy_node.callable, inputs).eval()

    flat = closed_loop_windows(cl_system, policy_node, cfg, sysid_cfg, test_dfs,
                               nm_mapping, norm_stats)
    physical = physical_windows(flat, inputs)
    with torch.no_grad():
        if mpc is None:
            reference = graph(*physical.values()).double().numpy()
        else:   # every window planned from scratch, as act_batch plans them
            mpc.set_rollout_state(None)
            reference = mpc(*flat.values()).double().numpy()
    count = len(reference)

    u_stretch = {c: norm_stats[c]["stretch"] for c in nm_mapping["U"]}
    export_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        policy_dir = Path(tmp) / "policy"
        policy_dir.mkdir()
        spec = _spec(export_id, run_id, cfg, inputs, nm_mapping, description,
                     describe_controller(policy_node, cfg, sysid_cfg, nm_mapping),
                     graph_format)
        if mpc is None:
            # Trace on two real windows: batch >= 2, see trace.export_graph.
            write_graph(graph, tuple(x[:2] for x in physical.values()),
                        [GRAPH_NAMES[s.key] for s in inputs], ["u"],
                        policy_dir / "policy", graph_format, exporter, export_id, run_id,
                        onnx2c)
        else:
            # The planner runs through the package's model; the staged copy
            # lets the self-check below run it.
            shutil.copytree(Path(out_dir) / "model", Path(tmp) / "model")
            spec["mpc"] = _mpc_block(mpc, cfg, nm_mapping)
            if spec["mpc"]["power"]:
                _write_power_graph(cfg, nm_mapping, mpc, inputs, policy_dir, export_id, run_id)
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
        stretch = np.array([u_stretch[c] for c in nm_mapping["U"]])
        within = (np.abs(actions - reference) / stretch <= AGREEMENT_TOL).all(1).mean()
        if mpc is None and within < 1 or within < MPC_AGREEMENT_SHARE:
            raise RuntimeError(
                "REFUSING TO WRITE: the exported policy does not reproduce the "
                f"trained one: {within * 100:.1f}% of windows agree to "
                f"{AGREEMENT_TOL:.0e} of each command's stretch"
                + ("" if mpc is None else f", {MPC_AGREEMENT_SHARE * 100:.0f}% required")
                + f". Worst per column: {error}")

        shipped = np.unique(np.linspace(0, count - 1, min(fixture_windows, count)).astype(int))
        _write_reference(policy_dir / "reference", spec["inputs"], named, reference,
                         shipped, nm_mapping["U"], {
                             "export_id": export_id, "run_id": run_id,
                             "tolerance": {"u": u_tolerance, **(
                                 {} if mpc is None else {"share": MPC_AGREEMENT_SHARE})},
                             "agreement_at_export": {
                                 "windows": count,
                                 "share_within_tolerance": float(within),
                                 "tolerance_fraction_of_stretch": AGREEMENT_TOL,
                                 "u_max_abs": error,
                             },
                             "exported_on": provenance(graph_format, exporter),
                         })
        install(Path(tmp), Path(out_dir), "policy")

    return PolicyExport(out_dir=Path(out_dir), error=error, windows_verified=count,
                        windows_shipped=len(shipped), share=float(within))


def _check_mpc_package(graph_format: str, out_dir: Path) -> None:
    """An MPC policy plans through torch model graphs in the same package."""
    if graph_format != "torch":
        raise ValueError("policy: mpc plans with gradients at run time, through the "
                         "model graphs: export it with --format torch.")
    spec = out_dir / "model" / "model.json"
    if not spec.exists() or json.loads(spec.read_text(encoding="utf-8"))["graph_format"] != "torch":
        raise FileNotFoundError(
            f"policy: mpc plans through the surrogate, so {out_dir}/model must hold it "
            "in torch format first (`nt export policy` writes it for an MPC run).")


def _mpc_block(mpc: MPC, cfg, nm_mapping: dict[str, list[str]]) -> dict:
    """What the runtime's planner needs: `policy.json`'s `mpc`."""
    objective = resolve_objective(cfg, nm_mapping)
    keys, input_map = _inner_keys(objective.r_columns)
    bounds = with_command_bounds(torch.nn.Identity(), cfg, nm_mapping, keys, input_map,
                                 mpc.umin, mpc.umax)
    order = ([nm_mapping["U"][j] for j in bounds.columns]
             if isinstance(bounds, _ReferenceBounds) else [])

    def side(raw) -> dict:
        name, offset = _bound_side(raw)
        return {"value": offset} if name is None else {"column": name, "offset": offset}

    return {
        "horizon": mpc.horizon, "iters": mpc.iters, "lr": mpc.lr, "hmax": mpc.hmax,
        "umin": mpc.umin.tolist(), "umax": mpc.umax.tolist(),
        "command_bounds": [{"column": c, "low": side(cfg.command_bounds[c][0]),
                            "high": side(cfg.command_bounds[c][1])} for c in order],
        "terms": [{"name": t.name, "column": t.column, "target": t.target, "map": t.map,
                   "norm": t.norm, "weight": t.weight, "gain": t.gain}
                  for t in objective.terms],
        "power": objective.reads_power,
    }


class _PowerGraph(torch.nn.Module):
    """`power_metric` over planned trajectories, for the MPC runtime."""

    def __init__(self, hook, nm_mapping: dict[str, list[str]]):
        super().__init__()
        self.hook, self.nm_mapping = hook, nm_mapping

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.hook(y, u, d, self.nm_mapping)


def _write_power_graph(cfg, nm_mapping, mpc: MPC, inputs: list[PolicyInput],
                       policy_dir: Path, export_id: str, run_id: str) -> None:
    """Trace the power hook on two typical trajectories one horizon long."""
    typical = {spec.key: torch.tensor(spec.shift) for spec in inputs}
    args = tuple(x.expand(2, mpc.horizon, -1).clone()
                 for x in (typical["yn"], 0.5 * (mpc.umin + mpc.umax), typical["D"]))
    write_graph(_PowerGraph(load_control_hooks(cfg).power_metric, nm_mapping), args,
                ["y", "u", "d"], ["power"], policy_dir / "power", "torch", "dynamo",
                export_id, run_id, "onnx2c")


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
