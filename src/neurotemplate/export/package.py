"""
Export a trained dynamics model as a self-contained consumer package.

    <out>/
      README.md            generated: the I/O contract, how to run it
      pyproject.toml       generated: numpy + onnxruntime, nothing else
      validate.py          written once; add project-specific checks to it
      example.py           written once
      <package>/           the shipped runtime (export/runtime.py, verbatim)
      model/               warm_start.onnx, step.onnx, model.json
      data/                example_input.csv, expected_output.csv,
                           expected_state.csv, REFERENCE.json

Everything is built in a staging directory and copied into `<out>` only once
the graphs agree with the trained model over the whole split, and the staged
package passes its own `validate` check. A failed export leaves `<out>`
untouched. `validate.py` and `example.py` are the project's to edit, so a
re-export never overwrites them; `model/MODEL_CARD.md`, if the project keeps
one, is never touched either.
"""

from __future__ import annotations

import io
import json
import platform
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from string import Template
from typing import Optional

import onnx
import onnxruntime
import pandas as pd
import torch

from neurotemplate.export import runtime
from neurotemplate.export.graphs import Step, WarmStart
from neurotemplate.export.spec import build_spec
from neurotemplate.export.state import StateSlot, probe_state_slots
from neurotemplate.export.trace import OPSET, export_graph
from neurotemplate.export.verify import (
    AGREEMENT_TOL, Agreement, compare, onnx_rollout, require_agreement,
    tolerances, torch_rollout,
)

TEMPLATES = Path(__file__).parent / "templates"

#: Written into the CSVs. `%.6f` does not round-trip float32 (it keeps 83 of
#: 1000 normalised-state values and 493 of 1000 kW values exact); `%.9g`
#: round-trips all of them, so the file format never sets the tolerance floor.
FLOAT_FORMAT = "%.9g"

#: Files the project owns after the first export.
WRITE_ONCE = ("validate.py", "example.py")


@dataclass(frozen=True)
class ExportResult:
    out_dir: Path
    agreement: Agreement
    steps_verified: int
    steps_shipped: int
    state_width: int
    kept: list[str]     # project-owned files left as they were


def export_package(*, dyn_sys, cfg, nm_mapping: dict[str, list[str]],
                   norm_stats: dict[str, dict], frame: pd.DataFrame,
                   timestep_seconds: float, run_id: str, out_dir: Path,
                   package: str, description: Optional[str] = None,
                   exporter: str = "dynamo", fixture_steps: Optional[int] = None,
                   version: str = "0.1.0") -> ExportResult:
    """Export, verify over every row of `frame`, and write the package.

    `frame` is a held-out split in physical units, indexed by timestamp. The
    agreement check covers all of it; the shipped fixture is its first
    `in_steps + fixture_steps` rows (all of it by default).
    """
    if not package.isidentifier():
        raise ValueError(f"package name {package!r} is not a Python identifier.")
    columns = nm_mapping["Y"] + nm_mapping["U"] + nm_mapping["D"]
    frame = _round_trip(frame, columns)
    steps = len(frame) - cfg.in_steps
    shipped = steps if fixture_steps is None else fixture_steps
    if not 1 <= shipped <= steps:
        raise ValueError(f"the split has {steps} steps after warm-up; cannot "
                         f"ship a fixture of {shipped}.")

    dyn_sys = dyn_sys.cpu().eval()
    export_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        slots = _write_model(stage / "model", dyn_sys, cfg, nm_mapping, norm_stats,
                             exporter, export_id, run_id)
        spec = build_spec(export_id=export_id, run_id=run_id, cfg=cfg,
                          dyn_sys=dyn_sys, nm_mapping=nm_mapping,
                          norm_stats=norm_stats, timestep_seconds=timestep_seconds,
                          slots=slots, description=description)
        (stage / "model" / "model.json").write_text(json.dumps(spec, indent=2) + "\n")

        reference = torch_rollout(dyn_sys, nm_mapping, norm_stats, frame)
        agreement = compare(reference, onnx_rollout(stage / "model", frame), nm_mapping["Y"])
        require_agreement(agreement, norm_stats)

        _write_fixture(stage / "data", frame.iloc[:cfg.in_steps + shipped],
                       reference, slots, nm_mapping, norm_stats, cfg.in_steps,
                       reference_info={
                           "export_id": export_id, "run_id": run_id,
                           "exporter": exporter, "steps_verified": steps,
                           "agreement": agreement,
                       })
        _write_scaffold(stage, spec, package, version)

        check = runtime.check_reference(stage)
        if not check.ok:
            raise RuntimeError("REFUSING TO WRITE: the staged package fails its "
                               "own reference check:\n" + check.report())
        kept = _install(stage, Path(out_dir), package)

    return ExportResult(out_dir=Path(out_dir), agreement=agreement,
                        steps_verified=steps, steps_shipped=shipped,
                        state_width=spec["state"]["width"], kept=kept)


def _round_trip(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """`frame[columns]` exactly as the shipped CSV will read back, so the
    torch reference is computed from the values a consumer actually has."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"the split is missing mapped column(s) {missing}.")
    if frame[columns].isna().any().any():
        raise ValueError("the split has NaNs in mapped columns; a rollout over "
                         "it has no defined answer.")
    buffer = io.StringIO()
    frame[columns].to_csv(buffer, float_format=FLOAT_FORMAT, index_label="timestamp")
    buffer.seek(0)
    return pd.read_csv(buffer, index_col="timestamp")


def _write_model(model_dir: Path, dyn_sys, cfg, nm_mapping, norm_stats,
                 exporter: str, export_id: str, run_id: str) -> list[StateSlot]:
    """Trace both graphs into `model_dir`; return the lifted state layout."""
    model_dir.mkdir(parents=True)
    model = dyn_sys.nodes[0].callable
    widths = [len(nm_mapping[key]) for key in ("Y", "U", "D")]
    batch, steps = 2, cfg.in_steps   # batch 2: see trace.export_graph

    slots = probe_state_slots(model, *(torch.randn(batch, steps * n) for n in widths))
    width = sum(slot.width for slot in slots)

    def physical(key: str) -> torch.Tensor:
        cols = nm_mapping[key]
        shift = torch.tensor([norm_stats[c]["shift"] for c in cols])
        stretch = torch.tensor([norm_stats[c]["stretch"] for c in cols])
        return shift + stretch * torch.randn(batch, steps, len(cols))

    window = (physical("Y"), physical("U"), physical("D"))
    names = ["y", "u", "d"]
    outputs = ["y_next", "state_next"] if width else ["y_next"]
    graphs = {
        "warm_start": (WarmStart(model, nm_mapping, norm_stats, slots), window, names),
        "step": (Step(model, nm_mapping, norm_stats, slots),
                 window + ((torch.randn(batch, width),) if width else ()),
                 names + (["state"] if width else [])),
    }
    for name, (graph, args, inputs) in graphs.items():
        path = model_dir / f"{name}.onnx"
        proto = export_graph(graph, args, inputs, outputs, path, exporter)
        # Ties the graph to its model.json; the runtime checks it on load.
        onnx.helper.set_model_props(proto, {
            "export_id": export_id, "opset": str(OPSET), "run_id": run_id,
        })
        onnx.save(proto, str(path))
    return slots


def _write_fixture(data_dir: Path, frame: pd.DataFrame, reference, slots,
                   nm_mapping, norm_stats, in_steps: int,
                   reference_info: dict) -> None:
    """The input window and the trained model's own output and state on it."""
    data_dir.mkdir(parents=True)
    steps = len(frame) - in_steps
    index = frame.index[in_steps:]
    frame.to_csv(data_dir / "example_input.csv", float_format=FLOAT_FORMAT,
                 index_label="timestamp")
    pd.DataFrame(reference.y[:steps], index=index, columns=nm_mapping["Y"]).to_csv(
        data_dir / "expected_output.csv", float_format=FLOAT_FORMAT,
        index_label="timestamp")
    if reference.state.shape[1]:
        state_columns = [f"{slot.path}[{i}]" for slot in slots for i in range(slot.width)]
        pd.DataFrame(reference.state[:steps], index=index, columns=state_columns).to_csv(
            data_dir / "expected_state.csv", float_format=FLOAT_FORMAT,
            index_label="timestamp")

    agreement: Agreement = reference_info["agreement"]
    document = {
        "source": "the trained torch model (NeuroTemplate SystemPreview), "
                  "not the exported graphs",
        "export_id": reference_info["export_id"],
        "run_id": reference_info["run_id"],
        "steps": steps,
        "tolerance": tolerances(nm_mapping["Y"], norm_stats),
        "agreement_at_export": {
            "steps": reference_info["steps_verified"],
            "tolerance_fraction_of_stretch": AGREEMENT_TOL,
            "y_max_abs": agreement.y_error,
            "state_max_abs": agreement.state_error,
        },
        "exported_on": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "neurotemplate": metadata.version("neurotemplate"),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "exporter": reference_info["exporter"],
        },
    }
    (data_dir / "REFERENCE.json").write_text(json.dumps(document, indent=2) + "\n")


def _render(name: str, **values: str) -> str:
    return Template((TEMPLATES / name).read_text()).substitute(values)


def _write_scaffold(stage: Path, spec: dict, package: str, version: str) -> None:
    """The consumer-facing files around `model/` and `data/`."""
    (stage / package).mkdir()
    shutil.copyfile(Path(runtime.__file__), stage / package / "runtime.py")
    (stage / package / "__init__.py").write_text(_render("__init__.py.tmpl"))

    title = spec["description"] or f"{spec['trained_from']['model']} dynamics model"
    values = {"package": package, "title": title, "version": version,
              "distribution": package.replace("_", "-")}
    (stage / "pyproject.toml").write_text(_render("pyproject.toml.tmpl", **values))
    (stage / "validate.py").write_text(_render("validate.py.tmpl", **values))
    (stage / "example.py").write_text(_render("example.py.tmpl", **values))
    (stage / "README.md").write_text(_render(
        "README.md.tmpl", **values, contract=_contract_table(spec),
        history_steps=str(spec["history_steps"]),
        timestep_seconds=f"{spec['timestep_seconds']:g}",
        state_width=str(spec["state"]["width"]),
    ))


def _contract_table(spec: dict) -> str:
    """The README's column table, from model.json."""
    rows = ["| role | column | unit |", "|---|---|---|"]
    for role, key in (("output (y)", "y_columns"), ("control (u)", "u_columns"),
                      ("disturbance (d)", "d_columns")):
        for column in spec[key]:
            rows.append(f"| {role} | `{column}` | {spec['units'][column] or '—'} |")
    return "\n".join(rows)


def _install(stage: Path, out_dir: Path, package: str) -> list[str]:
    """Copy the staged package into `out_dir`; return the files kept as-is."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model", "data", package):
        target = out_dir / name
        if name == "model" and target.exists():
            # A project's model card lives beside the graphs; keep it.
            for item in target.iterdir():
                if item.name != "MODEL_CARD.md":
                    item.unlink()
            shutil.copytree(stage / name, target, dirs_exist_ok=True)
            continue
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(stage / name, target)
    kept = []
    for name in ("README.md", "pyproject.toml", *WRITE_ONCE):
        if name in WRITE_ONCE and (out_dir / name).exists():
            kept.append(name)
            continue
        shutil.copyfile(stage / name, out_dir / name)
    return kept
