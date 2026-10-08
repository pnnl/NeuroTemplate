"""
The consumer package every export writes into, and what the exports share.

    <out>/
      README.md            generated from whatever the package holds
      pyproject.toml       generated: numpy + onnxruntime, nothing else
      validate.py          written once; add project-specific checks to it
      example.py           written once
      <package>/           the shipped runtime (export/runtime.py, verbatim)
      cpp/                 the same runtime in C++, plus validate and example
                           programs and their CMakeLists.txt (export/cpp/, verbatim)
      model/               a dynamics model: warm_start.onnx, step.onnx,
                           model.json, reference/   (`nt export model`)
      policy/              a control policy: policy.onnx, policy.json,
                           reference/               (`nt export policy`)

An export builds its artifact directory in a staging directory, and installs
it only once the staged package passes its own reference check — so a failed
export leaves `<out>` untouched. Installing replaces that one artifact
directory and regenerates the generated files and both runtimes; the other
artifact, the project's `validate.py` / `example.py`, and anything else the
project keeps at the top level (a `MODEL_CARD.md`, say) are left alone.
"""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from string import Template

import onnx
import onnxruntime
import torch

from neurotemplate.export import runtime
from neurotemplate.export.trace import OPSET

TEMPLATES = Path(__file__).parent / "templates"
#: The C++ runner, shipped verbatim into `<out>/cpp/`.
CPP = Path(__file__).parent / "cpp"

#: Written into every shipped CSV. `%.6f` does not round-trip float32 (it
#: keeps 83 of 1000 normalised-state values and 493 of 1000 kW values exact);
#: `%.9g` round-trips all of them, so the file format never sets the
#: tolerance floor.
FLOAT_FORMAT = "%.9g"

#: Top-level files the project owns after the first export.
WRITE_ONCE = ("validate.py", "example.py")


def check_package_name(package: str) -> None:
    if not package.isidentifier():
        raise ValueError(f"package name {package!r} is not a Python identifier.")


def stamp(proto: onnx.ModelProto, path: Path, export_id: str, run_id: str) -> None:
    """Tie a graph to the json written beside it; the runtime checks this on load."""
    onnx.helper.set_model_props(proto, {
        "export_id": export_id, "opset": str(OPSET), "run_id": run_id,
    })
    onnx.save(proto, str(path))


def provenance(exporter: str) -> dict:
    """Where and with what a reference was produced, for REFERENCE.json."""
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "neurotemplate": metadata.version("neurotemplate"),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "exporter": exporter,
    }


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def install(stage: Path, out_dir: Path, artifact: str, package: str,
            version: str) -> list[str]:
    """Check the staged `artifact` directory, then move it into `out_dir`.

    Returns the project-owned files that were kept as they were.
    """
    failed = [check for check in runtime.check_reference(stage) if not check.ok]
    if failed:
        raise RuntimeError("REFUSING TO WRITE: the staged package fails its own "
                           "reference check:\n" + "\n".join(c.report() for c in failed))

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir / artifact, ignore_errors=True)
    shutil.copytree(stage / artifact, out_dir / artifact)
    return _write_package_files(out_dir, package, version)


def _render(name: str, **values: str) -> str:
    return Template((TEMPLATES / name).read_text(encoding="utf-8")).substitute(values)


def _write_runtime(root: Path, package: str) -> None:
    (root / package).mkdir(exist_ok=True)
    shutil.copyfile(Path(runtime.__file__), root / package / "runtime.py")
    (root / package / "__init__.py").write_text(_render("__init__.py.tmpl"),
                                                encoding="utf-8")
    # File by file, so a build directory someone keeps inside cpp/ survives.
    (root / "cpp").mkdir(exist_ok=True)
    for source in CPP.iterdir():
        shutil.copyfile(source, root / "cpp" / source.name)


def _write_package_files(out_dir: Path, package: str, version: str) -> list[str]:
    """Everything around the artifact directories, from what `out_dir` holds."""
    _write_runtime(out_dir, package)
    specs = {name: json.loads((out_dir / name / f"{name}.json").read_text(encoding="utf-8"))
             for name in ("model", "policy") if (out_dir / name).is_dir()}
    title = next((s["description"] for s in specs.values() if s["description"]),
                 "NeuroTemplate export")
    values = {"package": package, "title": title, "version": version,
              "distribution": package.replace("_", "-")}

    sections = []
    if "model" in specs:
        sections.append(_render("README_model.md.tmpl", **values,
                                contract=_model_contract(specs["model"]),
                                history_steps=str(specs["model"]["history_steps"]),
                                timestep_seconds=f"{specs['model']['timestep_seconds']:g}",
                                state_width=str(specs["model"]["state"]["width"])))
    if "policy" in specs:
        sections.append(_render("README_policy.md.tmpl", **values,
                                contract=_policy_contract(specs["policy"])))
    (out_dir / "README.md").write_text(
        _render("README.md.tmpl", **values, sections="\n\n".join(sections)),
        encoding="utf-8")
    (out_dir / "pyproject.toml").write_text(_render("pyproject.toml.tmpl", **values),
                                          encoding="utf-8")

    kept = []
    for name in WRITE_ONCE:
        if (out_dir / name).exists():
            kept.append(name)
        else:
            (out_dir / name).write_text(_render(f"{name}.tmpl", **values),
                                       encoding="utf-8")
    return kept


def _model_contract(spec: dict) -> str:
    rows = ["| role | column | unit |", "|---|---|---|"]
    for role, key in (("output (y)", "y_columns"), ("control (u)", "u_columns"),
                      ("disturbance (d)", "d_columns")):
        for column in spec[key]:
            rows.append(f"| {role} | `{column}` | {spec['units'][column] or '—'} |")
    return "\n".join(rows)


def _policy_contract(spec: dict) -> str:
    rows = ["| input | steps | current row | columns |", "|---|---|---|---|"]
    for name, entry in spec["inputs"].items():
        columns = ", ".join(f"`{c}`" for c in entry["columns"])
        rows.append(f"| `{name}` | {entry['steps']} | {entry['current_row']} | {columns} |")
    rows.append(f"| **output** `u` | | | {', '.join(f'`{c}`' for c in spec['u_columns'])} |")
    return "\n".join(rows)
