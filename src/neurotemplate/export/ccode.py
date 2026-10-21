"""
The C format: each ONNX graph compiled to C by onnx2c, plus the contract as
a C header and the shipped C runtime (`export/c/`).

onnx2c (https://github.com/kraiskil/onnx2c) turns one ONNX graph into one C
file: the weights as constant arrays, the forward pass as straight-line
functions, no heap and no I/O. It is the user's to build; the export only
looks for it, at the commit `ONNX2C_COMMIT` names, and refuses any other. It
needs a static graph, so `prepare` fixes the batch at 1 and lets onnxruntime
fold everything that is then constant, and drops two attributes the dynamo
exporter writes that onnx2c refuses to parse.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort

#: The shipped C runtime: the drivers, validate.c and example.c.
SOURCES = Path(__file__).parent / "c"

#: The onnx2c the C format is verified against: upstream 271d0e5 plus the fix
#: for keepdims=0 reductions, which upstream onnx2c declares with the wrong
#: shape (a pull request upstream at the time of writing). docs/EXPORT.md repeats
#: both; change them together, and move to upstream once the fix is merged.
ONNX2C_REPOSITORY = "https://github.com/EladMichael/onnx2c"
ONNX2C_COMMIT = "e2ee84b0f6451835fc6749b298c85c0af475220c"

#: Digits onnx2c prints each weight with. Nine round-trips any float32
#: exactly; onnx2c's default of 20 only makes the source longer.
PRECISION = 9


def library_name() -> str:
    """The shared library `make` builds and the Python runtime loads."""
    return "libnt.dylib" if sys.platform == "darwin" else "libnt.so"


def check_tools(onnx2c: str) -> None:
    """Fail fast, before any tracing, when a tool the C format needs is
    missing, or onnx2c is not the pinned commit."""
    short = ONNX2C_COMMIT[:7]
    needed = (f"Exporting with --format c requires onnx2c at commit {short} on "
              f"PATH, or named with --onnx2c. To build it (needs cmake and "
              f"protobuf):\n"
              f"  git clone {ONNX2C_REPOSITORY} && cd onnx2c\n"
              f"  git checkout {ONNX2C_COMMIT} && git submodule update --init\n"
              f"  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && "
              f"cmake --build build --target onnx2c\n"
              f"then put build/onnx2c on PATH.")
    if shutil.which(onnx2c) is None:
        raise FileNotFoundError(f"onnx2c not found ({onnx2c!r}). {needed}")
    version = subprocess.run([onnx2c, "-v"], capture_output=True, text=True).stdout
    found = re.search(r"Commit: (\w+)", version)
    if found is None or not ONNX2C_COMMIT.startswith(found.group(1)):
        raise RuntimeError(f"{onnx2c} reports {version.strip()!r}, not commit "
                           f"{short}. {needed}")
    if shutil.which(compiler()) is None:
        raise FileNotFoundError(f"no C compiler ({compiler()!r}); the C format "
                                "compiles its graphs to verify them. Set CC.")


def compiler() -> str:
    return os.environ.get("CC", "cc")


def generate(onnx_path: Path, c_path: Path, function: str, onnx2c: str) -> None:
    """Write `c_path`: the graph in `onnx_path` as one C function, `function`."""
    with tempfile.TemporaryDirectory() as tmp:
        static = Path(tmp) / "static.onnx"
        prepare(onnx_path, static)
        # Only the `unionize` pass (shared buffers for intermediate tensors):
        # `fold_casts` folds casts of constants, which `prepare` already did,
        # and crashes on a cast from bool.
        result = subprocess.run(
            [onnx2c, "-f", function, "-P", str(PRECISION), "-p", "unionize", "-l", "1",
             str(static)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"onnx2c failed on {onnx_path.name} (exit "
                           f"{result.returncode}):\n{result.stderr[-2000:]}")
    c_path.write_text(result.stdout, encoding="utf-8")


def prepare(source: Path, target: Path) -> None:
    """`source` with the batch fixed at 1 and every constant folded, as onnx2c needs."""
    proto = onnx.load(str(source))
    for value in list(proto.graph.input) + list(proto.graph.output):
        value.type.tensor_type.shape.dim[0].dim_value = 1
    del proto.graph.value_info[:]       # inferred for a dynamic batch
    with tempfile.TemporaryDirectory() as tmp:
        fixed = Path(tmp) / "fixed.onnx"
        onnx.save(proto, str(fixed))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(target)
        ort.InferenceSession(str(fixed), options, providers=["CPUExecutionProvider"])
    proto = onnx.load(str(target))
    for node in proto.graph.node:
        # allowzero only changes a Reshape whose target shape holds a 0, which
        # a static graph's never does; noop_with_empty_axes=0 is the default.
        # onnx2c refuses both rather than ignoring them.
        keep = [a for a in node.attribute
                if not (node.op_type == "Reshape" and a.name == "allowzero")
                and not (a.name == "noop_with_empty_axes" and a.i == 0)]
        del node.attribute[:]
        node.attribute.extend(keep)
    del proto.graph.value_info[:]
    onnx.save(proto, str(target))


def model_header(spec: dict, y_tolerance: dict[str, float], state_tolerance: float) -> str:
    """`model/model.h`: the model's contract as C constants, and its two graphs."""
    ny, nu, nd = (len(spec[k]) for k in ("y_columns", "u_columns", "d_columns"))
    h, width = spec["history_steps"], spec["state"]["width"]
    window = f"[1][{h}]"
    state_in = f", const float state[1][{width}]" if width else ""
    state_out = f", float state_next[1][{width}]" if width else ""
    stats = spec["norm_stats"]
    lines = [
        "/* Generated by NeuroTemplate's export: the contract of model/. */",
        "#ifndef NT_MODEL_CONTRACT_H",
        "#define NT_MODEL_CONTRACT_H",
        "",
        f'#define NT_MODEL_EXPORT_ID "{spec["export_id"]}"',
        f"#define NT_MODEL_NY {ny}",
        f"#define NT_MODEL_NU {nu}",
        f"#define NT_MODEL_ND {nd}",
        f"#define NT_MODEL_HISTORY {h}",
        f"#define NT_MODEL_STATE {width}",
        "#define NT_MODEL_STATE_SLOTS " + str(max(width, 1)) + "  /* STATE, or 1 when 0 */",
        f"#define NT_MODEL_TIMESTEP_SECONDS {spec['timestep_seconds']!r}",
        f"#define NT_MODEL_VERIFIED {int(spec['verified'])}",
        "",
        _strings("NT_MODEL_Y_COLUMNS", "NT_MODEL_NY", spec["y_columns"]),
        _strings("NT_MODEL_U_COLUMNS", "NT_MODEL_NU", spec["u_columns"]),
        _strings("NT_MODEL_D_COLUMNS", "NT_MODEL_ND", spec["d_columns"]),
        "/* Each column's mean in the training data: inputs for an example. */",
        _floats("NT_MODEL_Y_TYPICAL", "NT_MODEL_NY", [stats[c]["shift"] for c in spec["y_columns"]]),
        _floats("NT_MODEL_U_TYPICAL", "NT_MODEL_NU", [stats[c]["shift"] for c in spec["u_columns"]]),
        _floats("NT_MODEL_D_TYPICAL", "NT_MODEL_ND", [stats[c]["shift"] for c in spec["d_columns"]]),
        "/* How closely validate.c must reproduce reference/, in physical units. */",
        _floats("NT_MODEL_Y_TOLERANCE", "NT_MODEL_NY", [y_tolerance[c] for c in spec["y_columns"]]),
        f"#define NT_MODEL_STATE_TOLERANCE {state_tolerance!r}",
        "",
        "/* The graphs, generated by onnx2c. The batch is fixed at 1. */",
        f"void nt_warm_start_graph(const float y{window}[{ny}], const float u{window}[{nu}],",
        f"                         const float d{window}[{nd}], float y_next[1][{ny}]{state_out});",
        f"void nt_step_graph(const float y{window}[{ny}], const float u{window}[{nu}],",
        f"                   const float d{window}[{nd}]{state_in}, float y_next[1][{ny}]{state_out});",
        "",
        "#endif",
    ]
    return "\n".join(lines) + "\n"


def policy_header(spec: dict, typical: dict[str, list[float]],
                  u_tolerance: dict[str, float]) -> str:
    """`policy/policy.h`: the policy's contract as C constants, and its graph.
    `typical` is each input's per-column mean in the training data."""
    inputs = spec["inputs"]
    lines = [
        "/* Generated by NeuroTemplate's export: the contract of policy/. */",
        "#ifndef NT_POLICY_CONTRACT_H",
        "#define NT_POLICY_CONTRACT_H",
        "",
        f'#define NT_POLICY_EXPORT_ID "{spec["export_id"]}"',
        f"#define NT_POLICY_NU {len(spec['u_columns'])}",
        f"#define NT_POLICY_HAS_U_HISTORY {int('u_history' in inputs)}",
        f"#define NT_POLICY_HAS_R {int('r' in inputs)}",
    ]
    for name, entry in inputs.items():
        upper = name.upper()
        lines += [f"#define NT_POLICY_{upper}_STEPS {entry['steps']}",
                  f"#define NT_POLICY_{upper}_N {len(entry['columns'])}",
                  f"#define NT_POLICY_{upper}_CURRENT_ROW {entry['current_row']}"]
    lines += [""]
    for name, entry in inputs.items():
        lines.append(_strings(f"NT_POLICY_{name.upper()}_COLUMNS",
                              f"NT_POLICY_{name.upper()}_N", entry["columns"]))
    lines.append("/* Each column's mean in the training data: inputs for an example. */")
    for name in inputs:
        lines.append(_floats(f"NT_POLICY_{name.upper()}_TYPICAL",
                             f"NT_POLICY_{name.upper()}_N", typical[name]))
    lines += [
        _strings("NT_POLICY_U_COLUMNS", "NT_POLICY_NU", spec["u_columns"]),
        "/* How closely validate.c must reproduce reference/, in physical units. */",
        _floats("NT_POLICY_U_TOLERANCE", "NT_POLICY_NU",
                [u_tolerance[c] for c in spec["u_columns"]]),
        "",
        "/* The graph, generated by onnx2c. The batch is fixed at 1. */",
    ]
    args = [f"const float {name}[1][{entry['steps']}][{len(entry['columns'])}]"
            for name, entry in inputs.items()]
    lines += [f"void nt_policy_graph({', '.join(args)}, float u[1][{len(spec['u_columns'])}]);",
              "", "#endif"]
    return "\n".join(lines) + "\n"


def _strings(name: str, size: str, values: list[str]) -> str:
    body = ", ".join(_c_string(v) for v in values)
    return f"static const char *const {name}[{size}] = {{{body}}};"


def _floats(name: str, size: str, values: list[float]) -> str:
    body = ", ".join(f"{float(v)!r}f" for v in values)
    return f"static const float {name}[{size}] = {{{body}}};"


def _c_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def compile_library(root: Path, artifact: str) -> None:
    """Build `root`'s shared library from one artifact's graphs and driver, as
    the package Makefile would, so the export can verify what it ships."""
    sources = sorted((root / artifact).glob("*.c")) + [SOURCES / f"nt_{artifact}.c"]
    command = [compiler(), "-O2", "-shared", "-fPIC", f"-I{root}", f"-I{SOURCES}",
               "-o", str(root / library_name()), *map(str, sources), "-lm"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"compiling the generated C failed:\n{result.stderr[-3000:]}")
