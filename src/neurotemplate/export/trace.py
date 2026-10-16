"""
Trace a graph wrapper to ONNX, check it, and audit its operators; or capture
it as a torch program.

The opset is pinned here and recorded in `model.json`; the runtime asserts it
on load, because the opset is a compatibility surface the template now owns.

Both formats drop the debugging record torch's exporter attaches to every
node — the Python stack that produced it, the module path, the FX node. No
runtime reads it. It was most of a shipped file (85-90% of the 2R2C graphs),
and its stack traces carried the exporting machine's absolute paths into
every delivery.

The operator audit is the loud failure for an unexportable construct. Every
forward path in the template is pure tensor code today, and the operators it
traces to are a small, core-ONNX set with no custom kernels — which is what
lets a C++ or Java driver, or an onnxruntime minimal build, run the graph. A
new construct that traces to something outside that set fails here, at
export, rather than on a consumer's machine.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import onnx
import torch
import torch.nn as nn

#: The opset every exported graph targets.
OPSET = 18

#: The exporters `export_graph` accepts. `dynamo` (torch.export) is the
#: supported path; `legacy` is TorchScript tracing, deprecated since torch 2.9
#: but kept because it traces many times faster, which the test suite uses.
EXPORTERS = ("dynamo", "legacy")

#: Every operator the template's forward paths are known to trace to, across
#: all families, heads, the prestack and the policy. Extend it deliberately:
#: a new entry is a new requirement on every consumer's runtime.
ALLOWED_OPS = frozenset({
    "Add", "Cast", "Clip", "Concat", "Constant", "ConstantOfShape", "Div",
    "Equal", "Erf", "Exp", "Expand", "GRU", "Gather", "Gemm", "Greater",
    "Identity", "Max", "Min", "Mul", "Neg", "Range", "Reciprocal", "ReduceMean",
    "ReduceSum", "Relu", "Reshape", "ScatterND", "Shape", "Sigmoid", "Slice",
    "Softplus", "Split", "Squeeze", "Sub", "Tanh", "Transpose", "Unsqueeze",
    "Where",
})


def export_graph(graph: nn.Module, args: tuple[torch.Tensor, ...],
                 input_names: list[str], output_names: list[str],
                 path: Path, exporter: str = "dynamo") -> onnx.ModelProto:
    """Trace `graph` on `args` to a single self-contained `.onnx` file.

    Only the batch axis (axis 0 of every input and output) is dynamic. Every
    other dimension — history length, channel counts, state width — is frozen
    at trace time, and ONNX Runtime rejects a wrong-shaped input by name.
    Trace with a batch of at least 2, or torch.export specialises it to 1.
    """
    if exporter not in EXPORTERS:
        raise ValueError(f"unknown exporter {exporter!r}; expected one of {EXPORTERS}")
    graph.eval()
    path = Path(path)
    with torch.no_grad(), warnings.catch_warnings():
        # torch.onnx is chatty about deprecations and trace-time constants
        # that are, here, deliberate (the block loop unrolls by design).
        warnings.simplefilter("ignore")
        if exporter == "dynamo":
            torch.onnx.export(
                graph, args, str(path),
                input_names=input_names, output_names=output_names,
                dynamic_shapes=_dynamic_batch(args),
                opset_version=OPSET, dynamo=True, external_data=False,
                verbose=False,
            )
        else:
            torch.onnx.export(
                graph, args, str(path),
                input_names=input_names, output_names=output_names,
                dynamic_axes={name: {0: "batch"} for name in input_names + output_names},
                opset_version=OPSET, dynamo=False,
            )
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto, full_check=True)
    audit_operators(proto, path.name)
    strip_debug_info(proto)
    onnx.save(proto, str(path))
    return proto


def export_program(graph: nn.Module, args: tuple[torch.Tensor, ...],
                   input_names: list[str]) -> torch.export.ExportedProgram:
    """Capture `graph` on `args` as a torch program, for `torch.export.save`.

    The same capture the dynamo exporter starts from, stopped before the
    translation to ONNX. The program is called by keyword, `input_names`
    being `graph.forward`'s own parameter names. Only the batch axis is
    dynamic, as in `export_graph`. Not under `no_grad`: a program captured
    there would run without gradients, and being differentiable is much of
    the point of shipping one.
    """
    graph.eval()
    # Cloned: the program saves its example inputs, and a slice of a larger
    # tensor (a policy's two traced windows) would save all of it.
    kwargs = {name: x.clone() for name, x in zip(input_names, args, strict=True)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = torch.export.export(
            graph, (), kwargs=kwargs,
            dynamic_shapes={name: {0: torch.export.Dim.AUTO} for name in kwargs})
    for node in program.graph.nodes:
        node.meta.pop("stack_trace", None)
    return program


def _dynamic_batch(args: tuple[torch.Tensor, ...]) -> tuple[dict, ...]:
    """Axis 0 of every input dynamic, nothing else.

    AUTO rather than a named Dim: nn.GRU guards on `batch != 1`, which a
    strict named Dim refuses outright. AUTO lets the guard through, and the
    resulting graph is correct at batch 1 too (the shipped runtime runs every
    family at batch 1).
    """
    return tuple({0: torch.export.Dim.AUTO} for _ in args)


def strip_debug_info(proto: onnx.ModelProto) -> None:
    """Drop the exporter's per-node debugging record (see the module docstring)."""
    for nodes in [proto.graph.node, *(function.node for function in proto.functions)]:
        for node in nodes:
            node.doc_string = ""
            del node.metadata_props[:]


def operators(proto: onnx.ModelProto) -> set[str]:
    """Every operator type the graph uses, including inside local functions."""
    ops = {node.op_type for node in proto.graph.node}
    for function in proto.functions:
        ops |= {node.op_type for node in function.node}
    return ops


def audit_operators(proto: onnx.ModelProto, label: str) -> None:
    """Refuse a graph that uses an operator outside `ALLOWED_OPS`."""
    unknown = sorted(operators(proto) - ALLOWED_OPS)
    if unknown:
        raise ValueError(
            f"{label} uses operator(s) {unknown} outside the audited set. A "
            "consumer's runtime may not implement them. If they are core "
            "ONNX and genuinely needed, add them to trace.ALLOWED_OPS."
        )
