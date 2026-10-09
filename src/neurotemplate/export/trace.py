"""
Trace a graph wrapper to ONNX, check it, and audit its operators.

The opset is pinned here and recorded in `model.json`; the runtime asserts it
on load, because the opset is a compatibility surface the template now owns.

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
#: but kept because it writes noticeably smaller files.
EXPORTERS = ("dynamo", "legacy")

#: Every operator the template's forward paths are known to trace to, across
#: all families, heads, the prestack and the policy. Extend it deliberately:
#: a new entry is a new requirement on every consumer's runtime.
ALLOWED_OPS = frozenset({
    "Add", "Cast", "Ceil", "Clip", "Concat", "Constant", "ConstantOfShape", "Div",
    "Equal", "Erf", "Exp", "Expand", "Floor", "GRU", "Gather", "GatherND", "Gemm",
    "Greater",
    "Identity", "Max", "Min", "Mul", "Neg", "Range", "Reciprocal", "ReduceMean",
    "ReduceSum", "Relu", "Reshape", "Round", "ScatterND", "Shape", "Sigmoid", "Slice",
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
            # AUTO rather than a named Dim: nn.GRU guards on `batch != 1`,
            # which a strict named Dim refuses outright. AUTO lets the guard
            # through, and the resulting graph is correct at batch 1 too
            # (tests/test_export.py runs every family at batch 1).
            batch = torch.export.Dim.AUTO
            torch.onnx.export(
                graph, args, str(path),
                input_names=input_names, output_names=output_names,
                dynamic_shapes=tuple({0: batch} for _ in args),
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
    return proto


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
