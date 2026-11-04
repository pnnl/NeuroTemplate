"""
Export trained models as standalone packages: ONNX graphs, torch programs, or C.

The trained graph is serialised, not reimplemented: `graphs` wraps a trained
model so that normalisation, window layout and the hidden state are all
inside the traced graph, `state` lifts hidden state into explicit tensors via
the stateful-block protocol, `trace` exports and audits, `ccode` turns the
ONNX into C, `verify` proves the result reproduces the trained system, and
`package` writes a package around the shipped `runtime` (and, for C, the C
runner in `c/`). Entry points: `nt export model`, `nt export policy`.
"""
