"""
Export trained models as ONNX packages that run without torch.

The trained graph is serialised, not reimplemented: `graphs` wraps a trained
`SSMGrad` so that normalisation, window layout and the hidden state are all
inside the traced graph, `state` lifts hidden state into explicit tensors via
the stateful-block protocol, `trace` exports and audits, `verify` proves the
result reproduces the trained system, and `package` writes a consumer
package around the shipped `runtime` and its C++ twin in `cpp/`. Entry
points: `nt export model`, `nt export policy`.
"""
