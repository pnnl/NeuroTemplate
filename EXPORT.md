# Exporting

`nt export model` ships a trained surrogate and `nt export policy` a trained
policy, as a standalone package that runs without NeuroTemplate,
Neuromancer or the training code. The export serializes the trained network
rather than reimplementing it. Normalization, the window layout and any
hidden state are inside the exported graphs, so a package takes physical
units in and gives physical units out.

```bash
nt export model  --run_id <model_run>  --data data/test.csv --out delivery --format onnx
nt export policy --run_id <policy_run> --out delivery --format onnx
```

Both may write to the same `--out`, giving one package with a model and a
policy. `nt export model --help` lists the remaining flags.

## Three formats

The format decides what the recipient has to install. A package holds one
format, so a model and a policy sharing a package share it too.

| `--format` | The package needs | Runs from |
|---|---|---|
| `onnx` (default) | numpy and onnxruntime | Python |
| `torch` | numpy and torch | Python, or torch directly |
| `c` | a C compiler; numpy for the Python runner | C, or Python over the compiled C |

**`onnx` is the general-purpose choice.** ONNX is an open standard, and the
graphs use only a small, audited set of its operators.

**`torch` is for people who already have torch.** Each graph is a torch
program (`torch.export`, a `.pt2` file), which loads in plain torch with no
model code and is an ordinary differentiable module. Use it to plan through
the model with gradients, embed it in a larger model, or hand a model to a
collaborator who should not have to install NeuroTemplate at a matching
version. On Linux, a plain `pip install torch` pulls the CUDA build, several
GB; a CPU-only recipient installs from PyTorch's CPU index.

**`c` is for the smallest and fastest deployment.** The graphs are compiled
to plain C by [onnx2c](https://github.com/kraiskil/onnx2c): weights as
constant arrays, the forward pass as straight-line functions, no heap and
no I/O. The package carries a C runner around them, which suits a
microcontroller as well as a host, and a Python runner that calls the same
compiled C through ctypes. The batch is fixed at 1.

Measured on one model and one policy (macOS arm64, Python 3.11): a 2R2C
zone plus an MLP power block, 13,897 parameters, history 12, hidden state
width 1, and a linear residual policy. Rollouts ran over a 2,004-step test
split. Installed sizes exclude the interpreter.

| | onnx | torch | c, from Python | c, from C |
|---|---|---|---|---|
| Installed to run it | 99 MB | 598 MB | 22 MB (numpy) | nothing |
| Model graphs as shipped | 195 KB | 620 KB | 600 KB of C source | 127 KB compiled |
| Model, per rollout step | 48 µs | 343 µs | 13 µs | 9 µs |
| Policy, per command | 16 µs | 93 µs | 5.5 µs | 0.3 µs |

From Python, the runner's own windowing in numpy is most of every step. A
model is small arithmetic, and the rest is per-call overhead, which C
removes. Treat these as orders of magnitude for your own models, not as
figures to quote.

## The C format needs onnx2c

The export calls onnx2c, which you build from source and put on PATH, or
name with `--onnx2c`. It also compiles the generated C with the host's
compiler (`cc`, or `CC`) to verify it. The export never installs either one.

The C format is pinned to one onnx2c commit, `e2ee84b`: upstream plus the
fix for reductions with `keepdims=0`, which is a pull request upstream at
the time of writing. The export reads the commit from `onnx2c -v` and refuses
any other, printing these steps. Building needs cmake and protobuf
(`brew install cmake protobuf`, or your platform's packages):

```bash
git clone https://github.com/EladMichael/onnx2c && cd onnx2c
git checkout e2ee84b0f6451835fc6749b298c85c0af475220c && git submodule update --init
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target onnx2c
```

Then put `build/onnx2c` on PATH. The pin lives in
[export/ccode.py](src/neurotemplate/export/ccode.py) (`ONNX2C_COMMIT`).

## What the package holds

```
delivery/
  README.md        what it is, what to install, how to run it
  runtime.py       the Python runner: Model, Policy
  example.py       runs the model and the policy, and times them
  validate.py      reproduces reference/ on this machine (verified exports only)
  model/           warm_start and step graphs, model.json, reference/
  policy/          the policy graph, policy.json, reference/
```

A C package adds `Makefile`, the C runner (`nt_model.c/.h`,
`nt_policy.c/.h`), `example.c`, `validate.c`, and a generated `model.h` and
`policy.h` holding each artifact's contract as C constants. `make` builds
`./example`, `./validate`, and the library `runtime.py` loads. Nothing is
compiled before it reaches the recipient.

Every file around the artifact directories is regenerated on each export.
Anything else you keep in the package directory is left alone. The graphs
ship without the debugging record torch's exporter attaches to every node,
which carried the exporting machine's file paths.

## Checked before anything is written

- **A model** is rolled out over every row of `--data`, a held-out split,
  once by the trained system and once by the staged package through its
  own runtime. For C, that means the generated code, compiled. The export
  refuses to write unless the two agree.
- **A policy** is run through the trained closed loop on the control run's
  test split. The exported policy must issue the same command for every
  window the trained one saw.

The shipped `reference/` fixtures come from the trained torch model, not
from the exported graphs. So `validate`, on the recipient's machine, checks
the whole chain: trained model, export, runtime and platform. The fixture
holds the whole of `--data` by default; `--fixture_steps` ships a shorter
one, and it is the largest part of most packages.

A model whose own rollout blows up on `--data` cannot be verified, because
float round-off grows with it. The export asks whether to write it anyway,
and with no terminal to ask on the answer is no. Such a model says
`verified: false`, warns when loaded, and ships no fixture and no
`validate`. A model that diverges on held-out data is the thing to worry
about, so the check is not loosened to fit one.

ONNX packages have been verified on Linux aarch64 and x86-64 and on macOS
arm64. Torch and C packages have been verified on macOS arm64 only.

## Using a package

```python
from runtime import Model, Policy

model = Model("model")
forecast = model.predict(frame)       # {output column: array}, open loop after warm-up
state = model.warmup(history)         # the last history_steps rows of every column
state = model.step(state, inputs)     # one step; inputs holds every u and d column

policy = Policy("policy")
command = policy.act(y, d, r)         # {u column: value}
```

That is the same in every format. The generated README states each input's
columns, window length and units, and covers the C runner (C) or direct use
from torch (torch).

## From Python

`neurotemplate.export.dynamics.export_model` and
`neurotemplate.export.policy.export_policy` take the objects
`load_trained_dynamics` and `load_trained_policy` return, with the same
options as the command line (`graph_format`, `exporter`, `onnx2c`, ...).
[export/cli.py](src/neurotemplate/export/cli.py) shows the calls.

## Extending the library

- **The operator list is a contract.** `ALLOWED_OPS` in
  [export/trace.py](src/neurotemplate/export/trace.py) lists every ONNX
  operator the template's models trace to, and an export using any other
  fails. A new entry is a new requirement on every recipient's runtime, and
  for C, onnx2c has to implement it, along with any casts it brings. That is
  why the ordinal head looks its levels up in floats only.
- **A new stateful family exports for free** if it implements
  `rollout_state()` and `set_rollout_state(state)`, exposing its hidden
  state as one `[batch, width]` tensor, the way `RCBlock` and `GRUBlock` do.
- **`--exporter legacy`** traces ONNX with TorchScript instead of
  `torch.export`, for the onnx and c formats. Torch has deprecated it. It
  traces much faster, which the test suite uses.
