"""
Run an exported NeuroTemplate dynamics model.

This file ships verbatim inside every exported package. It depends on numpy,
onnxruntime and the standard library only: no torch, no neuromancer, no
neurotemplate, no pandas. It knows nothing about RC networks, MLPs or GRUs.
Everything numeric — normalisation, window layout, the model, the hidden
state's seeding — lives inside the two graphs; this file only moves windows
of physical values in and out of them.

    model/warm_start.onnx   (y, u, d)        -> (y_next[, state])
    model/step.onnx         (y, u, d[, state]) -> (y_next[, state])
    model/model.json        columns, units, history length, state width

`y`, `u`, `d` are `[1, history_steps, n]` windows in physical units, oldest
row first, and row -1 of `u`/`d` is the input that drives this step. The
first step of a rollout runs `warm_start`, which seeds any hidden state from
the window; every later step runs `step` and carries the state explicitly.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import onnxruntime as ort

#: The `model.json` layout this runtime reads. Bumped on any change to it.
FORMAT = 1


@dataclass(frozen=True)
class State:
    """Everything the model needs to take one more step.

    `outputs` is the latest prediction (or, straight after `warmup`, the last
    measured row), keyed by column. The rest is the model's memory: the
    output window, the inputs that preceded this step, and the hidden state
    (`None` until the first step has run). Treat it as opaque and pass it
    back to `step`.
    """
    outputs: dict[str, float]
    y_window: np.ndarray          # [history_steps, ny]
    u_history: np.ndarray         # [history_steps - 1, nu]
    d_history: np.ndarray         # [history_steps - 1, nd]
    hidden: Optional[np.ndarray]  # [1, state_width], None when cold


def _session(path: Path) -> ort.InferenceSession:
    # One thread: these graphs are small enough that thread handoff costs
    # more than it saves at batch 1.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class Model:
    """An exported dynamics model. `Model("model")` loads a model directory.

      `warmup(history)`      recent measurements -> a State
      `step(state, inputs)`  one timestep -> a new State (the old one is untouched)
      `rollout(frame)`       warm up on the first history_steps rows, then run
                             open loop over the rest -> (outputs, hidden states)
      `predict(frame)`       `rollout`, as {column: array}
    """

    def __init__(self, model_dir: str | Path = "model"):
        model_dir = Path(model_dir)
        self.spec: dict = json.loads((model_dir / "model.json").read_text())
        if self.spec["format"] != FORMAT:
            raise ValueError(
                f"{model_dir}/model.json is format {self.spec['format']}; this "
                f"runtime reads format {FORMAT}. Re-export, or use the "
                "runtime.py shipped alongside that model.")
        self.y_columns: list[str] = self.spec["y_columns"]
        self.u_columns: list[str] = self.spec["u_columns"]
        self.d_columns: list[str] = self.spec["d_columns"]
        self.history_steps: int = self.spec["history_steps"]
        self.timestep_seconds: float = self.spec["timestep_seconds"]
        self.state_width: int = self.spec["state"]["width"]
        self._warm_start = _session(model_dir / "warm_start.onnx")
        self._step = _session(model_dir / "step.onnx")
        for name, sess in (("warm_start", self._warm_start), ("step", self._step)):
            self._check_graph(name, sess)

    def _check_graph(self, name: str, sess: ort.InferenceSession) -> None:
        """The graph must come from the same export as model.json."""
        meta = sess.get_modelmeta().custom_metadata_map
        expected = {"export_id": self.spec["export_id"], "opset": str(self.spec["opset"])}
        found = {key: meta.get(key) for key in expected}
        if found != expected:
            raise ValueError(
                f"{name}.onnx does not belong to this model.json: graph has "
                f"{found}, model.json expects {expected}.")

    # ── one step ─────────────────────────────────────────────────

    def warmup(self, history: Mapping[str, np.ndarray]) -> State:
        """A State from the last `history_steps` rows of recent measurements.

        `history` maps every y, u and d column to a 1-D array, oldest first
        (a pandas DataFrame works). The u and d values in the final row are
        not used: they are the inputs to the first `step`.
        """
        s = self.history_steps
        y = self._columns(history, self.y_columns, "history")
        u = self._columns(history, self.u_columns, "history")
        d = self._columns(history, self.d_columns, "history")
        if len(y) < s:
            raise ValueError(f"warmup needs at least {s} rows of history; got {len(y)}.")
        return self._state(y[-s:], u[-s:-1], d[-s:-1], hidden=None)

    def step(self, state: State, inputs: Mapping[str, float]) -> State:
        """Advance one timestep under this step's inputs (every u and d column)."""
        missing = [c for c in self.u_columns + self.d_columns if c not in inputs]
        if missing:
            raise KeyError(f"step() is missing input column(s) {missing}.")
        u_now = np.array([inputs[c] for c in self.u_columns], dtype=np.float32)
        d_now = np.array([inputs[c] for c in self.d_columns], dtype=np.float32)
        return self._advance(state, u_now, d_now)

    def _advance(self, state: State, u_now: np.ndarray, d_now: np.ndarray) -> State:
        u_window = np.vstack([state.u_history, u_now[None]])
        d_window = np.vstack([state.d_history, d_now[None]])
        feed = {"y": state.y_window[None], "u": u_window[None], "d": d_window[None]}
        cold = state.hidden is None
        if self.state_width and not cold:
            feed["state"] = state.hidden
        outputs = (self._warm_start if cold else self._step).run(None, feed)
        hidden = outputs[1] if self.state_width else np.zeros((1, 0), np.float32)
        y_window = np.vstack([state.y_window[1:], outputs[0]])
        return self._state(y_window, u_window[1:], d_window[1:], hidden)

    def _state(self, y_window, u_history, d_history, hidden) -> State:
        return State(
            outputs={c: float(v) for c, v in zip(self.y_columns, y_window[-1])},
            y_window=y_window, u_history=u_history, d_history=d_history,
            hidden=hidden,
        )

    # ── a whole rollout ──────────────────────────────────────────

    def rollout(self, frame: Mapping[str, np.ndarray],
                steps: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Warm up on the first `history_steps` rows, then run open loop.

        After the warm-up the model feeds on its own predictions and reads
        only the u and d columns, so this is a genuine multi-step forecast.
        Step k predicts row `history_steps + k` from the inputs in the row
        before it. Returns `(outputs [steps, ny], hidden [steps, state_width])`.
        """
        s = self.history_steps
        y = self._columns(frame, self.y_columns, "frame")
        u = self._columns(frame, self.u_columns, "frame")
        d = self._columns(frame, self.d_columns, "frame")
        n = len(y) - s if steps is None else steps
        if n < 1 or s + n > len(y):
            raise ValueError(
                f"frame has {len(y)} rows; cannot predict {n} step(s) after "
                f"{s} rows of warm-up.")
        state = self._state(y[:s], u[:s - 1], d[:s - 1], hidden=None)
        outputs, hidden = [], []
        for k in range(n):
            state = self._advance(state, u[s + k - 1], d[s + k - 1])
            outputs.append(state.y_window[-1])
            hidden.append(state.hidden[0])
        return np.array(outputs), np.array(hidden).reshape(n, self.state_width)

    def predict(self, frame: Mapping[str, np.ndarray],
                steps: Optional[int] = None) -> dict[str, np.ndarray]:
        """`rollout`'s outputs as `{y column: array}`."""
        outputs, _ = self.rollout(frame, steps)
        return {c: outputs[:, i] for i, c in enumerate(self.y_columns)}

    # ── helpers ──────────────────────────────────────────────────

    def _columns(self, frame: Mapping[str, np.ndarray], columns: list[str],
                 what: str) -> np.ndarray:
        """`[rows, len(columns)]` float32, in `columns` order."""
        missing = [c for c in self.y_columns + columns if c not in frame]
        if missing:
            raise KeyError(f"{what} is missing column(s) {missing}.")
        rows = len(frame[self.y_columns[0]])
        return np.array([np.asarray(frame[c], dtype=np.float32) for c in columns],
                        dtype=np.float32).reshape(len(columns), rows).T

    def describe(self) -> str:
        s = self.spec
        minutes = self.history_steps * self.timestep_seconds / 60
        return (
            f"{s['description'] or 'NeuroTemplate model'}  "
            f"(run {s['trained_from']['run_id'][:8]}, {s['trained_from']['model']})\n"
            f"  {len(self.y_columns)} outputs, {len(self.u_columns)} controls, "
            f"{len(self.d_columns)} disturbances\n"
            f"  timestep {self.timestep_seconds:g} s, warm-up {self.history_steps} "
            f"steps ({minutes:g} min), hidden state width {self.state_width}"
        )


# ── reading the shipped data ─────────────────────────────────────


def read_csv(path: str | Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """A CSV whose first column is an index -> (index, {column: float64 array})."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    index = [row[0] for row in body]
    columns = {name: np.array([float(row[i]) for row in body])
               for i, name in enumerate(header) if i > 0}
    return index, columns


@dataclass(frozen=True)
class ReferenceCheck:
    """This machine's rollout against the trained model's own, column by column."""
    y_error: dict[str, float]
    y_tolerance: dict[str, float]
    state_error: float
    state_tolerance: float

    @property
    def ok(self) -> bool:
        return (all(self.y_error[c] <= self.y_tolerance[c] for c in self.y_error)
                and self.state_error <= self.state_tolerance)

    def report(self) -> str:
        worst = max(self.y_error, key=lambda c: self.y_error[c] / self.y_tolerance[c])
        lines = [
            f"  outputs  worst column {worst}: {self.y_error[worst]:.2e} "
            f"(tolerance {self.y_tolerance[worst]:.0e})",
            f"  state    max |difference| {self.state_error:.2e} "
            f"(tolerance {self.state_tolerance:.0e})",
        ]
        bad = [c for c in self.y_error if self.y_error[c] > self.y_tolerance[c]]
        if bad:
            lines.append(f"  outputs out of tolerance: {bad}")
        if self.state_error > self.state_tolerance:
            lines.append("  the hidden state disagrees: warm-up or step is wrong, "
                         "not just the output layer")
        return "\n".join(lines)


def check_reference(root: str | Path = ".") -> ReferenceCheck:
    """Reproduce `data/expected_*.csv` with this machine's onnxruntime.

    The expected files were written by the trained torch model at export
    time, not by these graphs, so passing means the whole chain — trained
    model, export, this runtime, this platform — reproduces the model.
    """
    root = Path(root)
    reference = json.loads((root / "data" / "REFERENCE.json").read_text())
    model = Model(root / "model")
    _, frame = read_csv(root / "data" / "example_input.csv")
    _, expected = read_csv(root / "data" / "expected_output.csv")
    steps = len(next(iter(expected.values())))
    outputs, hidden = model.rollout(frame, steps)

    y_error = {c: float(np.abs(outputs[:, i] - expected[c]).max())
               for i, c in enumerate(model.y_columns)}
    state_error = 0.0
    if model.state_width:
        _, expected_state = read_csv(root / "data" / "expected_state.csv")
        target = np.stack(list(expected_state.values()), axis=1)
        state_error = float(np.abs(hidden - target).max())
    tolerance = reference["tolerance"]
    return ReferenceCheck(y_error, tolerance["y"], state_error, tolerance["state"])
