"""
Run exported NeuroTemplate models: a dynamics model, a control policy, or both.

This file ships verbatim inside every exported package. It depends on numpy
and the standard library, plus whatever runs the package's graphs
(`graph_format` in the json): onnxruntime for `.onnx` files, torch for `.pt2`
programs, or, for C, the library the package's Makefile builds, through
ctypes. It imports only that one. No neuromancer, no neurotemplate, no
pandas. It knows nothing about RC networks, MLPs or GRUs. Everything
numeric — normalisation, window layout, the network, the hidden state's
seeding — lives inside the graphs; this file only moves windows of physical
values in and out of them.

    model/warm_start.*      (y, u, d)          -> (y_next[, state])
    model/step.*            (y, u, d[, state]) -> (y_next[, state])
    model/model.json        columns, units, history length, state width
    policy/policy.*         (y, d[, r]) -> u
    policy/policy.json      each input's columns and window length
    policy/power.pt2        MPC only: (y, u, d) trajectories -> power

A policy exported from an MPC run has no policy graph: `policy.json` says how
it plans (`mpc`), and it plans through the model graphs beside it, in torch.

Every window is `[batch, steps, n]` in physical units, oldest row first. For
the dynamics model, row -1 of `u`/`d` is the input that drives this step; the
first step of a rollout runs `warm_start`, which seeds any hidden state from
the window, and every later step runs `step` with the state passed explicitly.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

#: The `model.json` layout this runtime reads. Bumped on any change to it.
FORMAT = 3

#: The file inside a `.pt2` that ties it to the json beside it.
STAMP = "neurotemplate_stamp.json"


#: Said on loading a model exported unverified.
UNVERIFIED = ("model/ was exported UNVERIFIED: the trained model blew up on "
              "the verification data, so these graphs were never checked "
              "against it and no reference/ ships.")


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


def _load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec["format"] != FORMAT:
        raise ValueError(
            f"{path} is format {spec['format']}; this runtime reads format "
            f"{FORMAT}. Use the runtime.py shipped alongside it, or re-export.")
    return spec


def _graph(directory: Path, name: str, spec: dict):
    """One graph, as `run({input: array}) -> [output arrays]`. It must come
    from the same export as `spec`."""
    if spec["graph_format"] == "onnx":
        return _onnx_graph(directory / f"{name}.onnx", spec)
    if spec["graph_format"] == "torch":
        return _torch_graph(directory / f"{name}.pt2", spec)
    if spec["graph_format"] == "c":
        return _c_graph(directory, name, spec)
    raise ValueError(f"unknown graph_format {spec['graph_format']!r}.")


def _onnx_graph(path: Path, spec: dict):
    import onnxruntime as ort

    options = ort.SessionOptions()
    # One thread: these graphs are small enough that thread handoff costs
    # more than it saves at batch 1.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Basic optimizations only. The default level fuses Gemm + Relu into a
    # Microsoft-specific operator, which an onnxruntime built with only the
    # graphs' own operators does not have; basic keeps to standard ONNX.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    _check_stamp(path, sess.get_modelmeta().custom_metadata_map,
                 {"export_id": spec["export_id"], "opset": str(spec["opset"])})
    return lambda feed: sess.run(None, feed)


def _torch_module(path: Path, spec: dict):
    """A `.pt2` program as a torch module, checked against `spec`."""
    import torch

    stamp = {STAMP: ""}
    with warnings.catch_warnings():
        # torch's loader reads the weights through a read-only buffer and
        # says so on every load; the weights are copied into the module.
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        module = torch.export.load(str(path), extra_files=stamp).module()
    _check_stamp(path, json.loads(stamp[STAMP] or "{}"),
                 {"export_id": spec["export_id"]})
    return module


def _torch_graph(path: Path, spec: dict):
    import torch

    module = _torch_module(path, spec)

    def run(feed):
        with torch.no_grad():
            out = module(**{name: torch.from_numpy(x) for name, x in feed.items()})
        return [x.numpy() for x in (out if isinstance(out, tuple) else (out,))]
    return run


def _c_graph(directory: Path, name: str, spec: dict):
    """A generated C graph, from the library the package's Makefile builds.

    The C graphs take one window at a time, so a batch runs row by row."""
    import ctypes

    root = directory.resolve().parent
    library_path = root / ("libnt.dylib" if sys.platform == "darwin" else "libnt.so")
    if not library_path.exists():
        raise FileNotFoundError(f"{library_path} not found: run `make` in {root} first.")
    library = ctypes.CDLL(str(library_path))
    stamp = ctypes.c_char_p.in_dll(library, f"nt_{directory.name}_export_id").value
    _check_stamp(library_path, {"export_id": stamp.decode()},
                 {"export_id": spec["export_id"]})
    function = getattr(library, f"nt_{name}_graph")
    function.restype = None
    if "inputs" in spec:                                     # a policy
        shapes = {k: (e["steps"], len(e["columns"])) for k, e in spec["inputs"].items()}
        widths = [len(spec["u_columns"])]
    else:
        h, width = spec["history_steps"], spec["state"]["width"]
        shapes = {k: (h, len(spec[f"{k}_columns"])) for k in ("y", "u", "d")}
        shapes["state"] = (width,)
        widths = [len(spec["y_columns"])] + ([width] if width else [])
    pointer = ctypes.POINTER(ctypes.c_float)

    def run(feed):
        # C reads whatever memory it is pointed at, so check what onnxruntime
        # and torch would: every window's shape.
        for key, x in feed.items():
            if np.shape(x)[1:] != shapes[key]:
                raise ValueError(f"input {key!r} has windows of shape "
                                 f"{np.shape(x)[1:]}; this graph takes {shapes[key]}.")
        inputs = [np.ascontiguousarray(x, dtype=np.float32) for x in feed.values()]
        outputs = [np.empty((len(inputs[0]), width), np.float32) for width in widths]
        for row in range(len(inputs[0])):
            function(*(x[row].ctypes.data_as(pointer) for x in inputs + outputs))
        return outputs
    return run


def _check_stamp(path: Path, stamp: Mapping[str, str], expected: dict) -> None:
    found = {key: stamp.get(key) for key in expected}
    if found != expected:
        raise ValueError(f"{path.name} does not belong to the json beside it: "
                         f"graph has {found}, json expects {expected}. Re-export, "
                         "or for C, rebuild with `make`.")


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
        self.spec = _load_spec(model_dir / "model.json")
        self.y_columns: list[str] = self.spec["y_columns"]
        self.u_columns: list[str] = self.spec["u_columns"]
        self.d_columns: list[str] = self.spec["d_columns"]
        self.history_steps: int = self.spec["history_steps"]
        self.timestep_seconds: float = self.spec["timestep_seconds"]
        self.state_width: int = self.spec["state"]["width"]
        self.verified: bool = self.spec["verified"]
        self._warm_start = _graph(model_dir, "warm_start", self.spec)
        self._step = _graph(model_dir, "step", self.spec)
        if not self.verified:
            warnings.warn(UNVERIFIED, stacklevel=2)

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
        outputs = (self._warm_start if cold else self._step)(feed)
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
            + ("" if self.verified else "\n  " + UNVERIFIED)
        )


class Policy:
    """An exported control policy. `Policy("policy")` loads a policy directory.

    Stateless: one graph, one call per control step. Each input is a window
    of physical values, oldest row first, whose length, columns and alignment
    `policy.json` states (`inputs.<name>.steps` / `.columns` /
    `.current_row`): `y` and `d` end at the current step, and `d` and `r` may
    run past it into the known future (forecast weather; the reference
    columns — a comfort band, a setpoint schedule, a tariff). `u_history`,
    for a policy that reads past commands, holds the commands applied over
    the last `steps` steps and ends the step before the current one.
    """

    def __init__(self, policy_dir: str | Path = "policy"):
        policy_dir = Path(policy_dir)
        self.spec = _load_spec(policy_dir / "policy.json")
        self.inputs: dict[str, dict] = self.spec["inputs"]
        self.u_columns: list[str] = self.spec["u_columns"]
        if "mpc" in self.spec:
            self._planner = _Planner(policy_dir, self.spec)
        else:
            self._graph = _graph(policy_dir, "policy", self.spec)
        self._previous = None   # MPC's last plan, between `act` calls

    def act(self, y, u_history=None, d=None, r=None,
            time_s: Optional[float] = None) -> dict[str, float]:
        """The command for one step: `{u column: value}`.

        Each argument is `[steps, n]` for that input (see `self.inputs`);
        `u_history` only for a policy that reads past commands, `r` only for one
        whose objective names reference columns. `time_s`, for an MPC policy
        only, is the step's deadline: planning stops once that many seconds
        have passed, if it has not converged first.
        """
        windows = {"y": y, "u_history": u_history, "d": d, "r": r}
        feed = self._feed({name: None if windows[name] is None
                           else np.asarray(windows[name])[None] for name in self.inputs})
        if "mpc" in self.spec:
            u, self._previous = self._planner.plan(feed, self._previous, time_s)
        elif time_s is not None:
            raise ValueError("time_s is an MPC policy's planning deadline; this "
                             "policy is one graph call and does not plan.")
        else:
            u = self._graph(feed)[0]
        return {c: float(v) for c, v in zip(self.u_columns, u[0])}

    def act_batch(self, windows: Mapping[str, np.ndarray]) -> np.ndarray:
        """Commands for a batch of windows, `{input: [batch, steps, n]}` ->
        `[batch, n_u]`. For scoring many scenarios at once. An MPC policy
        plans each from scratch here; `act` starts from its last plan."""
        feed = self._feed(windows)
        if "mpc" in self.spec:
            return self._planner.plan(feed, None)[0]
        return self._graph(feed)[0]

    def reset(self) -> None:
        """Forget MPC's last plan, as at the start of a new episode."""
        self._previous = None

    def _feed(self, windows: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [name for name in self.inputs if windows.get(name) is None]
        if missing:
            raise KeyError(f"policy input(s) {missing} not supplied.")
        return {name: np.asarray(windows[name], dtype=np.float32) for name in self.inputs}

    def describe(self) -> str:
        s = self.spec
        shapes = ", ".join(f"{name} {spec['steps']}x{len(spec['columns'])}"
                           for name, spec in self.inputs.items())
        plans = ("" if "mpc" not in s else
                 f"\n  plans {s['mpc']['horizon']} steps ahead through model/, "
                 f"until {s['mpc']['patience']} Adam steps pass without the objective "
                 f"improving by {s['mpc']['rtol']:g} of it (at most "
                 f"{s['mpc']['max_iters']} steps)")
        return (f"{s['description'] or 'NeuroTemplate policy'}  "
                f"(run {s['trained_from']['run_id'][:8]}, {s['trained_from']['policy']})\n"
                f"  commands {len(self.u_columns)} controls from windows {shapes}" + plans)


class _Planner:
    """MPC, as `policy: mpc` runs it in training (`control.mpc`).

    For each window: a free command per step over the next `horizon` steps,
    held as its share of the actuation range in [0, 1], `umin + share *
    (umax - umin)` mapped onto `control_bounds`, rolled through the model
    graphs beside the policy and scored by the run's objective table. Adam
    at `lr` steps the share, clamped back into [0, 1] after every step,
    starting from the last plan shifted by one step (or, without one, from
    the last command applied). Each window stops once `patience` steps pass
    without its objective improving by more than `rtol` of it, and keeps
    the best plan it saw; the solve ends when every window has stopped, at
    `max_iters` steps, or at a deadline the caller passes. The first step's
    command is the action.
    """

    def __init__(self, policy_dir: Path, spec: dict):
        import torch

        self.torch = torch
        m = spec["mpc"]
        self.horizon, self.lr, self.hmax = m["horizon"], m["lr"], m["hmax"]
        self.rtol, self.patience, self.max_iters = m["rtol"], m["patience"], m["max_iters"]
        self.umin, self.umax = torch.tensor(m["umin"]), torch.tensor(m["umax"])
        self.bounds, self.terms = m["control_bounds"], m["terms"]
        model_dir = policy_dir.parent / "model"
        model = _load_spec(model_dir / "model.json")
        if model["graph_format"] != "torch":
            raise ValueError("an MPC policy plans with gradients, through torch "
                             "model graphs; model/ is not torch.")
        self.history, self.stateful = model["history_steps"], model["state"]["width"] > 0
        self.warm_start = _torch_module(model_dir / "warm_start.pt2", model)
        self.step = _torch_module(model_dir / "step.pt2", model)
        self.power = _torch_module(policy_dir / "power.pt2", spec) if m["power"] else None
        columns = {"y": spec["inputs"]["y"]["columns"], "u": spec["u_columns"],
                   "d": spec["inputs"]["d"]["columns"],
                   "r": spec["inputs"]["r"]["columns"] if "r" in spec["inputs"] else []}
        self.where = {c: (role, i) for role, cs in columns.items() for i, c in enumerate(cs)}

    def plan(self, feed: dict[str, np.ndarray], previous, time_s: Optional[float] = None):
        """(first commands [batch, nu], the plan to start the next step from).
        `time_s` stops the solve once that many seconds have passed."""
        torch = self.torch
        deadline = None if time_s is None else time.perf_counter() + time_s
        y, d = torch.from_numpy(feed["y"]), torch.from_numpy(feed["d"])
        applied = torch.from_numpy(feed["u_history"])
        r = torch.from_numpy(feed["r"]) if "r" in feed else None
        if previous is None:
            start = applied[:, -1:].expand(-1, self.horizon, -1)
            share = ((start - self.umin) / (self.umax - self.umin).clamp_min(1e-9)).clamp(0, 1)
        else:
            share = torch.cat([previous[:, 1:], previous[:, -1:]], dim=1)
        share = share.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([share], lr=self.lr)
        n = len(share)
        best = torch.full((n,), torch.inf)
        reference = best.clone()
        best_share = share.detach().clone()
        wait = torch.zeros(n, dtype=torch.long)
        stopped = torch.zeros(n, dtype=torch.bool)
        steps = 0
        while True:          # the library's `control.solver.solve`, step for step
            values = self._objective(share, y, applied, d, r)
            with torch.no_grad():
                v = values.detach()
                better = ~stopped & (v < best)
                best = torch.where(better, v, best)
                best_share[better] = share[better]
                improved = ~stopped & (torch.isinf(reference)
                                       | (v < reference - self.rtol * reference.abs()))
                reference = torch.where(improved, v, reference)
                wait = torch.where(improved, torch.zeros_like(wait), wait + 1)
                stopped |= wait >= self.patience
            if stopped.all() or steps == self.max_iters:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
            optimizer.zero_grad()
            # Summed over windows: each is its own problem (control.solver).
            values.sum().backward()
            optimizer.step()
            steps += 1
            with torch.no_grad():
                share.clamp_(0, 1)
                share[stopped] = best_share[stopped]
        with torch.no_grad():
            u = self._commands(best_share, r)
        return u[:, 0].numpy(), best_share

    def _commands(self, share, r):
        """The plan's commands, [batch, horizon, nu], inside every bound."""
        u = self.umin + share * (self.umax - self.umin)
        columns = list(u.unbind(-1))
        for entry in self.bounds:          # in the order the sides read each other
            j = self.where[entry["column"]][1]
            low = self._side(entry["low"], columns, r)
            high = self.torch.maximum(self._side(entry["high"], columns, r), low)
            share = (columns[j] - self.umin[j]) / (self.umax[j] - self.umin[j])
            columns[j] = low + share * (high - low)
        return self.torch.stack(columns, dim=-1)

    def _side(self, side: dict, columns: list, r):
        if "value" in side:
            return self.torch.full_like(columns[0], side["value"])
        role, i = self.where[side["column"]]
        # A command's bound reads the reference at the command's own step.
        x = columns[i] if role == "u" else r[:, self.hmax - 1:self.hmax - 1 + self.horizon, i]
        return x + side["offset"]

    def _rollout(self, u, y, applied, d):
        """The model's outputs after each planned command, [batch, horizon, ny]."""
        torch, h = self.torch, self.history
        ys = list(y.unbind(1))
        us = torch.cat([applied, u], dim=1)
        state, outputs = None, []
        for k in range(self.horizon):
            now = self.hmax - 1 + k
            feed = {"y": torch.stack(ys[now - h + 1:now + 1], dim=1),
                    "u": us[:, now - h + 1:now + 1], "d": d[:, now - h + 1:now + 1]}
            if state is not None:
                feed["state"] = state
            out = (self.warm_start if k == 0 else self.step)(**feed)
            y_next, state = out if self.stateful else (out, None)
            ys.append(y_next)
            outputs.append(y_next)
        return torch.stack(outputs, dim=1)

    def _objective(self, share, y, applied, d, r):
        """The run's objective per window, [batch]."""
        u = self._commands(share, r)
        ahead = slice(self.hmax, self.hmax + self.horizon)
        series = {"y": self._rollout(u, y, applied, d), "u": u, "d": d[:, ahead],
                  "r": None if r is None else r[:, ahead]}

        def named(name):
            if name == "power":
                return self.power(y=series["y"], u=u, d=series["d"])
            role, i = self.where[name]
            return series[role][..., i]

        total = 0.0
        for term in self.terms:
            h = named(term["column"])
            if term["target"] is not None:
                h = h - named(term["target"])
            if term["map"] == "relu":
                h = self.torch.relu(h)
            elif term["map"] == "nrelu":
                h = self.torch.relu(-h)
            elif term["map"] == "first_difference":
                h = h[:, 1:] - h[:, :-1]
            weight = None if term["weight"] is None else named(term["weight"])
            if weight is not None and term["map"] == "first_difference":
                weight = weight[:, 1:]
            total = total + term["gain"] * _power_mean(self.torch, h, term["norm"], weight)
        return total


def _power_mean(torch, h, q: float, weight):
    """Per window `(mean_t w·|h|^q)^(1/q)`, [batch]: the training
    objective's reduction (`control.losses.window_power_mean`)."""
    a = h.abs() if q == 1 else h.abs() ** q
    if weight is not None:
        a = a * weight
    per_window = a.mean(dim=1)
    if q != 1:
        positive = per_window > 0
        safe = torch.where(positive, per_window, torch.ones_like(per_window))
        per_window = torch.where(positive, safe ** (1.0 / q), torch.zeros_like(per_window))
    return per_window


# ── reproducing the shipped reference ────────────────────────────


def read_csv(path: str | Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """A CSV whose first column is an index -> (index, {column: float64 array})."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    index = [row[0] for row in body]
    columns = {name: np.array([float(row[i]) for row in body])
               for i, name in enumerate(header) if i > 0}
    return index, columns


def window_columns(name: str, entry: Mapping) -> list[str]:
    """The fixture columns holding one `[steps, n]` window of input `name`
    (a `policy.json` input entry), time-major: `y[0].zone T`, ... The C
    validate.c reads them in this order."""
    return [f"{name}[{t}].{column}"
            for t in range(entry["steps"]) for column in entry["columns"]]


def read_windows(path: str | Path, inputs: Mapping[str, Mapping]) -> dict[str, np.ndarray]:
    """Policy windows stored one per row (`policy/reference/inputs.csv`) ->
    `{input: [windows, steps, n]}` float32, ready for `Policy.act_batch`."""
    _, table = read_csv(path)
    return {name: np.stack([table[c] for c in window_columns(name, entry)], axis=1)
                    .astype(np.float32).reshape(-1, entry["steps"], len(entry["columns"]))
            for name, entry in inputs.items()}


@dataclass(frozen=True)
class ReferenceCheck:
    """This machine's answers against the trained model's own, per column."""
    name: str                     # "model" or "policy"
    error: dict[str, float]
    tolerance: dict[str, float]
    state_error: float = 0.0
    state_tolerance: float = 0.0
    # MPC only: the share of windows whose every command is within tolerance,
    # and the share required. Planning through a surrogate with a latch or a
    # clamp can take another path from a one-ulp difference, on a few windows.
    share: Optional[float] = None
    required_share: Optional[float] = None

    @property
    def ok(self) -> bool:
        if self.share is not None:
            return self.share >= self.required_share
        return (all(self.error[c] <= self.tolerance[c] for c in self.error)
                and self.state_error <= self.state_tolerance)

    def report(self) -> str:
        worst = max(self.error, key=lambda c: self.error[c] / self.tolerance[c])
        lines = [f"  {self.name}: worst column {worst}: {self.error[worst]:.2e} "
                 f"(tolerance {self.tolerance[worst]:.0e})"]
        if self.name == "model":
            lines.append(f"  {self.name}: hidden state max |difference| "
                         f"{self.state_error:.2e} (tolerance {self.state_tolerance:.0e})")
        bad = [c for c in self.error if self.error[c] > self.tolerance[c]]
        if bad:
            where = "on some windows" if self.share is not None else ""
            lines.append(f"  {self.name}: out of tolerance {where}: {bad}".replace("  :", ":"))
        if self.state_error > self.state_tolerance:
            lines.append("  the hidden state disagrees: warm-up or step is wrong, "
                         "not just the output layer")
        if self.share is not None:
            lines.append(f"  {self.name}: {self.share * 100:.1f}% of windows within "
                         f"tolerance (required {self.required_share * 100:.0f}%)")
        return "\n".join(lines)


def check_reference(root: str | Path = ".") -> list[ReferenceCheck]:
    """Reproduce every shipped `<artifact>/reference/` with this machine's
    runtime. An artifact without one (a model exported unverified) has
    nothing to reproduce and is not checked.

    The expected files were written by the trained torch model at export
    time, not by these graphs, so passing means the whole chain — trained
    model, export, this runtime, this platform — reproduces the model.
    """
    root = Path(root)
    if not ((root / "model").is_dir() or (root / "policy").is_dir()):
        raise FileNotFoundError(f"{root} holds neither model/ nor policy/.")
    checks = []
    if (root / "model" / "reference").is_dir():
        checks.append(_check_model(root / "model"))
    if (root / "policy" / "reference").is_dir():
        checks.append(_check_policy(root / "policy"))
    return checks


def _check_model(model_dir: Path) -> ReferenceCheck:
    model = Model(model_dir)
    reference = json.loads((model_dir / "reference" / "REFERENCE.json").read_text(encoding="utf-8"))
    _, frame = read_csv(model_dir / "reference" / "example_input.csv")
    _, expected = read_csv(model_dir / "reference" / "expected_output.csv")
    steps = len(next(iter(expected.values())))
    outputs, hidden = model.rollout(frame, steps)

    error = {c: float(np.abs(outputs[:, i] - expected[c]).max())
             for i, c in enumerate(model.y_columns)}
    state_error = 0.0
    if model.state_width:
        _, expected_state = read_csv(model_dir / "reference" / "expected_state.csv")
        target = np.stack(list(expected_state.values()), axis=1)
        state_error = float(np.abs(hidden - target).max())
    tolerance = reference["tolerance"]
    return ReferenceCheck("model", error, tolerance["y"], state_error, tolerance["state"])


def _check_policy(policy_dir: Path) -> ReferenceCheck:
    reference = json.loads((policy_dir / "reference" / "REFERENCE.json").read_text(encoding="utf-8"))
    policy = Policy(policy_dir)
    actions = policy.act_batch(read_windows(policy_dir / "reference" / "inputs.csv",
                                            policy.inputs))
    _, expected = read_csv(policy_dir / "reference" / "expected_actions.csv")
    differences = np.stack([np.abs(actions[:, i] - expected[c])
                            for i, c in enumerate(policy.u_columns)], axis=1)
    error = {c: float(differences[:, i].max()) for i, c in enumerate(policy.u_columns)}
    tolerance = reference["tolerance"]
    if "share" not in tolerance:
        return ReferenceCheck("policy", error, tolerance["u"])
    within = (differences <= np.array([tolerance["u"][c] for c in policy.u_columns])).all(1)
    return ReferenceCheck("policy", error, tolerance["u"], share=float(within.mean()),
                          required_share=tolerance["share"])
