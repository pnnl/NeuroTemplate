"""
Export: the exported package must BE the trained model.

Every case goes through `export_model`, which rolls the trained
`SystemPreview` and the exported package (through the shipped runtime) out
over a whole synthetic split and refuses to write unless they agree, then
runs the package's own reference check. So one export per case covers the
graphs, the state lifting, the physical-units window layout and the runtime
together. Every mapping has keys with more than one column, because a
channel-major window layout agrees with the real (time-major) one only when
a key has a single column.

Cases: each family, the per-block split, each head type, and the
heterogeneous rc_network + mlp composition with a hinge and a mixed head stack
— the case the old hand-written deliveries could not express. That one runs
under the default (dynamo) exporter in every format (ONNX, torch programs,
C); the rest use the faster legacy tracer, which traces the same forward
code. The ONNX graphs must ship without the exporter's per-node debugging
record, and a torch program must load without NeuroTemplate.

Plus two failure modes that would ship a wrong model silently: a step graph
with its hidden state baked in as a constant, and an export that disagrees
with the trained model being written anyway; and a model that blows up on
the split, which exports unverified only when the user says so, and then
ships nothing to validate. And the shipped C runner, built and run against a
package holding both a model and a policy, then an unverified model.

The C cases need onnx2c (with the keepdims=0 reduction fix) and make on
PATH, and are skipped without them.
"""

import json
import shutil
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest
import torch

from neurotemplate.export import dynamics as export_dynamics
from neurotemplate.export.graphs import Step
from neurotemplate.export.dynamics import export_model
from neurotemplate.export.runtime import Model, check_reference
from neurotemplate.export.state import probe_state_slots
from neurotemplate.export.trace import export_graph
from neurotemplate.export.verify import Rollout
from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.contract import contract_model
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.heads import (
    BoundedHead, HeadStack, RegressionHead,
)
from model_fixtures import model_cfg, uniform_past

needs_c = pytest.mark.skipif(
    shutil.which("onnx2c") is None or shutil.which("make") is None,
    reason="needs onnx2c and make")

HISTORY_STEPS = 4
ROWS = 60

MAPPING = {
    "Y": ["north T", "south T"],
    "U": ["north sp", "south sp", "ahu fan"],
    "D": ["oat", "ghi"],
    "t": ["sec_elapsed"],
}
SPLIT = {
    "north": {"Y": ["north T"], "U": ["north sp"]},
    "south": {"Y": ["south T"], "U": ["south sp"]},
}
MIXED_MAPPING = {
    "Y": ["north T", "north Q", "south T", "power"],
    "U": ["north sp", "south sp"],
    "D": ["oat", "ghi"],
    "t": ["sec_elapsed"],
}
#: North is a declared network behind a thermostat block; south an mlp zone whose
#: setpoint reaches it through a hinge; power an mlp meter on north's heat.
MIXED = {
    "hvac": {
        "Y": ["north Q"], "U": ["north sp"], "D": [], "reads": ["north T"], "model": "thermostat",
        "params": {"thermostat": {"cooling": {"setpoint": "north sp", "gain": [1.0, 5.0],
                                              "capacity": [2.0, 8.0], "offset": [-0.5, 0.5]},
                                  "form": "tracker+offset", "gain_limit": 12.0}},
    },
    "north": {
        "Y": ["north T"], "U": [], "D": ["oat", "ghi"], "driven_by": ["north Q"],
        "model": "rc_network", "ambient": "oat", "solar": "ghi", "scale": {"north Q": 1.0},
        "params": {
            "rc_network": {"nodes": {"Ti": [0.5, 2.0], "Tm": [5.0, 20.0]},
                           "edges": {"Ti-Tm": [0.2, 0.8], "Ti-ambient": [2.0, 8.0]},
                           "inputs": {"north Q": {"Ti": 1.0}, "ghi": {"Ti": [0.001, 0.01]}},
                           "noise": {"Ti": 0.1, "Tm": 0.05, "sensor": 0.05}}},
    },
    "south": {"Y": ["south T"], "U": ["south sp"], "model": "mlp",
              "inputs": {"south sp": {"target": "south T", "map": "nrelu"}}},
    "power": {"Y": ["power"], "U": [], "reads": ["north Q"], "model": "mlp"},
}


UNIFORM = uniform_past(MAPPING, HISTORY_STEPS)
#: The thermal blocks' reads are not in the table; the mlp blocks' are.
MIXED_HISTORY = uniform_past(MIXED_MAPPING, HISTORY_STEPS, at_zero=("north sp", "north T"))


def make_cfg(**over) -> Config:
    return model_cfg(**{"model": "nssm", "past": UNIFORM, "steps_per_hour": 12.0,
                        **over})


def split(mapping) -> tuple[pd.DataFrame, dict]:
    """A random physical-units split over the mapping, and its norm stats."""
    rng = np.random.default_rng(0)
    columns = mapping["Y"] + mapping["U"] + mapping["D"]
    frame = pd.DataFrame(
        {c: 20.0 + 3 * i + rng.normal(0, 1 + i, ROWS) for i, c in enumerate(columns)},
        index=pd.date_range("2025-07-01", periods=ROWS, freq="5min"),
    )
    stats = {c: {"shift": float(frame[c].mean()), "stretch": float(frame[c].std()),
                 "min": float(frame[c].min()), "max": float(frame[c].max())}
             for c in columns}
    return frame, stats


def export(tmp_path, cfg, mapping, block_spec=None, heads=None, exporter="legacy",
           **options):
    frame, stats = split(mapping)
    torch.manual_seed(0)
    head_stack = HeadStack(heads, mapping["Y"]) if heads else None
    dyn_sys = get_dynamics(cfg, mapping, stats, head_stack=head_stack,
                           block_spec=block_spec)
    return export_model(
        dyn_sys=dyn_sys, cfg=cfg, nm_mapping=mapping, norm_stats=stats,
        frame=frame, timestep_seconds=300.0, run_id="0" * 32,
        out_dir=tmp_path / "pkg", exporter=exporter, **options,
    )


def make(package):
    """Build a C package's runner and library, as its README says to."""
    subprocess.run(["make"], cwd=package, check=True, capture_output=True)


CASES = {
    "nssm": (make_cfg(model="nssm", past=UNIFORM), MAPPING, None, None),
    "gru, 2 layers": (make_cfg(model="gru", num_layers=2, past=UNIFORM), MAPPING, None, None),
    "mlp per block": (make_cfg(model="mlp", past=UNIFORM), MAPPING, SPLIT, None),
    "bounded heads": (make_cfg(model="nssm", past=UNIFORM), MAPPING, None,
                      lambda: [BoundedHead(lo=0.0, hi=float("inf")),
                               BoundedHead(lo=-1.0, hi=100.0)]),
}


@pytest.mark.parametrize("name", CASES)
def test_every_family_and_head_exports_as_the_trained_model(tmp_path, name):
    cfg, mapping, block_spec, heads = CASES[name]
    result = export(tmp_path, cfg, mapping, block_spec, heads() if heads else None)
    assert result.steps_verified == ROWS - HISTORY_STEPS
    assert all(check.ok for check in check_reference(result.out_dir))


@pytest.mark.parametrize("graph_format", [
    "onnx", "torch", pytest.param("c", marks=needs_c)])
def test_heterogeneous_model_with_a_hinge_and_heads_exports(tmp_path, graph_format):
    """The exit criterion: rc_network + mlp, a hinge, a bounded head."""
    result = export(tmp_path, make_cfg(past=MIXED_HISTORY), MIXED_MAPPING, MIXED,
                    [RegressionHead()] * 3 + [BoundedHead(0.0, float("inf"))], exporter="dynamo",
                    graph_format=graph_format)
    assert result.state_width == 2                  # the network's two nodes
    if graph_format == "c":
        make(result.out_dir)                        # the Python runner's library
    assert all(check.ok for check in check_reference(result.out_dir))
    # The package drives itself from its own files, with no training code.
    model = Model(result.out_dir / "model")
    assert model.y_columns == MIXED_MAPPING["Y"]

    model_dir = result.out_dir / "model"
    if graph_format == "onnx":
        # The exporter's stack traces carry this machine's paths: never shipped.
        for path in model_dir.glob("*.onnx"):
            assert not any(node.metadata_props or node.doc_string
                           for node in onnx.load(str(path)).graph.node)
    elif graph_format == "torch":
        loads = ("import sys, torch\n"
                 f"torch.export.load({str(model_dir / 'step.pt2')!r}).module()\n"
                 "assert not any(m.startswith('neurotemplate') for m in sys.modules)")
        subprocess.run([sys.executable, "-c", loads], check=True, cwd=tmp_path)


def test_step_graph_follows_the_state_it_is_handed(tmp_path):
    """A trace that does not lift the GRU's hidden state bakes it in as a
    constant: the graph then answers the same whatever state it is given.
    The exported step must track torch for two different states."""
    cfg = make_cfg(model="gru", past=UNIFORM)
    frame, stats = split(MAPPING)
    torch.manual_seed(0)
    model = contract_model(get_dynamics(cfg, MAPPING, stats)).eval()
    windows = [torch.randn(2, HISTORY_STEPS, len(MAPPING[k])) for k in ("Y", "U", "D")]
    slots = probe_state_slots(model, *windows)
    width = sum(slot.width for slot in slots)
    graph = Step(model, slots)
    path = tmp_path / "step.onnx"
    export_graph(graph, (*windows, torch.randn(2, width)), ["y", "u", "d", "state"],
                 ["y_next", "state_next"], path, exporter="legacy")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    answers = []
    for seed in (1, 2):
        state = torch.randn(2, width, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            expected, _ = graph(*windows, state)
        got, _ = sess.run(None, {"y": windows[0].numpy(), "u": windows[1].numpy(),
                                 "d": windows[2].numpy(), "state": state.numpy()})
        np.testing.assert_allclose(got, expected.numpy(), atol=1e-5)
        answers.append(got)
    assert np.abs(answers[0] - answers[1]).max() > 1e-3


def test_an_export_that_disagrees_is_not_written(tmp_path, monkeypatch):
    real = export_dynamics.torch_rollout

    def off_by_a_degree(*args):
        reference = real(*args)
        return Rollout(y=reference.y + 1.0, state=reference.state)

    monkeypatch.setattr(export_dynamics, "torch_rollout", off_by_a_degree)
    with pytest.raises(RuntimeError, match="REFUSING TO WRITE"):
        export(tmp_path, make_cfg(), MAPPING)
    assert not (tmp_path / "pkg").exists()


def _blow_up(monkeypatch):
    """Make the trained model's own rollout go non-finite, as a diverging
    model's does."""
    real = export_dynamics.torch_rollout

    def diverging(*args):
        reference = real(*args)
        y = reference.y.copy()
        y[-1] = np.nan
        return Rollout(y=y, state=reference.state)

    monkeypatch.setattr(export_dynamics, "torch_rollout", diverging)


# ── policy ───────────────────────────────────────────────────────


def _d_future(steps: int) -> dict[str, int]:
    from control_fixtures import DIST
    return {c: steps for c in DIST}


def _priced(d_future: int, **overrides):
    """A policy reading the band and a tariff, each previewed differently."""
    from control_fixtures import (
        BAND_GAINS, BAND_LOSSES, LOW, POWER_LOSSES, TARIFF, TARIFF_SCALE,
    )
    return dict(losses={**BAND_LOSSES, "cost": POWER_LOSSES["cost"]},
                gains={**BAND_GAINS, "cost": 1.0},
                reference_scale={TARIFF: TARIFF_SCALE},
                future={TARIFF: 3, LOW: 1, **_d_future(d_future)}, **overrides)


@pytest.mark.parametrize("overrides, graph_format", [
    # Plain policy: its D window runs past the present (preview), it reads
    # the last two commands, and every key has several columns, so a wrong
    # window layout shows up.
    ({"future": _d_future(2), "u_past": 2}, "torch"),
    # Everything a policy can wrap around its net: a range read per step, and a
    # tariff it previews.
    (_priced(1, control_bounds={"Zone Cooling Setpoint [°C]": ["Comfort Low [°C]",
                                                               "Comfort High [°C]"]}), "onnx"),
], ids=["plain, torch", "per-step bounds, priced, onnx"])
def test_trained_policy_exports_as_the_trained_policy(tmp_path, overrides, graph_format):
    from control_fixtures import full_setup
    from neurotemplate.control.closed_loop import build_closed_loop
    from neurotemplate.control.losses import resolve_objective
    from neurotemplate.control.policy import build_policy
    from neurotemplate.export.policy import (
        closed_loop_windows, export_policy, physical_windows, policy_graph, policy_inputs,
    )

    s = full_setup(**overrides)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    closed_loop = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    result = export_policy(
        policy_node=policy, cl_system=closed_loop, cfg=cfg, sysid_cfg=sysid_cfg,
        nm_mapping=nm, norm_stats=ns, test_dfs=[s["frame"]], run_id="0" * 32,
        out_dir=tmp_path / "pkg", graph_format=graph_format,
        exporter="legacy" if graph_format == "onnx" else "dynamo",
    )
    assert all(check.ok for check in check_reference(result.out_dir))
    # The reference columns are named inputs, in the order the policy reads
    # them; the past commands are `u_history`, ending the step before the current one.
    spec = json.loads((result.out_dir / "policy" / "policy.json").read_text())
    reads_u = "u_past" in overrides
    assert list(spec["inputs"]) == ["y"] + (["u_history"] if reads_u else []) + ["d", "r"]
    assert spec["inputs"]["r"]["columns"] == resolve_objective(cfg, nm).r_columns
    if reads_u:
        history = spec["inputs"]["u_history"]
        assert history["steps"] == history["current_row"] == 2

    # The graph's physical windows are the closed loop's own windows: the
    # wrapper around the trained net must issue the net's own commands.
    inputs = policy_inputs(policy, cfg, nm, ns)
    flat = closed_loop_windows(closed_loop, policy, cfg, sysid_cfg, [s["frame"]], nm, ns)
    physical = physical_windows(flat, inputs).values()
    with torch.no_grad():
        wrapped = policy_graph(policy.callable, inputs)(*physical)
        direct = policy.callable(*flat.values())
    assert torch.allclose(wrapped, direct, atol=1e-4)


@pytest.mark.parametrize("graph_format, reads_d", [
    ("onnx", True), ("torch", False), pytest.param("c", False, marks=needs_c),
], ids=["onnx, with d", "torch, no d", "c, no d"])
def test_a_package_takes_only_what_the_policy_reads(tmp_path, graph_format, reads_d):
    """Listed columns are the inputs, whichever role they have: no wrapper has to
    rebuild a meter the policy never reads, and an input nothing reads is not one."""
    from control_fixtures import DIST, HIGH, LOW, SETPT, ZONE, full_setup
    from neurotemplate.control.closed_loop import build_closed_loop
    from neurotemplate.control.policy import build_policy
    from neurotemplate.export.policy import export_policy

    window = {ZONE: {"past": "10m"}, SETPT: {"past": "5m"}, LOW: {"past": 0, "future": 0},
              HIGH: {"past": 0, "future": "5m"}}
    if reads_d:
        window[DIST[1]] = {"past": 0, "future": "5m"}
    s = full_setup(policy_window=window)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    result = export_policy(
        policy_node=policy, cl_system=build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg),
        cfg=cfg, sysid_cfg=sysid_cfg, nm_mapping=nm, norm_stats=ns, test_dfs=[s["frame"]],
        run_id="0" * 32, out_dir=tmp_path / "pkg", graph_format=graph_format,
        exporter="legacy" if graph_format == "onnx" else "dynamo",
    )
    inputs = json.loads((result.out_dir / "policy" / "policy.json").read_text())["inputs"]
    assert inputs["y"]["columns"] == [ZONE]
    assert inputs["u_history"]["columns"] == [SETPT]
    assert inputs["r"]["columns"] == [HIGH, LOW]
    assert ("d" in inputs) == reads_d
    if reads_d:
        assert inputs["d"]["columns"] == [DIST[1]]
    if graph_format == "c":
        make(result.out_dir)
        subprocess.run(["./validate"], cwd=result.out_dir, check=True, capture_output=True)
    assert all(check.ok for check in check_reference(result.out_dir))


# ── the C runner ─────────────────────────────────────────────────


@needs_c
def test_c_runner_reproduces_the_reference(tmp_path, monkeypatch):
    """Build a C package's runner and run its `validate` on a model with
    hidden state and a priced policy in one package, then again once the
    model is re-exported unverified. This is what keeps the C drivers in step
    with the generated graphs and the fixtures runtime.py reads."""
    from control_fixtures import full_setup
    from neurotemplate.control.closed_loop import build_closed_loop
    from neurotemplate.control.policy import build_policy
    from neurotemplate.export.policy import export_policy

    package = export(tmp_path, make_cfg(model="gru"), MAPPING, graph_format="c").out_dir
    s = full_setup(**_priced(1), u_past=1)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    export_policy(
        policy_node=policy, cl_system=build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg),
        cfg=cfg, sysid_cfg=sysid_cfg, nm_mapping=nm, norm_stats=ns,
        test_dfs=[s["frame"]], run_id="0" * 32, out_dir=package, graph_format="c",
    )

    make(package)
    result = subprocess.run(["./validate"], cwd=package, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] model" in result.stdout and "[PASS] policy" in result.stdout
    assert all(check.ok for check in check_reference(package))    # through ctypes
    # Compiled C reads whatever it is pointed at; the Python runner must
    # refuse a wrongly shaped window rather than hand it over.
    from neurotemplate.export.runtime import Policy
    with pytest.raises(ValueError, match="windows of shape"):
        Policy(package / "policy").act_batch(
            {name: np.zeros((1, 1, 1)) for name in ("y", "u_history", "d", "r")})

    _blow_up(monkeypatch)
    export(tmp_path, make_cfg(model="gru"), MAPPING, graph_format="c",
           ask=lambda question: True)
    make(package)
    result = subprocess.run(["./validate"], cwd=package, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] model" not in result.stdout and "[PASS] policy" in result.stdout


def test_an_mpc_run_exports_a_planner_beside_its_surrogate(tmp_path):
    """`policy: mpc` ships no policy graph: the package plans through its own
    model/, in torch, and reproduces the trained planner's commands."""
    from control_fixtures import (
        HIGH, LOW, TARIFF, full_setup, make_nm_mapping, make_tracking_plant,
    )
    from neurotemplate.control.closed_loop import build_closed_loop
    from neurotemplate.control.controller import build_controller
    from neurotemplate.export.policy import export_policy

    future = {c: 4 for c in (TARIFF, LOW, HIGH, *make_nm_mapping()["D"])}
    s = full_setup(**{**_priced(0), "future": future}, policy="mpc", horizon="20m",
                   solver={"max_iters": 5})
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    dyn_sys, plant = make_tracking_plant(sysid_cfg, nm)
    with pytest.raises(FileNotFoundError, match="torch format first"):
        export_policy(policy_node=build_controller(cfg, sysid_cfg, nm, ns, plant),
                      cl_system=None, cfg=cfg, sysid_cfg=sysid_cfg, nm_mapping=nm,
                      norm_stats=ns, test_dfs=[s["frame"]], run_id="0" * 32,
                      out_dir=tmp_path / "pkg", graph_format="torch")
    export_model(dyn_sys=dyn_sys, cfg=sysid_cfg, nm_mapping=nm, norm_stats=ns,
                 frame=s["plant"], timestep_seconds=300.0, run_id="1" * 32,
                 out_dir=tmp_path / "pkg", graph_format="torch")
    mpc = build_controller(cfg, sysid_cfg, nm, ns, plant)
    result = export_policy(
        policy_node=mpc, cl_system=build_closed_loop(plant, mpc, cfg, sysid_cfg),
        cfg=cfg, sysid_cfg=sysid_cfg, nm_mapping=nm, norm_stats=ns,
        test_dfs=[s["frame"]], run_id="0" * 32, out_dir=tmp_path / "pkg",
        graph_format="torch")

    assert all(check.ok for check in check_reference(result.out_dir))
    spec = json.loads((result.out_dir / "policy" / "policy.json").read_text())
    assert spec["mpc"]["horizon"] == 4 and spec["mpc"]["power"]
    assert not (result.out_dir / "policy" / "policy.pt2").exists()
