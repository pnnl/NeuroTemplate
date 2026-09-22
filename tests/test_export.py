"""
ONNX export: the exported package must BE the trained model.

Every case goes through `export_package`, which rolls the trained
`SystemPreview` and the exported package (through the shipped runtime) out
over a whole synthetic split and refuses to write unless they agree, then
runs the package's own reference check. So one export per case covers the
graphs, the state lifting, the physical-units window layout and the runtime
together. Every mapping has keys with more than one column, because a
channel-major window layout agrees with the real (time-major) one only when
a key has a single column.

Cases: each family, the per-block split, each head type, and the
heterogeneous 2r2c + mlp composition with a prestack and a mixed head stack
— the case the old hand-written deliveries could not express. That one runs
under the default (dynamo) exporter; the rest use the faster legacy tracer,
which traces the same forward code.

Plus two failure modes that would ship a wrong model silently: a step graph
with its hidden state baked in as a constant, and an export that disagrees
with the trained model being written anyway.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from neurotemplate.export import package as export_package_module
from neurotemplate.export.graphs import Step
from neurotemplate.export.package import export_package
from neurotemplate.export.runtime import Model, check_reference
from neurotemplate.export.state import probe_state_slots
from neurotemplate.export.trace import export_graph, session
from neurotemplate.export.verify import Rollout
from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.heads import (
    BinaryHead, BoundedHead, HeadStack, OrdinalHead, RegressionHead,
)

IN_STEPS = 4
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
    "Y": ["north T", "power"],
    "U": ["north sp", "ahu fan"],
    "D": ["oat", "ghi"],
    "t": ["sec_elapsed"],
}
MIXED = {
    "north": {
        "Y": ["north T"], "U": ["north sp", "ahu fan"], "model": "2r2c",
        "ambient": "oat",
        "inputs": {"north sp": {"kind": "demand", "sign": -1, "state": "north T"},
                   "ahu fan": {"kind": "linear", "sign": +1}},
    },
    "power": {"Y": ["power"], "reads": ["north T"], "model": "mlp"},
}


def make_cfg(**over) -> Config:
    cfg = Config()
    cfg.model = "2r2c"
    cfg.in_steps = IN_STEPS
    cfg.log2_hidden_size = 3
    cfg.num_layers = 1
    cfg.steps_per_hour = 12.0
    cfg.device = torch.device("cpu")
    cfg.verbose = False
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


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


def export(tmp_path, cfg, mapping, block_spec=None, heads=None, exporter="legacy"):
    frame, stats = split(mapping)
    torch.manual_seed(0)
    head_stack = HeadStack(heads, mapping["Y"]) if heads else None
    dyn_sys = get_dynamics(cfg, mapping, head_stack=head_stack,
                           block_spec=block_spec, norm_stats=stats)
    return export_package(
        dyn_sys=dyn_sys, cfg=cfg, nm_mapping=mapping, norm_stats=stats,
        frame=frame, timestep_seconds=300.0, run_id="0" * 32,
        out_dir=tmp_path / "pkg", package="exported", exporter=exporter,
    )


CASES = {
    "2r2c": (make_cfg(), MAPPING, SPLIT, None),
    "nssm": (make_cfg(model="nssm"), MAPPING, None, None),
    "mlp": (make_cfg(model="mlp"), MAPPING, None, None),
    "lssm": (make_cfg(model="lssm"), MAPPING, None, None),
    "gru, 2 layers": (make_cfg(model="gru", num_layers=2), MAPPING, None, None),
    "mlp per block": (make_cfg(model="mlp", per_block=True), MAPPING, SPLIT, None),
    "gru per block": (make_cfg(model="gru", per_block=True), MAPPING, SPLIT, None),
    "binary + bounded heads": (make_cfg(model="nssm"), MAPPING, None,
                               lambda: [BinaryHead(), BoundedHead(lo=-1.0, hi=1.0)]),
}


@pytest.mark.parametrize("name", CASES)
def test_every_family_and_head_exports_as_the_trained_model(tmp_path, name):
    cfg, mapping, block_spec, heads = CASES[name]
    result = export(tmp_path, cfg, mapping, block_spec, heads() if heads else None)
    assert result.steps_verified == ROWS - IN_STEPS
    assert check_reference(result.out_dir).ok


def test_heterogeneous_model_with_prestack_and_heads_exports(tmp_path):
    """The exit criterion: 2r2c + mlp, a demand prestack, an ordinal head."""
    result = export(tmp_path, make_cfg(), MIXED_MAPPING, MIXED,
                    [RegressionHead(), OrdinalHead(n_levels=3)], exporter="dynamo")
    assert result.state_width == 1                  # the one zone's mass
    assert check_reference(result.out_dir).ok
    # The package drives itself from its own files, with no training code.
    model = Model(result.out_dir / "model")
    assert model.y_columns == MIXED_MAPPING["Y"]


def test_step_graph_follows_the_state_it_is_handed(tmp_path):
    """A trace that does not lift the GRU's hidden state bakes it in as a
    constant: the graph then answers the same whatever state it is given.
    The exported step must track torch for two different states."""
    cfg = make_cfg(model="gru")
    frame, stats = split(MAPPING)
    torch.manual_seed(0)
    model = get_dynamics(cfg, MAPPING, norm_stats=stats).nodes[0].callable.eval()
    windows = [torch.randn(2, IN_STEPS, len(MAPPING[k])) for k in ("Y", "U", "D")]
    slots = probe_state_slots(model, *(w.reshape(2, -1) for w in windows))
    width = sum(slot.width for slot in slots)
    graph = Step(model, MAPPING, stats, slots)
    path = tmp_path / "step.onnx"
    export_graph(graph, (*windows, torch.randn(2, width)), ["y", "u", "d", "state"],
                 ["y_next", "state_next"], path, exporter="legacy")

    sess = session(path)
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
    real = export_package_module.torch_rollout

    def off_by_a_degree(*args):
        reference = real(*args)
        return Rollout(y=reference.y + 1.0, state=reference.state)

    monkeypatch.setattr(export_package_module, "torch_rollout", off_by_a_degree)
    with pytest.raises(RuntimeError, match="REFUSING TO WRITE"):
        export(tmp_path, make_cfg(), MAPPING, SPLIT)
    assert not (tmp_path / "pkg").exists()
