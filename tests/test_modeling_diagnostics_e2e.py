"""
End-to-end wiring test for the sys-id diagnostics.

`nt model train` runs four physics assertions on top of the fit metrics and
threads all of them into one reportcard and one set of MLflow keys. The pieces
have unit tests; this runs `evaluate_and_plot` against the real fixture hooks
so a signature drifting between them fails here instead of at the end of a real
training run.

The fixture's `synth_cfg` returns a populated `SynthSpec`, which is also the
point: the subsystem used to be disabled in the library's own fixture, which
is how it stayed untested long enough to grow a bug.
"""

from pathlib import Path

import pytest
import torch

from neurotemplate.modeling.pipeline import evaluate_and_plot, log_test_results
from control_fixtures import (
    make_frozen_dynamics, make_nm_mapping, make_norm_stats, make_raw_df,
    make_sysid_cfg,
)
import nt_model as hooks
from fake_logger import FakeLogger
from model_fixtures import unit_stats


def _evaluate(tmp_path, synth: bool = True):
    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=120)
    norm_stats = make_norm_stats(raw)
    cfg = make_sysid_cfg()
    cfg.test_horizon, cfg.test_stride = "100m", "2h"
    cfg.synth_n_scenarios, cfg.synth_hours = 4, 0.5
    cfg.plot, cfg.progress = False, "silent"
    dyn_sys, _ = make_frozen_dynamics(cfg, nm, norm_stats)

    spec = hooks.synth_cfg(nm, norm_stats) if synth else None
    return evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys, test_dfs=[raw],
        nm_mapping=nm, norm_stats=norm_stats, synth_cfg=spec,
        freq_s=300.0, plot_run_dir=Path(tmp_path),
    ), cfg, nm, norm_stats


def test_the_fixture_hooks_produce_a_usable_spec(tmp_path):
    results, _, _, _ = _evaluate(tmp_path)

    names = [r.name for r in results.physics]
    # The step-response case is generated from `responds` + the setpoint's
    # recorded range, not hand-written, so its name carries the swept target.
    assert names[0] == "Equilibrium"
    assert names[-1] == "Free decay"
    assert any(n.startswith("Step response: Zone Cooling Setpoint")
               for n in names)
    for r in results.physics:
        assert isinstance(r.passed, bool)
        assert r.headline and r.channel == "Zone Air Temperature [°C]"

    # Readouts are summarised but never asserted on.
    assert results.readouts
    assert all(r.name != "Zone Air Temperature [°C]" for r in results.readouts)


def test_the_checks_do_not_disturb_the_surrogate(tmp_path):
    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=120)
    norm_stats = make_norm_stats(raw)
    cfg = make_sysid_cfg()
    cfg.test_horizon, cfg.test_stride = "100m", "2h"
    cfg.synth_n_scenarios, cfg.synth_hours = 4, 0.5
    cfg.plot, cfg.progress = False, "silent"
    dyn_sys, _ = make_frozen_dynamics(cfg, nm, norm_stats)
    before = [p.clone() for p in dyn_sys.parameters()]

    evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys, test_dfs=[raw],
        nm_mapping=nm, norm_stats=norm_stats,
        synth_cfg=hooks.synth_cfg(nm, norm_stats),
        freq_s=300.0, plot_run_dir=Path(tmp_path),
    )

    for p, p0 in zip(dyn_sys.parameters(), before):
        assert torch.equal(p, p0)
        assert p.grad is None
