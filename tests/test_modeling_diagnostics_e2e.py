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
from neurotemplate.common.data_utils import normalize_df
import modeling_user_defined as hooks


class _FakeLogger:
    def __init__(self):
        self.metrics, self.dicts = {}, {}

    def log_metrics(self, d):
        self.metrics.update(d)

    def log_dict(self, d, name):
        self.dicts[name] = d


def _evaluate(tmp_path, synth: bool = True):
    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=120)
    norm_stats = make_norm_stats(raw)
    cfg = make_sysid_cfg()
    cfg.test_window, cfg.test_stride = 24, 24
    cfg.synth_n_scenarios, cfg.synth_hours = 4, 0.5
    cfg.plot, cfg.verbose = False, False
    dyn_sys, _ = make_frozen_dynamics(cfg, nm)

    spec = hooks.synth_cfg(nm, norm_stats) if synth else None
    return evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys, test_dfs=[normalize_df(raw, norm_stats)],
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


def test_opting_out_leaves_the_other_diagnostics_alone(tmp_path):
    results, _, _, _ = _evaluate(tmp_path, synth=False)

    assert results.physics is None
    assert results.readouts is None
    assert results.metrics and results.stability


def test_everything_logged_is_json_serializable(tmp_path):
    import json

    results, _, nm, _ = _evaluate(tmp_path)
    logger = _FakeLogger()
    log_test_results(logger, results, nm)

    for payload in logger.dicts.values():
        json.dumps(payload)
    assert set(logger.dicts) >= {
        "test_metrics.json", "rollout_stability.json",
        "physics_checks.json", "readout_channels.json",
    }
    assert logger.metrics["physics_checks_total"] == 4.0
    assert 0.0 <= logger.metrics["physics_checks_passed"] <= 4.0
    assert all(isinstance(v, float) for v in logger.metrics.values())


def test_the_checks_do_not_disturb_the_surrogate(tmp_path):
    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=120)
    norm_stats = make_norm_stats(raw)
    cfg = make_sysid_cfg()
    cfg.test_window, cfg.test_stride = 24, 24
    cfg.synth_n_scenarios, cfg.synth_hours = 4, 0.5
    cfg.plot, cfg.verbose = False, False
    dyn_sys, _ = make_frozen_dynamics(cfg, nm)
    before = [p.clone() for p in dyn_sys.parameters()]

    evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys, test_dfs=[normalize_df(raw, norm_stats)],
        nm_mapping=nm, norm_stats=norm_stats,
        synth_cfg=hooks.synth_cfg(nm, norm_stats),
        freq_s=300.0, plot_run_dir=Path(tmp_path),
    )

    for p, p0 in zip(dyn_sys.parameters(), before):
        assert torch.equal(p, p0)
        assert p.grad is None


# ── Staged channels: classification metrics and the persistence bar ──

def test_a_staged_channel_is_scored_as_a_classifier_and_against_persistence(tmp_path):
    """With an ordinal head on the stage channel the rollout is also read as
    a classifier — balanced accuracy, level shares — and persistence gets
    the same treatment so the reportcard can say whether the surrogate
    clears copy-forward. Both must reach the page and MLflow."""
    from neurotemplate.modeling.dynamics import get_dynamics
    from neurotemplate.modeling.heads import HeadStack, OrdinalHead, RegressionHead
    from control_fixtures import STAGE

    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=120)
    norm_stats = make_norm_stats(raw)
    cfg = make_sysid_cfg()
    cfg.test_window, cfg.test_stride = 24, 24
    cfg.plot, cfg.verbose = False, False

    stats = norm_stats[STAGE]
    levels = torch.tensor([(v - stats["shift"]) / stats["stretch"] for v in (0.0, 1.0, 2.0)])
    heads = [OrdinalHead(n_levels=3, level_values=levels) if col == STAGE
             else RegressionHead() for col in nm["Y"]]
    dyn_sys = get_dynamics(cfg, nm, head_stack=HeadStack(heads, nm["Y"]))
    dyn_sys.eval()

    results = evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys, test_dfs=[normalize_df(raw, norm_stats)],
        nm_mapping=nm, norm_stats=norm_stats, synth_cfg=None,
        freq_s=300.0, plot_run_dir=Path(tmp_path),
    )
    assert [lm.name for lm in results.levels] == [STAGE]
    assert results.levels[0].levels == pytest.approx([0.0, 1.0, 2.0], abs=1e-4)
    assert [m.name for m in results.persistence] == nm["Y"]
    assert [lm.name for lm in results.persistence_levels] == [STAGE]
    # Persistence never leaves the seeded level within a window, so its
    # predicted shares are whatever the seeds were — but it is a real number.
    assert 0.0 <= results.persistence_levels[0].balanced_accuracy <= 1.0

    logger = _FakeLogger()
    log_test_results(logger, results, nm)
    assert "level_metrics.json" in logger.dicts
    assert "persistence_metrics.json" in logger.dicts
    key = [k for k in logger.metrics if k.endswith("_balanced_acc") and k.startswith("test_")]
    assert len(key) == 1
