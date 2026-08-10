"""
End-to-end wiring test for the sys-id diagnostics.

`nt model train` runs four physics assertions on top of the fit metrics, and
threads all of them into one reportcard and one set of MLflow keys. The pieces have unit tests; this runs `evaluate_and_plot`
against the real fixture hooks so a signature drifting between them fails here
instead of at the end of a real training run.

The fixture's `synth_cfg` returns a populated `SynthSpec`, which is also the
point: the subsystem used to be disabled in the library's own fixture, which
is how it stayed untested long enough to grow a bug.
"""

from pathlib import Path

import pytest
import torch

from neurotemplate.modeling.pipeline import evaluate_and_plot, log_test_results
from neurotemplate.modeling.reportcard import render_reportcard
from neurotemplate.modeling.dynamics import describe_dynamics
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
    cfg.synth_n_scenarios, cfg.synth_n_steps = 4, 6
    cfg.plot, cfg.verbose = False, False
    dyn_sys, _ = make_frozen_dynamics(cfg, nm)

    spec = hooks.synth_cfg(nm, norm_stats) if synth else None
    return evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys, test_dfs=[normalize_df(raw, norm_stats)],
        nm_mapping=nm, norm_stats=norm_stats, synth_cfg=spec,
        freq_s=300.0, plot_run_dir=Path(tmp_path),
    ), cfg, nm, norm_stats


def test_every_diagnostic_reaches_the_reportcard(tmp_path):
    results, cfg, nm, norm_stats = _evaluate(tmp_path)

    md = render_reportcard(
        cfg=cfg, arch=describe_dynamics(make_frozen_dynamics(cfg, nm)[0], cfg),
        test_metrics=results.metrics, nm_mapping=nm,
        stability=results.stability,
        physics=results.physics, readouts=results.readouts, run_name="x",
        timestamp="2026-01-01T00:00:00",
    )
    for title in ("## Test Metrics (per channel)", "## Rollout stability",
                  "## Physics checks", "## Readout channels"):
        assert title in md, f"missing section: {title}"


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
    cfg.synth_n_scenarios, cfg.synth_n_steps = 4, 6
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


def test_the_plot_path_writes_figures_and_their_data(tmp_path):
    """A figure is only written if its data lands beside it."""
    import pandas as pd

    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=120)
    # plot_predictions anchors its time axis to the source frame's timestamps.
    raw.index = pd.date_range("2026-01-01", periods=len(raw), freq="5min")
    norm_stats = make_norm_stats(raw)
    cfg = make_sysid_cfg()
    cfg.test_window, cfg.test_stride = 24, 24
    cfg.synth_n_scenarios, cfg.synth_n_steps, cfg.synth_n_plot = 4, 6, 2
    cfg.plot, cfg.verbose = True, False
    dyn_sys, _ = make_frozen_dynamics(cfg, nm)

    evaluate_and_plot(
        cfg=cfg, dyn_sys=dyn_sys, test_dfs=[normalize_df(raw, norm_stats)],
        nm_mapping=nm, norm_stats=norm_stats,
        synth_cfg=hooks.synth_cfg(nm, norm_stats),
        freq_s=300.0, plot_run_dir=Path(tmp_path),
    )

    synth_dir = Path(tmp_path) / "synthetic"
    pdfs = sorted(synth_dir.glob("*.pdf"))
    csvs = sorted((synth_dir / "synth_data").glob("*.csv"))
    assert len(pdfs) == 4 * cfg.synth_n_plot      # four scenarios
    assert [p.stem for p in pdfs] == [c.stem for c in csvs]
