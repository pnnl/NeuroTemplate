"""
Reportcard renderer tests.

Includes a snapshot test against a golden file. To regenerate the golden
file (e.g. after an intentional rendering change), run:

    REGEN_GOLDEN=1 uv run --extra dev pytest tests/test_reportcard.py -k snapshot
"""

import os
from pathlib import Path

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.common.config import dict_to_dataclass
from neurotemplate.common.metrics import per_channel_metrics
from neurotemplate.modeling.reportcard import render_reportcard

import torch

GOLDEN_DIR = Path(__file__).parent / "golden"


def _make_arch():
    """Architecture dict shaped like models.describe_dynamics output."""
    return {
        "factory": "nssm",
        "separable": True,
        "residual": True,
        "in_steps": 12,
        "ny": 2,
        "nu": 1,
        "nd": 1,
        "param_count": 1234,
        "blocks": {
            "fy": {"class": "MLP", "param_count": 500, "hsizes": [32, 32]},
            "fu": {"class": "MLP", "param_count": 500, "hsizes": [32, 32]},
            "fd": {"class": "MLP", "param_count": 234, "hsizes": [32, 32]},
        },
    }


def _make_metrics():
    y_true = torch.linspace(0, 10, 50).reshape(1, 50, 1).repeat(1, 1, 2)
    y_pred = y_true + 0.1
    return per_channel_metrics(y_pred, y_true, ["zone_a [°C]", "zone_b [°C]"])


def _make_cfg():
    cfg = dict_to_dataclass({
                "model": "nssm",
        "in_steps": 12,
        "lr": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "train_pred_steps": 24,
        "dev_pred_steps": 48,
        "test_window": 287,
        "curriculum_steps": 1,
        "norm_type": "min_max",
        "seed": 42,
        "ref_gain": 1.0,
        "synth_gain": 0.0,
        "grad_gain": 0.0,
    }, Config)
    return cfg


def test_render_returns_string():
    md = render_reportcard(
        cfg=_make_cfg(),
        arch=_make_arch(),
        test_metrics=_make_metrics(),
        nm_mapping={
            "Y": ["zone_a [°C]", "zone_b [°C]"],
            "U": ["setpoint"],
            "D": ["outdoor"],
        },
        run_name="unit-test-run",
        timestamp="2026-05-08T12:00:00",
        best_dev_loss=0.0123456,
        train_time_s=42.0,
        test_time_s=0.5,
        split_sizes={"train": 1000, "dev": 200, "test": 200},
    )
    assert isinstance(md, str)
    assert "# Reportcard:" in md
    assert "Test Metrics" in md
    assert "zone_a" in md


def test_render_handles_missing_optional_blocks():
    """fu/fd may be None for non-separable models (mlp/gru)."""
    arch = _make_arch()
    arch["separable"] = False
    arch["blocks"]["fu"] = None
    arch["blocks"]["fd"] = None
    md = render_reportcard(
        cfg=_make_cfg(), arch=arch,
        test_metrics=_make_metrics(),
        nm_mapping={"Y": ["a"], "U": [], "D": []},
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "fy" in md
    # fu/fd lines should be absent
    assert "**fu:**" not in md
    assert "**fd:**" not in md


def test_snapshot_matches_golden():
    """Pin the rendered output for the NSSM-M baseline. Update via REGEN_GOLDEN=1."""
    md = render_reportcard(
        cfg=_make_cfg(),
        arch=_make_arch(),
        test_metrics=_make_metrics(),
        nm_mapping={
            "Y": ["zone_a [°C]", "zone_b [°C]"],
            "U": ["setpoint"],
            "D": ["outdoor"],
        },
        run_name="snapshot-baseline",
        timestamp="2026-05-08T12:00:00",
        best_dev_loss=0.0123456,
        train_time_s=42.0,
        test_time_s=0.5,
        split_sizes={"train": 1000, "dev": 200, "test": 200},
    )

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / "reportcard_nssm_baseline.md"

    if os.environ.get("REGEN_GOLDEN"):
        golden_path.write_text(md)

    if not golden_path.exists():
        # First run: write it and pass with a clear note.
        golden_path.write_text(md)
        return

    expected = golden_path.read_text()
    assert md == expected, (
        f"Reportcard rendering changed.\n"
        f"If intentional, regenerate via: REGEN_GOLDEN=1 pytest {__file__}"
    )


def _authority(delta: float):
    from neurotemplate.modeling.probe import (
        ActuatorAuthority, AuthorityReport, ChannelEnvelope,
    )
    return AuthorityReport(
        authority=[ActuatorAuthority("zone_a [°C]", "setpoint", delta,
                                     y_range=10.0, u_lo=20.0, u_hi=26.0)],
        envelope=[ChannelEnvelope("zone_a [°C]", 21.0, 25.0, 18.0, 30.0)],
        settle_steps=24, n_windows=32,
    )


def _render_with_authority(delta: float) -> str:
    return render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["zone_a [°C]"], "U": ["setpoint"], "D": ["outdoor"]},
        authority=_authority(delta),
        run_name="x", timestamp="2026-01-01T00:00:00",
    )


def test_authority_section_reports_the_sweep_and_the_envelope():
    md = _render_with_authority(delta=-3.0)
    assert "## Actuator authority" in md
    assert "32 test windows" in md and "24 rollout steps" in md
    assert "Achievable envelope" in md
    assert "⚠️ no authority" not in md


def test_an_actuator_that_moves_nothing_is_flagged():
    """0.05 °C on a 10 °C channel is 0.5% — below any usable authority."""
    md = _render_with_authority(delta=0.05)
    assert "⚠️ no authority" in md
    assert "**⚠️ No authority:**" in md
    assert "`setpoint`" in md


def test_authority_section_skipped_without_a_probe():
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["a"], "U": [], "D": []},
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "## Actuator authority" not in md


def test_stability_section_states_the_rollout_horizon():
    """The gate's horizon must be visible, not inferred from the config section.

    Two of four measured submissions passed the ramp check at a 47-step window
    and diverged past ~300 steps on the same data. The diagnostic was correct;
    the reader had no way to see it had been asked about 47 steps.
    """
    from neurotemplate.common.metrics import rollout_stability

    cfg = _make_cfg()
    cfg.test_window = 47
    y = torch.linspace(20.0, 26.0, 60).reshape(1, 60, 1).repeat(1, 1, 2)
    md = render_reportcard(
        cfg=cfg, arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["zone_a [°C]", "zone_b [°C]"], "U": [], "D": []},
        stability=rollout_stability(y, y, ["zone_a [°C]", "zone_b [°C]"]),
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "**47-step**" in md
