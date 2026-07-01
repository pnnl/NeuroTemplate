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
    """fu/fd may be None for non-separable models (mlp/tcn/gru)."""
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
