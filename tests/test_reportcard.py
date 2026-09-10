"""
Reportcard renderer tests.

Includes a snapshot test against a golden file. To regenerate the golden
file (e.g. after an intentional rendering change), run:

    REGEN_GOLDEN=1 uv run --extra dev pytest tests/test_reportcard.py -k snapshot
"""

import os
from pathlib import Path

import pytest

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.common.config import dict_to_dataclass
from neurotemplate.common.metrics import level_metrics, per_channel_metrics
from neurotemplate.modeling.heads import HeadStack, OrdinalHead, RegressionHead
from neurotemplate.modeling.reportcard import (
    EnsembleSpread,
    SeedSpread,
    render_reportcard,
    spread_stats,
)

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


def test_a_per_block_model_shows_each_blocks_time_constants_and_family():
    """The one number a 2r2c run exists to be checked on has to reach the
    page, labelled with the block and the channel it belongs to. With
    mixed families the family is the interesting new fact, so both blocks
    here are 2r2c to pin the label; test_a_mixed_family_model_labels_each_
    blocks_own_family (below) covers the mixed case."""
    arch = _make_arch()
    arch["factory"] = "2r2c"
    arch["blocks"] = {"fy": {
        "class": "BlockStack", "param_count": 2504, "out_features": 2,
        "blocks": {
            "north": {"class": "RCBlock", "param_count": 1252, "model": "2r2c",
                      "time_constants_h": {"north T": {"fast": 0.6, "slow": 13.4}}},
            "south": {"class": "RCBlock", "param_count": 1252, "model": "2r2c",
                      "time_constants_h": {"south T": {"fast": 0.4, "slow": 21.0}}},
        },
    }, "fu": None, "fd": None}
    md = render_reportcard(
        cfg=_make_cfg(), arch=arch, test_metrics=_make_metrics(),
        nm_mapping={"Y": ["north T", "south T"], "U": [], "D": []},
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "block `north` (2r2c)" in md and "block `south` (2r2c)" in md
    assert "north T: tau_fast=0.60 h, tau_slow=13.40 h" in md
    assert "south T: tau_fast=0.40 h, tau_slow=21.00 h" in md


def test_a_mixed_family_model_labels_each_blocks_own_family():
    """The family name is what makes a mixed-family block stack readable:
    without it, two very differently-shaped sub-blocks look unrelated."""
    arch = _make_arch()
    arch["blocks"] = {"fy": {
        "class": "BlockStack", "param_count": 999, "out_features": 2,
        "blocks": {
            "north": {"class": "RCBlock", "param_count": 500, "model": "2r2c",
                      "time_constants_h": {"north T": {"fast": 0.6, "slow": 13.4}}},
            "power": {"class": "MLP", "param_count": 499, "model": "mlp"},
        },
    }, "fu": None, "fd": None}
    md = render_reportcard(
        cfg=_make_cfg(), arch=arch, test_metrics=_make_metrics(),
        nm_mapping={"Y": ["north T", "power"], "U": [], "D": []},
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "block `north` (2r2c)" in md
    assert "block `power` (mlp)" in md


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


def _physics(passed: bool) -> list:
    """Two verdicts: one that always passes, one the caller decides."""
    from neurotemplate.modeling.scenarios import ScenarioResult
    return [
        ScenarioResult("Equilibrium", True, "`zone_a [°C]` held to 0.01.",
                       "zone_a [°C]", 0.01, 0.24),
        ScenarioResult("Free decay", passed,
                       "`zone_a [°C]` closed 0.4% of a 8.3 gap to ambient.",
                       "zone_a [°C]", 0.004, 0.02),
    ]


def test_physics_section_counts_the_verdicts():
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["zone_a [°C]"], "U": [], "D": ["outdoor"]},
        physics=_physics(passed=False),
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "## Physics checks" in md
    assert "1 of 2 passed" in md
    assert "**FAIL**" in md
    # Named by scenario *and* channel, since each scenario reports on each.
    assert "**⚠️ Failed:** `Free decay | zone_a [°C]`" in md


def test_physics_section_stays_quiet_when_everything_passes():
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["zone_a [°C]"], "U": [], "D": ["outdoor"]},
        physics=_physics(passed=True),
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "2 of 2 passed" in md
    assert "⚠️ Failed" not in md


def test_physics_section_skipped_when_the_use_case_opted_out():
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["a"], "U": [], "D": []},
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "## Physics checks" not in md


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


# ── Seed ensemble spread (cfg.ensemble_seeds > 1) ─────────────────

@pytest.mark.parametrize("values,expected", [
    ([3.0, 1.0, 2.0], SeedSpread(min=1.0, median=2.0, max=3.0)),      # odd count
    ([1.0, 2.0, 3.0, 4.0], SeedSpread(min=1.0, median=2.5, max=4.0)),  # even count
    ([5.0], SeedSpread(min=5.0, median=5.0, max=5.0)),                # single seed
])
def test_spread_stats_reports_min_median_max(values, expected):
    """Hand-constructed per-seed value lists — the correctness this dataclass
    exists to deliver, independent of anything reportcard-shaped."""
    assert spread_stats(values) == expected


def _make_ensemble(**over) -> EnsembleSpread:
    fields = dict(
        n_seeds=3,
        rmse={"zone_a [°C]": SeedSpread(0.40, 0.43, 0.47)},
        physics={"Equilibrium | zone_a [°C]": SeedSpread(0.01, 0.02, 0.03)},
        tau_slow={"north T": SeedSpread(12.0, 13.4, 15.0)},
    )
    fields.update(over)
    return EnsembleSpread(**fields)


def test_ensemble_section_is_absent_when_there_is_no_ensemble():
    """ensemble=None (cfg.ensemble_seeds == 1) must not add a section — this
    is what keeps that case's rendering identical to a run built before the
    ensemble feature existed."""
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["zone_a [°C]", "zone_b [°C]"], "U": [], "D": []},
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "Seed ensemble spread" not in md


def test_ensemble_section_reports_min_median_max_per_metric():
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["zone_a [°C]", "zone_b [°C]"], "U": [], "D": []},
        ensemble=_make_ensemble(),
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "## Seed ensemble spread" in md
    assert "Across **3** independently-trained seeds" in md
    # RMSE and tau_slow tables follow metrics_to_markdown's plain-name style.
    assert "| zone_a [°C] | 0.4 | 0.43 | 0.47 |" in md
    assert "| north T | 12 | 13.4 | 15 |" in md
    # The physics table follows scenarios.checks_to_markdown's backtick style.
    assert "| Equilibrium | `zone_a [°C]` | 0.01 | 0.02 | 0.03 |" in md


def test_ensemble_section_skips_empty_metric_families():
    """A 2r2c-free ensemble has no tau_slow to report, and one with
    synth_cfg=None has no physics — each empty dict must render nothing."""
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["zone_a [°C]", "zone_b [°C]"], "U": [], "D": []},
        ensemble=_make_ensemble(physics={}, tau_slow={}),
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "## Seed ensemble spread" in md
    assert "tau_slow" not in md
    assert "Physics-check values" not in md


# ── Level metrics and the persistence reference ───────────────────

def _make_levels(pred, true):
    return [level_metrics(torch.tensor(pred), torch.tensor(true), [0.0, 61728.4],
                          "furnace [W]")]


def test_a_staged_channel_is_read_as_a_classifier_beside_persistence():
    """A head that never fires has to be visible as one: balanced accuracy
    at the floor, predicted share 0% at the rare level, and the persistence
    row beside it so 'ties copy-forward' is a fact on the page."""
    true = [0] * 9 + [1]
    stack = HeadStack([OrdinalHead(n_levels=2, decision_thresholds=torch.tensor([0.3])),
                       RegressionHead()], ["furnace [W]", "zone_a [°C]"])
    norm_stats = {
        "furnace [W]": {"shift": 0.0, "stretch": 61728.4, "min": 0.0, "max": 61728.4,
                        "norm_type": "min_max"},
        "zone_a [°C]": {"shift": 0.0, "stretch": 1.0, "min": 0.0, "max": 10.0,
                        "norm_type": "min_max"},
    }
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["furnace [W]", "zone_a [°C]"], "U": [], "D": []},
        levels=_make_levels([0] * 10, true),
        persistence=_make_metrics(),
        persistence_levels=_make_levels([0] * 9 + [1], true),
        head_stack=stack, norm_stats=norm_stats,
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "## Level metrics (staged channels)" in md
    assert "| furnace [W] | 0.500 | 1.000 |" in md
    assert "| furnace [W] | 61728.4 | 10.0% | 0.0% | 0.000 |" in md
    assert "**Persistence reference**" in md
    # The decision rule is reported in physical units: 0.3 * 61728.4.
    assert "`furnace [W]`: 1.852e+04" in md


def test_a_run_without_levelled_heads_has_no_level_section():
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["zone_a [°C]", "zone_b [°C]"], "U": [], "D": []},
        levels=[], persistence=_make_metrics(), persistence_levels=[],
        run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "Level metrics" not in md
    assert "**Persistence reference**" in md


def test_ensemble_section_reports_balanced_accuracy_for_staged_channels():
    ensemble = EnsembleSpread(
        n_seeds=3, rmse={}, physics={}, tau_slow={},
        balanced_acc={"dx [W]": SeedSpread(min=0.5, median=0.73, max=0.85)},
    )
    md = render_reportcard(
        cfg=_make_cfg(), arch=_make_arch(), test_metrics=_make_metrics(),
        nm_mapping={"Y": ["zone_a [°C]", "zone_b [°C]"], "U": [], "D": []},
        ensemble=ensemble, run_name="x", timestamp="2026-01-01T00:00:00",
    )
    assert "Test balanced accuracy (staged channels)" in md
    assert "| dx [W] | 0.5 | 0.73 | 0.85 |" in md
