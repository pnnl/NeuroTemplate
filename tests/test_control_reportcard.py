"""
Control reportcard renderer tests.

Includes a snapshot test against a golden file. To regenerate it (e.g. after
an intentional rendering change), run:

    REGEN_GOLDEN=1 uv run pytest tests/test_control_reportcard.py -k snapshot
"""

import os
from pathlib import Path

import torch

from neurotemplate.common.metrics import per_channel_metrics
from neurotemplate.control.metrics import (
    actuation_metrics,
    comfort_metrics,
    energy_metrics,
    response_audit,
)
from neurotemplate.control.objective import (
    ObjectiveBalance,
    TermBalance,
    TermShape,
)
from neurotemplate.control.pipeline import ControlResults
from neurotemplate.control.reportcard import render_control_reportcard
from control_fixtures import make_control_cfg, make_sysid_cfg

GOLDEN_DIR = Path(__file__).parent / "golden"

_NM = {"Y": ["zone [°C]"], "U": ["setpoint [°C]"], "D": ["outdoor [°C]"],
       "t": ["sec_elapsed"]}
_CONTROLLED = ["zone [°C]"]


def _make_arch():
    """Architecture dict shaped like control.policy.describe_policy output."""
    return {
        "family": "mlp",
        "nonlin": "gelu",
        "bound_method": "sigmoid_scale",
        "insize": 27,
        "outsize": 1,
        "h_ctrl": 12,
        "d_preview": 0,
        "band_preview": 0,
        "price_preview": 0,
        "reads_price": False,
        "hsizes": [64, 64],
        "param_count": 6017,
        "u_cols": ["setpoint [°C]"],
        "umin_phys": [22.0],
        "umax_phys": [28.0],
        "quantized": True,
        "setpoint_step": 0.5,
    }


_U_LO, _U_HI = torch.tensor([22.0]), torch.tensor([28.0])


def _make_results(offset: float = 1.0, power_level: float = 1500.0):
    """ControlResults for a trajectory held `offset` above the band.

    Shaped to exercise every section: the band jumps at the midpoint (so the
    response audit has something to score), the channel is out of band from
    then on, and the setpoint mostly sits on its lower bound — one window
    answers the jump, the other ignores it.
    """
    n = 8
    y = torch.full((2, n, 1), 22.0)
    y[:, n // 2:, :] = 25.0 + offset
    # ymin steps 19 -> 20 halfway, which is a band jump that leaves the
    # channel above the band without changing any excursion.
    ymin = torch.full((2, n, 1), 20.0)
    ymin[:, :n // 2, :] = 19.0
    ymax = torch.full((2, n, 1), 25.0)
    mid = 0.5 * (ymin + ymax)
    power = torch.full((2, n), power_level)
    comfort = comfort_metrics(y, ymin, ymax, _CONTROLLED, step_hours=0.25)

    u = torch.full((2, n, 1), 22.0)      # pinned at the lower bound
    u[1, n // 2:, :] = 23.0              # ...except one window, late

    return ControlResults(
        comfort_violation_rate=0.5,
        # Single controlled channel, so the pooled figure equals the
        # per-channel one: half the steps in-band (0 excursion), half
        # `offset` above -> mean excursion = offset / 2.
        comfort_mean_exceedance=comfort[0].mean_exceedance,
        comfort=comfort,
        mean_abs_du=float((u[:, 1:, :] - u[:, :-1, :]).abs().mean()),
        actuation=actuation_metrics(u, _NM["U"], _U_LO, _U_HI),
        response=response_audit(y, ymin, ymax, u, _CONTROLLED, _NM["U"],
                                _U_LO, _U_HI),
        tracking=per_channel_metrics(y, mid, _CONTROLLED),
        energy=energy_metrics(power, step_hours=0.25),
        eval_time_s=0.5,
    )


def _make_balance():
    """An effort term worth 3x the comfort term — the mis-scaled failure."""
    return ObjectiveBalance(
        terms=[
            TermBalance("comfort_lower_loss", 1.0, 0.1, 1.0 / 9.0),
            TermBalance("comfort_upper_loss", 1.5, 0.15, 1.5 / 8.5),
            TermBalance("du_loss", 0.5, 0.05, 0.5 / 9.5),
            TermBalance("effort_loss", 7.0, 0.7, 7.0 / 3.0),
        ],
        total=10.0,
    )


def _make_shapes():
    """A monotone effort term — reported, and deliberately not flagged."""
    return [TermShape(
        u_name="setpoint [°C]",
        grid=[22.0, 23.5, 25.0, 26.5, 28.0],
        values=[4.0, 3.0, 2.0, 1.0, 0.5],
        argmin=4,
    )]


def _render(**kwargs):
    cfg = make_control_cfg()
    cfg.run = "snapshot-control"
    defaults = dict(
        cfg=cfg,
        sysid_cfg=make_sysid_cfg(),
        arch=_make_arch(),
        nm_mapping=_NM,
        controlled=_CONTROLLED,
        results=_make_results(),
        balance=_make_balance(),
        shapes=_make_shapes(),
        eval_band_seed=12345,
        run_name="snapshot-control",
        timestamp="2026-05-08T12:00:00",
        best_dev_loss=0.0123456,
        train_time_s=42.0,
    )
    defaults.update(kwargs)
    return render_control_reportcard(**defaults)


def test_all_sections_present():
    md = _render()
    for title in (
        "## Summary",
        "## Policy architecture",
        "## Scenario",
        "## Objective balance",
        "## Closed-loop performance",
        "## Actuation",
        "## Band-jump response",
        "## Baseline comparison",
        "## Tracking error vs. comfort-band midpoint",
        "## Hyperparameters",
    ):
        assert title in md, f"missing section: {title}"


def test_objective_section_flags_a_dominant_term():
    """Scale is the failure worth asserting on, and the only one asserted."""
    md = _render()
    assert "One term is running the objective" in md
    assert "`effort_loss` at 2.3×" in md


def test_objective_section_reports_term_shape_without_judging_it():
    """The sweep is information. A monotone effort term is normal."""
    md = _render()
    assert "Effort term swept across each actuation box" in md
    assert "28" in md                      # where the minimum landed
    assert "minimised on a bound" not in md
    assert "No value of `effort_weight`" not in md


def test_objective_section_skipped_without_a_diagnostic_pass():
    md = _render(balance=None, shapes=None)
    assert "## Objective balance" not in md


def test_actuation_bounds_reported_in_physical_units():
    md = _render()
    assert "`setpoint [°C]`: [22, 28]" in md


def test_linear_policy_renders_without_hidden_sizes():
    arch = _make_arch()
    arch.update(family="linear", nonlin="n/a (linear)", hsizes=[],
                quantized=False)
    md = _render(arch=arch)
    assert "single affine map" in md
    assert "**Setpoint quantization:** off" in md


def test_missing_baseline_says_why():
    md = _render(baseline=None)
    assert "build_baseline_policy" in md
    assert "| Baseline |" not in md


def test_baseline_comparison_renders_signed_delta():
    """A baseline that violates twice as often must show a negative Δ on the
    violation column (the learned policy is better) and a positive Δ on power."""
    learned = _make_results(offset=1.0, power_level=1500.0)
    baseline = _make_results(offset=2.0, power_level=1000.0)
    baseline.comfort_violation_rate = 1.0

    md = _render(results=learned, baseline=baseline)

    assert "| Learned policy |" in md
    assert "| Baseline |" in md
    delta_row = next(l for l in md.splitlines() if l.startswith("| Δ "))
    cells = [c.strip() for c in delta_row.strip("|").split("|")]
    # Column order: label, mean exceedance, violation rate, degree-hours,
    # mean |Δu|, mean power, peak power.
    assert cells[2] == "-50"       # violation rate delta, percentage points
    assert cells[5] == "+500"      # mean power delta


def test_energy_marked_unavailable_without_power_metric():
    results = _make_results()
    results.energy = None
    md = _render(results=results)
    assert "power_metric not implemented" in md
    assert results.energy_proxy is None


def test_snapshot_matches_golden():
    """Pin the rendered output. Update via REGEN_GOLDEN=1."""
    md = _render(baseline=_make_results(offset=2.0, power_level=1000.0))

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / "control_reportcard_baseline.md"

    if os.environ.get("REGEN_GOLDEN"):
        golden_path.write_text(md)

    if not golden_path.exists():
        # First run: write it and pass with a clear note.
        golden_path.write_text(md)
        return

    expected = golden_path.read_text()
    assert md == expected, (
        f"Control reportcard rendering changed.\n"
        f"If intentional, regenerate via: REGEN_GOLDEN=1 pytest {__file__}"
    )
