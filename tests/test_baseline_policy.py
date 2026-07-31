"""
Rule-based baseline controller + the learned-vs-baseline comparison.

The comparison is only meaningful if both controllers are scored against the
*same* comfort-band realization, so that claim gets its own test.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.pipeline import evaluate_baseline, evaluate_closed_loop
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import horizons
from control_user_defined import actuation_bounds, build_baseline_policy
from control_fixtures import full_setup


def _setup():
    s = full_setup()
    s["test_dfs"] = [s["norm_df"]]
    return s


# ── The baseline node itself ─────────────────────────────────────


def test_baseline_node_io():
    s = _setup()
    node = build_baseline_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    assert node.input_keys == ["ymin", "ymax"]
    assert node.output_keys == ["U"]


def test_baseline_has_no_trainable_parameters():
    """It is a rule, not a model — it must never reach the optimizer."""
    s = _setup()
    node = build_baseline_policy(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    assert [p for p in node.parameters() if p.requires_grad] == []


def test_baseline_emits_band_midpoint_clamped_to_actuator_bounds():
    s = _setup()
    nm, ns = s["nm"], s["norm_stats"]
    node = build_baseline_policy(s["cfg"], s["sysid_cfg"], nm, ns)
    umin, umax = actuation_bounds(nm, ns)

    # A band comfortably inside the actuator range comes through as its midpoint.
    ymin = torch.full((2, 1), float(umin) + 0.1)
    ymax = torch.full((2, 1), float(umax) - 0.1)
    u = node.callable(ymin, ymax)
    assert torch.allclose(u, 0.5 * (ymin + ymax))

    # A band above the actuator ceiling is clamped, not extrapolated.
    high = torch.full((2, 1), float(umax) + 5.0)
    assert torch.allclose(node.callable(high, high), umax.expand_as(high))


def test_baseline_output_stays_within_bounds_in_closed_loop():
    s = _setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    _, _, hmax = horizons(cfg, sysid_cfg)

    node = build_baseline_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], node, cfg, sysid_cfg)
    results, _ = evaluate_closed_loop(
        cfg, sysid_cfg, cl, s["test_dfs"], nm, ns,
    )
    assert 0.0 <= results.comfort_violation_rate <= 1.0
    assert results.comfort  # one entry per controlled channel


# ── The comparison harness ───────────────────────────────────────


def test_evaluate_baseline_returns_results():
    s = _setup()
    out = evaluate_baseline(
        s["cfg"], s["sysid_cfg"], s["dyn_node"], s["test_dfs"],
        s["nm"], s["norm_stats"],
    )
    assert out is not None
    assert out.comfort and out.tracking


def test_missing_hook_returns_none_rather_than_raising(monkeypatch):
    """No build_baseline_policy just means no comparison — not a crash."""
    import control_user_defined as cud
    s = _setup()
    monkeypatch.delattr(cud, "build_baseline_policy")

    assert evaluate_baseline(
        s["cfg"], s["sysid_cfg"], s["dyn_node"], s["test_dfs"],
        s["nm"], s["norm_stats"],
    ) is None


def test_hook_returning_none_skips_comparison(monkeypatch):
    import control_user_defined as cud
    s = _setup()
    monkeypatch.setattr(cud, "build_baseline_policy",
                        lambda *a, **k: None, raising=True)

    assert evaluate_baseline(
        s["cfg"], s["sysid_cfg"], s["dyn_node"], s["test_dfs"],
        s["nm"], s["norm_stats"],
    ) is None


def test_baseline_and_learned_see_an_identical_comfort_band(monkeypatch):
    """The whole comparison rests on this: both rollouts are scored against
    the same fixed-seed band realization on the same windows."""
    import neurotemplate.control.scenarios as scenarios
    s = _setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]

    seen = []
    real = scenarios.build_window_dict_multi

    def spy(*args, **kwargs):
        data = real(*args, **kwargs)
        seen.append((data["ymin"].clone(), data["ymax"].clone()))
        return data

    monkeypatch.setattr(
        "neurotemplate.control.pipeline.build_window_dict_multi", spy
    )

    learned = build_policy(cfg, sysid_cfg, nm, ns)
    evaluate_closed_loop(
        cfg, sysid_cfg, build_closed_loop(s["dyn_node"], learned, cfg, sysid_cfg),
        s["test_dfs"], nm, ns,
    )
    evaluate_baseline(
        cfg, sysid_cfg, s["dyn_node"], s["test_dfs"], nm, ns,
    )

    assert len(seen) == 2
    (ymin_a, ymax_a), (ymin_b, ymax_b) = seen
    assert torch.equal(ymin_a, ymin_b)
    assert torch.equal(ymax_a, ymax_b)


def test_baseline_rejects_mismatched_controlled_and_u_counts():
    """The midpoint rule assumes one setpoint per controlled zone; a mismatch
    must fail loudly rather than broadcast something wrong."""
    s = _setup()
    nm = dict(s["nm"])
    nm["U"] = nm["U"] + nm["U"]  # two setpoints, still one controlled zone
    with pytest.raises(ValueError, match="one U per controlled channel"):
        build_baseline_policy(s["cfg"], s["sysid_cfg"], nm, s["norm_stats"])
