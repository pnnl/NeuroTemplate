"""The reference rule as a baseline controller, and the comparison harness.

The rule the residual policy is anchored to is the same rule scored as the
baseline — one definition, two consumers — so these tests cover both roles.
The comparison is only meaningful if both controllers see the *same* comfort
band realization, so that claim gets its own test.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.pipeline import (
    baseline_node,
    clears_reference,
    evaluate_baseline,
    evaluate_closed_loop,
)
from neurotemplate.control.policy import build_policy, build_reference_node
from neurotemplate.control.scenarios import horizons
from control_user_defined import actuation_bounds, reference_control
from control_fixtures import full_setup


def _setup(**overrides):
    s = full_setup(**overrides)
    s["test_dfs"] = [s["norm_df"]]
    return s


# ── The reference rule as a Node ─────────────────────────────────


def test_reference_node_io():
    s = _setup()
    node = build_reference_node(s["cfg"], s["sysid_cfg"], s["nm"],
                                s["norm_stats"])
    assert node.input_keys == ["yn", "ymin", "ymax", "D"]
    assert node.output_keys == ["U"]


def test_reference_node_has_no_trainable_parameters():
    """It is a rule, not a model — it must never reach the optimizer."""
    s = _setup()
    node = build_reference_node(s["cfg"], s["sysid_cfg"], s["nm"],
                                s["norm_stats"])
    assert [p for p in node.parameters() if p.requires_grad] == []


def test_reference_emits_band_midpoint_clamped_to_actuator_bounds():
    s = _setup()
    nm, ns = s["nm"], s["norm_stats"]
    umin, umax = actuation_bounds(nm, ns)

    def rule(ymin, ymax):
        yn = torch.zeros(ymin.shape[0], len(nm["Y"]))
        d = torch.zeros(ymin.shape[0], len(nm["D"]))
        return reference_control(yn, ymin, ymax, d, nm, ns)

    # A band comfortably inside the actuator range comes through as its midpoint.
    ymin = torch.full((2, 1), float(umin) + 0.1)
    ymax = torch.full((2, 1), float(umax) - 0.1)
    assert torch.allclose(rule(ymin, ymax), 0.5 * (ymin + ymax))

    # A band above the actuator ceiling is clamped, not extrapolated.
    high = torch.full((2, 1), float(umax) + 5.0)
    assert torch.allclose(rule(high, high), umax.expand_as(high))


def test_reference_node_reads_the_current_step_in_the_closed_loop():
    """The Node has no input_map, so every signal arrives unwindowed — which
    is what makes _ReferenceRule's slicing the identity here.

    Also pins the one-step offset: the rollout starts at `hmax - 1`, so the
    command at `U[hmax + k]` was chosen from the band at `hmax - 1 + k`.
    """
    s = _setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    node = build_reference_node(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], node, cfg, sysid_cfg)
    results, out = evaluate_closed_loop(
        cfg, sysid_cfg, cl, s["test_dfs"], nm, ns,
    )
    _, _, hmax = horizons(cfg, sysid_cfg)
    P = out["U"].shape[1] - hmax
    band = slice(hmax - 1, hmax - 1 + P)
    mid = 0.5 * (out["ymin"][:, band, :] + out["ymax"][:, band, :])
    umin, umax = actuation_bounds(nm, ns)
    assert torch.allclose(out["U"][:, hmax:, :], mid.clamp(umin, umax),
                          atol=1e-5)
    assert 0.0 <= results.comfort_violation_rate <= 1.0


def test_reference_rejects_mismatched_controlled_and_u_counts():
    """The midpoint rule assumes one setpoint per controlled zone; a mismatch
    must fail loudly rather than broadcast something wrong."""
    s = _setup()
    nm = dict(s["nm"])
    nm["U"] = nm["U"] + nm["U"]  # two setpoints, still one controlled zone
    with pytest.raises(ValueError, match="one U per controlled channel"):
        reference_control(torch.zeros(2, len(nm["Y"])), torch.zeros(2, 1),
                          torch.ones(2, 1), torch.zeros(2, len(nm["D"])),
                          nm, s["norm_stats"])


# ── Hook resolution ──────────────────────────────────────────────


def test_evaluate_baseline_returns_results():
    s = _setup()
    out = evaluate_baseline(
        s["cfg"], s["sysid_cfg"], s["dyn_node"], s["test_dfs"],
        s["nm"], s["norm_stats"],
    )
    assert out is not None
    assert out.comfort and out.tracking


def test_no_rule_at_all_returns_none_rather_than_raising(monkeypatch):
    """No reference and no baseline hook just means no comparison."""
    import control_user_defined as cud
    s = _setup()
    monkeypatch.delattr(cud, "reference_control")

    assert evaluate_baseline(
        s["cfg"], s["sysid_cfg"], s["dyn_node"], s["test_dfs"],
        s["nm"], s["norm_stats"],
    ) is None


def test_the_escape_hatch_hook_is_still_honoured(monkeypatch):
    """A rule a memoryless function cannot express still gets a Node in."""
    import control_user_defined as cud
    s = _setup()
    monkeypatch.delattr(cud, "reference_control")
    monkeypatch.setattr(cud, "build_baseline_policy", lambda *a, **k: None,
                        raising=False)

    assert baseline_node(s["cfg"], s["sysid_cfg"], s["nm"],
                         s["norm_stats"]) is None


def test_defining_both_rules_is_an_error(monkeypatch):
    """Two definitions of one rule is two places for a cooling inset to drift."""
    import control_user_defined as cud
    s = _setup()
    monkeypatch.setattr(cud, "build_baseline_policy", lambda *a, **k: None,
                        raising=False)
    with pytest.raises(ValueError, match="both reference_control and"):
        baseline_node(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])


# ── The comparison itself ────────────────────────────────────────


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


def test_clears_reference_compares_mean_exceedance():
    """The decision rule: a policy that does not beat its own reference on the
    metric this module says to tune on has not earned its deployment."""
    from dataclasses import replace

    s = _setup()
    results = evaluate_baseline(
        s["cfg"], s["sysid_cfg"], s["dyn_node"], s["test_dfs"],
        s["nm"], s["norm_stats"],
    )
    exceedance = results.comfort_mean_exceedance

    assert clears_reference(results, None) is None
    assert clears_reference(results, results) is True
    assert clears_reference(
        results, replace(results, comfort_mean_exceedance=exceedance - 1.0)
    ) is False
    assert clears_reference(
        results, replace(results, comfort_mean_exceedance=exceedance + 1.0)
    ) is True
