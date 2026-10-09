"""The reference rule as a baseline controller, and the comparison harness.

The rule the residual policy is anchored to is the same rule scored as the
baseline — one definition, two consumers — so these tests cover both roles.
"""

import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.pipeline import (
    clears_reference,
    evaluate_baseline,
    evaluate_closed_loop,
)
from neurotemplate.control.policy import build_reference_node
from neurotemplate.control.scenarios import horizons, scored_steps
from control_user_defined import actuation_bounds, reference_control
from control_fixtures import HIGH, LOW, SETPT, full_setup


def _setup(**overrides):
    s = full_setup(**overrides)
    s["test_dfs"] = [s["norm_df"]]
    return s


# ── The reference rule as a Node ─────────────────────────────────


def test_reference_node_has_no_trainable_parameters():
    """It is a rule, not a model — it must never reach the optimizer."""
    s = _setup()
    node = build_reference_node(s["cfg"], s["sysid_cfg"], s["nm"],
                                s["norm_stats"])
    assert [p for p in node.parameters() if p.requires_grad] == []


def test_reference_reads_the_band_by_name_and_clamps_to_actuator_bounds():
    s = _setup()
    nm, ns = s["nm"], s["norm_stats"]
    umin, umax = actuation_bounds(nm, ns)
    sp = ns[SETPT]

    def rule(low_c, high_c):
        """The rule's command, in °C, for a band given in °C."""
        r_now = {name: torch.full((2,), (value - ns[name]["shift"]) / ns[name]["stretch"])
                 for name, value in ((LOW, low_c), (HIGH, high_c))}
        u = reference_control(torch.zeros(2, len(nm["Y"])), r_now,
                              torch.zeros(2, len(nm["D"])), nm, ns)
        return u * sp["stretch"] + sp["shift"]

    # A band inside the actuator range comes through as its midpoint.
    assert torch.allclose(rule(sp["mean"] - 0.5, sp["mean"] + 0.3),
                          torch.full((2, 1), sp["mean"] - 0.1), atol=1e-4)
    # A band above the actuator ceiling is clamped, not extrapolated.
    top = sp["max"] + 5.0
    ceiling = umax * sp["stretch"] + sp["shift"]
    assert torch.allclose(rule(top, top), ceiling.expand(2, 1), atol=1e-4)


def test_reference_node_reads_the_current_step_in_the_closed_loop():
    """The Node has no input_map, so every signal arrives unwindowed — which
    is what makes _ReferenceRule's slicing the identity here.

    Also pins the alignment: the command at `U[hmax - 1 + k]` was chosen from
    the band at that same index, and applied in the same rollout step.
    """
    s = _setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    node = build_reference_node(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], node, cfg, sysid_cfg)
    results, out = evaluate_closed_loop(
        cfg, sysid_cfg, cl, s["test_dfs"], nm, ns,
    )
    _, _, hmax = horizons(cfg, sysid_cfg)
    commands = scored_steps("U", hmax)
    P = out["U"][:, commands].shape[1]
    r = out["R"][:, commands][:, :P]
    r_now = {HIGH: r[..., 0].reshape(-1), LOW: r[..., 1].reshape(-1)}
    n = r_now[LOW].shape[0]
    expected = reference_control(torch.zeros(n, len(nm["Y"])), r_now,
                                 torch.zeros(n, len(nm["D"])), nm, ns)
    expected = expected.reshape(out["U"][:, commands].shape)
    assert torch.allclose(out["U"][:, commands, :], expected, atol=1e-5)
    assert 0.0 <= results.comfort_violation_rate <= 1.0


# ── Hook resolution ──────────────────────────────────────────────


def test_no_rule_at_all_returns_none_rather_than_raising(monkeypatch):
    """No reference and no baseline hook just means no comparison."""
    import control_user_defined as cud
    s = _setup()
    monkeypatch.delattr(cud, "reference_control")

    assert evaluate_baseline(
        s["cfg"], s["sysid_cfg"], s["dyn_node"], s["test_dfs"],
        s["nm"], s["norm_stats"],
    ) is None


# ── The comparison itself ────────────────────────────────────────


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
