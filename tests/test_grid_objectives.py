"""The power terms: energy, cost and peak as rows of the declared objective.

All three are reductions of the *same* per-step power signal — a mean, a
weighted mean and a power mean — so the tests here check that a weighted term
computes what its formula says on a real rollout, that every power term
reaches the policy parameters through the surrogate, and that naming `power`
fails fast rather than substituting a proxy when the hook is missing.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import (
    build_closed_loop,
    build_control_loss,
    build_problem,
)
from neurotemplate.control.losses import resolve_objective
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import build_window_dict, horizons, scored_steps
from control_fixtures import (
    BAND_GAINS, BAND_LOSSES, POWER_LOSSES, TARIFF, TARIFF_SCALE, full_setup,
)


def _setup(**gains):
    """The band plus the three power terms; only the named gains are on."""
    return full_setup(
        losses={**BAND_LOSSES, **POWER_LOSSES},
        gains={**{k: 0.0 for k in BAND_GAINS}, "energy": 0.0, "cost": 0.0,
               "peak": 0.0, **gains},
        reference_scale={TARIFF: TARIFF_SCALE},
    )


def _rolled_out(s):
    """A real closed-loop rollout dict — what the loss is actually handed."""
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    data = build_window_dict(s["norm_df"], cfg, sysid_cfg, nm,
                             P=cfg.train_pred_steps, stride=cfg.history_steps)
    return cl(data)


def _cost(s, out):
    loss = build_control_loss(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    return float(loss(out)["cost_loss"].detach())


def _power_of(s, out):
    """The scored-horizon power the loss terms all read."""
    import control_user_defined as hooks
    _, _, hmax = horizons(s["cfg"], s["sysid_cfg"])
    return hooks.power_metric(
        out["yn"][:, scored_steps("yn", hmax), :],
        out["U"][:, scored_steps("U", hmax), :],
        out["D"][:, scored_steps("D", hmax), :],
        s["nm"], s["norm_stats"],
    )


def _tariff_index(s):
    """The tariff's position in R."""
    return resolve_objective(s["cfg"], s["nm"]).r_columns.index(TARIFF)


# ── The cost term ────────────────────────────────────────────────


def test_cost_is_the_tariff_weighted_mean_power():
    s = _setup(cost=1.0)
    _, _, hmax = horizons(s["cfg"], s["sysid_cfg"])
    out = _rolled_out(s)

    tariff = out["R"][:, hmax:, _tariff_index(s)] * TARIFF_SCALE   # $/kWh
    expected = (_power_of(s, out) * tariff).mean()
    assert _cost(s, out) == pytest.approx(float(expected.detach()), rel=1e-5)


def test_shifting_the_tariff_changes_the_cost():
    """The cheap guard on price/power co-indexing: the fixture tariff
    alternates every step, so a one-step misalignment would move the cost."""
    s = _setup(cost=1.0)
    out = _rolled_out(s)
    shifted = dict(out)
    shifted["R"] = out["R"].clone()
    j = _tariff_index(s)
    shifted["R"][..., j] = out["R"][..., j].roll(1, dims=1)
    assert _cost(s, out) != pytest.approx(_cost(s, shifted))


def test_a_weight_of_zeros_costs_exactly_zero():
    """No shift on a scaled reference column: a zero price is a zero cost."""
    s = _setup(cost=1.0)
    out = _rolled_out(s)
    out["R"][..., _tariff_index(s)] = 0.0
    assert _cost(s, out) == 0.0


def test_naming_power_without_the_hook_fails_fast(monkeypatch):
    import control_user_defined as hooks
    s = _setup(cost=1.0)
    monkeypatch.setattr(hooks, "POWER_METRIC_IMPLEMENTED", False)
    with pytest.raises(NotImplementedError, match="power_metric"):
        build_control_loss(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])


# ── Every power term reaches the policy ──────────────────────────


@pytest.mark.parametrize("name", ["energy", "cost", "peak"])
def test_every_power_term_reaches_the_policy_parameters(name):
    """The chain is theta -> U -> surrogate -> yn -> power_metric -> term. A
    power metric that read only recorded D would have a zero gradient and
    train to nothing while looking fine."""
    s = _setup(**{name: 1.0})
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    problem = build_problem(build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg),
                            build_control_loss(cfg, sysid_cfg, nm, ns))
    data = build_window_dict(s["norm_df"], cfg, sysid_cfg, nm,
                             P=cfg.train_pred_steps, stride=cfg.history_steps)
    out = problem({**data, "name": "train"})
    params = [p for p in policy.parameters() if p.requires_grad]
    grads = torch.autograd.grad(out[f"train_{name}_loss"], params,
                                allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
