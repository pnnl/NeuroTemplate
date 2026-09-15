"""The two grid-response objective terms: energy cost and soft peak.

Both are reductions of the *same* per-step power signal, so the tests here
check three things per term: it computes what its formula says on a
hand-computed batch, it reaches the policy parameters through the surrogate,
and it fails fast rather than substituting a proxy when its hook is missing.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import (
    _soft_peak,
    build_closed_loop,
    build_control_loss,
    build_problem,
)
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import build_window_dict, horizons
from control_fixtures import (
    PRICE_SCALE,
    alternating_price,
    full_setup,
    install_price,
)


def _rolled_out(s, **kw):
    """A real closed-loop rollout dict — what the loss is actually handed."""
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    data = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps, **kw
    )
    return policy, cl(data), data


def _term(loss, data, name):
    return float(loss(data)[name].detach())


def _power_of(s, out):
    """The scored-horizon power the loss terms all read."""
    import control_user_defined as hooks
    _, _, hmax = horizons(s["cfg"], s["sysid_cfg"])
    return hooks.power_metric(
        out["yn"][:, hmax:, :], out["U"][:, hmax:, :], out["D"][:, hmax:, :],
        s["nm"], s["norm_stats"],
    )


# ── The cost term ────────────────────────────────────────────────


def test_cost_is_the_price_weighted_mean_power(tmp_path):
    s = full_setup(comfort_weight=0.0, du_weight=0.0, energy_weight=0.0,
                   cost_weight=1.0)
    install_price(s["cfg"], tmp_path)
    _, _, hmax = horizons(s["cfg"], s["sysid_cfg"])
    _, out, data = _rolled_out(s)

    loss = build_control_loss(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])
    got = _term(loss, {**out, "price": data["price"]}, "cost_loss")

    power = _power_of(s, out)
    price = data["price"][:, hmax:, 0] * PRICE_SCALE
    assert got == pytest.approx(float((power * price).mean().detach()), rel=1e-5)


def test_shifting_the_tariff_changes_the_cost(tmp_path):
    """The cheap guard on price/power co-indexing: if the cost were reading a
    mis-aligned slice, rolling the tariff by a step would not move it. The
    tariff has to alternate per step for this to bite — a test window is under
    an hour long, so a realistic series is nearly constant across it."""
    s = full_setup(comfort_weight=0.0, du_weight=0.0, energy_weight=0.0,
                   cost_weight=1.0)
    install_price(s["cfg"], tmp_path, series=alternating_price())
    _, out, data = _rolled_out(s)
    loss = build_control_loss(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])

    straight = _term(loss, {**out, "price": data["price"]}, "cost_loss")
    shifted = _term(loss, {**out, "price": data["price"].roll(1, dims=1)},
                    "cost_loss")
    assert straight != pytest.approx(shifted)


def test_cost_without_a_tariff_hook_fails_fast():
    s = full_setup(cost_weight=1.0)
    with pytest.raises(NotImplementedError, match="train_price_path"):
        build_control_loss(s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"])


# ── The peak term ────────────────────────────────────────────────


def test_power_mean_is_monotone_in_q_and_bounded_by_the_max():
    p = torch.tensor([[1.0, 2.0, 3.0, 10.0]])
    values = [float(_soft_peak(p, q)) for q in (1.0, 2.0, 4.0, 8.0, 32.0)]
    assert values == sorted(values)
    assert values[-1] <= float(p.max()) + 1e-6
    assert values[-1] > values[0]


def test_a_flat_profile_scores_below_a_spiky_one_of_equal_energy():
    """The whole point of the term: same integral, different shape."""
    flat = torch.tensor([[4.0, 4.0, 4.0, 4.0]])
    spiky = torch.tensor([[0.0, 0.0, 0.0, 16.0]])
    assert float(flat.sum()) == float(spiky.sum())
    assert float(_soft_peak(flat, 4.0)) < float(_soft_peak(spiky, 4.0))


def test_peak_reduces_per_window_then_averages_over_the_batch():
    """A pooled peak would give gradient to the worst window and none to the
    rest; check the two-window arithmetic by hand."""
    p = torch.tensor([[0.0, 2.0], [6.0, 6.0]])
    q = 4.0
    w0 = ((0.0**q + 2.0**q) / 2) ** (1 / q)
    w1 = 6.0
    assert float(_soft_peak(p, q)) == pytest.approx((w0 + w1) / 2, rel=1e-6)


def test_peak_gradient_is_dense_over_time_not_one_hot():
    """A hard max would put one non-zero entry per window and the policy would
    learn to shave a single timestep."""
    p = torch.tensor([[1.0, 2.0, 3.0, 10.0]], requires_grad=True)
    _soft_peak(p, 4.0).backward()
    assert (p.grad != 0).all()
    # ...but weighted toward the high-power steps.
    assert p.grad[0, -1] > p.grad[0, 0]


# ── Both terms reach the policy ──────────────────────────────────


@pytest.mark.parametrize(
    "name, weight",
    [("energy_loss", "energy_weight"),
     ("cost_loss", "cost_weight"),
     ("peak_loss", "peak_weight")],
)
def test_every_power_term_reaches_the_policy_parameters(tmp_path, name,
                                                        weight):
    """The chain is theta -> U -> (STE quantize) -> surrogate -> yn ->
    power_metric -> term. A power metric that read only recorded D would have a
    zero gradient and train to nothing while looking fine."""
    s = full_setup(**{"comfort_weight": 0.0, "du_weight": 0.0,
                      "energy_weight": 0.0, weight: 1.0})
    install_price(s["cfg"], tmp_path)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]

    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm, ns)
    problem = build_problem(cl, loss)

    data = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    )
    out = problem({**data, "name": "train"})
    params = [p for p in policy.parameters() if p.requires_grad]
    grads = torch.autograd.grad(out[f"train_{name}"], params,
                                allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)


# ── The cost reaches the reportcard ──────────────────────────────


def test_price_stays_free_of_gradient_inside_the_loss(tmp_path):
    s = full_setup(comfort_weight=0.0, du_weight=0.0, energy_weight=0.0,
                   cost_weight=1.0)
    install_price(s["cfg"], tmp_path)
    _, out, data = _rolled_out(s)
    assert data["price"].requires_grad is False


def test_an_all_zero_power_window_does_not_poison_the_peak_gradient():
    """M_q's root has an infinite derivative at 0, and power_metric's clamp
    makes all-zero windows entirely plausible. Pin that it stays finite."""
    p = torch.zeros(1, 4, requires_grad=True)
    _soft_peak(p, 4.0).backward()
    assert torch.isfinite(p.grad).all()
