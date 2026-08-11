"""The tariff signal: how it enters the rollout and what it must satisfy.

Price is exogenous data — not a state, not a control, not a disturbance the
surrogate consumes — so it rides in the window dict and the dynamics never
sees it. These tests pin the three properties the cost objective rests on:
it is shaped like `t`, it is aligned to `t`, and it carries no gradient.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import (
    build_price,
    build_window_dict,
    build_window_dict_multi,
    price_scale,
)
from control_fixtures import (
    CHEAP,
    EXPENSIVE,
    PRICE_SCALE,
    full_setup,
    install_price_hook,
    tou_price,
)


def _window(s, **kw):
    return build_window_dict(
        s["norm_df"], s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"],
        P=s["cfg"].train_pred_steps, stride=s["cfg"].in_steps, **kw
    )


# ── Presence is the switch ───────────────────────────────────────


def test_no_hook_means_no_price_key():
    """Phase 0 changes nothing for a run without a tariff."""
    s = full_setup()
    assert "price" not in _window(s)
    assert price_scale(s["cfg"]) is None


def test_hook_adds_a_price_signal_shaped_like_t(monkeypatch):
    install_price_hook(monkeypatch)
    s = full_setup()
    data = _window(s)
    assert data["price"].shape == data["t"].shape
    assert price_scale(s["cfg"]) == pytest.approx(PRICE_SCALE)


def test_price_is_aligned_to_the_timestamps_of_its_own_window(monkeypatch):
    """The price at index i is the price at the timestamp of yn[i], which is
    what makes the cost term's co-indexing un-gettable-wrong."""
    install_price_hook(monkeypatch)
    s = full_setup()
    data = _window(s)
    expected = tou_price(data["t"])
    assert torch.equal(data["price"], expected)


def test_price_carries_no_gradient(monkeypatch):
    install_price_hook(monkeypatch)
    s = full_setup()
    assert _window(s)["price"].requires_grad is False


def test_price_pools_across_files(monkeypatch):
    install_price_hook(monkeypatch)
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    pooled = build_window_dict_multi(
        [s["norm_df"], s["norm_df"]], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    )
    single = _window(s)
    assert pooled["price"].shape[0] == 2 * single["price"].shape[0]
    assert torch.equal(pooled["price"][: single["price"].shape[0]],
                       single["price"])


# ── Fail fast on a tariff that would break the cost gradient ─────


def test_missing_price_scale_is_an_error(monkeypatch):
    import control_user_defined as cud
    s = full_setup()
    monkeypatch.setattr(cud, "price_signal", tou_price, raising=False)
    monkeypatch.delattr(cud, "PRICE_SCALE", raising=False)
    with pytest.raises(ValueError, match="PRICE_SCALE"):
        price_scale(s["cfg"])


def test_non_positive_price_scale_is_an_error(monkeypatch):
    s = full_setup()
    install_price_hook(monkeypatch, scale=0.0)
    with pytest.raises(ValueError, match="PRICE_SCALE"):
        price_scale(s["cfg"])


def test_negative_price_is_an_error(monkeypatch):
    s = full_setup()
    install_price_hook(monkeypatch, signal=lambda t: -torch.ones_like(t))
    with pytest.raises(ValueError, match="negative price"):
        build_price(s["cfg"], torch.zeros(2, 3, 1))


def test_wrong_shape_is_an_error(monkeypatch):
    s = full_setup()
    install_price_hook(monkeypatch, signal=lambda t: torch.ones_like(t)[:, :1, :])
    with pytest.raises(ValueError, match="must return the shape of t"):
        build_price(s["cfg"], torch.zeros(2, 3, 1))


# ── Passthrough ──────────────────────────────────────────────────


def test_price_survives_the_rollout_untouched(monkeypatch):
    """SystemPreview copies the input dict and only restacks keys a node
    writes, so an extra key reaches the loss and `out` unchanged."""
    install_price_hook(monkeypatch)
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    data = _window(s)
    cl = build_closed_loop(
        s["dyn_node"], build_policy(cfg, sysid_cfg, nm, ns), cfg, sysid_cfg,
    )
    out = cl(data)
    assert torch.equal(out["price"], data["price"])


def test_the_tariff_actually_varies_over_a_day(monkeypatch):
    """A flat price makes the cost term a rescaled energy term, so the fixture
    tariff has to have two levels for the cost tests to mean anything."""
    install_price_hook(monkeypatch)
    s = full_setup()
    t = torch.arange(0, 24 * 3600, 3600, dtype=torch.float32)[None, :, None]
    price = build_price(s["cfg"], t) * PRICE_SCALE
    assert float(price.min()) == pytest.approx(CHEAP)
    assert float(price.max()) == pytest.approx(EXPENSIVE)
