"""The tariff signal: how it enters the rollout and what it must satisfy.

Price is exogenous data — not a state, not a control, not a disturbance the
surrogate consumes — so it rides in the window dict and the dynamics never
sees it. It is *sampled* per window from a real historical series, exactly
like the comfort band, rather than derived from the clock: a tariff that is a
deterministic function of hour-of-day is a tariff the policy can learn as the
clock, and D is hour-of-day correlated enough to make that a real shortcut.

These tests pin the properties the cost objective rests on: the shape, the
physical→scaled conversion, reproducibility under a seeded generator, and
that price windows are drawn independently of the weather they are scored
beside.
"""

import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import (
    build_price,
    build_window_dict,
    build_window_dict_multi,
    horizons,
    price_scale,
)
from control_fixtures import (
    PRICE_SCALE,
    PRICE_SERIES,
    flat_price,
    full_setup,
    install_price_hook,
    sampled_price,
)


def _window(s, generator=None):
    return build_window_dict(
        s["norm_df"], s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"],
        P=s["cfg"].train_pred_steps, stride=s["cfg"].in_steps,
        generator=generator,
    )


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── Presence is the switch ───────────────────────────────────────


def test_no_hook_means_no_price_key():
    """A run without a tariff is byte-for-byte the run it was before."""
    s = full_setup()
    assert "price" not in _window(s)
    assert price_scale(s["cfg"]) is None


def test_hook_adds_a_price_window_per_sample(monkeypatch):
    install_price_hook(monkeypatch)
    s = full_setup()
    data = _window(s, _gen())
    assert data["price"].shape == (data["t"].shape[0], data["t"].shape[1], 1)
    assert price_scale(s["cfg"]) == pytest.approx(PRICE_SCALE)


def test_the_dict_carries_the_scaled_price_and_the_hook_returns_dollars(monkeypatch):
    """One clear contract: the hook returns $/kWh, the library normalizes."""
    install_price_hook(monkeypatch, signal=flat_price(0.6))
    s = full_setup()
    assert torch.allclose(_window(s)["price"],
                          torch.full_like(_window(s)["price"], 0.6 / PRICE_SCALE))


def test_price_carries_no_gradient(monkeypatch):
    install_price_hook(monkeypatch)
    s = full_setup()
    assert _window(s, _gen())["price"].requires_grad is False


# ── Sampled, not derived ─────────────────────────────────────────


def test_price_windows_come_out_of_the_series(monkeypatch):
    """Every sampled window is a contiguous slice of the historical record."""
    install_price_hook(monkeypatch)
    s = full_setup()
    price = _window(s, _gen())["price"] * PRICE_SCALE
    W = price.shape[1]
    slices = torch.stack([
        PRICE_SERIES[i:i + W, 0] for i in range(PRICE_SERIES.shape[0] - W + 1)
    ])
    for w in range(price.shape[0]):
        assert (slices == price[w, :, 0]).all(dim=1).any(), (
            "sampled window is not a contiguous slice of the price series"
        )


def test_the_same_seed_draws_the_same_tariff(monkeypatch):
    """What makes the learned/baseline comparison apples-to-apples: the
    evaluation reseeds from EVAL_BAND_SEED, so both see the same prices."""
    install_price_hook(monkeypatch)
    s = full_setup()
    assert torch.equal(_window(s, _gen(7))["price"], _window(s, _gen(7))["price"])


def test_enabling_price_leaves_the_band_realization_untouched(monkeypatch):
    """Price is drawn after the band, so turning it on does not silently
    re-roll every comfort band and invalidate a comparison across runs."""
    s = full_setup()
    before = _window(s, _gen(3))
    install_price_hook(monkeypatch)
    after = _window(s, _gen(3))
    assert torch.equal(before["ymin"], after["ymin"])
    assert torch.equal(before["ymax"], after["ymax"])


def test_price_windows_are_independent_of_the_weather_windows(monkeypatch):
    """The deliberate decoupling, stated as sharply as it can be: the weather
    windows are a deterministic stride over the file, the price windows are
    drawn. Two runs therefore share the former and differ in the latter, so
    the price↔weather correlation cannot serve as a second route to learning
    the clock."""
    install_price_hook(monkeypatch)
    s = full_setup()
    a, b = _window(s, _gen(1)), _window(s, _gen(2))
    assert torch.equal(a["D"], b["D"])
    assert not torch.equal(a["price"], b["price"])


def test_price_pools_across_files(monkeypatch):
    install_price_hook(monkeypatch)
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    pooled = build_window_dict_multi(
        [s["norm_df"], s["norm_df"]], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps, generator=_gen(5),
    )
    assert pooled["price"].shape[0] == pooled["yn"].shape[0]
    assert pooled["price"].shape[1] == pooled["D"].shape[1]


# ── Fail fast on a tariff the cost term cannot use ───────────────


def test_missing_price_scale_is_an_error(monkeypatch):
    import control_user_defined as cud
    s = full_setup()
    monkeypatch.setattr(cud, "price_signal", sampled_price, raising=False)
    monkeypatch.delattr(cud, "PRICE_SCALE", raising=False)
    with pytest.raises(ValueError, match="PRICE_SCALE"):
        price_scale(s["cfg"])


def test_non_positive_price_scale_is_an_error(monkeypatch):
    s = full_setup()
    install_price_hook(monkeypatch, scale=0.0)
    with pytest.raises(ValueError, match="PRICE_SCALE"):
        price_scale(s["cfg"])


def test_wrong_shape_is_an_error(monkeypatch):
    s = full_setup()
    install_price_hook(monkeypatch,
                       signal=lambda n, w, g=None: torch.ones(n, w))
    with pytest.raises(ValueError, match=r"must return \[4, 6, 1\]"):
        build_price(s["cfg"], 4, 6)


def test_a_gap_in_the_price_series_is_an_error(monkeypatch):
    """A NaN here would propagate silently into the cost term."""
    s = full_setup()
    install_price_hook(
        monkeypatch,
        signal=lambda n, w, g=None: torch.full((n, w, 1), float("nan")),
    )
    with pytest.raises(ValueError, match="non-finite"):
        build_price(s["cfg"], 2, 3)


def test_a_negative_price_is_allowed(monkeypatch):
    """Markets do pay to absorb load, and the cost term should want it."""
    s = full_setup()
    install_price_hook(monkeypatch, signal=flat_price(-0.05))
    price = build_price(s["cfg"], 2, 3)
    assert float(price.max()) < 0


# ── Passthrough ──────────────────────────────────────────────────


def test_price_survives_the_rollout_untouched(monkeypatch):
    """SystemPreview copies the input dict and only restacks keys a node
    writes, so an extra key reaches the loss and `out` unchanged."""
    install_price_hook(monkeypatch)
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    data = _window(s, _gen())
    cl = build_closed_loop(
        s["dyn_node"], build_policy(cfg, sysid_cfg, nm, ns), cfg, sysid_cfg,
    )
    out = cl(data)
    assert torch.equal(out["price"], data["price"])


def test_the_fixture_tariff_varies_within_and_across_days():
    """A flat price makes the cost term a rescaled energy term, and a price
    that repeats daily is learnable as the clock. The fixture series has to
    avoid both for the cost tests to mean anything."""
    day = 24 * 12
    assert PRICE_SERIES[:day].min() < PRICE_SERIES[:day].max()
    assert not torch.equal(PRICE_SERIES[:day], PRICE_SERIES[day:2 * day])
