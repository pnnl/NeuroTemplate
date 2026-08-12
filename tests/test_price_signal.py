"""The tariff signal: how it enters the rollout and what it must satisfy.

Price is exogenous data — not a state, not a control, not a disturbance the
surrogate consumes — so it rides in the window dict and the dynamics never
sees it. It is *sampled* per window from a real historical CSV, one per
split, exactly like the comfort band, rather than derived from the clock: a
tariff that is a deterministic function of hour-of-day is a tariff the policy
can learn as the clock, and D is hour-of-day correlated enough to make that a
real shortcut.

These tests pin the properties the cost objective rests on: the shape, the
physical->scaled conversion, reproducibility under a seeded generator, and
that price windows are drawn independently of the weather they are scored
beside.
"""

import pandas as pd
import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import (
    _load_price_series,
    build_price,
    build_window_dict,
    build_window_dict_multi,
    horizons,
    price_scale,
)
from control_fixtures import (
    PRICE_SCALE,
    PRICE_SERIES,
    _write_price_csv,
    flat_price,
    full_setup,
    install_price,
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


def test_hook_adds_a_price_window_per_sample(tmp_path):
    s = full_setup()
    install_price(s["cfg"], tmp_path)
    data = _window(s, _gen())
    assert data["price"].shape == (data["t"].shape[0], data["t"].shape[1], 1)
    assert price_scale(s["cfg"]) == pytest.approx(PRICE_SCALE)


def test_the_dict_carries_the_scaled_price_and_the_csv_carries_dollars(tmp_path):
    """One clear contract: the CSV carries $/kWh, the library normalizes."""
    s = full_setup()
    install_price(s["cfg"], tmp_path, series=flat_price(0.6))
    assert torch.allclose(_window(s)["price"],
                          torch.full_like(_window(s)["price"], 0.6 / PRICE_SCALE))


def test_price_carries_no_gradient(tmp_path):
    s = full_setup()
    install_price(s["cfg"], tmp_path)
    assert _window(s, _gen())["price"].requires_grad is False


# ── Sampled, not derived ─────────────────────────────────────────


def test_price_windows_come_out_of_the_series(tmp_path):
    """Every sampled window is a contiguous slice of the historical record."""
    s = full_setup()
    install_price(s["cfg"], tmp_path)
    price = _window(s, _gen())["price"] * PRICE_SCALE
    W = price.shape[1]
    slices = torch.stack([
        PRICE_SERIES[i:i + W, 0] for i in range(PRICE_SERIES.shape[0] - W + 1)
    ])
    for w in range(price.shape[0]):
        assert (slices == price[w, :, 0]).all(dim=1).any(), (
            "sampled window is not a contiguous slice of the price series"
        )


def test_the_same_seed_draws_the_same_tariff(tmp_path):
    """What makes the learned/baseline comparison apples-to-apples: the
    evaluation reseeds from EVAL_BAND_SEED, so both see the same prices."""
    s = full_setup()
    install_price(s["cfg"], tmp_path)
    assert torch.equal(_window(s, _gen(7))["price"], _window(s, _gen(7))["price"])


def test_enabling_price_leaves_the_band_realization_untouched(tmp_path):
    """Price is drawn after the band, so turning it on does not silently
    re-roll every comfort band and invalidate a comparison across runs."""
    s = full_setup()
    before = _window(s, _gen(3))
    install_price(s["cfg"], tmp_path)
    after = _window(s, _gen(3))
    assert torch.equal(before["ymin"], after["ymin"])
    assert torch.equal(before["ymax"], after["ymax"])


def test_price_windows_are_independent_of_the_weather_windows(tmp_path):
    """The deliberate decoupling, stated as sharply as it can be: the weather
    windows are a deterministic stride over the file, the price windows are
    drawn. Two runs therefore share the former and differ in the latter, so
    the price↔weather correlation cannot serve as a second route to learning
    the clock."""
    s = full_setup()
    install_price(s["cfg"], tmp_path)
    a, b = _window(s, _gen(1)), _window(s, _gen(2))
    assert torch.equal(a["D"], b["D"])
    assert not torch.equal(a["price"], b["price"])


def test_price_repeat_upsamples_a_coarser_file(tmp_path):
    """price_repeat expands a file coarser than the plant's timestep (e.g.
    hourly price over 5-minute data) onto the plant's per-step grid."""
    path = str(tmp_path / "hourly.csv")
    _write_price_csv(path, torch.tensor([[0.1], [0.2], [0.3]]))
    series = _load_price_series(path, None, 12)
    assert series.shape == (36, 1)
    assert torch.allclose(series[:12], torch.full((12, 1), 0.1))
    assert torch.allclose(series[12:24], torch.full((12, 1), 0.2))
    assert torch.allclose(series[24:], torch.full((12, 1), 0.3))


def test_multiple_price_columns_requires_price_column(tmp_path):
    path = str(tmp_path / "multi.csv")
    idx = pd.date_range("2026-01-01", periods=5, freq="5min")
    pd.DataFrame({"timestamp": idx, "a": [0.1] * 5, "b": [0.2] * 5}).to_csv(
        path, index=False,
    )
    with pytest.raises(ValueError, match="price_column"):
        _load_price_series(path, None, 1)


def test_price_column_picks_the_named_column(tmp_path):
    path = str(tmp_path / "multi.csv")
    idx = pd.date_range("2026-01-01", periods=5, freq="5min")
    pd.DataFrame({"timestamp": idx, "a": [0.1] * 5, "b": [0.2] * 5}).to_csv(
        path, index=False,
    )
    series = _load_price_series(path, "b", 1)
    assert torch.allclose(series, torch.full((5, 1), 0.2))


def test_price_pools_across_files(tmp_path):
    s = full_setup()
    install_price(s["cfg"], tmp_path)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    pooled = build_window_dict_multi(
        [s["norm_df"], s["norm_df"]], cfg, sysid_cfg, nm, ns,
        P=cfg.train_pred_steps, stride=cfg.in_steps, generator=_gen(5),
    )
    assert pooled["price"].shape[0] == pooled["yn"].shape[0]
    assert pooled["price"].shape[1] == pooled["D"].shape[1]


# ── Fail fast on a tariff the cost term cannot use ───────────────


def test_missing_price_scale_is_an_error():
    s = full_setup()
    s["cfg"].train_price_path = "unused.csv"
    with pytest.raises(ValueError, match="price_scale"):
        price_scale(s["cfg"])


def test_non_positive_price_scale_is_an_error():
    s = full_setup()
    s["cfg"].train_price_path = "unused.csv"
    s["cfg"].price_scale = 0.0
    with pytest.raises(ValueError, match="price_scale"):
        price_scale(s["cfg"])


def test_a_gap_in_the_price_series_is_an_error(tmp_path):
    """A NaN here would propagate silently into the cost term."""
    s = full_setup()
    install_price(s["cfg"], tmp_path, series=flat_price(float("nan"), length=10))
    with pytest.raises(ValueError, match="non-finite"):
        build_price(s["cfg"], 2, 3, s["cfg"].test_price_path)


def test_a_negative_price_is_allowed(tmp_path):
    """Markets do pay to absorb load, and the cost term should want it."""
    s = full_setup()
    install_price(s["cfg"], tmp_path, series=flat_price(-0.05))
    price = build_price(s["cfg"], 2, 3, s["cfg"].test_price_path)
    assert float(price.max()) < 0


# ── Passthrough ──────────────────────────────────────────────────


def test_price_survives_the_rollout_untouched(tmp_path):
    """SystemPreview copies the input dict and only restacks keys a node
    writes, so an extra key reaches the loss and `out` unchanged."""
    s = full_setup()
    install_price(s["cfg"], tmp_path)
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
