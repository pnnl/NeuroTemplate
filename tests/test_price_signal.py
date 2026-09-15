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
    build_price,
    build_window_dict,
    price_scale,
)
from control_fixtures import (
    MIDNIGHT,
    PLANT_ORIGIN,
    PRICE_SCALE,
    PRICE_SERIES,
    PRICE_STEPS_PER_DAY,
    flat_price,
    full_setup,
    install_price,
    time_of_day_price,
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


# ── Fail fast on a tariff the cost term cannot use ───────────────


def test_a_gap_in_the_price_series_is_an_error(tmp_path):
    """A NaN here would propagate silently into the cost term."""
    s = full_setup()
    install_price(s["cfg"], tmp_path, series=flat_price(float("nan"), length=10))
    with pytest.raises(ValueError, match="non-finite"):
        build_price(s["cfg"], 2, 3, s["cfg"].test_price_path)


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


# ── price_align ──────────────────────────────────────────────────
#
# The tariff is sampled independently of the weather, which by default puts
# its evening peak at a random hour of the plant's day. That is the right
# default — it is what stops "read the price" collapsing into "read the
# clock" — but it also means a real 8 pm peak is never drawn at 8 pm, so
# pre-cooling cannot be seen in a plot or scored by hour. `price_align: day`
# snaps the offset to a whole day, keeping the day random and the hours real.


def _price_starts(s, tmp_path, align, start=MIDNIGHT, start_step=0, days=10,
                  seed=0):
    """The first price value of every window, in physical units.

    The fixture series encodes step-of-day (`start_step` says where the file's
    first row sits within its own day), so these values *are* the time of day
    each window's tariff begins at.
    """
    s["cfg"].price_align = align
    install_price(s["cfg"], tmp_path,
                  series=time_of_day_price(days, start_step=start_step),
                  scale=1.0, start=start)
    data = _window(s, generator=_gen(seed))
    return data["price"][:, 0, 0]


def _plant_time_of_day(s):
    """Each window's start as a step of the plant's own day.

    Windows stride by `in_steps` from the fixture record's 06:35 origin.
    """
    origin = pd.Timestamp(PLANT_ORIGIN)
    origin_step = (origin.hour * 60 + origin.minute) // 5
    n = _window(s)["yn"].shape[0]
    stride = s["cfg"].in_steps
    return (origin_step + stride * torch.arange(n)) % PRICE_STEPS_PER_DAY


def test_day_alignment_draws_the_tariff_on_the_plants_clock(tmp_path):
    """The property the whole option exists for: an 8 pm peak at the plant's
    8 pm, so a pre-cool shows up where a buildings person would look for it."""
    s = full_setup()
    expected = _plant_time_of_day(s)
    starts = _price_starts(s, tmp_path, "day")

    assert starts.numel() > 1, "need several windows for this to mean anything"
    assert torch.equal(starts.long(), expected)


def test_day_alignment_reads_the_price_files_own_start_time(tmp_path):
    """A price export need not begin at midnight, so where its rows sit within
    their day is read from its first timestamp rather than assumed."""
    s = full_setup()
    expected = _plant_time_of_day(s)
    # Same tariff, written from 06:00 instead of midnight: the encoded series
    # is shifted to match, so the answer must not move.
    starts = _price_starts(s, tmp_path, "day", start="2026-01-01 06:00",
                           start_step=72)

    assert torch.equal(starts.long(), expected)
