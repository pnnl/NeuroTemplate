import torch

from control_user_defined import (
    power_metric,
    power_channels,
    plot_context_channels,
)
from control_fixtures import (
    make_nm_mapping, make_raw_df, make_norm_stats,
    FAN_P, DX_P, DIST,
)


def test_power_channels_resolved():
    assert power_channels(make_nm_mapping()) == [FAN_P, DX_P]


def test_power_channels_empty_when_unmetered():
    """No power columns in Y turns the power row off rather than raising."""
    nm = make_nm_mapping()
    nm["Y"] = [c for c in nm["Y"] if c not in (FAN_P, DX_P)]
    assert power_channels(nm) == []


def test_plot_context_channels_picks_outdoor_drybulb():
    assert plot_context_channels(make_nm_mapping()) == [DIST[0]]


def test_power_metric_matches_fan_plus_dx():
    nm = make_nm_mapping()
    raw = make_raw_df()
    ns = make_norm_stats(raw)

    # Build a normalized yn from known physical values:
    # fan power = 400 W, DX power = 2000 W -> P = 2400 W per (b, t) entry.
    y = nm["Y"]
    b, t, ny = 2, 4, len(y)
    yn = torch.zeros(b, t, ny)
    phys_vals = {FAN_P: 400.0, DX_P: 2000.0}
    for i, c in enumerate(y):
        phys = phys_vals.get(c, 0.0)
        s = ns[c]
        yn[:, :, i] = (phys - s["shift"]) / s["stretch"]

    P = power_metric(
        yn, torch.zeros(b, t, len(nm["U"])), torch.zeros(b, t, len(nm["D"])), nm, ns,
    )
    assert P.shape == (b, t)
    assert torch.all(P >= 0)
    assert torch.allclose(P, torch.full((b, t), 2400.0), rtol=1e-4)


def test_power_metric_clamps_negative_to_zero():
    # Surrogate could predict slightly-negative power near zero; metric must
    # floor at 0 W rather than report negative electrical power.
    nm = make_nm_mapping()
    ns = make_norm_stats(make_raw_df())
    y = nm["Y"]
    yn = torch.zeros(1, 3, len(y))
    phys_vals = {FAN_P: -50.0, DX_P: -100.0}
    for i, c in enumerate(y):
        phys = phys_vals.get(c, 0.0)
        s = ns[c]
        yn[:, :, i] = (phys - s["shift"]) / s["stretch"]
    P = power_metric(
        yn, torch.zeros(1, 3, len(nm["U"])), torch.zeros(1, 3, len(nm["D"])), nm, ns,
    )
    assert torch.allclose(P, torch.zeros_like(P))
