import torch

from control_user_defined import power_metric
from control_fixtures import (
    make_nm_mapping, make_raw_df, make_norm_stats, FAN_P, DX_P,
)


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
