import torch

from neurotemplate.control.scenarios import (
    build_loaders,
    build_window_dict,
    build_window_dict_multi,
    horizons,
)
from control_fixtures import full_setup


def test_window_dict_shapes():
    s = full_setup()
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    P = cfg.train_pred_steps
    W = hmax + P

    d = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, s["norm_stats"],
        P=P, stride=cfg.in_steps,
    )
    n = d["yn"].shape[0]
    assert n > 0
    assert d["yn"].shape == (n, hmax, len(nm["Y"]))
    assert d["U"].shape == (n, hmax, len(nm["U"]))
    assert d["D"].shape == (n, W, len(nm["D"]))
    assert d["t"].shape == (n, W, 1)
    from control_user_defined import regulated
    n_ctrl = len(regulated(nm)[0])
    assert d["ymin"].shape == (n, W, n_ctrl)
    assert d["ymax"].shape == (n, W, n_ctrl)


def test_comfort_band_ordered():
    s = full_setup()
    d = build_window_dict(
        s["norm_df"], s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"],
        P=s["cfg"].train_pred_steps, stride=s["cfg"].in_steps,
    )
    # ymax strictly above ymin everywhere (band_width > 0, normalized).
    assert torch.all(d["ymax"] > d["ymin"])


def test_build_loaders_iterates():
    s = full_setup()
    train, dev = build_loaders(
        s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"],
        [s["norm_df"]], [s["norm_df"]],
    )
    batch = next(iter(train))
    for k in ("yn", "U", "D", "ymin", "ymax"):
        assert k in batch
    assert next(iter(dev)) is not None


def test_build_window_dict_multi_pools_and_tracks_source():
    """Two copies of the same df produce 2x as many windows as one,
    with a source_idx that partitions cleanly into two runs."""
    s = full_setup()
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    single = build_window_dict(
        s["norm_df"], cfg, sysid_cfg, nm, s["norm_stats"],
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    )
    multi = build_window_dict_multi(
        [s["norm_df"], s["norm_df"]], cfg, sysid_cfg, nm, s["norm_stats"],
        P=cfg.train_pred_steps, stride=cfg.in_steps,
    )
    n_single = single["yn"].shape[0]
    assert multi["yn"].shape[0] == 2 * n_single
    src = multi["source_idx"].tolist()
    assert src == [0] * n_single + [1] * n_single
