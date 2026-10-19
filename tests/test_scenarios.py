import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import (
    build_loaders,
    build_window_dict,
    build_window_dict_multi,
    horizons,
    load_control_splits,
)
from control_fixtures import (
    HIGH, LOW, full_setup, make_reference_df, write_splits,
)

# The file-backed tests reuse one plant file for every split.
pytestmark = pytest.mark.filterwarnings("ignore:.*timestamps from training")


def _windows(s, df=None):
    cfg = s["cfg"]
    return build_window_dict(
        s["norm_df"] if df is None else df, cfg, s["sysid_cfg"], s["nm"],
        P=cfg.train_pred_steps, stride=cfg.history_steps,
    )


def test_window_dict_shapes():
    s = full_setup()
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    W = hmax + cfg.train_pred_steps

    d = _windows(s)
    n = d["yn"].shape[0]
    assert n > 0
    assert d["yn"].shape == (n, hmax, len(nm["Y"]))
    # One shorter than yn: the policy writes U before the dynamics reads it.
    assert d["U"].shape == (n, hmax - 1, len(nm["U"]))
    assert d["D"].shape == (n, W, len(nm["D"]))
    assert d["t"].shape == (n, W, 1)
    # The objective names the band's two columns, sorted; the tariff column
    # in the frame is named by nothing, so it is not windowed.
    assert d["R"].shape == (n, W, 2)


def test_build_loaders_iterates():
    s = full_setup()
    train, dev = build_loaders(
        s["cfg"], s["sysid_cfg"], s["nm"], [s["norm_df"]], [s["norm_df"]],
    )
    batch = next(iter(train))
    for k in ("yn", "U", "D", "R"):
        assert k in batch
    assert next(iter(dev)) is not None


def test_build_window_dict_multi_pools_and_tracks_source():
    """Two copies of the same df produce 2x as many windows as one,
    with a source_idx that partitions cleanly into two runs."""
    s = full_setup()
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    single = _windows(s)
    multi = build_window_dict_multi(
        [s["norm_df"], s["norm_df"]], cfg, sysid_cfg, nm,
        P=cfg.train_pred_steps, stride=cfg.history_steps,
    )
    n_single = single["yn"].shape[0]
    assert multi["yn"].shape[0] == 2 * n_single
    src = multi["source_idx"].tolist()
    assert src == [0] * n_single + [1] * n_single


def test_the_reference_survives_the_rollout_untouched():
    """SystemPreview copies the input dict and only restacks keys a node
    writes, so R reaches the loss and `out` unchanged — and never the
    dynamics, whose Node reads only yn, U and D."""
    s = full_setup()
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    data = _windows(s)
    cl = build_closed_loop(
        s["dyn_node"], build_policy(cfg, sysid_cfg, nm, ns), cfg, sysid_cfg,
    )
    assert torch.equal(cl(data)["R"], data["R"])
    assert "R" not in s["dyn_node"].input_keys


# ── Reference files ──────────────────────────────────────────────


def test_three_reference_draws_give_three_times_the_windows(tmp_path):
    """Several draws are several files against one plant record."""
    s = full_setup()
    cfg, nm, ns, plant = s["cfg"], s["nm"], s["norm_stats"], s["plant"]
    write_splits(cfg, tmp_path, plant, [make_reference_df(plant, seed=i)
                                        for i in range(3)])
    train, _, _ = load_control_splits(cfg, nm, ns)
    write_splits(cfg, tmp_path, plant, [make_reference_df(plant)])
    one, _, _ = load_control_splits(cfg, nm, ns)

    pooled = build_window_dict_multi(train, cfg, s["sysid_cfg"], nm,
                                     P=cfg.train_pred_steps, stride=cfg.history_steps)
    single = build_window_dict_multi(one, cfg, s["sysid_cfg"], nm,
                                     P=cfg.train_pred_steps, stride=cfg.history_steps)
    assert pooled["yn"].shape[0] == 3 * single["yn"].shape[0]
    # Same plant windows three times over, each against its own draw.
    n = single["yn"].shape[0]
    assert torch.equal(pooled["D"][:n], pooled["D"][n:2 * n])
    assert not torch.equal(pooled["R"][:n], pooled["R"][n:2 * n])


def test_a_loaded_reference_is_normalised_with_its_targets_stats(tmp_path):
    s = full_setup()
    cfg, nm, ns, plant = s["cfg"], s["nm"], s["norm_stats"], s["plant"]
    reference = make_reference_df(plant)
    write_splits(cfg, tmp_path, plant, [reference])
    (train,), _, _ = load_control_splits(cfg, nm, ns)
    zone = ns[nm["Y"][0]]
    expected = (reference[LOW] - zone["shift"]) / zone["stretch"]
    assert torch.allclose(torch.tensor(train[LOW].to_numpy()),
                          torch.tensor(expected.to_numpy()), atol=1e-6)
    assert HIGH in train.columns


def test_a_reference_off_its_plant_files_timestamps_is_refused(tmp_path):
    """Refused, not reconciled — and the message names both ranges."""
    s = full_setup()
    plant = s["plant"]
    shifted = make_reference_df(plant)
    shifted.index = shifted.index + (shifted.index[1] - shifted.index[0])
    write_splits(s["cfg"], tmp_path, plant, [shifted])
    with pytest.raises(ValueError, match=(
            rf"reference covers {shifted.index[0]} to {shifted.index[-1]}.*"
            rf"plant file {plant.index[0]} to {plant.index[-1]}")):
        load_control_splits(s["cfg"], s["nm"], s["norm_stats"])
