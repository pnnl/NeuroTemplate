import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.policy import build_policy
from neurotemplate.control.scenarios import (
    build_window_dict,
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
