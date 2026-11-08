import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.losses import resolve_objective
from neurotemplate.control.policy import _NormalizedNet, build_policy
from neurotemplate.control.windows import (
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
        s["frame"] if df is None else df, cfg, s["sysid_cfg"], s["nm"],
        P=cfg.horizon_steps, stride=cfg.history_steps,
    )


def test_window_dict_shapes():
    s = full_setup()
    cfg, sysid_cfg, nm = s["cfg"], s["sysid_cfg"], s["nm"]
    _, _, hmax = horizons(cfg, sysid_cfg)
    W = hmax + cfg.horizon_steps

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


def test_a_loaded_reference_is_read_as_written_and_the_policy_scales_it_by_its_target(tmp_path):
    s = full_setup()
    cfg, nm, ns, plant = s["cfg"], s["nm"], s["norm_stats"], s["plant"]
    reference = make_reference_df(plant)
    write_splits(cfg, tmp_path, plant, [reference])
    (train,), _, _ = load_control_splits(cfg, nm)
    assert torch.allclose(torch.tensor(train[LOW].to_numpy()),
                          torch.tensor(reference[LOW].to_numpy()))

    # The band's floor reaches the policy's net normalised like the zone it bounds.
    policy = build_policy(cfg, s["sysid_cfg"], nm, ns)
    net = next(m for m in policy.callable.modules() if isinstance(m, _NormalizedNet))
    i = policy.input_keys.index("R")
    j = resolve_objective(cfg, nm).r_columns.index(LOW)
    zone = ns[nm["Y"][0]]
    assert getattr(net, f"_shift_{i}")[j].item() == pytest.approx(zone["shift"])
    assert getattr(net, f"_stretch_{i}")[j].item() == pytest.approx(zone["stretch"])
    assert HIGH in train.columns


def test_a_draw_replaces_what_it_shares_with_the_record_it_matches(tmp_path):
    s = full_setup()
    cfg, nm, plant = s["cfg"], s["nm"], s["plant"]
    base, later = make_reference_df(plant), make_reference_df(plant, seed=1)
    write_splits(cfg, tmp_path, plant, [base, later[[LOW]]])
    train, dev, test = load_control_splits(cfg, nm)
    assert len(train) == 2 and len(dev) == len(test) == 1     # dev and test take no draws
    assert train[1][LOW].tolist() == pytest.approx(later[LOW].tolist())
    assert train[1][HIGH].tolist() == pytest.approx(base[HIGH].tolist())
    assert train[0][LOW].tolist() == pytest.approx(base[LOW].tolist())


def test_a_draw_matching_no_record_is_refused_and_names_the_ranges(tmp_path):
    s = full_setup()
    plant = s["plant"]
    base = make_reference_df(plant)
    shifted = base.copy()
    shifted.index = shifted.index + (shifted.index[1] - shifted.index[0])
    write_splits(s["cfg"], tmp_path, plant, [base, shifted])
    with pytest.raises(ValueError, match=rf"covers {shifted.index[0]}.*matches no training record"):
        load_control_splits(s["cfg"], s["nm"])


def test_a_policy_reads_what_its_window_lists_and_names_the_rest():
    """No window entry, no input: a meter it does not read is not in the policy's
    inputs, and neither is a key none of whose columns it reads. The tariff's whole
    window is unread here, so R is the band's alone."""
    from control_fixtures import DIST, FAN_P, SETPT, ZONE
    window = {ZONE: {"past": "10m"}, DIST[0]: {"past": 0, "future": "5m"},
              LOW: {"past": 0, "future": 0}, HIGH: {"past": 0, "future": 0}}
    s = full_setup(policy_window=window)
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    assert policy.input_keys == ["yn", "D", "R"]
    r_columns = resolve_objective(cfg, nm).r_columns
    from neurotemplate.control.policy import read_columns, unread_columns
    assert read_columns(cfg, nm, r_columns)["yn"] == [ZONE]
    assert FAN_P in unread_columns(cfg, nm, r_columns) and SETPT in unread_columns(cfg, nm, r_columns)
    # past 2 + 1 of the zone, past 0 + 1 + future 1 of one D, one row of each band edge,
    # and time to go.
    net = next(m for m in policy.callable.modules() if isinstance(m, _NormalizedNet))
    assert net.base.in_features == 3 + 2 + 2 + 1
    u = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)(_windows(s))["U"]
    assert torch.isfinite(u).all()
