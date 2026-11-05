"""control_bounds: the one way to bound a command, a fixed range or one read per step."""
import pytest
import torch

from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.losses import resolve_objective
from neurotemplate.control.policy import _per_step_bounds, build_policy, control_box
from neurotemplate.control.scenarios import build_window_dict, horizons
from control_fixtures import SETPT, full_setup

LOW, HIGH = "Comfort Low [°C]", "Comfort High [°C]"


def _rollout(**overrides):
    s = full_setup(**overrides)
    cfg, sysid_cfg = s["cfg"], s["sysid_cfg"]
    policy = build_policy(cfg, sysid_cfg, s["nm"], s["norm_stats"])
    net = policy.callable
    while not hasattr(net, "linear"):
        net = net.base
    with torch.no_grad():
        net.linear[-1].bias.normal_(0, 5)          # push the commands around the box
    windows = build_window_dict(s["frame"], cfg, sysid_cfg, s["nm"],
                                P=cfg.train_pred_steps, stride=cfg.history_steps)
    out = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)(windows)
    return s, out


def test_a_bounded_setpoint_stays_inside_the_band_at_every_step():
    s, out = _rollout(control_bounds={SETPT: [LOW, HIGH]})
    _, _, hmax = horizons(s["cfg"], s["sysid_cfg"])
    nm = s["nm"]
    u = out["U"][:, hmax - 1:, nm["U"].index(SETPT)]
    r_columns = resolve_objective(s["cfg"], nm).r_columns
    # The command at U[i] was chosen from the signals at step i, the band's
    # included (see scenarios.scored_steps), so both are read from hmax - 1.
    low = out["R"][:, hmax - 1:, r_columns.index(LOW)]
    high = out["R"][:, hmax - 1:, r_columns.index(HIGH)]
    steps = min(u.shape[1], low.shape[1])
    assert (u[:, :steps] >= low[:, :steps] - 1e-4).all()
    assert (u[:, :steps] <= high[:, :steps] + 1e-4).all()


def test_bounds_name_columns_the_policy_can_read():
    with pytest.raises(ValueError, match="not a command or a reference column"):
        _rollout(control_bounds={SETPT: ["Nowhere [°C]", HIGH]})


def test_a_column_not_listed_takes_its_recorded_range_and_a_number_replaces_it():
    s = full_setup()
    umin, umax = control_box(s["cfg"], s["nm"], s["norm_stats"], "cpu")
    stats = s["norm_stats"][SETPT]
    assert umin.item() == pytest.approx(stats["min"]) and umax.item() == pytest.approx(stats["max"])
    s = full_setup(control_bounds={SETPT: [22.0, 26.0]})
    umin, umax = control_box(s["cfg"], s["nm"], s["norm_stats"], "cpu")
    assert (umin.item(), umax.item()) == (22.0, 26.0)


def test_a_fixed_range_holds_the_commands_and_needs_no_per_step_map():
    s, out = _rollout(control_bounds={SETPT: [24.0, 25.0]})
    _, _, hmax = horizons(s["cfg"], s["sysid_cfg"])
    u = out["U"][:, hmax - 1:, 0]
    assert (u >= 24.0 - 1e-4).all() and (u <= 25.0 + 1e-4).all()


def test_bounds_name_u_columns():
    s = full_setup(control_bounds={"Nowhere [°C]": [0.0, 1.0]})
    with pytest.raises(ValueError, match="not U columns"):
        control_box(s["cfg"], s["nm"], s["norm_stats"], "cpu")


def test_a_bound_may_sit_a_margin_inside_a_band_edge():
    """[column, offset]: the command stays at least the margin under the band's
    ceiling, so equipment that starts past its setpoint does not overshoot."""
    s, out = _rollout(control_bounds={SETPT: [LOW, [HIGH, -0.5]]})
    _, _, hmax = horizons(s["cfg"], s["sysid_cfg"])
    nm = s["nm"]
    u = out["U"][:, hmax - 1:, nm["U"].index(SETPT)]
    r_columns = resolve_objective(s["cfg"], nm).r_columns
    high = out["R"][:, hmax - 1:, r_columns.index(HIGH)]
    steps = min(u.shape[1], high.shape[1])
    assert (u[:, :steps] <= high[:, :steps] - 0.5 + 1e-4).all()


def _ordered(bounds, u):
    """`_per_step_bounds` over two commands in °C on a stub
    policy that emits `u`; returns the bounded commands."""
    heat, cool = "Heat [°C]", "Cool [°C]"
    nm = {"U": [heat, cool], "D": []}
    base = lambda *args: u                      # noqa: E731
    net = _per_step_bounds(base, bounds, nm, ["yn"], {}, [],
                           torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    return net(torch.zeros(1)), heat, cool


def test_a_command_bounded_by_another_stays_ordered_whatever_the_declaration_order():
    """Cooling at least 1 °C above heating at every step, with the two declared
    either way round: the bounds are mapped in the order they read each other."""
    u = torch.rand(256, 2)
    for bounds in ({"Heat [°C]": [15.0, 22.0], "Cool [°C]": [["Heat [°C]", 1.0], 30.0]},
                   {"Cool [°C]": [["Heat [°C]", 1.0], 30.0], "Heat [°C]": [15.0, 22.0]}):
        out, _, _ = _ordered(bounds, u)
        assert (out[:, 1] >= out[:, 0] + 1.0 - 1e-5).all()
        assert (out[:, 0] >= 15.0 - 1e-5).all() and (out[:, 0] <= 22.0 + 1e-5).all()


def test_bounds_that_read_each_other_in_a_cycle_are_refused():
    with pytest.raises(ValueError, match="cycle"):
        _ordered({"Heat [°C]": [15.0, "Cool [°C]"], "Cool [°C]": ["Heat [°C]", 30.0]},
                 torch.rand(4, 2))
