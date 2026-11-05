"""RC networks fitted by Kalman maximum likelihood, and the ladder over them."""

import math

import pytest

import torch

from neurotemplate.modeling import ladder, network as nw, thermostat as th
from neurotemplate.modeling.ladder import Rung

DT_H = 0.5
HEAT = "Heat [kW]"


def _drive(steps: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A day-cycling outdoor temperature and heat switched between levels
    every few hours, the way a test sequence excites a zone."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(steps, dtype=nw.DTYPE) * DT_H
    ambient = 15 + 6 * torch.sin(2 * math.pi * t / 24) + torch.randn(steps, generator=g,
                                                                     dtype=nw.DTYPE) * 0.3
    levels = torch.tensor([-6.0, 0.0, 4.0], dtype=nw.DTYPE)
    hold = int(3 / DT_H)
    heat = levels[torch.randint(0, 3, (steps // hold + 1,), generator=g)].repeat_interleave(hold)
    return ambient, heat[:steps, None]


def _simulate(network, values, steps=1500, seed=0) -> nw.Series:
    """The stochastic network itself: process noise on every node, sensor noise."""
    ambient, inputs = _drive(steps, seed)
    a, b, q, r = nw.discrete_model(network, nw.as_tensors(values), DT_H)
    g = torch.Generator().manual_seed(seed + 1)
    chol = torch.linalg.cholesky(q + 1e-12 * torch.eye(q.shape[0], dtype=nw.DTYPE))
    x = torch.full((len(network.nodes),), 18.0, dtype=nw.DTYPE)
    drive = torch.cat([ambient[:, None], inputs], dim=1)
    ys = []
    for k in range(steps):
        ys.append(x[0] + r.sqrt() * torch.randn((), generator=g, dtype=nw.DTYPE))
        x = a @ x + b @ drive[k] + chol @ torch.randn(len(x), generator=g, dtype=nw.DTYPE)
    return nw.Series(torch.stack(ys), ambient, inputs, DT_H)


TRUTH = {"C_Ti": 1.5, "C_Tm": 20.0, "C_Te": 6.0, "R_Ti_Tm": 0.4, "R_Ti_Te": 0.8,
         "R_Te_ambient": 4.0, "R_Ti_ambient": 3.0, "sigma_Ti": 0.15, "sigma_Tm": 0.05,
         "sigma_Te": 0.05, "sigma_y": 0.03, f"gain_{HEAT}_Ti": 1.0}


def _truth(network):
    return {k: TRUTH[k] for k in network.parameter_names()}


def test_the_likelihood_is_the_hand_worked_one_state_filter():
    net = Rung().network((HEAT,), None, (), {HEAT: 1.0})
    values = _truth(net)
    series = _simulate(net, values, steps=200)
    c, r_o, sigma, sigma_y = (values[k] for k in ("C_Ti", "R_Ti_ambient", "sigma_Ti", "sigma_y"))
    a = math.exp(-DT_H / (r_o * c))
    b_ambient, b_heat = 1 - a, (1 - a) * r_o          # zero-order hold of 1/(RC), 1/C
    q = sigma ** 2 * r_o * c / 2 * (1 - a ** 2)        # the exact OU step variance
    r = sigma_y ** 2 + nw.SENSOR_FLOOR
    # The stationary predicted variance: P = a²P r / (P + r) + q, a quadratic.
    p = ((q + r * (a * a - 1)) + math.sqrt((q + r * (a * a - 1)) ** 2 + 4 * q * r)) / 2
    gain, s = p / (p + r), p + r
    y, amb, heat = series.y.tolist(), series.ambient.tolist(), series.inputs[:, 0].tolist()
    x = a * y[0] + b_ambient * amb[0] + b_heat * heat[0]
    nll = 0.0
    for k in range(1, len(y)):
        e = y[k] - x
        nll += 0.5 * (math.log(2 * math.pi * s) + e * e / s)
        x = a * (x + gain * e) + b_ambient * amb[k] + b_heat * heat[k]
    theta = nw.pack(net, values)
    assert math.isclose(float(nw.negative_log_likelihood(net, theta, [series])), nll,
                        rel_tol=1e-9)


def test_a_fit_on_its_own_network_recovers_the_parameters_within_their_error_bars():
    net = Rung(mass=True).network((HEAT,), None, (), {HEAT: 1.0})
    truth = _truth(net)
    data = [_simulate(net, truth, steps=2000, seed=3)]
    fit = nw.fit(net, data)
    for name in ("C_Ti", "C_Tm", "R_Ti_Tm", "R_Ti_ambient"):
        error = abs(math.log(fit.values[name] / truth[name]))
        assert error < 3 * fit.log_se[name], (name, fit.values[name], truth[name])
    assert nw.declaration(fit)["inputs"][HEAT] == {"Ti": 1.0}


def _select(rung: Rung, seed: int) -> ladder.Selection:
    truth_net = rung.network((HEAT,), None, (), {HEAT: 1.0})
    data = [_simulate(truth_net, _truth(truth_net), steps=1500, seed=seed)]

    def fit_rung(r, parents):
        start = parents[0].values if parents else None
        return [nw.fit(r.network((HEAT,), None, (), {HEAT: 1.0}), data, start)]

    return ladder.forward_select(fit_rung, solar=False)


def test_the_ladder_climbs_to_the_three_node_network_that_made_the_data():
    assert _select(Rung(mass=True, envelope=True), seed=5).chosen == "TiTmTe"


def test_the_ladder_does_not_climb_past_the_two_node_network_that_made_the_data():
    assert _select(Rung(mass=True), seed=7).chosen == "TiTm"


def test_an_extension_whose_new_mode_the_data_cannot_place_is_not_selected():
    """A decoupled node (a mode slower than the record) buys likelihood
    from noise; the ladder stops below it rather than selecting it."""
    from types import SimpleNamespace

    def stub(ll, n_params, unplaced):
        return SimpleNamespace(log_likelihood=ll, n_params=n_params,
                               unplaced_modes=[1e9] * unplaced, bic=-2 * ll)

    fits = {"Ti": stub(0.0, 4, 0), "TiTm": stub(50.0, 8, 1), "TiTe": stub(1.0, 8, 0)}
    selection = ladder.forward_select(lambda rung, parents: [fits[rung.name]], solar=False)
    assert selection.chosen == "Ti"
    assert selection.steps[0].unidentified == ["TiTm"]
    assert selection.steps[0].best == "TiTe"


def test_a_declaration_round_trips_with_its_boxes_and_pins():
    net = Rung(mass=True).network((HEAT,), None, (), {HEAT: 1.0})
    decl = {"nodes": {"Ti": [1.0, 2.0], "Tm": 20.0},
            "edges": {"Ti-Tm": [0.2, 0.8], "Ti-ambient": [2.0, 4.0]},
            "inputs": {HEAT: {"Ti": 1.0}},
            "noise": {"Ti": 0.1, "Tm": 0.05, "sensor": 0.03}}
    parsed, boxes, noise = nw.parse_declaration(decl)
    assert parsed.nodes == net.nodes and set(parsed.edges) == set(net.edges)
    assert parsed.pinned == {"C_Tm": 20.0, f"gain_{HEAT}_Ti": 1.0}
    assert boxes["R_Ti_Tm"] == (0.2, 0.8) and noise["sigma_y"] == 0.03


def test_the_thermostat_fit_recovers_a_trackers_capacity_and_offset():
    from neurotemplate.modeling import thermostat as th

    g = torch.Generator().manual_seed(11)
    steps = 3000
    setpoint = 24 + torch.randint(-2, 3, (steps // 12 + 1,), generator=g).repeat_interleave(12)
    setpoint = setpoint[:steps].to(nw.DTYPE)
    zone = setpoint + torch.randn(steps, generator=g, dtype=nw.DTYPE) * 1.5
    outdoor = torch.full((steps,), 30.0, dtype=nw.DTYPE)
    gain, capacity, offset = 20.0, 12.0, 0.4
    heat = -(gain * (zone - setpoint - offset)).clamp(0, capacity)
    measured = torch.cat([torch.zeros(1, dtype=nw.DTYPE), heat[:-1]])   # the meter reads at k+1
    measured = measured + torch.randn(steps, generator=g, dtype=nw.DTYPE) * 0.5
    record = th.Record(zone, {"cooling": setpoint}, outdoor, measured)
    fit = th.fit(th.Form(offset=True), ("cooling",), [record])
    for name, truth, log in (("capacity", capacity, True), ("offset", offset, False)):
        value, se = fit.values[f"cooling.{name}"], fit.se[f"cooling.{name}"]
        error = abs(math.log(value / truth)) if log else abs(value - truth)
        assert error < 3 * se, (name, value, truth, se)


def _block_and_ladder(net, values, series):
    """A zone's declaration, and the ladder's filter over the same rows: the
    nodes after the last measurement, and the step's matrices."""
    decl = {"nodes": {"Ti": values["C_Ti"], "Tm": values["C_Tm"]},
            "edges": {"Ti-Tm": values["R_Ti_Tm"], "Ti-ambient": values["R_Ti_ambient"]},
            "inputs": {HEAT: {"Ti": 1.0}},
            "noise": {"Ti": values["sigma_Ti"], "Tm": values["sigma_Tm"],
                      "sensor": values["sigma_y"]}}
    nodes = nw.filtered_states(net, values, series)[-2]
    a, b, q, r = nw.discrete_model(net, nw.as_tensors(values), DT_H)
    return decl, nodes + nw.steady_gain(a, q, r) * (series.y[-1] - nodes[0]), a, b


def test_the_zone_filters_and_steps_exactly_as_the_ladder_does():
    """The heat over the step arrives held at its row, whoever produced it."""
    from neurotemplate.modeling.thermal import RCZone

    net = Rung(mass=True).network((HEAT,), None, (), {HEAT: 1.0})
    values = _truth(net)
    series = _simulate(net, values, steps=60, seed=2)
    decl, nodes, a, b = _block_and_ladder(net, values, series)
    zone = RCZone(decl, [HEAT], window=60, dt_h=DT_H).eval()
    command = series.inputs[:, 0].float()[None]
    with torch.no_grad():
        next_zone = zone(series.y.float()[None], series.ambient.float()[None], command)
    expected = a @ nodes + b @ torch.tensor([float(series.ambient[-1]), float(command[0, -1])],
                                            dtype=nw.DTYPE)
    assert torch.allclose(zone.rollout_state()[0].double(), expected, atol=1e-4)
    assert math.isclose(float(next_zone[0]), float(expected[0]), abs_tol=1e-4)


def _two_blocks(decl, thermostat_decl, history_steps=60):
    from neurotemplate.modeling.blocks import build_blocks
    from neurotemplate.modeling.dynamics import get_dynamics
    from model_fixtures import model_cfg, unit_stats

    mapping = {"Y": ["zone", HEAT], "U": ["sp"], "D": ["out"], "t": ["sec_elapsed"]}
    spec = {"hvac": {"Y": [HEAT], "U": ["sp"], "D": [], "reads": ["zone"], "model": "thermostat",
                     "params": {"thermostat": thermostat_decl}},
            "room": {"Y": ["zone"], "U": [], "D": ["out"], "driven_by": [HEAT], "ambient": "out",
                     "scale": {HEAT: 1.0}, "model": "rc_network", "params": {"rc_network": decl}}}
    cfg = model_cfg(model="rc_network", past={"zone": 0, HEAT: 0, "sp": 0, "out": 0},
                    context=f"{history_steps * 30}m" if history_steps else 0,
                    steps_per_hour=1 / DT_H)
    assert [b.name for b in build_blocks(mapping, spec)] == ["hvac", "room"]
    return get_dynamics(cfg, mapping, unit_stats(mapping), block_spec=spec)


def test_a_zone_behind_a_thermostat_block_sees_the_heat_of_the_step_being_taken():
    from neurotemplate.modeling.contract import contract_model

    net = Rung(mass=True).network((HEAT,), None, (), {HEAT: 1.0})
    values = _truth(net)
    series = _simulate(net, values, steps=60, seed=2)
    decl, nodes, a, b = _block_and_ladder(net, values, series)
    system = _two_blocks(decl, {"cooling": {"setpoint": "sp", "gain": 4.0, "capacity": 6.0},
                                "form": "tracker", "gain_limit": 40.0}).eval()
    meter = torch.cat([torch.zeros(1, dtype=nw.DTYPE), series.inputs[:-1, 0]])
    y = torch.stack([series.y, meter], dim=1).float()[None]
    u = torch.full((1, 60, 1), 20.0)
    d = series.ambient.float()[None, :, None]
    with torch.no_grad():
        out = contract_model(system)(y, u, d)[0]
    # The thermostat answers the measured zone, and the network steps on that.
    heat = -min(max(4.0 * (float(series.y[-1]) - 20.0), 0.0), 6.0)
    expected = a @ nodes + b @ torch.tensor([float(series.ambient[-1]), heat], dtype=nw.DTYPE)
    assert math.isclose(float(out[1]), heat, abs_tol=1e-4)
    assert math.isclose(float(out[0]), float(expected[0]), abs_tol=1e-4)


def test_a_capacity_near_zero_leaves_every_gradient_finite():
    """A fitted outdoor slope can take a capacity to 1e-25 at some hour. The heat survives
    that; without a floor on the smoothing width the gradient does not, and one NaN gradient
    is every weight lost."""
    capacity = torch.tensor([1e-25, 8.0], dtype=torch.float32, requires_grad=True)
    gain = torch.tensor(10.0, requires_grad=True)
    th.soft_clip(gain * torch.tensor([0.1, 0.1]), capacity, smooth=True).sum().backward()
    assert torch.isfinite(capacity.grad).all() and torch.isfinite(gain.grad).all()

    p = {"cooling.gain": gain, "cooling.on_at": torch.tensor(0.1),
         "cooling.off_at": torch.tensor(0.1)}
    on, _ = th._switches(p, "cooling", torch.tensor([0.1, 0.1]), capacity)
    assert torch.isfinite(torch.autograd.grad(on.sum(), capacity)[0]).all()

    width = torch.tensor([1e-3, 0.4, 3.7], dtype=torch.float32)
    assert torch.equal(th._floored_width(width), width)     # above the floor, nothing changes


def test_a_network_that_pins_nothing_is_refused():
    """Temperature alone places only gain / C: nothing pinned, nothing physical."""
    decl = {"nodes": {"Ti": [0.5, 2.0]}, "edges": {"Ti-ambient": [1.0, 4.0]},
            "inputs": {HEAT: {"Ti": [0.5, 2.0]}}, "noise": {"Ti": 0.1, "sensor": 0.1}}
    with pytest.raises(ValueError, match="pins nothing"):
        _two_blocks(decl, {"cooling": {"setpoint": "sp", "gain": 4.0, "capacity": 6.0},
                           "form": "tracker", "gain_limit": 40.0})


def test_hidden_nodes_without_context_are_refused():
    decl = {"nodes": {"Ti": 1.0, "Tm": 10.0}, "edges": {"Ti-Tm": 1.0, "Ti-ambient": 2.0},
            "inputs": {HEAT: {"Ti": 1.0}}, "noise": {"Ti": 0.1, "Tm": 0.1, "sensor": 0.1}}
    with pytest.raises(ValueError, match="context"):
        _two_blocks(decl, {"cooling": {"setpoint": "sp", "gain": 4.0, "capacity": 6.0},
                           "form": "tracker", "gain_limit": 40.0}, history_steps=0)


def test_only_the_heat_gain_is_unidentified_beside_c_and_r():
    """The scale a declaration must state: scaling C, the gains and 1 / R together
    leaves every prediction, and so the likelihood, where it was."""
    net = nw.Network(("Ti", "Tm"), (("Ti", "Tm"), ("Ti", nw.AMBIENT)), ((HEAT, "Ti"),))
    base = {"C_Ti": 1.5, "C_Tm": 12.0, "R_Ti_Tm": 2.0, "R_Ti_ambient": 4.0,
            f"gain_{HEAT}_Ti": 1.0, "sigma_Ti": 0.1, "sigma_Tm": 0.05, "sigma_y": 0.03}
    scaled = {k: v * 3 if k[:2] in ("C_", "ga") else v / 3 if k.startswith("R_") else v
              for k, v in base.items()}
    series = _simulate(Rung(mass=True).network((HEAT,), None, (), {HEAT: 1.0}), _truth(
        Rung(mass=True).network((HEAT,), None, (), {HEAT: 1.0})), steps=60, seed=3)
    series = nw.Series(series.y, series.ambient, series.inputs, DT_H)
    start = torch.zeros(1, dtype=nw.DTYPE)
    nll = [float(nw.conditioned_filter(net, nw.as_tensors(v), series, start).nll())
           for v in (base, scaled)]
    assert math.isclose(nll[0], nll[1], rel_tol=1e-9)


def test_a_pinned_capacitance_lets_the_ladder_place_a_gain_as_a_cop():
    """`scale` can pin a C instead: then an input's gain is the fit, and it comes
    back at the truth, in the form a power's COP would."""
    truth_net = Rung().network((HEAT,), None, (), {HEAT: 2.5})
    values = {"C_Ti": 1.5, "R_Ti_ambient": 3.0, f"gain_{HEAT}_Ti": 2.5,
              "sigma_Ti": 0.05, "sigma_y": 0.02}
    series = _simulate(truth_net, values, steps=2000, seed=5)
    net = Rung().network((), None, (HEAT,), {"C_Ti": 1.5})
    assert net.pinned == {"C_Ti": 1.5}
    fit = nw.fit(net, [series], max_iter=200)
    assert abs(fit.values[f"gain_{HEAT}_Ti"] - 2.5) < 0.25
    decl = nw.declaration(fit)
    parsed, _, _ = nw.parse_declaration(decl)
    assert parsed.pinned == {"C_Ti": 1.5}


def test_a_plain_forward_is_a_model_with_the_penalty_and_named_columns():
    """The same parts, wired in a module of your own: its contract call is all the
    library needs, the sign penalty included, and the state survives the penalty."""
    from neurotemplate.modeling.contract import named_columns
    from neurotemplate.modeling.config import ModelingConfig
    from neurotemplate.modeling.gradient_sampling import SampledGradientPenalty, signed_model
    from neurotemplate.modeling.thermal import RCZone, thermostat
    from model_fixtures import unit_stats

    mapping = {"Y": ["zone", HEAT], "U": ["sp"], "D": ["out"], "t": ["sec_elapsed"]}
    net = Rung(mass=True).network((HEAT,), None, (), {HEAT: 1.0})
    decl = _block_and_ladder(net, _truth(net), _simulate(net, _truth(net), steps=12, seed=2))[0]
    hvac_decl = {"cooling": {"setpoint": "sp", "gain": 4.0, "capacity": 6.0}, "form": "tracker",
                 "gain_limit": 40.0}

    class Building(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hvac = thermostat(hvac_decl)
            self.zone = RCZone(decl, [HEAT], window=12, dt_h=DT_H)

        def forward(self, y, u, d):
            c = named_columns(mapping, y, u, d)
            heat = self.hvac(u[:, -1], c["zone"][:, -1])
            held = torch.cat([c[HEAT][:, 1:], heat[:, None]], dim=1)
            return torch.stack([self.zone(c["zone"], c["out"], held), heat], dim=-1)

    model = Building()
    stats = unit_stats(mapping)
    masks = {"U": torch.zeros(1, 1, 2, 1)}
    masks["U"][0, 0, 1, 0] = 1            # a higher cooling setpoint cools less: heat rises
    cfg = ModelingConfig()
    penalty = SampledGradientPenalty(signed_model(model, mapping, stats, 12), masks, 8, "cpu", cfg)
    window = [torch.randn(2, 12, n) for n in (2, 1, 1)]
    model(*window)
    held = model.zone.rollout_state()
    assert float(penalty.compute()) == 0.0
    assert model.zone.rollout_state() is held

    masks["U"][0, 0, 1, 0] = -1           # the wrong sign is scored, and reaches the gain
    penalty = SampledGradientPenalty(signed_model(model, mapping, stats, 12), masks, 8, "cpu", cfg)
    torch.manual_seed(0)
    value = penalty.compute()
    assert float(value) > 0.0
    assert torch.autograd.grad(value, next(model.hvac.parameters()), allow_unused=True)[0] is not None


def test_a_family_finds_only_its_own_params():
    """Priors for a family a block is swapped out of stay put; a missing entry is loud."""
    decl = {"nodes": {"Ti": 1.0}, "edges": {"Ti-ambient": 2.0}, "inputs": {HEAT: {"Ti": 1.0}},
            "noise": {"Ti": 0.1, "sensor": 0.1}}
    from neurotemplate.modeling.dynamics import get_dynamics
    from model_fixtures import model_cfg, unit_stats
    mapping = {"Y": ["zone"], "U": [HEAT], "D": ["out"], "t": ["sec_elapsed"]}
    spec = {"room": {"Y": ["zone"], "U": [HEAT], "D": ["out"], "ambient": "out",
                     "scale": {HEAT: 1.0}, "params": {"rc_network": decl}}}

    def build(**over):
        cfg = model_cfg(past={c: 0 for c in ("zone", HEAT, "out")}, steps_per_hour=1 / DT_H,
                        **over)
        return get_dynamics(cfg, mapping, unit_stats(mapping), block_spec=spec)

    build(model="rc_network")
    build(model="mlp")
    spec["room"]["params"] = {}
    with pytest.raises(ValueError, match="rc_network. entry"):
        build(model="rc_network")
    spec["room"]["params"] = {"rc_netwrok": decl}
    with pytest.raises(ValueError, match="rc_netwrok"):
        build(model="mlp")
