"""RC networks fitted by Kalman maximum likelihood, and the ladder over them."""

import math

import torch

from neurotemplate.modeling import ladder, network as nw
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
    net = Rung().network(HEAT, None, ())
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
    net = Rung(mass=True).network(HEAT, None, ())
    truth = _truth(net)
    data = [_simulate(net, truth, steps=2000, seed=3)]
    fit = nw.fit(net, data)
    for name in ("C_Ti", "C_Tm", "R_Ti_Tm", "R_Ti_ambient"):
        error = abs(math.log(fit.values[name] / truth[name]))
        assert error < 3 * fit.log_se[name], (name, fit.values[name], truth[name])
    assert nw.declaration(fit)["inputs"][HEAT] == {"Ti": 1.0}


def _select(rung: Rung, seed: int) -> ladder.Selection:
    truth_net = rung.network(HEAT, None, ())
    data = [_simulate(truth_net, _truth(truth_net), steps=1500, seed=seed)]

    def fit_rung(r, parents):
        start = parents[0].values if parents else None
        return [nw.fit(r.network(HEAT, None, ()), data, start)]

    return ladder.forward_select(fit_rung, solar=False)


def test_the_ladder_climbs_to_the_three_node_network_that_made_the_data():
    assert _select(Rung(mass=True, envelope=True), seed=5).chosen == "TiTmTe"


def test_the_ladder_does_not_climb_past_the_two_node_network_that_made_the_data():
    assert _select(Rung(mass=True), seed=7).chosen == "TiTm"


def test_a_declaration_round_trips_with_its_boxes_and_pins():
    net = Rung(mass=True).network(HEAT, None, ())
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


def test_the_trained_block_filters_and_steps_exactly_as_the_ladder_does():
    from neurotemplate.modeling.rc_network import NetworkBlock

    net = Rung(mass=True).network(HEAT, None, ())
    values = _truth(net)
    series = _simulate(net, values, steps=60, seed=2)
    decl = {"nodes": {"Ti": values["C_Ti"], "Tm": values["C_Tm"]},
            "edges": {"Ti-Tm": values["R_Ti_Tm"], "Ti-ambient": values["R_Ti_ambient"]},
            "inputs": {HEAT: {"Ti": 1.0}},
            "noise": {"Ti": values["sigma_Ti"], "Tm": values["sigma_Tm"],
                      "sensor": values["sigma_y"]}}
    thermostat = {"cooling": {"setpoint": "sp", "gain": 4.0, "capacity": 6.0}, "form": "tracker"}
    columns = ("zone", HEAT, "sp", "out")
    stats = {c: {"shift": 0.0, "stretch": 1.0} for c in columns}
    block = NetworkBlock("z", ("zone", HEAT), ("sp",), ("out",), HEAT, "out",
                         {"network": decl, "thermostat": thermostat}, window=60, dt_h=DT_H,
                         norm_stats=stats)
    heat_meter = torch.cat([torch.zeros(1, dtype=nw.DTYPE), series.inputs[:-1, 0]])
    y = torch.stack([series.y, heat_meter], dim=1)          # the meter reads a step late
    x = torch.cat([y.flatten(), torch.tensor([20.0], dtype=nw.DTYPE), series.ambient])[None]
    with torch.no_grad():
        increment, state = block.step(x.float(), None, block.discrete())

    # The ladder's filter over the same rows, then one step of the network
    # under the thermostat's heat.
    ladder_series = nw.Series(series.y, series.ambient, series.inputs, DT_H)
    nodes = nw.filtered_states(net, values, ladder_series)[-2]
    a, b, q, r = nw.discrete_model(net, nw.as_tensors(values), DT_H)
    nodes = nodes + nw.steady_gain(a, q, r) * (series.y[-1] - nodes[0])
    with torch.no_grad():
        gain = block.thermostat_values()["cooling.gain"].item()
    heat = -min(max(gain * (float(nodes[0]) - 20.0), 0.0), 6.0)
    expected = a @ nodes + b @ torch.tensor([float(series.ambient[-1]), heat], dtype=nw.DTYPE)
    assert torch.allclose(state[0].double(), expected, atol=1e-4)
    assert math.isclose(float(increment[0, 1]), heat - float(heat_meter[-1]), abs_tol=1e-4)
