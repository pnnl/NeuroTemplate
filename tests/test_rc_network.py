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
