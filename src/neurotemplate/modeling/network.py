"""
RC networks for a thermal zone, fitted by Kalman-filter maximum likelihood.

The grey-box method of Bacher & Madsen 2011 (*Identifying suitable models for
the heat dynamics of buildings*): a linear stochastic state-space model in
continuous time whose states are node temperatures, fitted to a measured zone
temperature by maximum likelihood. Pure functions over torch tensors in
float64; `nt data ladder` and the `rc_network` family call them.

The network
===========

    nodes    each a capacitance C (kWh/K). The first is the zone air, the one
             node a sensor reads.
    edges    each a resistance R (K/kW) between two nodes, or between a node
             and `ambient`, the outdoor temperature.
    inputs   each column enters one node as gain · value, in kW. Delivered
             heat is in kW already and its gain is pinned at 1; that pin is
             what puts C and R in physical units. Solar irradiance enters
             through a fitted aperture (kW per W/m²).

    C_i dT_i/dt = Σ_edges (T_j − T_i) / R_ij + Σ_inputs gain · u + noise

A passive network with a path to ambient is stable for any positive R and C,
so every parameter is a positive scale fitted in log space and no point of
the parameter space is unstable. Time is in hours throughout.

Noise
=====

Each node takes a Wiener push of its own size σ_i (K/√h): what the network
leaves out. The sensor adds white noise of size σ_y (K). Both are fitted, and
the steady-state Kalman gain they imply is the observer the model exports.

Discretisation
==============

Inputs are averaged over each step (heat is a rate, so the average keeps the
energy) and held, so the exact discrete model at step `dt` is the zero-order
hold `A = e^{A_c dt}`, `B = A_c⁻¹(A − I)B_c`, both read off one matrix
exponential. Continuous-time parameters mean the same thing at any step,
which is why a fit at 30 min initialises one at 5.

What resolution to fit at
=========================

Kalman maximum likelihood scores one-step predictions. At 5 min that is a
score on sensor noise and thermostat cycling, and it shows: on two plant
buildings the 5 min fit shortened or lost the slow mode and doubled the 24 h
rollout error of the 30 min fit. So the likelihood runs at a coarse step (the
ladder's job), and the fine fit is the rollout loss of training, seeded by
this filter and held inside boxes from the coarse error bars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

AMBIENT = "ambient"
DTYPE = torch.float64
#: Added to the sensor's noise variance (K²), a hundredth of a millikelvin
#: in σ: far below any sensor, and enough that the filter's Riccati solve
#: never divides by zero when the fit drives the sensor noise to nothing.
SENSOR_FLOOR = 1e-10


@dataclass(frozen=True)
class Network:
    """A network's structure: which nodes, which edges, where each input
    enters, and which parameters are pinned.

    `pinned` maps a parameter name (see `parameter_names`) to the value it is
    held at; delivered heat's gain is the usual one.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    inputs: tuple[tuple[str, str], ...]          # (column, node)
    pinned: dict[str, float] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        known = set(self.nodes) | {AMBIENT}
        problems = [f"edge {e} names an unknown node" for e in self.edges
                    if not set(e) <= known or e[0] == e[1]]
        problems += [f"input {c!r} enters unknown node {n!r}" for c, n in self.inputs
                     if n not in self.nodes]
        if not any(AMBIENT in e for e in self.edges):
            problems.append("no edge reaches ambient, so the network has no path to it")
        problems += [f"pinned {k!r} is not a parameter" for k in self.pinned
                     if k not in self.parameter_names()]
        if problems:
            raise ValueError(f"network {self.name or self.nodes}: " + "; ".join(problems))

    @property
    def zone(self) -> str:
        return self.nodes[0]

    def parameter_names(self) -> list[str]:
        """Every physical and noise parameter, in a fixed order."""
        return ([f"C_{n}" for n in self.nodes]
                + [f"R_{a}_{b}" for a, b in self.edges]
                + [f"gain_{c}_{n}" for c, n in self.inputs]
                + [f"sigma_{n}" for n in self.nodes]
                + ["sigma_y"])

    def free_names(self) -> list[str]:
        return [k for k in self.parameter_names() if k not in self.pinned]


@dataclass
class Series:
    """Data for one fit, on one regular grid of step `dt_h` hours.

    `y` [T] is the zone temperature sampled at each step; `ambient` [T] and
    `inputs` [T, m] are averaged over the step that starts there, in the
    order of the network's `inputs`. Several contiguous files are fitted
    as separate `Series` whose log-likelihoods add.
    """

    y: torch.Tensor
    ambient: torch.Tensor
    inputs: torch.Tensor
    dt_h: float


# ── Parameters ──────────────────────────────────────────────────

def unpack(network: Network, theta: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every parameter's value from the free ones in log space."""
    values, i = {}, 0
    for name in network.parameter_names():
        if name in network.pinned:
            values[name] = torch.tensor(float(network.pinned[name]), dtype=DTYPE)
        else:
            values[name] = torch.exp(theta[i])
            i += 1
    return values


def pack(network: Network, values: dict[str, float]) -> torch.Tensor:
    """The free parameters, log space, from values by name."""
    return torch.tensor([math.log(values[k]) for k in network.free_names()], dtype=DTYPE)


def continuous_matrices(network: Network, p: dict[str, torch.Tensor]
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """`(A_c, B_c)` per hour; B_c's columns are ambient, then the inputs."""
    n, index = len(network.nodes), {node: i for i, node in enumerate(network.nodes)}
    capacity = torch.stack([p[f"C_{node}"] for node in network.nodes])
    conductance = torch.zeros(n, n, dtype=DTYPE)
    to_ambient = torch.zeros(n, dtype=DTYPE)
    for a, b in network.edges:
        g = 1 / p[f"R_{a}_{b}"]
        if AMBIENT in (a, b):
            node = index[b if a == AMBIENT else a]
            to_ambient = to_ambient + g * _unit(n, node)
        else:
            i, j = index[a], index[b]
            conductance = conductance + g * (_pair(n, i, j) + _pair(n, j, i))
    laplacian = torch.diag(conductance.sum(1) + to_ambient) - conductance
    a_c = -laplacian / capacity[:, None]
    columns = [to_ambient / capacity]
    for column, node in network.inputs:
        columns.append(p[f"gain_{column}_{node}"] * _unit(n, index[node]) / capacity)
    return a_c, torch.stack(columns, dim=1)


def _unit(n: int, i: int) -> torch.Tensor:
    e = torch.zeros(n, dtype=DTYPE)
    e[i] = 1.0
    return e


def _pair(n: int, i: int, j: int) -> torch.Tensor:
    m = torch.zeros(n, n, dtype=DTYPE)
    m[i, j] = 1.0
    return m


def discretize(a_c: torch.Tensor, b_c: torch.Tensor, noise: torch.Tensor, dt_h: float
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(A, B, Q)` for one step of `dt_h` hours under zero-order hold, with
    `noise` the per-node σ (K/√h).

    Q is the stationary covariance's one-step deficit, `Σ − A Σ Aᵀ`, with Σ
    solving `A_c Σ + Σ A_cᵀ + diag(σ²) = 0`: exact because an RC network is
    always stable, and unlike Van Loan's method it never exponentiates
    `−A_c`, which overflows for a mode much faster than the step."""
    n, m = a_c.shape[0], b_c.shape[1]
    top = torch.cat([a_c, b_c], dim=1)
    augmented = torch.cat([top, torch.zeros(m, n + m, dtype=DTYPE)], dim=0)
    expm = torch.linalg.matrix_exp(augmented * dt_h)
    a, b = expm[:n, :n], expm[:n, n:]
    eye = torch.eye(n, dtype=DTYPE)
    lyapunov = torch.kron(eye, a_c) + torch.kron(a_c, eye)
    sigma = torch.linalg.solve(lyapunov, -torch.diag(noise ** 2).reshape(-1)).reshape(n, n)
    sigma = (sigma + sigma.T) / 2
    q = sigma - a @ sigma @ a.T
    return a, b, (q + q.T) / 2


def discrete_model(network: Network, p: dict[str, torch.Tensor], dt_h: float):
    """`(A, B, Q, r)`: the step at `dt_h` and the sensor's noise variance."""
    a_c, b_c = continuous_matrices(network, p)
    noise = torch.stack([p[f"sigma_{node}"] for node in network.nodes])
    a, b, q = discretize(a_c, b_c, noise, dt_h)
    return a, b, q, p["sigma_y"] ** 2 + SENSOR_FLOOR


def time_constants_h(network: Network, p: dict[str, torch.Tensor]) -> torch.Tensor:
    """The network's time constants in hours, fastest first: −1/λ over the
    eigenvalues of A_c, which are real and negative for an RC network."""
    a_c, _ = continuous_matrices(network, p)
    # A_c = −C⁻¹L with L symmetric, so C^½ A_c C^-½ is symmetric with the
    # same eigenvalues; the similarity needs no gradient of its own.
    root = torch.stack([p[f"C_{n}"] for n in network.nodes]).detach().sqrt()
    symmetric = root[:, None] * a_c / root[None, :]
    eigenvalues = torch.linalg.eigvalsh((symmetric + symmetric.T) / 2)
    return torch.sort(-1 / eigenvalues).values


# ── The Kalman filter ───────────────────────────────────────────

def affine_scan(f: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix compositions of the affine steps `x ← f_k x + c_k`, time first:
    `f` [T, ..., n, n] and `c` [T, ..., n], broadcasting over the middle
    axes. Returns `(F, c')` with `x_{k+1} = F_k x_0 + c'_k`. Hillis–Steele,
    so T steps cost log2(T) batched matmuls rather than T Python steps."""
    steps, d = f.shape[0], 1
    while d < steps:
        f, c = (torch.cat([f[:d], f[d:] @ f[:-d]]),
                torch.cat([c[:d], (f[d:] @ c[:-d, ..., None])[..., 0] + c[d:]]))
        d *= 2
    return f, c


def stationary_covariance(a, q, r, iterations: int = 60) -> torch.Tensor:
    """The predicted state covariance the filter settles to: the filtering
    Riccati equation's fixed point, by the structure-preserving doubling
    algorithm (each iteration doubles the horizon, so a mode of any length
    converges in a few dozen small solves, where iterating the recursion
    would take as many steps as the slowest mode is long)."""
    n = a.shape[0]
    eye = torch.eye(n, dtype=DTYPE)
    observe = torch.zeros(n, n, dtype=DTYPE)
    observe[0, 0] = 1.0
    a_k, g_k, h_k = a.T, observe / r, q
    for _ in range(iterations):
        w = torch.linalg.solve(eye + g_k @ h_k, eye)
        a_next = a_k @ w @ a_k
        g_k = g_k + a_k @ w @ g_k @ a_k.T
        h_next = h_k + a_k.T @ h_k @ w @ a_k
        settled = float(torch.max(torch.abs(h_next - h_k)).detach()) < 1e-14
        a_k, h_k = a_next, h_next
        if settled:
            break
    return (h_k + h_k.T) / 2


@dataclass
class Filtered:
    """One pass of the filter: per-step innovations (measured minus
    predicted) and their variances, and the predicted states [T + 1, n]:
    row k is the state before y_k is measured."""

    innovations: torch.Tensor
    variances: torch.Tensor
    states: torch.Tensor

    def nll(self) -> torch.Tensor:
        """−log L, constant included."""
        return 0.5 * (torch.log(2 * math.pi * self.variances)
                      + self.innovations ** 2 / self.variances).sum()


def kalman_filter(a, b, q, r, y, drive, x0) -> Filtered:
    """The stationary filter over `y` [T] with `drive` [T, 1 + m] (ambient,
    then the inputs), from the predicted state `x0`: measure y_k, then
    predict to k+1 with step k's drive. The zone is state 0.

    Stationary: the gain is the steady-state one throughout, as if the
    filter had always been running. That costs a little accuracy in the
    first steps of a series and buys a recursion that is affine in the
    state, `x ← A(I − K e₀ᵀ)x + A K y + B u`, which runs as a scan."""
    n = a.shape[0]
    cov = stationary_covariance(a, q, r)
    variance = cov[0, 0] + r
    gain = cov[:, 0] / variance
    observe = torch.zeros(n, dtype=DTYPE)
    observe[0] = 1.0
    f = a @ (torch.eye(n, dtype=DTYPE) - torch.outer(gain, observe))
    c = (a @ gain)[None, :] * y[:, None] + drive @ b.T
    prefix_f, prefix_c = affine_scan(f.expand(y.shape[0], -1, -1), c)
    states = torch.cat([x0[None], prefix_f @ x0 + prefix_c])
    return Filtered(y - states[:-1, 0], variance.expand(y.shape[0]), states)


def steady_gain(a, q, r) -> torch.Tensor:
    """The steady-state Kalman gain K for the zone measurement. With K the
    observer is the fixed linear update `x̂ ← x̂ + K (y − x̂[0])`, then the
    model step."""
    cov = stationary_covariance(a, q, r)
    return cov[:, 0] / (cov[0, 0] + r)


def drive_matrix(series: Series) -> torch.Tensor:
    return torch.cat([series.ambient[:, None], series.inputs], dim=1)


def hidden_count(network: Network) -> int:
    return len(network.nodes) - 1


def conditioned_filter(network: Network, p: dict[str, torch.Tensor], series: Series,
                       hidden_start: torch.Tensor) -> Filtered:
    """The filter over `series` conditioned on its first measurement: the
    zone starts at y₀ and each hidden node at y₀ plus its offset in
    `hidden_start`, which is estimated, as in Bacher & Madsen. Conditioning
    on y₀ rather than estimating the zone's start too keeps the likelihood
    bounded as the sensor noise goes to zero. Innovations start at y₁."""
    a, b, q, r = discrete_model(network, p, series.dt_h)
    x0 = series.y[0] + torch.cat([torch.zeros(1, dtype=DTYPE), hidden_start])
    drive = drive_matrix(series)
    return kalman_filter(a, b, q, r, series.y[1:], drive[1:], a @ x0 + b @ drive[0])


def negative_log_likelihood(network: Network, theta: torch.Tensor,
                            data: list[Series]) -> torch.Tensor:
    """−log L summed over the series. `theta` is the free parameters in log
    space, then each series' hidden-node starts (see `conditioned_filter`)."""
    k, h = len(network.free_names()), hidden_count(network)
    p = unpack(network, theta[:k])
    total = torch.zeros((), dtype=DTYPE)
    for i, series in enumerate(data):
        total = total + conditioned_filter(
            network, p, series, theta[k + i * h:k + (i + 1) * h]).nll()
    return total


# ── Fitting ─────────────────────────────────────────────────────

@dataclass
class Fit:
    network: Network
    values: dict[str, float]
    log_likelihood: float
    n_obs: int
    #: Free parameters, initial states included: what a likelihood-ratio
    #: test and BIC count.
    n_params: int
    #: Standard error of each free parameter's log, from the inverse Hessian;
    #: infinite for one the data does not place (see `summarize`).
    log_se: dict[str, float]
    #: Correlation between the free parameters' logs, in `free_names` order.
    correlation: torch.Tensor
    taus_h: list[float]
    #: Each time constant's standard error in hours, by the delta method.
    taus_se_h: list[float]
    #: One-step innovations per series, for the whiteness check.
    innovations: list[torch.Tensor]
    converged: bool

    @property
    def bic(self) -> float:
        return -2 * self.log_likelihood + self.n_params * math.log(self.n_obs)


def default_start(network: Network) -> dict[str, float]:
    """A plausible small building: zone air 1 kWh/K, every other node 10,
    every resistance 5 K/kW, gains 0.005 (an aperture in kW per W/m²), noise
    0.1 K/√h per node and 0.05 K on the sensor."""
    start = {}
    for name in network.parameter_names():
        if name.startswith("C_"):
            start[name] = 1.0 if name == f"C_{network.zone}" else 10.0
        elif name.startswith("R_"):
            start[name] = 5.0
        elif name.startswith("gain_"):
            start[name] = 0.005
        elif name == "sigma_y":
            start[name] = 0.05
        else:
            start[name] = 0.1
    return start


#: The optimizer sees each log-parameter through `LOG_LIMIT · tanh(θ / LOG_LIMIT)`,
#: so a line search's trial step can never reach a scale (e^25 ≈ 7·10¹⁰ either
#: way) where the matrix exponential overflows. Near any sensible value the
#: map is the identity.
LOG_LIMIT = 25.0


def fit(network: Network, data: list[Series], start: Optional[dict[str, float]] = None,
        max_iter: int = 300) -> Fit:
    """Maximum likelihood by L-BFGS over the free parameters in log space
    and each series' hidden-node starts.

    `start` (by name) overrides `default_start`, so a parent rung's fit
    seeds its extensions; every hidden node starts at the first measured
    zone temperature.
    """
    values = default_start(network)
    values.update({k: v for k, v in (start or {}).items() if k in values})
    k = len(network.free_names())
    hidden = torch.zeros(hidden_count(network) * len(data), dtype=DTYPE)
    raw = torch.cat([LOG_LIMIT * torch.atanh(pack(network, values) / LOG_LIMIT), hidden])
    raw.requires_grad_(True)

    def squash(t):
        return torch.cat([LOG_LIMIT * torch.tanh(t[:k] / LOG_LIMIT), t[k:]])

    optimizer = torch.optim.LBFGS([raw], lr=1, max_iter=max_iter, tolerance_grad=1e-6,
                                  tolerance_change=1e-9, history_size=20,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = negative_log_likelihood(network, squash(raw), data)
        if not torch.isfinite(loss):
            # A trial step into a degenerate network (a solve gone singular):
            # report it as far worse than anything seen, so the line search
            # backs off instead of comparing against NaN.
            return torch.tensor(1e12, dtype=DTYPE)
        loss.backward()
        return loss

    optimizer.step(closure)
    converged = optimizer.state[raw]["n_iter"] < max_iter
    return summarize(network, squash(raw).detach(), data, converged)


#: A log-parameter within this of `LOG_LIMIT` has run off to zero or infinity
#: (a noise the data does not need, a gain it cannot see): it is reported as
#: such and left out of the curvature.
COLLAPSE_MARGIN = 5.0
#: Curvature below this share of the largest is no curvature: the direction
#: is not placed by the data, and every parameter on it gets an infinite
#: error bar rather than a pseudo-inverse's false zero.
FLAT = 1e-9


def standard_errors(hessian: torch.Tensor, collapsed: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(covariance, se, unplaced)` from the Hessian of −log L, leaving out the
    `collapsed` parameters (at a limit). A direction with no curvature (see
    `FLAT`) is unplaced: every parameter on it gets an infinite error bar,
    and the covariance holds only the placed directions."""
    n = hessian.shape[0]
    keep = (~collapsed).nonzero().squeeze(1)
    sub = hessian[keep][:, keep]
    curvature, directions = torch.linalg.eigh((sub + sub.T) / 2)
    placed = curvature > FLAT * curvature.abs().max()
    inverse = torch.where(placed, 1 / curvature, torch.zeros_like(curvature))
    covariance = torch.zeros(n, n, dtype=DTYPE)
    covariance[keep[:, None], keep[None, :]] = (directions * inverse) @ directions.T
    unplaced = torch.ones(n, dtype=torch.bool)
    unplaced[keep] = (directions[:, ~placed] ** 2).sum(1) > 1e-6
    se = torch.where(unplaced, torch.full((n,), math.inf, dtype=DTYPE),
                     torch.diagonal(covariance).clamp(min=0).sqrt())
    return covariance, se, unplaced


def summarize(network: Network, theta: torch.Tensor, data: list[Series],
              converged: bool = True) -> Fit:
    """The fit's likelihood, error bars, time constants and innovations at
    `theta` (as `negative_log_likelihood` reads it)."""
    k = len(network.free_names())

    def nll(t):
        return negative_log_likelihood(network, t, data)

    collapsed = torch.zeros(theta.shape[0], dtype=torch.bool)
    collapsed[:k] = theta[:k].abs() > LOG_LIMIT - COLLAPSE_MARGIN
    covariance, se, unplaced = standard_errors(
        torch.autograd.functional.hessian(nll, theta), collapsed)
    finite = torch.where(unplaced, torch.ones_like(se), se)
    correlation = (covariance / torch.outer(finite, finite))[:k, :k]

    def taus(t):
        return time_constants_h(network, unpack(network, t[:k]))

    jac = torch.autograd.functional.jacobian(taus, theta)
    taus_var = torch.einsum("ij,jk,ik->i", jac, covariance, jac)
    taus_unplaced = (jac[:, unplaced].abs() > 1e-9).any(1)
    p = unpack(network, theta[:k])
    innovations, h = [], hidden_count(network)
    with torch.no_grad():
        for i, series in enumerate(data):
            filtered = conditioned_filter(network, p, series,
                                          theta[k + i * h:k + (i + 1) * h])
            innovations.append(filtered.innovations / filtered.variances.sqrt())
    return Fit(
        network=network,
        values={name: float(v) for name, v in p.items()},
        log_likelihood=-float(nll(theta)),
        n_obs=sum(series.y.shape[0] - 1 for series in data),
        n_params=theta.shape[0],
        log_se={name: float(s) for name, s in zip(network.free_names(), se)},
        correlation=correlation,
        taus_h=[float(t) for t in taus(theta)],
        taus_se_h=[math.inf if bad else float(v) for bad, v in
                   zip(taus_unplaced, taus_var.clamp(min=0).sqrt())],
        innovations=innovations,
        converged=converged,
    )


# ── Rollout ─────────────────────────────────────────────────────

def as_tensors(values: dict[str, float]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=DTYPE) for k, v in values.items()}


def filtered_states(network: Network, values: dict[str, float], series: Series
                    ) -> torch.Tensor:
    """Predicted states [T, n] filtering `series` with every node starting at
    y₀: row k is the state before y_{k+1} is measured, so row T − 1 is one
    step past the series. Long enough a series forgets the start."""
    with torch.no_grad():
        return conditioned_filter(network, as_tensors(values), series,
                                  torch.zeros(hidden_count(network), dtype=DTYPE)).states


def rollout(network: Network, values: dict[str, float], x0: torch.Tensor,
            drive: torch.Tensor, dt_h: float) -> torch.Tensor:
    """Node temperatures [T + 1, n] open loop from `x0` under `drive`
    [T, 1 + m]: row k is the state before step k's drive acts."""
    a, b, _, _ = discrete_model(network, as_tensors(values), dt_h)
    steps = drive.shape[0]
    with torch.no_grad():
        prefix_f, prefix_c = affine_scan(a.expand(steps, -1, -1), drive @ b.T)
        return torch.cat([x0[None], prefix_f @ x0 + prefix_c])


# ── The declaration ─────────────────────────────────────────────
#
#   {"nodes":  {"Ti": C, "Tm": C},                    kWh/K, zone first
#    "edges":  {"Ti-ambient": R, "Ti-Tm": R},          K/kW
#    "inputs": {"Heat [kW]": {"Ti": 1.0}, "GHI [W/m²]": {"Ti": gain}},
#    "noise":  {"Ti": σ, "Tm": σ, "sensor": σ_y}}      K/√h per node, K
#
# C, R and each gain are a scalar, which pins it, or a box [low, high];
# noise is a scalar per node and for the sensor.


def _round(x: float) -> float:
    return float(f"{x:.4g}")


def suggested_box(value: float, log_se: float, z: float = 3.0) -> list[float]:
    """`[value / f, value · f]` with `f = exp(z · se)`: symmetric in log space
    around the estimate, so a `BoundedParameter` starts a fit at it. A
    parameter the data could not place gets a box a factor of 1000 either
    way, which is the honest answer."""
    factor = math.exp(min(z * log_se, math.log(1000.0)))
    return [_round(value / factor), _round(value * factor)]


def at_limit(fit: Fit) -> list[str]:
    """The free parameters that ran off to zero or infinity (see
    `COLLAPSE_MARGIN`)."""
    bound = math.exp(LOG_LIMIT - COLLAPSE_MARGIN)
    return [k for k in fit.network.free_names()
            if not 1 / bound < fit.values[k] < bound]


def declaration(fit: Fit) -> dict:
    """A fit as a network declaration: boxes from its error bars (pinned
    parameters as scalars) and its noise as fitted. A gain that ran off to
    zero is left out: the data says that input does not reach that node."""
    net, values = fit.network, fit.values
    dropped = set(at_limit(fit))

    def box(name):
        if name in net.pinned:
            return _round(net.pinned[name])
        return suggested_box(values[name], fit.log_se[name])

    inputs: dict[str, dict] = {}
    for column, node in net.inputs:
        name = f"gain_{column}_{node}"
        if not (name in dropped and values[name] < 1):
            inputs.setdefault(column, {})[node] = box(name)
    return {
        "nodes": {node: box(f"C_{node}") for node in net.nodes},
        "edges": {f"{a}-{b}": box(f"R_{a}_{b}") for a, b in net.edges},
        "inputs": inputs,
        "noise": {**{node: _round(values[f"sigma_{node}"]) for node in net.nodes},
                  "sensor": _round(values["sigma_y"])},
        # A record, not a declaration: the time constants this fit found, with
        # their error bars (null where the data does not place one), for the
        # model card to hold a later fit against.
        "time_constants_h": [[_round(t), None if math.isinf(se) else _round(se)]
                             for t, se in zip(fit.taus_h, fit.taus_se_h)],
    }


def parse_declaration(decl: dict, name: str = "") -> tuple[Network, dict[str, tuple[float, float]],
                                                           dict[str, float]]:
    """`(network, boxes, noise)` from a declaration: the structure, every
    C/R/gain as a `(low, high)` box (a pin is `low == high`), and the noise
    by parameter name. Every problem is named at once."""
    problems = []
    keys = set(decl) - {"time_constants_h"}
    if keys != {"nodes", "edges", "inputs", "noise"}:
        problems.append("keys must be nodes, edges, inputs and noise (and the ladder's "
                        f"time_constants_h); got {sorted(decl)}")
        raise ValueError(f"network {name!r}: " + "; ".join(problems))
    boxes: dict[str, tuple[float, float]] = {}

    def take(param, value, what):
        pair = value if isinstance(value, (list, tuple)) else [value, value]
        if (len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  for v in pair) or not 0 < pair[0] <= pair[1]):
            problems.append(f"{what} must be a positive number or [low, high]; got {value!r}")
        else:
            boxes[param] = (float(pair[0]), float(pair[1]))

    nodes = tuple(decl["nodes"])
    for node, value in decl["nodes"].items():
        take(f"C_{node}", value, f"node {node!r}")
    edges = []
    for key, value in decl["edges"].items():
        ends = tuple(key.split("-"))
        if len(ends) != 2:
            problems.append(f"edge {key!r} must be two node names joined by '-'")
            continue
        edges.append(ends)
        take(f"R_{ends[0]}_{ends[1]}", value, f"edge {key!r}")
    inputs = []
    for column, entries in decl["inputs"].items():
        if not isinstance(entries, dict) or not entries:
            problems.append(f"input {column!r} must map node -> gain; got {entries!r}")
            continue
        for node, value in entries.items():
            inputs.append((column, node))
            take(f"gain_{column}_{node}", value, f"input {column!r} into {node!r}")
    noise = {}
    for key in (*nodes, "sensor"):
        value = decl["noise"].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"noise {key!r} must be a positive number; got {value!r}")
        else:
            noise["sigma_y" if key == "sensor" else f"sigma_{key}"] = float(value)
    extra = sorted(set(decl["noise"]) - {*nodes, "sensor"})
    if extra:
        problems.append(f"noise names unknown node(s) {extra}")
    if problems:
        raise ValueError(f"network {name!r}:\n  - " + "\n  - ".join(problems))
    pinned = {k: lo for k, (lo, hi) in boxes.items() if lo == hi}
    return Network(nodes, tuple(edges), tuple(inputs), pinned, name), boxes, noise


def box_centres(boxes: dict[str, tuple[float, float]]) -> dict[str, float]:
    """The geometric centre of each box: where a bounded fit starts."""
    return {k: math.sqrt(lo * hi) for k, (lo, hi) in boxes.items()}
