"""
The RC ladder: forward selection over a thermal zone's RC networks.

Bacher & Madsen 2011's procedure, on the rungs below. Start at the simplest
network; fit every one-step extension of the current one, keep the one with
the best likelihood, and test it against its parent with a likelihood-ratio
test, 2(ℓ₁ − ℓ₀) against χ²(Δp). Stop when the best extension's p-value is
above 5%. Nested pairs only: siblings such as `TiTm` and `TiTe` are compared
by likelihood, never tested against each other.

The rungs (node names as in the paper):

    Ti      zone air, to ambient                               1R1C
    +Tm     interior mass, coupled to the zone only            (`TiTm`: the classic 2R2C)
    +Te     envelope between zone and ambient, replacing the zone's own
            edge to ambient                                    (`TiTe`)
    +Am/Ae  solar into the mass or envelope too, one more aperture
    +Ria    a direct zone–ambient edge beside the envelope

What the block's `scale` names is pinned at its value: a heat in kW at 1, or a
C from design data. Temperature alone places only gain / C, so without a pin C,
R and every gain are placed only up to a common factor. Every other input (solar,
an occupancy schedule, a heat whose gain is the fit) enters the zone through a
fitted gain.
Each extension adds a node's C, R, process noise and start (four
parameters), or one gain or resistance. A node with no capacity to speak of
is its parent's edge again, so every extension nests its parent at a
boundary, as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import torch

from neurotemplate.modeling.network import (
    AMBIENT, DTYPE, Fit, Network, Series, drive_matrix, filtered_states, rollout)

#: The forward selection's stopping rule: the best extension must beat its
#: parent at this level.
P_STOP = 0.05
#: Ljung–Box lags, in hours of innovations.
WHITENESS_HOURS = 12.0


@dataclass(frozen=True)
class Rung:
    mass: bool = False
    envelope: bool = False
    solar_mass: bool = False
    solar_envelope: bool = False
    direct: bool = False

    @property
    def name(self) -> str:
        return ("Ti" + "Tm" * self.mass + "Te" * self.envelope
                + "Am" * self.solar_mass + "Ae" * self.solar_envelope
                + "Ria" * self.direct)

    def extensions(self, solar: bool) -> list["Rung"]:
        """The one-step extensions; the solar ones only with a solar column."""
        out = []
        if not self.mass:
            out.append(replace(self, mass=True))
        if not self.envelope:
            out.append(replace(self, envelope=True))
        if solar and self.mass and not self.solar_mass:
            out.append(replace(self, solar_mass=True))
        if solar and self.envelope and not self.solar_envelope:
            out.append(replace(self, solar_envelope=True))
        if self.envelope and not self.direct:
            out.append(replace(self, direct=True))
        return out

    def network(self, drives: tuple[str, ...], solar: Optional[str], gains: tuple[str, ...],
                scale: dict[str, float]) -> Network:
        """The rung over its inputs: `drives` (the heats, first) and `gains`
        enter the zone, `solar` wherever the rung routes it; `scale` holds an
        input's gain or a parameter at a value."""
        nodes = ("Ti",) + ("Tm",) * self.mass + ("Te",) * self.envelope
        edges = []
        if self.mass:
            edges.append(("Ti", "Tm"))
        if self.envelope:
            edges += [("Ti", "Te"), ("Te", AMBIENT)]
        if not self.envelope or self.direct:
            edges.append(("Ti", AMBIENT))
        inputs = [(column, "Ti") for column in drives]
        if solar is not None:
            inputs.append((solar, "Ti"))
            if self.solar_mass:
                inputs.append((solar, "Tm"))
            if self.solar_envelope:
                inputs.append((solar, "Te"))
        inputs += [(column, "Ti") for column in gains]
        pinned = {name: float(value) for name, value in scale.items()
                  if not any(column == name for column, _ in inputs)}
        pinned.update({f"gain_{column}_{node}": float(scale[column])
                       for column, node in inputs if column in scale})
        return Network(nodes, tuple(edges), tuple(inputs), pinned=pinned, name=self.name)


def chi2_sf(x: float, df: int) -> float:
    """P(χ²(df) > x)."""
    if x <= 0:
        return 1.0
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64),
                                         torch.tensor(x / 2, dtype=torch.float64)))


def lr_p_value(parent_ll: float, parent_p: int, child_ll: float, child_p: int) -> float:
    """The likelihood-ratio test's p-value for a child nesting its parent."""
    return chi2_sf(2 * (child_ll - parent_ll), child_p - parent_p)


def ljung_box(residuals: list[torch.Tensor], lags: int) -> float:
    """Ljung–Box p-value over standardised innovations, series pooled by
    summing their statistics: small means what the network leaves out still
    has structure in time."""
    statistic, df = 0.0, 0
    for e in residuals:
        e = e - e.mean()
        n = e.shape[0]
        lags_here = min(lags, n - 1)
        denominator = float((e * e).sum())
        for k in range(1, lags_here + 1):
            rho = float((e[k:] * e[:-k]).sum()) / denominator
            statistic += n * (n + 2) * rho ** 2 / (n - k)
        df += lags_here
    return chi2_sf(statistic, df)


@dataclass
class Step:
    """One iteration of the selection: the candidates fitted from `parent`,
    the best one, and its test."""

    parent: str
    candidates: dict[str, float]      # name -> log-likelihood
    best: Optional[str]               # None when no candidate was admissible
    p_value: float
    #: Candidates left out because they add a mode the data cannot place in
    #: some zone (see `Fit.unplaced_modes`).
    unidentified: list[str]


@dataclass
class Selection:
    fits: dict[str, list[Fit]]        # rung name -> one fit per zone
    rungs: dict[str, Rung]
    parents: dict[str, str]           # rung name -> the rung it extended
    steps: list[Step]
    chosen: str

    def log_likelihood(self, name: str) -> float:
        return sum(f.log_likelihood for f in self.fits[name])

    def n_params(self, name: str) -> int:
        return sum(f.n_params for f in self.fits[name])

    def bic(self, name: str) -> float:
        return sum(f.bic for f in self.fits[name])

    def by_bic(self) -> str:
        return min(self.fits, key=self.bic)


def _identified(child: list[Fit], parent: list[Fit]) -> bool:
    """No zone gains a mode the data cannot place: the extension's new node
    is measured, not a near-free node whose likelihood gain is noise."""
    return all(len(c.unplaced_modes) <= len(p.unplaced_modes)
               for c, p in zip(child, parent))


def forward_select(fit_rung: Callable[[Rung, Optional[list[Fit]]], list[Fit]],
                   solar: bool, p_stop: float = P_STOP, bic: bool = False) -> Selection:
    """Forward selection from `Ti`. `fit_rung(rung, parent_fits)` fits one
    rung on every zone, seeded from its parent's fits, and returns one fit
    per zone; the likelihoods add over zones, so one zone and several
    sharing a rung run the same selection.

    An extension is admissible only if it adds no mode the data cannot place
    in any zone. With `bic`, the best one must also lower the summed BIC: a
    likelihood summed over many zones clears the ratio test for almost any
    extra parameter, which the penalty per parameter and observation does not.
    """
    start = Rung()
    fits = {start.name: fit_rung(start, None)}
    rungs, parents, steps = {start.name: start}, {}, []
    current = start

    def total(name):
        return sum(f.log_likelihood for f in fits[name])

    def summed_bic(name):
        return sum(f.bic for f in fits[name])

    while True:
        candidates = current.extensions(solar)
        if not candidates:
            break
        for rung in candidates:
            if rung.name not in fits:
                fits[rung.name] = fit_rung(rung, fits[current.name])
                rungs[rung.name] = rung
                parents[rung.name] = current.name
        admissible = [r for r in candidates
                      if _identified(fits[r.name], fits[current.name])]
        unidentified = [r.name for r in candidates if r not in admissible]
        scores = {r.name: total(r.name) for r in candidates}
        if not admissible:
            steps.append(Step(current.name, scores, None, 1.0, unidentified))
            break
        best = max(admissible, key=lambda r: total(r.name))
        p = lr_p_value(total(current.name), sum(f.n_params for f in fits[current.name]),
                       total(best.name), sum(f.n_params for f in fits[best.name]))
        steps.append(Step(current.name, scores, best.name, p, unidentified))
        if p > p_stop or (bic and summed_bic(best.name) >= summed_bic(current.name)):
            break
        current = best
    return Selection(fits, rungs, parents, steps, current.name)


def whiteness_lags(dt_h: float) -> int:
    return max(1, round(WHITENESS_HOURS / dt_h))


# ── Data ────────────────────────────────────────────────────────

def zone_series(frame, zone: str, ambient: str, columns: list[str], y_inputs: tuple[str, ...],
                every: int, dt_h: float) -> Series:
    """One file on a grid of `every` rows, the step from grid point K to
    K + 1 driven by the rows between them: the zone sampled at each point;
    ambient and every input column averaged over the step's rows (a rate's
    average keeps its energy). The timing is by role: a U or D row is held
    over the step that starts there, a Y row measured over the step that
    ends there, so `y_inputs`, the inputs that are Y of the model (a heat
    behind a thermostat), are averaged one row later. `dt_h` is the grid's
    step in hours."""
    steps = (len(frame) - 1) // every

    def averaged(column):
        start = 1 if column in y_inputs else 0
        values = frame[column].to_numpy(dtype=float)[start:start + steps * every]
        return torch.tensor(values, dtype=DTYPE).reshape(steps, every).mean(dim=1)

    zone_values = torch.tensor(frame[zone].to_numpy(dtype=float)[:steps * every:every],
                               dtype=DTYPE)
    inputs = (torch.stack([averaged(c) for c in columns], dim=1) if columns
              else torch.zeros(steps, 0, dtype=DTYPE))
    return Series(zone_values, averaged(ambient), inputs, dt_h)


def network_columns(network: Network) -> list[str]:
    """The input columns a network's drive reads, in its `inputs` order."""
    return [column for column, _ in network.inputs]


def rollout_rmse(network: Network, values: dict[str, float], series: Series,
                 horizon: int, burn_in: int, every: int) -> Optional[tuple[float, float]]:
    """`(first hour, whole horizon)` RMSE of open-loop rollouts `horizon`
    steps long, started every `every` steps after `burn_in` steps of
    filtering, each from the state the filter holds there; None when the
    series is too short for one."""
    states = filtered_states(network, values, series)
    drive = drive_matrix(series)
    hour = max(1, round(1 / series.dt_h))
    errors = []
    for start in range(burn_in, series.y.shape[0] - horizon, every):
        path = rollout(network, values, states[start], drive[start + 1:start + horizon],
                       series.dt_h)[:, 0]
        errors.append(path - series.y[start + 1:start + 1 + horizon])
    if not errors:
        return None
    errors = torch.stack(errors)
    return (float(errors[:, :hour].pow(2).mean().sqrt()), float(errors.pow(2).mean().sqrt()))
