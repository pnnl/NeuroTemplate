"""
Physics checks: does the surrogate behave like a building when nobody is
driving it?

A test split tells you how closely the model tracks recorded data. It cannot
tell you whether the model has learned any physics, because the recorded data
never contains the situations that would show it: the HVAC fully off, the
weather stepping to a new value, an actuator held at one end of its range.
Those are cheap to synthesize and the answers are unambiguous.

Four scenarios, each an assertion rather than a score:

  Equilibrium  — everything off, ambient equal to the indoor state. The states
                 must stay put. A model that drifts here is integrating
                 something it should not.
  Step response — from equilibrium, drive one actuator off its rest value, to
                 each end of its range in turn. Every state with a declared
                 sign must move the declared way, measured against the same
                 seeds left at rest so the model's own drift cancels. Zero
                 response means the policy will have no lever; a wrong-signed
                 response means it will pull it backwards. Coverage is
                 generated from `SynthSpec.responds`, so no actuator goes
                 untested by omission.
  Reachability — from equilibrium, push every actuator that moves a state, and
                 every disturbance with a declared sign, the same way. The span
                 between the two directions is what a comfort band has to fit
                 inside.
  Free decay   — from equilibrium, teleport the building to a new climate with
                 the HVAC off. The states must drift toward ambient and must
                 not sail past it.

Every check reports a number in physical units and a pass/fail, not a loss:
"closed 4% of an 8.3 °C gap in 24 steps" is a finding; "free_decay_loss:
0.0271" is not. Reject a surrogate on these before spending an hour training a
policy against it — a fit metric cannot make this call, and the record has an
architecture that fit *better* and controlled 2.7x worse.

Reachability answers what an earlier `modeling.probe` module tried to: how far
the actuators can drive each channel. It does so from a declared equilibrium
with declared push directions, rather than by holding co-actuators at the
arithmetic midpoint of their range over whatever weather one test file
happened to contain.

These are deliberately extrapolations. Holding ambient equal to a 24 °C indoor
state puts a surrogate trained on a 9-18 °C spring outside its data, and that
is the point — a model that only behaves inside its training distribution will
be asked to leave it the moment a controller starts steering. Read a failure
as "this is where it stops making sense", not as a scoring error.

These are assertions only. An earlier version also exposed each scenario as a
differentiable training penalty behind a `synth_gain` weight; no campaign ever
switched it on, and it forced every scenario to stay differentiable for a path
nobody used. Physicality is still enforced during training — by the gradient
sign penalty in `gradient_sampling`, which is the mechanism suited to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from neurotemplate._typing import beartype
from neurotemplate.modeling.contract import Normalized, contract_model


# Pass thresholds, all expressed as a share of a state channel's own recorded
# range so they carry across plants and units without retuning.
EQUILIBRIUM_DRIFT_FRAC = 0.02   # drift at rest that still counts as "at rest"
RESPONSE_MOVE_FRAC = 0.01       # movement below this is no response at all
DECAY_MIN_CLOSED = 0.02         # share of the ambient gap that must be closed
DECAY_MAX_CLOSED = 1.10         # more than this is overshoot, not relaxation


#: A control whose `off` value sits this close to a range end (as a share of
#: its range) has nothing to sweep on that side.
_SAME_VALUE_FRAC = 1e-6


@dataclass(frozen=True)
class SynthSpec:
    """What the physics checks need to know about this plant.

    Four fields, and only two of them are per-column work:

      states      Y channels that behave like temperatures — they hold when
                  nothing drives them and relax toward `ambient`. Naming these
                  is required: a power channel is not a temperature, and
                  treating one as though it were is how this subsystem used to
                  sample "equilibrium temperatures" out of a 0-61728 W range.

                  Only these channels are asserted on. Equilibria are sampled
                  from the window every one of them has been observed at (the
                  intersection of their ranges), so naming a channel with a far
                  wider span — supply air against a zone, say — narrows nothing
                  but does subject that channel to the assertions. Leave it out
                  to have it seeded at rest and reported without a verdict.
                  Zones held at different temperatures may share no recorded
                  temperature at all; they are then seeded in the gap between
                  their ranges (see `_state_range`).
      ambient     the D column the states relax toward.
      off         physical value of each column when nothing is running.
                  **Required for every control column** and only for those:
                  an actuator's "off" is not guessable (a cooling setpoint is
                  off at its *maximum*, not its minimum). Any other column left
                  out is held at its recorded minimum, which is what "no
                  forcing" means for solar, occupancy, power and flow — list it
                  in `off` if your plant idles somewhere else.

                  For a *setpoint* actuator, "off" has to be off across the
                  whole `states` range, since equilibria are sampled from all
                  of it. A cooling setpoint of 26 °C is a call for cooling on
                  a zone that reaches 28, and the equilibrium check will
                  rightly report the resulting drift. Put such a setpoint
                  outside the states' range.
      responds    `{state: {control: +1 or -1}}` — the sign of that state's
                  movement as that control *increases*. Required for every
                  control, and the step response and reachability checks are
                  generated from it, so an actuator cannot be silently left
                  untested and one that pushes two zones opposite ways can be
                  said so.

                  This is the same shape as the gradient-mapping JSON but a
                  different claim, and the two are deliberately not shared.
                  `grad_path` declares partials that hold at *every* point in
                  state space, which is why its U half is usually empty: a
                  cooling setpoint above the current zone temperature commands
                  nothing, so ∂zone/∂setpoint is 0 there and positive only
                  where cooling is active — piecewise, not a universal sign.
                  What is declared here is a *transition*: start at rest, move
                  this control off its `off` value, and the scenario has built
                  the state where the actuator is live. That has a definite
                  answer where the universal partial does not.

    Everything is in physical units, as the model reads them.
    """
    states: list[str]
    ambient: str
    off: dict[str, float]
    responds: dict[str, dict[str, int]] = field(default_factory=dict)

    def validate(self, nm_mapping: dict[str, list[str]]) -> None:
        """Fail with a list of exactly what to fix, before anything runs."""
        y, u, d = nm_mapping["Y"], nm_mapping["U"], nm_mapping["D"]
        problems = []

        if not self.states:
            problems.append("`states` is empty; name the Y channels that "
                            "behave like temperatures.")
        unknown = [c for c in self.states if c not in y]
        if unknown:
            problems.append(f"`states` names columns that are not in Y: "
                            f"{unknown}. Y = {y}")
        if self.ambient not in d:
            problems.append(f"`ambient` ({self.ambient!r}) is not a D column. "
                            f"D = {d}")

        missing = [c for c in u if c not in self.off]
        if missing:
            problems.append(f"`off` must give a value for every control "
                            f"column; missing: {missing}")
        known = set(y) | set(u) | set(d)
        stray = [c for c in self.off if c not in known]
        if stray:
            problems.append(f"`off` names columns that are not in the "
                            f"mapping: {stray}")

        covered = set()
        for state, edges in self.responds.items():
            if state not in self.states:
                problems.append(f"`responds` keys must be state channels; "
                                f"{state!r} is not in states={self.states}")
            for col, sign in edges.items():
                if col not in u:
                    problems.append(f"`responds[{state!r}]` names {col!r}, "
                                    f"which is not a control. U = {u}")
                else:
                    covered.add(col)
                if sign not in (-1, 1):
                    problems.append(
                        f"`responds[{state!r}][{col!r}]` is {sign}; expected "
                        "-1 or +1 — the sign of the state's movement as the "
                        "control increases."
                    )

        untested = [c for c in u if c not in covered]
        if untested and not problems:
            problems.append(
                f"`responds` declares no state that these controls move: "
                f"{untested}. Every control needs at least one entry, or its "
                "step response and reachability go untested — which is the "
                "gap this field exists to close. Give the sign of each state's "
                "movement as the control increases (a cooling setpoint that "
                "warms the zone as it rises is +1)."
            )

        if problems:
            raise ValueError(
                "synth_cfg is not usable with this column mapping:\n  - "
                + "\n  - ".join(problems)
            )


#: Order statistics reported for each readout channel.
READOUT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class ReadoutSummary:
    """What a non-state Y channel did during one scenario. No verdict.

    Readouts are seeded at their resting value and then left alone: the model's
    own predictions feed back at every step, so a readout is free to go
    anywhere and nothing asserts otherwise. That freedom is not harmless —
    those predictions are inputs to the next state prediction, so a readout
    emitting 4000 W at equilibrium with the compressor off can drag the zone
    with it, and the drift would look unattributable.

    Reporting the spread costs a slice and a quantile on a rollout that already
    ran, and it separates `0/0/0/0/0` from `0/1200/3000/4200/5800` — which is
    the whole question a reader has. What a readout *should* do is
    scenario-dependent (hold at off during equilibrium, move under a step
    response), so the library takes no position and prints the numbers.
    """
    scenario: str
    name: str
    seeded_at: float        # the resting value it started from
    quantiles: list[float]  # at READOUT_QUANTILES, in physical units

    @property
    def key(self) -> str:
        return f"{self.scenario} | {self.name}"

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario, "channel": self.name,
            "seeded_at": self.seeded_at,
            **{f"q{int(q * 100)}": v
               for q, v in zip(READOUT_QUANTILES, self.quantiles)},
        }


@dataclass
class ScenarioResult:
    """One physics assertion on one state channel, and the number behind it.

    Every scenario reports one of these per channel in `SynthSpec.states`. An
    earlier version collapsed each scenario to its single worst channel, which
    made naming a second state destructive: one wide-ranging channel failing
    took the channel you cared about out of the report entirely. Per channel,
    the classification is a soft label — it decides what a channel is *seeded*
    with, not whether you get to see it.
    """
    name: str          # the scenario, e.g. "Equilibrium"
    passed: bool
    headline: str      # one sentence, physical units, safe to print as-is
    channel: str       # the state channel this verdict is about
    value: float       # the measured quantity the check turned on
    tolerance: float   # the threshold it was compared against

    @property
    def key(self) -> str:
        """Stable identifier for one (scenario, channel) verdict."""
        return f"{self.name} | {self.channel}"

    def to_dict(self) -> dict:
        return {"passed": self.passed, "headline": self.headline,
                "scenario": self.name, "channel": self.channel,
                "value": self.value, "tolerance": self.tolerance}


# ── Input construction ───────────────────────────────────────────


def _resting_value(col: str, spec: SynthSpec, norm_stats: dict) -> float:
    """`off` if the spec gives one, else the column's recorded minimum.

    The single fallback in this module, and it is deliberate: "no forcing"
    really is the minimum for irradiance, occupancy, power, flow and stage
    signals, which is most of what is left once the controls are covered.
    Controls are never defaulted — `SynthSpec.validate` demands them.
    """
    if col in spec.off:
        return float(spec.off[col])
    return float(norm_stats[col]["min"])


def _state_range(spec: SynthSpec, norm_stats: dict) -> tuple[float, float]:
    """Temperatures to seed equilibria at: the ones nearest every state's data.

    `highest min .. lowest max`, never all of Y. When the state channels'
    recorded ranges overlap, that is their *intersection*, not their union.
    At equilibrium every state holds the same temperature and ambient holds it
    too, so the honest place to sample one is where all of them have data. The
    union is actively wrong once a second state is named: on a real RTU whose
    zone spans 18.9-27.3 °C and whose supply air spans 8.6-54.5 °C, it would
    assert the building holds steady at 51 °C indoors *and* out — 28 °C past
    anything the ambient channel ever recorded.

    When the ranges do not overlap — conditioned zones held at different
    temperatures (20.4-21.4 °C beside 21.8-23.6 °C) never share one — the
    same two numbers bound the gap between them, the temperatures closest to
    every state's data. A building at rest does reach a common temperature;
    the data just never shows it, so each state seeded outside its own range
    says by how much in its Equilibrium headline (`_outside_range`).

    With a single state channel this is its own range.
    """
    highest_min = max(norm_stats[c]["min"] for c in spec.states)
    lowest_max = min(norm_stats[c]["max"] for c in spec.states)
    return min(highest_min, lowest_max), max(highest_min, lowest_max)


def _equilibrium_inputs(t_phys, nm_mapping, norm_stats, spec, device):
    """(y, u, d) rows for a plant sitting at `t_phys` with nothing on.

    State channels take `t_phys`; every other column takes its resting value.
    Ambient takes `t_phys` too, so there is no gradient across the envelope —
    that is what makes it an equilibrium rather than a slow decay.
    """
    n = t_phys.shape[0]

    def column(cols, per_col):
        out = torch.zeros(n, len(cols), device=device)
        for i, col in enumerate(cols):
            out[:, i] = per_col(col)
        return out

    def rest(col):
        return torch.full((n,), _resting_value(col, spec, norm_stats),
                       device=device)

    y = column(nm_mapping["Y"],
               lambda c: t_phys
               if c in spec.states else rest(c))
    u = column(nm_mapping["U"], rest)
    d = column(nm_mapping["D"],
               lambda c: t_phys
               if c == spec.ambient else rest(c))
    return y, u, d


def _rollout_data(yn_row, u_rows, d_rows, history_steps, n_steps, ny, device):
    """A SystemPreview input dict from per-phase rows.

    `u_rows` / `d_rows` are each (history_row, rollout_row); pass the same row
    twice for a signal that does not change at the rollout boundary. `Y` only
    sets the rollout length — the System never reads its values here.
    """
    n = yn_row.shape[0]
    total = (history_steps - 1) + n_steps

    def phased(rows):
        history, rollout = rows
        return torch.cat([
            history.unsqueeze(1).expand(-1, history_steps - 1, -1),
            rollout.unsqueeze(1).expand(-1, n_steps, -1),
        ], dim=1).contiguous()

    return {
        "yn": yn_row.unsqueeze(1).expand(-1, history_steps, -1).contiguous(),
        "Y": torch.zeros(n, total, ny, device=device),
        "U": phased(u_rows),
        "D": phased(d_rows),
    }


def _declared_demands(system) -> dict[str, dict]:
    """`{U column: {"kind", "state", "sign", ...}}` for every `demand` the
    surrogate declares, or {}.

    Read off the model's own `PreStack` rather than the block spec, so the
    checks and the trained run cannot disagree about what is a thermostat.
    Only a built-in family declares demands.
    """
    model = contract_model(system)
    pre_stack = model.core.pre_stack if isinstance(model, Normalized) else None
    if pre_stack is None:
        return {}
    return {col: entry for col, entry in pre_stack.to_config().items()
            if entry["kind"] == "demand"}


# ── Scenarios ────────────────────────────────────────────────────


@dataclass
class ScenarioOutcome:
    """Everything one scenario measured: verdicts on states, stats on readouts.

    Both come off the same rollouts. Drawing a second sample for the readouts
    would let the printed spread describe a different set of scenarios than the
    verdict beside it — the shape of a bug this module has already had once.
    """
    verdicts: list[ScenarioResult]
    readouts: list[ReadoutSummary]

    def __add__(self, other: "ScenarioOutcome") -> "ScenarioOutcome":
        return ScenarioOutcome(self.verdicts + other.verdicts,
                               self.readouts + other.readouts)


class _Scenario:
    """Shared plumbing: sample equilibria, roll out, read the state channels."""

    def __init__(self, system, nm_mapping, norm_stats, spec: SynthSpec,
                 history_steps, n_scenarios, n_steps, device, grad_masks=None):
        self.system = system
        self.nm_mapping = nm_mapping
        self.norm_stats = norm_stats
        self.spec = spec
        # Only the reachability scenario reads these, for the direction to push
        # each disturbance. Empty/None means disturbances stay at rest.
        self.grad_masks = grad_masks or {}
        self.history_steps = history_steps
        self.n_scenarios = n_scenarios
        self.n_steps = n_steps
        self.device = device
        self.ny = len(nm_mapping["Y"])
        self.state_idx = [nm_mapping["Y"].index(c) for c in spec.states]
        self.t_min, self.t_max = _state_range(spec, norm_stats)
        self.demands = _declared_demands(system)

    def _outside_range(self, col: str) -> float:
        """How far the seed range reaches past `col`'s recorded range; 0 when
        the states overlap."""
        stats = self.norm_stats[col]
        return max(0.0, self.t_max - stats["max"], stats["min"] - self.t_min)

    def _sample_equilibrium(self, n, lo=None, hi=None):
        lo = self.t_min if lo is None else lo
        hi = self.t_max if hi is None else hi
        return torch.rand(n, device=self.device) * (hi - lo) + lo

    def _states_physical(self, y_pred):
        """[n, n_steps + 1, n_states] in physical units, starting at rest.

        Index 0 is the last seeded step (the equilibrium the rollout starts
        from), so `[:, k] - [:, 0]` is movement caused by the rollout.
        """
        return y_pred[:, self.history_steps - 1:, self.state_idx]

    def check(self, n: Optional[int] = None) -> list[ScenarioResult]:
        """Just the verdicts. `evaluate` also returns the readout statistics."""
        return self.evaluate(n).verdicts

    def _readouts(self, y_pred, scenario: str) -> list[ReadoutSummary]:
        """Order statistics per non-state Y channel, from an existing rollout.

        Measured over the free-running steps only: the seeded prefix is the
        channel's own `off` value by construction, and including it would plant
        a known constant in the min or max. It is reported separately as
        `seeded_at` so the spread has a reference point.
        """
        free = y_pred[:, self.history_steps:, :]
        qs = torch.tensor(READOUT_QUANTILES, device=free.device)
        return [
            ReadoutSummary(
                scenario=scenario,
                name=name,
                seeded_at=_resting_value(name, self.spec, self.norm_stats),
                quantiles=torch.quantile(
                    free[:, :, i].reshape(-1), qs
                ).tolist(),
            )
            for i, name in enumerate(self.nm_mapping["Y"])
            if name not in self.spec.states
        ]

    def _tolerance(self, frac: float) -> list[float]:
        """Per-state-channel threshold, `frac` of that channel's own range.

        Floored above zero so a channel that was constant in training — which
        the reportcard already warns about separately — gives a strict check
        rather than a division by zero.
        """
        return [max(frac * self._range(c), 1e-9) for c in self.spec.states]

    def _range(self, col: str) -> float:
        return self.norm_stats[col]["max"] - self.norm_stats[col]["min"]

    def _sweeps(self) -> list[tuple[str, float, int]]:
        """Every (control, target, direction) worth driving from rest.

        A control is swept to each end of its recorded range that differs from
        its `off` value, so a cooling setpoint whose off *is* its maximum gets
        one sweep downward rather than a no-op upward one. `direction` is +1
        when the target is above `off` and -1 when below, which is what turns
        `responds` (a sign against increasing the control) into the movement
        this particular sweep should produce.
        """
        sweeps = []
        for col in self.nm_mapping["U"]:
            stats = self.norm_stats[col]
            off = _resting_value(col, self.spec, self.norm_stats)
            span = max(stats["max"] - stats["min"], 1e-12)
            for target in (stats["min"], stats["max"]):
                if abs(target - off) / span <= _SAME_VALUE_FRAC:
                    continue
                sweeps.append((col, float(target), 1 if target > off else -1))
        return sweeps


class EquilibriumScenario(_Scenario):
    """Nothing on, ambient equal to indoors. The states must hold."""

    label = "Equilibrium"

    def _data(self, n):
        t = self._sample_equilibrium(n)
        y, u, d = _equilibrium_inputs(
            t, self.nm_mapping, self.norm_stats, self.spec, self.device,
        )
        return _rollout_data(y, (u, u), (d, d), self.history_steps, self.n_steps,
                             self.ny, self.device)

    def _drift(self, y_pred):
        """[n, n_states] mean |movement| away from rest, in physical units."""
        s = self._states_physical(y_pred)
        return (s[:, 1:, :] - s[:, 0:1, :]).abs().mean(dim=1)

    def evaluate(self, n: Optional[int] = None) -> ScenarioOutcome:
        n = n or self.n_scenarios
        with torch.no_grad():
            y_pred = self.system(self._data(n))["yn"]
            drift = self._drift(y_pred).mean(dim=0)      # [n_states]
            readouts = self._readouts(y_pred, self.label)

        tol = self._tolerance(EQUILIBRIUM_DRIFT_FRAC)
        results = []
        for i, channel in enumerate(self.spec.states):
            value, limit = float(drift[i]), tol[i]
            passed = value <= limit
            verdict = (
                f"Held to {value:.3g} over {self.n_steps} steps"
                if passed else
                f"Drifted {value:.3g} over {self.n_steps} steps with nothing "
                "running — either the model is integrating something it "
                "should not, or a control's `off` value is not off across the "
                "whole range equilibria are sampled from"
            )
            outside = self._outside_range(channel)
            seeded = (
                f" Seeded up to {outside:.3g} outside this channel's recorded "
                "range: the states share no recorded temperature."
                if outside > 0 else ""
            )
            results.append(ScenarioResult(
                name=self.label, passed=passed,
                headline=f"{verdict} (limit {limit:.3g}).{seeded}",
                channel=channel, value=value, tolerance=limit,
            ))
        return ScenarioOutcome(results, readouts)

    def trace(self, n: int) -> dict:
        with torch.no_grad():
            data = self._data(n)
            y_pred = self.system(data)["yn"]
        return _trace(self.label, data, y_pred, self.nm_mapping,
                      self.spec, self.history_steps)


class StepResponseScenario(_Scenario):
    """From equilibrium, one actuator off its rest value. States must respond.

    Every control is swept to each end of its range that differs from `off`,
    so coverage is generated rather than hand-written: an actuator cannot be
    left untested by omission, and both directions of a mid-range control are
    exercised. Each state with a declared sign in `responds` gets its own
    verdict, so an actuator that warms one zone while cooling another is
    expressible — a single per-actuation direction could not say that.

    The response is the step's *effect*: the move with the step minus the move
    of the same seeds left at rest. Measured from rest alone, a model's own
    drift decides the verdict — the same −0.23 °C drift once failed a heating
    sweep as "the wrong way" and passed a cooling sweep beside it.

    A control declared `demand` is a thermostat: the clamp makes it command
    nothing while the zone is on the inactive side of the target. Its sweep
    is seeded only where the target is active (a cooling target below the
    zone, a heating target above it), and a target inactive across the whole
    seed range asserts nothing and is skipped — its answer is zero by
    construction. The pairing is read off the model, so a `demand` whose
    zone is not one of `states` is swept unrestricted.
    """

    label = "Step response"

    def _seed_range(self, col: str, target: float) -> Optional[tuple[float, float]]:
        """Where to seed `col`'s sweep to `target`, or None to skip it."""
        demand = self.demands.get(col)
        if demand is None or demand["state"] not in self.spec.states:
            return self.t_min, self.t_max
        if demand["sign"] < 0:      # cooling: active with the zone above
            lo, hi = max(self.t_min, target), self.t_max
        else:                       # heating: active with the zone below
            lo, hi = self.t_min, min(self.t_max, target)
        return (lo, hi) if lo < hi else None

    def _data(self, t, col: str, target: Optional[float]):
        """Seeds `t`; `col` stepped to `target` after the history, or held at
        rest throughout when `target` is None."""
        n = t.shape[0]
        y, u_off, d = _equilibrium_inputs(
            t, self.nm_mapping, self.norm_stats, self.spec, self.device,
        )
        u_on = u_off.clone()
        if target is not None:
            j = self.nm_mapping["U"].index(col)
            u_on[:, j] = torch.full((n,), target, device=self.device)
        return _rollout_data(y, (u_off, u_on), (d, d), self.history_steps,
                             self.n_steps, self.ny, self.device)

    def _move(self, y_pred):
        """[n, n_states] signed movement from rest to the end of the rollout."""
        s = self._states_physical(y_pred)
        return s[:, -1, :] - s[:, 0, :]

    def evaluate(self, n: Optional[int] = None) -> ScenarioOutcome:
        n = n or self.n_scenarios
        results, readouts = [], []
        for col, target, moved in self._sweeps():
            seeds = self._seed_range(col, target)
            if seeds is None:
                continue
            label = f"{self.label}: {col} to {target:.4g}"
            with torch.no_grad():
                t = self._sample_equilibrium(n, *seeds)
                y_pred = self.system(self._data(t, col, target))["yn"]
                y_rest = self.system(self._data(t, col, None))["yn"]
                move = (self._move(y_pred) - self._move(y_rest)).mean(dim=0)
                readouts += self._readouts(y_pred, label)

            tol = self._tolerance(RESPONSE_MOVE_FRAC)
            for i, channel in enumerate(self.spec.states):
                # `responds` is signed against the control *increasing*; this
                # sweep may be driving it the other way.
                sign = self.spec.responds.get(channel, {}).get(col)
                if sign is None:
                    continue      # no declared relationship, nothing to assert
                want = sign * moved
                value, limit = float(move[i]), tol[i]
                if want * value >= limit:
                    passed, verdict = True, (
                        f"Moved {value:+.3g} over {self.n_steps} steps, net of "
                        "the drift at rest"
                    )
                elif abs(value) < limit:
                    passed, verdict = False, (
                        f"Did not respond ({value:+.3g} over {self.n_steps} "
                        "steps, net of the drift at rest) — the policy will "
                        "have no lever here"
                    )
                else:
                    passed, verdict = False, (
                        f"Moved {value:+.3g} net of the drift at rest, the "
                        "wrong way — the policy will pull this actuator "
                        "backwards"
                    )
                results.append(ScenarioResult(
                    name=label, passed=passed,
                    headline=f"{verdict} "
                             f"(wanted {'+' if want > 0 else '-'}"
                             f"{limit:.3g} or more).",
                    channel=channel, value=value, tolerance=limit,
                ))
        return ScenarioOutcome(results, readouts)

    def trace(self, n: int) -> dict:
        for col, target, _ in self._sweeps():
            seeds = self._seed_range(col, target)
            if seeds is None:
                continue
            with torch.no_grad():
                data = self._data(self._sample_equilibrium(n, *seeds), col, target)
                y_pred = self.system(data)["yn"]
            return _trace(f"{self.label}: {col} to {target:.4g}", data, y_pred,
                          self.nm_mapping, self.spec, self.history_steps)
        return {}


class ReachabilityScenario(_Scenario):
    """From equilibrium, everything pushed the same way. How far can it get?

    For each state and each direction, every control that moves that state is
    driven to whichever end of its range pushes it further, and every exogenous
    input with a declared gradient sign is driven the same way. The span
    between the two directions is the state's reachable envelope under this
    surrogate — the number a comfort band has to fit inside.

    Step response and reachability are deliberately different questions. The
    step response *isolates* one actuator, holding ambient equal to indoors so
    there is no envelope gradient to confound it. Reachability deliberately
    *conflates* actuators and weather, because it is asking for a bound rather
    than an attribution.

    Exogenous directions come from the gradient mapping (`cfg.grad_path`),
    whose D half is a genuine universal partial: more outdoor heat warms the
    zone anywhere in the state space. Without a mapping the disturbances stay
    at their equilibrium values and the span reported is actuator-only, which
    the headline says.
    """

    label = "Reachability"

    #: A state whose two directions land closer together than this share of its
    #: recorded range is not controllable on this surrogate.
    MIN_SPAN_FRAC = 0.02

    def _push_targets(self, channel: str, direction: int):
        """(control targets, disturbance targets) that drive `channel` further.

        `direction` is +1 to push the state up and -1 to push it down. A sign
        of +1 means the state rises with the input, so the input goes to its
        maximum when pushing up and its minimum when pushing down.
        """
        def target(col: str, sign: int) -> float:
            s = self.norm_stats[col]
            return s["max"] if sign * direction > 0 else s["min"]

        controls = {
            col: target(col, sign)
            for col, sign in self.spec.responds.get(channel, {}).items()
        }
        disturbances = {
            col: target(col, sign)
            for col, sign in self._ambient_signs(channel).items()
        }
        return controls, disturbances

    def _ambient_signs(self, channel: str) -> dict[str, int]:
        """Declared d(state)/d(disturbance) signs, from the gradient mapping."""
        mask = self.grad_masks.get("D") if self.grad_masks else None
        if mask is None:
            return {}
        i = self.nm_mapping["Y"].index(channel)
        return {
            col: int(mask[0, 0, i, j])
            for j, col in enumerate(self.nm_mapping["D"])
            if int(mask[0, 0, i, j]) != 0
        }

    def _data(self, channel: str, direction: int, n, t=None):
        # Both directions must start from the *same* equilibrium sample, or the
        # span between them picks up the sampling spread — an inert surrogate
        # would report a reachable range it does not have.
        t = self._sample_equilibrium(n) if t is None else t
        y, u_off, d_rest = _equilibrium_inputs(
            t, self.nm_mapping, self.norm_stats, self.spec, self.device,
        )
        controls, disturbances = self._push_targets(channel, direction)

        def driven(row, key, targets):
            out = row.clone()
            for col, value in targets.items():
                j = self.nm_mapping[key].index(col)
                out[:, j] = torch.full((n,), float(value), device=self.device)
            return out

        return _rollout_data(
            y,
            (u_off, driven(u_off, "U", controls)),
            (d_rest, driven(d_rest, "D", disturbances)),
            self.history_steps, self.n_steps, self.ny, self.device,
        )

    def _settled(self, channel: str, direction: int, n, t) -> tuple[float, object]:
        """Mean final value of `channel` when pushed, plus the raw rollout."""
        i = self.spec.states.index(channel)
        y_pred = self.system(self._data(channel, direction, n, t))["yn"]
        return float(self._states_physical(y_pred)[:, -1, i].mean()), y_pred

    def evaluate(self, n: Optional[int] = None) -> ScenarioOutcome:
        n = n or self.n_scenarios
        results, readouts = [], []
        for channel in self.spec.states:
            with torch.no_grad():
                t = self._sample_equilibrium(n)
                hi, y_up = self._settled(channel, +1, n, t)
                lo, _ = self._settled(channel, -1, n, t)
            readouts += self._readouts(y_up, f"{self.label} (up)")

            span = hi - lo
            limit = max(self.MIN_SPAN_FRAC * self._range(channel), 1e-9)
            weather = "" if self._ambient_signs(channel) else (
                " Disturbances held at rest — no gradient mapping, so this is "
                "the actuators alone."
            )
            ends = (f"pushed down it settles at {lo:.4g}, pushed up "
                    f"{hi:.4g}, over {self.n_steps} steps")
            if span >= limit:
                passed, verdict = True, (
                    f"Reaches {lo:.4g} to {hi:.4g} — a span of {span:.4g} "
                    f"over {self.n_steps} steps"
                )
            elif span < 0:
                # Not a narrow envelope: the two directions came out crossed,
                # so the surrogate moves against every declared sign at once.
                passed, verdict = False, (
                    f"Responds backwards — {ends}, so driving everything "
                    f"toward a higher value lands {-span:.4g} lower. The "
                    "declared signs in `responds` and this surrogate disagree"
                )
            else:
                passed, verdict = False, (
                    f"Reaches only {lo:.4g} to {hi:.4g} — a span of "
                    f"{span:.4g} with everything pushed both ways. No policy "
                    "can hold a band this surrogate cannot reach"
                )
            results.append(ScenarioResult(
                name=self.label, passed=passed,
                headline=f"{verdict}.{weather}",
                channel=channel, value=span, tolerance=limit,
            ))
        return ScenarioOutcome(results, readouts)

    def trace(self, n: int) -> dict:
        channel = self.spec.states[0]
        with torch.no_grad():
            data = self._data(channel, +1, n)
            y_pred = self.system(data)["yn"]
        return _trace(f"{self.label}: {channel} (up)", data, y_pred,
                      self.nm_mapping, self.spec, self.history_steps)


class FreeDecayScenario(_Scenario):
    """From equilibrium, a new climate with the HVAC off. The states must relax."""

    label = "Free decay"

    #: Smallest indoor-to-ambient gap worth asserting anything about, as a
    #: share of the states' range — the same scale-free convention as the
    #: pass thresholds, so it carries to a plant measured in anything.
    MIN_GAP_FRAC = 0.1

    def _data(self, n):
        t_in = self._sample_equilibrium(n)
        a = self.norm_stats[self.spec.ambient]
        t_out = torch.rand(n, device=self.device) * (a["max"] - a["min"]) + a["min"]
        # Push tiny gaps out so the assertion has something to bite on; the
        # direction of the (possibly zero) gap is preserved.
        min_gap = self.MIN_GAP_FRAC * (self.t_max - self.t_min)
        gap = t_out - t_in
        sign = torch.where(gap >= 0, 1.0, -1.0)
        t_out = torch.where(gap.abs() < min_gap, t_in + min_gap * sign, t_out)

        y, u, d_before = _equilibrium_inputs(
            t_in, self.nm_mapping, self.norm_stats, self.spec, self.device,
        )
        d_after = d_before.clone()
        j = self.nm_mapping["D"].index(self.spec.ambient)
        d_after[:, j] = t_out

        data = _rollout_data(y, (u, u), (d_before, d_after), self.history_steps,
                             self.n_steps, self.ny, self.device)
        return data, t_in, t_out

    def _closed_fraction(self, y_pred, t_out):
        """[n, n_states] share of the ambient gap closed by the rollout's end.

        1.0 means the state reached ambient, 0 means it never moved, negative
        means it moved away, and above 1 means it sailed past.
        """
        s = self._states_physical(y_pred)
        start = s[:, 0, :]
        gap = t_out.unsqueeze(1) - start
        return (s[:, -1, :] - start) / gap

    def evaluate(self, n: Optional[int] = None) -> ScenarioOutcome:
        n = n or self.n_scenarios
        with torch.no_grad():
            data, t_in, t_out = self._data(n)
            y_pred = self.system(data)["yn"]
            closed = self._closed_fraction(y_pred, t_out).mean(dim=0)
            mean_gap = float((t_out - t_in).abs().mean())
            readouts = self._readouts(y_pred, self.label)

        results = []
        for i, channel in enumerate(self.spec.states):
            value = float(closed[i])
            if value < DECAY_MIN_CLOSED:
                passed, verdict = False, (
                    f"Closed {value:.1%} of a {mean_gap:.3g} gap to ambient in "
                    f"{self.n_steps} steps — this surrogate has no envelope "
                    "coupling to speak of"
                )
            elif value > DECAY_MAX_CLOSED:
                passed, verdict = False, (
                    f"Overshot ambient by {value - 1:.1%} of a {mean_gap:.3g} "
                    f"gap in {self.n_steps} steps"
                )
            else:
                passed, verdict = True, (
                    f"Closed {value:.1%} of a {mean_gap:.3g} gap to ambient in "
                    f"{self.n_steps} steps"
                )
            results.append(ScenarioResult(
                name=self.label, passed=passed, headline=f"{verdict}.",
                channel=channel, value=value, tolerance=DECAY_MIN_CLOSED,
            ))
        return ScenarioOutcome(results, readouts)

    def trace(self, n: int) -> dict:
        with torch.no_grad():
            data, _, _ = self._data(n)
            y_pred = self.system(data)["yn"]
        return _trace(self.label, data, y_pred, self.nm_mapping,
                      self.spec, self.history_steps)


def _trace(name, data, y_pred, nm_mapping, spec, history_steps) -> dict:
    """Physical-unit trajectories for the plots, one entry per scenario."""
    j = nm_mapping["D"].index(spec.ambient)
    return {
        "name": name,
        "predictions": y_pred.detach().cpu(),
        "ambient_traj": data["D"][:, :, j].detach().cpu(),
        "channel_names": nm_mapping["Y"],
        "state_names": spec.states,
        "seed_steps": history_steps,
    }


# ── Entry points ─────────────────────────────────────────────────


@beartype
def build_scenarios(
    system,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict,
    synth_cfg: Optional[SynthSpec],
    history_steps: int,
    n_scenarios: int,
    n_steps: int,
    device,
    grad_masks: Optional[dict] = None,
) -> Optional[list]:
    """The scenario objects for this run, or None when the subsystem is off.

    `synth_cfg=None` (the `modeling_user_defined.synth_cfg` hook's opt-out) is
    the single switch; every caller checks for None rather than re-deriving
    whether the subsystem applies. The spec is validated here, so a mapping it
    cannot describe fails at construction with the list of what to fix.
    """
    if synth_cfg is None:
        return None
    synth_cfg.validate(nm_mapping)

    common = dict(
        system=system, nm_mapping=nm_mapping, norm_stats=norm_stats,
        spec=synth_cfg, history_steps=history_steps, n_scenarios=n_scenarios,
        n_steps=n_steps, device=device, grad_masks=grad_masks,
    )
    return [
        EquilibriumScenario(**common),
        StepResponseScenario(**common),
        ReachabilityScenario(**common),
        FreeDecayScenario(**common),
    ]


@beartype
def run_scenarios(scenarios: list, n: Optional[int] = None) -> ScenarioOutcome:
    """Every scenario's verdicts and readout statistics, flattened."""
    total = ScenarioOutcome([], [])
    for s in scenarios:
        total = total + s.evaluate(n)
    return total


@beartype
def checks_to_markdown(results: list[ScenarioResult]) -> str:
    rows = ["| Scenario | Channel | Verdict | What happened |",
            "|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r.name} | `{r.channel}` | "
                    f"{'pass' if r.passed else '**FAIL**'} | {r.headline} |")
    return "\n".join(rows)


@beartype
def checks_to_dict(results: list[ScenarioResult]) -> dict:
    """Keyed by scenario *and* channel, since each scenario reports on every
    state channel."""
    return {r.key: r.to_dict() for r in results}


@beartype
def readouts_to_markdown(readouts: list[ReadoutSummary]) -> str:
    """Order statistics per readout channel, per scenario. No verdicts."""
    rows = ["| Scenario | Channel | Seeded at | Min | 25% | Median | 75% | Max |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in readouts:
        cells = " | ".join(f"{v:.4g}" for v in r.quantiles)
        rows.append(f"| {r.scenario} | `{r.name}` | {r.seeded_at:.4g} | "
                    f"{cells} |")
    return "\n".join(rows)


@beartype
def readouts_to_dict(readouts: list[ReadoutSummary]) -> dict:
    return {r.key: r.to_dict() for r in readouts}


@beartype
def format_checks(results: list[ScenarioResult]) -> str:
    """Plain-text version for stdout."""
    lines = ["Physics checks (synthetic scenarios):"]
    for r in results:
        mark = "ok  " if r.passed else "FAIL"
        lines.append(f"  [{mark}] {r.key}: {r.headline}")
    return "\n".join(lines)
