"""
Physics checks: does the surrogate behave like a building when nobody is
driving it?

A test split tells you how closely the model tracks recorded data. It cannot
tell you whether the model has learned any physics, because the recorded data
never contains the situations that would show it: the HVAC fully off, the
weather stepping to a new value, an actuator held at one end of its range.
Those are cheap to synthesize and the answers are unambiguous.

Three scenarios, each an assertion rather than a score:

  Equilibrium  — everything off, ambient equal to the indoor state. The states
                 must stay put. A model that drifts here is integrating
                 something it should not.
  Step response — from equilibrium, apply one named actuation. The states must
                 move, and in the declared direction. Zero response means the
                 policy will have no lever; a wrong-signed response means it
                 will pull it backwards.
  Free decay   — from equilibrium, teleport the building to a new climate with
                 the HVAC off. The states must drift toward ambient and must
                 not sail past it.

Every check reports a number in physical units and a pass/fail, not a loss:
"closed 4% of an 8.3 °C gap in 24 steps" is a finding; "free_decay_loss:
0.0271" is not. Reject a surrogate on these before spending an hour training a
policy against it — a fit metric cannot make this call, and the record has an
architecture that fit *better* and controlled 2.7x worse.

Relationship to `modeling.probe`: the probe sweeps actuators over *recorded*
disturbances and reports how far they move each channel. These scenarios put
the plant in a state the data never visited and assert the sign and stability
of what happens. The probe measures authority you have; this says whether the
model is physically coherent at all.

The same rollouts also back a differentiable training penalty (`compute()`,
wired through `cfg.synth_gain`), but the assertions run whether or not the
penalty is switched on, and that is the intended way to use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from neurotemplate._typing import beartype


# Pass thresholds, all expressed as a share of a state channel's own recorded
# range so they carry across plants and units without retuning.
EQUILIBRIUM_DRIFT_FRAC = 0.02   # drift at rest that still counts as "at rest"
RESPONSE_MOVE_FRAC = 0.01       # movement below this is no response at all
DECAY_MIN_CLOSED = 0.02         # share of the ambient gap that must be closed
DECAY_MAX_CLOSED = 1.10         # more than this is overshoot, not relaxation


@dataclass(frozen=True)
class Actuation:
    """One named way of driving the plant, and where it should push the states.

    `values` maps control columns to physical values; controls left out stay at
    their `off` value, so a heating action does not have to restate the cooling
    setpoint. `direction` is +1 when the states should rise and -1 when they
    should fall — that sign is the whole assertion.
    """
    name: str
    values: dict[str, float]
    direction: int


@dataclass(frozen=True)
class SynthSpec:
    """What the physics checks need to know about this plant.

    Four fields, and only two of them are per-column work:

      states      Y channels that behave like temperatures — they hold when
                  nothing drives them and relax toward `ambient`. Naming these
                  is required: a power channel is not a temperature, and
                  treating one as though it were is how this subsystem used to
                  sample "equilibrium temperatures" out of a 0-61728 W range.
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
      actuations  optional; without any, the step-response check is skipped.

    Everything is in physical units; the library normalizes.
    """
    states: list[str]
    ambient: str
    off: dict[str, float]
    actuations: Sequence[Actuation] = field(default_factory=tuple)

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

        for act in self.actuations:
            bad = [c for c in act.values if c not in u]
            if bad:
                problems.append(f"actuation {act.name!r} names non-control "
                                f"columns: {bad}. U = {u}")
            if act.direction not in (-1, 1):
                problems.append(f"actuation {act.name!r} has direction "
                                f"{act.direction}; expected -1 or +1.")

        if problems:
            raise ValueError(
                "synth_cfg is not usable with this column mapping:\n  - "
                + "\n  - ".join(problems)
            )


@dataclass
class ScenarioResult:
    """One physics assertion, its verdict, and the number behind it."""
    name: str
    passed: bool
    headline: str      # one sentence, physical units, safe to print as-is
    channel: str       # the state channel the verdict was decided on
    value: float       # the measured quantity the check turned on
    tolerance: float   # the threshold it was compared against
    penalty: float     # the differentiable term's value, for the training path

    def to_dict(self) -> dict:
        return {"passed": self.passed, "headline": self.headline,
                "channel": self.channel, "value": self.value,
                "tolerance": self.tolerance, "penalty": self.penalty}


# ── Input construction ───────────────────────────────────────────


def _normalize(value, col: str, norm_stats: dict):
    stats = norm_stats[col]
    if stats["stretch"] == 0:
        raise ValueError(
            f"Column '{col}' has zero stretch in norm_stats — cannot "
            "normalize it (constant in the training data?)."
        )
    return (value - stats["shift"]) / stats["stretch"]


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
    """Physical range spanned by the state channels — never by all of Y."""
    return (min(norm_stats[c]["min"] for c in spec.states),
            max(norm_stats[c]["max"] for c in spec.states))


def _equilibrium_inputs(t_phys, nm_mapping, norm_stats, spec, device):
    """Normalized (y, u, d) rows for a plant sitting at `t_phys` with nothing on.

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
        return _normalize(
            torch.full((n,), _resting_value(col, spec, norm_stats),
                       device=device),
            col, norm_stats,
        )

    y = column(nm_mapping["Y"],
               lambda c: _normalize(t_phys, c, norm_stats)
               if c in spec.states else rest(c))
    u = column(nm_mapping["U"], rest)
    d = column(nm_mapping["D"],
               lambda c: _normalize(t_phys, c, norm_stats)
               if c == spec.ambient else rest(c))
    return y, u, d


def _actuated_u(u_off, nm_mapping, norm_stats, act: Actuation, device):
    """`u_off` with the actuation's columns overridden."""
    u = u_off.clone()
    n = u.shape[0]
    for col, value in act.values.items():
        j = nm_mapping["U"].index(col)
        u[:, j] = _normalize(
            torch.full((n,), float(value), device=device), col, norm_stats,
        )
    return u


def _rollout_data(yn_row, u_rows, d_rows, in_steps, n_steps, ny, device):
    """A SystemPreview input dict from per-phase rows.

    `u_rows` / `d_rows` are each (history_row, rollout_row); pass the same row
    twice for a signal that does not change at the rollout boundary. `Y` only
    sets the rollout length — the System never reads its values here.
    """
    n = yn_row.shape[0]
    total = (in_steps - 1) + n_steps

    def phased(rows):
        history, rollout = rows
        return torch.cat([
            history.unsqueeze(1).expand(-1, in_steps - 1, -1),
            rollout.unsqueeze(1).expand(-1, n_steps, -1),
        ], dim=1).contiguous()

    return {
        "yn": yn_row.unsqueeze(1).expand(-1, in_steps, -1).contiguous(),
        "Y": torch.zeros(n, total, ny, device=device),
        "U": phased(u_rows),
        "D": phased(d_rows),
    }


def _to_physical(y_norm, cols, norm_stats):
    """[n, steps, ny] normalized -> physical, per column."""
    phys = torch.zeros_like(y_norm)
    for i, col in enumerate(cols):
        s = norm_stats[col]
        phys[:, :, i] = y_norm[:, :, i] * s["stretch"] + s["shift"]
    return phys


# Kept under its historical name: modeling/pipeline.py unnormalizes test
# rollouts with it and has nothing to do with scenarios.
def _unnormalize_y(predictions, nm_mapping, norm_stats):
    """Convert normalized Y predictions [n, steps, ny] to physical units."""
    return _to_physical(predictions, nm_mapping["Y"], norm_stats)


# ── Scenarios ────────────────────────────────────────────────────


class _Scenario:
    """Shared plumbing: sample equilibria, roll out, read the state channels."""

    def __init__(self, system, nm_mapping, norm_stats, spec: SynthSpec,
                 in_steps, n_scenarios, n_steps, device):
        self.system = system
        self.nm_mapping = nm_mapping
        self.norm_stats = norm_stats
        self.spec = spec
        self.in_steps = in_steps
        self.n_scenarios = n_scenarios
        self.n_steps = n_steps
        self.device = device
        self.ny = len(nm_mapping["Y"])
        self.state_idx = [nm_mapping["Y"].index(c) for c in spec.states]
        self.t_min, self.t_max = _state_range(spec, norm_stats)

    def _sample_equilibrium(self, n, lo=None, hi=None):
        lo = self.t_min if lo is None else lo
        hi = self.t_max if hi is None else hi
        return torch.rand(n, device=self.device) * (hi - lo) + lo

    def _states_physical(self, y_pred):
        """[n, n_steps + 1, n_states] in physical units, starting at rest.

        Index 0 is the last seeded step (the equilibrium the rollout starts
        from), so `[:, k] - [:, 0]` is movement caused by the rollout.
        """
        phys = _to_physical(y_pred, self.nm_mapping["Y"], self.norm_stats)
        return phys[:, self.in_steps - 1:, self.state_idx]

    def _tolerance(self, frac: float) -> list[float]:
        """Per-state-channel threshold, `frac` of that channel's own range.

        Floored above zero so a channel that was constant in training — which
        the reportcard already warns about separately — gives a strict check
        rather than a division by zero.
        """
        return [max(frac * self._range(c), 1e-9) for c in self.spec.states]

    def _range(self, col: str) -> float:
        return self.norm_stats[col]["max"] - self.norm_stats[col]["min"]


class EquilibriumScenario(_Scenario):
    """Nothing on, ambient equal to indoors. The states must hold."""

    label = "Equilibrium"

    def _data(self, n):
        t = self._sample_equilibrium(n)
        y, u, d = _equilibrium_inputs(
            t, self.nm_mapping, self.norm_stats, self.spec, self.device,
        )
        return _rollout_data(y, (u, u), (d, d), self.in_steps, self.n_steps,
                             self.ny, self.device)

    def _drift(self, y_pred):
        """[n, n_states] mean |movement| away from rest, in physical units."""
        s = self._states_physical(y_pred)
        return (s[:, 1:, :] - s[:, 0:1, :]).abs().mean(dim=1)

    @staticmethod
    def _penalty(s):
        return (s[:, 1:, :] - s[:, 0:1, :]).pow(2).mean()

    def compute(self):
        y_pred = self.system(self._data(self.n_scenarios))["yn"]
        return self._penalty(self._states_physical(y_pred))

    def check(self, n: Optional[int] = None) -> list[ScenarioResult]:
        n = n or self.n_scenarios
        with torch.no_grad():
            y_pred = self.system(self._data(n))["yn"]
            drift = self._drift(y_pred).mean(dim=0)      # [n_states]
            # Same rollout as the verdict, so the two numbers agree.
            penalty = float(self._penalty(self._states_physical(y_pred)))

        tol = self._tolerance(EQUILIBRIUM_DRIFT_FRAC)
        worst = max(range(len(tol)), key=lambda i: float(drift[i]) / tol[i])
        value, limit = float(drift[worst]), tol[worst]
        channel = self.spec.states[worst]
        passed = value <= limit
        verdict = (
            f"held to {value:.3g} over {self.n_steps} steps"
            if passed else
            f"drifted {value:.3g} over {self.n_steps} steps with nothing "
            "running — either the model is integrating something it should "
            "not, or a control's `off` value is not off across the whole "
            "range equilibria are sampled from"
        )
        return [ScenarioResult(
            name=self.label, passed=passed,
            headline=f"`{channel}` {verdict} (limit {limit:.3g}).",
            channel=channel, value=value, tolerance=limit, penalty=penalty,
        )]

    def trace(self, n: int) -> dict:
        with torch.no_grad():
            data = self._data(n)
            y_pred = self.system(data)["yn"]
        return _trace(self.label, data, y_pred, self.nm_mapping,
                      self.norm_stats, self.spec, self.in_steps)


class StepResponseScenario(_Scenario):
    """From equilibrium, one named actuation. The states must move, correctly."""

    label = "Step response"

    def _data(self, act: Actuation, n):
        t = self._sample_equilibrium(n)
        y, u_off, d = _equilibrium_inputs(
            t, self.nm_mapping, self.norm_stats, self.spec, self.device,
        )
        u_on = _actuated_u(u_off, self.nm_mapping, self.norm_stats, act,
                           self.device)
        return _rollout_data(y, (u_off, u_on), (d, d), self.in_steps,
                             self.n_steps, self.ny, self.device)

    def _move(self, y_pred):
        """[n, n_states] signed movement from rest to the end of the rollout."""
        s = self._states_physical(y_pred)
        return s[:, -1, :] - s[:, 0, :]

    def compute(self):
        total = torch.zeros((), device=self.device)
        for act in self.spec.actuations:
            y_pred = self.system(self._data(act, self.n_scenarios))["yn"]
            s = self._states_physical(y_pred)
            deltas = s[:, 1:, :] - s[:, :-1, :]
            # Penalize only steps that move against the declared direction.
            total = total + torch.relu(-act.direction * deltas).mean()
        return total

    def check(self, n: Optional[int] = None) -> list[ScenarioResult]:
        n = n or self.n_scenarios
        results = []
        for act in self.spec.actuations:
            with torch.no_grad():
                y_pred = self.system(self._data(act, n))["yn"]
                move = self._move(y_pred).mean(dim=0)     # [n_states]
                s = self._states_physical(y_pred)
                deltas = s[:, 1:, :] - s[:, :-1, :]
                penalty = float(torch.relu(-act.direction * deltas).mean())

            tol = self._tolerance(RESPONSE_MOVE_FRAC)
            # Worst channel = the one furthest from doing what was asked.
            worst = min(range(len(tol)),
                        key=lambda i: act.direction * float(move[i]) / tol[i])
            value, limit = float(move[worst]), tol[worst]
            channel = self.spec.states[worst]
            signed = act.direction * value
            if signed >= limit:
                passed, verdict = True, (
                    f"moved {value:+.3g} over {self.n_steps} steps"
                )
            elif abs(value) < limit:
                passed, verdict = False, (
                    f"did not respond ({value:+.3g} over {self.n_steps} "
                    "steps) — the policy will have no lever here"
                )
            else:
                passed, verdict = False, (
                    f"moved {value:+.3g}, the wrong way — the policy will "
                    "pull this actuator backwards"
                )
            results.append(ScenarioResult(
                name=f"{self.label}: {act.name}", passed=passed,
                headline=f"`{channel}` {verdict} "
                         f"(wanted {'+' if act.direction > 0 else '-'}"
                         f"{limit:.3g} or more).",
                channel=channel, value=value, tolerance=limit,
                penalty=penalty,
            ))
        return results

    def trace(self, n: int) -> dict:
        if not self.spec.actuations:
            return {}
        act = self.spec.actuations[0]
        with torch.no_grad():
            data = self._data(act, n)
            y_pred = self.system(data)["yn"]
        return _trace(f"{self.label}: {act.name}", data, y_pred,
                      self.nm_mapping, self.norm_stats, self.spec,
                      self.in_steps)


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
        d_after[:, j] = _normalize(t_out, self.spec.ambient, self.norm_stats)

        data = _rollout_data(y, (u, u), (d_before, d_after), self.in_steps,
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

    @staticmethod
    def _penalty(s, t_out):
        """Movement away from ambient, plus anything outside [start, ambient]."""
        toward = torch.sign(t_out.unsqueeze(1) - s[:, 0, :]).unsqueeze(1)
        deltas = s[:, 1:, :] - s[:, :-1, :]
        wrong_way = torch.relu(-deltas * toward).mean()
        target = t_out[:, None, None].expand_as(s[:, 0:1, :])
        lo = torch.min(s[:, 0:1, :], target)
        hi = torch.max(s[:, 0:1, :], target)
        overshoot = (torch.relu(s[:, 1:, :] - hi).pow(2)
                     + torch.relu(lo - s[:, 1:, :]).pow(2)).mean()
        return wrong_way + overshoot

    def compute(self):
        data, _, t_out = self._data(self.n_scenarios)
        y_pred = self.system(data)["yn"]
        return self._penalty(self._states_physical(y_pred), t_out)

    def check(self, n: Optional[int] = None) -> list[ScenarioResult]:
        n = n or self.n_scenarios
        with torch.no_grad():
            data, t_in, t_out = self._data(n)
            y_pred = self.system(data)["yn"]
            closed = self._closed_fraction(y_pred, t_out).mean(dim=0)
            mean_gap = float((t_out - t_in).abs().mean())
            # Same rollout as the verdict, so the two numbers agree.
            penalty = float(self._penalty(self._states_physical(y_pred), t_out))

        worst = min(range(len(closed)), key=lambda i: float(closed[i]))
        value = float(closed[worst])
        channel = self.spec.states[worst]
        if value < DECAY_MIN_CLOSED:
            passed, verdict = False, (
                f"closed {value:.1%} of a {mean_gap:.3g} gap to ambient in "
                f"{self.n_steps} steps — this surrogate has no envelope "
                "coupling to speak of"
            )
        elif value > DECAY_MAX_CLOSED:
            passed, verdict = False, (
                f"overshot ambient by {value - 1:.1%} of a {mean_gap:.3g} "
                f"gap in {self.n_steps} steps"
            )
        else:
            passed, verdict = True, (
                f"closed {value:.1%} of a {mean_gap:.3g} gap to ambient in "
                f"{self.n_steps} steps"
            )
        return [ScenarioResult(
            name=self.label, passed=passed,
            headline=f"`{channel}` {verdict}.",
            channel=channel, value=value, tolerance=DECAY_MIN_CLOSED,
            penalty=penalty,
        )]

    def trace(self, n: int) -> dict:
        with torch.no_grad():
            data, _, _ = self._data(n)
            y_pred = self.system(data)["yn"]
        return _trace(self.label, data, y_pred, self.nm_mapping,
                      self.norm_stats, self.spec, self.in_steps)


def _trace(name, data, y_pred, nm_mapping, norm_stats, spec, in_steps) -> dict:
    """Physical-unit trajectories for the plots, one entry per scenario."""
    j = nm_mapping["D"].index(spec.ambient)
    a = norm_stats[spec.ambient]
    return {
        "name": name,
        "predictions": _to_physical(
            y_pred, nm_mapping["Y"], norm_stats).detach().cpu(),
        "ambient_traj": (
            data["D"][:, :, j] * a["stretch"] + a["shift"]).detach().cpu(),
        "channel_names": nm_mapping["Y"],
        "state_names": spec.states,
        "seed_steps": in_steps,
    }


# ── Entry points ─────────────────────────────────────────────────


@beartype
def build_scenarios(
    system,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict,
    synth_cfg: Optional[SynthSpec],
    in_steps: int,
    n_scenarios: int,
    n_steps: int,
    device,
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
        spec=synth_cfg, in_steps=in_steps, n_scenarios=n_scenarios,
        n_steps=n_steps, device=device,
    )
    return [
        EquilibriumScenario(**common),
        StepResponseScenario(**common),
        FreeDecayScenario(**common),
    ]


@beartype
def run_checks(scenarios: list, n: Optional[int] = None) -> list[ScenarioResult]:
    """Every scenario's verdict, flattened. Needs no `synth_gain`."""
    return [r for s in scenarios for r in s.check(n)]


@beartype
def checks_to_markdown(results: list[ScenarioResult]) -> str:
    rows = ["| Scenario | Verdict | What happened |", "|---|---|---|"]
    for r in results:
        rows.append(f"| {r.name} | {'pass' if r.passed else '**FAIL**'} | "
                    f"{r.headline} |")
    return "\n".join(rows)


@beartype
def checks_to_dict(results: list[ScenarioResult]) -> dict:
    return {r.name: r.to_dict() for r in results}


@beartype
def format_checks(results: list[ScenarioResult]) -> str:
    """Plain-text version for stdout."""
    lines = ["Physics checks (synthetic scenarios):"]
    for r in results:
        mark = "ok  " if r.passed else "FAIL"
        lines.append(f"  [{mark}] {r.name}: {r.headline}")
    return "\n".join(lines)
