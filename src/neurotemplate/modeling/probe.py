"""
Counterfactual actuator probe: what does the surrogate think the controls do?

Fit metrics answer "does the rollout track the recorded data". They cannot
answer "does this surrogate believe the actuators do anything", and the two
come apart in practice: a better-fitting surrogate whose cooling input barely
moved the zone produced 2.7x the closed-loop comfort violation of a
worse-fitting one that had a real cooling response. Nothing in RMSE saw it,
and the cost of finding out was an hour of policy training per candidate.

The probe is one sweep of cheap rollouts. Hold D at its recorded values,
replace the whole U trajectory with a constant, roll out, and read the settled
Y. Two things fall out of the same rollouts:

  authority — how far each control input alone moves each Y channel across its
              full actuation range. Near zero means the policy has no lever on
              that channel, whatever the objective asks for.
  envelope  — the span of settled values each Y channel can be driven to. A
              reference band outside that span is not achievable by any policy,
              and a violation rate scored against it measures the weather.

Everything here is a counterfactual on a frozen model, so it is seconds of
work and it runs before a controller is ever trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurotemplate._typing import beartype


# An actuator whose full swing moves a channel by less than this share of the
# channel's own observed range is not a lever the policy can pull. 2% is well
# below any authority worth controlling with and well above rollout noise.
DEAD_AUTHORITY_FRAC = 0.02

# The probe's numbers are means over windows, so a few dozen windows is
# plenty; the cap keeps its cost flat on long test splits.
MAX_PROBE_WINDOWS = 32


@dataclass
class ActuatorAuthority:
    """How far one control input moves one Y channel, in physical units."""
    y_name: str
    u_name: str
    delta: float        # settled y at u_max minus settled y at u_min
    y_range: float      # the channel's observed physical range, for scale
    u_lo: float         # actuation range swept, in physical units
    u_hi: float

    @property
    def relative(self) -> float:
        """|delta| as a share of the channel's own range. 0 means no lever."""
        return abs(self.delta) / self.y_range if self.y_range > 0 else 0.0

    def is_dead(self, frac: float = DEAD_AUTHORITY_FRAC) -> bool:
        return self.relative < frac


@dataclass
class ChannelEnvelope:
    """The span of settled values a Y channel can be driven to."""
    name: str
    lo: float        # lowest settled mean across the sweep
    hi: float        # highest settled mean across the sweep
    data_lo: float   # the channel's recorded minimum, for context
    data_hi: float

    def covers(self, band_lo: float, band_hi: float) -> bool:
        """Whether [band_lo, band_hi] overlaps what the plant can hold."""
        return not (band_hi < self.lo or band_lo > self.hi)


@dataclass
class AuthorityReport:
    authority: list[ActuatorAuthority]
    envelope: list[ChannelEnvelope]
    settle_steps: int
    n_windows: int

    def dead_actuators(self, frac: float = DEAD_AUTHORITY_FRAC) -> list[str]:
        """U columns with no authority over *any* Y channel."""
        u_names = []
        for a in self.authority:
            if a.u_name not in u_names:
                u_names.append(a.u_name)
        return [
            u for u in u_names
            if all(a.is_dead(frac) for a in self.authority if a.u_name == u)
        ]

    def envelope_for(self, name: str) -> ChannelEnvelope:
        for e in self.envelope:
            if e.name == name:
                return e
        raise KeyError(f"No envelope for channel {name!r}")


@beartype
def u_bounds_from_norm_stats(
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-U (umin, umax) in normalized space, from the observed data range.

    The same default `control_user_defined.actuation_bounds` uses, so a
    sys-id-side probe sweeps the box the policy will later be squashed into.
    """
    u_cols = nm_mapping["U"]

    def norm(col: str, key: str) -> float:
        s = norm_stats[col]
        return (s[key] - s["shift"]) / s["stretch"]

    return (torch.tensor([norm(c, "min") for c in u_cols]),
            torch.tensor([norm(c, "max") for c in u_cols]))


def _subsample_windows(data: dict, max_windows: int) -> dict:
    """Evenly spaced subset of the batch axis, or `data` when already small."""
    n = next(iter(data.values())).shape[0]
    if n <= max_windows:
        return data
    device = next(iter(data.values())).device
    idx = torch.linspace(0, n - 1, max_windows, device=device).round().long()
    return {k: v.index_select(0, idx) for k, v in data.items()}


@beartype
def probe_authority(
    dyn_sys,
    data: dict,
    in_steps: int,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    umin: torch.Tensor,
    umax: torch.Tensor,
    *,
    settle_steps: int = -1,
    max_windows: int = MAX_PROBE_WINDOWS,
) -> AuthorityReport:
    """Sweep the actuation box on a frozen surrogate; report authority + envelope.

    Args:
        dyn_sys:  the rollout System (`get_dynamics` / `load_trained_dynamics`).
        data:     a sys-id window dict — keys `yn`, `U`, `D`, `t` — as built by
                  `get_sw_dict_multi` (drop `source_idx` first).
        in_steps: the surrogate's history window, so the transient before the
                  rollout starts is excluded.
        umin/umax: actuation box in NORMALIZED space, one entry per U column.
        settle_steps: rollout steps discarded as transient. -1 (the default)
                  drops the first half of the rollout.
        max_windows: cap on probe windows; see MAX_PROBE_WINDOWS.

    The sweep is one-at-a-time: every control is held at its midpoint except
    the one being probed, which is held at each end of its range. That is
    2*nu + 1 rollouts, and it reads authority off a single actuator at a time
    the way an operator would test one.
    """
    u_cols, y_cols = nm_mapping["U"], nm_mapping["Y"]
    if not u_cols:
        raise ValueError("Cannot probe authority: nm_mapping['U'] is empty.")
    if umin.shape != (len(u_cols),) or umax.shape != (len(u_cols),):
        raise ValueError(
            f"umin/umax must have one entry per U column ({len(u_cols)}); got "
            f"{tuple(umin.shape)} and {tuple(umax.shape)}"
        )

    data = _subsample_windows(data, max_windows)
    device = data["U"].device
    umin, umax = umin.to(device), umax.to(device)
    umid = 0.5 * (umin + umax)

    pred_steps = data["U"].shape[1] - in_steps
    if pred_steps < 2:
        raise ValueError(
            f"Probe windows are {data['U'].shape[1]} steps but the surrogate "
            f"consumes {in_steps} as history; need at least 2 rollout steps."
        )
    if settle_steps < 0:
        settle_steps = pred_steps // 2

    shift = torch.tensor([norm_stats[c]["shift"] for c in y_cols], device=device)
    stretch = torch.tensor([norm_stats[c]["stretch"] for c in y_cols],
                           device=device)

    def settled(u_const: torch.Tensor) -> torch.Tensor:
        """[ny] mean settled physical value per Y channel, U held constant."""
        d = {k: v.clone() for k, v in data.items()}
        d["U"][:, :, :] = u_const
        with torch.no_grad():
            out = dyn_sys(d)
        # System output prepends the initial condition, hence the trailing -1.
        yn = out["yn"][:, in_steps + settle_steps:-1, :]
        return (yn * stretch + shift).mean(dim=(0, 1))

    sweep = [settled(umid)]
    authority: list[ActuatorAuthority] = []
    for j, u_name in enumerate(u_cols):
        u_lo, u_hi = umid.clone(), umid.clone()
        u_lo[j], u_hi[j] = umin[j], umax[j]
        y_at_lo, y_at_hi = settled(u_lo), settled(u_hi)
        sweep += [y_at_lo, y_at_hi]

        u_stats = norm_stats[u_name]
        for i, y_name in enumerate(y_cols):
            y_stats = norm_stats[y_name]
            authority.append(ActuatorAuthority(
                y_name=y_name,
                u_name=u_name,
                delta=float(y_at_hi[i] - y_at_lo[i]),
                y_range=y_stats["max"] - y_stats["min"],
                u_lo=float(umin[j]) * u_stats["stretch"] + u_stats["shift"],
                u_hi=float(umax[j]) * u_stats["stretch"] + u_stats["shift"],
            ))

    stacked = torch.stack(sweep)  # [n_sweep_points, ny]
    lows, highs = stacked.min(dim=0).values, stacked.max(dim=0).values
    envelope = [
        ChannelEnvelope(
            name=y_name,
            lo=float(lows[i]), hi=float(highs[i]),
            data_lo=norm_stats[y_name]["min"], data_hi=norm_stats[y_name]["max"],
        )
        for i, y_name in enumerate(y_cols)
    ]

    return AuthorityReport(
        authority=authority,
        envelope=envelope,
        settle_steps=settle_steps,
        n_windows=data["U"].shape[0],
    )


# ── Rendering ────────────────────────────────────────────────────


@beartype
def authority_to_markdown(report: AuthorityReport) -> str:
    """Authority table, one row per (Y channel, control input) pair."""
    rows = [
        "| Channel | Control | Swept range | Δ channel | Δ / channel range | |",
        "|---|---|---|---:|---:|---|",
    ]
    for a in report.authority:
        flag = "⚠️ no authority" if a.is_dead() else ""
        rows.append(
            f"| {a.y_name} | {a.u_name} | {a.u_lo:.4g} → {a.u_hi:.4g} | "
            f"{a.delta:+.4g} | {a.relative:.1%} | {flag} |"
        )
    return "\n".join(rows)


@beartype
def envelope_to_markdown(report: AuthorityReport) -> str:
    """Achievable-envelope table, one row per Y channel."""
    rows = [
        "| Channel | Achievable lo | Achievable hi | Recorded lo | Recorded hi |",
        "|---|---:|---:|---:|---:|",
    ]
    for e in report.envelope:
        rows.append(
            f"| {e.name} | {e.lo:.4g} | {e.hi:.4g} | "
            f"{e.data_lo:.4g} | {e.data_hi:.4g} |"
        )
    return "\n".join(rows)


@beartype
def envelope_vs_band_to_markdown(
    report: AuthorityReport,
    controlled: list[str],
    band_lo: float,
    band_hi: float,
) -> str:
    """Per controlled channel: what the plant can hold vs. what is being asked.

    "Match the band to the envelope before tuning anything" is finding #1 of
    every campaign in the record, and the library will otherwise happily
    accept a band whose ceiling the plant cannot reach and report a violation
    rate that is really a measurement of the weather.
    """
    rows = [
        "| Channel | Achievable lo | Achievable hi | Band lo | Band hi | |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in controlled:
        e = report.envelope_for(name)
        if not e.covers(band_lo, band_hi):
            flag = "⚠️ band is entirely unreachable"
        elif band_lo < e.lo or band_hi > e.hi:
            flag = "⚠️ band extends past the envelope"
        else:
            flag = ""
        rows.append(
            f"| {name} | {e.lo:.4g} | {e.hi:.4g} | "
            f"{band_lo:.4g} | {band_hi:.4g} | {flag} |"
        )
    return "\n".join(rows)


@beartype
def authority_to_dict(report: AuthorityReport) -> dict:
    """JSON-serializable probe result (for MLflow / disk)."""
    return {
        "settle_steps": report.settle_steps,
        "n_windows": report.n_windows,
        "authority": [
            {
                "channel": a.y_name, "control": a.u_name,
                "delta": a.delta, "relative": a.relative,
                "u_lo": a.u_lo, "u_hi": a.u_hi, "dead": a.is_dead(),
            }
            for a in report.authority
        ],
        "envelope": {
            e.name: {"lo": e.lo, "hi": e.hi,
                     "data_lo": e.data_lo, "data_hi": e.data_hi}
            for e in report.envelope
        },
    }
