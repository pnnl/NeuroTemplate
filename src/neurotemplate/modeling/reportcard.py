"""
Per-run reportcard: the verdicts on one trained model, short enough to read.

A markdown summary dropped in each run's artifact directory and logged to
MLflow. It reports exceptions, not inventories: a healthy channel is counted,
not listed, and each check is explained once rather than on every row. The
full output of every instrument is in the run's `results/` folder, one JSON
file each, for tools and for looking a channel up.

Sections (in order):

  Verdict       — one line per instrument: accuracy, whiteness, physics,
                  rollout, staged channels, and any data warning
  Model         — family, size, history; identical blocks collapsed to one line
  Data          — split sizes, role counts, head kinds, normalization
  Accuracy      — rollout RMSE against persistence, and skill; the worst
                  channels when there are many
  Whiteness     — one-step-ahead residuals (`whiteness`), grouped by the
                  period that holds the most residual power
  Staged channels — balanced accuracy against persistence, decision thresholds
  Physics       — per check, how many channels pass, and the failures
  Rollout       — channels that ramp or drift well past the truth; omitted
                  when none do
  Seed ensemble — the spread across `cfg.ensemble_seeds` models, when > 1
  Hyperparameters — one line

Single entry point: render_reportcard(...) returns the markdown string.
The caller (pipeline.py) is responsible for writing it to disk and
logging it to MLflow.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from neurotemplate.common.data_utils import degenerate_mapped_columns
from neurotemplate.modeling.bounded import NEAR_EDGE, edge
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.calibration import Calibration
from neurotemplate.modeling.heads import BoundedHead, HeadStack, OrdinalHead
from neurotemplate.modeling.whiteness import PERIOD_BANDS, Whiteness
from neurotemplate.common.metrics import (
    ChannelMetrics,
    LevelMetrics,
    RolloutStability,
    skill,
)
from neurotemplate.modeling.scenarios import (
    DECAY_MIN_CLOSED,
    FreeDecayScenario,
    ScenarioResult,
)
from neurotemplate.common.reportcard import RESULTS_DIR, _names, _section

from neurotemplate._typing import beartype

#: A table with more rows than this shows only its worst rows.
MAX_ROWS = 12
#: How many worst rows a capped table shows.
WORST_ROWS = 8
#: A rollout drifting more than this multiple of the truth's drift is flagged.
DRIFT_RATIO_WARN = 2.0


# ── Seed ensemble spread ──────────────────────────────────────────
#
# Every number elsewhere on the reportcard comes from one trained model —
# one draw from whatever the seed happened to give. `pipeline.train_and_
# evaluate` optionally trains `cfg.ensemble_seeds` models (see that field's
# comment in ModelingConfig) and hands the per-seed results here to be
# turned into the spread these types describe. `spread_stats` is also used
# directly by cli/optuna.py to report the spread of a trial's per-seed dev
# losses — the same order-statistics question, asked of a different metric.


@dataclass(frozen=True)
class SeedSpread:
    """min/median/max of one metric across an ensemble of seeds.

    Order statistics, not mean +/- std: a handful of seeds is nowhere near
    enough to trust a Gaussian assumption, and printing a stray min or max
    is honest about that instead of implying more precision than a handful
    of draws support.
    """
    min: float
    median: float
    max: float


def spread_stats(values: list[float]) -> SeedSpread:
    """Build a SeedSpread from one metric's per-seed values."""
    return SeedSpread(min=min(values), median=statistics.median(values), max=max(values))


@dataclass(frozen=True)
class EnsembleSpread:
    """Spread across `cfg.ensemble_seeds` independently-trained models.

    Built by pipeline.train_and_evaluate from the same TestResults and
    describe_dynamics structures a single-seed run already produces — see
    ModelingConfig.ensemble_seeds. Each dict is keyed by whatever
    distinguishes one of its rows from another; empty dicts render nothing.
    """
    n_seeds: int
    rmse: dict[str, SeedSpread]        # channel -> spread
    physics: dict[str, SeedSpread]     # "scenario | channel" -> spread
    tau_slow: dict[str, SeedSpread]    # zone -> spread of its slowest time constant (rc_network)
    # channel -> spread, levelled (binary / ordinal) channels only. Rare-level
    # heads vary far more across seeds than their RMSE lets on.
    balanced_acc: dict[str, SeedSpread] = field(default_factory=dict)
    # "scenario | channel" -> how many seeds passed the check. The loud
    # signal: a verdict that flips between seeds is not a property of the
    # model family, whatever the values' spread looks like.
    physics_passed: dict[str, int] = field(default_factory=dict)


def _spread_table(rows: list[tuple[str, SeedSpread]]) -> list[str]:
    """A 4-column min/median/max markdown table, one row per named spread."""
    lines = ["| Channel | Min | Median | Max |", "|---|---:|---:|---:|"]
    for name, s in rows:
        lines.append(f"| `{name}` | {s.min:.4g} | {s.median:.4g} | {s.max:.4g} |")
    return lines


def _widest(spreads: dict[str, SeedSpread]) -> list[tuple[str, SeedSpread]]:
    """The spreads widest relative to their median, at most MAX_ROWS."""
    def width(s: SeedSpread) -> float:
        return (s.max - s.min) / abs(s.median) if s.median else float("inf")
    return sorted(spreads.items(), key=lambda kv: -width(kv[1]))[:MAX_ROWS]


def _ensemble_section(ensemble: Optional[EnsembleSpread]) -> Optional[str]:
    """Min/median/max across the seed ensemble; None without one."""
    if ensemble is None:
        return None
    body = [f"Across **{ensemble.n_seeds}** seeds (`cfg.seed` .. `cfg.seed + "
            f"{ensemble.n_seeds - 1}`). Every other number on this card is the "
            "first seed's; this is the error bar around them.\n"]
    for title, spreads in (("Test RMSE", ensemble.rmse),
                           ("`tau_slow` (hours)", ensemble.tau_slow),
                           ("Balanced accuracy", ensemble.balanced_acc)):
        if spreads:
            shown = _widest(spreads)
            more = (f" (the {len(shown)} widest of {len(spreads)})"
                    if len(shown) < len(spreads) else "")
            body += [f"**{title}**{more}:\n", *_spread_table(shown), ""]
    flips = {key: n for key, n in ensemble.physics_passed.items()
             if 0 < n < ensemble.n_seeds}
    if ensemble.physics:
        body.append("**Physics verdicts that change with the seed:** "
                    + (", ".join(f"`{key}` ({n}/{ensemble.n_seeds})"
                                 for key, n in sorted(flips.items()))
                       if flips else "none") + ".")
    return _section("Seed ensemble spread", "\n".join(body))


# ── Header and verdict ────────────────────────────────────────────


def _header(name: str, timestamp: str, trained: bool,
            best_dev_loss: Optional[float], train_time_s: Optional[float],
            test_time_s: Optional[float]) -> str:
    parts = [timestamp]
    if not trained:
        parts.append("not trained (`lr: 0`): the model as built")
    if best_dev_loss is not None:
        parts.append(f"dev loss {best_dev_loss:.4g}")
    if train_time_s is not None:
        parts.append(f"trained {train_time_s:.0f} s")
    if test_time_s is not None:
        parts.append(f"test rollout {test_time_s:.2f} s")
    return f"# Reportcard: {name}\n\n" + " · ".join(parts) + "\n"


def ranked_skills(metrics: list[ChannelMetrics],
            persistence: list[ChannelMetrics]) -> list[tuple[ChannelMetrics, float, float]]:
    """(metrics, persistence RMSE, skill) per channel, worst skill first;
    a channel persistence already predicts exactly (NaN skill) goes last."""
    reference = {m.name: m.rmse for m in persistence}
    rows = [(m, reference[m.name], skill(m.rmse, reference[m.name])) for m in metrics]
    return sorted(rows, key=lambda r: (math.isnan(r[2]), r[2]))


def _verdict_section(
    rows: list[tuple[ChannelMetrics, float, float]],
    white: list[Whiteness],
    physics: Optional[list[ScenarioResult]],
    flagged: list[RolloutStability],
    test_window: int,
    levels: list[LevelMetrics],
    persistence_levels: list[LevelMetrics],
    warnings: list[str],
) -> str:
    lines = []
    scored = [r for r in rows if not math.isnan(r[2])]
    if scored:
        beats = sum(r[2] > 0 for r in scored)
        median = statistics.median(r[2] for r in scored)
        worst = ", ".join(f"`{m.name}` {s:.2f}" for m, _, s in scored[:3])
        lines.append(f"- **Accuracy:** beats persistence on {beats} of {len(scored)} "
                     f"channels, median skill {median:.2f}; worst {worst}.")
    if white:
        n_white = sum(w.white for w in white)
        lines.append(f"- **One-step residuals:** {n_white} of {len(white)} continuous "
                     "channels white" + ("." if n_white == len(white)
                                         else "; the rest carry structure (Whiteness)."))
    if physics is not None:
        by_check = _group_checks(physics)
        counts = ", ".join(f"{check} {sum(r.passed for r in rs)}/{len(rs)}"
                           for check, rs in by_check.items())
        passed = sum(r.passed for r in physics)
        lines.append(f"- **Physics:** {passed} of {len(physics)} checks pass ({counts}).")
    lines.append(
        f"- **Rollout:** {len(flagged)} channel(s) ramp or drift past "
        f"{DRIFT_RATIO_WARN:g}× the truth over {test_window} steps (Rollout)."
        if flagged else
        f"- **Rollout:** no channel ramps or drifts past {DRIFT_RATIO_WARN:g}× "
        f"the truth over {test_window} steps.")
    if levels:
        reference = {lm.name: lm.balanced_accuracy for lm in persistence_levels}
        worst = min(levels, key=lambda lm: lm.balanced_accuracy)
        lines.append(f"- **Staged channels:** worst balanced accuracy `{worst.name}` "
                     f"{worst.balanced_accuracy:.2f} (persistence "
                     f"{reference.get(worst.name, float('nan')):.2f}).")
    lines += [f"- ⚠️ {w}" for w in warnings]
    lines.append(f"\nEvery instrument's full output is in `{RESULTS_DIR}/` beside this card.")
    return _section("Verdict", "\n".join(lines))


# ── Model ─────────────────────────────────────────────────────────


def _lag_list(lags: dict[str, int]) -> str:
    return ", ".join(f"{column} {lag}" for column, lag in lags.items())


def _lags_by_value(lags: dict[str, int]) -> str:
    """Columns grouped by their lag, longest first: `lag 5: a, b, … (+n)`."""
    groups: dict[int, list[str]] = {}
    for column, lag in lags.items():
        groups.setdefault(lag, []).append(column)
    return "; ".join(f"lag {lag}: {_names(cols)}"
                     for lag, cols in sorted(groups.items(), reverse=True))


def _block_parts(info: dict) -> list[str]:
    parts = [f"`{info['class']}`", f"{info['param_count']:,} params"]
    if info.get("hsizes"):
        parts.append(f"hsizes {info['hsizes']}")
    for key in ("hidden_size", "num_layers"):
        if key in info:
            parts.append(f"{key} {info[key]}")
    return parts


def _network_lines(info: dict, indent: str) -> list[str]:
    """An rc_network block's time constants against the ladder's, and the
    share of its stability bound each thermostat gain uses."""
    if "thermostat" in info:
        shares = ", ".join(f"{side} {share:.0%}" for side, share in info["gain_share"].items())
        return [f"{indent}- thermostat {info['thermostat']}: gain at {shares} of "
                "the bound past which the loop rings"]
    if "network_time_constants_h" not in info:
        return []
    taus = ", ".join(f"{t:.3g}" for t in info["network_time_constants_h"])
    line = f"{indent}- network {info['network']}: time constants {taus} h"
    coarse = info.get("coarse_time_constants_h")
    if coarse:
        line += "; the ladder's " + ", ".join(
            f"{t:.3g} ± {'?' if se is None else f'{se:.2g}'}" for t, se in coarse) + " h"
    lines = [line]
    if info.get("left_coarse_bar"):
        lines.append(f"{indent}- ⚠️ outside three of the ladder's error bars: "
                     f"{info['left_coarse_bar']} h — training moved a time constant "
                     "the likelihood had placed.")
    return lines


def _edge_lines(bounded: dict, indent: str) -> list[str]:
    """The bounded parameters the box stopped, one line per parameter and
    edge: the box, not the data, set those values."""
    lines = []
    for name, b in bounded.items():
        if b["pinned"]:
            continue
        for side in ("lower", "upper"):
            hits = [f"{b['value'][label]:.3g}" if len(b["value"]) == 1
                    else f"{label} {b['value'][label]:.3g}"
                    for label, fraction in b["fraction"].items() if edge(fraction) == side]
            if hits:
                lines.append(f"{indent}  - {name} [{b['low']:g}, {b['high']:g}], {side}: "
                             + _names(hits, quote=False))
    if not lines:
        return []
    return [f"{indent}- ⚠️ At a box edge (within {NEAR_EDGE:.0%} of it), so the box, "
            "not the data, set these; every value is in "
            f"`{RESULTS_DIR}/architecture.json`:"] + lines


def _forcing_lines(forcing: Optional[dict], indent: str) -> list[str]:
    """Which column feeds which branch of a forcing block. A demand routed
    into the mlp voids the block's stability guarantee, so it is called out."""
    if not forcing:
        return []
    branches = []
    if forcing.get("ambient"):
        branches.append(f"ambient {forcing['ambient']['column']}")
    branches += [f"linear {c} (k={t['k']:.3g})" for c, t in forcing.get("linear", {}).items()]
    branches += [f"{'latched' if t['latched'] else 'saturating'} {c} "
                 f"(w={t['w']:.3g}, slope_share={t['slope_share']:.2f})"
                 for c, t in forcing.get("saturating", {}).items()]
    if forcing.get("mlp"):
        branches.append(f"mlp {', '.join(forcing['mlp'])}")
    lines = [f"{indent}- forcing: " + "; ".join(branches)]
    if forcing.get("demand_in_mlp"):
        lines.append(f"{indent}- ⚠️ demand routed into the mlp: "
                     f"{', '.join(forcing['demand_in_mlp'])} — the stability "
                     "guarantee does not hold for this block")
    return lines


def _own_lines(info: dict, indent: str) -> list[str]:
    """What a single block reports about itself beyond its size."""
    return (_network_lines(info, indent)
            + _edge_lines(info.get("bounded", {}), indent)
            + _forcing_lines(info.get("forcing"), indent))


def _block_signature(info: dict) -> tuple:
    """Blocks with the same signature print as one line."""
    return (info.get("model"), info["class"], info["param_count"], str(info.get("hsizes")),
            info.get("hidden_size"), info.get("num_layers"),
            tuple(info.get("lags", {}).values()))


def _stack_lines(info: dict) -> list[str]:
    """One line per group of identical plain blocks; a block with something
    of its own to say (time constants, a parameter at an edge, forcing)
    gets its own line."""
    groups: dict[tuple, list[str]] = {}
    lines = []
    for name, sub in info["blocks"].items():
        if sub is None:
            continue
        own = _own_lines(sub, "  ")
        if own:
            lines.append(f"- block `{name}` ({sub.get('model')}): " + ", ".join(_block_parts(sub)))
            lines += own
        else:
            groups.setdefault(_block_signature(sub), []).append(name)
    for names in groups.values():
        first = info["blocks"][names[0]]
        label = (f"block `{names[0]}`" if len(names) == 1
                 else f"{len(names)} blocks {_names(names, limit=3)}")
        each = "" if len(names) == 1 else " each"
        line = f"- {label} ({first.get('model')}): " + ", ".join(_block_parts(first)) + each
        if "lags" in first:
            line += (f"; reads {_lag_list(first['lags'])}" if len(names) == 1
                     else f"; `{names[0]}` reads {_lag_list(first['lags'])}")
        lines.append(line)
    return lines


def _architecture_section(arch: dict) -> str:
    flags = [f for f, on in (("separable", arch["separable"]),
                             ("residual", arch["residual"])) if on]
    lines = [
        f"- `{arch['factory']}`" + (f" ({', '.join(flags)})" if flags else "")
        + f", {arch['param_count']:,} trainable params, history {arch['history_steps']} steps; "
        f"ny={arch['ny']}, nu={arch['nu']}, nd={arch['nd']}",
        f"- History: {_lags_by_value(arch['history'])}",
    ]
    for key in ("fy", "fu", "fd"):
        info = arch.get("blocks", {}).get(key)
        if info is None:
            continue
        if info.get("blocks"):
            lines += _stack_lines(info)
        else:
            lines.append(f"- {key}: " + ", ".join(_block_parts(info)))
            lines += _own_lines(info, "  ")
    for column, entry in (arch.get("pre_stack") or {}).items():
        lines.append(f"- derived input {column}: {entry['kind']} of {entry['state']}, "
                     f"sign {entry['sign']:+d}, scale {entry['scale']:.3g}")
    return _section("Model", "\n".join(lines))


# ── Data ──────────────────────────────────────────────────────────


def _head_label(head) -> str:
    """Short label for a Head instance, e.g. 'bounded [0, +∞]' or 'ordinal K=4'."""
    label = head.kind or type(head).__name__
    if hasattr(head, "lo") and hasattr(head, "hi"):
        lo = "−∞" if head.lo == float("-inf") else f"{head.lo:g}"
        hi = "+∞" if head.hi == float("inf") else f"{head.hi:g}"
        label += f" [{lo}, {hi}]"
    if hasattr(head, "n_levels"):
        label += f" K={head.n_levels}"
    return label


def _norm_line(nm_mapping: dict, norm_stats: dict) -> str:
    """The commonest normalization, and the columns that differ from it."""
    columns = [c for role in ("Y", "U", "D") for c in nm_mapping.get(role, [])]
    kinds = {c: norm_stats.get(c, {}).get("norm_type", "?") for c in columns}
    common = Counter(kinds.values()).most_common(1)[0][0]
    others: dict[str, list[str]] = {}
    for c, kind in kinds.items():
        if kind != common:
            others.setdefault(kind, []).append(c)
    return (f"- Normalization: {common}"
            + "".join(f"; {kind}: {_names(cols)}" for kind, cols in others.items()))


def _outside_bounds(head_stack: Optional[HeadStack],
                    test_df: Optional[pd.DataFrame]) -> list[str]:
    """Bounded channels whose test data leaves the head's [lo, hi]: the model
    cannot reach those samples, so the channel was set up to fail."""
    if head_stack is None or test_df is None:
        return []
    out = []
    for name, head in zip(head_stack.channel_names, head_stack.heads):
        if not isinstance(head, BoundedHead) or name not in test_df.columns:
            continue
        vals = test_df[name].to_numpy()
        n_out = int(((vals < head.lo) | (vals > head.hi)).sum())
        if n_out:
            out.append(f"`{name}` has {n_out} test samples ({100 * n_out / vals.size:.1f}%) "
                       f"outside its head's {_head_label(head)}, which it cannot reach.")
    return out


def _data_section(nm_mapping: dict, split_sizes: Optional[dict[str, dict[str, int]]],
                  head_stack: Optional[HeadStack], norm_stats: Optional[dict]) -> str:
    """`split_sizes` is `{split: {"rows": int, "files": int}}`."""
    lines = []
    if split_sizes:
        lines.append("- Rows: " + " · ".join(
            f"{split} {v['rows']:,}" + (f" in {v['files']} files" if v["files"] > 1 else "")
            for split, v in split_sizes.items()))
    roles = " · ".join(f"{role} {len(nm_mapping.get(role, []))}" for role in ("Y", "U", "D"))
    lines.append(f"- Columns: {roles} (by name in `nm_mapping.json`)")
    if head_stack is not None:
        kinds = Counter(_head_label(h) for h in head_stack.heads)
        lines.append("- Heads: " + ", ".join(f"{n} {kind}" for kind, n in kinds.items()))
    if norm_stats is not None:
        lines.append(_norm_line(nm_mapping, norm_stats))
    return _section("Data", "\n".join(lines))


# ── Accuracy ──────────────────────────────────────────────────────


def _accuracy_section(rows: list[tuple[ChannelMetrics, float, float]],
                      test_window: int) -> str:
    shown = rows if len(rows) <= MAX_ROWS else rows[:WORST_ROWS]
    body = [f"RMSE of {test_window}-step rollouts, in each channel's units, against "
            "persistence (hold the last seeded value). Skill = 1 − RMSE / "
            "persistence; at or below 0 the model adds nothing to a copy-forward.\n"]
    if len(shown) < len(rows):
        body[0] += (f"The {len(shown)} lowest-skill of {len(rows)} channels; all are in "
                    f"`{RESULTS_DIR}/test_metrics.json`.\n")
    body += ["| Channel | RMSE | Persistence | Skill |", "|---|---:|---:|---:|"]
    for m, reference, s in shown:
        body.append(f"| `{m.name}` | {m.rmse:.4g} | {reference:.4g} | "
                    + ("n/a" if math.isnan(s) else f"{s:.2f}") + " |")
    return _section("Accuracy", "\n".join(body))


# ── Whiteness ─────────────────────────────────────────────────────


def _whiteness_section(white: list[Whiteness]) -> Optional[str]:
    if not white:
        return None
    body = ["One-step-ahead residuals: every test step predicted from the recorded "
            "history before it. A model that has the dynamics leaves white noise — "
            "ACF inside ±1.96/√n and a straight cumulated periodogram (Bacher & "
            "Madsen 2011). Structure is grouped by the period holding the most "
            "residual power: under an hour is a missing fast state or lag, daily "
            "is a daily driver no input carries, the longest periods a slow drift.\n"]
    structured = [w for w in white if not w.white]
    if structured:
        body += ["| Most residual power at | Channels | ACF at lag 1 |", "|---|---|---|"]
        for label, _, _ in PERIOD_BANDS:
            group = [w for w in structured if w.period_band == label]
            if group:
                lo, hi = min(w.lag1 for w in group), max(w.lag1 for w in group)
                lag1 = f"{lo:.2f}" if lo == hi else f"{lo:.2f} to {hi:.2f}"
                body.append(f"| {label} | {len(group)}: {_names([w.name for w in group])} | "
                            f"{lag1} |")
    clean = [w.name for w in white if w.white]
    body.append(f"\nWhite: {len(clean)} of {len(white)}"
                + (f" ({_names(clean)})." if clean else "."))
    return _section("Whiteness", "\n".join(body))


# ── Staged channels ───────────────────────────────────────────────


def _levels_section(levels: list[LevelMetrics], persistence_levels: list[LevelMetrics],
                    head_stack: Optional[HeadStack],
                    calibration: Optional[dict[str, Calibration]]) -> Optional[str]:
    """Balanced accuracy is mean per-level recall: a constant predictor scores
    1/K, and that is the floor, not 0. RMSE ranks a head that never fires
    above one that does, so a staged channel is read here."""
    if not levels:
        return None
    reference = {lm.name: lm.balanced_accuracy for lm in persistence_levels}
    body = ["Balanced accuracy (mean per-level recall; a constant predictor scores "
            f"1/K). Per-level shares and recall: `{RESULTS_DIR}/level_metrics.json`.\n",
            "| Channel | Balanced acc | Persistence | Thresholds |", "|---|---:|---:|---|"]
    heads = {} if head_stack is None else dict(zip(head_stack.channel_names, head_stack.heads))
    for lm in levels:
        head = heads.get(lm.name)
        # With use_ste=False the smooth activation is fed back and the snap
        # never runs, so there is no decision rule to report.
        rule = (", ".join(f"{t:.4g}" for t in head.decision_thresholds.tolist())
                if isinstance(head, OrdinalHead) and head.use_ste else "—")
        if calibration and lm.name in calibration:
            c = calibration[lm.name]
            rule += f" (calibrated, share error {c.trace[c.kept].error:.3f})"
        body.append(f"| `{lm.name}` | {lm.balanced_accuracy:.3f} | "
                    f"{reference.get(lm.name, float('nan')):.3f} | {rule} |")
    return _section("Staged channels", "\n".join(body))


# ── Physics ───────────────────────────────────────────────────────


def _group_checks(results: list[ScenarioResult]) -> dict[str, list[ScenarioResult]]:
    """Results by check, in run order; a step response's actuator is dropped
    from its name ("Step response: X to v" -> "Step response")."""
    groups: dict[str, list[ScenarioResult]] = {}
    for r in results:
        groups.setdefault(r.name.split(":")[0], []).append(r)
    return groups


def _physics_section(results: list[ScenarioResult], hours: Optional[float],
                     n_steps: Optional[int]) -> str:
    """Assertions on synthetic states the recorded data never contains: the
    HVAC fully off, the weather stepped, a zone left to decay. Each check runs
    once per state channel, so a channel you rely on cannot hide behind one
    you do not."""
    horizon = (f" over {hours:g} h ({n_steps} steps)"
               if hours is not None and n_steps is not None else "")
    body = [f"Each check puts the model in a state the data never visits{horizon} "
            "and asserts what has to happen next, per state channel. A controller "
            "goes to these states, so failing here is not out-of-scope.\n",
            "| Check | Pass | Failed: value (limit) |", "|---|---:|---|"]
    examples = []
    for check, rs in _group_checks(results).items():
        failed = [r for r in rs if not r.passed]
        cells = _names([f"{r.channel} {r.value:.3g} ({r.tolerance:.3g})" for r in failed],
                       quote=False)
        body.append(f"| {check} | {len(rs) - len(failed)}/{len(rs)} | {cells or '—'} |")
        if failed:
            examples.append(f"- {check}, e.g. `{failed[0].channel}`: {failed[0].headline}")
    body += [""] + examples if examples else []
    if hours is not None and any(r.name == FreeDecayScenario.label for r in results):
        # closed = 1 - exp(-hours / tau) clears DECAY_MIN_CLOSED for any tau
        # up to this, so the check cannot falsify a slow-mode prior.
        longest_h = hours / -math.log(1 - DECAY_MIN_CLOSED)
        body.append(f"\nFree decay tests the sign of the envelope coupling, not its "
                    f"rate: any slow mode under ~{longest_h:.0f} h passes.")
    return _section("Physics", "\n".join(body))


# ── Rollout ───────────────────────────────────────────────────────


def _flagged(stability: list[RolloutStability]) -> list[RolloutStability]:
    return [s for s in stability
            if s.looks_like_ramp() or s.drift_ratio > DRIFT_RATIO_WARN]


def _rollout_section(flagged: list[RolloutStability], test_window: int) -> Optional[str]:
    """Shape, not error: a model that stopped responding to its inputs emits
    a ramp a short window's RMSE can hide."""
    if not flagged:
        return None
    ramps = [s.name for s in flagged if s.looks_like_ramp()]
    drifts = [s for s in flagged if not s.looks_like_ramp()]
    body = [f"Over {test_window}-step rollouts; closed loop runs far longer, so free-run "
            "a flagged channel over a long horizon before training a controller on it.\n"]
    if ramps:
        body.append(f"- Ramping (monotone, unlike the truth): {_names(ramps, limit=WORST_ROWS)}")
    if drifts:
        worst = sorted(drifts, key=lambda s: -s.drift_ratio)
        body.append(f"- Drifting past {DRIFT_RATIO_WARN:g}× the truth: " + _names(
            [f"{s.name} {s.drift_ratio:.1f}×" for s in worst], limit=WORST_ROWS, quote=False))
    return _section("Rollout", "\n".join(body))


# ── Hyperparameters ───────────────────────────────────────────────


def _hyperparameters_line(cfg: ModelingConfig) -> str:
    gains = ", ".join(f"{name} {gain}" for name, gain in
                      (("ref", cfg.ref_gain), ("grad", cfg.grad_gain)) if gain != 0)
    return _section("Hyperparameters", (
        f"lr {cfg.lr} · batch {cfg.batch_size} · epochs {cfg.epochs} · pred steps "
        f"{cfg.train_pred_steps}/{cfg.dev_pred_steps} (train/dev) · test window "
        f"{cfg.test_window} · curriculum {cfg.curriculum_steps} (ratio "
        f"{cfg.curriculum_ratio}) · seed {cfg.seed} · loss gains {gains or 'none'}"))


@beartype
def render_reportcard(
    cfg: ModelingConfig,
    arch: dict,
    test_metrics: list[ChannelMetrics],
    nm_mapping: dict[str, list[str]],
    *,
    stability: Optional[list[RolloutStability]] = None,
    physics: Optional[list[ScenarioResult]] = None,
    physics_steps: Optional[int] = None,
    whiteness: Optional[list[Whiteness]] = None,
    levels: Optional[list[LevelMetrics]] = None,
    persistence: Optional[list[ChannelMetrics]] = None,
    persistence_levels: Optional[list[LevelMetrics]] = None,
    calibration: Optional[dict[str, Calibration]] = None,
    ensemble: Optional[EnsembleSpread] = None,
    head_stack: Optional[HeadStack] = None,
    norm_stats: Optional[dict] = None,
    test_df: Optional[pd.DataFrame] = None,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    best_dev_loss: Optional[float] = None,
    train_time_s: Optional[float] = None,
    test_time_s: Optional[float] = None,
    split_sizes: Optional[dict[str, dict[str, int]]] = None,
) -> str:
    """
    Render a markdown reportcard for one training run.

    Args:
        cfg:           Resolved ModelingConfig used for the run.
        arch:          Output of models.describe_dynamics(dyn_sys, cfg).
        test_metrics:  Output of metrics.per_channel_metrics(...).
        persistence:   The same for persistence, the bar `test_metrics` is
                       scored against; required for the accuracy section.
        whiteness:     One-step residual whiteness per continuous channel.
        levels:        Classification metrics for the binary / ordinal
                       channels, beside `persistence_levels`.
        calibration:   Optional {channel: Calibration} from
                       calibrate_decision_thresholds.
        ensemble:      Optional EnsembleSpread from cfg.ensemble_seeds > 1.
        head_stack:    Optional HeadStack, physical units: head kinds,
                       decision thresholds, and bounded-coverage warnings
                       against `test_df`.
        run_name:      MLflow run name (falls back to cfg.run).
        timestamp:     ISO-ish timestamp string (defaults to now()).
        split_sizes:   Optional {split: {"rows": N, "files": K}} per split.

    Returns:
        Markdown string. The caller writes it to disk and logs it.
    """
    name = run_name if run_name is not None else cfg.run
    ts = timestamp if timestamp is not None else datetime.now().isoformat(
        timespec="seconds"
    )
    rows = ranked_skills(test_metrics, persistence or [])
    white = whiteness or []
    flagged = _flagged(stability or [])

    warnings = _outside_bounds(head_stack, test_df)
    if norm_stats is not None:
        frozen = degenerate_mapped_columns(norm_stats, nm_mapping)
        if frozen:
            warnings.append(
                f"Constant in this data: {_names(frozen)}. Normalization had nothing "
                "to scale, so they stay in raw units; one that varies at evaluation "
                "time arrives far out of range. Exclude it or widen the excitation.")

    parts = [
        _header(name, ts, cfg.lr > 0, best_dev_loss, train_time_s, test_time_s),
        _verdict_section(rows, white, physics, flagged, cfg.test_window,
                         levels or [], persistence_levels or [], warnings),
        _architecture_section(arch),
        _data_section(nm_mapping, split_sizes, head_stack, norm_stats),
        _accuracy_section(rows, cfg.test_window),
    ]
    optional = [
        _whiteness_section(white),
        _levels_section(levels or [], persistence_levels or [], head_stack, calibration),
        None if physics is None else _physics_section(physics, cfg.synth_hours, physics_steps),
        _rollout_section(flagged, cfg.test_window),
        _ensemble_section(ensemble),
    ]
    parts += [p for p in optional if p is not None]
    parts.append(_hyperparameters_line(cfg))
    return "\n".join(parts)
