"""
Offline diagnostics: read a dataset, compute one statistic, print a
recommendation. No training, no pipeline wiring, no extension point.

Two decisions in this library are made by hand today, from the column's
*name* rather than its data: which head a Y channel gets (`data_head_config`)
and the 2R2C storage prior (a block's `tau_slow_h` box). Both are questions
the data can answer, so this module answers them:

  recommend_head   — 1-D k-means over a Y column's values; recommends
                     RegressionHead / BinaryHead / OrdinalHead / BoundedHead
                     from the SHAPE of the distribution, not the column name.
  estimate_decay   — finds runs where the actuators sit at their `off`
                     values for a while and fits an exponential envelope
                     decay, giving the 2R2C storage prior (`tau_slow`)
                     something to be checked against instead of just
                     asserted.

Everything here is pure: numpy arrays in, dataclasses/strings out. The CLI
(`modeling/cli/data.py`) owns loading the dataset and hooks and printing
these results; that split is what makes the statistics testable with plain
synthetic arrays instead of a real dataset and config file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── nt data heads ─────────────────────────────────────────────────

# K sweep upper bound. Staged HVAC equipment (off/stage1/stage2/...) rarely
# has more than a handful of discrete states; six is generous headroom
# without paying for an unbounded search.
K_MAX_DEFAULT = 6

# A sample farther than this fraction of the SMALLEST gap between adjacent
# cluster centers counts as sitting in the gap (unoccupied space between
# modes) rather than as noise around a mode. A quarter of the smallest gap
# keeps the "occupied" band tight around each mode without counting ordinary
# sensor noise as gap occupancy.
GAP_TOL_FRAC = 0.25

# A K whose cluster means explain at least this share of the column's total
# variance is doing real work, not carving noise into extra modes.
HIGH_VARIANCE_EXPLAINED = 0.90

# A K whose gaps hold no more than this share of samples has genuinely
# unoccupied space between modes -- the "staged, not smeared" signature.
LOW_GAP_OCCUPANCY = 0.05

# A mode holding less than this share of the samples is a rare event. Every
# default in the training stack -- MSE, a midpoint snap, NRMSE -- is
# indifferent to a level this rare by construction, so the recommendation
# says so rather than handing over a head whose rare level will be ignored.
RARE_MODE_FRAC = 0.05

# A column with at least this share of samples sitting at one end of its
# observed range has a real floor/ceiling (a clipped valve, a solar
# irradiance that floors at 0 overnight), not just its natural min/max.
BOUNDARY_MASS_FRAC = 0.05

# Samples within this fraction of the column's range from an endpoint count
# as "sitting at" that endpoint. Saturation/clipping repeats the same float
# exactly, so this can be tight -- same convention as scenarios.py's
# _SAME_VALUE_FRAC for "this value sits at a range end".
BOUNDARY_TOL_FRAC = 1e-6


@dataclass(frozen=True)
class LevelFit:
    """One K in the k-means sweep."""
    n_levels: int
    centers: tuple[float, ...]      # physical units, sorted ascending
    variance_explained: float       # share of total variance explained by cluster means
    gap_occupancy: float            # share of samples NOT close to a center
    mode_shares: tuple[float, ...]  # share of samples nearest each center


@dataclass(frozen=True)
class HeadRecommendation:
    """The recommendation for one Y column."""
    column: str
    kind: str                       # "regression" | "binary" | "ordinal" | "bounded"
    evidence: str                   # human-readable finding, see recommend_head
    fits: tuple[LevelFit, ...]       # the K sweep, for the report
    snippet: str                    # pasteable constructor call


def _kmeans_1d(x: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """1-D Lloyd's-algorithm k-means. Returns k sorted cluster centers.

    Centers are initialized at the k quantiles that evenly divide the data
    (e.g. k=3 -> the 1/6, 3/6, 5/6 quantiles) so the starting spread already
    tracks the data instead of an arbitrary random draw. Each iteration
    assigns every sample to its nearest center, then recomputes each center
    as the mean of the samples assigned to it, until the centers stop
    moving.

    A cluster that wins no sample is re-seeded at the sample farthest from
    its own center. On a mostly-off channel (a compressor at exactly 0 on
    80% of steps) several quantile seeds coincide at 0, ties go to the
    first, and without the re-seed the duplicates would sit there for good —
    every K would collapse to K=2 plus phantom centers.
    """
    quantile_positions = (np.arange(k) + 0.5) / k
    centers = np.quantile(x, quantile_positions)
    for _ in range(max_iter):
        distances = np.abs(x[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            in_cluster = x[labels == i]
            if in_cluster.size > 0:
                new_centers[i] = in_cluster.mean()
            else:
                farthest = np.argmax(distances[np.arange(len(x)), labels])
                new_centers[i] = x[farthest]
                # Taken: the next empty cluster must not land on it too.
                distances[farthest, :] = 0.0
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return np.sort(centers)


def fit_levels(x: np.ndarray, k_max: int = K_MAX_DEFAULT) -> tuple[LevelFit, ...]:
    """Sweep 1-D k-means over K = 2..k_max, one `LevelFit` per K.

    Stops early if the column has fewer than K distinct values -- k-means
    cannot place more clusters than there are points to put in them.
    """
    x = np.asarray(x, dtype=float)
    total_var = float(np.var(x))
    n_unique = len(np.unique(x))

    fits = []
    for k in range(2, k_max + 1):
        if k > n_unique:
            break
        centers = _kmeans_1d(x, k)
        distances = np.abs(x[:, None] - centers[None, :])
        nearest_dist = distances.min(axis=1)
        nearest = distances.argmin(axis=1)
        mode_shares = tuple(float(np.mean(nearest == i)) for i in range(k))

        # Share of variance explained by "which cluster this sample is in":
        # 1 minus the leftover (within-cluster) mean squared distance over
        # the column's total variance. A perfect k-way split scores 1.0; a
        # single mode split arbitrarily into k pieces scores low.
        if total_var > 0:
            variance_explained = 1.0 - float(np.mean(nearest_dist ** 2)) / total_var
        else:
            variance_explained = 1.0

        smallest_gap = float(np.diff(centers).min())
        gap_tol = GAP_TOL_FRAC * smallest_gap
        gap_occupancy = float(np.mean(nearest_dist > gap_tol))

        fits.append(LevelFit(
            n_levels=k,
            centers=tuple(centers.tolist()),
            variance_explained=variance_explained,
            gap_occupancy=gap_occupancy,
            mode_shares=mode_shares,
        ))
    return tuple(fits)


def _best_fit(fits: tuple[LevelFit, ...]) -> Optional[LevelFit]:
    """Smallest K past the elbow: the first K whose cluster means explain
    HIGH_VARIANCE_EXPLAINED of the variance while LOW_GAP_OCCUPANCY of
    samples sit in the gaps between them. None if no K qualifies."""
    for fit in fits:
        if (fit.variance_explained >= HIGH_VARIANCE_EXPLAINED
                and fit.gap_occupancy <= LOW_GAP_OCCUPANCY):
            return fit
    return None


def _k_sweep_line(fits: tuple[LevelFit, ...], elbow: Optional[int] = None) -> str:
    if not fits:
        return "K sweep: fewer than 2 distinct values, nothing to sweep"
    line = " | ".join(
        f"K={fit.n_levels}: {fit.variance_explained * 100:.1f}% var explained"
        for fit in fits
    )
    if elbow is not None:
        line += f"  (elbow at {elbow})"
    return line


def _occupancy_lines(centers: list[float], shares: list[float]) -> list[str]:
    """Where the mass sits, and a warning when a mode is a rare event."""
    lines = ["mode shares: " + ", ".join(
        f"{c:g} -> {s * 100:.1f}%" for c, s in zip(centers, shares))]
    rare = [(c, s) for c, s in zip(centers, shares) if s < RARE_MODE_FRAC]
    if rare:
        lines.append(
            "RARE LEVEL: " + ", ".join(f"{c:g} holds {s * 100:.1f}%" for c, s in rare)
            + f" of samples (under {RARE_MODE_FRAC * 100:.0f}%). This is a "
            "rare-event channel: an unweighted loss and a midpoint snap "
            "will learn to never emit it, and NRMSE will reward that. Keep "
            "balance_levels and calibrate_thresholds on, and judge the head "
            "by the reportcard's level metrics, not its NRMSE."
        )
    return lines


def _scope_note(column: str) -> str:
    return (
        f"Scope: this only reads {column!r}'s marginal distribution. A "
        "channel can be genuinely multimodal because an exogenous variable "
        "switches it between operating regimes while it stays continuous "
        "within each -- that would look staged here and isn't. Check the "
        "physical channel before taking the recommendation on faith."
    )


def _ordinal_snippet(column: str, values: list[float]) -> str:
    values_repr = ", ".join(f"{v:g}" for v in values)
    return f"OrdinalHead(n_levels={len(values)}, level_values=[{values_repr}])"


def _bounded_snippet(column: str, lo: float, hi: float) -> str:
    lo_expr = "float('-inf')" if math.isinf(lo) else f"{lo:g}"
    hi_expr = "float('inf')" if math.isinf(hi) else f"{hi:g}"
    return f"BoundedHead({lo_expr}, {hi_expr})"


def recommend_head(
    x: np.ndarray,
    column: str,
    family: Optional[str] = None,
    k_max: int = K_MAX_DEFAULT,
) -> HeadRecommendation:
    """Recommend a head for one Y column from the SHAPE of its distribution.

    The test is not "how many distinct values" -- a staged compressor at
    0/4.2/8.1 kW carries sensor noise on every sample and so has thousands
    of distinct floats, but is discrete by shape: values cluster tightly and
    the space between clusters is empty. `fit_levels` (1-D k-means, K=2..
    k_max) captures exactly that: `variance_explained` says how much of the
    spread is "which mode" rather than noise, `gap_occupancy` says how much
    of the data sits in the empty space between modes.

    Decision rules, in order:
      1. Exactly one distinct value: nothing to recommend from.
      2. Exactly two distinct values, {0, 1}: BinaryHead().
      3. A K in the sweep clears both thresholds (high variance explained,
         low gap occupancy): OrdinalHead at that K. This also covers "exactly
         two distinct values otherwise" -- k-means on two-valued data always
         explains ~100% of the variance at K=2 with 0% gap occupancy, so it
         falls out of the same rule rather than needing a separate branch.
      4. A large share of samples sit exactly at one or both range ends,
         with nothing in the sweep above qualifying as staged: BoundedHead.
      5. Otherwise: RegressionHead().

    `family` is the model family that produces **this column** -- not the
    run's `cfg.model`. With per-block families those differ: a run whose
    `cfg.model` is "2r2c" can still produce a power channel under an `mlp`
    block, and that channel has no conflict at all. Callers get the per-
    channel answer from `dynamics.family_by_channel`; None means "unknown",
    and no conflict is claimed.

    When the family is "2r2c" and the recommendation is ordinal, the evidence
    says so: `OrdinalHead.residual_compatible` is False and a 2r2c block
    requires its channels' heads to be residual-compatible (`get_dynamics`
    raises otherwise), so the recommendation cannot be taken as given -- that
    channel's family has to change, or a different head be accepted.

    What this cannot see: the marginal distribution only. A channel driven
    into several continuous regimes by some other variable looks the same
    as staged equipment here. See `_scope_note`, always appended.
    """
    # `+ 0.0` turns IEEE −0.0 (simulator output, `clip(lower=0)`) into 0.0,
    # so no floor prints as "-0" in the evidence or the pasteable snippet.
    x = np.asarray(x, dtype=float) + 0.0
    fits = fit_levels(x, k_max)
    uniques = np.unique(x)

    if len(uniques) < 2:
        evidence = (f"only one distinct value ({uniques[0]:g}) observed; "
                    f"nothing to recommend from.\n{_scope_note(column)}")
        return HeadRecommendation(column, "regression", evidence, fits, "RegressionHead()")

    if len(uniques) == 2 and set(uniques.tolist()) == {0.0, 1.0}:
        share_one = float(np.mean(x == 1.0))
        lines = ["exactly two values observed: 0, 1 (0/1 flag)",
                 *_occupancy_lines([0.0, 1.0], [1.0 - share_one, share_one]),
                 _k_sweep_line(fits), _scope_note(column)]
        return HeadRecommendation(column, "binary", "\n".join(lines), fits, "BinaryHead()")

    best = _best_fit(fits)
    if best is not None:
        centers = list(best.centers)
        lines = [
            f"{best.n_levels} modes at " + ", ".join(f"{c:g}" for c in centers),
            f"{(1 - best.gap_occupancy) * 100:.1f}% of samples close to a mode; "
            f"gaps hold {best.gap_occupancy * 100:.1f}%",
            *_occupancy_lines(centers, list(best.mode_shares)),
            _k_sweep_line(fits, elbow=best.n_levels),
        ]
        if family == "2r2c":
            lines.append(
                "2r2c conflict: this channel is produced by a 2r2c block, "
                "whose channels need residual-compatible heads, and "
                "OrdinalHead.residual_compatible is False -- get_dynamics "
                "will raise. Recommended anyway on the data's shape; move "
                "this channel to another family, or accept a different head."
            )
        lines.append(_scope_note(column))
        snippet = _ordinal_snippet(column, centers)
        return HeadRecommendation(column, "ordinal", "\n".join(lines), fits, snippet)

    lo, hi = float(x.min()), float(x.max())
    span = hi - lo
    tol = BOUNDARY_TOL_FRAC * span if span > 0 else 0.0
    share_lo = float(np.mean(np.abs(x - lo) <= tol))
    share_hi = float(np.mean(np.abs(x - hi) <= tol))
    pinned_lo = share_lo > BOUNDARY_MASS_FRAC
    pinned_hi = share_hi > BOUNDARY_MASS_FRAC
    if pinned_lo or pinned_hi:
        pinned = []
        if pinned_lo:
            pinned.append(f"{share_lo * 100:.1f}% pinned at the floor ({lo:g})")
        if pinned_hi:
            pinned.append(f"{share_hi * 100:.1f}% pinned at the ceiling ({hi:g})")
        evidence = (f"{'; '.join(pinned)}, continuous interior\n"
                    f"{_k_sweep_line(fits)}\n{_scope_note(column)}")
        bound_lo = lo if pinned_lo else float("-inf")
        bound_hi = hi if pinned_hi else float("inf")
        snippet = _bounded_snippet(column, bound_lo, bound_hi)
        return HeadRecommendation(column, "bounded", evidence, fits, snippet)

    top = fits[0] if fits else None
    var_pct = f"{top.variance_explained * 100:.1f}%" if top else "n/a"
    evidence = (f"unimodal: best K in the sweep explains only {var_pct} of "
                f"variance with samples spread through the gaps -- no "
                f"separable modes\n{_k_sweep_line(fits)}\n{_scope_note(column)}")
    return HeadRecommendation(column, "regression", evidence, fits, "RegressionHead()")


# ── nt data decay ──────────────────────────────────────────────────

# Runs shorter than this give too few points for a reliable slope, and a
# building rarely holds its actuators off for less than an hour anyway. If
# this misses real free-decay episodes in your data, lower it with
# --min-hours.
MIN_RUN_HOURS_DEFAULT = 4.0

# A control sits "at its off value" when within this fraction of its own
# observed range -- same scale-free convention as scenarios.py's
# EQUILIBRIUM_DRIFT_FRAC, for a recorded actuator rather than a synthesized
# equilibrium.
OFF_TOLERANCE_FRAC = 0.02

# A time constant longer than this multiple of the window it was fitted in is
# extrapolated, not measured: over a span well short of tau the decay is
# indistinguishable from a straight line, so the slope -- and tau with it --
# is set by noise rather than by the plant. Rejecting those windows is
# arithmetic about what the span can resolve, not a judgment about buildings.
MAX_TAU_PER_SPAN = 3.0


@dataclass(frozen=True)
class DecayWindow:
    """One free-decay episode: an off-mask run long enough to fit a slope."""
    start: int              # row index where the window begins
    n_steps: int
    tau_h: float
    half_life_h: float
    r_squared: float         # fit quality of log|gap| vs t, for the evidence


def actuator_off_mask(
    values: dict[str, np.ndarray],
    off: dict[str, float],
    tol_frac: float = OFF_TOLERANCE_FRAC,
) -> np.ndarray:
    """Boolean mask, True where every actuator in `values` sits within
    `tol_frac` of its own observed range from its declared `off` value.

    `off` must carry an entry for every key in `values` -- the caller is
    responsible for filling in "recorded minimum" for any control the
    `SynthSpec.off` hook left out, per that dataclass's own docstring.
    """
    n = len(next(iter(values.values())))
    mask = np.ones(n, dtype=bool)
    for col, x in values.items():
        rng = float(x.max() - x.min())
        tol = tol_frac * rng
        mask &= np.abs(x - off[col]) <= tol
    return mask


def find_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Maximal True-runs of `mask` at least `min_len` samples long, as
    (start, stop) index pairs with `stop` exclusive."""
    runs = []
    start = None
    # Sentinel False closes a run that reaches the end of the array.
    for i, is_on in enumerate(np.append(mask, False)):
        if is_on and start is None:
            start = i
        elif not is_on and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def _fit_log_decay(t_hours: np.ndarray, gap: np.ndarray) -> Optional[tuple[float, float]]:
    """OLS slope of log|gap| against t_hours. Returns (tau_h, r_squared), or
    None when the window cannot support a fit:

      - the slope must be negative (the gap shrinking),
      - the gap must never touch zero (log undefined; `estimate_decay` has
        already cut the run at its first crossing),
      - and the fitted tau must be short enough that this window could have
        resolved it (`MAX_TAU_PER_SPAN`).

    The last one is what separates a measurement from an extrapolation, and
    r_squared does not catch it: a 5-hour window fits a 127-hour constant
    happily, because over a span that short the exponential *is* a line."""
    if np.any(gap == 0):
        return None
    log_abs_gap = np.log(np.abs(gap))
    slope, intercept = np.polyfit(t_hours, log_abs_gap, 1)
    if slope >= 0:
        return None
    tau_h = -1.0 / slope
    span_h = float(t_hours[-1] - t_hours[0])
    if tau_h > MAX_TAU_PER_SPAN * span_h:
        return None
    predicted = slope * t_hours + intercept
    ss_res = float(np.sum((log_abs_gap - predicted) ** 2))
    ss_tot = float(np.sum((log_abs_gap - log_abs_gap.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return tau_h, r_squared


def _before_first_crossing(gap: np.ndarray) -> int:
    """Length of the prefix of `gap` before it first reaches or changes sign."""
    crossed = np.sign(gap) != np.sign(gap[0])
    return int(np.argmax(crossed)) if crossed.any() else len(gap)


def estimate_decay(
    t_hours: np.ndarray,
    state: np.ndarray,
    ref: np.ndarray,
    off: np.ndarray,
    min_run_steps: int,
) -> tuple[DecayWindow, ...]:
    """Find free-decay windows and fit an envelope time constant in each.

    `off` is a boolean mask (see `actuator_off_mask`), True where the plant
    is uncoupled. Within each run of at least `min_run_steps` consecutive
    True samples, regress log|state - ref| on elapsed time. A run is cut at
    the gap's first sign change and only the part before it is fitted: the
    reference is usually recorded ambient, which moves through the day, so a
    genuine overnight decay often crosses it once the morning warms, and the
    decay before the crossing is exactly the measurement wanted. A run whose
    remaining prefix is shorter than `min_run_steps`, or that is not an
    exponential decay toward `ref` (flat, growing, or too short to resolve
    the tau it fits), contributes no window rather than a bogus tau -- see
    `_fit_log_decay` for the rejections.

    Returns one `DecayWindow` per run that fits cleanly, in time order.
    Empty means no qualifying window was found in this data -- the `tau_slow_h`
    box then has to come from what is known about the building, held wide.
    """
    gap = np.asarray(state, dtype=float) - np.asarray(ref, dtype=float)
    windows = []
    for start, stop in find_runs(np.asarray(off, dtype=bool), min_run_steps):
        stop = start + _before_first_crossing(gap[start:stop])
        if stop - start < min_run_steps:
            continue
        t0 = t_hours[start:stop] - t_hours[start]
        fit = _fit_log_decay(t0, gap[start:stop])
        if fit is None:
            continue
        tau_h, r_squared = fit
        windows.append(DecayWindow(
            start=start, n_steps=stop - start, tau_h=tau_h,
            half_life_h=tau_h * math.log(2), r_squared=r_squared,
        ))
    return tuple(windows)


def summarize_taus(
    windows: tuple[DecayWindow, ...],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """(median, min, max) tau_h across windows, or (None, None, None)."""
    if not windows:
        return None, None, None
    taus = [w.tau_h for w in windows]
    return float(np.median(taus)), min(taus), max(taus)


# ── Latching: does the equipment remember whether it was running? ──

#: Bin edges for `latch_table`, in the state's units of error. Fine near zero,
#: where a thermostat's deadband lives.
LATCH_EDGES = (-math.inf, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, math.inf)


@dataclass(frozen=True)
class LatchRow:
    """How often the equipment runs in one bin of error, by its previous step."""
    low: float
    high: float
    p_after_running: Optional[float]
    n_after_running: int
    p_after_idle: Optional[float]
    n_after_idle: int


def latch_table(running: np.ndarray, error: np.ndarray,
                edges: tuple[float, ...] = LATCH_EDGES) -> list[LatchRow]:
    """P(running now) at the same error, split by whether it ran the step before.

    `running` is a boolean per step, `error` the thermostat's error (positive
    while the equipment is called for), both over one contiguous record. A
    memoryless response gives the same probability in both columns of a bin;
    a latch gives a high one after running and a low one after idling, across
    the bins either side of zero.
    """
    running = np.asarray(running, dtype=bool)
    error = np.asarray(error, dtype=float)
    now, before, e = running[1:], running[:-1], error[1:]
    rows = []
    for low, high in zip(edges, edges[1:]):
        in_bin = (e > low) & (e <= high)
        after_on, after_off = in_bin & before, in_bin & ~before
        rows.append(LatchRow(
            low, high,
            float(now[after_on].mean()) if after_on.any() else None, int(after_on.sum()),
            float(now[after_off].mean()) if after_off.any() else None, int(after_off.sum()),
        ))
    return rows


@dataclass(frozen=True)
class LatchEstimate:
    """What a record's starts and stops say about a thermostat stage, as
    (25th, 50th, 75th) percentiles, in the units a demand declaration takes;
    None when the record has no event to read."""
    start_error: Optional[tuple[float, float, float]]  # error when a stopped stage starts
    stop_error: Optional[tuple[float, float, float]]   # error when a running stage stops
    capacity: Optional[tuple[float, float, float]]     # state units per hour at a switch


def _quartiles(values: list[float]) -> Optional[tuple[float, float, float]]:
    if not values:
        return None
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return float(q25), float(q50), float(q75)


def latch_estimates(records: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                    sign: int, step_h: float) -> LatchEstimate:
    """The error at each start and stop, and the jump in the state's rate
    across it, from `(running, error, state)` records, each contiguous.

    Read at the events, conditions on either side of a switch are nearly the
    same, so the jump in rate is the stage's own effect: the capacity a
    saturating branch declares, in `sign`'s direction. A comparison of all
    running steps against all idle ones would not be: the equipment runs
    under heavier loads than it idles under. Staged equipment switching one
    stage makes it a lower bound. Advice for centring a box, not a fit.
    """
    starts, stops, jumps = [], [], []
    for running, error, state in records:
        running = np.asarray(running, dtype=bool)
        error, state = np.asarray(error, dtype=float), np.asarray(state, dtype=float)
        rate = np.diff(state) / step_h          # rate[k]: what step k's state did next
        for k in range(1, len(running) - 1):
            if running[k] == running[k - 1]:
                continue
            (starts if running[k] else stops).append(error[k])
            switched_on = 1.0 if running[k] else -1.0
            jumps.append(sign * switched_on * (rate[k] - rate[k - 1]))
    return LatchEstimate(_quartiles(starts), _quartiles(stops), _quartiles(jumps))


def pool_latch_tables(tables: list[list[LatchRow]]) -> list[LatchRow]:
    """Bin-wise counts and probabilities pooled across several records."""
    pooled = []
    for rows in zip(*tables):
        def pooled_p(p_key, n_key):
            n = sum(getattr(r, n_key) for r in rows)
            if not n:
                return None, 0
            return sum((getattr(r, p_key) or 0.0) * getattr(r, n_key) for r in rows) / n, n
        p_on, n_on = pooled_p("p_after_running", "n_after_running")
        p_off, n_off = pooled_p("p_after_idle", "n_after_idle")
        pooled.append(LatchRow(rows[0].low, rows[0].high, p_on, n_on, p_off, n_off))
    return pooled
