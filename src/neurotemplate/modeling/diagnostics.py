"""
Offline diagnostics: read a dataset, compute one statistic, print a
recommendation. No training, no pipeline wiring, no extension point.

Which head a Y channel gets (`data_head_config`) is otherwise chosen from
the column's *name*; it is a question the data can answer:

  recommend_head   — 1-D k-means over a Y column's values; recommends
                     RegressionHead / BinaryHead / OrdinalHead / BoundedHead
                     from the SHAPE of the distribution, not the column name.

A thermal zone's network and thermostat have their own commands (`ladder`,
`thermostat`).

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
    `cfg.model` is "rc_network" can still produce a power channel under an `mlp`
    block, and that channel has no conflict at all. Callers get the per-
    channel answer from `dynamics.family_by_channel`; None means "unknown",
    and no conflict is claimed.

    When the family is "rc_network" and the recommendation is ordinal, the evidence
    says so: `OrdinalHead.residual_compatible` is False and an rc_network block
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
        if family == "rc_network":
            lines.append(
                "rc_network conflict: this channel is produced by an rc_network "
                "block, whose channels need residual-compatible heads, and "
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
