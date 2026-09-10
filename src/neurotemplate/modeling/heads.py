"""
Configurable per-channel output heads.

Each Y channel has a Head that decides two things:

  - activation: how to map the model's raw pre-activation output `z` to
    a value usable for autoregressive state feedback and for comparison
    against ground truth.
  - loss:       how to score a per-channel prediction against the truth.

The dynamics model produces a raw [batch, raw_width] output. `HeadStack`
slices that output per channel, applies each head's activation to build
the activated state [batch, n_channels] that loops back into the model,
and aggregates per-channel losses during training.

Discrete forward, smooth backward (straight-through estimator)
=============================================================

Each head exposes a smooth `activate(z)` (used for gradients) and a
discrete `decode(z)` (snap to level / clamp to bound / etc.). When a
head has no meaningful discrete form (regression, binary), `decode`
defaults to `activate` and the rest of the machinery is a no-op.

`HeadStack.activate` — the one thing the autoregressive loop and the
loss both call — combines them per channel with the straight-through
trick

    y_ste = y_soft + (y_hard - y_soft).detach()

so that the *value* fed back to the loop and consumed by the loss is
the snapped one, but gradients flow through the smooth surrogate as if
no snapping happened. This keeps the loop state clean (discrete
channels actually see discrete states; the model can't get stuck at an
intermediate fixed point) without breaking BPTT — including for
controller-in-loop training, which depends on differentiability all
the way through. The gradient is an approximation: it describes
sensitivity in the smooth surrogate, not the discrete forward
function. That's the well-known STE tradeoff — works well in practice
for shallow non-differentiabilities, but the closer the controller's
job is to *triggering* transitions across thresholds, the more the
gradient can mislead.

The combine is opt-out per head via the `use_ste` flag (default True
on heads that override `decode`). With `use_ste=False`, the head
exposes its smooth `activate(z)` directly to the loop — useful for
comparing runs with and without snapped feedback.

Gradient flow design
====================

The activation choice is a real design decision, not a styling one — a
saturated activation kills training before the optimiser sees a signal.

  regression
    Identity activation. ∂y/∂z = 1 everywhere. Standard MSE loss.

  binary
    Sigmoid for state feedback. Loss is BCE on the activated probability
    σ(z), optionally class-weighted (see "Rare levels" below). The
    BCE-with-logits form would be marginally more stable at extreme
    |z|, but we keep the loss interface uniform (all losses operate on
    activated state) so the dynamics System can expose one output
    tensor — clarity wins over the small stability margin until a real
    run shows the difference.

  bounded
    Three activation modes selected by which side is finite. All three
    take a sharpness `β` controlling how quickly the activation
    saturates near its bound(s):

      Fully bounded  (both lo, hi finite):
          y = lo + (hi - lo) · σ(β·z/(hi-lo))   [gap-scaled sigmoid]
          ∂y/∂z saturates on both sides.

      Lower-bounded only (hi = +∞):
          y = lo + softplus(z, β)
          ∂y/∂z = σ(β·z) > 0; linear (≈ 1) on the open side,
          saturates only as z → -∞.

      Upper-bounded only (lo = -∞):
          y = hi - softplus(-z, β)       [mirror of above]

    ReLU is explicitly rejected for all three forms — ∂ReLU/∂z = 0 for
    z < 0 would stall any channel that drifts to the wrong side.
    Pass float('inf') / float('-inf') on the open side. Loss is MSE on
    activated values for all three modes. `decode` clamps the smooth
    activation to [lo, hi] for exact-bound forward output through the
    STE combine.

  ordinal
    K-level ordered output via a sum-of-sigmoids discretizer over K-1
    fixed thresholds. Activation: ŷ = v_0 + Σ_k σ(β·(z - θ_k)/Δv_k) · Δv_k
    with k = 1..K-1, bounded structurally to [v_0, v_{K-1}]. The
    sharpness `β` controls how closely ŷ snaps to the nearest level
    between transitions (larger β gives crisper steps at the cost of
    weaker gradients away from any threshold). Thresholds are fixed at
    the midpoints between consecutive level_values: the caller already
    picks `level_values`, so the midpoint placement is optimal by
    construction. Learnable thresholds would just give the model a
    redundant degree of freedom and threshold drift breaks the
    predict-then-snap semantics. Loss is MSE on ŷ vs the target level,
    optionally class-weighted. `decode` snaps to the exact level
    corresponding to the count of *decision* thresholds exceeded — a
    separate set from the activation thresholds, see "Rare levels".
    Tradeoff: gradient w.r.t. z vanishes when z drifts far from every
    threshold, and the vanishing region grows with β. If saturation
    becomes a problem, lower β or swap MSE for cumulative-link NLL.

Rare levels: class weights and the decision rule
================================================

Staged equipment is usually a rare-event channel — a furnace that fires
on 2% of samples, a fan that reaches its top stage on 5%. Two things in
a plain MSE/BCE head are indifferent to that by construction, and both
are fixed inside the head with no change to the System's output:

  level_weights
    Per-level loss weights, indexed by the *target's* level. The loss is
    Σ w·err / Σ w, so all-ones (the default) reproduces the unweighted
    mean exactly and the loss scale does not move with the weights.
    Inverse training frequency (`inverse_frequency_weights`) makes each
    level count equally regardless of how often it occurs; the pipeline
    sets that by default (`cfg.balance_levels`). Measured on a
    2.5%-positive compressor channel, this is the difference between a
    coin flip (ROC-AUC 0.74 ± 0.27 across seeds) and a model (0.94 ±
    0.01) — MSE alone does not reliably train on such a channel.

  decision_thresholds (ordinal only)
    Where `decode` snaps. Defaults to the same midpoints the activation
    uses, which is a p = 0.5 boundary — a balanced prior the data does
    not have. On a rare level the ranking z produces is usually fine and
    the snap point is what is wrong: moving the threshold to maximise
    balanced accuracy recovered +0.11 on training data alone, +0.17
    calibrated on dev, with no retraining. The pipeline calibrates these
    on the dev split after training (`cfg.calibrate_thresholds`), and
    they round-trip through `config()` so a reloaded run decides the
    same way it was evaluated. The activation thresholds stay at the
    midpoints: learning them under the training loss buys nothing.

Both were tested one-step-ahead with true history; neither has been
verified to repair the autoregressive collapse of a rare channel, which
is where the surrogate actually fails. Treat the classification metrics
on the reportcard as the check.

Residual prediction (y_{k+1} = f(...) + y_k) is only meaningful for
heads whose pre-activation `z` lives in the same space as `y_k`. That's
regression. Binary, bounded, and ordinal operate in different spaces
(logit / softplus-shifted / pre-threshold) where adding y_k mixes
units. The dynamics model asks each head whether it accepts a residual
via `Head.residual_compatible`.

Serialization
=============

Heads are reconstructed at load time from a JSON-safe config produced
by `HeadStack.to_config()` and consumed by `HeadStack.from_config()`.
The config records each head's `kind` plus its constructor arguments
in head-space — load_state_dict afterwards restores trained parameters.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurotemplate._typing import beartype


# ── Head base class ──────────────────────────────────────────────


class Head(nn.Module, ABC):
    """
    One head per Y channel.

    Subclasses set:
      kind                  : short string identifier used in the
                              serialized config (e.g. "regression").
      width                 : number of raw model outputs this head
                              consumes (1 for every head in this module —
                              kept on the class so future variable-width
                              heads only need to override one attribute).
      residual_compatible   : True if the dynamics model may add y_prev
                              to z before activation.

    Subclasses implement:
      activate(z) -> y_soft          : smooth, differentiable forward
                                       value. Used as the backward path
                                       in the STE combine.
      loss(y_state, target) -> scalar: per-channel training loss on the
                                       STE-combined state.
      config() -> dict               : JSON-safe constructor kwargs so the
                                       topology can be rebuilt at load time.

    Subclasses may override:
      decode(z) -> y_hard            : snapped / discrete forward value
                                       (clamp to bound, snap to level).
                                       Default: same as activate, which
                                       makes the STE combine a no-op.

    The loop and the loss both consume the STE-combined value via
    HeadStack.activate, so subclasses do not need to do anything special
    to participate — implement decode where there is a meaningful
    discrete form, leave it alone otherwise.
    """

    kind: str = ""
    width: int = 1
    residual_compatible: bool = False
    # Per-head straight-through-estimator toggle. Overridden in __init__
    # for heads where the user might want to disable snapping in the
    # autoregressive loop. No effect on heads whose decode() defaults
    # to activate() — the combine is a no-op there.
    use_ste: bool = True

    @abstractmethod
    def activate(self, z: torch.Tensor) -> torch.Tensor:
        """Smooth, differentiable [..., width] -> [...] activation."""

    @abstractmethod
    def loss(self, y_state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scalar loss for STE-combined [...] vs target [...]."""

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Snapped / discrete forward value. Default: no snapping — equal to
        activate(z), making the STE combine a no-op for this head."""
        return self.activate(z)

    def levels(self) -> Optional[torch.Tensor]:
        """The discrete values this head's output stands for, ascending, or
        None for a continuous head. Non-None heads are scored as classifiers
        (balanced accuracy, per-level recall) on top of the regression
        metrics, and their loss can be class-weighted via `level_weights`."""
        return None

    def level_index(self, y: torch.Tensor) -> torch.Tensor:
        """Index into `levels()` of the level nearest each value of `y`.
        Works for both the truth (exactly on a level) and an emitted value
        (snapped for ordinal, a probability for binary — nearest of {0, 1}
        is the p = 0.5 rule)."""
        levels = self.levels()
        if levels is None:
            raise TypeError(f"{type(self).__name__} has no levels to index")
        return (y.unsqueeze(-1) - levels).abs().argmin(dim=-1)

    def config(self) -> dict:
        """JSON-safe kwargs needed to reconstruct this head. Default: none."""
        return {}


# ── Concrete heads ───────────────────────────────────────────────


class RegressionHead(Head):
    """Identity activation, MSE loss."""

    kind = "regression"
    width = 1
    residual_compatible = True

    def activate(self, z: torch.Tensor) -> torch.Tensor:
        return z.squeeze(-1)

    def loss(self, y_state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_state, target)


class BinaryHead(Head):
    """Sigmoid activation, BCE on activated probability. Target must be 0/1.

    `level_weights` (shape (2,), default ones) weights each sample's BCE by
    its target class; see the module docstring. The emitted value is a
    probability, not a decision, so there is no decision threshold here —
    the classification metrics read it at p = 0.5.
    """

    kind = "binary"
    width = 1
    residual_compatible = False

    def __init__(self, level_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("_levels", torch.tensor([0.0, 1.0]), persistent=False)
        self.register_buffer(
            "level_weights", _checked_level_weights(level_weights, 2), persistent=False,
        )

    def activate(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z.squeeze(-1))

    def levels(self) -> torch.Tensor:
        return self._levels

    def loss(self, y_state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.level_weights[self.level_index(target)]
        bce = F.binary_cross_entropy(y_state, target, reduction="none")
        return (w * bce).sum() / w.sum()

    def config(self) -> dict:
        return {"level_weights": self.level_weights.tolist()}


class BoundedHead(Head):
    """
    Bounded output with three activation modes selected by which side of
    the bound is finite:

      Fully bounded  (both lo, hi finite):
          y = lo + (hi - lo) * sigmoid(β · z / (hi - lo))
      Lower bounded  (hi = +inf):
          y = lo + softplus(z, β)
      Upper bounded  (lo = -inf):
          y = hi - softplus(-z, β)

    Pass `float('inf')` / `float('-inf')` on the open side for the
    half-bounded cases. At least one side must be finite — both-infinite
    is a regression head and is rejected.

    MSE on the activated value. All three activations have nonzero
    gradient everywhere; softplus' derivative is sigmoid(β·z) > 0 and is
    linear (no saturation) on the open side, so half-bounded heads have
    *gentler* gradient behavior than fully-bounded ones — only the
    closed side ever saturates.

    `lo` / `hi` are interpreted in head-space (whatever space the data
    is consumed in). Callers building heads from physical bounds should
    project them with `project_physical_to_norm` first.

    `decode` clamps the smooth band to [lo, hi] so the STE combine emits
    exact-bound values forward while gradient flows through the smooth
    activation backward.
    """

    kind = "bounded"
    width = 1
    residual_compatible = False

    def __init__(self, lo: float, hi: float, beta: float = 5.0,
                 use_ste: bool = True):
        super().__init__()
        lo_finite = math.isfinite(lo)
        hi_finite = math.isfinite(hi)
        if not (lo_finite or hi_finite):
            raise ValueError(
                f"{type(self).__name__} requires at least one finite bound "
                "(use RegressionHead for an unbounded channel)"
            )
        if lo_finite and hi_finite and hi <= lo:
            raise ValueError(
                f"{type(self).__name__} requires hi > lo, got lo={lo} hi={hi}"
            )
        if beta <= 0:
            raise ValueError(
                f"{type(self).__name__} requires beta > 0, got beta={beta}"
            )
        self.lo = float(lo)
        self.hi = float(hi)
        self.beta = float(beta)
        self.use_ste = bool(use_ste)
        self._lo_finite = lo_finite
        self._hi_finite = hi_finite

    def _to_band(self, z: torch.Tensor) -> torch.Tensor:
        if self._lo_finite and self._hi_finite:
            # Gap-scaled sigmoid: dividing by (hi-lo) makes β express
            # "transition width as a fraction of the band" so the same
            # β works at any absolute band scale.
            return self.lo + (self.hi - self.lo) * torch.sigmoid(
                self.beta * z / (self.hi - self.lo)
            )
        if self._lo_finite:
            return self.lo + F.softplus(z, beta=self.beta)
        return self.hi - F.softplus(-z, beta=self.beta)

    def activate(self, z: torch.Tensor) -> torch.Tensor:
        return self._to_band(z.squeeze(-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Smooth band activation, hard-clamped to [lo, hi]. torch.clamp
        treats infinite bounds as no-ops, so half-bounded heads work
        without special-casing."""
        return self._to_band(z.squeeze(-1)).clamp(self.lo, self.hi)

    def loss(self, y_state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_state, target)

    def config(self) -> dict:
        return {
            "lo": _float_to_json(self.lo),
            "hi": _float_to_json(self.hi),
            "beta": self.beta,
            "use_ste": self.use_ste,
        }


class OrdinalHead(Head):
    """
    K-level ordinal head: sum-of-sigmoids over K-1 fixed thresholds,
    MSE on the activated value vs the target level.

    Activation:   ŷ = v_0 + Σ_k σ(β·(z − θ_k)/Δv_k) · Δv_k,
                  k = 1..K-1,  ŷ ∈ [v_0, v_{K-1}].
    Loss:         MSE(ŷ, target).
    Decode:       snap to v_k where k = count of thresholds exceeded.

    Width = 1 (one scalar z per channel). The dynamics model's outsize
    is unchanged, so this head drops into existing factories without any
    output-dim surgery.

    Thresholds are fixed (registered buffer) at the values given by
    `init_thresholds`, or at the midpoints between consecutive
    `level_values` when not specified. The caller already picks
    `level_values`, so the midpoint placement is optimal by
    construction — learnable thresholds would just give the model a
    redundant degree of freedom (it could shift z or shift θ to express
    the same correction) and threshold drift breaks the
    "predict-then-snap" semantics of the head.

    `residual_compatible = False`. Two reasons:
      1. Ordinal sequences are typically piecewise constant with rare
         step transitions. Δlevel is overwhelmingly zero, so the
         residual `f(...)` would have to learn the "predict zero with
         rare spikes" distribution — a classic rare-event regression
         failure mode in which the min-MSE solution collapses to the
         mean (~0) and never transitions.
      2. y_prev lives in level space [v_0, v_{K-1}]; f(...) lives in
         pre-threshold z-space. The 1:1 correspondence assumed by an
         additive residual depends on threshold spacing, which the
         model does not own. Non-residual sidesteps the muddle.

    `decision_thresholds` (K-1, default: the same midpoints) is where
    `decode` snaps; `level_weights` (K, default ones) weights the loss by
    the target's level. Both exist because a rare level is invisible to
    an unweighted MSE and to a midpoint snap — see the module docstring.
    Neither is a trained parameter: they are constructor arguments that
    round-trip through `config()`, and the pipeline sets them after
    training from the data rather than the optimiser.

    `level_values`, `init_thresholds` and `decision_thresholds` live in
    head-space. Callers constructing the head from physical units should
    project them through `project_physical_to_norm` first.
    """

    kind = "ordinal"
    width = 1
    residual_compatible = False

    def __init__(
        self,
        n_levels: int,
        init_thresholds: Optional[torch.Tensor] = None,
        level_values: Optional[torch.Tensor] = None,
        beta: float = 15.0,
        use_ste: bool = True,
        decision_thresholds: Optional[torch.Tensor] = None,
        level_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if n_levels < 2:
            raise ValueError(f"OrdinalHead requires n_levels >= 2, got {n_levels}")
        if beta <= 0:
            raise ValueError(f"OrdinalHead requires beta > 0, got beta={beta}")
        self.n_levels = int(n_levels)
        self.beta = float(beta)
        self.use_ste = bool(use_ste)

        # level_values: K output values the head can emit, in head-space.
        # Default is the integer-index sequence [0, 1, ..., K-1].
        if level_values is None:
            level_values = torch.arange(n_levels, dtype=torch.float32)
        else:
            level_values = level_values.to(torch.float32)
            if level_values.shape != (n_levels,):
                raise ValueError(
                    f"level_values must have shape ({n_levels},), "
                    f"got {tuple(level_values.shape)}"
                )
            gaps_v = level_values[1:] - level_values[:-1]
            if (gaps_v <= 0).any():
                raise ValueError(
                    f"level_values must be strictly increasing, got {level_values.tolist()}"
                )
        self.register_buffer("level_values", level_values)
        # Cached Δv_k = v_k − v_{k-1} for the gap-scaled sigmoid. Shape (K-1,).
        self.register_buffer("level_gaps", level_values[1:] - level_values[:-1])

        # Activation thresholds: fixed at init time. Default is the
        # midpoints between consecutive level_values. Registered as a
        # buffer so it moves with .to(device) and round-trips through
        # state_dict.
        midpoints = 0.5 * (level_values[:-1] + level_values[1:])
        self.register_buffer(
            "thresholds", _checked_thresholds(init_thresholds, midpoints, "init_thresholds"),
        )
        # Decision thresholds: where decode snaps. Not in state_dict — the
        # pipeline calibrates them after the weights are saved, so config()
        # is their single source of truth (load_state_dict must not undo it).
        self.register_buffer(
            "decision_thresholds",
            _checked_thresholds(decision_thresholds, midpoints, "decision_thresholds"),
            persistent=False,
        )
        self.register_buffer(
            "level_weights", _checked_level_weights(level_weights, n_levels),
            persistent=False,
        )

    def activate(self, z: torch.Tensor) -> torch.Tensor:
        z = z.squeeze(-1)
        # Gap-scaled sum of sigmoids: each sigmoid's argument is divided
        # by its own level gap Δv_k, so β has a consistent meaning
        # ("transition width as a fraction of the gap") regardless of
        # level scale. With the default level_values = [0,1,...,K-1] all
        # Δv_k = 1 and the formula collapses to the textbook form.
        gaps = self.level_gaps                    # (K-1,)
        sigmoid_args = self.beta * (z.unsqueeze(-1) - self.thresholds) / gaps
        steps = torch.sigmoid(sigmoid_args) * gaps  # (..., K-1)
        return self.level_values[0] + steps.sum(dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Exact level value: snap to the k-th level corresponding to the
        count of decision thresholds exceeded. Returns one of self.level_values."""
        z = z.squeeze(-1)
        idx = (z.unsqueeze(-1) > self.decision_thresholds).sum(dim=-1)
        return self.level_values[idx]

    def levels(self) -> torch.Tensor:
        return self.level_values

    def loss(self, y_state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.level_weights[self.level_index(target)]
        return (w * (y_state - target) ** 2).sum() / w.sum()

    def config(self) -> dict:
        return {
            "n_levels": self.n_levels,
            "beta": self.beta,
            "level_values": self.level_values.tolist(),
            "use_ste": self.use_ste,
            "decision_thresholds": self.decision_thresholds.tolist(),
            "level_weights": self.level_weights.tolist(),
        }


def _checked_thresholds(
    given: Optional[torch.Tensor], default: torch.Tensor, name: str,
) -> torch.Tensor:
    """`given` as float32 with the shape of `default` and strictly increasing,
    or `default` when not given."""
    if given is None:
        return default.clone()
    thresholds = given.to(torch.float32)
    if thresholds.shape != default.shape:
        raise ValueError(
            f"{name} must have shape {tuple(default.shape)}, "
            f"got {tuple(thresholds.shape)}"
        )
    if (thresholds[1:] <= thresholds[:-1]).any():
        raise ValueError(
            f"{name} must be strictly increasing, got {thresholds.tolist()}"
        )
    return thresholds


def _checked_level_weights(given: Optional[torch.Tensor], n_levels: int) -> torch.Tensor:
    """`given` as a positive float32 (n_levels,) tensor, or ones when not given."""
    if given is None:
        return torch.ones(n_levels)
    weights = given.to(torch.float32)
    if weights.shape != (n_levels,):
        raise ValueError(
            f"level_weights must have shape ({n_levels},), got {tuple(weights.shape)}"
        )
    if (weights <= 0).any():
        raise ValueError(f"level_weights must be positive, got {weights.tolist()}")
    return weights


def inverse_frequency_weights(y: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Per-level weights ∝ 1 / (share of `y` nearest each level), scaled so
    they average 1. A level `y` never visits gets the weight of a single
    sample, so it is neither infinite nor silently dropped."""
    idx = (y.reshape(-1, 1) - levels).abs().argmin(dim=-1)
    counts = torch.bincount(idx, minlength=len(levels)).clamp(min=1).to(torch.float32)
    weights = 1.0 / counts
    return weights * len(levels) / weights.sum()


# ── Helpers exposed to users building heads from physical units ──


def project_physical_to_norm(x: float, stats: dict[str, float]) -> float:
    """Map a single physical value to head-space via the column's affine norm.

    ±inf passes through unchanged so half-bounded heads stay half-bounded
    regardless of normalization.
    """
    if math.isinf(x):
        return x
    return (x - stats["shift"]) / stats["stretch"]


def bounds_from_norm(stats: dict[str, float]) -> tuple[float, float]:
    """Channel min/max projected into head-space using the column's norm stats."""
    shift, stretch = stats["shift"], stats["stretch"]
    return ((stats["min"] - shift) / stretch, (stats["max"] - shift) / stretch)


# ── Head stack ──────────────────────────────────────────────────


# Built after the concrete head classes so the lookup is exhaustive.
_KIND_TO_CLASS: dict[str, type[Head]] = {
    cls.kind: cls
    for cls in (RegressionHead, BinaryHead, BoundedHead, OrdinalHead)
}


def _float_to_json(x: float):
    """Encode floats so JSON can represent infinities portably."""
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _json_to_float(x) -> float:
    if isinstance(x, str):
        if x == "inf":
            return float("inf")
        if x == "-inf":
            return float("-inf")
        raise ValueError(f"Unrecognized float string in head config: {x!r}")
    return float(x)


class HeadStack(nn.Module):
    """
    Bundle of per-channel heads over one Y vector.

    Slices the model's raw [batch, raw_width] output by channel, applies
    each head's activation, and aggregates per-channel losses.
    """

    @beartype
    def __init__(self, heads: list[Head], channel_names: list[str]):
        super().__init__()
        if len(heads) != len(channel_names):
            raise ValueError(
                f"heads ({len(heads)}) and channel_names ({len(channel_names)}) "
                "must have the same length"
            )
        if len(set(channel_names)) != len(channel_names):
            raise ValueError(f"channel_names must be unique, got {channel_names}")
        self.heads = nn.ModuleList(heads)
        self.channel_names = list(channel_names)

        # Precompute slice into the raw model output per channel.
        offsets, cursor = [], 0
        for h in heads:
            offsets.append((cursor, cursor + h.width))
            cursor += h.width
        self._slices = offsets
        self.raw_width = cursor
        self.n_channels = len(heads)

    def z_for(self, z_raw: torch.Tensor, i: int) -> torch.Tensor:
        """Channel i's slice of the raw model output."""
        lo, hi = self._slices[i]
        return z_raw[..., lo:hi]

    def forward(self, z_raw: torch.Tensor) -> torch.Tensor:
        """`activate`, reachable through nn.Module.__call__ so a forward hook
        can observe the pre-activation z — the dynamics model calls the
        stack this way, and threshold calibration relies on it."""
        return self.activate(z_raw)

    def activate(self, z_raw: torch.Tensor) -> torch.Tensor:
        """[..., raw_width] -> [..., n_channels] activated state.

        For each head: when `head.use_ste=True` (default), apply the
        straight-through combine — returned value matches `decode(z)`
        (snapped/discrete where defined), gradient flows through
        `activate(z)` (smooth surrogate). When `head.use_ste=False`,
        return the smooth `activate(z)` directly, skipping the snap.
        Heads without a `decode` override have `decode == activate`, so
        STE is a no-op regardless of the flag.
        """
        outs = []
        for i in range(self.n_channels):
            z = self.z_for(z_raw, i)
            head = self.heads[i]
            y_soft = head.activate(z)
            if head.use_ste:
                y_hard = head.decode(z)
                outs.append(y_soft + (y_hard - y_soft).detach())
            else:
                outs.append(y_soft)
        return torch.stack(outs, dim=-1)

    def apply_residual(self, z_raw: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        """
        Add y_prev to the pre-activation z for residual-compatible channels.

        For width-1 heads this is just z[..., i] += y_prev[..., i] guarded by
        residual_compatible. The non-residual channels are left untouched.

        Args:
            z_raw:  [..., raw_width] pre-activation model output
            y_prev: [..., n_channels] last step's activated state

        Returns:
            z_raw with residual added on the residual-compatible channels.
        """
        out = z_raw.clone()
        for i, head in enumerate(self.heads):
            if not head.residual_compatible:
                continue
            if head.width != 1:
                raise ValueError(
                    f"Residual on width>1 head not supported (channel "
                    f"{self.channel_names[i]}, width={head.width})"
                )
            lo, hi = self._slices[i]
            out[..., lo:hi] = out[..., lo:hi] + y_prev[..., i:i + 1]
        return out

    def loss(self, y_state: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Mean across channels of per-channel head losses.

        Mean (not sum) so swapping head types does not silently change the
        overall ref_loss magnitude: an all-regression stack reproduces a
        blanket-MSE on the full Y tensor to floating-point precision.

        Args:
            y_state: [..., n_channels] activated model output
            y_true:  [..., n_channels] ground truth
        """
        total = y_state.new_zeros(())
        for i, head in enumerate(self.heads):
            total = total + head.loss(y_state[..., i], y_true[..., i])
        return total / self.n_channels

    # ── Serialization ──────────────────────────────────────────

    def to_config(self) -> dict[str, dict]:
        """Serialize to a JSON-safe {column_name: {kind, ...kwargs}} dict.

        The trained parameters live in state_dict; this captures the
        topology so load_model can rebuild a stack with the right shape
        before state_dict is restored.
        """
        out: dict[str, dict] = {}
        for name, head in zip(self.channel_names, self.heads):
            entry = {"kind": head.kind}
            entry.update(head.config())
            out[name] = entry
        return out

    @classmethod
    def from_config(
        cls, data: dict[str, dict], channel_names: list[str]
    ) -> "HeadStack":
        """Inverse of to_config.

        Channels missing from `data` default to RegressionHead so older
        configs (which only stored non-default entries) keep loading.
        Channel names present in `data` but not in `channel_names` raise.
        """
        unknown = set(data) - set(channel_names)
        if unknown:
            raise ValueError(
                f"head config references columns not in channel_names: {sorted(unknown)}"
            )
        heads: list[Head] = []
        for name in channel_names:
            entry = data.get(name, {"kind": "regression"})
            kind = entry["kind"]
            if kind not in _KIND_TO_CLASS:
                raise ValueError(
                    f"Unknown head kind '{kind}' for channel '{name}'. "
                    f"Expected one of: {sorted(_KIND_TO_CLASS)}."
                )
            kwargs = {k: v for k, v in entry.items() if k != "kind"}
            if "lo" in kwargs:
                kwargs["lo"] = _json_to_float(kwargs["lo"])
            if "hi" in kwargs:
                kwargs["hi"] = _json_to_float(kwargs["hi"])
            for key in ("level_values", "decision_thresholds", "level_weights"):
                if key in kwargs:
                    kwargs[key] = torch.tensor(kwargs[key], dtype=torch.float32)
            heads.append(_KIND_TO_CLASS[kind](**kwargs))
        return cls(heads, channel_names)
