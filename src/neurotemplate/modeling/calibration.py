"""
Two data-driven settings for staged (binary / ordinal) channels that the
optimiser cannot be trusted with, set from the data instead:

  balance_level_weights        — before training: each levelled head's
                                 loss weights := inverse sqrt training frequency
                                 of its levels (`cfg.balance_levels`).
  calibrate_decision_thresholds — after training: each ordinal head's
                                 decision thresholds := the balanced-
                                 accuracy optimum on dev rollouts
                                 (`cfg.calibrate_thresholds`).

Why the thresholds are not trained: gradient descent on MSE places the
snap where the model fires at roughly the base rate, which on a 2%-
positive channel is a whisker away from "never". Moving the same scalar
against balanced accuracy instead — no retraining — was the largest
single effect in the head bake-off (+0.11 on training data, +0.17 on
dev). The pipeline already selects on dev (that is what `patience`
does), so calibrating one more scalar per threshold there is not a new
category of choice.

The z that is calibrated against comes from the *rollout* on dev at
`dev_pred_steps`, the same regime the dev loss is measured in, not from
teacher-forced one-step prediction. That is deliberate: the surrogate's
job is the rollout, and a threshold tuned on true-history z is applied
across a distribution shift when the model eats its own output. The
calibrated rule changes what is fed back afterwards, so the tuned
thresholds are checked with a second dev rollout: a channel whose
realized balanced accuracy did not improve keeps its old thresholds and
says so. That is the one guard; the reportcard's level metrics on the
test split remain the real check.

A level the dev split never visits cannot be calibrated for (no recall
to maximise), so such a channel keeps its current thresholds and says
so in a warning rather than quietly optimising the other levels.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from pandas import DataFrame

from neurotemplate._typing import beartype
from neurotemplate.common.data_utils import get_sw_dict_multi, pool_dfs
from neurotemplate.common.metrics import level_metrics
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.heads import HeadStack, OrdinalHead, inverse_sqrt_frequency_weights


# Thresholds are searched over this many quantiles of the dev z. Balanced
# accuracy is a step function of the threshold with at most one step per
# sample, so the search is exact once candidates outnumber the rare level's
# occurrences, and 201 already does on any dev split worth calibrating on.
N_CANDIDATES = 201


@dataclass(frozen=True)
class Calibration:
    """What calibration did to one ordinal channel: the thresholds it ended
    on and the dev balanced accuracy realized in a rollout before and after.
    `after == before` with unchanged thresholds means the tuned rule was
    tried and did not hold up in the rollout."""
    thresholds: list[float]
    dev_before: float
    dev_after: float


@beartype
def balance_level_weights(head_stack: HeadStack, train_dfs: list[DataFrame]) -> None:
    """Set every levelled head's `level_weights` to the inverse square-root frequency of
    its levels in the (normalized) training data. Heads without levels are
    left alone."""
    pool = pool_dfs(train_dfs)
    for name, head in zip(head_stack.channel_names, head_stack.heads):
        levels = head.levels()
        if levels is None:
            continue
        y = torch.tensor(pool[name].to_numpy(), dtype=torch.float32, device=levels.device)
        head.level_weights.copy_(inverse_sqrt_frequency_weights(y, levels))


def collect_rollout_z(dyn_sys, data: dict, in_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll `dyn_sys` out on `data` and return the pre-activation z it produced
    at every scored step, [batch, steps, raw_width], with the matching truth
    [batch, steps, ny]. Read through a forward hook on the head stack, so the
    System's output contract is untouched."""
    head_stack = dyn_sys.nodes[0].callable.head_stack
    seen: list[torch.Tensor] = []
    handle = head_stack.register_forward_hook(lambda _m, inp, _out: seen.append(inp[0].detach()))
    try:
        with torch.no_grad():
            out = dyn_sys(data)
    finally:
        handle.remove()
    # The rollout runs one step past the last truth (yn has T+1 entries);
    # drop that final z so z[:, t] lines up with Y[:, in_steps + t].
    z = torch.stack(seen, dim=1)[:, :-1]
    return z, out["Y"][:, in_steps:]


def _balanced_accuracy(z: torch.Tensor, thresholds: torch.Tensor,
                       true_idx: torch.Tensor, n_levels: int) -> float:
    pred_idx = (z.unsqueeze(-1) > thresholds).sum(dim=-1)
    return level_metrics(pred_idx, true_idx, list(range(n_levels)), "").balanced_accuracy


@beartype
def tune_thresholds(z: torch.Tensor, true_idx: torch.Tensor,
                    init: torch.Tensor) -> torch.Tensor:
    """The K-1 thresholds on 1-D `z` that maximise balanced accuracy against
    `true_idx`, starting from `init`. Coordinate ascent: each threshold in
    turn tries every quantile candidate between its neighbours and keeps the
    best, until a full sweep changes nothing. Exact for K=2 (one threshold,
    one pass); for K>2 a local optimum, which is what a 2-D grid found too.
    A candidate has to be strictly better to displace `init`, so a split the
    data cannot improve keeps the midpoints."""
    n_levels = len(init) + 1
    candidates = torch.unique(torch.quantile(
        z, torch.linspace(0.0, 1.0, N_CANDIDATES, device=z.device),
    ))
    thresholds = init.clone()
    best = _balanced_accuracy(z, thresholds, true_idx, n_levels)
    improved = True
    while improved:
        improved = False
        for k in range(len(thresholds)):
            lo = thresholds[k - 1] if k > 0 else -torch.inf
            hi = thresholds[k + 1] if k + 1 < len(thresholds) else torch.inf
            for c in candidates:
                if not (lo < c < hi):
                    continue
                trial = thresholds.clone()
                trial[k] = c
                score = _balanced_accuracy(z, trial, true_idx, n_levels)
                if score > best + 1e-9:
                    best, thresholds, improved = score, trial, True
    return thresholds


def _rollout_z(dyn_sys, cfg, dev_dfs, nm_mapping):
    data = get_sw_dict_multi(
        dfs=dev_dfs, cfg=cfg, nm_mapping=nm_mapping,
        steps=cfg.in_steps + cfg.dev_pred_steps,
    )
    data.pop("source_idx")
    was_training = dyn_sys.training
    dyn_sys.eval()
    z_all, y_true = collect_rollout_z(dyn_sys, data, cfg.in_steps)
    dyn_sys.train(was_training)
    return z_all, y_true


@beartype
def calibrate_decision_thresholds(
    dyn_sys,
    cfg: ModelingConfig,
    dev_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
) -> dict[str, Calibration]:
    """Move every ordinal head's `decision_thresholds` to the balanced-
    accuracy optimum on dev rollouts at `cfg.dev_pred_steps`, in place on
    the System's own head stack, then confirm with a second rollout that the
    new rule really scores better once it is the one being fed back —
    reverting any channel where it does not. Returns {channel: Calibration}
    for every ordinal channel that could be calibrated; a channel with a
    level dev never visits, or with `use_ste=False` (its snap never runs),
    is skipped with a warning. Empty without ordinal heads."""
    head_stack = dyn_sys.nodes[0].callable.head_stack
    if head_stack is None or not any(isinstance(h, OrdinalHead) for h in head_stack.heads):
        return {}

    ordinal = []
    for i, (name, head) in enumerate(zip(head_stack.channel_names, head_stack.heads)):
        if not isinstance(head, OrdinalHead):
            continue
        if not head.use_ste:
            # The loop feeds back the smooth activation, so decode — and any
            # threshold it snaps at — never runs. Nothing to calibrate.
            warnings.warn(
                f"{name!r}: OrdinalHead(use_ste=False) feeds back its smooth "
                "activation and never snaps, so its decision thresholds have "
                "no effect and are not calibrated.",
                stacklevel=2,
            )
            continue
        ordinal.append((i, name, head))
    if not ordinal:
        return {}

    def realized(z_all, y_true, i, head):
        """Dev balanced accuracy of the head's current rule on this rollout."""
        z = head_stack.z_for(z_all, i).reshape(-1)
        true_idx = head.level_index(y_true[..., i]).reshape(-1)
        return _balanced_accuracy(z, head.decision_thresholds, true_idx, head.n_levels)

    z_all, y_true = _rollout_z(dyn_sys, cfg, dev_dfs, nm_mapping)
    tuned: dict[str, tuple[torch.Tensor, float]] = {}   # name -> (old thresholds, before)
    for i, name, head in ordinal:
        z = head_stack.z_for(z_all, i).reshape(-1)
        true_idx = head.level_index(y_true[..., i]).reshape(-1)
        missing = [k for k in range(head.n_levels) if not (true_idx == k).any()]
        if missing:
            warnings.warn(
                f"{name!r}: dev split never visits level(s) {missing} of "
                f"{head.level_values.tolist()}, so its decision thresholds "
                "cannot be calibrated and stay at "
                f"{head.decision_thresholds.tolist()}.",
                stacklevel=2,
            )
            continue
        old = head.decision_thresholds.clone()
        before = _balanced_accuracy(z, old, true_idx, head.n_levels)
        head.decision_thresholds.copy_(tune_thresholds(z, true_idx, old))
        tuned[name] = (old, before)

    if not tuned:
        return {}

    # The tuned rule now shapes the fed-back state; score it as it will run.
    z_all, y_true = _rollout_z(dyn_sys, cfg, dev_dfs, nm_mapping)
    out: dict[str, Calibration] = {}
    for i, name, head in ordinal:
        if name not in tuned:
            continue
        old, before = tuned[name]
        after = realized(z_all, y_true, i, head)
        if after < before:
            warnings.warn(
                f"{name!r}: the calibrated decision thresholds "
                f"{head.decision_thresholds.tolist()} scored {after:.3f} balanced "
                f"accuracy on dev rollouts against {before:.3f} at "
                f"{old.tolist()} once fed back, so they were reverted. The z "
                "distribution shifts when the model eats its own output; this "
                "channel's decision rule was not recoverable that way.",
                stacklevel=2,
            )
            head.decision_thresholds.copy_(old)
            after = before
        out[name] = Calibration(head.decision_thresholds.tolist(), before, after)
    return out
