"""
Two data-driven settings for staged (binary / ordinal) channels that the
optimiser cannot be trusted with, set from the data instead:

  balance_level_weights         — before training: each levelled head's
                                  loss weights := inverse sqrt training
                                  frequency of its levels (`cfg.balance_levels`).
  calibrate_decision_thresholds — after training: each ordinal head's
                                  decision thresholds := share-matched
                                  against dev rollouts (`cfg.calibrate_thresholds`).

Why the thresholds are not trained: gradient descent on MSE places the
snap where the model fires at roughly the base rate, which on a 2%-
positive channel is a whisker away from "never" — see heads.py's "Rare
levels" section for the mechanism. Moving the threshold after training,
against a criterion that actually notices the rare level, recovers most
of that without retraining.

Share-matching sets each threshold so the LEVEL SHARE the model predicts
equals the level share the data actually has — a quantile of z, closed
form, not a search. That is also what makes it the right criterion for a
surrogate rather than a balanced-accuracy argmax: balanced accuracy is
the crossing point of two CONDITIONAL densities, p(z | rare) and
p(z | common), and the rare one is estimated from a handful of samples —
the same samples a changed decision rule relocates in the next rollout
(see the loop below). Share-matching reads a quantile of the MARGINAL
distribution of z instead, which most of the data does not move when the
rule changes — a smaller, better-behaved exposure to the same feedback
loop, not a different problem.

The z calibration is measured against comes from the *rollout* on dev at
`dev_pred_steps` — the regime the dev loss is measured in and the one the
surrogate is judged in, not teacher-forced one-step prediction.

A level the dev split never visits cannot be calibrated for (no target
share to match), so such a channel keeps its current thresholds and says
so in a warning. Likewise a channel whose `OrdinalHead` has
`use_ste=False`: its `decode()` never runs in the loop, so there is
nothing a threshold could change.

Every rollout tried is kept, not just the thresholds that were: the
reportcard renders the full trace (`Calibration.trace`) so the search is
visible, not just its outcome.
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


# Upper bound on "set thresholds, roll out, see what share they actually
# produced" passes per run. Each pass is one dev rollout — cheap enough to
# spend a handful on. A channel that has not settled by then keeps
# whichever pass matched the true shares best, not necessarily the last.
MAX_CALIBRATION_ITERATIONS = 5

# Two threshold vectors this close (head-space) count as the same rule —
# calibration stops moving a channel that has converged rather than
# spending its remaining iterations reproducing the same rollout.
CONVERGED_ATOL = 1e-4


@dataclass(frozen=True)
class CalibrationStep:
    """One dev rollout in the threshold search for one channel.

    `error` — the L1 distance between the level shares this rollout
    actually produced and the true level shares — is what the search
    minimises. `balanced_accuracy` rides along for context on the
    reportcard; it is not the criterion being optimised.
    """
    thresholds: list[float]
    pred_shares: list[float]
    error: float
    balanced_accuracy: float


@dataclass(frozen=True)
class Calibration:
    """What calibration did to one ordinal channel.

    `thresholds` / `true_shares` mirror `trace[kept]` and the fixed target
    it was matched against, for convenience. `kept` is not always the last
    entry — a later pass can overshoot, and the lowest-error one is kept.
    """
    thresholds: list[float]
    true_shares: list[float]
    trace: tuple[CalibrationStep, ...]
    kept: int


@beartype
def balance_level_weights(head_stack: HeadStack, train_dfs: list[DataFrame]) -> None:
    """Set every levelled head's `level_weights` to the inverse square-root
    frequency of its levels in the (normalized) training data. Heads
    without levels are left alone."""
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
def share_match_thresholds(z: torch.Tensor, true_idx: torch.Tensor, n_levels: int) -> torch.Tensor:
    """The K-1 thresholds on 1-D `z` that make each level's PREDICTED share
    equal its TRUE share in `true_idx`: the k-th threshold is the quantile
    of `z` at the cumulative true share up to level k. An order statistic,
    not a search — exact for any K, with no local optimum to land in."""
    true_shares = torch.bincount(true_idx, minlength=n_levels).float() / true_idx.numel()
    cum_share = torch.cumsum(true_shares, dim=0)[:-1]
    return torch.quantile(z, cum_share)


def _score(z: torch.Tensor, true_idx: torch.Tensor, thresholds: torch.Tensor,
          n_levels: int) -> CalibrationStep:
    """What `thresholds` actually produced on this rollout: the level
    shares, the share-matching error, and balanced accuracy for context."""
    pred_idx = (z.unsqueeze(-1) > thresholds).sum(dim=-1)
    pred_shares = torch.bincount(pred_idx, minlength=n_levels).float() / pred_idx.numel()
    true_shares = torch.bincount(true_idx, minlength=n_levels).float() / true_idx.numel()
    error = (pred_shares - true_shares).abs().sum().item()
    ba = level_metrics(pred_idx, true_idx, list(range(n_levels)), "").balanced_accuracy
    return CalibrationStep(thresholds.tolist(), pred_shares.tolist(), error, ba)


@beartype
def calibrate_decision_thresholds(
    dyn_sys,
    cfg: ModelingConfig,
    dev_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
    max_iterations: int = MAX_CALIBRATION_ITERATIONS,
) -> dict[str, Calibration]:
    """Move every ordinal head's `decision_thresholds` to the share-matched
    rule found over up to `max_iterations` dev rollouts, in place on the
    System's own head stack. Returns {channel: Calibration} for every
    ordinal channel that could be calibrated; a channel with a level dev
    never visits, or with `use_ste=False` (its snap never runs), is
    skipped with a warning. Empty without ordinal heads."""
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

    state = {
        name: {"thresholds": head.decision_thresholds.clone(), "trace": [],
              "true_shares": None, "active": True}
        for _, name, head in ordinal
    }

    # decode()'s output is what gets fed back as the next step's input, so
    # the thresholds shape the z the NEXT threshold would be chosen from —
    # this is a fixed point, not a one-shot lookup. Iterate a bounded number
    # of "set thresholds, roll out, see what share they actually produced"
    # passes and keep whichever one matched the true shares best, rather
    # than trusting wherever the last pass happened to land.
    for _ in range(max_iterations):
        if not any(s["active"] for s in state.values()):
            break
        for _, name, head in ordinal:
            head.decision_thresholds.copy_(state[name]["thresholds"])
        z_all, y_true = _rollout_z(dyn_sys, cfg, dev_dfs, nm_mapping)
        for i, name, head in ordinal:
            s = state[name]
            if not s["active"]:
                continue
            z = head_stack.z_for(z_all, i).reshape(-1)
            true_idx = head.level_index(y_true[..., i]).reshape(-1)
            true_counts = torch.bincount(true_idx, minlength=head.n_levels)
            if (true_counts == 0).any():
                missing = [k for k in range(head.n_levels) if true_counts[k] == 0]
                warnings.warn(
                    f"{name!r}: dev split never visits level(s) {missing} of "
                    f"{head.level_values.tolist()}, so its decision thresholds "
                    f"cannot be calibrated and stay at {s['thresholds'].tolist()}.",
                    stacklevel=2,
                )
                s["active"] = False
                continue
            s["true_shares"] = (true_counts.float() / true_idx.numel()).tolist()
            s["trace"].append(_score(z, true_idx, s["thresholds"], head.n_levels))
            candidate = share_match_thresholds(z, true_idx, head.n_levels)
            if (candidate[1:] <= candidate[:-1]).any():
                # A tie in z's distribution collapsed two thresholds together;
                # nothing more to try.
                s["active"] = False
                continue
            if torch.allclose(candidate, s["thresholds"], atol=CONVERGED_ATOL):
                s["active"] = False
                continue
            s["thresholds"] = candidate

    out: dict[str, Calibration] = {}
    for _, name, head in ordinal:
        s = state[name]
        if not s["trace"]:
            continue
        kept = min(range(len(s["trace"])), key=lambda j: s["trace"][j].error)
        head.decision_thresholds.copy_(torch.tensor(s["trace"][kept].thresholds))
        out[name] = Calibration(
            thresholds=s["trace"][kept].thresholds,
            true_shares=s["true_shares"],
            trace=tuple(s["trace"]),
            kept=kept,
        )
    return out
