"""
The training loop, for a model and a policy alike.

A fixed budget of `iters` steps, as in the amortized-weights paper's protocol:

  - **Schedule.** Every parameter group's rate ramps linearly from about zero to its
    peak over the first `RAMP` of the steps, then follows a cosine down to
    `FINAL_SHARE` of that peak. A group with its own peak (`physical_lr`) follows the
    same curve.
  - **Clip.** The loss is divided by its value on the first batch, so it starts at 1
    whatever its units, and the gradient norm is clipped at `CLIP`. Adam is
    scale-invariant, so the division changes only what the clip means.
  - **Dev.** `EVALUATIONS` times, evenly spaced and at the last step, the dev set is
    scored and the best weights so far are kept. Training ends with them loaded.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch

from neurotemplate.common.progress import Progress

#: Share of the steps the rate ramps up over.
RAMP = 0.05
#: Where the cosine ends, as a share of the peak: the paper's 5e-3 to 5e-5.
FINAL_SHARE = 0.01
#: Gradient-norm clip, on the loss divided by its first-batch value.
CLIP = 1.0
#: How many times a run scores the dev set.
EVALUATIONS = 20


@dataclass
class Fitted:
    best_state: dict
    best_dev: float


def move_to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def schedule(iters: int) -> Callable[[int], float]:
    """The rate at step `i` (0-based) as a share of the peak: a linear ramp, then a
    cosine from 1 to `FINAL_SHARE`."""
    ramp = max(int(RAMP * iters), 1)

    def share(i: int) -> float:
        if i < ramp:
            return (i + 1) / ramp
        progress = (i - ramp) / max(iters - ramp, 1)
        return FINAL_SHARE + (1 - FINAL_SHARE) * 0.5 * (1 + math.cos(math.pi * progress))
    return share


def _score_dev(problem, dev_batches: Iterable[dict], device) -> dict:
    """The dev set's last batch's output, with the mean over its batches as `mean_dev_loss`."""
    problem.eval()
    with torch.no_grad():
        losses = []
        for batch in dev_batches:
            output = problem(move_to_device(batch, device))
            losses.append(output["dev_loss"])
    output["mean_dev_loss"] = torch.mean(torch.stack(losses))
    return output


def _loss_scale(loss: torch.Tensor) -> float:
    """The first batch's training loss, which every step's loss is divided by."""
    scale = loss.item()
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            f"The training loss on the first batch is {scale}, before any step. Training "
            "divides by it so that one clip and one rate fit every project, which needs a "
            "positive, finite loss. A zero loss has nothing to descend; a non-finite one "
            "comes from a rollout that blows up or reads a missing value.")
    return scale


def fit(problem, optimizer, train_batches: Iterator[dict], dev_batches: Iterable[dict], *,
        iters: int, device, progress: Progress, logger=None) -> Fitted:
    """Train `problem` for `iters` steps, leaving its best weights loaded."""
    if logger is not None:
        logger.log_weights(problem)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(iters))
    eval_every = max(iters // EVALUATIONS, 1)
    best_dev, best_state = float("inf"), deepcopy(problem.state_dict())
    scale = None
    losses = []
    progress.start(iters)
    for step in range(1, iters + 1):
        problem.train()
        output = problem(move_to_device(next(train_batches), device))
        loss = output["train_loss"]
        if scale is None:
            scale = _loss_scale(loss)
        optimizer.zero_grad()
        (loss / scale).backward()
        torch.nn.utils.clip_grad_norm_(problem.parameters(), CLIP)
        optimizer.step()
        scheduler.step()
        # Detached: a held loss holds its whole graph.
        output = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
        losses.append(output["train_loss"])
        if step % eval_every and step != iters:
            continue

        output["mean_train_loss"] = torch.mean(torch.stack(losses))
        losses = []
        output = {**output, **_score_dev(problem, dev_batches, device)}
        dev = output["mean_dev_loss"].item()
        if dev < best_dev:
            best_dev, best_state = dev, deepcopy(problem.state_dict())
        if logger is not None:
            logger.log_metrics(output, step=step)
        progress.update(step, output["mean_train_loss"].item(), dev)
    progress.finish()
    problem.load_state_dict(best_state)
    return Fitted(best_state, best_dev)

