"""
The training loop, for a model and a policy alike.

Each step draws a batch, descends the loss and clips the gradient. Every
`eval_every` steps it scores the dev set, keeps the best weights so far and logs.
It ends with the best weights loaded.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch

from neurotemplate.common.progress import Progress


@dataclass
class Fitted:
    best_state: dict
    best_dev: float


def move_to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


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


def fit(problem, optimizer, next_batch: Callable[[], dict], dev_batches: Iterable[dict], *,
        iters: int, eval_every: int, clip: float, patience: int, warmup: int, device,
        progress: Progress, logger=None) -> Fitted:
    """Train `problem` for `iters` steps, leaving its best weights loaded."""
    if logger is not None:
        logger.log_weights(problem)
    best_dev, best_state = float("inf"), deepcopy(problem.state_dict())
    bad = 0
    losses = []
    progress.start(iters)
    for step in range(1, iters + 1):
        problem.train()
        output = problem(move_to_device(next_batch(), device))
        loss = output["train_loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(problem.parameters(), clip)
        optimizer.step()
        # Detached: a held loss holds its whole graph.
        output = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
        losses.append(output["train_loss"])
        if step % eval_every:
            continue

        evaluation = step // eval_every - 1
        output["mean_train_loss"] = torch.mean(torch.stack(losses))
        losses = []
        output = {**output, **_score_dev(problem, dev_batches, device)}
        dev = output["mean_dev_loss"].item()
        if dev < best_dev:
            best_dev, best_state = dev, deepcopy(problem.state_dict())
            bad = 0
        elif evaluation > warmup:
            bad += 1
        if logger is not None:
            logger.log_metrics(output, step=evaluation)
        progress.update(step, output["mean_train_loss"].item(), dev)
        if bad > patience:
            progress.stopped_early()
            break
    progress.finish()
    problem.load_state_dict(best_state)
    return Fitted(best_state, best_dev)
