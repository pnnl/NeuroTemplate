"""
Training loop and utilities.

Uses a standard Neuromancer Problem (no subclass needed). Physics losses
are proper Neuromancer Loss objects inside PenaltyLoss, so they flow
through Problem.forward() and appear in the Trainer's output dict with
the correct labels (e.g., 'train_grad_penalty'). See loss.py for which
terms are train-only.
"""

import os
import random
from datetime import datetime
from typing import Optional, Union

import numpy as np
import torch
from pandas import DataFrame

from neurotemplate._typing import beartype

from neurotemplate.common.data_utils import NormStats, get_sw_dict_multi, model_window_keys
from neurotemplate.common.sampler import WindowSampler, fixed_batches
from neurotemplate.common.fit import fit
from neurotemplate.common.progress import Progress
from neurotemplate.common.tracking import RunLog, save_checkpoint
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.contract import contract_model
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.loss import build_loss
from neurotemplate.modeling.bounded import physical_parameters
from neurotemplate.modeling.config import ModelingConfig

from neurotemplate.nm.problem import Problem
from neurotemplate.nm.system import System


# ── Seed / Device helpers ────────────────────────────────────────

@beartype
def set_seeds(seed: int) -> int:
    if seed < 0:
        seed = int(datetime.now().timestamp())
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@beartype
def resolve_device(requested: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve the torch device to use, and make it torch's default device
    (`torch.set_default_device`) for the rest of the process: tensors the
    loaders and the rollout create without an explicit device land there.

    Raises RuntimeError if a specific device is requested but unavailable.
    """
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device, but MPS is unavailable.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # CPU is torch's own default, and installing it anyway costs a Python
    # hook on every torch call (~15% of a grey-box training step); None also
    # clears a device an earlier call set.
    torch.set_default_device(None if device.type == "cpu" else device)
    return device


def set_threads(threads: Optional[int]) -> None:
    """Torch's CPU thread count for the rest of the process; None means one per
    core. Process-wide, which is why each run sets it from its own config rather
    than any library code deciding it. Importing this package starts OpenMP at
    one thread (see `neurotemplate/__init__.py`), so None has to ask for the rest."""
    torch.set_num_threads(os.cpu_count() if threads is None else threads)


# ── Build Trainer ────────────────────────────────────────────────

@beartype
def build_trainer(
    cfg: ModelingConfig,
    train_dfs: list[DataFrame],
    dev_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
    grad_masks: dict[str, torch.Tensor],
    norm_stats: NormStats,
    head_stack: Optional[HeadStack] = None,
    block_spec: Optional[dict[str, dict]] = None,
    logger: Optional[RunLog] = None,
) -> tuple:
    """
    The Problem, its optimizer, the training draws and the dev batches.

    The Problem contains:
      - A SystemPreview node for autoregressive rollout
      - A PenaltyLoss with tracking + physics objectives

    No custom Problem subclass needed — physics losses are proper
    Neuromancer Loss objects inside PenaltyLoss.
    """
    # SystemPreview with 'yn' feedback loop for autoregressive rollout
    system = get_dynamics(
        cfg, nm_mapping, norm_stats, head_stack=head_stack, block_spec=block_spec,
    ).to(cfg.device)

    # The loss reads the head stack the model owns: a Stepper rescales its own copy.
    head_stack = getattr(contract_model(system), "head_stack", None)

    # Single PenaltyLoss with all objectives:
    #   - Tracking: ref_loss
    #   - Physics:  grad_penalty (Loss object, train-only)
    loss = build_loss(
        cfg=cfg,
        system=system,
        grad_masks=grad_masks,
        nm_mapping=nm_mapping,
        norm_stats=norm_stats,
        head_stack=head_stack,
    )

    # Standard Neuromancer Problem — no subclass needed
    problem = Problem([system], loss).to(cfg.device)

    optimizer = torch.optim.Adam(_parameter_groups(problem, cfg), lr=cfg.lr)

    steps = cfg.history_steps + cfg.horizon_steps
    train_batches = WindowSampler(train_dfs, model_window_keys(cfg, nm_mapping, steps), steps,
                                  cfg.batch_size, cfg.seed, cfg.device)
    dev_data = get_sw_dict_multi(dev_dfs, cfg, nm_mapping, steps, cfg.dev_stride_steps)
    dev_data.pop("source_idx")
    dev_batches = fixed_batches(dev_data, cfg.batch_size, "dev")
    if cfg.lr > 0:
        check_parameters_reach_loss(problem, next(train_batches))

    return problem, optimizer, train_batches, dev_batches


def _parameter_groups(problem: Problem, cfg: ModelingConfig) -> list[dict]:
    """Network weights at `cfg.lr`; boxed physical parameters at
    `cfg.physical_lr` (see `bounded.physical_parameters`)."""
    physical = physical_parameters(problem)
    physical_ids = {id(p) for p in physical}
    rest = [p for p in problem.parameters() if id(p) not in physical_ids]
    groups = [{"params": rest}]
    if physical:
        groups.append({"params": physical, "lr": cfg.physical_lr})
    return groups


@beartype
def check_parameters_reach_loss(problem: Problem, batch: dict) -> None:
    """Raise if a trainable parameter has no gradient path to the training
    loss on `batch`: it would never move, though `lr > 0` says it should.

    One forward and one backward pass. A parameter meant to stay put sets
    `requires_grad=False` and is not checked. A path whose gradient is merely
    zero here (a saturated clamp) is not this check's: it may move later.
    """
    trainable = [(name, p) for name, p in problem.named_parameters() if p.requires_grad]
    loss = problem(batch)[f"{batch['name']}_loss"]
    grads = torch.autograd.grad(loss, [p for _, p in trainable], allow_unused=True)
    cut = [name for (name, _), grad in zip(trainable, grads) if grad is None]
    if cut:
        raise ValueError(
            f"{len(cut)} trainable parameter(s) have no gradient path to the training "
            f"loss, so they would never train: {cut}. Something between them and the "
            "output leaves the autograd graph (numpy, .item(), .detach(), an integer "
            "op), or they are not used. Freeze one deliberately with "
            "requires_grad_(False), or set lr: 0 to keep the model as built.")


# ── Train ────────────────────────────────────────────────────────

@beartype
def fit_dynamics(
    cfg: ModelingConfig,
    train_dfs: list[DataFrame],
    dev_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
    grad_masks: dict[str, torch.Tensor],
    norm_stats: NormStats,
    head_stack: Optional[HeadStack] = None,
    block_spec: Optional[dict[str, dict]] = None,
    logger: Optional[RunLog] = None,
) -> tuple[System, Optional[float]]:
    """Train one model, leaving its best weights loaded, and log its checkpoint.

    At `lr: 0` there is nothing to train: the model is kept as built, and there
    is no dev loss to report (None).

    Returns (trained_system, best_dev_loss).
    """
    problem, optimizer, train_batches, dev_batches = build_trainer(
        cfg=cfg,
        train_dfs=train_dfs,
        dev_dfs=dev_dfs,
        nm_mapping=nm_mapping,
        grad_masks=grad_masks,
        norm_stats=norm_stats,
        head_stack=head_stack,
        block_spec=block_spec,
        logger=logger,
    )
    system = problem.nodes[0]
    if cfg.lr == 0:
        if cfg.progress != "silent":
            print("lr: 0: nothing to train; the model is kept as built.")
        if logger is not None:
            logger.log_weights(problem)
            save_checkpoint(logger, problem.state_dict())
        return system, None
    fitted = fit(problem, optimizer, train_batches, dev_batches,
                 iters=cfg.iters, device=cfg.device, progress=Progress(cfg.progress),
                 logger=logger)
    if logger is not None:
        save_checkpoint(logger, fitted.best_state)
    return system, fitted.best_dev
