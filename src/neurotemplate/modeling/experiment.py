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
from dataclasses import replace
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from pandas import DataFrame

from neurotemplate._typing import beartype

from neurotemplate.common.data_utils import NormStats, get_loader
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.loss import build_loss
from neurotemplate.modeling.rc import physical_parameters
from neurotemplate.modeling.config import ModelingConfig

from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.system import System
from neuromancer.loggers import MLFlowLogger


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
def resolve_device(requested: Optional[str] = None) -> torch.device:
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
    torch.set_default_device(device)
    return device


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
    logger: Optional[MLFlowLogger] = None,
) -> Trainer:
    """
    Construct a Neuromancer Trainer with a standard Problem.

    The Problem contains:
      - A SystemPreview node for autoregressive rollout
      - A PenaltyLoss with tracking + physics objectives

    No custom Problem subclass needed — physics losses are proper
    Neuromancer Loss objects inside PenaltyLoss.
    """
    if cfg.verbose:
        print("Building dynamics system...")

    # SystemPreview with 'yn' feedback loop for autoregressive rollout
    system = get_dynamics(
        cfg, nm_mapping, head_stack=head_stack, block_spec=block_spec,
        norm_stats=norm_stats,
    ).to(cfg.device)

    # The raw SSMGrad model lives inside the System's Node.
    # Physics losses need direct access to it for Jacobians and
    # training-mode checks. The loss must read the head stack the model
    # owns (get_dynamics attaches its own copy), not the caller's — one
    # stack, so nothing calibrated on it can go stale in the loss.
    raw_model = system.nodes[0].callable
    head_stack = raw_model.head_stack

    if cfg.verbose:
        print("Building loss (tracking + physics)...")

    # Single PenaltyLoss with all objectives:
    #   - Tracking: ref_loss
    #   - Physics:  grad_penalty (Loss object, train-only)
    loss = build_loss(
        cfg=cfg,
        raw_model=raw_model,
        system=system,
        grad_masks=grad_masks,
        head_stack=head_stack,
    )

    if cfg.verbose:
        print("Building Problem...")

    # Standard Neuromancer Problem — no subclass needed
    problem = Problem([system], loss).to(cfg.device)

    if cfg.verbose:
        # problem.show()
        print("Building optimizer...")

    optimizer = torch.optim.Adam(_parameter_groups(problem, cfg), lr=cfg.lr)

    if cfg.verbose:
        print("Building data loaders...")

    train_loader = get_loader(
        dfs=train_dfs, cfg=cfg, nm_mapping=nm_mapping,
        name="train",
        steps=cfg.in_steps + cfg.train_pred_steps,
    )
    dev_loader = get_loader(
        dfs=dev_dfs, cfg=cfg, nm_mapping=nm_mapping,
        name="dev",
        steps=cfg.in_steps + cfg.dev_pred_steps,
    )

    if cfg.verbose:
        print("Building Trainer...")

    trainer = Trainer(
        problem=problem,
        train_data=train_loader,
        dev_data=dev_loader,
        optimizer=optimizer,
        logger=logger,
        patience=cfg.patience,
        warmup=cfg.warmup,
        epochs=cfg.epochs,
        device=cfg.device,
        epoch_verbose=cfg.epoch_verbose,
        eval_metric="mean_dev_loss",
        lr_scheduler=True,
    )

    return trainer


def _parameter_groups(problem: Problem, cfg: ModelingConfig) -> list[dict]:
    """Network weights at `cfg.lr`; a 2R2C block's physical scalars at
    `cfg.rc_branch_lr` (see `rc.physical_parameters`)."""
    physical = physical_parameters(problem)
    physical_ids = {id(p) for p in physical}
    rest = [p for p in problem.parameters() if id(p) not in physical_ids]
    groups = [{"params": rest}]
    if physical:
        groups.append({"params": physical, "lr": cfg.rc_branch_lr})
    return groups


# ── Child Logger ─────────────────────────────────────────────────

@beartype
def create_child_logger(
    cfg: ModelingConfig,
    global_logger: MLFlowLogger,
    pred_steps: int,
    curriculum_step: int,
    save_weights: bool = False,
) -> MLFlowLogger:
    """Create a nested MLflow child run for a curriculum stage.

    The child logs the stage's own config — its rollout length and run name —
    built as a copy, so the caller's `cfg` is never touched.
    """
    child_savedir = f"{cfg.savedir}/pred_steps_{pred_steps}"
    # MLFlowLogger.clean_up() unconditionally rmtree's savedir, but never
    # creates it; for a curriculum stage that writes no local artifact the
    # dir is absent and clean_up() raises FileNotFoundError. Create it up
    # front so the stage tears down cleanly.
    os.makedirs(child_savedir, exist_ok=True)

    stage_cfg = replace(cfg, train_pred_steps=pred_steps,
                        run=f"Curriculum Step {curriculum_step}")
    child_logger = MLFlowLogger(
        stage_cfg,
        savedir=child_savedir,
        stdout=global_logger.stdout,
        verbosity=cfg.epoch_verbose if cfg.verbose else 0,
        logout=global_logger.logout,
        nested=True,
        save_weights=save_weights,
    )

    child_logger._mlflow.log_param("curriculum_step", curriculum_step)
    return child_logger


# ── Train with Curriculum ────────────────────────────────────────

@beartype
def train_with_curriculum(
    cfg: ModelingConfig,
    train_dfs: list[DataFrame],
    dev_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
    grad_masks: dict[str, torch.Tensor],
    norm_stats: NormStats,
    head_stack: Optional[HeadStack] = None,
    block_spec: Optional[dict[str, dict]] = None,
    global_logger: Optional[MLFlowLogger] = None,
) -> tuple[System, float]:
    """
    Train with curriculum learning: progressively increase train_pred_steps.

    Returns (trained_system, best_dev_loss).
    """
    trainer = build_trainer(
        cfg=cfg,
        train_dfs=train_dfs,
        dev_dfs=dev_dfs,
        nm_mapping=nm_mapping,
        grad_masks=grad_masks,
        norm_stats=norm_stats,
        head_stack=head_stack,
        block_spec=block_spec,
        logger=global_logger,
    )

    pred_steps = cfg.train_pred_steps
    for i in range(1 + cfg.curriculum_steps):
        if cfg.verbose:
            print(f"Training with {pred_steps} prediction steps "
                  f"(curriculum step {i}/{cfg.curriculum_steps})")

        child_logger = None
        if (global_logger is not None) and (cfg.curriculum_steps > 0):
            save_weights = cfg.save_config == "all"
            child_logger = create_child_logger(
                cfg=cfg,
                global_logger=global_logger,
                pred_steps=pred_steps,
                curriculum_step=i,
                save_weights=save_weights,
            )
            trainer.logger = child_logger

        trainer.train()  # leaves the best weights loaded

        if child_logger is not None:
            child_logger.clean_up()
            trainer.logger = None

        if i < cfg.curriculum_steps:
            pred_steps = int(cfg.curriculum_ratio * pred_steps)

            trainer.train_data = get_loader(
                dfs=train_dfs, cfg=cfg, nm_mapping=nm_mapping,
                name="train",
                steps=cfg.in_steps + pred_steps,
            )
            _reset_early_stopping(trainer)

    # Extract the trained System (first node of the Problem)
    dyn_sys = trainer.model.nodes[0]

    if (global_logger is not None) and (cfg.save_config == "last_step"):
        global_logger.log_artifacts({
            "best_model_state_dict.pth": trainer.best_model,
        })

    return dyn_sys, trainer.get_devloss()


def _reset_early_stopping(trainer: Trainer) -> None:
    """Give the next curriculum step its own full patience window.

    Neuromancer's Trainer has no public reset, so this writes its
    bookkeeping fields directly — and checks they exist first, because
    setting a renamed field would silently create a new attribute and stop
    resetting anything.
    """
    fields = {"badcount": 0, "best_devloss": torch.inf, "current_epoch": 0}
    missing = [name for name in fields if not hasattr(trainer, name)]
    if missing:
        raise AttributeError(
            f"neuromancer's Trainer no longer has {missing}; curriculum "
            "training cannot reset early stopping between steps."
        )
    for name, value in fields.items():
        setattr(trainer, name, value)
