"""
Training loop and utilities.

Uses a standard Neuromancer Problem (no subclass needed). Physics losses
are proper Neuromancer Loss objects inside PenaltyLoss, so they flow
through Problem.forward() and appear in the Trainer's output dict with
the correct labels (e.g., 'train_grad_penalty').

Physics losses are active in both train and eval. During training they
use the tunable gain; during eval they use a fixed weight of 1.0 so
that dev_loss rewards physics compliance without being gameable.
"""

import os
import torch
import numpy as np
from typing import Optional
from pandas import DataFrame
import random
from datetime import datetime



from neurotemplate._typing import beartype

from neurotemplate.common.data_utils import get_loader
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.loss import build_loss
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
    Resolve the torch device to use.

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
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
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
    norm_stats: dict[str, dict[str, float]],
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
    ).to(cfg.device)

    # The raw SSMGrad model lives inside the System's Node.
    # Physics losses need direct access to it for Jacobians and
    # training-mode checks.
    raw_model = system.nodes[0].callable

    if cfg.verbose:
        print("Building loss (tracking + physics)...")

    # Single PenaltyLoss with all objectives:
    #   - Tracking: ref_loss, vel_loss, sgn_loss (Variable expressions)
    #   - Physics:  grad_penalty (Loss object, train-only)
    loss = build_loss(
        cfg=cfg,
        raw_model=raw_model,
        system=system,
        grad_masks=grad_masks,
        nm_mapping=nm_mapping,
        norm_stats=norm_stats,
        head_stack=head_stack,
    )

    if cfg.verbose:
        print("Building Problem...")

    # Standard Neuromancer Problem — no subclass needed
    problem = Problem([system], loss).to(cfg.device)

    if cfg.verbose:
        # problem.show()
        print("Building optimizer...")

    optimizer = torch.optim.Adam(problem.parameters(), lr=cfg.lr)

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


# ── Child Logger ─────────────────────────────────────────────────

@beartype
def create_child_logger(
    cfg: ModelingConfig,
    global_logger: MLFlowLogger,
    pred_steps: int,
    curriculum_step: int,
    save_weights: bool = False,
) -> MLFlowLogger:
    """Create a nested MLflow child run for a curriculum stage."""
    default_pred_steps = cfg.train_pred_steps
    cfg.train_pred_steps = pred_steps

    child_savedir = f"{cfg.savedir}/pred_steps_{pred_steps}"
    # MLFlowLogger.clean_up() unconditionally rmtree's savedir, but never
    # creates it; for a curriculum stage that writes no local artifact the
    # dir is absent and clean_up() raises FileNotFoundError. Create it up
    # front so the stage tears down cleanly.
    os.makedirs(child_savedir, exist_ok=True)

    tmp_run_name = cfg.run
    cfg.run = f"Curriculum Step {curriculum_step}"
    child_logger = MLFlowLogger(
        cfg,
        savedir=child_savedir,
        stdout=global_logger.stdout,
        verbosity=cfg.epoch_verbose if cfg.verbose else 0,
        logout=global_logger.logout,
        nested=True,
        save_weights=save_weights,
    )
    cfg.run = tmp_run_name
    cfg.train_pred_steps = default_pred_steps

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
    norm_stats: dict[str, dict[str, float]],
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

        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)

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
            trainer.badcount = 0
            trainer.best_devloss = torch.inf
            trainer.current_epoch = 0

    # Extract the trained System (first node of the Problem)
    dyn_sys = trainer.model.nodes[0]

    if (global_logger is not None) and (cfg.save_config == "last_step"):
        global_logger.log_artifacts({
            "best_model_state_dict.pth": trainer.best_model,
            "best_model.pth": trainer.model,
        })

    best_devloss = trainer.best_devloss
    if hasattr(best_devloss, "item"):
        best_devloss = best_devloss.item()

    return dyn_sys, best_devloss
