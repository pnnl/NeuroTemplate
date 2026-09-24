"""
Optuna hyperparameter search followed by full retraining with best params.

Phase 1: Run N trials with short training (trial_epochs), optimizing the
         combined dev_loss (single-objective, minimize).
Phase 2: Retrain the best trial's config with full epochs and full
         eval/reportcard/plot via pipeline.train_and_evaluate.

The per-trial loop in `objective()` calls train_with_curriculum directly —
it deliberately skips eval/plot/reportcard for speed.
"""

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter as tic
from typing import Optional

from optuna import Trial, create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from pandas import DataFrame

from neurotemplate._typing import beartype

from neurotemplate.common.config import load_config_from_cli, MODEL_CONFIG_DIR
from neurotemplate.common.data_utils import get_grad_masks
from neurotemplate.common.run_summary import write_run_summary
from neurotemplate.common.tracking import tracked_run
from neurotemplate.hooks import load_modeling_hooks
from neurotemplate.modeling.cli.train import LOGGED_METRICS, print_finished
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.experiment import resolve_device, set_seeds, train_with_curriculum
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.pipeline import (
    prepare_data, run_summary_metrics, train_and_evaluate,
)
from neurotemplate.modeling.reportcard import spread_stats


@beartype
def objective(
    trial: Trial,
    cfg: ModelingConfig,
    train_dfs: list[DataFrame],
    dev_dfs: list[DataFrame],
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    head_stack: Optional[HeadStack],
    block_spec: Optional[dict[str, dict]] = None,
) -> float:
    """Train under suggested hyperparameters across multiple seeds; return mean dev loss.

    Prints the seeds' min/median/max dev loss (via `spread_stats`) when
    `cfg.verbose` — free, since the losses are already in hand, and it says
    whether the search is discriminating between trials or chasing noise.
    """
    trial_cfg = load_modeling_hooks(cfg).hyper_args(trial, cfg)
    trial_cfg.epochs = trial_cfg.trial_epochs
    trial_cfg.verbose = False
    trial_cfg.epoch_verbose = trial_cfg.epochs + 1
    trial_cfg.plot = False

    trial_grad_masks = get_grad_masks(
        grad_path=trial_cfg.grad_path, nm_mapping=nm_mapping
    )

    losses = []
    for i in range(cfg.trial_seeds):
        set_seeds(cfg.seed + i)

        _, dev_loss = train_with_curriculum(
            cfg=trial_cfg,
            train_dfs=train_dfs,
            dev_dfs=dev_dfs,
            nm_mapping=nm_mapping,
            grad_masks=trial_grad_masks,
            norm_stats=norm_stats,
            head_stack=head_stack,
            block_spec=block_spec,
            global_logger=None,
        )

        losses.append(dev_loss)
        trial.report(sum(losses) / len(losses), step=i)
        if trial.should_prune():
            raise TrialPruned()

    if cfg.verbose:
        spread = spread_stats(losses)
        print(f"  trial {trial.number}: dev_loss across {len(losses)} seeds "
              f"min={spread.min:.4f} median={spread.median:.4f} "
              f"max={spread.max:.4f}")

    return sum(losses) / len(losses)


@beartype
def main() -> None:
    cfg, cli = load_config_from_cli(
        sys.argv[1:],
        cls=ModelingConfig,
        config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
    )

    if cfg.n_trials <= 0:
        raise ValueError("n_trials must be > 0")

    cfg.device = resolve_device(cfg.device)
    cfg.seed = set_seeds(cfg.seed)
    hooks = load_modeling_hooks(cfg)

    if cfg.verbose:
        print(f"Device: {cfg.device}")
        print(f"Seed:   {cfg.seed}")
        print("=" * 60)
        print("PHASE 1: Hyperparameter Search (Optuna)")
        print("=" * 60)
        print("Loading data...")

    # Shared by every trial and the final retrain.
    data = prepare_data(cfg, hooks)

    # ── Phase 1: Optuna search ───────────────────────────────────
    optuna_dir = Path(cfg.optuna_dir)
    optuna_dir.mkdir(parents=True, exist_ok=True)
    cfg.study_name += datetime.now().strftime("_%b%d_%H_%M")
    db_path = optuna_dir / f"{cfg.study_name}.sqlite3"

    study = create_study(
        direction="minimize",
        study_name=cfg.study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )

    if cfg.verbose:
        print(f"Study: {cfg.study_name}")
        print(f"DB:    {db_path}")
        print(f"Trials: {cfg.n_trials}")

    search_t0 = tic()
    study.optimize(
        lambda trial: objective(
            trial, cfg, data.train_dfs, data.dev_dfs,
            data.nm_mapping, data.norm_stats, data.head_stack, data.block_spec,
        ),
        n_trials=cfg.n_trials,
        n_jobs=1,
        show_progress_bar=cfg.verbose,
    )
    search_time_s = tic() - search_t0

    if cfg.verbose:
        print(f"\nSearch time: {search_time_s:.1f}s")
        print(f"Best dev loss: {study.best_value:.6f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

    # ── Phase 2: Full retrain ────────────────────────────────────
    unknown = [k for k in study.best_params if not hasattr(cfg, k)]
    if unknown:
        raise ValueError(f"Best params not in ModelingConfig: {unknown}")
    cfg = replace(cfg, **study.best_params, verbose=True,
                  run=cfg.study_name + "_result")
    cfg.validate()
    data.grad_masks = get_grad_masks(grad_path=cfg.grad_path, nm_mapping=data.nm_mapping)

    print("\n" + "=" * 60)
    print("PHASE 2: Full Training with Best Hyperparameters")
    print("=" * 60)
    if cfg.ensemble_seeds > 1:
        print(f"Ensemble: {cfg.ensemble_seeds} seeds "
              f"({cfg.seed}..{cfg.seed + cfg.ensemble_seeds - 1})")

    with tracked_run(cfg, stdout=["train_loss", "dev_loss"],
                     logout=LOGGED_METRICS) as logger:
        logger.log_dict(study.best_params, name="best_params.json")
        result = train_and_evaluate(
            cfg, logger=logger, extra_metrics={"search_time": search_time_s},
            **vars(data),
        )
        write_run_summary(
            logger, stage="model", config_path=cli.config, cfg=cfg,
            metrics=run_summary_metrics(result["test_results"]),
            mapping=data.nm_mapping,
        )
        print_finished(logger.run.info.run_id,
                       result["test_results"].normalized_rmse)
        print(f"Optuna dashboard: uv run optuna-dashboard sqlite:///{db_path}")


if __name__ == "__main__":
    main()
