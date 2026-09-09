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
from pathlib import Path
from time import perf_counter as tic
from datetime import datetime
from typing import Optional

from optuna import Trial, create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from pandas import DataFrame

from neurotemplate._typing import beartype

from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.common.config import load_config_from_cli, MODEL_CONFIG_DIR
from neuromancer.loggers import MLFlowLogger

from neurotemplate.common.data_utils import (
    normalize_df,
    compute_norm_stats,
    warn_degenerate_columns,
    get_data_splits,
    get_grad_masks,
)
from neurotemplate.modeling.experiment import resolve_device, set_seeds, train_with_curriculum
from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.pipeline import resolve_block_spec, train_and_evaluate
from neurotemplate.modeling.reportcard import spread_stats
from neurotemplate.hooks import load_modeling_hooks


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
    cfg, _cli = load_config_from_cli(
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
    data_mapping = hooks.data_mapping
    data_head_config = hooks.data_head_config
    column_norm_config = hooks.column_norm_config
    build_synth_cfg = hooks.synth_cfg

    if cfg.verbose:
        print(f"Device: {cfg.device}")
        print(f"Seed:   {cfg.seed}")
        print("=" * 60)
        print("PHASE 1: Hyperparameter Search (Optuna)")
        print("=" * 60)
        print("Loading data...")

    # ── Data, normalization, synth_cfg (all shared across trials) ─
    train_raws, dev_raws, test_raws, freq_s = get_data_splits(cfg)
    cfg.steps_per_hour = 3600.0 / freq_s
    nm_mapping = data_mapping(df=train_raws[0], cfg=cfg)
    block_spec = resolve_block_spec(hooks, nm_mapping, cfg)

    norm_overrides = column_norm_config(nm_mapping)
    import pandas as _pd
    train_pool = train_raws[0] if len(train_raws) == 1 else _pd.concat(train_raws)
    norm_stats = compute_norm_stats(
        df=train_pool, norm_type=cfg.norm_type, column_overrides=norm_overrides,
    )
    # Optional user hook: post-process norm_stats (e.g. tie two columns to a
    # shared shift/stretch). No-op unless the hooks module defines it.
    if hasattr(hooks, "tie_norm_stats"):
        norm_stats = hooks.tie_norm_stats(nm_mapping, norm_stats, cfg)
    warn_degenerate_columns(norm_stats, nm_mapping)
    train_dfs = [normalize_df(r, norm_stats) for r in train_raws]
    dev_dfs = [normalize_df(r, norm_stats) for r in dev_raws]
    test_dfs = [normalize_df(r, norm_stats) for r in test_raws]
    synth_cfg = build_synth_cfg(nm_mapping, norm_stats)
    head_stack = data_head_config(nm_mapping, norm_stats)

    # ── Phase 1: Optuna search ───────────────────────────────────
    optuna_dir = Path(cfg.optuna_dir)
    optuna_dir.mkdir(parents=True, exist_ok=True)
    cfg.study_name += datetime.now().strftime("_%b%d_%H_%M")
    db_path = optuna_dir / f"{cfg.study_name}.sqlite3"
    storage = f"sqlite:///{db_path}"

    study = create_study(
        direction="minimize",
        study_name=cfg.study_name,
        storage=storage,
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
            trial, cfg, train_dfs, dev_dfs,
            nm_mapping, norm_stats, head_stack, block_spec,
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
    if cfg.verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: Full Training with Best Hyperparameters")
        print("=" * 60)
        if cfg.ensemble_seeds > 1:
            print(f"Ensemble: {cfg.ensemble_seeds} seeds "
                  f"({cfg.seed}..{cfg.seed + cfg.ensemble_seeds - 1})")

    set_seeds(cfg.seed)
    unknown = [k for k in study.best_params if not hasattr(cfg, k)]
    if unknown:
        raise ValueError(f"Best params not in ModelingConfig: {unknown}")
    for key, value in study.best_params.items():
        setattr(cfg, key, value)

    cfg.verbose = True
    grad_masks = get_grad_masks(grad_path=cfg.grad_path, nm_mapping=nm_mapping)

    cfg.run = cfg.study_name + "_result"
    global_logger = MLFlowLogger(
        args=cfg,
        savedir=cfg.savedir,
        stdout=["train_loss", "dev_loss"],
        verbosity=cfg.epoch_verbose,
        logout=[
            "train_loss", "dev_loss",
            "ref_loss", "sgn_loss", "vel_loss",
            "grad_penalty",
            "time", "rmse",
        ],
        nested=False,
    )
    global_logger.log_dict(study.best_params, name="best_params.json")

    result = train_and_evaluate(
        cfg=cfg,
        train_dfs=train_dfs, dev_dfs=dev_dfs,
        test_dfs=test_dfs,
        nm_mapping=nm_mapping, norm_stats=norm_stats,
        grad_masks=grad_masks, synth_cfg=synth_cfg,
        freq_s=freq_s, logger=global_logger,
        head_stack=head_stack, block_spec=block_spec,
        extra_metrics={"search_time": search_time_s},
    )

    if hasattr(global_logger, "run"):
        r_id = global_logger.run.info.run_id
        print(f"Finished run: {r_id}, "
              f"test_rmse: {result['test_results'].aggregate_rmse:.3f}")
        print("MLflow UI:    uv run mlflow ui")
        print(f"Load model:   uv run load_model.py --run_id {r_id}")
        print("Load Optuna Dashboard for this study: \n")
        print(f"uv run optuna-dashboard sqlite:///optuna_dbs/{cfg.study_name}.sqlite3")

    global_logger.clean_up()


if __name__ == "__main__":
    main()
