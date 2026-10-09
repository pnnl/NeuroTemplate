"""
Entry point for training with physics-informed losses.

Thin wrapper: this file owns CLI parsing, the run name, and the MLflow run.
Data prep is pipeline.prepare_data; training, eval, plotting and the
reportcard are pipeline.train_and_evaluate.

Usage:
    nt model train --config model_configs/modeling.yaml
    nt model train --config model_configs/modeling.yaml \
        --set grad_gain=1.0
"""

import sys

from neurotemplate._typing import beartype

from neurotemplate.common.config import load_config_from_cli, MODEL_CONFIG_DIR
from neurotemplate.common.run_summary import write_run_summary
from neurotemplate.common.tracking import tracked_run
from neurotemplate.hooks import load_modeling_hooks
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.experiment import resolve_device, set_seeds
from neurotemplate.modeling.pipeline import (
    prepare_data, run_summary_metrics, train_and_evaluate,
)

#: Loss terms echoed to stdout each epoch, and the substrings of the metric
#: keys logged to MLflow. Shared with `nt model optuna`'s final retrain.
STDOUT_METRICS = ["train_loss", "train_ref_loss", "train_grad_penalty", "dev_loss"]
LOGGED_METRICS = ["train_loss", "dev_loss", "ref_loss", "grad_penalty", "time",
                  "rmse"]


def print_finished(run_id: str, normalized_rmse: float) -> None:
    """The closing lines of a training run: its id and how to look at it."""
    print(f"Finished run: {run_id}, test_rmse: {normalized_rmse:.3f}")
    print("MLflow UI:  uv run mlflow ui")
    print(f"Load model: uv run nt model load --run_id {run_id} --data <test csv>")


@beartype
def main() -> None:
    cfg, cli = load_config_from_cli(
        sys.argv[1:],
        cls=ModelingConfig,
        config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
    )

    cfg.device = resolve_device(cfg.device)
    cfg.seed = set_seeds(cfg.seed)
    hooks = load_modeling_hooks(cfg)

    if cfg.verbose:
        print(f"Device: {cfg.device}")
        print(f"Seed:   {cfg.seed}")
        if cfg.ensemble_seeds > 1:
            print(f"Ensemble: {cfg.ensemble_seeds} seeds "
                  f"({cfg.seed}..{cfg.seed + cfg.ensemble_seeds - 1})")
        if cfg.grad_gain > 0:
            print(f"Gradient penalty: gain={cfg.grad_gain}, "
                  f"samples={cfg.grad_n_samples}")
        print("Loading data...")

    data = prepare_data(cfg, hooks)

    # With a curriculum, each step is a nested child run under this parent.
    cfg.run = cfg.model.upper() + " " + cfg.run
    run_name = cfg.run + " Parent" if cfg.curriculum_steps else cfg.run

    with tracked_run(cfg, stdout=STDOUT_METRICS, logout=LOGGED_METRICS,
                     run_name=run_name) as logger:
        result = train_and_evaluate(cfg, logger=logger, **vars(data))
        write_run_summary(
            logger, stage="model", config_path=cli.config, cfg=cfg,
            metrics=run_summary_metrics(result["test_results"]),
            mapping=data.nm_mapping,
        )
        if cfg.verbose:
            print_finished(logger.run.info.run_id,
                           result["test_results"].normalized_rmse)


if __name__ == "__main__":
    main()
