"""
Entry point for training with physics-informed losses.

Thin wrapper: this file owns CLI parsing, run-name policy, status files,
and the MLflow logger lifecycle. Everything else (data prep, training,
eval, plotting, reportcard) is in pipeline.train_and_evaluate.

Usage:
    nt model train --config model_configs/modeling.yaml
    nt model train --config model_configs/modeling.yaml \
        --set grad_gain=1.0
"""

import sys
from pathlib import Path

from neurotemplate._typing import beartype

from neurotemplate.common.config import load_config_from_cli, MODEL_CONFIG_DIR
from neurotemplate.modeling.config import ModelingConfig
from neuromancer.loggers import MLFlowLogger

from neurotemplate.common.data_utils import (
    normalize_df,
    compute_norm_stats,
    get_data_splits,
    get_grad_masks,
    warn_degenerate_columns,
)
from neurotemplate.modeling.experiment import resolve_device, set_seeds
from neurotemplate.modeling.pipeline import train_and_evaluate
from neurotemplate.hooks import load_modeling_hooks


@beartype
def main() -> None:

    cfg, _cli = load_config_from_cli(
        sys.argv[1:],
        cls=ModelingConfig,
        config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
    )

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
        if cfg.grad_gain > 0:
            print(f"Gradient penalty: gain={cfg.grad_gain}, "
                  f"samples={cfg.grad_n_samples}")

    # ── Data, normalization, masks, synth_cfg (computed once) ────
    if cfg.verbose:
        print("Loading data...")

    train_raws, dev_raws, test_raws, freq_s = get_data_splits(cfg)
    cfg.steps_per_hour = 3600.0 / freq_s
    # Column classification only needs the schema — first train df is
    # enough since `_validate_split_columns` already enforced cross-file
    # column match.
    nm_mapping = data_mapping(df=train_raws[0], cfg=cfg)
    zone_spec = hooks.zone_mapping(nm_mapping, cfg) if hasattr(hooks, "zone_mapping") else None

    norm_overrides = column_norm_config(nm_mapping)
    # Stats pooled across all training files so per-column ranges reflect
    # the union of training data. The concatenated frame is built once and
    # never used for windowing — windowing stays per-file.
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

    grad_masks = get_grad_masks(grad_path=cfg.grad_path, nm_mapping=nm_mapping)
    synth_cfg = build_synth_cfg(nm_mapping, norm_stats)
    head_stack = data_head_config(nm_mapping, norm_stats)

    # ── Run name & MLflow logger ─────────────────────────────────
    cfg.run = cfg.model.upper() + " " + cfg.run
    parent_run_name = cfg.run
    if cfg.curriculum_steps:
        cfg.run += " Parent"

    if cfg.verbose:
        print("Setting up MLflow logger...")

    global_logger = MLFlowLogger(
        args=cfg,
        savedir=cfg.savedir,
        stdout=["train_loss", "train_ref_loss", "train_grad_penalty",
                "dev_loss"],
        verbosity=cfg.epoch_verbose if cfg.verbose else 0,
        logout=[
            "train_loss", "dev_loss",
            "ref_loss", "sgn_loss", "vel_loss",
            "grad_penalty",
            "time", "rmse",
        ],
        nested=False,
    )
    cfg.run = parent_run_name

    # ── Train + evaluate + reportcard + plot ─────────────────────
    result = train_and_evaluate(
        cfg=cfg,
        train_dfs=train_dfs, dev_dfs=dev_dfs,
        test_dfs=test_dfs,
        nm_mapping=nm_mapping, norm_stats=norm_stats,
        grad_masks=grad_masks, synth_cfg=synth_cfg,
        freq_s=freq_s, logger=global_logger,
        head_stack=head_stack, zone_spec=zone_spec,
    )

    if cfg.verbose and hasattr(global_logger, "run"):
        r_id = global_logger.run.info.run_id
        print(f"Finished run: {r_id}, "
              f"test_rmse: {result['test_results'].aggregate_rmse:.3f}")
        print(f"MLflow UI:  uv run mlflow ui")
        print(f"Load model: uv run nt model load --run_id {r_id}")

    global_logger.clean_up()


if __name__ == "__main__":
    main()
