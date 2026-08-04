"""
DPC control pipeline entry point.

  nt control train --config control_configs/control.yaml \
      --set sysid_run_id=<mlflow run id>

Loads a frozen sys-id surrogate, builds a closed loop (policy + frozen
dynamics), trains the policy under a comfort-band objective, evaluates the
closed loop, and writes a reportcard.
"""

from __future__ import annotations

import sys
from pathlib import Path

from neurotemplate._typing import beartype

from neuromancer.loggers import MLFlowLogger

from neurotemplate.common.data_utils import get_data_splits, normalize_df
from neurotemplate.modeling.experiment import resolve_device, set_seeds

from neurotemplate.control.config import ControlConfig, load_config_from_cli, CONTROL_CONFIG_DIR
from neurotemplate.control.load import load_frozen_dynamics
from neurotemplate.control.scenarios import build_loaders, horizons
from neurotemplate.control.policy import build_policy, describe_policy
from neurotemplate.control.closed_loop import build_closed_loop, build_control_loss, build_problem
from neurotemplate.control.reportcard import render_control_reportcard
from neurotemplate.control.pipeline import (
    EVAL_BAND_SEED,
    train_policy,
    evaluate_baseline,
    evaluate_closed_loop,
    log_control_results,
    write_reportcard,
    plot_closed_loop,
)
from neurotemplate.hooks import load_control_hooks


@beartype
def main() -> None:
    cfg, _cli = load_config_from_cli(
        sys.argv[1:],
        cls=ControlConfig,
        config_dir=CONTROL_CONFIG_DIR,
        default_config_path=CONTROL_CONFIG_DIR / "control.yaml",
    )
    cfg.validate()

    cfg.device = resolve_device(cfg.device)
    cfg.seed = set_seeds(cfg.seed)

    # Point MLflow at the sys-id tracking store *before* loading the run.
    import mlflow
    mlflow.set_tracking_uri(cfg.location)

    if cfg.verbose:
        print(f"Device: {cfg.device}  Seed: {cfg.seed}")
        print(f"Tracking URI: {cfg.location}")
        print(f"Loading frozen surrogate (run {cfg.sysid_run_id})...")

    dyn_sys, dyn_node, sysid_cfg, nm_mapping, norm_stats = (
        load_frozen_dynamics(cfg.sysid_run_id, cfg.run_dir, cfg.weights)
    )

    h_sys, h_ctrl, hmax = horizons(cfg, sysid_cfg)
    if cfg.verbose:
        print(f"Horizons: H_sys={h_sys} H_ctrl={h_ctrl} "
              f"train_pred_steps={cfg.train_pred_steps}")
        print("Loading data...")

    # get_data_splits returns one list per split (length >= 1). Each file
    # is normalized independently and the resulting list flows through the
    # control loaders, which window each file in isolation — no rollout
    # spans two files. See docs/multi-file-data.md.
    train_raws, dev_raws, test_raws, _ = get_data_splits(cfg)
    train_dfs = [normalize_df(r, norm_stats) for r in train_raws]
    dev_dfs = [normalize_df(r, norm_stats) for r in dev_raws]
    test_dfs = [normalize_df(r, norm_stats) for r in test_raws]

    train_loader, dev_loader = build_loaders(
        cfg, sysid_cfg, nm_mapping, norm_stats, train_dfs, dev_dfs,
    )

    if cfg.verbose:
        print("Building policy + closed loop + loss...")
    policy_node = build_policy(cfg, sysid_cfg, nm_mapping, norm_stats)
    cl_system = build_closed_loop(dyn_node, policy_node, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm_mapping, norm_stats)
    problem = build_problem(cl_system, loss).to(cfg.device)


    cfg.run = "DPC " + cfg.run
    logger = MLFlowLogger(
        args=cfg,
        savedir=cfg.savedir,
        stdout=[
            "train_loss", "dev_loss", "train_comfort_lower_loss",
            "train_comfort_upper_loss", "train_du_loss", "train_effort_loss"
        ],
        verbosity=cfg.epoch_verbose if cfg.verbose else 0,
        logout=[
            "train_loss", "dev_loss", "time", "comfort_lower_loss",
            "comfort_upper_loss", "du_loss", "effort"
        ],
        nested=False,
    )

    if cfg.verbose:
        print("Training policy (dynamics frozen)...")
    cl_system, best_dev, train_time_s = train_policy(
        cfg, problem, train_loader, dev_loader, logger=logger,
    )

    if cfg.verbose:
        print("Evaluating closed loop on test split...")
    results, out = evaluate_closed_loop(
        cfg, sysid_cfg, cl_system, test_dfs, nm_mapping, norm_stats,
    )

    # Optional rule-based comparison. Scored on the same windows and the same
    # fixed-seed band; None when the use case defines no baseline.
    baseline = evaluate_baseline(
        cfg, sysid_cfg, dyn_node, test_dfs, nm_mapping, norm_stats,
    )
    if cfg.verbose:
        if baseline is None:
            print("No baseline defined (build_baseline_policy hook absent).")
        else:
            print("Baseline comfort violation rate: "
                  f"{baseline.comfort_violation_rate * 100:.2f}%")

    if cfg.plot:
        plot_closed_loop(
            cfg, sysid_cfg, out, nm_mapping, norm_stats,
            Path(cfg.savedir) / "plots",
        )

    controlled, _ = load_control_hooks(cfg).controlled_channels(nm_mapping)
    arch = describe_policy(policy_node, cfg, sysid_cfg, nm_mapping, norm_stats)
    md = render_control_reportcard(
        cfg, sysid_cfg, arch, nm_mapping, controlled, results,
        eval_band_seed=EVAL_BAND_SEED, baseline=baseline,
        best_dev_loss=best_dev, train_time_s=train_time_s,
    )
    write_reportcard(md, Path(cfg.savedir), logger=logger)
    log_control_results(logger, results, baseline=baseline)

    print(f"Comfort violation rate: "
          f"{results.comfort_violation_rate * 100:.2f}%")
    print(f"Mean |Δsetpoint|: {results.mean_abs_du:.4g}")
    if hasattr(logger, "run"):
        rid = logger.run.info.run_id
        print(f"Finished DPC run: {rid}")
        with open("latest_control_run_id.dat", "w") as f:
            f.write(rid)

    # Push savedir (plots + reportcard) to MLflow before clean_up rmtrees it.
    if hasattr(logger, "log_artifacts"):
        logger.log_artifacts({})
    logger.clean_up()


if __name__ == "__main__":
    main()
