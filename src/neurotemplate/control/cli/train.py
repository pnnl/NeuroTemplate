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

import mlflow

from neurotemplate.common.config import CONTROL_CONFIG_DIR, load_config_from_cli
from neurotemplate.common.data_utils import get_data_splits, normalize_df
from neurotemplate.common.run_summary import write_run_summary
from neurotemplate.common.tracking import tracked_run
from neurotemplate.modeling.dynamics import describe_dynamics
from neurotemplate.modeling.experiment import resolve_device, set_seeds

from neurotemplate.control.config import ControlConfig
from neurotemplate.control.load import load_frozen_dynamics
from neurotemplate.control.scenarios import build_loaders, eval_pred_steps, horizons
from neurotemplate.control.policy import (
    build_policy,
    describe_policy,
    regulated_channels,
)
from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem, power_term,
)
from neurotemplate.control.objective import (
    format_objective_report,
    objective_balance,
    objective_notes,
    objective_to_dict,
    power_diagnostics,
    run_objective_pass,
    term_shapes,
)
from neurotemplate.control.reportcard import render_control_reportcard
from neurotemplate.control.pipeline import (
    EVAL_BAND_SEED,
    clears_reference,
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
    cfg, cli = load_config_from_cli(
        sys.argv[1:],
        cls=ControlConfig,
        config_dir=CONTROL_CONFIG_DIR,
        default_config_path=CONTROL_CONFIG_DIR / "control.yaml",
    )
    cfg.device = resolve_device(cfg.device)
    cfg.seed = set_seeds(cfg.seed)

    # Point MLflow at the sys-id tracking store *before* loading the run.
    mlflow.set_tracking_uri(cfg.location)

    if cfg.verbose:
        print(f"Device: {cfg.device}  Seed: {cfg.seed}")
        print(f"Tracking URI: {cfg.location}")
        print(f"Loading frozen surrogate (run {cfg.sysid_run_id})...")

    dyn_sys, dyn_node, sysid_cfg, nm_mapping, norm_stats = (
        load_frozen_dynamics(cfg.sysid_run_id, cfg.run_dir, cfg.weights)
    )
    surrogate_arch = describe_dynamics(dyn_sys, sysid_cfg)

    h_sys, h_ctrl, hmax = horizons(cfg, sysid_cfg)
    eval_pred_steps(cfg, hmax)   # fail now, not after the policy has trained
    if cfg.verbose:
        print(f"Horizons: H_sys={h_sys} H_ctrl={h_ctrl} "
              f"train_pred_steps={cfg.train_pred_steps}")
        print("Loading data...")

    # get_data_splits returns one list per split (length >= 1). Each file
    # is normalized independently and the resulting list flows through the
    # control loaders, which window each file in isolation — no rollout
    # spans two files.
    train_raws, dev_raws, test_raws, _ = get_data_splits(cfg)
    train_dfs = [normalize_df(r, norm_stats) for r in train_raws]
    dev_dfs = [normalize_df(r, norm_stats) for r in dev_raws]
    test_dfs = [normalize_df(r, norm_stats) for r in test_raws]

    train_loader, dev_loader = build_loaders(
        cfg, sysid_cfg, nm_mapping, norm_stats, train_dfs, dev_dfs,
    )

    # Whether this band is reachable at all is a sys-id question, answered by
    # the reachability scenario on the surrogate's own reportcard. Read it
    # there before spending an hour training against a band nothing can hold.
    hooks = load_control_hooks(cfg)
    u_bounds = hooks.actuation_bounds(nm_mapping, norm_stats)
    regulated, _ = regulated_channels(hooks, nm_mapping)

    if cfg.verbose:
        print("Building policy + closed loop + loss...")
    policy_node = build_policy(cfg, sysid_cfg, nm_mapping, norm_stats)
    arch = describe_policy(policy_node, cfg, sysid_cfg, nm_mapping, norm_stats)
    cl_system = build_closed_loop(dyn_node, policy_node, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm_mapping, norm_stats)
    problem = build_problem(cl_system, loss).to(cfg.device)

    # One forward pass before training: which term is actually being
    # optimised, and does the power term have an interior minimum at all.
    # Both answers cost seconds here and an iteration each if skipped.
    balance_batch = next(iter(train_loader))
    balance_out = run_objective_pass(problem, balance_batch)
    balance = objective_balance(loss, balance_out)
    power = power_term(cfg, nm_mapping, norm_stats)
    power_weighted = (cfg.energy_weight > 0 or cfg.cost_weight > 0
                      or cfg.peak_weight > 0)
    if power is None or not power_weighted:
        shapes, power_diag = [], None
    else:
        # One sweep serves all three power-derived terms: cost and peak are
        # reductions of the same signal, so sweeping them separately would
        # just rescale this.
        shapes = term_shapes(power, balance_out, hmax, nm_mapping["U"],
                             *u_bounds, norm_stats)
        power_diag = power_diagnostics(power, balance_out, hmax)
    notes = objective_notes(cfg, arch)
    print(format_objective_report(balance, shapes, power_diag, notes))

    cfg.run = "DPC " + cfg.run
    with tracked_run(
        cfg,
        stdout=[
            "train_loss", "dev_loss", "train_comfort_lower_loss",
            "train_comfort_upper_loss", "train_du_loss", "train_energy_loss",
            "train_cost_loss", "train_peak_loss",
        ],
        logout=[
            "train_loss", "dev_loss", "time", "comfort_lower_loss",
            "comfort_upper_loss", "du_loss", "energy_loss", "cost_loss",
            "peak_loss",
        ],
    ) as logger:
        if cfg.verbose:
            print("Training policy (dynamics frozen)...")
        cl_system, best_dev, train_time_s = train_policy(
            cfg, problem, train_loader, dev_loader, logger=logger,
        )
        # The same batch through the trained policy: the shares a weight can
        # be tuned against (a residual's Δu term is exactly zero before).
        trained_balance = objective_balance(
            loss, run_objective_pass(problem, balance_batch))
        if cfg.verbose:
            print("Objective shares after training: " + ", ".join(
                f"{t.name} {t.share * 100:.0f}%" for t in trained_balance.terms))

        if cfg.verbose:
            print("Evaluating closed loop on test split...")
        results, out = evaluate_closed_loop(
            cfg, sysid_cfg, cl_system, test_dfs, nm_mapping, norm_stats,
        )

        # Optional rule-based comparison. Scored on the same windows and the
        # same fixed-seed band; None when the use case defines no baseline.
        baseline = evaluate_baseline(
            cfg, sysid_cfg, dyn_node, test_dfs, nm_mapping, norm_stats,
        )
        clears = clears_reference(results, baseline)
        if cfg.verbose:
            if baseline is None:
                print("No baseline defined (no reference_control hook).")
            else:
                print("Baseline comfort violation rate: "
                      f"{baseline.comfort_violation_rate * 100:.2f}%")
                print("Learned policy clears its reference: "
                      f"{'yes' if clears else 'NO — ship the reference'}")

        if cfg.plot:
            # `t` is elapsed seconds from each file's first row, so the plots
            # need the files' wall-clock origins to be stamped with real dates.
            plot_closed_loop(
                cfg, sysid_cfg, out, nm_mapping, norm_stats,
                Path(cfg.savedir) / "plots",
                origins=[df.index[0] for df in test_dfs],
            )

        md = render_control_reportcard(
            cfg, sysid_cfg, arch, nm_mapping, regulated, results,
            eval_band_seed=EVAL_BAND_SEED, surrogate_arch=surrogate_arch,
            baseline=baseline,
            clears_reference=clears, best_dev_loss=best_dev,
            train_time_s=train_time_s,
            balance=balance, trained_balance=trained_balance, shapes=shapes,
            power=power_diag, objective_notes=notes,
        )
        write_reportcard(md, Path(cfg.savedir), logger=logger)
        log_control_results(logger, results, baseline=baseline)
        logger.log_dict(objective_to_dict(balance, shapes, power_diag, notes,
                                          trained_balance),
                        name="objective_diagnostics.json")

        write_run_summary(
            logger, stage="control", config_path=cli.config, cfg=cfg,
            metrics={"control_violation_rate": results.comfort_violation_rate,
                     "control_mean_abs_du": results.mean_abs_du},
        )

        print(f"Comfort violation rate: "
              f"{results.comfort_violation_rate * 100:.2f}%")
        print(f"Mean |Δsetpoint|: {results.mean_abs_du:.4g}")
        rid = logger.run.info.run_id
        print(f"Finished DPC run: {rid}")

        # Push savedir (plots + reportcard) to MLflow before it is removed.
        logger.log_artifacts({})


if __name__ == "__main__":
    main()
