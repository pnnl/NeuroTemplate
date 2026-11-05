"""
One tracked control run, start to finish — `nt control train` without the
command line:

    from neurotemplate.common.config import load_config
    from neurotemplate.control.config import ControlConfig
    from neurotemplate.control.train import train_control

    cfg = load_config("control_configs/control.yaml", ControlConfig,
                      {"sysid_run_id": run.run_id})
    policy_run = train_control(cfg)
    policy_run.run_id, policy_run.results.comfort_violation_rate

Loads a frozen sys-id surrogate, builds a closed loop (controller + frozen
dynamics), trains the policy under the declared objective (`policy: mpc`
trains nothing), evaluates the closed loop, and writes a reportcard.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from neurotemplate._typing import beartype

import mlflow

from neurotemplate.common.config import fields_set, settings_report
from neurotemplate.common.reportcard import RESULTS_DIR
from neurotemplate.common.run_summary import write_run_summary
from neurotemplate.common.tracking import save_checkpoint, tracked_run
from neurotemplate.modeling.experiment import resolve_device, set_seeds, set_threads

from neurotemplate.control.config import ControlConfig
from neurotemplate.control.load import load_frozen_dynamics
from neurotemplate.control.load import latest_model_run
from neurotemplate.control.losses import (
    extend_norm_stats, fill_reference_scale, resolve_objective,
)
from neurotemplate.control.scenarios import (
    build_loaders, horizons, load_control_splits, shortest_test_rollout,
)
from neurotemplate.control.controller import build_controller, describe_controller
from neurotemplate.control.policy import unread_columns
from neurotemplate.control.sensitivity import command_sensitivity
from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem, power_term,
)
from neurotemplate.control.objective import (
    format_objective_report,
    objective_balance,
    objective_notes,
    objective_to_dict,
    objective_warnings,
    power_diagnostics,
    run_objective_pass,
)
from neurotemplate.control.reportcard import render_control_reportcard
from neurotemplate.control.pipeline import (
    ControlResults,
    train_policy,
    evaluate_closed_loop,
    log_control_results,
    write_reportcard,
    plot_closed_loop,
)
from neurotemplate.hooks import CONTROL_SNAPSHOT, load_control_hooks, snapshot_hooks


#: What each run names even when you left it alone. Everything else sits in
#: `nt docs config`, and appears on "You set" only when you set it.
COMMON_SETTINGS = ("sysid_run_id", "policy", "policy_window", "horizon", "test_horizon",
                   "train_stride", "dev_stride", "test_stride", "reference_scale", "epochs",
                   "lr", "batch_size", "log2_policy_hidden", "policy_layers", "seed")


@dataclass
class TrainedPolicy:
    """One finished control run: its MLflow id and what it produced."""
    run_id: str
    cl_system: object
    results: ControlResults   # the closed loop on the test split
    best_dev_loss: Optional[float]     # None for `policy: mpc`, which trains nothing
    train_time_s: Optional[float]


@beartype
def train_control(cfg: ControlConfig,
                  config_path: Optional[str] = None) -> TrainedPolicy:
    """Train, evaluate and log one policy as its own MLflow run.

    `cfg` comes from `load_config` or is built in code; it is validated
    here and not modified: the run resolves its device, seed and run name
    on a copy.
    `config_path` is the file `cfg` came from, recorded in the run summary;
    None for a config built in code.
    """
    given = fields_set(cfg)
    cfg = deepcopy(cfg)
    cfg.validate()
    cfg.device = resolve_device(cfg.device)
    set_threads(cfg.threads)
    cfg.seed = set_seeds(cfg.seed)

    # Point MLflow at the sys-id tracking store *before* loading the run.
    mlflow.set_tracking_uri(cfg.location)
    cfg.sysid_run_id = cfg.sysid_run_id or latest_model_run()

    dyn_sys, dyn_node, sysid_cfg, nm_mapping, sysid_stats = (
        load_frozen_dynamics(cfg.sysid_run_id, cfg.run_dir, cfg.weights,
                             device=cfg.device)
    )
    # The windows mean nothing before the surrogate says the timestep.
    cfg.steps_per_hour = sysid_cfg.steps_per_hour
    # The objective's names resolve against the surrogate's columns, so this
    # is the first point a misnamed column can fail — before any data loads.
    objective = resolve_objective(cfg, nm_mapping)
    cfg.fill_window_table(objective.regulated, nm_mapping, objective.r_columns)
    objective = resolve_objective(cfg, nm_mapping)

    _, _, hmax = horizons(cfg, sysid_cfg)
    # One list per split (length >= 1); the loaders window each file in
    # isolation — no rollout spans two files.
    train_dfs, dev_dfs, test_dfs = load_control_splits(cfg, nm_mapping)
    cfg.fill_common_windows(shortest_test_rollout(cfg, sysid_cfg, test_dfs))
    cfg.check_windows()
    fill_reference_scale(objective, cfg, train_dfs)
    norm_stats = extend_norm_stats(objective, cfg, sysid_stats)
    if cfg.progress != "silent":
        print(settings_report(cfg, given, COMMON_SETTINGS))
    train_loader, dev_loader = build_loaders(
        cfg, sysid_cfg, nm_mapping, train_dfs, dev_dfs,
    )

    # Whether a band is reachable at all is a sys-id question, answered by
    # the reachability scenario on the surrogate's own reportcard. Read it
    # there before spending an hour training against a band nothing can hold.
    hooks = load_control_hooks(cfg)

    # Raises if no output has a gradient path to U: nothing could be trained
    # or planned through the surrogate. Partial zero shares are the reportcard's.
    sensitivity = command_sensitivity(dyn_sys, train_dfs, nm_mapping, cfg.device)
    if cfg.progress != "silent":
        for c in sensitivity:
            if c.zero_share == 1.0:
                print(f"WARNING: the surrogate's outputs never move with {c.column!r} "
                      "in any recorded window (an exact zero gradient); a controller "
                      "cannot learn or plan that command.")

    trains = cfg.policy != "mpc"   # MPC plans through the surrogate; nothing to train
    if trains and cfg.progress != "silent":
        unread = unread_columns(cfg, nm_mapping, objective.r_columns)
        if unread:
            print(f"Not read by the policy (not in policy_window): {unread}")
    policy_node = build_controller(cfg, sysid_cfg, nm_mapping, norm_stats, dyn_node)
    arch = describe_controller(policy_node, cfg, sysid_cfg, nm_mapping)
    cl_system = build_closed_loop(dyn_node, policy_node, cfg, sysid_cfg)
    loss = build_control_loss(cfg, sysid_cfg, nm_mapping)
    problem = build_problem(cl_system, loss).to(cfg.device)

    # One forward pass before training: which term is actually being
    # optimised, and does the power term have an interior minimum at all.
    # Both answers cost seconds here and an iteration each if skipped.
    balance_batch = next(iter(train_loader))
    balance_out = run_objective_pass(problem, balance_batch)
    balance = objective_balance(loss, balance_out)
    power = power_term(cfg, nm_mapping)
    power_diag = (power_diagnostics(power, balance_out, hmax)
                  if power is not None and objective.reads_power else None)
    notes = objective_notes(objective, cfg)
    if cfg.progress in ("live", "log"):
        print(format_objective_report(balance, power_diag, notes, trains))
    elif cfg.progress == "summary":
        for line in objective_warnings(balance) + [f"NOTE: {n}" for n in notes]:
            print(line)

    cfg.run = ("DPC " if trains else "MPC ") + cfg.run
    term_keys = [term.key for term in objective.terms]
    with tracked_run(
        cfg,
        stdout=["train_loss", "dev_loss"] + [f"train_{k}" for k in term_keys],
        logout=["train_loss", "dev_loss", "time"] + term_keys,
    ) as logger:
        snapshot_hooks(hooks, CONTROL_SNAPSHOT, logger)
        best_dev = train_time_s = trained_balance = None
        if trains:
            cl_system, best_dev, train_time_s = train_policy(
                cfg, problem, train_loader, dev_loader, logger=logger,
            )
            # The same batch through the trained policy: the shares a weight can
            # be tuned against.
            trained_balance = objective_balance(
                loss, run_objective_pass(problem, balance_batch))
            if cfg.progress in ("live", "log"):
                print("Objective shares after training: " + ", ".join(
                    f"{t.name} {t.share * 100:.0f}%" for t in trained_balance.terms))
        else:
            save_checkpoint(logger, problem.state_dict())
            policy_node.callable.solves.clear()   # count the evaluation's solves only

        results, out = evaluate_closed_loop(
            cfg, sysid_cfg, cl_system, test_dfs, nm_mapping, norm_stats,
        )
        if not trains:
            arch["solves"] = list(policy_node.callable.solves)

        if cfg.plot:
            # `t` is elapsed seconds from each file's first row, so the plots
            # need the files' wall-clock origins to be stamped with real dates.
            plot_closed_loop(
                cfg, sysid_cfg, out, nm_mapping, norm_stats,
                Path(cfg.savedir) / "plots",
                origins=[df.index[0] for df in test_dfs],
            )

        md = render_control_reportcard(
            cfg, sysid_cfg, arch, nm_mapping, results,
            best_dev_loss=best_dev,
            train_time_s=train_time_s,
            balance=balance, trained_balance=trained_balance,
            power=power_diag, objective_notes=notes,
            sensitivity=sensitivity,
        )
        write_reportcard(md, Path(cfg.savedir), logger=logger)
        log_control_results(logger, results)
        logger.log_dict(objective_to_dict(balance, power_diag, notes,
                                          trained_balance),
                        name=f"{RESULTS_DIR}/objective_diagnostics.json")
        logger.log_dict({c.column: c.zero_share for c in sensitivity},
                        name=f"{RESULTS_DIR}/command_sensitivity.json")

        write_run_summary(
            logger, stage="control", config_path=config_path, cfg=cfg,
            metrics={"control_violation_rate": results.comfort_violation_rate,
                     "control_mean_abs_du": results.mean_abs_du},
        )

        # Push savedir (plots + reportcard) to MLflow before it is removed.
        logger.log_artifacts({})

    return TrainedPolicy(
        run_id=logger.run.info.run_id,
        cl_system=cl_system,
        results=results,
        best_dev_loss=best_dev,
        train_time_s=train_time_s,
    )
