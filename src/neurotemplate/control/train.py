"""
One tracked DPC training run, start to finish — `nt control train` without
the command line:

    from neurotemplate.common.config import load_config
    from neurotemplate.control.config import ControlConfig
    from neurotemplate.control.train import train_control

    cfg = load_config("control_configs/control.yaml", ControlConfig,
                      {"sysid_run_id": run.run_id})
    policy_run = train_control(cfg)
    policy_run.run_id, policy_run.results.comfort_violation_rate

Loads a frozen sys-id surrogate, builds a closed loop (policy + frozen
dynamics), trains the policy under the declared objective, evaluates the
closed loop, and writes a reportcard.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from neurotemplate._typing import beartype

import mlflow

from neurotemplate.common.run_summary import write_run_summary
from neurotemplate.common.tracking import save_checkpoint, tracked_run
from neurotemplate.modeling.dynamics import describe_dynamics
from neurotemplate.modeling.experiment import resolve_device, set_seeds, set_threads

from neurotemplate.control.config import ControlConfig
from neurotemplate.control.load import load_frozen_dynamics
from neurotemplate.control.plan import plan_on_surrogate
from neurotemplate.control.losses import extend_norm_stats, resolve_objective
from neurotemplate.control.scenarios import (
    build_loaders, eval_pred_steps, horizons, load_control_splits,
)
from neurotemplate.control.controller import build_controller, describe_controller
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


@dataclass
class TrainedPolicy:
    """One finished control run: its MLflow id and what it produced."""
    run_id: str
    cl_system: object
    results: ControlResults   # the closed loop on the test split
    best_dev_loss: float
    train_time_s: float


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
    cfg = deepcopy(cfg)
    cfg.validate()
    cfg.device = resolve_device(cfg.device)
    set_threads(cfg.threads)
    cfg.seed = set_seeds(cfg.seed)

    # Point MLflow at the sys-id tracking store *before* loading the run.
    mlflow.set_tracking_uri(cfg.location)

    dyn_sys, dyn_node, sysid_cfg, nm_mapping, sysid_stats = (
        load_frozen_dynamics(cfg.sysid_run_id, cfg.run_dir, cfg.weights,
                             device=cfg.device)
    )
    surrogate_arch = describe_dynamics(dyn_sys, sysid_cfg)
    # The objective's names resolve against the surrogate's columns, so this
    # is the first point a misnamed column can fail — before any data loads.
    objective = resolve_objective(cfg, nm_mapping)
    norm_stats = extend_norm_stats(objective, cfg, sysid_stats)

    _, _, hmax = horizons(cfg, sysid_cfg)
    eval_pred_steps(cfg, hmax)   # fail now, not after the policy has trained
    # One list per split (length >= 1), each plant file joined with its
    # reference file; the loaders window each file in isolation — no rollout
    # spans two files.
    train_dfs, dev_dfs, test_dfs = load_control_splits(cfg, nm_mapping)
    train_loader, dev_loader = build_loaders(
        cfg, sysid_cfg, nm_mapping, train_dfs, dev_dfs,
    )

    # Whether a band is reachable at all is a sys-id question, answered by
    # the reachability scenario on the surrogate's own reportcard. Read it
    # there before spending an hour training against a band nothing can hold.
    hooks = load_control_hooks(cfg)

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
        print(format_objective_report(balance, power_diag, notes))
    elif cfg.progress == "summary":
        for line in objective_warnings(balance) + [f"NOTE: {n}" for n in notes]:
            print(line)

    cfg.run = "DPC " + cfg.run
    term_keys = [term.key for term in objective.terms]
    with tracked_run(
        cfg,
        stdout=["train_loss", "dev_loss"] + [f"train_{k}" for k in term_keys],
        logout=["train_loss", "dev_loss", "time"] + term_keys,
    ) as logger:
        snapshot_hooks(hooks, CONTROL_SNAPSHOT, logger)
        if cfg.policy == "mpc":     # nothing to train: the surrogate is the controller
            save_checkpoint(logger, problem.state_dict())
            best_dev, train_time_s = float("nan"), 0.0
        else:
            cl_system, best_dev, train_time_s = train_policy(
                cfg, problem, train_loader, dev_loader, logger=logger,
            )
        # The same batch through the trained policy: the shares a weight can
        # be tuned against (a residual's Δu term is exactly zero before).
        trained_balance = objective_balance(
            loss, run_objective_pass(problem, balance_batch))
        if cfg.progress in ("live", "log"):
            print("Objective shares after training: " + ", ".join(
                f"{t.name} {t.share * 100:.0f}%" for t in trained_balance.terms))

        results, out = evaluate_closed_loop(
            cfg, sysid_cfg, cl_system, test_dfs, nm_mapping, norm_stats,
        )
        plan = None
        if cfg.plan_iters:
            plan = plan_on_surrogate(cfg, sysid_cfg, cl_system, policy_node, test_dfs,
                                     nm_mapping, norm_stats)
            if cfg.progress != "silent":
                print(f"Best plan on the surrogate: objective {plan.gap * 100:.0f}% below "
                      "the controller's (control reportcard, Controller gap).")

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
            surrogate_arch=surrogate_arch,
            best_dev_loss=best_dev,
            train_time_s=train_time_s,
            balance=balance, trained_balance=trained_balance,
            power=power_diag, objective_notes=notes, plan=plan,
        )
        write_reportcard(md, Path(cfg.savedir), logger=logger)
        log_control_results(logger, results)
        if plan is not None:
            logger.log_metrics({"plan_objective": plan.plan.total,
                                "policy_objective": plan.policy.total,
                                "controller_gap": plan.gap,
                                "plan_violation_rate": plan.results.comfort_violation_rate})
        logger.log_dict(objective_to_dict(balance, power_diag, notes,
                                          trained_balance),
                        name="objective_diagnostics.json")

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
