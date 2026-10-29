"""
Train the DPC policy, evaluate the closed loop, write a reportcard.

Mirrors modeling/pipeline.py in shape. Only the policy trains — the surrogate is
frozen, so the optimizer is built over requires_grad params only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter as tic
from typing import Optional

import torch
from matplotlib import rc_context

from neuromancer.trainer import Trainer

from neurotemplate.common.data_utils import NormStats, axis_label
from neurotemplate.common.progress import Progress
from neurotemplate.common.metrics import (
    ChannelMetrics,
    metric_key,
    metrics_to_dict,
    per_channel_metrics,
)
from neurotemplate.common.reportcard import write_reportcard as _write_reportcard
from neurotemplate.common.tracking import save_checkpoint
from neurotemplate.control.metrics import (
    ActuationMetrics,
    ComfortMetrics,
    EnergyMetrics,
    JumpResponse,
    ResidualMetrics,
    actuation_metrics,
    actuation_to_dict,
    comfort_metrics,
    comfort_to_dict,
    energy_metrics,
    hours_per_step,
    residual_metrics,
    response_audit,
    response_to_dict,
)
from neurotemplate.control.losses import POWER, Objective, resolve_objective
from neurotemplate.control.policy import reference_over_horizon
from neurotemplate.control.scenarios import (
    build_window_dict_multi,
    eval_pred_steps,
    horizons,
    scored_series,
    scored_steps,
)
from neurotemplate.hooks import load_control_hooks
from neurotemplate.modeling.plotting import PLOT_STYLE, unique_day_dir

from neurotemplate._typing import beartype


@dataclass
class ControlResults:
    comfort_violation_rate: float      # pooled fraction of (channel, step) outside band
    comfort_mean_exceedance: float     # pooled mean excursion over ALL steps
    comfort: list[ComfortMetrics]      # per regulated channel
    mean_abs_du: float                 # mean |ΔU| over every U column (physical units)
    actuation: list[ActuationMetrics]  # per-control effort + saturation
    response: list[JumpResponse]       # reaction to band jumps
    tracking: list[ChannelMetrics]     # error vs band midpoint, two-sided bands
    energy: Optional[EnergyMetrics]    # None if power_metric not implemented
    eval_time_s: float
    # Per anchored U column; empty without a residual.
    residual: list[ResidualMetrics] = field(default_factory=list)


# ── Train (policy only) ──────────────────────────────────────────


@beartype
def train_policy(
    cfg,
    problem,
    train_loader,
    dev_loader,
    logger=None,
) -> tuple[object, float, float]:
    """Returns (trained closed-loop System, best_dev_loss, train_time_s)."""
    trainable = [p for p in problem.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "No trainable parameters — the policy is frozen or empty. The "
            "dynamics must be frozen but the policy must not be."
        )
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)

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
        eval_metric="mean_dev_loss",
        callback=Progress(cfg.progress, cfg.update_every),
        # Neuromancer prints its own epoch line only when there is no logger.
        epoch_verbose=cfg.epochs + 1,
    )
    t0 = tic()
    trainer.train()  # leaves the best weights loaded
    train_time_s = tic() - t0
    if logger is not None:
        save_checkpoint(logger, trainer.best_model)

    cl_system = trainer.model.nodes[0]
    return cl_system, trainer.get_devloss(), train_time_s


# ── Evaluate (long closed-loop rollout) ──────────────────────────


@beartype
def band_trajectories(
    objective: Objective,
    out: dict,
    hmax: int,
    nm_mapping: dict[str, list[str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(y, lower, upper) for the regulated channels over the scored horizon.

    Each is [b, P, n_regulated] in physical units, in `objective.regulated`
    order. A band with no lower (upper) bound reads -inf (+inf) there, so the
    excursion arithmetic needs no special case.
    """
    def column(name):
        return scored_series(objective, name, out, hmax, nm_mapping)

    y = torch.stack([column(b.target) for b in objective.bands], dim=-1)

    def edge(name, fill):
        return (torch.full_like(y[..., 0], fill) if name is None
                else column(name))

    lower = torch.stack([edge(b.lower, -torch.inf) for b in objective.bands], dim=-1)
    upper = torch.stack([edge(b.upper, torch.inf) for b in objective.bands], dim=-1)
    return y, lower, upper


@beartype
def evaluate_closed_loop(
    cfg,
    sysid_cfg,
    cl_system,
    test_dfs: list,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> tuple[ControlResults, dict]:
    """Roll the trained closed loop on the test split; pooled physical metrics.

    `test_dfs` is a list of one or more test dataframes. Windows
    from different files are pooled along the batch axis (no rollout spans
    two files). Metrics are computed once across the pooled rollouts.
    """
    data, source_idx = evaluation_windows(cfg, sysid_cfg, test_dfs, nm_mapping)
    cl_system.nsteps = eval_pred_steps(cfg, horizons(cfg, sysid_cfg)[2])
    t0 = tic()
    with torch.no_grad():
        out = cl_system(data)
    eval_time_s = tic() - t0
    # Plumb source_idx through `out` so plot_closed_loop can route per-batch.
    out["source_idx"] = source_idx
    return score_rollout(cfg, sysid_cfg, out, nm_mapping, norm_stats, eval_time_s), out


@beartype
def evaluation_windows(cfg, sysid_cfg, test_dfs: list,
                       nm_mapping: dict[str, list[str]]) -> tuple[dict, torch.Tensor]:
    """The test split's closed-loop windows, and which file each came from.

    The test split's reference columns are fixed data, so every run — and
    every controller scored on it — is scored against the same references.
    A closed loop rolls them out over `eval_pred_steps`.
    """
    _, _, hmax = horizons(cfg, sysid_cfg)
    data = build_window_dict_multi(
        test_dfs, cfg, sysid_cfg, nm_mapping, P=eval_pred_steps(cfg, hmax),
        stride=cfg.test_stride,
    )
    # source_idx is bookkeeping for plotting only; the closed-loop System
    # doesn't expect it among its inputs.
    return data, data.pop("source_idx")


@beartype
def score_rollout(
    cfg,
    sysid_cfg,
    out: dict,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
    eval_time_s: float,
    residual: bool = True,
) -> ControlResults:
    """Pooled physical metrics of one finished evaluation rollout. `residual`
    scores the deployed deviation from the reference rule, when the run has
    one; a controller that is no residual is not scored against a box."""
    hooks = load_control_hooks(cfg)
    _, _, hmax = horizons(cfg, sysid_cfg)
    objective = resolve_objective(cfg, nm_mapping)
    regulated = objective.regulated
    device = sysid_cfg.device

    y_ctrl, lower, upper = band_trajectories(objective, out, hmax, nm_mapping)
    step_hours = hours_per_step(out["t"][:, scored_steps("t", hmax), :])
    comfort = comfort_metrics(y_ctrl, lower, upper, regulated, step_hours)
    # Two pooled headline numbers across all regulated channels. The rate is
    # what an operator asks about; the exceedance is what to tune on, because a
    # threshold count saturates and, scored against a surrogate rather than the
    # plant, its argmin can select a substantially worse controller.
    excursion = (lower - y_ctrl).clamp_min(0.0) + (y_ctrl - upper).clamp_min(0.0)
    violation_rate = (excursion > 0).float().mean().item()
    mean_exceedance = excursion.mean().item()

    # The midpoint exists only for a band with both edges.
    two_sided = [j for j, b in enumerate(objective.bands)
                 if b.lower is not None and b.upper is not None]
    mid = 0.5 * (lower + upper)
    tracking = per_channel_metrics(y_ctrl[..., two_sided], mid[..., two_sided],
                                   [regulated[j] for j in two_sided])

    u_cols = nm_mapping["U"]
    U = out["U"][:, scored_steps("U", hmax), :]
    mean_abs_du = (U[:, 1:, :] - U[:, :-1, :]).abs().mean().item()

    # Per-control behaviour, in the same physical units as the commands. The
    # bounds come from the hook the policy is actually squashed into, so
    # "at bound" means the limit the policy was given, not the data's spread.
    u_lo, u_hi = (b.to(device=device, dtype=torch.float32)
                  for b in hooks.actuation_bounds(nm_mapping, norm_stats))
    actuation = actuation_metrics(U, u_cols, u_lo, u_hi)
    response = response_audit(
        y_ctrl, lower, upper, U, regulated, u_cols, u_lo, u_hi,
    )

    energy = None
    if getattr(hooks, "POWER_METRIC_IMPLEMENTED", False):
        power = hooks.power_metric(
            out["yn"][:, scored_steps("yn", hmax), :],
            out["U"][:, scored_steps("U", hmax), :],
            out["D"][:, scored_steps("D", hmax), :],
            nm_mapping,
        )
        # Each weight a power term carries (a tariff, a carbon series) rides
        # through the rollout untouched, so it is co-indexed with the power it
        # prices.
        weights = {
            term.weight: scored_series(objective, term.weight, out, hmax, nm_mapping)
            for term in objective.terms
            if term.column == POWER and term.weight is not None
        }
        energy = energy_metrics(power, step_hours, weights)

    # How much of each box a residual policy actually used. Replaying the
    # (memoryless) rule on the recorded trajectory gives the command it would
    # have issued at each step, so U - u_ref is the deviation that was deployed.
    deviations = []
    if residual and cfg.residual_delta:
        u_ref = reference_over_horizon(cfg, out, hmax, nm_mapping)
        for column, ref in u_ref.items():
            j = u_cols.index(column)
            deviations.append(residual_metrics(
                column, U[:, :, j], ref,
                tuple(float(side) for side in cfg.residual_delta[column]),
            ))

    return ControlResults(
        comfort_violation_rate=violation_rate,
        comfort_mean_exceedance=mean_exceedance,
        comfort=comfort,
        mean_abs_du=mean_abs_du,
        actuation=actuation,
        response=response,
        tracking=tracking,
        energy=energy,
        eval_time_s=eval_time_s,
        residual=deviations,
    )


# ── MLflow metric logging ────────────────────────────────────────


def log_control_results(logger, results: ControlResults) -> None:
    """Flatten closed-loop metrics into MLflow-safe metric keys + log dict.

    Mirrors modeling.pipeline.log_test_results.
    """
    if logger is None:
        return
    logger.log_metrics(_flat_control_metrics(results))
    logger.log_dict(_control_metrics_dict(results), name="control_metrics.json")


def _flat_control_metrics(results: ControlResults) -> dict:
    prefix = "control"
    flat = {
        f"{prefix}_violation_rate": results.comfort_violation_rate,
        f"{prefix}_mean_exceedance": results.comfort_mean_exceedance,
        f"{prefix}_mean_abs_du": results.mean_abs_du,
    }
    for m in results.comfort:
        safe = metric_key(m.name)
        flat[f"{prefix}_{safe}_violation_rate"] = m.violation_rate
        flat[f"{prefix}_{safe}_mean_exceedance"] = m.mean_exceedance
        flat[f"{prefix}_{safe}_rms_exceedance"] = m.rms_exceedance
        flat[f"{prefix}_{safe}_mean_violation"] = m.mean_violation
        flat[f"{prefix}_{safe}_max_violation"] = m.max_violation
        flat[f"{prefix}_{safe}_degree_hours"] = m.degree_hours
    for a in results.actuation:
        safe = metric_key(a.name)
        flat[f"{prefix}_{safe}_mean_abs_du"] = a.mean_abs_du
        flat[f"{prefix}_{safe}_frac_saturated"] = a.frac_saturated
    for r in results.response:
        safe = f"{metric_key(r.name)}_{r.direction}"
        flat[f"{prefix}_{safe}_actuated"] = r.frac_actuated
        flat[f"{prefix}_{safe}_resolved"] = r.frac_resolved
    if results.energy is not None:
        flat[f"{prefix}_mean_power"] = results.energy.mean_power
        flat[f"{prefix}_total_energy"] = results.energy.total_energy
        flat[f"{prefix}_peak_power"] = results.energy.peak_power
        flat[f"{prefix}_mean_window_peak"] = results.energy.mean_window_peak
        for name, value in results.energy.weighted_energy.items():
            flat[f"{prefix}_{metric_key(name)}_weighted_energy"] = value
    for r in results.residual:
        safe = metric_key(r.name)
        flat[f"{prefix}_{safe}_residual_below"] = r.below_cap
        flat[f"{prefix}_{safe}_residual_above"] = r.above_cap
        flat[f"{prefix}_{safe}_mean_abs_delta"] = r.mean_abs_delta
        # Absent for an edge at 0, where the fraction is vacuous.
        for side, frac in (("below", r.frac_at_below), ("above", r.frac_at_above)):
            if frac is not None:
                flat[f"{prefix}_{safe}_frac_delta_at_{side}"] = frac
    return flat


def _control_metrics_dict(results: ControlResults) -> dict:
    return {
        "violation_rate": results.comfort_violation_rate,
        "mean_exceedance": results.comfort_mean_exceedance,
        "mean_abs_du": results.mean_abs_du,
        "comfort": comfort_to_dict(results.comfort),
        "actuation": actuation_to_dict(results.actuation),
        "response": response_to_dict(results.response),
        "energy": None if results.energy is None else asdict(results.energy),
        "residual": [asdict(r) for r in results.residual],
        "tracking": metrics_to_dict(results.tracking),
    }


@rc_context(PLOT_STYLE)
@beartype
def plot_closed_loop(
    cfg,
    sysid_cfg,
    out,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
    plot_dir: Path,
    origins: list,
) -> None:
    """
    One figure per rollout window: the regulated channels and their bands,
    the commands, and (if present) metered power. Each window goes in its own
    day-stamped folder under `plot_dir` with the underlying trajectory CSV
    alongside, so restyling never requires re-running the rollout.

    `origins` is one wall-clock start timestamp per test file, in the order
    the files were loaded (`[df.index[0] for df in test_dfs]`). It is required
    rather than optional because `t` is *elapsed* seconds from its own file's
    first row, so without the origins every plot is stamped 1970 — see the
    timestamp block below.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    hooks = load_control_hooks(cfg)
    _, _, hmax = horizons(cfg, sysid_cfg)
    objective = resolve_objective(cfg, nm_mapping)
    regulated = objective.regulated
    u_cols = nm_mapping["U"]

    y, lower, upper = (x.cpu() for x in band_trajectories(
        objective, out, hmax, nm_mapping))
    # U[k] is the command that drove y[k] (see scored_steps).
    U = out["U"][:, scored_steps("U", hmax), :].cpu()

    # Context traces overlaid on the regulated axis, and the optional power
    # row. Both come from the use case — the library has no opinion about
    # which columns are interesting.
    d_cols, y_cols = nm_mapping["D"], nm_mapping["Y"]
    context = {
        name: out["D"][:, scored_steps("D", hmax), d_cols.index(name)].cpu()
        for name in hooks.plot_context_channels(nm_mapping)
    }
    power = {
        name: out["yn"][:, scored_steps("yn", hmax), y_cols.index(name)].clamp_min(0.0).cpu()
        for name in hooks.power_channels(nm_mapping)
    }

    # Reference columns that are not a band edge — a tariff, a cap — on their
    # own axis. A price trace next to the power trace is how load shifting
    # reads at a glance, and how a precool that never happened becomes obvious.
    edges = {name for b in objective.bands for name in (b.lower, b.upper)}
    traces = {
        name: scored_series(objective, name, out, hmax, nm_mapping).cpu()
        for name in objective.r_columns if name not in edges
    }

    # Per-step wall-clock timestamps. `t` is `sec_elapsed`, which `get_data`
    # rebases to zero at its file's first row, so it is elapsed seconds and
    # not POSIX seconds — reading it as POSIX stamps every figure 1970-01-01
    # and only preserves hour-of-day for files that happen to start at
    # midnight. `source_idx` says which file each window came from, so the
    # window's real clock is that file's origin plus its elapsed seconds.
    t_elapsed = out["t"][:, scored_steps("t", hmax), 0].cpu().numpy()
    source_idx = out["source_idx"].cpu().numpy()
    if len(origins) <= int(source_idx.max()):
        raise ValueError(
            f"plot_closed_loop got {len(origins)} origins but windows "
            f"reference file index {int(source_idx.max())}. Pass one origin "
            "per test file: [df.index[0] for df in test_dfs]."
        )
    P = y.shape[1]
    steps = list(range(P))
    dt_s = float(t_elapsed[0, 1] - t_elapsed[0, 0])
    steps_per_hour = max(1, int(round(3600.0 / dt_s)))
    tick_steps = list(range(0, P, steps_per_hour))
    tick_labels = [str(i // steps_per_hour) for i in tick_steps]

    plot_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 3 if power else 2
    n_windows = y.shape[0]
    for w in range(n_windows):
        ts = origins[int(source_idx[w])] + pd.to_timedelta(t_elapsed[w], unit="s")
        day_name = ts[0].strftime("%b_%d")
        date_str = ts[0].strftime("%b-%d")
        day_dir = unique_day_dir(plot_dir, day_name)
        day_dir.mkdir(parents=True, exist_ok=True)

        # Trajectory CSV alongside the figure so plots can be restyled
        # without re-running the rollout.
        csv_data = {}
        for j, band in enumerate(objective.bands):
            csv_data[band.target] = y[w, :, j].numpy()
            if band.lower is not None:
                csv_data[f"{band.target} lower"] = lower[w, :, j].numpy()
            if band.upper is not None:
                csv_data[f"{band.target} upper"] = upper[w, :, j].numpy()
        for j, name in enumerate(u_cols):
            csv_data[name] = U[w, :, j].numpy()
        for series in (context, power, traces):
            for name, values in series.items():
                csv_data[name] = values[w].numpy()
        pd.DataFrame(csv_data, index=ts).rename_axis("timestamp").to_csv(
            day_dir / f"closed_loop_{date_str}.csv"
        )

        fig, axes = plt.subplots(
            n_rows, 1, sharex=True, figsize=(10, 3 * n_rows),
        )
        ax0, ax1 = axes[0], axes[1]
        colors = [ax0.plot(steps, y[w, :, j], label=name, lw=1.0)[0].get_color()
                  for j, name in enumerate(regulated)]
        # One band when every regulated channel shares it; otherwise each
        # channel's own band, in its line's colour. A one-sided band is drawn
        # as its edge.
        shared = bool((lower[w] == lower[w, :, :1]).all()
                      and (upper[w] == upper[w, :, :1]).all())
        bands = ([(0, "green", "band")] if shared else
                 [(j, colors[j], f"{name} band") for j, name in enumerate(regulated)])
        for j, color, label in bands:
            lo, hi = lower[w, :, j], upper[w, :, j]
            if torch.isfinite(lo).all() and torch.isfinite(hi).all():
                ax0.fill_between(steps, lo, hi, color=color, alpha=0.12, label=label)
                continue
            for edge in (lo, hi):
                if torch.isfinite(edge).all():
                    ax0.plot(steps, edge, color=color, lw=0.8, linestyle=":",
                             label=label)
        for name, series in context.items():
            ax0.plot(steps, series[w], label=name, lw=0.9, linestyle="--")
        ax0.set_ylabel(axis_label(norm_stats, regulated, "Regulated"))
        ax0.set_title(f"Closed loop — {ts[0].strftime('%Y-%m-%d')}")
        ax0.legend(fontsize=6, ncol=2)
        for j, name in enumerate(u_cols):
            ax1.plot(steps, U[w, :, j], label=name, lw=1.0)
        ax1.set_ylabel(axis_label(norm_stats, u_cols, "Control input"))
        ax1.legend(fontsize=6, ncol=2)
        if power:
            ax2 = axes[2]
            for name, series in power.items():
                ax2.plot(steps, series[w], label=name, lw=1.0)
            ax2.set_ylabel(axis_label(norm_stats, list(power), "Power"))
            ax2.legend(fontsize=6, ncol=2)
            bottom_ax = ax2
        else:
            bottom_ax = ax1
        if traces:
            ax_ref = bottom_ax.twinx()
            for name, values in traces.items():
                ax_ref.plot(steps, values[w], lw=0.9, linestyle=":", label=name)
            ax_ref.set_ylabel(axis_label(norm_stats, list(traces), "Reference"))
            ax_ref.legend(fontsize=6, loc="upper right")
        bottom_ax.set_xlabel("Time [h]")
        bottom_ax.set_xticks(tick_steps)
        bottom_ax.set_xticklabels(tick_labels)
        fig.tight_layout()
        fig.savefig(day_dir / f"closed_loop_{date_str}.png", dpi=110)
        plt.close(fig)


def write_reportcard(markdown: str, out_dir: Path, logger=None) -> Path:
    """Write the control reportcard to <out_dir>/control_reportcard.md + MLflow."""
    return _write_reportcard(
        markdown, out_dir, "control_reportcard.md", logger=logger
    )
