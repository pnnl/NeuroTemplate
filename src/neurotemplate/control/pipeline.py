"""
Train the DPC policy, evaluate the closed loop, write a reportcard.

Mirrors src/pipeline.py in shape. Only the policy trains — the surrogate is
frozen, so the optimizer is built over requires_grad params only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter as tic
from typing import Optional

import torch

from neuromancer.trainer import Trainer

from neurotemplate.common.metrics import (
    ChannelMetrics,
    _safe_metric_key,
    metrics_to_dict,
    per_channel_metrics,
)
from neurotemplate.common.reportcard import write_reportcard as _write_reportcard
from neurotemplate.control.metrics import (
    ComfortMetrics,
    EnergyMetrics,
    comfort_metrics,
    comfort_to_dict,
    energy_metrics,
    hours_per_step,
)
from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.scenarios import horizons, build_window_dict_multi
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


# Fixed seed for the evaluation comfort-band realization. Keeps the headline
# violation metric reproducible and comparable across configs/surrogates.
EVAL_BAND_SEED = 12345


@dataclass
class ControlResults:
    comfort_violation_rate: float      # pooled fraction of (zone, step) outside band
    comfort: list[ComfortMetrics]      # per-zone comfort detail
    mean_abs_du: float                 # mean |Δsetpoint| (physical units)
    tracking: list[ChannelMetrics]     # per-zone error vs band midpoint
    energy: Optional[EnergyMetrics]    # None if power_metric not implemented
    eval_time_s: float

    @property
    def energy_proxy(self) -> Optional[float]:
        """Mean power, or None when the use case meters no power."""
        return None if self.energy is None else self.energy.mean_power


def _phys(norm_stats, cols, device):
    shift = torch.tensor([norm_stats[c]["shift"] for c in cols], device=device)
    stretch = torch.tensor([norm_stats[c]["stretch"] for c in cols],
                           device=device)
    return shift, stretch


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
        epoch_verbose=cfg.epoch_verbose,
        eval_metric="mean_dev_loss",
    )

    t0 = tic()
    best = trainer.train()
    trainer.model.load_state_dict(best)
    train_time_s = tic() - t0

    best_dev = trainer.best_devloss
    if hasattr(best_dev, "item"):
        best_dev = best_dev.item()

    cl_system = trainer.model.nodes[0]
    return cl_system, float(best_dev), train_time_s


# ── Evaluate (long closed-loop rollout) ──────────────────────────


@beartype
def evaluate_closed_loop(
    cfg,
    sysid_cfg,
    cl_system,
    test_dfs: list,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> tuple[ControlResults, dict]:
    """Roll the trained closed loop on the test split; pooled physical metrics.

    `test_dfs` is a list of one or more normalized test dataframes. Windows
    from different files are pooled along the batch axis (no rollout spans
    two files). Metrics are computed once across the pooled rollouts.
    """
    hooks = load_control_hooks(cfg)
    _, _, hmax = horizons(cfg, sysid_cfg)
    controlled, ctrl_idx = hooks.controlled_channels(nm_mapping)
    device = sysid_cfg.device

    # Deterministic comfort band for evaluation: a fixed seed (independent of
    # cfg.seed) so every config/run is scored against the *same* band
    # realization and results are reproducible & comparable.
    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(EVAL_BAND_SEED)
    # test_window is the total window (matches sys-id); rollout = test_window - hmax.
    eval_pred_steps = cfg.test_window - hmax
    if eval_pred_steps <= 0:
        raise ValueError(
            f"test_window ({cfg.test_window}) must be > hmax ({hmax}); "
            "otherwise there is no rollout after the warmup."
        )
    data = build_window_dict_multi(
        test_dfs, cfg, sysid_cfg, nm_mapping, norm_stats,
        P=eval_pred_steps, stride=cfg.test_stride,
        band=cfg.test_band(), generator=eval_gen,
    )
    # source_idx is bookkeeping for plotting only; the closed-loop System
    # doesn't expect it among its inputs.
    source_idx = data.pop("source_idx")

    cl_system.nsteps = eval_pred_steps
    t0 = tic()
    with torch.no_grad():
        out = cl_system(data)
    eval_time_s = tic() - t0
    # Plumb source_idx through `out` so plot_closed_loop can route per-batch.
    out["source_idx"] = source_idx

    y_shift, y_stretch = _phys(norm_stats, controlled, device)
    y_ctrl = out["yn"][:, hmax:, ctrl_idx] * y_stretch + y_shift  # [b,P,nc]
    ymin = data["ymin"][:, hmax:, :] * y_stretch + y_shift
    ymax = data["ymax"][:, hmax:, :] * y_stretch + y_shift

    step_hours = hours_per_step(out["t"][:, hmax:, :])
    comfort = comfort_metrics(y_ctrl, ymin, ymax, controlled, step_hours)
    # Pooled headline rate across all controlled channels.
    violation_rate = ((y_ctrl < ymin) | (y_ctrl > ymax)).float().mean().item()

    mid = 0.5 * (ymin + ymax)
    tracking = per_channel_metrics(y_ctrl, mid, controlled)

    u_cols = nm_mapping["U"]
    u_shift, u_stretch = _phys(norm_stats, u_cols, device)
    U = out["U"][:, hmax:, :] * u_stretch + u_shift
    mean_abs_du = (U[:, 1:, :] - U[:, :-1, :]).abs().mean().item()

    energy = None
    if getattr(hooks, "POWER_METRIC_IMPLEMENTED", False):
        power = hooks.power_metric(
            out["yn"][:, hmax:, :], out["U"][:, hmax:, :], out["D"][:, hmax:, :],
            nm_mapping, norm_stats,
        )
        energy = energy_metrics(power, step_hours)

    return ControlResults(
        comfort_violation_rate=violation_rate,
        comfort=comfort,
        mean_abs_du=mean_abs_du,
        tracking=tracking,
        energy=energy,
        eval_time_s=eval_time_s,
    ), out


# ── MLflow metric logging ────────────────────────────────────────


def log_control_results(
    logger,
    results: ControlResults,
    baseline: Optional[ControlResults] = None,
) -> None:
    """Flatten closed-loop metrics into MLflow-safe metric keys + log dict.

    Mirrors modeling.pipeline.log_test_results. Baseline metrics are logged
    under a `baseline_` prefix so a single run holds both sides of the
    comparison and they can be diffed across runs in the MLflow UI.
    """
    if logger is None:
        return

    flat = _flat_control_metrics(results, prefix="control")
    payload = {"learned": _control_metrics_dict(results)}
    if baseline is not None:
        flat.update(_flat_control_metrics(baseline, prefix="baseline"))
        payload["baseline"] = _control_metrics_dict(baseline)

    logger.log_metrics(flat)
    if hasattr(logger, "log_dict"):
        logger.log_dict(payload, name="control_metrics.json")


def _flat_control_metrics(results: ControlResults, prefix: str) -> dict:
    flat = {
        f"{prefix}_violation_rate": results.comfort_violation_rate,
        f"{prefix}_mean_abs_du": results.mean_abs_du,
    }
    for m in results.comfort:
        safe = _safe_metric_key(m.name)
        flat[f"{prefix}_{safe}_violation_rate"] = m.violation_rate
        flat[f"{prefix}_{safe}_mean_violation"] = m.mean_violation
        flat[f"{prefix}_{safe}_max_violation"] = m.max_violation
        flat[f"{prefix}_{safe}_degree_hours"] = m.degree_hours
    if results.energy is not None:
        flat[f"{prefix}_mean_power"] = results.energy.mean_power
        flat[f"{prefix}_total_energy"] = results.energy.total_energy
        flat[f"{prefix}_peak_power"] = results.energy.peak_power
        flat[f"{prefix}_mean_window_peak"] = results.energy.mean_window_peak
    return flat


def _control_metrics_dict(results: ControlResults) -> dict:
    return {
        "violation_rate": results.comfort_violation_rate,
        "mean_abs_du": results.mean_abs_du,
        "comfort": comfort_to_dict(results.comfort),
        "energy": None if results.energy is None else asdict(results.energy),
        "tracking": metrics_to_dict(results.tracking),
    }


# ── Rule-based baseline (optional) ───────────────────────────────


@beartype
def evaluate_baseline(
    cfg,
    sysid_cfg,
    dyn_node,
    test_dfs: list,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> Optional[ControlResults]:
    """Score the use case's rule-based controller on the same test rollouts.

    `control_user_defined.build_baseline_policy(cfg, sysid_cfg, nm_mapping,
    norm_stats) -> Node` is optional: without it there is simply nothing to
    compare against and this returns None. Same shape as the
    POWER_METRIC_IMPLEMENTED check — the hook's presence is the switch.

    The comparison is apples-to-apples because evaluate_closed_loop reseeds
    the band from EVAL_BAND_SEED, so the baseline sees a byte-identical
    comfort band on byte-identical test windows. The baseline is never
    trained: it has no trainable parameters by construction.
    """
    build = getattr(load_control_hooks(cfg), "build_baseline_policy", None)
    if build is None:
        return None

    baseline_node = build(cfg, sysid_cfg, nm_mapping, norm_stats)
    if baseline_node is None:
        return None

    cl = build_closed_loop(dyn_node, baseline_node, cfg, sysid_cfg)
    results, _ = evaluate_closed_loop(
        cfg, sysid_cfg, cl, test_dfs, nm_mapping, norm_stats,
    )
    return results


@beartype
def plot_closed_loop(
    cfg,
    sysid_cfg,
    out,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    plot_dir: Path,
) -> None:
    """
    One figure per rollout window: controlled zone temps + comfort band,
    setpoint(s), and (if present) fan/DX power. Each window goes in its own
    day-stamped folder under `plot_dir` with the underlying trajectory CSV
    alongside, so restyling never requires re-running the rollout.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from neurotemplate.modeling.plotting import _unique_day_dir

    hooks = load_control_hooks(cfg)
    _, _, hmax = horizons(cfg, sysid_cfg)
    controlled, ctrl_idx = hooks.controlled_channels(nm_mapping)
    device = sysid_cfg.device
    y_shift, y_stretch = _phys(norm_stats, controlled, device)
    u_cols = nm_mapping["U"]
    u_shift, u_stretch = _phys(norm_stats, u_cols, device)

    y = (out["yn"][:, hmax:, ctrl_idx] * y_stretch + y_shift).cpu()
    ymin = (out["ymin"][:, hmax:, :] * y_stretch + y_shift).cpu()
    ymax = (out["ymax"][:, hmax:, :] * y_stretch + y_shift).cpu()
    U = (out["U"][:, hmax:, :] * u_stretch + u_shift).cpu()

    # Context traces overlaid on the temperature axis (denormalized), and the
    # optional power row. Both come from the use case — the library has no
    # opinion about which columns are interesting.
    d_cols = nm_mapping["D"]
    context = {}
    for name in hooks.plot_context_channels(nm_mapping):
        s = norm_stats[name]
        idx = d_cols.index(name)
        context[name] = (
            out["D"][:, hmax:, idx] * s["stretch"] + s["shift"]
        ).cpu()

    y_cols = nm_mapping["Y"]
    power = {}
    for name in hooks.power_channels(nm_mapping):
        s = norm_stats[name]
        idx = y_cols.index(name)
        power[name] = (
            out["yn"][:, hmax:, idx] * s["stretch"] + s["shift"]
        ).clamp_min(0.0).cpu()

    # Per-step real timestamps (sec_elapsed is POSIX seconds; see
    # data_utils._engineer_features).
    t_unix = out["t"][:, hmax:, 0].cpu().numpy()
    P = y.shape[1]
    steps = list(range(P))
    dt_s = float(t_unix[0, 1] - t_unix[0, 0])
    steps_per_hour = max(1, int(round(3600.0 / dt_s)))
    tick_steps = list(range(0, P, steps_per_hour))
    tick_labels = [str(i // steps_per_hour) for i in tick_steps]

    plot_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 3 if power else 2
    n_windows = y.shape[0]
    for w in range(n_windows):
        ts = pd.to_datetime(t_unix[w], unit="s")
        day_name = ts[0].strftime("%b_%d")
        date_str = ts[0].strftime("%b-%d")
        day_dir = _unique_day_dir(plot_dir, day_name)
        day_dir.mkdir(parents=True, exist_ok=True)

        # Trajectory CSV alongside the figure so plots can be restyled
        # without re-running the rollout.
        csv_data = {}
        for j, name in enumerate(controlled):
            csv_data[name] = y[w, :, j].numpy()
            csv_data[f"{name} ymin"] = ymin[w, :, j].numpy()
            csv_data[f"{name} ymax"] = ymax[w, :, j].numpy()
        for j, name in enumerate(u_cols):
            csv_data[name] = U[w, :, j].numpy()
        for name, series in context.items():
            csv_data[name] = series[w].numpy()
        for name, series in power.items():
            csv_data[name] = series[w].numpy()
        pd.DataFrame(csv_data, index=ts).rename_axis("timestamp").to_csv(
            day_dir / f"closed_loop_{date_str}.csv"
        )

        fig, axes = plt.subplots(
            n_rows, 1, sharex=True, figsize=(10, 3 * n_rows),
        )
        ax0, ax1 = axes[0], axes[1]
        for j, name in enumerate(controlled):
            ax0.plot(steps, y[w, :, j], label=name, lw=1.0)
        ax0.fill_between(
            steps, ymin[w, :, 0], ymax[w, :, 0],
            color="green", alpha=0.12, label="comfort band",
        )
        for name, series in context.items():
            ax0.plot(steps, series[w], label=name, lw=0.9, linestyle="--")
        ax0.set_ylabel("Temp [°C]")
        ax0.set_title(f"Closed loop — {ts[0].strftime('%Y-%m-%d')}")
        ax0.legend(fontsize=6, ncol=2)
        for j, name in enumerate(u_cols):
            ax1.plot(steps, U[w, :, j], label=name, lw=1.0)
        ax1.set_ylabel("Setpoint [°C]")
        ax1.legend(fontsize=6, ncol=2)
        if power:
            ax2 = axes[2]
            for name, series in power.items():
                ax2.plot(steps, series[w], label=name, lw=1.0)
            ax2.set_ylabel("Power")
            ax2.legend(fontsize=6, ncol=2)
            bottom_ax = ax2
        else:
            bottom_ax = ax1
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
