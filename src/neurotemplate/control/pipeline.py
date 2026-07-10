"""
Train the DPC policy, evaluate the closed loop, write a reportcard.

Mirrors src/pipeline.py in shape. Only the policy trains — the surrogate is
frozen, so the optimizer is built over requires_grad params only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter as tic
from typing import Optional

import torch

from neuromancer.trainer import Trainer

from neurotemplate.common.metrics import ChannelMetrics, per_channel_metrics, metrics_to_markdown
from neurotemplate.control.scenarios import horizons, build_window_dict_multi
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


# Fixed seed for the evaluation comfort-band realization. Keeps the headline
# violation metric reproducible and comparable across configs/surrogates.
EVAL_BAND_SEED = 12345


@dataclass
class ControlResults:
    comfort_violation_rate: float      # fraction of (zone, step) outside band
    mean_abs_du: float                 # mean |Δsetpoint| (physical units)
    tracking: list[ChannelMetrics]     # per-zone error vs band midpoint
    energy_proxy: Optional[float]      # mean power if power_metric exists
    eval_time_s: float


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

    below = y_ctrl < ymin
    above = y_ctrl > ymax
    violation_rate = (below | above).float().mean().item()

    mid = 0.5 * (ymin + ymax)
    tracking = per_channel_metrics(y_ctrl, mid, controlled)

    u_cols = nm_mapping["U"]
    u_shift, u_stretch = _phys(norm_stats, u_cols, device)
    U = out["U"][:, hmax:, :] * u_stretch + u_shift
    mean_abs_du = (U[:, 1:, :] - U[:, :-1, :]).abs().mean().item()

    energy_proxy = None
    if getattr(hooks, "POWER_METRIC_IMPLEMENTED", False):
        power = hooks.power_metric(
            out["yn"][:, hmax:, :], out["U"][:, hmax:, :], out["D"][:, hmax:, :],
            nm_mapping, norm_stats,
        )
        energy_proxy = power.mean().item()

    return ControlResults(
        comfort_violation_rate=violation_rate,
        mean_abs_du=mean_abs_du,
        tracking=tracking,
        energy_proxy=energy_proxy,
        eval_time_s=eval_time_s,
    ), out


# ── Reportcard ───────────────────────────────────────────────────


@beartype
def render_control_reportcard(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    results: ControlResults,
    *,
    best_dev_loss: Optional[float] = None,
    train_time_s: Optional[float] = None,
) -> str:
    h_sys, h_ctrl, hmax = horizons(cfg, sysid_cfg)
    controlled, _ = load_control_hooks(cfg).controlled_channels(nm_mapping)
    energy = (
        f"{results.energy_proxy:.4g}" if results.energy_proxy is not None
        else "n/a (power_metric not implemented)"
    )
    lines = [
        f"# DPC Control Reportcard — {cfg.run}",
        "",
        "## Setup",
        f"- Sys-id run: `{cfg.sysid_run_id}`  (model: {sysid_cfg.model})",
        f"- Horizons: H_sys={h_sys}, H_ctrl={h_ctrl}, "
        f"train_pred_steps={cfg.train_pred_steps}, test_window={cfg.test_window}",
        f"- Controlled channels ({len(controlled)}): "
        f"{', '.join(controlled)}",
        f"- Control U ({len(nm_mapping['U'])}): "
        f"{', '.join(nm_mapping['U'])}",
        f"- Train band: lo ~ U[{cfg.train_band().lo}, "
        f"{cfg.train_band().hi}] °C, width ~ U[{cfg.train_band().width_min}, "
        f"{cfg.train_band().width_max}] °C, segments={cfg.train_band().sub_samples}",
        f"- Test band:  lo ~ U[{cfg.test_band().lo}, "
        f"{cfg.test_band().hi}] °C, width ~ U[{cfg.test_band().width_min}, "
        f"{cfg.test_band().width_max}] °C, segments={cfg.test_band().sub_samples}",
        f"- Weights: comfort={cfg.comfort_weight}, du={cfg.du_weight}, "
        f"effort={cfg.effort_weight}",
        "",
        "## Closed-loop performance (physical units)",
        f"- Comfort violation rate: "
        f"**{results.comfort_violation_rate * 100:.2f}%**",
        f"- Mean |Δsetpoint|: {results.mean_abs_du:.4g}",
        f"- Energy proxy: {energy}",
        f"- Eval rollout time: {results.eval_time_s:.2f}s",
    ]
    if best_dev_loss is not None:
        lines.append(f"- Best dev loss: {best_dev_loss:.6g}")
    if train_time_s is not None:
        lines.append(f"- Train time: {train_time_s:.1f}s")
    lines += [
        "",
        "## Tracking error vs. comfort-band midpoint",
        metrics_to_markdown(results.tracking),
    ]
    return "\n".join(lines)


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

    # Outdoor dry-bulb temp for context (denormalized).
    d_cols = nm_mapping["D"]
    od_idx = next(
        (i for i, c in enumerate(d_cols)
         if "outdoor" in c.lower() and "dry-bulb" in c.lower()),
        None,
    )
    if od_idx is not None:
        od_col = d_cols[od_idx]
        s = norm_stats[od_col]
        D_od = (out["D"][:, hmax:, od_idx] * s["stretch"] + s["shift"]).cpu()
    else:
        od_col, D_od = None, None

    # Optional third row: HVAC fan + DX power (denormalized W).
    if hooks.has_power_channels(nm_mapping):
        fan_idx, dx_idx = hooks._power_channel_indices(nm_mapping)
        y_cols = nm_mapping["Y"]
        fan_name, dx_name = y_cols[fan_idx], y_cols[dx_idx]
        s_fan, s_dx = norm_stats[fan_name], norm_stats[dx_name]
        P_fan = (out["yn"][:, hmax:, fan_idx] * s_fan["stretch"]
                 + s_fan["shift"]).clamp_min(0.0).cpu()
        P_dx = (out["yn"][:, hmax:, dx_idx] * s_dx["stretch"]
                + s_dx["shift"]).clamp_min(0.0).cpu()
    else:
        fan_name = dx_name = None
        P_fan = P_dx = None

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
    n_rows = 3 if P_fan is not None else 2
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
        if D_od is not None:
            csv_data[od_col] = D_od[w].numpy()
        if P_fan is not None:
            csv_data[fan_name] = P_fan[w].numpy()
            csv_data[dx_name] = P_dx[w].numpy()
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
        if D_od is not None:
            ax0.plot(steps, D_od[w], label=od_col, lw=0.9,
                     color="tab:orange", linestyle="--")
        ax0.set_ylabel("Temp [°C]")
        ax0.set_title(f"Closed loop — {ts[0].strftime('%Y-%m-%d')}")
        ax0.legend(fontsize=6, ncol=2)
        for j, name in enumerate(u_cols):
            ax1.plot(steps, U[w, :, j], label=name, lw=1.0)
        ax1.set_ylabel("Setpoint [°C]")
        ax1.legend(fontsize=6, ncol=2)
        if P_fan is not None:
            ax2 = axes[2]
            ax2.plot(steps, P_fan[w], label=fan_name, lw=1.0)
            ax2.plot(steps, P_dx[w], label=dx_name, lw=1.0)
            ax2.set_ylabel("Power [W]")
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
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "control_reportcard.md"
    path.write_text(markdown)
    if logger is not None and hasattr(logger, "_mlflow"):
        logger._mlflow.log_text(markdown, "control_reportcard.md")
    return path
