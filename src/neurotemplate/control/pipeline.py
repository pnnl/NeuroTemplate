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
    ActuationMetrics,
    ComfortMetrics,
    EnergyMetrics,
    JumpResponse,
    ResidualMetrics,
    SATURATION_TOL,
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
from neurotemplate.control.closed_loop import build_closed_loop
from neurotemplate.control.policy import (
    build_reference_node,
    reference_over_horizon,
    regulated_channels,
)
from neurotemplate.control.scenarios import (
    build_window_dict_multi,
    horizons,
    price_scale,
)
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


# Fixed seed for the evaluation comfort-band realization. Keeps the headline
# violation metric reproducible and comparable across configs/surrogates.
EVAL_BAND_SEED = 12345


@dataclass
class ControlResults:
    comfort_violation_rate: float      # pooled fraction of (zone, step) outside band
    comfort_mean_exceedance: float     # pooled mean excursion over ALL steps
    comfort: list[ComfortMetrics]      # per-zone comfort detail
    mean_abs_du: float                 # mean |Δsetpoint| (physical units)
    actuation: list[ActuationMetrics]  # per-control effort + saturation
    response: list[JumpResponse]       # reaction to reference-band jumps
    tracking: list[ChannelMetrics]     # per-zone error vs band midpoint
    energy: Optional[EnergyMetrics]    # None if power_metric not implemented
    eval_time_s: float
    residual: Optional[ResidualMetrics] = None   # None unless residual_delta

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
    regulated, reg_idx = regulated_channels(hooks, nm_mapping)
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
        band=cfg.test_band(), price_path=cfg.test_price_path, generator=eval_gen,
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

    y_shift, y_stretch = _phys(norm_stats, regulated, device)
    y_ctrl = out["yn"][:, hmax:, reg_idx] * y_stretch + y_shift  # [b,P,nc]
    ymin = data["ymin"][:, hmax:, :] * y_stretch + y_shift
    ymax = data["ymax"][:, hmax:, :] * y_stretch + y_shift

    step_hours = hours_per_step(out["t"][:, hmax:, :])
    comfort = comfort_metrics(y_ctrl, ymin, ymax, regulated, step_hours)
    # Two pooled headline numbers across all regulated channels. The rate is
    # what an operator asks about; the exceedance is what to tune on, because a
    # threshold count saturates and, scored against a surrogate rather than the
    # plant, its argmin can select a substantially worse controller.
    excursion = (ymin - y_ctrl).clamp_min(0.0) + (y_ctrl - ymax).clamp_min(0.0)
    violation_rate = (excursion > 0).float().mean().item()
    mean_exceedance = excursion.mean().item()

    mid = 0.5 * (ymin + ymax)
    tracking = per_channel_metrics(y_ctrl, mid, regulated)

    u_cols = nm_mapping["U"]
    u_shift, u_stretch = _phys(norm_stats, u_cols, device)
    U = out["U"][:, hmax:, :] * u_stretch + u_shift
    mean_abs_du = (U[:, 1:, :] - U[:, :-1, :]).abs().mean().item()

    # Per-control behaviour, in the same physical units as the commands. The
    # bounds come from the hook the policy is actually squashed into, so
    # "at bound" means the limit the policy was given, not the data's spread.
    umin_n, umax_n = hooks.actuation_bounds(nm_mapping, norm_stats)
    u_lo = umin_n.to(device) * u_stretch + u_shift
    u_hi = umax_n.to(device) * u_stretch + u_shift
    actuation = actuation_metrics(U, u_cols, u_lo, u_hi)
    response = response_audit(
        y_ctrl, ymin, ymax, U, regulated, u_cols, u_lo, u_hi,
    )

    energy = None
    if getattr(hooks, "POWER_METRIC_IMPLEMENTED", False):
        power = hooks.power_metric(
            out["yn"][:, hmax:, :], out["U"][:, hmax:, :], out["D"][:, hmax:, :],
            nm_mapping, norm_stats,
        )
        # The tariff rides through the rollout untouched, so it is already
        # co-indexed with the power it prices.
        price = (None if "price" not in out else
                 out["price"][:, hmax:, 0] * price_scale(cfg))
        energy = energy_metrics(power, step_hours, price)

    # How much of the ±Δ box a residual policy actually used. Replaying the
    # (memoryless) rule on the recorded trajectory gives the command it would
    # have issued at each step, so U - u_ref is the deviation that was deployed.
    residual = None
    if cfg.residual_delta is not None:
        u_ref = reference_over_horizon(cfg, out, hmax, nm_mapping, norm_stats)
        tol = (0.5 * cfg.setpoint_step if cfg.setpoint_step > 0
               else SATURATION_TOL * max(cfg.residual_delta, 1e-12))
        residual = residual_metrics(
            U, u_ref * u_stretch + u_shift, cfg.residual_delta, tol,
        )

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
        residual=residual,
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
        flat["control_clears_reference"] = float(
            clears_reference(results, baseline)
        )

    logger.log_metrics(flat)
    if hasattr(logger, "log_dict"):
        logger.log_dict(payload, name="control_metrics.json")


def _flat_control_metrics(results: ControlResults, prefix: str) -> dict:
    flat = {
        f"{prefix}_violation_rate": results.comfort_violation_rate,
        f"{prefix}_mean_exceedance": results.comfort_mean_exceedance,
        f"{prefix}_mean_abs_du": results.mean_abs_du,
    }
    for m in results.comfort:
        safe = _safe_metric_key(m.name)
        flat[f"{prefix}_{safe}_violation_rate"] = m.violation_rate
        flat[f"{prefix}_{safe}_mean_exceedance"] = m.mean_exceedance
        flat[f"{prefix}_{safe}_rms_exceedance"] = m.rms_exceedance
        flat[f"{prefix}_{safe}_mean_violation"] = m.mean_violation
        flat[f"{prefix}_{safe}_max_violation"] = m.max_violation
        flat[f"{prefix}_{safe}_degree_hours"] = m.degree_hours
    for a in results.actuation:
        safe = _safe_metric_key(a.name)
        flat[f"{prefix}_{safe}_mean_abs_du"] = a.mean_abs_du
        flat[f"{prefix}_{safe}_frac_saturated"] = a.frac_saturated
    for r in results.response:
        safe = f"{_safe_metric_key(r.name)}_{r.direction}"
        flat[f"{prefix}_{safe}_actuated"] = r.frac_actuated
        flat[f"{prefix}_{safe}_resolved"] = r.frac_resolved
    if results.energy is not None:
        flat[f"{prefix}_mean_power"] = results.energy.mean_power
        flat[f"{prefix}_total_energy"] = results.energy.total_energy
        flat[f"{prefix}_peak_power"] = results.energy.peak_power
        flat[f"{prefix}_mean_window_peak"] = results.energy.mean_window_peak
        if results.energy.energy_cost is not None:
            flat[f"{prefix}_energy_cost"] = results.energy.energy_cost
    if results.residual is not None:
        flat[f"{prefix}_residual_delta"] = results.residual.delta_cap
        flat[f"{prefix}_mean_abs_delta"] = results.residual.mean_abs_delta
        # Absent at Δ = 0, where a zero-width box makes the fraction vacuous.
        if results.residual.frac_at_cap is not None:
            flat[f"{prefix}_frac_delta_at_cap"] = results.residual.frac_at_cap
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
        "residual": (None if results.residual is None
                     else asdict(results.residual)),
        "tracking": metrics_to_dict(results.tracking),
    }


# ── Rule-based baseline (optional) ───────────────────────────────


@beartype
def baseline_node(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> Optional[object]:
    """The rule-based controller to score alongside the learned policy.

    Two hooks can supply one, and a use case must not define both:

      `reference_control` — the memoryless rule, and the one to prefer. It is
      also the residual policy's anchor, so using it here makes the baseline
      column and the δ = 0 floor the same controller by construction.
      `build_baseline_policy` — the escape hatch for a rule a memoryless
      function cannot express. Returns a Node, or None to skip the comparison.

    Neither defined means there is simply nothing to compare against.
    """
    hooks = load_control_hooks(cfg)
    rule = getattr(hooks, "reference_control", None)
    build = getattr(hooks, "build_baseline_policy", None)

    if rule is not None and build is not None:
        raise ValueError(
            "control_user_defined defines both reference_control and "
            "build_baseline_policy, so there are two definitions of one rule "
            "and two places for a cooling inset to drift apart. Keep "
            "reference_control (the residual policy is anchored to it) and "
            "delete the other — or keep build_baseline_policy if the rule "
            "genuinely cannot be written memorylessly."
        )
    if rule is not None:
        return build_reference_node(cfg, sysid_cfg, nm_mapping, norm_stats)
    return None if build is None else build(cfg, sysid_cfg, nm_mapping,
                                            norm_stats)


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

    The comparison is apples-to-apples because evaluate_closed_loop reseeds
    the band from EVAL_BAND_SEED, so the baseline sees a byte-identical
    comfort band on byte-identical test windows. The baseline is never
    trained: it has no trainable parameters by construction.

    With a residual policy this is also the δ = 0 score, which is what turns
    the residual's floor from rhetorical into something you can decide on —
    see `clears_reference`.
    """
    node = baseline_node(cfg, sysid_cfg, nm_mapping, norm_stats)
    if node is None:
        return None

    cl = build_closed_loop(dyn_node, node, cfg, sysid_cfg)
    results, _ = evaluate_closed_loop(
        cfg, sysid_cfg, cl, test_dfs, nm_mapping, norm_stats,
    )
    return results


@beartype
def clears_reference(results: ControlResults,
                     baseline: Optional[ControlResults]) -> Optional[bool]:
    """Did the learned policy beat the rule it is anchored to? None if no rule.

    Scored on mean exceedance rather than violation rate: a threshold count
    saturates, and measured against a surrogate rather than the plant its
    argmin can select a substantially worse controller.

    Why this needs saying at all: the floor a residual promises is not
    automatic. Neuromancer's Trainer checkpoints the best model after each
    epoch's optimizer steps, so the δ = 0 point is never itself a candidate —
    by the first evaluation δ is already non-zero. `evaluate_baseline` scores
    that point explicitly on a byte-identical band; this is the decision rule
    that makes the score mean something. **A submission that does not clear
    its own reference should ship the reference.**
    """
    if baseline is None:
        return None
    return results.comfort_mean_exceedance <= baseline.comfort_mean_exceedance


@beartype
def plot_closed_loop(
    cfg,
    sysid_cfg,
    out,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    plot_dir: Path,
    origins: list,
) -> None:
    """
    One figure per rollout window: regulated zone temps + comfort band,
    setpoint(s), and (if present) fan/DX power. Each window goes in its own
    day-stamped folder under `plot_dir` with the underlying trajectory CSV
    alongside, so restyling never requires re-running the rollout.

    `origins` is one wall-clock start timestamp per test file, in the order
    the files were loaded (`[df.index[0] for df in test_dfs]`). It is required
    rather than optional because `t` is *elapsed* seconds from its own file's
    first row, so without the origins every plot is stamped 1970 — see the
    timestamp block below.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from neurotemplate.modeling.plotting import _unique_day_dir

    hooks = load_control_hooks(cfg)
    _, _, hmax = horizons(cfg, sysid_cfg)
    regulated, reg_idx = regulated_channels(hooks, nm_mapping)
    device = sysid_cfg.device
    y_shift, y_stretch = _phys(norm_stats, regulated, device)
    u_cols = nm_mapping["U"]
    u_shift, u_stretch = _phys(norm_stats, u_cols, device)

    y = (out["yn"][:, hmax:, reg_idx] * y_stretch + y_shift).cpu()
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

    # The tariff, in dollars, on its own axis. A price trace next to the power
    # trace is how load shifting reads at a glance — and how a precool that
    # never happened becomes obvious.
    scale = price_scale(cfg)
    price = (None if "price" not in out else
             (out["price"][:, hmax:, 0] * scale).cpu())

    # Per-step wall-clock timestamps. `t` is `sec_elapsed`, which `get_data`
    # rebases to zero at its file's first row, so it is elapsed seconds and
    # not POSIX seconds — reading it as POSIX stamps every figure 1970-01-01
    # and only preserves hour-of-day for files that happen to start at
    # midnight. `source_idx` says which file each window came from, so the
    # window's real clock is that file's origin plus its elapsed seconds.
    t_elapsed = out["t"][:, hmax:, 0].cpu().numpy()
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
        day_dir = _unique_day_dir(plot_dir, day_name)
        day_dir.mkdir(parents=True, exist_ok=True)

        # Trajectory CSV alongside the figure so plots can be restyled
        # without re-running the rollout.
        csv_data = {}
        for j, name in enumerate(regulated):
            csv_data[name] = y[w, :, j].numpy()
            csv_data[f"{name} ymin"] = ymin[w, :, j].numpy()
            csv_data[f"{name} ymax"] = ymax[w, :, j].numpy()
        for j, name in enumerate(u_cols):
            csv_data[name] = U[w, :, j].numpy()
        for name, series in context.items():
            csv_data[name] = series[w].numpy()
        for name, series in power.items():
            csv_data[name] = series[w].numpy()
        if price is not None:
            csv_data["price"] = price[w].numpy()
        pd.DataFrame(csv_data, index=ts).rename_axis("timestamp").to_csv(
            day_dir / f"closed_loop_{date_str}.csv"
        )

        fig, axes = plt.subplots(
            n_rows, 1, sharex=True, figsize=(10, 3 * n_rows),
        )
        ax0, ax1 = axes[0], axes[1]
        for j, name in enumerate(regulated):
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
        if price is not None:
            ax_price = bottom_ax.twinx()
            ax_price.plot(steps, price[w], color="grey", lw=0.9,
                          linestyle=":", label="price")
            ax_price.set_ylabel("Price [$/kWh]")
            ax_price.legend(fontsize=6, loc="upper right")
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
