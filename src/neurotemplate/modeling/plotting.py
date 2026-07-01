import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

from neurotemplate.hooks import load_modeling_hooks

from neurotemplate._typing import beartype

# ---------------------- Plotting ---------------------- #

# Publication-friendly defaults. LINE_THICKNESS is applied as the default
# matplotlib line width; individual calls can still override via linewidth=.

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'


# Sanitize a string for filesystem-friendly filenames.
@beartype
def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)


# Find a unique day-based subdirectory within base_dir.
# If "May_14" exists, tries "May_14_1", "May_14_2", etc.
@beartype
def _unique_day_dir(base_dir: Path, day_name: str) -> Path:
    candidate = base_dir / day_name
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = base_dir / f"{day_name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


# Plot predicted vs. true signals and log artifacts when a logger is provided.
@beartype
def plot_predictions(
    cfg,
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    run_dir: str,
    verbose: bool = False,
    logger: Optional[object] = None,
) -> None:
    sub_plot = load_modeling_hooks(cfg).sub_plot

    def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
        if logger is None:
            return
        mlflow_mod = getattr(logger, "_mlflow", None)
        if mlflow_mod is not None:
            if artifact_path is None:
                mlflow_mod.log_artifact(path)
            else:
                mlflow_mod.log_artifact(path, artifact_path=artifact_path)
            return
        if hasattr(logger, "log_artifact"):
            if artifact_path is None:
                logger.log_artifact(path)
            else:
                logger.log_artifact(path, artifact_path=artifact_path)

    if not run_dir:
        raise ValueError("run_dir must be provided for plot outputs.")
    base_dir = Path(run_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    df_pred['timestamp'] = df_true.index[0] + pd.to_timedelta(df_pred['sec_elapsed'],unit='s')
    df_pred = df_pred.set_index('timestamp')

    common_index = df_true.index.intersection(df_pred.index)
    if len(common_index) == 0:
        raise ValueError("No overlapping timestamps between true and predicted series.")

    df_true = df_true.loc[common_index]
    df_pred = df_pred.loc[common_index]

    # Organize plots into day-based subdirectories (e.g., May_14/, May_14_1/)
    day_name = df_true.index[0].strftime('%b_%d')
    plot_dir = _unique_day_dir(base_dir, day_name)
    plot_dir.mkdir(parents=True, exist_ok=True)

    date_str = df_true.index[0].strftime('%b-%d')

    data_dir = plot_dir / "pred_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{date_str}.csv"
    df_pred.to_csv(csv_path)
    log_artifact(str(csv_path), artifact_path="predictions")

    for col in df_true.columns:
        if (col not in df_pred.columns) or (col == 'sec_elapsed'):
            continue

        col_safe = safe_name(str(col))

        pred = df_pred[col].values
        true = df_true[col].values
        ss_res = ((true - pred) ** 2).sum()
        ss_tot = ((true - true.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(((true - pred) ** 2).mean())

        sub = sub_plot(df_true, col)

        # Full validation horizon plot
        fig = plt.figure()
        if sub is None:
            ax = plt.subplot(111)
        else:
            # unpack column names and options
            sub_cols, sub_opts = sub
            ax = plt.subplot(211)
            ax2 = plt.subplot(212, sharex=ax)

        ax.plot(df_pred.index, df_pred[col], "k-", label="Predicted")
        ax.plot(df_true.index, df_true[col], "r--", label="True")
        ax.grid(True, alpha=0.3)

        # Lower Plot, shares x-axis
        if sub is not None:
            for i, s_c in enumerate(sub_cols):
                if s_c == col:
                    # Don't replot the same column
                    continue

                if s_c not in df_true:
                    raise ValueError(f"Subplot Column {s_c} not available!")

                ax2.plot(df_true.index, df_true[s_c], **sub_opts[i])
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        ax.set_title(f"{col} — RMSE: {rmse:.3f} - R²: {r2:.3f} ")
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()

        full_path = plot_dir / f"{col_safe}_{date_str}.pdf"
        fig.savefig(full_path)
        log_artifact(str(full_path), artifact_path=f"plots/{plot_dir.stem}")
        plt.close(fig)


# Plot synthetic scenario results — one figure per scenario, all zones together.
@beartype
def plot_synthetic(
    results: list[dict],
    run_dir: str,
    freq_s: float = 300.0,
    n_plot: int = 4,
    verbose: bool = False,
    temp_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Args:
        results: List of dicts from scenario.evaluate(), one per scenario type.
        run_dir: Directory to save plots.
        freq_s: Seconds per timestep, used for x-axis labeling.
        n_plot: Max number of scenario instances to plot per figure.
        temp_range: (T_min, T_max) baseline for y-axis. Expanded if data exceeds it.
    """
    plot_dir = Path(run_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        zone_names = result["zone_names"]
        name = result["name"]

        # Step response has modes (cooling/heating); others are a single entry.
        if "modes" in result:
            entries = [
                (mode_name, mode_data)
                for mode_name, mode_data in result["modes"].items()
            ]
        else:
            entries = [(None, result)]

        for mode_name, entry in entries:
            losses = entry["losses"]
            preds = entry["predictions"]
            u_traj = entry.get("u_traj")
            outdoor_temp_traj = entry.get("outdoor_temp_traj")

            # has_u = u_traj is not None
            has_u = False
            has_outdoor = outdoor_temp_traj is not None
            n_scenarios = preds.shape[0]
            total_steps = preds.shape[1]
            n_show = min(n_plot, n_scenarios)

            if len(losses) != n_scenarios:
                raise ValueError(
                    f"losses length ({len(losses)}) != batch dim ({n_scenarios}) "
                    f"for {name}" + (f" mode {mode_name}" if mode_name else "")
                )

            t_minutes = np.arange(total_steps) * freq_s / 60.0
            # D/U tensors have total_steps-1 entries; align with the first
            # total_steps-1 prediction steps (no input for the final step).
            t_input = t_minutes[:-1]

            seed_steps = result.get("seed_steps")

            # One figure per scenario instance
            for row in range(n_show):
                fig, ax = plt.subplots(1, 1)

                title = name
                if mode_name:
                    title += f" — {mode_name.replace('_', ' ').title()}"
                title += f" — Loss: {losses[row]:.4f}"
                fig.suptitle(title, fontsize=14)

                # Zone temp trajectories (solid)
                for z in range(len(zone_names)):
                    ax.plot(t_minutes, preds[row, :, z].numpy(), "-")

                # Vertical line at seed/prediction boundary
                if seed_steps is not None:
                    ax.axvline(
                        t_minutes[seed_steps - 1], color="gray",
                        linestyle="--", linewidth=1, label="Prediction start",
                    )

                # Outdoor temp: plot the actual tensor fed to the model
                if has_outdoor:
                    ax.step(t_input, outdoor_temp_traj[row].numpy(),
                            where="post", color="blue", linestyle=":", alpha=0.6)

                ax.set_ylabel("Temp [°C]")
                ax.set_xlabel("Time (minutes)")
                ax.grid(True, alpha=0.3)

                # Control inputs on twin axis (dashed, no labels)
                if has_u:
                    ax2 = ax.twinx()
                    for u_idx in range(u_traj.shape[2]):
                        ax2.step(t_input, u_traj[row, :, u_idx].numpy(),
                                 where="post", alpha=0.4, linestyle="--",
                                 color="green")
                    ax2.tick_params(right=False, labelright=False)

                # Legend
                legend_handles = [
                    plt.Line2D([], [], color="black", linestyle="-", label="Zone temps"),
                ]
                if seed_steps is not None:
                    legend_handles.append(
                        plt.Line2D([], [], color="gray", linestyle="--", linewidth=1,
                                   label="Prediction start"),
                    )
                if has_outdoor:
                    legend_handles.append(
                        plt.Line2D([], [], color="blue", linestyle=":", alpha=0.6,
                                   label="Outdoor temp"),
                    )
                if has_u:
                    legend_handles.append(
                        plt.Line2D([], [], color="green", linestyle="--", alpha=0.4,
                                   label="Control inputs"),
                    )
                ax.legend(handles=legend_handles, loc="best", fontsize=10)

                # Standardize y-axis: temp_range as baseline, expand for data
                if temp_range is not None:
                    data_min = preds[row].min().item()
                    data_max = preds[row].max().item()
                    if has_outdoor:
                        data_min = min(data_min, outdoor_temp_traj[row].min().item())
                        data_max = max(data_max, outdoor_temp_traj[row].max().item())
                    margin = 0.5
                    y_low = min(temp_range[0], data_min) - margin
                    y_high = max(temp_range[1], data_max) + margin
                    ax.set_ylim(y_low, y_high)

                fig.tight_layout()
                fname = safe_name(f"{name}_{mode_name}" if mode_name else name)
                save_path = plot_dir / f"{fname}_{row}.pdf"
                fig.savefig(save_path)
                if verbose:
                    print(f"Saved synthetic plot: {save_path}")
                plt.close(fig)

                # Save actual values to CSV alongside the plot
                csv_data = {"time_min": t_minutes}
                for z, zn in enumerate(zone_names):
                    csv_data[zn] = preds[row, :, z].numpy()
                if has_outdoor:
                    # Pad to match total_steps (inputs are 1 shorter)
                    outdoor_vals = outdoor_temp_traj[row].numpy()
                    csv_data["outdoor_temp"] = np.append(outdoor_vals, np.nan)
                if has_u:
                    u_names = result.get("u_names") or entry.get("u_names")
                    for u_idx in range(u_traj.shape[2]):
                        u_label = u_names[u_idx] if u_names else f"u_{u_idx}"
                        csv_data[u_label] = np.append(
                            u_traj[row, :, u_idx].numpy(), np.nan
                        )
                data_dir = plot_dir / "synth_data"
                data_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(csv_data).to_csv(
                    data_dir / f"{fname}_{row}.csv", index=False
                )
