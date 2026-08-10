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


def plot_synthetic(
    traces: list[dict],
    run_dir: str,
    freq_s: float = 300.0,
    n_plot: int = 4,
    verbose: bool = False,
    temp_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    One figure per synthetic-scenario instance: the state channels against the
    ambient signal they were driven with, with the seed/rollout boundary
    marked. The verdicts live in the reportcard; these are for looking at the
    shape when a verdict is surprising.

    Args:
        traces: dicts from `scenario.trace(n)`; empty ones are skipped.
        run_dir: directory for the figures (CSVs go in `synth_data/` beside).
        freq_s: seconds per timestep, for the x axis.
        n_plot: max instances drawn per scenario.
        temp_range: (lo, hi) baseline for the y axis; expanded to fit the data.
    """
    plot_dir = Path(run_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for trace in traces:
        if not trace:
            continue
        name = trace["name"]
        preds = trace["predictions"]
        ambient = trace["ambient_traj"]
        channels = trace["channel_names"]
        states = trace["state_names"]
        seed_steps = trace["seed_steps"]

        total_steps = preds.shape[1]
        t_minutes = np.arange(total_steps) * freq_s / 60.0
        # The input tensors are one step shorter than the prediction sequence
        # (no input drives the final state).
        t_input = t_minutes[:len(ambient[0])]

        for row in range(min(n_plot, preds.shape[0])):
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(name, fontsize=14)

            for label in states:
                z = channels.index(label)
                ax.plot(t_minutes, preds[row, :, z].numpy(), "-", label=label)
            ax.axvline(t_minutes[seed_steps - 1], color="gray",
                       linestyle="--", linewidth=1, label="Rollout start")
            ax.step(t_input, ambient[row].numpy(), where="post", color="blue",
                    linestyle=":", alpha=0.6, label="Ambient")

            ax.set_ylabel("Physical units")
            ax.set_xlabel("Time (minutes)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            if temp_range is not None:
                margin = 0.5
                data_min = min(preds[row].min().item(), ambient[row].min().item())
                data_max = max(preds[row].max().item(), ambient[row].max().item())
                ax.set_ylim(min(temp_range[0], data_min) - margin,
                            max(temp_range[1], data_max) + margin)

            fig.tight_layout()
            fname = safe_name(name)
            fig.savefig(plot_dir / f"{fname}_{row}.pdf")
            if verbose:
                print(f"Saved synthetic plot: {plot_dir / f'{fname}_{row}.pdf'}")
            plt.close(fig)

            csv_data = {"time_min": t_minutes}
            for z, label in enumerate(channels):
                csv_data[label] = preds[row, :, z].numpy()
            pad = total_steps - len(ambient[row])
            csv_data["ambient"] = np.append(
                ambient[row].numpy(), [np.nan] * pad
            )
            data_dir = plot_dir / "synth_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(csv_data).to_csv(
                data_dir / f"{fname}_{row}.csv", index=False
            )
