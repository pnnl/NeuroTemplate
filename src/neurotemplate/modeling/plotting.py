import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from neurotemplate.common.data_utils import axis_label, unit_of
from neurotemplate.hooks import load_modeling_hooks

from neurotemplate._typing import beartype

# ---------------------- Plotting ---------------------- #

# Publication-friendly defaults. LINE_THICKNESS is applied as the default
# matplotlib line width; individual calls can still override via linewidth=.

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

# Fixed-order categorical palette for synthetic-scenario state lines, plus the
# muted tones used for the ambient signal, rollout boundary, and gridlines.
_STATE_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
_AMBIENT_COLOR = "#898781"
_BOUNDARY_COLOR = "#c3c2b7"
_GRID_COLOR = "#e1e0d9"


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
        if logger is None or not hasattr(logger, "_mlflow"):
            return
        if artifact_path is None:
            logger._mlflow.log_artifact(path)
        else:
            logger._mlflow.log_artifact(path, artifact_path=artifact_path)

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
    norm_stats: dict,
    freq_s: float = 300.0,
    n_plot: int = 4,
    verbose: bool = False,
    fallback_unit: Optional[str] = None,
) -> None:
    """
    One figure per synthetic-scenario instance: the state channels against the
    ambient signal they were driven with, with the seed/rollout boundary
    marked. The verdicts live in the reportcard; these are for looking at the
    shape when a verdict is surprising.

    Args:
        traces: dicts from `scenario.trace(n)`; empty ones are skipped.
        run_dir: directory for the figures (CSVs go in `synth_data/` beside).
        norm_stats: per-column stats dict — supplies each trace's y-axis unit
            via each state column's own "[unit]" name suffix (see
            `common.data_utils.unit_of`). A trace's states can disagree
            (a mixed-unit Y across per-block families is exactly the case
            this exists for), in which case the axis goes unit-less rather
            than printing a wrong label.
        freq_s: seconds per timestep, for the x axis.
        n_plot: max instances drawn per scenario.
        fallback_unit: unit used for a state column whose name carries none
            of its own — the deprecated `cfg.synth_y_unit`. None column units
            with no fallback set just render "Temperature", unit-less.
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

        # Each state's own unit, falling back to `fallback_unit` (the
        # deprecated cfg.synth_y_unit) only for a column whose name carries
        # none. A one-off stand-in norm_stats so `axis_label` — the same
        # unit-agreement logic every other axis label uses — need not know
        # about the fallback at all.
        with_fallback = {c: {"unit": unit_of(norm_stats, c) or fallback_unit}
                          for c in states}
        ylabel = axis_label(with_fallback, states, "Temperature")

        for row in range(min(n_plot, preds.shape[0])):
            fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
            fig.suptitle(name, fontsize=13, fontweight="bold")

            for i, label in enumerate(states):
                z = channels.index(label)
                color = _STATE_COLORS[i % len(_STATE_COLORS)]
                ax.plot(t_minutes, preds[row, :, z].numpy(), "-",
                        color=color, linewidth=2, label=label)
            ax.axvline(t_minutes[seed_steps - 1], color=_BOUNDARY_COLOR,
                       linestyle="--", linewidth=1, label="Rollout start")
            ax.step(t_input, ambient[row].numpy(), where="post",
                    color=_AMBIENT_COLOR, linestyle=":", linewidth=1.5,
                    alpha=0.9, label="Ambient")

            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time (minutes)")
            ax.margins(y=0.15)
            ax.grid(True, alpha=0.6, color=_GRID_COLOR, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="best", fontsize=8, frameon=False)

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
