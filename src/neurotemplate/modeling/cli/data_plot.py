"""
`nt data plot`: look at a record one channel at a time.

    nt data plot data/train.csv
    nt data plot data/train.csv --column "Zone A Temp [degC]" --output zone.png

Opens a window with the record's channels listed down the left and one of
them plotted against wall-clock time; click a name, or step with ↑/↓.
`--output` writes one channel to an image instead, for an agent to read.

Run from an application directory, the list is grouped by the roles `data.yaml`
assigns (Y / U / D), and every other numeric column lands in "no role": a band, a
tariff, or a plant column the model does not read. Without a `data.yaml` it is one
flat list.

The file is read as recorded. A gap in the timestamps breaks the line rather
than being bridged, and nothing is filled, resampled or judged.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.frequencies import to_offset

from neurotemplate.common.data_file import ROLES, load_data_file
from neurotemplate.common.data_utils import read_timestamped_csv
from neurotemplate.modeling.plotting import PLOT_STYLE

UNMAPPED = "no role"
ALL = "columns"

# One line colour per group, from the library's categorical palette.
_GROUP_COLORS = {"Y": "#2a78d6", "U": "#eb6834", "D": "#1baf7a",
                 UNMAPPED: "#898781", ALL: "#2a78d6"}
_NAME_CHARS = 34  # list entries longer than this are cut; the title has it whole


def group_channels(columns: list[str],
                   mapping: Optional[dict[str, list[str]]]) -> dict[str, list[str]]:
    """The plottable columns by role, in file order; empty groups dropped.

    `mapping` is `data.yaml`'s `columns`, or None for a flat list. Names it gives
    that are not in this file are ignored, so a draw can be plotted from an application
    directory.
    """
    if mapping is None:
        return {ALL: list(columns)}
    groups = {}
    for role in ROLES:
        groups[role] = [c for c in columns if c in mapping.get(role, [])]
    mapped = {c for role in ROLES for c in mapping.get(role, [])}
    groups[UNMAPPED] = [c for c in columns if c not in mapped]
    return {group: names for group, names in groups.items() if names}


def break_at_gaps(series: pd.Series) -> tuple[pd.Series, int]:
    """`series` with a NaN one step after each gap, so the line breaks there;
    and how many gaps. The step is the record's most common spacing."""
    spacing = series.index.to_series().diff()
    step = spacing.mode().iloc[0]
    gap_starts = series.index[:-1][(spacing > step).to_numpy()[1:]]
    if len(gap_starts) == 0:
        return series, 0
    breaks = pd.Series(float("nan"), index=gap_starts + step)
    return pd.concat([series, breaks]).sort_index(), len(gap_starts)


def draw_channel(ax, series: pd.Series, group: str) -> None:
    """Plot one channel on `ax`, replacing whatever was there."""
    ax.clear()
    line, gaps = break_at_gaps(series)
    ax.plot(line.index, line.to_numpy(), linewidth=1.2, color=_GROUP_COLORS[group])
    title = series.name if group == ALL else f"{group} · {series.name}"
    step = to_offset(series.index.to_series().diff().mode().iloc[0]).freqstr
    note = (f"{len(series)} rows · step {step} · {gaps} gap(s) · "
            f"{int(series.isna().sum())} missing value(s)")
    ax.set_title(f"{title}\n{note}", fontsize=11, loc="left")
    ax.set_ylabel(series.name)
    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.figure.autofmt_xdate()


def _short(name: str) -> str:
    return name if len(name) <= _NAME_CHARS else name[:_NAME_CHARS - 1] + "…"


def show_window(df: pd.DataFrame, groups: dict[str, list[str]], first: str,
                title: str) -> None:
    """The interactive window: a clickable channel list beside the plot."""
    order = [c for names in groups.values() for c in names]
    group_of = {c: g for g, names in groups.items() for c in names}

    fig = plt.figure(figsize=(14, 6.5))
    fig.canvas.manager.set_window_title(title)
    list_ax = fig.add_axes([0.01, 0.03, 0.24, 0.94])
    list_ax.set_axis_off()
    plot_ax = fig.add_axes([0.33, 0.14, 0.64, 0.74])

    # One row per heading and per channel, top to bottom.
    rows = len(order) + len(groups) if len(groups) > 1 else len(order)
    row_height = min(0.045, 1.0 / max(rows, 1))
    fontsize = max(5.0, min(10.0, 420.0 / max(rows, 1)))
    entries = {}
    y = 1.0
    for group, names in groups.items():
        if len(groups) > 1:
            list_ax.text(0.0, y, group, fontsize=fontsize, fontweight="bold",
                         va="top", transform=list_ax.transAxes)
            y -= row_height
        for name in names:
            entries[name] = list_ax.text(
                0.04, y, _short(name), fontsize=fontsize, va="top",
                transform=list_ax.transAxes, picker=True)
            y -= row_height

    current = {"name": first}

    def select(name: str) -> None:
        current["name"] = name
        for other, text in entries.items():
            text.set_fontweight("bold" if other == name else "normal")
            text.set_color(_GROUP_COLORS[group_of[other]] if other == name else "black")
        draw_channel(plot_ax, df[name], group_of[name])
        fig.canvas.draw_idle()

    def on_pick(event) -> None:
        for name, text in entries.items():
            if event.artist is text:
                select(name)
                return

    def on_key(event) -> None:
        step = {"down": 1, "up": -1}.get(event.key)
        if step is not None:
            index = (order.index(current["name"]) + step) % len(order)
            select(order[index])

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)
    select(first)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nt data plot",
        description="Plot a record's channels one at a time, grouped by the roles in "
                    "data.yaml when one is in the working directory.")
    parser.add_argument("path", help="One CSV with a time column.")
    parser.add_argument("--column", default=None,
                        help="The channel to show first (or the one --output writes).")
    parser.add_argument("--output", default=None,
                        help="Write --column to this image instead of opening a window.")
    args = parser.parse_args(sys.argv[1:])

    df = read_timestamped_csv(args.path)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    skipped = [c for c in df.columns if c not in numeric]
    if skipped:
        print(f"Not plotted (not numeric): {skipped}")
    if not numeric:
        raise SystemExit(f"{args.path} has no numeric column to plot.")

    mapping = load_data_file("data.yaml").columns if Path("data.yaml").exists() else None
    groups = group_channels(numeric, mapping)
    if mapping is None:
        print("No data.yaml here: channels are listed ungrouped.")

    if args.column is not None and args.column not in numeric:
        raise SystemExit(f"--column {args.column!r} is not a numeric column of "
                         f"{args.path}; it has {numeric}")
    first = args.column or next(iter(groups.values()))[0]

    with plt.rc_context(PLOT_STYLE):
        if args.output is not None:
            if args.column is None:
                raise SystemExit("--output writes one channel: name it with --column.")
            fig, ax = plt.subplots(figsize=(11, 4.5))
            group = next(g for g, names in groups.items() if args.column in names)
            draw_channel(ax, df[args.column], group)
            fig.tight_layout()
            fig.savefig(args.output, dpi=120)
            print(f"Wrote {args.output}")
            return
        show_window(df, groups, first, title=Path(args.path).name)
