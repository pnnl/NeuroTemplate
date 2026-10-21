"""
`nt data heads`, `nt data decay` and `nt data latch` -- offline diagnostics
that read a dataset and print a recommendation. Both load the dataset the same way
`nt model train` does (see `train.py`'s first ~80 lines) and hand the raw
numbers to the pure functions in `modeling/diagnostics.py`; this file is
only argument parsing and printing.

Neither is wired into `nt model train`. `heads` prints a snippet the user
pastes into `data_head_config`; `decay` measures `tau_slow`, whose range is
the natural `tau_slow_h` box of a 2r2c block's `params`. Both require
judgment the pipeline does not have (is this multimodality really staged
equipment; is this run really a free decay), so neither substitutes a
measurement automatically. `decay` reads only the train split, matching how
`nt model train` pools data; a free-decay episode that lives only in dev or
test needs `train_data_path` pointed at it.

Usage:
    nt data heads --config model_configs/modeling.yaml
    nt data heads --config model_configs/modeling.yaml --column "Total DX Power [W]"
    nt data decay --config model_configs/modeling.yaml \
        --state "North Zone Air Temperature [°C]" --ref "Ambient Temperature [°C]"
    nt data decay --config model_configs/modeling.yaml \
        --state "Zone [°C]" --ref "Outdoor [°C]" --off-by "Total DX Power [W]"
    nt data latch --config model_configs/modeling.yaml --running "DX Power [W]" \
        --setpoint "Cooling Setpoint [°C]" --state "Zone [°C]" --sign -1
"""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd

from neurotemplate._typing import beartype

from neurotemplate.common.config import load_config_from_cli, MODEL_CONFIG_DIR
from neurotemplate.common.data_utils import compute_norm_stats, get_data_splits, pool_dfs
from neurotemplate.hooks import load_modeling_hooks
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.dynamics import family_by_channel
from neurotemplate.modeling.pipeline import resolve_block_spec, resolve_nm_mapping
from neurotemplate.modeling.diagnostics import (
    K_MAX_DEFAULT,
    LATCH_EDGES,
    MIN_RUN_HOURS_DEFAULT,
    OFF_TOLERANCE_FRAC,
    actuator_off_mask,
    estimate_decay,
    latch_table,
    pool_latch_tables,
    recommend_head,
    summarize_taus,
)


def _print_head_recommendation(rec) -> None:
    header, *rest = rec.evidence.splitlines()
    print(f"\n{rec.column}  —  {header}")
    for line in rest:
        print(f"  {line}")
    print()
    print(f"  {rec.snippet}")


@beartype
def heads_main() -> None:
    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--column", type=str, default=None,
                       help="Only this Y column (default: every Y column).")
    flags.add_argument("--k-max", type=int, default=K_MAX_DEFAULT,
                       help=f"Largest K tried in the k-means sweep (default {K_MAX_DEFAULT}).")
    cfg, args = load_config_from_cli(
        sys.argv[1:], cls=ModelingConfig, config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
        command_flags=flags,
    )

    hooks = load_modeling_hooks(cfg)
    train_raws, _dev_raws, _test_raws, _freq_s = get_data_splits(cfg)
    nm_mapping = resolve_nm_mapping(hooks, train_raws[0], cfg)
    train_pool = pool_dfs(train_raws)

    # Which family builds which channel, resolved exactly the way the trainer
    # resolves it, so a head/family conflict is reported per channel rather
    # than from the run-wide cfg.model — which is the wrong answer for every
    # block that declares its own family.
    families = family_by_channel(
        cfg, nm_mapping, resolve_block_spec(hooks, nm_mapping, cfg),
    )

    columns = [args.column] if args.column is not None else nm_mapping["Y"]
    unknown = [c for c in columns if c not in nm_mapping["Y"]]
    if unknown:
        raise SystemExit(f"--column {unknown} not in Y = {nm_mapping['Y']}")

    for col in columns:
        rec = recommend_head(
            train_pool[col].to_numpy(), col,
            family=families.get(col), k_max=args.k_max,
        )
        _print_head_recommendation(rec)
    print()


def _off_columns(args, spec, nm_mapping: dict[str, list[str]]) -> list[str]:
    """The columns that define "off" for `--state`: `--off-by` if given, else
    the controls `synth_cfg.responds` pairs with that state.

    Per state rather than all of U: in a multi-zone building every zone's
    actuator is never off at once, but each zone's own one often is.
    """
    if args.off_by:
        return args.off_by
    if not nm_mapping["U"]:
        return []
    paired = list(spec.responds.get(args.state, {})) if spec is not None else []
    if not paired:
        raise SystemExit(
            f"synth_cfg.responds pairs no control with {args.state!r}, so "
            "there is no actuator to call off. Name the columns that define "
            "off with --off-by (a control, or a meter that reads zero when "
            "the equipment is off)."
        )
    return paired


@beartype
def decay_main() -> None:
    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--state", type=str, required=True,
                       help="Y column to fit the free-decay envelope for.")
    flags.add_argument("--ref", type=str, required=True,
                       help="Reference column the state decays toward (usually ambient).")
    flags.add_argument("--off-by", type=str, action="append", default=[],
                       help="A column that defines 'off' (repeatable): a control, "
                            "or a meter that reads zero when the equipment is off. "
                            "Default: the controls synth_cfg.responds pairs with "
                            "--state.")
    flags.add_argument("--min-hours", type=float, default=MIN_RUN_HOURS_DEFAULT,
                       help=f"Minimum length of an off-run to fit, in hours "
                            f"(default {MIN_RUN_HOURS_DEFAULT:g}).")
    cfg, args = load_config_from_cli(
        sys.argv[1:], cls=ModelingConfig, config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
        command_flags=flags,
    )

    hooks = load_modeling_hooks(cfg)
    train_raws, _dev_raws, _test_raws, freq_s = get_data_splits(cfg)
    cfg.steps_per_hour = 3600.0 / freq_s
    nm_mapping = resolve_nm_mapping(hooks, train_raws[0], cfg)

    known_columns = nm_mapping["Y"] + nm_mapping["U"] + nm_mapping["D"]
    unknown = [c for c in (args.state, args.ref, *args.off_by) if c not in known_columns]
    if unknown:
        raise SystemExit(f"--state/--ref/--off-by column(s) {unknown} are not "
                         "in the data mapping.")

    train_pool = pool_dfs(train_raws)
    norm_overrides = hooks.column_norm_config(nm_mapping)
    norm_stats = compute_norm_stats(
        df=train_pool, norm_type=cfg.norm_type, column_overrides=norm_overrides,
    )
    spec = hooks.synth_cfg(nm_mapping, norm_stats)
    if spec is None and not args.off_by:
        raise SystemExit(
            "nt data decay reads which controls move --state, and their `off` "
            "values, from the synth_cfg hook, and synth_cfg(nm_mapping, "
            "norm_stats) returned None. Fill in synth_cfg in "
            "modeling_user_defined.py, or name the columns that define off "
            "with --off-by."
        )

    # A column the spec gives no `off` idles at its recorded minimum, per
    # SynthSpec.off's own docstring — zero, for a meter.
    off_columns = _off_columns(args, spec, nm_mapping)
    declared_off = spec.off if spec is not None else {}
    off_values = {
        col: declared_off.get(col, float(train_pool[col].min()))
        for col in off_columns
    }
    min_run_steps = max(2, round(args.min_hours * cfg.steps_per_hour))

    # Each raw file is processed on its own -- sec_elapsed and any off-run
    # only make sense within one contiguous recording.
    all_windows = []
    for raw_df in train_raws:
        t_hours = raw_df["sec_elapsed"].to_numpy(dtype=float) / 3600.0
        state = raw_df[args.state].to_numpy(dtype=float)
        ref = raw_df[args.ref].to_numpy(dtype=float)
        if off_columns:
            values = {col: raw_df[col].to_numpy(dtype=float) for col in off_columns}
            mask = actuator_off_mask(values, off_values)
        else:
            mask = pd.Series(True, index=raw_df.index).to_numpy()
        all_windows.extend(estimate_decay(t_hours, state, ref, mask, min_run_steps))

    print(f"Free-decay tau for {args.state!r} relative to {args.ref!r}")
    print(f"  off = every one of {off_columns} within "
          f"{OFF_TOLERANCE_FRAC * 100:.0f}% of its off value, for "
          f">= {args.min_hours:g}h: {len(all_windows)} qualifying window(s)")

    if not all_windows:
        print(
            "  No free-decay episode found in this data -- the plant is "
            "never uncoupled for long enough at these off values. If a "
            "setpoint is excited and never sits at its off value, define off "
            "by a meter instead (--off-by \"<power column>\"). The "
            "tau_slow_h box then has to come from what you know about the "
            "building: hold it wide, and read on the reportcard whether the "
            "fit presses against an edge."
        )
        return

    for w in all_windows:
        print(f"  window @ row {w.start} ({w.n_steps} steps): "
              f"tau={w.tau_h:.2f}h  half-life={w.half_life_h:.2f}h  r²={w.r_squared:.3f}")

    median, lo, hi = summarize_taus(tuple(all_windows))
    print(f"  median tau = {median:.2f}h  (range {lo:.2f}-{hi:.2f}h "
          f"across {len(all_windows)} windows)")
    print(f"  half-life at the median = {median * math.log(2):.2f}h")
    print(
        "  Report the range, not just the median -- the spread across "
        "windows IS the honest uncertainty on this estimate. It measures "
        "tau_slow: the range is the natural `tau_slow_h` box in the block's "
        "params, then read back off the reportcard where the fit landed in it."
    )


def latch_main() -> None:
    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--running", type=str, required=True,
                       help="A column that reads above its minimum while the equipment "
                            "runs: a meter, or a stage signal.")
    flags.add_argument("--setpoint", type=str, required=True,
                       help="The setpoint that calls the equipment.")
    flags.add_argument("--state", type=str, required=True,
                       help="The state the setpoint is compared with.")
    flags.add_argument("--sign", type=int, choices=(-1, 1), required=True,
                       help="The direction the state moves while the equipment runs, as "
                            "in a demand declaration: -1 for cooling, +1 for heating.")
    cfg, args = load_config_from_cli(
        sys.argv[1:], cls=ModelingConfig, config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
        command_flags=flags,
    )
    train_raws, _dev_raws, _test_raws, _freq_s = get_data_splits(cfg)
    missing = [c for c in (args.running, args.setpoint, args.state)
               if c not in train_raws[0].columns]
    if missing:
        raise SystemExit(f"column(s) {missing} are not in the training data.")

    # The error is positive while the equipment is called for, as the demand
    # feature is; each file on its own, since "the step before" needs contiguity.
    tables = []
    for raw in train_raws:
        meter = raw[args.running].to_numpy(dtype=float)
        floor = meter.min() + OFF_TOLERANCE_FRAC * (meter.max() - meter.min())
        error = args.sign * (raw[args.setpoint] - raw[args.state]).to_numpy(dtype=float)
        tables.append(latch_table(meter > floor, error))
    rows = pool_latch_tables(tables)

    print(f"How often {args.running!r} runs, by {args.state!r} minus "
          f"{args.setpoint!r} (sign {args.sign:+d}: positive while it is called for)")
    print("  error bin          after running        after idle")
    for r in rows:
        def cell(p, n):
            return f"{p:6.2f}  (n={n:5d})" if p is not None else f"{'-':>6s}  (n={n:5d})"
        print(f"  ({r.low:+5.2f}, {r.high:+5.2f}]   {cell(r.p_after_running, r.n_after_running)}"
              f"   {cell(r.p_after_idle, r.n_after_idle)}")
    print("  The same column twice means the equipment answers the error alone. A high "
          "'after running' beside a low 'after idle' in the bins around zero means it "
          "latches: declare `hysteresis: true` on the setpoint's demand entry.")
