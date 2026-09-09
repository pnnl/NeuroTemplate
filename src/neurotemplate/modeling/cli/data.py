"""
`nt data heads` and `nt data decay` -- offline diagnostics that read a
dataset and print a recommendation, described in full in
generalization_report.md §4.1-4.3. Both load the dataset the same way
`nt model train` does (see `train.py`'s first ~80 lines) and hand the raw
numbers to the pure functions in `modeling/diagnostics.py`; this file is
only argument parsing and printing.

Neither is wired into `nt model train`. `heads` prints a snippet the user
pastes into `data_head_config`; `decay` prints a number to compare against
the asserted `rc_tau_ambient_h` prior. Both require judgment the pipeline
does not have (is this multimodality really staged equipment; is this run
really a free decay), so neither substitutes a measurement automatically.

Usage:
    nt data heads --config model_configs/modeling.yaml
    nt data heads --config model_configs/modeling.yaml --column "Total DX Power [W]"
    nt data decay --config model_configs/modeling.yaml \
        --state "North Zone Air Temperature [°C]" --ref "Ambient Temperature [°C]"
"""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd

from neurotemplate._typing import beartype

from neurotemplate.common.config import load_config_from_cli, MODEL_CONFIG_DIR
from neurotemplate.common.data_utils import compute_norm_stats, get_data_splits
from neurotemplate.hooks import load_modeling_hooks
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.dynamics import family_by_channel
from neurotemplate.modeling.pipeline import resolve_block_spec
from neurotemplate.modeling.diagnostics import (
    K_MAX_DEFAULT,
    MIN_RUN_HOURS_DEFAULT,
    OFF_TOLERANCE_FRAC,
    actuator_off_mask,
    estimate_decay,
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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--column", type=str, default=None,
                         help="Only this Y column (default: every Y column).")
    parser.add_argument("--k-max", type=int, default=K_MAX_DEFAULT,
                         help=f"Largest K tried in the k-means sweep (default {K_MAX_DEFAULT}).")
    args, remaining = parser.parse_known_args(sys.argv[1:])

    cfg, _cli = load_config_from_cli(
        remaining, cls=ModelingConfig, config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
    )

    hooks = load_modeling_hooks(cfg)
    train_raws, _dev_raws, _test_raws, _freq_s = get_data_splits(cfg)
    nm_mapping = hooks.data_mapping(df=train_raws[0], cfg=cfg)
    train_pool = train_raws[0] if len(train_raws) == 1 else pd.concat(train_raws)

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


@beartype
def decay_main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state", type=str, default=None,
                         help="Y column to fit the free-decay envelope for.")
    parser.add_argument("--ref", type=str, default=None,
                         help="Reference column the state decays toward (usually ambient).")
    parser.add_argument("--min-hours", type=float, default=MIN_RUN_HOURS_DEFAULT,
                         help=f"Minimum length of an off-run to fit, in hours "
                              f"(default {MIN_RUN_HOURS_DEFAULT:g}).")
    args, remaining = parser.parse_known_args(sys.argv[1:])

    cfg, _cli = load_config_from_cli(
        remaining, cls=ModelingConfig, config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
    )
    # Checked after load_config_from_cli so --help/--print-config/etc still
    # work without also having to pass --state/--ref.
    if args.state is None or args.ref is None:
        raise SystemExit(
            "nt data decay requires --state and --ref (the Y column to fit "
            "and the reference/ambient column it decays toward)."
        )

    hooks = load_modeling_hooks(cfg)
    train_raws, _dev_raws, _test_raws, freq_s = get_data_splits(cfg)
    cfg.steps_per_hour = 3600.0 / freq_s
    nm_mapping = hooks.data_mapping(df=train_raws[0], cfg=cfg)

    known_columns = nm_mapping["Y"] + nm_mapping["U"] + nm_mapping["D"]
    for col in (args.state, args.ref):
        if col not in known_columns:
            raise SystemExit(f"--state/--ref column {col!r} is not in the data mapping.")

    train_pool = train_raws[0] if len(train_raws) == 1 else pd.concat(train_raws)
    norm_overrides = hooks.column_norm_config(nm_mapping)
    norm_stats = compute_norm_stats(
        df=train_pool, norm_type=cfg.norm_type, column_overrides=norm_overrides,
    )
    spec = hooks.synth_cfg(nm_mapping, norm_stats)
    if spec is None:
        raise SystemExit(
            "nt data decay reads the actuator `off` values from the synth_cfg "
            "hook, and synth_cfg(nm_mapping, norm_stats) returned None. Fill "
            "in synth_cfg in modeling_user_defined.py first."
        )

    # Every U column not named in spec.off idles at its recorded minimum,
    # per SynthSpec.off's own docstring.
    off_values = {
        col: spec.off.get(col, float(train_pool[col].min()))
        for col in nm_mapping["U"]
    }
    min_run_steps = max(2, round(args.min_hours * cfg.steps_per_hour))

    # Each raw file is processed on its own -- sec_elapsed and any off-run
    # only make sense within one contiguous recording.
    all_windows = []
    for raw_df in train_raws:
        t_hours = raw_df["sec_elapsed"].to_numpy(dtype=float) / 3600.0
        state = raw_df[args.state].to_numpy(dtype=float)
        ref = raw_df[args.ref].to_numpy(dtype=float)
        if nm_mapping["U"]:
            u_values = {col: raw_df[col].to_numpy(dtype=float) for col in nm_mapping["U"]}
            mask = actuator_off_mask(u_values, off_values)
        else:
            mask = pd.Series(True, index=raw_df.index).to_numpy()
        all_windows.extend(estimate_decay(t_hours, state, ref, mask, min_run_steps))

    print(f"Free-decay tau for {args.state!r} relative to {args.ref!r}")
    print(f"  actuators off (every U column within "
          f"{OFF_TOLERANCE_FRAC * 100:.0f}% of its off value) for "
          f">= {args.min_hours:g}h: {len(all_windows)} qualifying window(s)")

    if not all_windows:
        print(
            "  No free-decay episode found in this data -- the plant is "
            "never uncoupled for long enough at these off values. The "
            "rc_tau_ambient_h prior then has to come from an Optuna sweep "
            "instead: it is an ordinary hyperparameter there, not something "
            "this script can measure for you."
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
        "windows IS the honest uncertainty on this estimate. Compare it "
        "against rc_tau_ambient_h before training, then read tau_slow back "
        "off the reportcard."
    )
