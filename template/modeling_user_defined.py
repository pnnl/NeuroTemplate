"""
Modeling hooks — the glue between NeuroTemplate and *your* dataset.

The library calls each function below, gets back a dataset-specific answer
(column roles, head topology, normalization overrides, ...), and proceeds
without any further knowledge of your data.

The file must be named `modeling_user_defined.py` and live in the directory
you run `nt` from, or be pointed at by `user_hooks_path` in your YAML.

Every function here except `hyper_args`, `tie_norm_stats` and `block_mapping`
is imported by attribute with no default, so all of them must exist even when
the body is a trivial return. Fill in `data_mapping` first; the rest have
working defaults.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Optional

import pandas as pd

from neurotemplate._typing import beartype
from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.heads import (  # noqa: F401  — cited in comments
    BinaryHead,
    BoundedHead,
    HeadStack,
    OrdinalHead,
    RegressionHead,
    bounds_from_norm,
    project_physical_to_norm,
)
from neurotemplate.modeling.scenarios import (  # noqa: F401  — cited in synth_cfg
    SynthSpec,
)


# ---------------------------------------------------------------------
# Required: assign every dataframe column a role
# ---------------------------------------------------------------------
# Y = states the model predicts, U = inputs a controller can move,
# D = exogenous disturbances the model sees but cannot influence.
# Any column not listed is dropped.
Y_COLUMNS: list[str] = [
    # "Tank Temperature [degC]",
]
U_COLUMNS: list[str] = [
    # "Heater Command [-]",
]
D_COLUMNS: list[str] = [
    # "Ambient Temperature [degC]",
]


@beartype
def data_mapping(
    df: pd.DataFrame,
    cfg: Config,
) -> dict[str, list[str]]:
    """
    Return {"Y": [...], "U": [...], "D": [...], "t": ["sec_elapsed"]}.

    Names must match `df` columns exactly and appear in dataframe order —
    downstream indexing is positional. `cfg.exclude_columns` holds substring
    patterns to drop; a pattern matching nothing raises, to catch typos.

    `sec_elapsed` is added by the CSV loader from the timestamp index.
    """
    missing = [c for c in cfg.exclude_columns
               if not any(c in col for col in df.columns)]
    if missing:
        raise ValueError(
            f"cfg.exclude_columns lists patterns matching no columns: {missing}"
        )

    def keep(names: list[str]) -> list[str]:
        unknown = [c for c in names if c not in df.columns]
        if unknown:
            raise ValueError(f"Columns not in dataframe: {unknown}")
        return [
            c for c in df.columns
            if c in names and not any(p in c for p in cfg.exclude_columns)
        ]

    y_cols, u_cols, d_cols = keep(Y_COLUMNS), keep(U_COLUMNS), keep(D_COLUMNS)

    for (a_name, a), (b_name, b) in [
        (("Y", y_cols), ("U", u_cols)),
        (("Y", y_cols), ("D", d_cols)),
        (("U", u_cols), ("D", d_cols)),
    ]:
        overlap = set(a) & set(b)
        if overlap:
            raise ValueError(f"Columns in both {a_name} and {b_name}: {overlap}")

    if not y_cols:
        raise ValueError("Y is empty — the model has nothing to predict.")
    if "sec_elapsed" not in df.columns:
        raise ValueError("sec_elapsed column missing from dataframe.")

    return {"Y": y_cols, "U": u_cols, "D": d_cols, "t": ["sec_elapsed"]}


# ---------------------------------------------------------------------
# Required: one output head per Y channel
# ---------------------------------------------------------------------
@beartype
def data_head_config(
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> HeadStack:
    """
    Return a HeadStack with exactly one head per Y column, in Y order.

    Run `nt data heads` first — it reads the data and suggests a head (plus a
    pasteable snippet) for each Y column from the shape of its distribution,
    not its name.

    All-regression (below) is the right default. Reach for another head only
    when the channel really is discrete or bounded:

      RegressionHead()                  continuous, MSE loss
      BinaryHead()                      0/1 target, sigmoid + BCE
      BoundedHead(lo, hi)               clamped to [lo, hi]; pass
                                        float("inf") / float("-inf") on an
                                        open side for half-bounded
      OrdinalHead(n_levels=K)           K ordered levels, values 0..K-1
      OrdinalHead(n_levels=K, level_values=t)
                                        K ordered levels at given values

    A binary / ordinal channel is usually a rare-event channel (`nt data
    heads` flags a level under 5%), and the pipeline handles that for you:
    each levelled head's loss is weighted by inverse training frequency
    (`cfg.balance_levels`) and, after training, each ordinal head's decision
    thresholds are calibrated on dev (`cfg.calibrate_thresholds`). Judge such
    a channel by the reportcard's level metrics, not its NRMSE. To pin either
    by hand instead, pass `level_weights=` / `decision_thresholds=` here and
    switch the corresponding flag off.

    Bounds and level values live in head-space, i.e. AFTER normalization.
    Write them in physical units and project:

        stats = norm_stats[col]
        BoundedHead(*bounds_from_norm(stats))              # observed min/max
        BoundedHead(project_physical_to_norm(0.0, stats),
                    float("inf"))                          # physically >= 0
        OrdinalHead(n_levels=3, level_values=torch.tensor(
            [project_physical_to_norm(v, stats) for v in (0.0, 500.0, 1700.0)],
            dtype=torch.float32))
    """
    heads = []
    for col in nm_mapping["Y"]:
        # stats = norm_stats[col]
        # if "on/off" in col.lower():
        #     heads.append(BinaryHead())
        # elif "power" in col.lower():
        #     heads.append(BoundedHead(project_physical_to_norm(0.0, stats),
        #                              float("inf")))
        # else:
        heads.append(RegressionHead())
    return HeadStack(heads, nm_mapping["Y"])


# ---------------------------------------------------------------------
# Required: per-column normalization overrides
# ---------------------------------------------------------------------
@beartype
def column_norm_config(
    nm_mapping: dict[str, list[str]],
) -> dict[str, Optional[str]]:
    """
    Return {column_name: "min_max" | "abs_max" | "z_score" | None}.

    Columns not listed use the run-wide `cfg.norm_type`. Override where a
    column's distribution defeats the default: bounded non-negative signals
    prefer "min_max", already-0/1 flags prefer None (leave them alone).

    Return {} for no overrides.
    """
    overrides: dict[str, Optional[str]] = {}
    # for col in nm_mapping["Y"] + nm_mapping["U"] + nm_mapping["D"]:
    #     if "power" in col.lower():
    #         overrides[col] = "min_max"
    #     elif "occupancy" in col.lower():
    #         overrides[col] = None
    return overrides


# ---------------------------------------------------------------------
# Optional: split the states into independent blocks, each its own family
# ---------------------------------------------------------------------
@beartype
def block_mapping(
    nm_mapping: dict[str, list[str]],
    cfg: Config,
) -> Optional[dict[str, dict]]:
    """
    Group Y channels into blocks, each getting its own sub-model — its own
    parameters, its own architecture (via `model`), and its own hidden state
    if the family is stateful. Called only if this function exists; delete
    it, or return None, for a single block holding every channel.

    A block owns its own Y channels and its own U channels; it sees all of D
    by default, or a named subset; and any U channel no block claims is
    shared with all of them — which is what you want for one air handler
    serving several blocks. Y must partition exactly: each state in one
    block and no state left out. D has no such rule — it is exogenous and
    shared by nature, so several blocks may each claim the same column, or
    all of it, with no conflict.

        return {
            "north": {"Y": ["North Zone Temperature [degC]"],
                      "U": ["North Setpoint [degC]"],
                      "model": "2r2c"},
            "power": {"Y": ["Total Power [W]"],
                      "reads": ["North Zone Temperature [degC]"],
                      "D": ["Ambient Temperature [degC]"],
                      "model": "mlp"},
        }
        # "AHU Fan Speed [-]" is claimed by neither, so both blocks get it.

    `model` is a family name ("mlp", "gru", "2r2c") or omitted, meaning "use
    cfg.model" — so a spec that never sets `model` behaves exactly like a
    plain block split, one family for everyone. `reads` names other blocks'
    Y channels this block may also see, as an ordinary input (see the
    `power` block above, reading the zone it measures alongside).

    `model: 2r2c` is per-block whether or not this hook exists — with no
    hook each Y channel still gets its own RC network, driven by every U
    channel. Declaring blocks is how you say *which setpoint drives which
    block*. For `mlp` and `gru` the split only happens with `per_zone: true`
    or by naming `model` explicitly here.

    Blocks are independent by construction: no term in one block's update
    reads another block's state on the same step. `reads` is safe for every
    family except `2r2c`, whose stability guarantee requires its forcing map
    to never see state — a `2r2c` block with `reads` raises.

    `zone_mapping` is a deprecated alias for this hook, honored if
    `block_mapping` is absent.
    """
    return None


# ---------------------------------------------------------------------
# Optional: post-process norm_stats before they are frozen
# ---------------------------------------------------------------------
@beartype
def tie_norm_stats(
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
    cfg: Config,
) -> dict[str, dict[str, float]]:
    """
    Last chance to edit norm_stats after `compute_norm_stats`, before they
    are used and saved with the run. Called only if this function exists.

    Each column's shift/stretch is derived from that column alone, so two
    columns never share a scale. The one case that matters: if a U column
    and a Y column must be directly comparable in normalized space (e.g. a
    setpoint that the model passes through to a state), assign them the same
    shift/stretch here or the denormalized passthrough will be wrong.

        s = norm_stats["Setpoint [degC]"]
        for col in ("Setpoint [degC]", "Zone Temperature [degC]"):
            norm_stats[col].update(shift=s["shift"], stretch=s["stretch"])

    Also the place to fix a column's "unit" (used on plot axis labels):
    it's parsed from the column's own trailing "[...]" by
    `common.data_utils.parse_column_unit`, so a name that doesn't follow
    that convention, or a unit you'd rather display differently, is a
    one-line override here — `norm_stats[col]["unit"] = "kWh"` — not a
    reason to add a second mechanism for the same job.
    """
    return norm_stats


# ---------------------------------------------------------------------
# Required: the plant's resting state, for the physics checks
# ---------------------------------------------------------------------
@beartype
def synth_cfg(
    nm_mapping: dict,
    norm_stats: dict,
) -> Optional[SynthSpec]:
    """
    Three physics assertions run after training whatever else you configure:
    the states must hold at equilibrium, move the right way when an actuator is
    driven off its rest value, and relax toward ambient when the HVAC is off. They put the
    surrogate in states the recorded data never contains, so they answer a
    question no test split can — and a surrogate that fails them will not
    control well however good its NRMSE is.

    Filling this in is a few lines and worth doing. Return None to skip the
    checks entirely.

    Only two things are per-column work:

      states   Y channels that behave like temperatures — they hold when
               nothing drives them and relax toward ambient. Naming them is
               required, because writing an equilibrium temperature into a
               power channel produces silent garbage.
      off      the physical value of each CONTROL column when nothing is
               running. Required for controls and only for controls: an
               actuator's "off" is not guessable (a cooling setpoint is off
               at its *maximum*). Every other column defaults to its recorded
               minimum, which is what "no forcing" means for solar,
               occupancy, power and flow.

               For a setpoint actuator, pick a value that is off across the
               *whole* `states` range — equilibria are sampled from all of
               it, and a 26 °C cooling setpoint is a call for cooling on a
               zone that reaches 28.

      responds `{state: {control: +1 or -1}}` — the sign of each state's
               movement as that control INCREASES. Required for every control.
               The library sweeps each control from its `off` value to each end
               of its range and asserts every declared state moved the right
               way, so an actuator cannot go untested because nobody wrote a
               case for it, and one that warms one zone while cooling another
               is expressible.

               This looks like the `grad_path` mapping and is deliberately not
               the same thing. That file declares partials valid at *every*
               point in state space, which is why its U half is usually empty:
               a cooling setpoint above the current zone temperature commands
               nothing, so the partial is 0 there and positive only where
               cooling runs. Here the scenario has *built* the state where the
               actuator is live, so the sign is well defined.

        from neurotemplate.modeling.scenarios import SynthSpec

        cool_sp = "Zone Cooling Setpoint [degC]"
        zone = "Zone Air Temperature [degC]"
        return SynthSpec(
            states=[zone],
            ambient="Ambient Temperature [degC]",
            off={cool_sp: norm_stats[cool_sp]["max"]},
            # Raising the cooling setpoint asks for less cooling, so the zone
            # warms. The sweep drives it down from `off` and expects a fall.
            responds={zone: {cool_sp: +1}},
        )

    All values are PHYSICAL units; the library normalizes them, and validates
    the whole spec against the column mapping before anything runs.
    """
    return None


# ---------------------------------------------------------------------
# Required: subplot guidance for prediction plots
# ---------------------------------------------------------------------
@beartype
def sub_plot(
    df_true: pd.DataFrame,
    column_name: str,
) -> Optional[tuple[list[str], list[dict]]]:
    """
    For the plot of `column_name`, what goes on a second x-shared axis below?
    This is where the covariates you actually want to eyeball belong.

    Return None for a single axis, or (columns, kwargs) of equal length —
    each kwargs dict is splatted straight into `ax2.plot`:

        return (["Ambient Temperature [degC]", "Heater Command [-]"],
                [{"label": "ambient", "color": "tab:blue"},
                 {"label": "heater", "color": "tab:red", "linestyle": "--"}])
    """
    return None


# ---------------------------------------------------------------------
# Required only for `nt model optuna`: hyperparameter search space
# ---------------------------------------------------------------------
@beartype
def hyper_args(trial, base_cfg: Config) -> Config:
    """
    Return a copy of `base_cfg` with the fields this trial should vary.
    Any ModelingConfig field is fair game; keep the space small.
    """
    cfg = deepcopy(base_cfg)
    # cfg.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # cfg.log2_hidden_size = trial.suggest_int("log2_hidden_size", 4, 9)
    # cfg.num_layers = trial.suggest_int("num_layers", 1, 4)
    return cfg


# ---------------------------------------------------------------------
# Helper (not a hook): write the gradient-sign mapping JSON
# ---------------------------------------------------------------------
# `grad_gain > 0` requires `grad_path` to point at a JSON of known-sign
# partial derivatives: {output_col: {input_col: +1 | -1}}. Only list edges
# that follow from conservation laws and hold anywhere in the state space —
# the penalty is evaluated at random off-trajectory samples, so relationships
# that only hold under your supervisory controller will fire in the wrong
# places and hurt. Run `python modeling_user_defined.py` to write the file.
def write_gradient_mapping(path: str = "data/grad/mapping.json") -> str:
    mapping: dict[str, dict[str, int]] = {
        # "Tank Temperature [degC]": {
        #     "Tank Temperature [degC]": +1,   # thermal mass continuity
        #     "Ambient Temperature [degC]": +1,  # envelope conduction
        #     "Heater Command [-]": +1,        # heat in raises temperature
        # },
    }
    if not mapping:
        raise ValueError("write_gradient_mapping: fill in `mapping` first.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    return path


if __name__ == "__main__":
    print(write_gradient_mapping())
