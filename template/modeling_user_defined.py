"""
Modeling hooks — the glue between NeuroTemplate and *your* dataset.

The library calls each function below, gets back a dataset-specific answer
(column roles, head topology, normalization overrides, ...), and proceeds
without any further knowledge of your data.

The file must be named `modeling_user_defined.py` and live in the directory
you run `nt` from, or be pointed at by `user_hooks_path` in your YAML.

Every function here except `hyper_args` and `tie_norm_stats` is imported by
attribute with no default, so all of them must exist even when the body is a
trivial return. Fill in `data_mapping` first; the rest have working defaults.
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
    Actuation,
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
    the states must hold at equilibrium, move the right way under an
    actuation, and relax toward ambient when the HVAC is off. They put the
    surrogate in states the recorded data never contains, so they answer a
    question no test split can — and a surrogate that fails them will not
    control well however good its NRMSE is.

    Filling this in is a few lines and worth doing. Return None to skip the
    checks entirely; `synth_gain` must then stay 0.

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

    `actuations` is optional; each one names the control values to apply and
    the direction (+1 up, -1 down) the states should then move.

        from neurotemplate.modeling.scenarios import Actuation, SynthSpec

        cool_sp = "Zone Cooling Setpoint [degC]"
        return SynthSpec(
            states=["Zone Air Temperature [degC]"],
            ambient="Ambient Temperature [degC]",
            off={cool_sp: norm_stats[cool_sp]["max"]},
            actuations=[Actuation(
                name="cooling",
                values={cool_sp: norm_stats[cool_sp]["min"]},
                direction=-1,
            )],
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
