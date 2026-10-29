"""
Modeling hooks — the glue between NeuroTemplate and *your* dataset.

The library calls each function below for a dataset-specific answer (column
roles, output heads, model structure, physics checks) and knows nothing else
about your data. Each docstring is that hook's specification.

The file must be named `modeling_user_defined.py` and live in the directory
you run `nt` from, or be pointed at by `user_hooks_path` in your YAML. Fill
the hooks in here, in place: each docstring is that hook's specification
for the library version you have installed. A trained run keeps a copy of
this file, and everything that reloads the run uses the copy.

Every function here except `hyper_args` and `block_mapping` must exist, even
when its body is a trivial return. Fill in the column lists first; every
other hook runs as written. Then decide on model structure (`block_mapping`)
and the physics checks (`synth_cfg`) — those two shape what you get more
than any hyperparameter. The examples in comments are syntax: the column
names, and any number in them, belong to some other plant.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

from neurotemplate._typing import beartype
from neurotemplate.common.data_utils import NormStats
from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.heads import (  # noqa: F401  — cited in comments
    BinaryHead,
    BoundedHead,
    HeadStack,
    OrdinalHead,
    RegressionHead,
)
from neurotemplate.modeling.scenarios import (  # noqa: F401  — cited in synth_cfg
    SynthSpec,
)


# ---------------------------------------------------------------------
# Required: assign every dataframe column a role
# ---------------------------------------------------------------------
# Y = what the model predicts, U = what a controller can move,
# D = exogenous inputs: things you could know without the plant (weather, a
#     schedule). Anything the plant itself produces — a meter, a temperature
#     the plant affects — is Y or dropped, never D: a policy can only be fed
#     a D it has a forecast for. A price is not D either: the surrogate
#     would learn that price drives the plant. Prices go in reference files.
# Put in Y what a controller or its objective needs, and nothing else:
# every extra channel costs capacity, and a cumulative counter (an energy
# total) is an unbounded ramp no scaler can normalize.
# A column that is constant in the training split cannot be normalized and
# reaches the model in raw units; training warns about each one. Drop it:
# if it ever varies later, the rollout diverges from its first step.
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

    Names must match `df` columns exactly; `sec_elapsed` is added by the CSV
    loader from the timestamp index. `cfg.exclude_columns` holds substring
    patterns to drop. The library checks what this returns (unknown or
    doubly-assigned columns, an empty Y, an exclude pattern matching
    nothing), so the hook only has to say which column plays which role.
    """
    def keep(names: list[str]) -> list[str]:
        return [c for c in names if not any(p in c for p in cfg.exclude_columns)]

    return {"Y": keep(Y_COLUMNS), "U": keep(U_COLUMNS), "D": keep(D_COLUMNS),
            "t": ["sec_elapsed"]}


# ---------------------------------------------------------------------
# Required: one output head per Y channel
# ---------------------------------------------------------------------
@beartype
def data_head_config(
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> HeadStack:
    """
    Return a HeadStack with exactly one head per Y column, in Y order.

    Run `nt data heads` first — it reads the data and suggests a head (plus a
    pasteable snippet) for each Y column from the shape of its distribution,
    not its name.

    All-regression (below) is the right default. Reach for another head only
    when the channel really is discrete or bounded:

      RegressionHead()                  continuous, MSE loss
      BinaryHead()                      0/1 target, sigmoid + BCE; give the
                                        column None in column_norm_config
      BoundedHead(lo, hi)               clamped to [lo, hi]; pass
                                        float("inf") / float("-inf") on an
                                        open side for half-bounded
      OrdinalHead(n_levels=K)           K ordered levels, values 0..K-1
      OrdinalHead(n_levels=K, level_values=[v0, ...])
                                        K ordered levels at given values

    Levelled heads are rebalanced and threshold-calibrated for you
    (`balance_levels`, `calibrate_thresholds`). Judge them by the
    reportcard's level metrics against its persistence row, not by NRMSE:
    a regression head on staged equipment fits near the mean and can score
    well while never predicting the equipment on. Pin either by hand with
    `level_weights=` / `decision_thresholds=` and the flag off.

    Bounds and level values are in the column's physical units:

        BoundedHead(0.0, float("inf"))                     # physically >= 0
        OrdinalHead(n_levels=3, level_values=[0.0, 500.0, 1700.0])
    """
    heads = []
    for col in nm_mapping["Y"]:
        # if "on/off" in col.lower():
        #     heads.append(BinaryHead())
        # elif "power" in col.lower():
        #     heads.append(BoundedHead(0.0, float("inf")))
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
# Optional: model structure — blocks, a family per block, declared columns
# ---------------------------------------------------------------------
@beartype
def block_mapping(
    nm_mapping: dict[str, list[str]],
    cfg: Config,
) -> Optional[dict[str, dict]]:
    """
    Model structure: split Y into blocks, each its own sub-model with its own
    parameters, family (`model`) and hidden state. Return None (or delete
    this function) for one block holding every channel. The framework
    README's "Blocks" and "Declaring what a column is" sections explain each
    choice at length; this is the contract.

    Worth doing whenever the plant is several parts that interact mainly
    through shared inputs. A joint model has to learn that the
    cross-couplings are ~zero, and short records rarely teach it; separate
    blocks get that for free.

        return {
            "zone_a": {"Y": ["Zone A Temp [degC]"],
                       "U": ["Zone A Setpoint [degC]"],
                       "model": "2r2c", "params": {
                           "tau_fast_h": [lo, hi],      # see below
                           "tau_ambient_h": [lo, hi],
                           "tau_slow_h": [lo, hi]}},
            "meter":  {"Y": ["Total Power [kW]"],
                       "reads": ["Zone A Temp [degC]"],
                       "model": "mlp"},
        }

    Rules: every Y channel is in exactly one block. A block's U are its
    own; a U no block claims is shared by all. D is all of D unless a block
    names a subset, and blocks may overlap on it. `reads` lets a block see
    other blocks' Y as an ordinary input (not allowed in `2r2c`, whose
    stability guarantee depends on never reading state through a learned
    map). `model` omitted means `cfg.model`; under `mlp`/`gru` the split
    then needs `per_block: true`.

    `2r2c` is a thermal network: a state, a hidden mass, conduction to
    ambient. It cannot diverge under any input, and its storage reads off in
    hours, which is why it is the family to try for a temperature a
    controller will plan against. Each `2r2c` block states its storage prior
    in `params`, in hours, each a scalar (pinned) or [lo, hi] (fitted
    inside), in the order fast < ambient < slow with no overlap:
      tau_fast_h     how fast the state settles after a step of its actuator
      tau_ambient_h  the initial rate of a drift toward ambient, actuators off
      tau_slow_h     the same drift, once the mass is involved; `nt data
                     decay` estimates it from free-float periods, if the
                     data has any
    There is no default: a prior about a building is yours to state, held
    wide when you are unsure, and the reportcard says where each fit landed
    in its box. A fit pressed against an edge is the data disagreeing.

    A block may also declare what each column *is*, which changes the
    physics rather than the shapes:

        "zone_a": {
            ...,
            "ambient": "Outdoor Temp [degC]",
            "inputs": {
                "Zone A Setpoint [degC]": {"kind": "demand", "sign": -1,
                                           "state": "Zone A Temp [degC]",
                                           "capacity": [lo, hi]},
                "Solar [W/m2]":           {"kind": "linear", "sign": +1},
            },
        }

      demand   a setpoint, replaced for every family by the thermostat's
               clamped error against the block's own predicted state —
               relu(state − setpoint) for cooling (sign -1), the mirror for
               heating — so a setpoint past the state commands nothing, as
               on a real thermostat. Without it a model renders a setpoint
               as a smooth, always-active push, and a controller trained on
               it inherits that. Declare every setpoint that drives the
               block. In `2r2c` it drives a bounded, saturating heat input
               whose `capacity`, °C per hour at full stage, is a declared
               scalar or [lo, hi] (`nt data latch` estimates it); `"branch":
               "mlp"` routes it into the learned residual instead, losing the
               stability bound. Equipment that latches (starts past its
               setpoint and runs until the state is some way back) also sets
               `"hysteresis": {"on_at": [lo, hi], "off_at": [lo, hi]}`, in
               °C of error; `nt data latch` estimates both.
      linear   one sign-fixed coefficient times the physical value: metered
               heat or power delivered, irradiance. An actuator that is a
               power rather than a setpoint is `linear`, never `demand`.
      ambient  the outdoor-temperature column, as explicit conduction at the
               block's `tau_ambient_h` rate.

    `linear` and `ambient` act only inside `2r2c`; other families see those
    columns as ordinary inputs, with a build-time warning. Undeclared
    columns feed the block's learned residual, a small MLP (it maps only
    those columns to a heat flow, so it rarely needs width). A `2r2c` block
    that declares every column has no residual and no free constant heat
    gain: the most constrained model. Declaring only the setpoints leaves
    the rest to the residual. Which is better for your plant is a model
    iteration, decided by the physics checks and the rollout readouts, not
    by test NRMSE, which declarations are not expected to improve.
    """
    return None


# ---------------------------------------------------------------------
# Required: the plant's resting state, for the physics checks
# ---------------------------------------------------------------------
@beartype
def synth_cfg(
    nm_mapping: dict,
    norm_stats: dict,
) -> Optional[SynthSpec]:
    """
    Physics checks, run after training: equilibrium (states hold when nothing
    drives them), step response (each actuator moves each state the declared
    way, net of the model's drift at rest), reachability, and free decay (states relax toward ambient with the
    actuators off). They put the surrogate in states the recorded data never
    visits — which is where a controller will take it — and they are the
    readout most likely to tell you a surrogate is wrong for control when
    test error says it is fine. Returning None skips them, and you lose that.

      states   Y channels that hold at rest and relax toward `ambient`
               (temperatures, typically). Not meters: an equilibrium value
               written into a power channel is meaningless.
      off      the physical value of each U column when it asks for
               nothing. Not guessable — a cooling setpoint is off at its
               *maximum*, a heating setpoint at its *minimum* — so required
               for every U. Pick a value that is off across the whole
               `states` range, since equilibria are sampled from all of it.
               Other columns default to their recorded minimum ("no forcing"
               for irradiance, occupancy, flow); give an `off` for any
               column where the minimum is not "nothing".
      responds `{state: {control: +1 | -1}}`: the direction each state moves
               as that control INCREASES, from a state where the control is
               live. Required for every U. A `demand` declaration in
               `block_mapping` does not change these signs — they describe
               the column, not the feature built from it. It does change
               where a sweep starts: a demand's step is measured only from
               zone temperatures where the target calls for something, and
               an end of its range that never does is skipped.

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

    The checks are instruments, not gates. Read a failure as a diagnosis:
    a step that barely moves the state means its forcing is too weak; an
    equilibrium that drifts means something is integrated that should not be
    (a wrong ambient prior with a residual cancelling it, or an `off` value
    that is not off everywhere); a free decay far faster or slower than the
    plant could manage means the conduction is wrong. A surrogate that fails
    step response is not one to plan against. A small equilibrium drift
    after an honest attempt or two is worth noting and moving past.
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
def hyper_args(trial, cfg: Config) -> Config:
    """
    Set the fields this trial should vary on `cfg` (a copy) and return it.
    Any ModelingConfig field is fair game; keep the space small. Optuna
    selects on dev loss, which measures fit on the data you have — not how
    the model behaves where a controller will drive it.
    """
    # cfg.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # cfg.log2_hidden_size = trial.suggest_int("log2_hidden_size", 3, 6)
    # cfg.num_layers = trial.suggest_int("num_layers", 1, 3)
    # One column's lag. Moving the deepest lag also moves the stride, so
    # trials then train on different windows.
    # cfg.history = {**cfg.history,
    #                "Ambient Temperature [degC]": trial.suggest_int("ambient_lag", 0, 6)}
    return cfg


# ---------------------------------------------------------------------
# Helper (not a hook): write the gradient-sign mapping JSON
# ---------------------------------------------------------------------
# `grad_gain > 0` requires `grad_path` to point at a JSON of known-sign
# partial derivatives: {output_col: {input_col: +1 | -1}}. Only list edges
# that hold everywhere in state space — the penalty is evaluated at random
# off-trajectory points, so a relationship that holds only while some
# controller is active will fire in the wrong places — a raw setpoint's
# partial, for one, is zero wherever the setpoint is satisfied. A setpoint
# declared `demand` in block_mapping is differentiated as the demand feature
# instead, so its sign here is the declared `sign` (-1 for cooling).
# Run `python modeling_user_defined.py` to write the file.
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
