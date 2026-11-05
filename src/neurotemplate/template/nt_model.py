"""
Modeling hooks — the glue between NeuroTemplate and *your* dataset.

The library calls each function below for a dataset-specific answer (output heads,
model structure, physics checks) and knows nothing else about your data; the column
roles are in data.yaml. Each docstring is that hook's specification.

The file must be named `nt_model.py` and live in the directory
you run `nt` from, or be pointed at by `user_hooks_path` in your YAML. Fill
the hooks in here, in place: each docstring is that hook's specification
for the library version you have installed. A trained run keeps a copy of
this file, and everything that reloads the run uses the copy.

Every function here except `block_mapping` must exist, even
when its body is a trivial return. Every hook runs as written. Decide on model
structure (`block_mapping`)
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
    BoundedHead,
    HeadStack,
    RegressionHead,
)
from neurotemplate.modeling.scenarios import (  # noqa: F401  — cited in synth_cfg
    SynthSpec,
)


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

    All-regression (below) is the right default. Reach for another head only
    when the channel really is bounded:

      RegressionHead()                  continuous, MSE loss
      BoundedHead(lo, hi)               inside [lo, hi]; pass float("inf") /
                                        float("-inf") on an open side

    Bounds are in the column's physical units:

        BoundedHead(0.0, float("inf"))                     # physically >= 0
    """
    heads = []
    for col in nm_mapping["Y"]:
        # if "power" in col.lower():
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
    this function) for one block holding every channel. The "Blocks" section
    of `nt docs modeling` explains each choice at length; this is the contract.

    Worth doing whenever the plant is several parts that interact mainly
    through shared inputs. A joint model has to learn that the
    cross-couplings are ~zero, and short records rarely teach it; separate
    blocks get that for free.

        return {
            "hvac_a": {"Y": ["Zone A Heat [kW]"],
                       "U": ["Zone A Cool SP [degC]", "Zone A Heat SP [degC]"],
                       "D": ["Outdoor Temp [degC]"], "ambient": "Outdoor Temp [degC]",
                       "reads": ["Zone A Temp [degC]"], "model": "thermostat",
                       "params": {"thermostat": json.load(
                           open("ladder/hvac_a.thermostat.json"))}},
            "zone_a": {"Y": ["Zone A Temp [degC]"], "U": [],
                       "D": ["Outdoor Temp [degC]", "Solar [W/m2]", "Occupied [-]"],
                       "driven_by": ["Zone A Heat [kW]"], "model": "rc_network",
                       "ambient": "Outdoor Temp [degC]", "solar": "Solar [W/m2]",
                       "scale": {"Zone A Heat [kW]": 1.0},
                       "params": {"rc_network": json.load(open("ladder/zone_a.json"))}},
            "meter":  {"Y": ["Total Power [kW]"], "U": [], "D": ["Outdoor Temp [degC]"],
                       "driven_by": ["Zone A Heat [kW]"], "model": "mlp",
                       "residual": False},
        }

    Rules: every Y channel is in exactly one block. A block's U are its
    own; a U no block claims is shared by all. D is all of D unless a block
    names a subset, and blocks may overlap on it. `reads` lets a block see
    another block's Y as an ordinary input, as of the row before the step;
    `driven_by` as of this step, the block running after its producer. A
    block's `residual` overrides the run's: a meter says the value itself.
    `model` omitted means `cfg.model`, so a hook naming no family swaps every
    block from the config; `block_models` there overrides any one block.

    `rc_network` is a thermal zone: Y the zone temperature alone, its inputs
    a `driven_by` column (the heat a `thermostat` block produces), a U (a
    commanded heat) or a D, `ambient` the outdoor temperature it conducts to.
    Heat from supply flow × (supply − zone) temperature × 1.2 kJ/(m³K), or
    stage × capacity, in kW, positive heating; its rows mean what its role
    says (`nt guide setup`). `scale` is required and says what sets the units,
    since temperature alone places only gain / C: `{heat column: 1.0}` for a
    heat in kW, or `{"C_Ti": value}` to fit a power's gain as a COP.
    `thermostat` is Y the heat, U its setpoints, `reads` the zone, `ambient` the
    outdoor temperature (in its D), which some of its forms read.
    `context_steps` in the config is the record the zone's filter places its
    hidden nodes from. `params` maps a family to its declaration, every value a
    scalar (pinned) or [lo, hi] (fitted inside); only the family the block takes
    reads its entry:
      rc_network  Nodes (kWh/K), edges (K/kW), where each input enters and its
                  gain, and noise. `nt data ladder` fits a ladder of networks
                  and saves the one its tests pick; edit it freely, and
                  `--candidate` runs your own through the same tests.
      thermostat  Per side, the setpoint and its gain (kW/K), capacity (kW) and
                  offset (K), and more by form; and the `gain_limit` its zone's
                  capacitance sets. `nt data thermostat` saves it; a side the
                  record never exercised comes back null where the data placed
                  nothing: declare it.
    There is no default: a prior about a building is yours to state, and
    the commands are where the data states it. The reportcard says where
    each fit landed in its box. A fit pressed against an edge is the data
    disagreeing.

    Any other block may read a setpoint through a hinge, which changes the
    physics rather than the shapes:

        "zone_b": {
            ...,
            "inputs": {
                "Zone B Cool SP [degC]": {"target": "Zone B Temp [degC]", "map": "nrelu"},
                "Zone B Heat SP [degC]": {"target": "Zone B Temp [degC]", "map": "relu"},
            },
        }

    The setpoint is replaced by the thermostat's clamped error against the block's
    own predicted state — relu(zone − setpoint) for `nrelu`, relu(setpoint − zone)
    for `relu` — so a setpoint past the zone commands nothing, as on a real
    thermostat. Without it a model renders a setpoint as a smooth, always-active
    push, and a controller trained on it inherits that.

    Which structure is better for your plant is a model iteration, decided
    by the physics checks, the rollout and the whiteness, not by test RMSE,
    which declarations are not expected to improve.
    """
    return None


# ---------------------------------------------------------------------
# Optional: models of your own
# ---------------------------------------------------------------------
# Any torch module that honours the model contract gets training, the
# physics checks, reload, control (a policy or MPC) and export:
#
#     model(y, u, d) -> y_next
#       y, u, d  [batch, history_steps, n]   physical units, oldest row first;
#                                            row -1 of u and d drives this step
#       y_next   [batch, ny]
#
# Return {name: build(cfg, nm_mapping, norm_stats) -> nn.Module} and set
# `model: <name>` in the YAML. It is one module over every column, so
# block_mapping returns None; bound or level an output inside the model, or give
# it a `head_stack`. State that outlives the window goes through
# `rollout_state()` / `set_rollout_state(state)`, one [batch, width] tensor, None
# when cold (seed it from the window then). Keep the model's code, and any
# starting values, in this file: the run keeps a copy of it. The library's parts
# are yours to wire: `modeling.thermal`, `modeling.families`, `modeling.stepper`.
# `nt guide own-model` has the decisions; `nt docs modeling` the full contract.
#
# def custom_models():
#     return {"my_zone": lambda cfg, nm_mapping, norm_stats: MyZone(nm_mapping)}


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
               live. Required for every U. A hinge in `block_mapping` does not
               change these signs — they describe the column, not the feature
               built from it. It does change where a sweep starts: a hinge's
               step is measured only from zone temperatures where the target
               calls for something, and an end of its range that never does
               is skipped.

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

    All values are PHYSICAL units, as the model reads them; the library
    validates the whole spec against the column mapping before anything runs.

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
# Helper (not a hook): write the gradient-sign mapping JSON
# ---------------------------------------------------------------------
# `grad_gain > 0` requires `grad_path` to point at a JSON of known-sign
# partial derivatives: {output_col: {input_col: +1 | -1}}. Only list edges
# that hold everywhere in state space — the penalty is evaluated at random
# off-trajectory points, so a relationship that holds only while some
# controller is active will fire in the wrong places — a raw setpoint's
# partial, for one, is zero wherever the setpoint is satisfied. A hinged setpoint
# is differentiated as the hinge instead, so its sign is its push on the zone
# (-1 for a cooling `nrelu`).
# Run `python nt_model.py` to write the file.
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
