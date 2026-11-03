"""
User-defined glue between the library and a specific dataset.

Every function in this module is the library's extension point for the
caller: the library calls it, gets back the dataset-specific configuration
(column roles, head topology, normalization overrides, etc.), and proceeds
without any further dataset knowledge.

Docstrings here describe the API contract and behavior; the function
bodies hold the dataset-specific decisions.
"""
import pandas as pd
import os
import json
from typing import Optional, Tuple

from optuna import Trial
from neurotemplate.modeling.config import ModelingConfig as Config
import torch

from neurotemplate.common.data_utils import NormStats
from neurotemplate.modeling.heads import (
    HeadStack,
    OrdinalHead,
    RegressionHead,
)
from neurotemplate.modeling.scenarios import SynthSpec

if os.environ.get("RUNTIME_TYPING", 0):
    from beartype import beartype
else:
    # passthrough for no type checking
    def beartype(fn): return fn


def _col_match(col, exact_list, partial_list):
    return (
        (col in exact_list) or any(pattern in col for pattern in partial_list)
    )


@beartype
def data_mapping(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Assign each dataframe column to a role: Y (states), U (controls),
    D (disturbances), or t (time).

    Returns a dict with keys "Y", "U", "D", "t", each mapping to a list of
    column names from `df` in dataframe order. A column landing in two roles
    raises.

    The `t` slot is currently fixed to the "sec_elapsed" column produced by
    the data loader.
    """

    # Exact column names that should be dropped entirely.
    EXCLUDE_EXACT = [
        "Time [s]",
    ]
    # Partial patterns: any column whose name contains one of these is excluded.
    EXCLUDE_PARTIAL = []

    # Y — building model outputs.
    Y_EXACT   = []
    Y_PARTIAL = ["Air Temperature [°C]", "RTU", "Cooling Stage"]

    # U — controller outputs (per-zone cooling setpoints).
    U_EXACT   = []
    U_PARTIAL = ["Cooling Setpoint"]

    # D — exogenous disturbances.
    D_EXACT   = [
        "Outdoor Dry-Bulb Temperature [°C]",
        "Global Horizontal Irradiance [W/m²]",
        "Occupancy Signal [-]",
    ]
    D_PARTIAL = []

    y_set = set()
    u_set = set()
    d_set = set()

    for col in df:
        if _col_match(col, EXCLUDE_EXACT, EXCLUDE_PARTIAL):
            continue

        if _col_match(col, Y_EXACT, Y_PARTIAL):
            y_set.add(col)

        if _col_match(col, U_EXACT, U_PARTIAL):
            u_set.add(col)

        if _col_match(col, D_EXACT, D_PARTIAL):
            d_set.add(col)

    # Check for overlap between sets — any column landing in two groups is a
    # pattern-matching mistake; raise immediately with the offending columns.
    for (name_a, set_a), (name_b, set_b) in [
        (("Y", y_set), ("U", u_set)),
        (("Y", y_set), ("D", d_set)),
        (("U", u_set), ("D", d_set)),
    ]:
        overlap = set_a & set_b
        if overlap:
            raise ValueError(
                f"Overlap in columns belonging to {name_a} and {name_b} sets, "
                f"check pattern matching! Columns: {overlap}"
            )

    # Restore df column order for deterministic downstream indexing.
    y_cols = [col for col in df.columns if col in y_set]
    u_cols = [col for col in df.columns if col in u_set]
    d_cols = [col for col in df.columns if col in d_set]

    # sec_elapsed is created by the get_data process from the timestamps in the
    # input data file, so it should always be there. Confirm though.
    assert 'sec_elapsed' in df.columns, "sec_elapsed column not found in dataframe!"

    nm_mapping = {
        'Y': y_cols,
        'U': u_cols,
        'D': d_cols,
        't': ['sec_elapsed']
    }

    return nm_mapping

# ---------------------------------------------------------------------
# Optional: per-Y-channel output head configuration
# ---------------------------------------------------------------------
@beartype
def data_head_config(
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> HeadStack:
    """
    Return one Head per Y column, packaged as a HeadStack.

    Any numeric bounds / level values are in PHYSICAL units. Channels
    without a head-kind reason fall through to RegressionHead (identity
    activation, MSE loss).

    Available head types:
      - RegressionHead  : continuous target, MSE loss.
      - BinaryHead      : 0/1 target, sigmoid activation, BCE loss.
      - BoundedHead     : continuous target with soft bounds during training,
                          clamped to [lo, hi] in the STE forward path.
                            Fully bounded   (both finite)  : sigmoid-band.
                            Lower-bounded   (hi=+inf)      : lo + softplus(z).
                            Upper-bounded   (lo=-inf)      : hi - softplus(-z).
                          Pass `use_ste=False` to skip the snap and feed back
                          the smooth activation directly.
      - OrdinalHead     : K-level ordered output. Specify `level_values`
                          (in physical units) and let the
                          head place thresholds at the midpoints. Activation
                          interpolates smoothly between levels; decode snaps
                          to one of the level_values via the STE combine.
    """
    def _ordinal_with_levels(levels, use_ste=True):
        return OrdinalHead(n_levels=len(levels), level_values=levels,
                           beta=20, use_ste=use_ste)

    heads = []
    for col in nm_mapping["Y"]:
        col_lower = col.lower()
        if "supply air flow" in col_lower:
            heads.append(_ordinal_with_levels([0.0, 0.82, 1.25]))
        elif "fan power" in col_lower:
            heads.append(_ordinal_with_levels([0.0, 483.0, 1680.7]))
        elif "dx power" in col_lower:
            heads.append(_ordinal_with_levels([0.0, 4300.0, 8800.0], use_ste=False))
        elif "cooling stage" in col_lower:
            heads.append(OrdinalHead(n_levels=3))
        else:
            heads.append(RegressionHead())
    return HeadStack(heads, nm_mapping["Y"])


# ---------------------------------------------------------------------
# Optional: per-column normalization overrides
# ---------------------------------------------------------------------
@beartype
def column_norm_config(
    nm_mapping: dict[str, list[str]],
) -> dict[str, Optional[str]]:
    """
    Per-column overrides on the default normalization. Returns a mapping
    {column_name: norm_type}, where `norm_type` is one of the strings
    accepted by `compute_norm_stats` ("min_max", "abs_max", "z_score") or
    None to skip normalization entirely. Columns absent from the returned
    dict use the run-wide default from cfg.norm_type.
    """
    all_cols = nm_mapping["Y"] + nm_mapping["U"] + nm_mapping["D"]
    overrides: dict[str, Optional[str]] = {}
    for col in all_cols:
        col_lower = col.lower()
        if any(p in col_lower for p in ("flow", "irradiance", "power")):
            overrides[col] = "min_max"
        elif "occupancy" in col_lower:
            overrides[col] = None   # binary 0/1 — keep in natural units
    return overrides


# ---------------------------------------------------------------------
# Required: the plant's resting state, for the physics checks
# ---------------------------------------------------------------------
@beartype
def synth_cfg(
    nm_mapping: dict,
    norm_stats: dict,
) -> Optional[SynthSpec]:
    """The RTU strip-mall plant at rest, and the one way it can be driven.

    Only the zone air temperature behaves like a temperature — supply air,
    flow, stage and the two power channels are readouts, and naming them as
    states would have the equilibrium scenario sampling "temperatures" out of
    a 0-6000 W range. That distinction is the whole point of `states`.

    The single control is a cooling setpoint, so its "off" is its *maximum*
    (never calls for cooling) and full actuation is its minimum, which should
    drive the zone down.
    """
    (setpoint,) = nm_mapping["U"]
    ambient = next(c for c in nm_mapping["D"] if "Outdoor Dry-Bulb" in c)
    states = [c for c in nm_mapping["Y"] if "Zone Air Temperature" in c]
    return SynthSpec(
        states=states,
        ambient=ambient,
        off={setpoint: norm_stats[setpoint]["max"]},
        # Raising a cooling setpoint asks for less cooling, so the zone warms:
        # +1. The library sweeps the setpoint from its off value (the maximum)
        # down to its minimum and expects the zone to fall.
        responds={c: {setpoint: +1} for c in states},
    )


# ---------------------------------------------------------------------
# Required: Optuna hyperparameter suggestion
# ---------------------------------------------------------------------
@beartype
def hyper_args(
    trial: Trial,
    cfg: Config,
) -> Config:
    """Suggest high-impact hyperparameters for Optuna trial."""
    return cfg


# ---------------------------------------------------------------------
# Optional: subplot guidance
# ---------------------------------------------------------------------
@beartype
def sub_plot(
    df_true: pd.DataFrame, 
    column_name: str
) -> Optional[Tuple[list[str], list[dict]]]:
    # For a given column name that is being plotted
    # what should be plotted on the axis below it, sharing the x axis?
    # This is a place to put the co-variates of most interest!
    # 
    # default behavior:
    return None


# ---------------------------------------------------------------------
# Optional: gradient mapping JSON helper
# ---------------------------------------------------------------------
@beartype
def make_all_gradient_mappings(
    rel_path: str = "../data/grad",
) -> str:
    """
    Build and save gradient-sign mapping JSONs.

    Physics-only: each edge comes from energy or mass conservation and
    must hold at any state in the training distribution, independent of
    what the supervisory controller would do. Controller-mediated effects
    (e.g. outdoor temp → AHU fan flow, setpoint → zone temp) are
    deliberately omitted — they are correct on-manifold for the trained
    controller but spurious at the random sample points where the
    gradient penalty is evaluated.

    Writes one file per non-empty subset of {Y, U, D} as inputs so a run
    can pick how strict the prior should be. Outputs that aren't present
    in a given run's Y are filtered silently downstream by get_grad_masks.

    Format: {output_col: {input_col: +1 or -1, ...}, ...}

    Returns the directory the files were written to.
    """

    # --- D columns (disturbances) ---
    OUT = "Outdoor Dry-Bulb Temperature [°C]"
    GHI = "Global Horizontal Irradiance [W/m²]"
    OCC = "Occupancy Signal [-]"

    # --- per-zone names ---
    zones = [f"Zone {i} Air Temperature [°C]"           for i in range(1, 6)]
    vavT  = [f"VAV Box {i} Supply Air Temperature [°C]" for i in range(1, 6)]
    vavF  = [f"VAV Box {i} Supply Air Flow [m³/s]"      for i in range(1, 6)]

    # --- AHU output names ---
    AHU_COOL = "AHU Cooling Coil Cooling Power [W]"
    AHU_TSUP = "AHU Supply Air Temperature [°C]"

    mapping: dict[str, dict[str, int]] = {}

    # Zone temperature — zone energy balance.
    #   +outdoor/+GHI/+occ : envelope conduction, solar gain, internal gain
    #   +self              : thermal mass continuity
    #   +VAV i supply T    : warmer supply air -> warmer zone at fixed flow
    #   +AHU supply T      : strictly physical when VAV supply T isn't an
    #                        input to the model (configs without VAV in Y);
    #                        in configs with VAV in Y the one-step partial
    #                        at fixed VAV_T is ~0 and the one-sided sign
    #                        penalty doesn't fire — so this is a free
    #                        guardrail against spurious negative correlation.
    for i, z in enumerate(zones):
        mapping[z] = {
            OUT:      +1,
            GHI:      +1,
            OCC:      +1,
            z:        +1,
            vavT[i]:  +1,
            AHU_TSUP: +1,
        }

    # AHU supply air temperature — coil thermodynamics + air-column inertia.
    mapping[AHU_TSUP] = {
        AHU_TSUP: +1,   # self: coil and air-column thermal mass
        AHU_COOL: -1,   # more heat removed by coil -> colder supply
    }

    # VAV supply temperature — duct passthrough from AHU supply (reheat = 0).
    for vt in vavT:
        mapping[vt] = {
            vt:       +1,   # self: passthrough continuity
            AHU_TSUP: +1,   # warmer AHU supply -> warmer VAV supply
        }

    # AHU cooling power, AHU fan flow, AHU fan pressure rise, VAV box flows:
    # no physics-only prior. These are controller-commanded actuator outputs
    # whose next-step value is whatever the controller demands; any monotonic
    # relationship we could write down would be a closed-loop artifact, not a
    # conservation law.

    # --- Classify every input column in `mapping` as Y or D so we can
    # write the subset files. Cooling setpoints are intentionally *not*
    # listed as a valid U input: zone temp is not monotone in setpoint
    # (a setpoint far above current temp produces no actuator response),
    # so any setpoint→temp sign constraint is a controller artifact that
    # fires on the wrong half of the random-sample space. ---
    Y_cols = set(
        zones + vavT + vavF +
        ["AHU Cooling Coil Cooling Power [W]",
         "AHU Supply Fan Air Flow [m³/s]",
         "AHU Supply Fan Pressure Rise [Pa]",
         "AHU Supply Air Temperature [°C]"]
    )
    D_cols = {OUT, GHI, OCC}

    def filter_by_input_kinds(mp: dict, kinds: set[str]) -> dict:
        keep = set()
        if "y" in kinds: keep |= Y_cols
        if "d" in kinds: keep |= D_cols
        # 'u' deliberately contributes nothing — see note above.
        out = {}
        for y_col, deps in mp.items():
            subset = {k: v for k, v in deps.items() if k in keep}
            if subset:
                out[y_col] = subset
        return out

    os.makedirs(rel_path, exist_ok=True)
    # Remove any stale files from prior runs (e.g. older versions wrote
    # setpoint→zone-temp edges into yu.json / u.json; clean those out so
    # an old grad_path config can't silently keep loading them).
    for stale in os.listdir(rel_path):
        if stale.endswith(".json"):
            os.remove(os.path.join(rel_path, stale))

    for kinds in ("y", "u", "d", "yu", "yd", "ud", "yud"):
        sub = filter_by_input_kinds(mapping, set(kinds))
        if not sub:
            continue  # SampledGradientPenalty rejects empty mappings
        path = os.path.join(rel_path, f"{kinds}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sub, f, indent=2, sort_keys=True)

    return rel_path


if __name__ == "__main__":
    make_all_gradient_mappings()
