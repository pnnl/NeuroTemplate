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

from neurotemplate.modeling.config import ModelingConfig as Config
import torch

from neurotemplate.common.data_utils import NormStats
from neurotemplate.modeling.heads import (
    BoundedHead,
    HeadStack,
    RegressionHead,
)
from neurotemplate.metrics.physics import SynthSpec

if os.environ.get("RUNTIME_TYPING", 0):
    from beartype import beartype
else:
    # passthrough for no type checking
    def beartype(fn): return fn


# ---------------------------------------------------------------------
# Optional: per-Y-channel output head configuration
# ---------------------------------------------------------------------
@beartype
def output_heads(
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
      - BoundedHead     : continuous target inside [lo, hi].
                            Fully bounded   (both finite)  : sigmoid-band.
                            Lower-bounded   (hi=+inf)      : lo + softplus(z).
                            Upper-bounded   (lo=-inf)      : hi - softplus(-z).
    """
    heads = []
    for col in nm_mapping["Y"]:
        col_lower = col.lower()
        if any(p in col_lower for p in ("supply air flow", "fan power", "dx power")):
            heads.append(BoundedHead(0.0, float("inf")))
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
