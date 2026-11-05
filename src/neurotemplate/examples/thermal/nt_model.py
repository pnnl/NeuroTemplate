"""A zone behind a thermostat, with a meter: each signal its own block."""
import json

from neurotemplate.modeling.heads import BoundedHead, HeadStack, RegressionHead

ZONE, HEAT, POWER, SETPOINT = "Zone [°C]", "Heat [kW]", "Power [kW]", "Cooling SP [°C]"
OUTDOOR, SOLAR = "Outdoor [°C]", "Solar [W/m²]"


def block_mapping(nm_mapping, cfg):
    return {
        "hvac": {"Y": [HEAT], "U": [SETPOINT], "D": [OUTDOOR], "ambient": OUTDOOR, "reads": [ZONE],
                 "model": "thermostat",
                 "params": {"thermostat": json.load(open("ladder/hvac.thermostat.json"))}},
        "zone": {"Y": [ZONE], "U": [], "D": [OUTDOOR, SOLAR], "driven_by": [HEAT],
                 "model": "rc_network", "ambient": OUTDOOR, "solar": SOLAR, "scale": {HEAT: 1.0},
                 "params": {"rc_network": json.load(open("ladder/zone.json"))}},
        "meter": {"Y": [POWER], "U": [], "D": [OUTDOOR], "driven_by": [HEAT], "model": "mlp",
                  "predicts": "value"},
    }


def output_heads(nm_mapping, norm_stats):
    return HeadStack([BoundedHead(0.0, float("inf")) if c == POWER else RegressionHead()
                      for c in nm_mapping["Y"]], nm_mapping["Y"])
