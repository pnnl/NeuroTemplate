"""What a `power` loss term prices: the meter the model predicts."""

POWER = "Power [kW]"
POWER_METRIC_IMPLEMENTED = True


def power_channels(nm_mapping):
    return [POWER]


def plot_context_channels(nm_mapping):
    return ["Outdoor [°C]"]


def power_metric(yn, U, D, nm_mapping):
    return yn[:, :, nm_mapping["Y"].index(POWER)]
