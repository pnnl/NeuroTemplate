"""
Control hooks — the glue between the DPC machinery and *your* plant.

These answer what power costs and what the plots show. The objective — which
outputs are held inside which band, what power is priced at — is not a hook:
it is the `losses` table in control.yaml, over columns of the surrogate and
of your reference files (see `make_references.py`). What a command may reach is
`control_bounds` there too. The Y/U/D roles come from the frozen sys-id run.

Every hook works in physical units, the data's own.

The file must be named `nt_control.py` and live in the directory
you run `nt` from, or be pointed at by `user_hooks_path` in your YAML. Fill
the hooks in here, in place: each docstring is that hook's specification
for the library version you have installed. A control run keeps a copy of
this file, and reloading the run uses the copy.

Every required hook runs as written. The examples in comments are syntax:
the column names, and any number in them, belong to some other plant.
"""

from __future__ import annotations

import torch

from neurotemplate._typing import beartype
from neurotemplate.common.data_utils import NormStats


# ---------------------------------------------------------------------
# Required: which Y channels meter power
# ---------------------------------------------------------------------
@beartype
def power_channels(nm_mapping: dict[str, list[str]]) -> list[str]:
    """
    Y columns that meter power, in the order you want them plotted.

    Return [] when nothing is metered — that turns off the power row on the
    closed-loop plots. List them by name rather than by pattern: a name
    filter such as `"power" in c.lower()` also sweeps in channels nobody is
    billed for (a second fuel under an electricity tariff), and those then
    look like a cost to the objective.

        return ["Compressor Power [W]", "Fan Power [W]"]
    """
    return []


# ---------------------------------------------------------------------
# Required: context traces for the closed-loop plots
# ---------------------------------------------------------------------
@beartype
def plot_context_channels(nm_mapping: dict[str, list[str]]) -> list[str]:
    """
    D columns to overlay on the controlled-variable axis of the closed-loop
    plots — the driver that makes the trace legible (an ambient temperature,
    a load, irradiance). Return [] for none.
    """
    return []


# ---------------------------------------------------------------------
# The signal every loss term naming `power` reduces
# ---------------------------------------------------------------------
# By default the sum of `power_channels`: the meters the model predicts.
POWER_METRIC_IMPLEMENTED = True


@beartype
def power_metric(
    yn: torch.Tensor,
    U: torch.Tensor,
    D: torch.Tensor,
    nm_mapping: dict[str, list[str]],
) -> torch.Tensor:
    """
    Instantaneous power per timestep, shape [batch, time]: what a loss term
    naming `power` reduces (a mean for energy, a tariff-weighted mean for
    cost, a power mean for a peak), and what the reportcard integrates.

    Inputs are the trajectories in physical units: yn [batch, time, ny],
    U [batch, time, nu], D [batch, time, nd]. At each step, U is the command
    that drove the plant into that step's yn. The library converts no units:
    return Watts and the integral is in Wh.

    The default sums the meters `power_channels` names, as the model predicts
    them. What the policy learns from is this signal's gradient, so the
    meters must respond to the commands: model them in a block that sees the
    actuation (`nt guide thermal-zone`, step 4). Whether they do is the surrogate's question,
    answered on its own reportcard.

    Replace the body when nothing meters the power: a proxy from the physics
    of when the equipment runs, which does not decrease as the demand on it
    grows and is zero when it would not be called. A `relu` there stops the
    gradient on the inactive side; `soft_clamp_min`
    (`from neurotemplate.modeling.hinge import soft_clamp_min`) keeps the
    zero forward and a gradient back toward the active side.
    """
    channels = power_channels(nm_mapping)
    if not channels:
        raise NotImplementedError(
            "power_metric sums the meters power_channels names, and it names none. "
            "List the metered Y columns there, or write a proxy in power_metric.")
    return sum(yn[:, :, nm_mapping["Y"].index(c)] for c in channels)
