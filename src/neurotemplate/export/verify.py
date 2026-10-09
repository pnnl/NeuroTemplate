"""
Prove the exported graphs are the trained model, and produce the reference
that lets every consumer prove it again.

This is the one place that holds both models at once. It rolls the trained
NeuroTemplate system out over a whole split — the same `SystemPreview` that
training and evaluation use, not a reimplementation of it — and the exported
package out over the same rows through the shipped runtime, and compares the
outputs and the hidden-state trajectory step by step.

The torch side of that comparison is what ships as `expected_output.csv` and
`expected_state.csv`. A reference written by the export would make the
consumer's check circular: it would prove their runtime reproduces the
export, not the trained model, and any export bug would be baked into both
sides. Taken from torch, the consumer's check covers the whole chain.

Tolerances are fractions of each output column's normalisation stretch, so
one number means the same thing for a temperature, a power and a fraction.
The state lives in the model's internal (normalised) units and is compared
absolutely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from neurotemplate.export.runtime import Model
from neurotemplate.export.state import read_state
from neurotemplate.modeling.dynamics import stateful_blocks

#: Export time, same machine, ONNX Runtime vs torch. Measured at or near
#: float32 round-off for every family; the export refuses to write above it.
AGREEMENT_TOL = 1e-4
#: A consumer's machine, its onnxruntime vs the shipped torch reference. This
#: absorbs runtime *and* platform float differences, and cross-platform
#: reproducibility has not been measured, so it keeps 10x headroom.
REPRODUCE_TOL = 1e-3
#: How far, in stretches from its columns' means, the trained model's own
#: rollout may reach before the export calls it blown up. A rollout that grows
#: without bound amplifies float32 round-off until no agreement check can
#: pass, though the export is correct; one that stays within a hundred
#: stretches of its data is not that.
REACH_LIMIT = 100.0


@dataclass(frozen=True)
class Rollout:
    """One open-loop rollout: outputs in physical units, and the hidden state."""
    y: np.ndarray       # [steps, ny]
    state: np.ndarray   # [steps, state_width]


def torch_rollout(dyn_sys, nm_mapping: dict[str, list[str]],
                  norm_stats: dict[str, dict], frame: pd.DataFrame) -> Rollout:
    """The trained system's own rollout: seeded with the first `in_steps`
    rows of Y, then open loop over the rest of `frame`."""
    model = dyn_sys.nodes[0].callable
    blocks = stateful_blocks(model)

    def block(key: str) -> torch.Tensor:
        columns = nm_mapping[key]
        shift = np.array([norm_stats[c]["shift"] for c in columns])
        stretch = np.array([norm_stats[c]["stretch"] for c in columns])
        values = (frame[columns].to_numpy(dtype=np.float64) - shift) / stretch
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    data = {"Y": block("Y"), "U": block("U"), "D": block("D")}
    data["yn"] = data["Y"][:, :model.in_steps].clone()

    # The state is read after every forward call, which is one rollout step.
    states: list[torch.Tensor] = []

    def record_state(*_):
        if blocks:
            states.append(read_state(blocks).clone())

    hook = model.register_forward_hook(record_state)
    try:
        with torch.no_grad():
            yn = dyn_sys(data)["yn"][0].numpy()
    finally:
        hook.remove()

    # The System's last step consumes the final row's inputs and predicts a
    # row past the end of `frame`; keep only predictions of rows that exist.
    steps = len(frame) - model.in_steps
    yn = yn[model.in_steps:model.in_steps + steps]

    y_stats = [norm_stats[c] for c in nm_mapping["Y"]]
    y = yn.astype(np.float64) * [s["stretch"] for s in y_stats] + [s["shift"] for s in y_stats]
    state = (torch.cat(states)[:steps].numpy() if blocks
             else np.zeros((steps, 0), np.float32))
    return Rollout(y=y, state=state)


def onnx_rollout(model_dir, frame: pd.DataFrame) -> Rollout:
    """The exported package's rollout, through the runtime a consumer runs."""
    model = Model(model_dir)
    y, state = model.rollout({c: frame[c].to_numpy() for c in frame.columns})
    return Rollout(y=y, state=state)


@dataclass(frozen=True)
class Agreement:
    """Worst difference per output column (physical units) and in the state,
    and how far the trained model's own rollout strayed (`reach`: the
    largest |y - shift| / stretch it hit)."""
    y_error: dict[str, float]
    state_error: float
    reach: float

    def worst_fraction(self, norm_stats: dict[str, dict]) -> float:
        """The largest output error as a fraction of its column's stretch."""
        return max(err / norm_stats[c]["stretch"] for c, err in self.y_error.items())


def compare(reference: Rollout, exported: Rollout, y_columns: list[str],
            norm_stats: dict[str, dict]) -> Agreement:
    if reference.y.shape != exported.y.shape:
        raise ValueError(f"rollout shapes differ: torch {reference.y.shape}, "
                         f"onnx {exported.y.shape}")
    y_error = {c: float(np.abs(exported.y[:, i] - reference.y[:, i]).max())
               for i, c in enumerate(y_columns)}
    state_error = (float(np.abs(exported.state - reference.state).max())
                   if reference.state.size else 0.0)
    return Agreement(y_error=y_error, state_error=state_error,
                     reach=reach(reference, y_columns, norm_stats))


def reach(rollout: Rollout, y_columns: list[str],
          norm_stats: dict[str, dict]) -> float:
    """The largest |y - shift| / stretch a rollout hit; nan or inf when it
    went non-finite."""
    shift = np.array([norm_stats[c]["shift"] for c in y_columns])
    stretch = np.array([norm_stats[c]["stretch"] for c in y_columns])
    return float((np.abs(rollout.y - shift) / stretch).max())


def blew_up(reach: float) -> bool:
    """Did a rollout with this `reach` diverge (see `REACH_LIMIT`)?"""
    return not np.isfinite(reach) or reach > REACH_LIMIT


def require_agreement(agreement: Agreement, norm_stats: dict[str, dict]) -> None:
    """Refuse an export whose graphs are not the trained model."""
    worst = agreement.worst_fraction(norm_stats)
    if worst > AGREEMENT_TOL or agreement.state_error > AGREEMENT_TOL:
        raise RuntimeError(
            "REFUSING TO WRITE: the exported graphs do not reproduce the "
            f"trained model (worst output error {worst:.2e} of its column's "
            f"stretch, state error {agreement.state_error:.2e}; tolerance "
            f"{AGREEMENT_TOL:.0e}). Per-column errors: {agreement.y_error}. "
            f"The trained model's own rollout reached {agreement.reach:.3g} "
            "stretches from its columns' means: a rollout that grows far past "
            "its data grows float32 round-off with it, so a large reach means "
            "the model diverges on this split rather than that the export is "
            "wrong.")


def tolerances(y_columns: list[str], norm_stats: dict[str, dict]) -> dict:
    """The consumer-side tolerances, in each column's physical units."""
    return {
        "y": {c: REPRODUCE_TOL * norm_stats[c]["stretch"] for c in y_columns},
        "state": REPRODUCE_TOL,
    }
