"""
`Stepper`: the usual recipe, behind the contract.

    normalise → derive inputs → core → residual → heads → physical units

A family is its core: a module taking normalised `[batch, rows, n]` windows and
returning one value per output, `z [batch, ny]` (see `families`, `gru`). The
residual applies per output: `residual` is one flag or one per output, and an
output whose head reads its `z` as anything but the value takes none.

Using it is optional. A model of your own honours the contract however it likes,
and may build on this, or on `Scaler`, as much as it finds useful.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from neurotemplate.modeling.heads import HeadStack
from neurotemplate.modeling.prestack import PreStack

_KEYS = ("Y", "U", "D")


class Scaler(nn.Module):
    """A run's normalisation: each key's columns to and from the units a family
    computes in."""

    def __init__(self, nm_mapping: dict[str, list[str]], norm_stats: dict[str, dict]):
        super().__init__()
        for key in _KEYS:
            for field in ("shift", "stretch"):
                self.register_buffer(f"{key.lower()}_{field}", torch.tensor(
                    [float(norm_stats[c][field]) for c in nm_mapping[key]],
                    dtype=torch.float32), persistent=False)

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        """`x` [..., n] of key `key` ("Y", "U" or "D") in normalised units."""
        key = key.lower()
        return (x - getattr(self, f"{key}_shift")) / getattr(self, f"{key}_stretch")

    def physical_y(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.y_stretch + self.y_shift


class Stepper(nn.Module):
    """A core, in normalised units, behind the physical contract.

    `head_stack` is declared in physical units and rescaled here.
    """

    def __init__(self, core: nn.Module, nm_mapping: dict[str, list[str]],
                 norm_stats: dict[str, dict], history_steps: int,
                 residual: Union[bool, Sequence[bool]] = True,
                 head_stack: Optional[HeadStack] = None,
                 pre_stack: Optional[PreStack] = None):
        super().__init__()
        self.core = core
        self.scale = Scaler(nm_mapping, norm_stats)
        self.history_steps = history_steps
        self.ny, self.nu, self.nd = (len(nm_mapping[key]) for key in _KEYS)
        if head_stack is not None and head_stack.raw_width != self.ny:
            raise ValueError(f"head_stack.raw_width={head_stack.raw_width} does not match "
                             f"ny={self.ny}; a model's heads are width 1.")
        # A new stack, in the core's units; post-training calibration writes
        # its decision thresholds into this one, never the caller's.
        self.head_stack = None if head_stack is None else head_stack.normalized(norm_stats)
        self.pre_stack = pre_stack
        flags = [residual] * self.ny if isinstance(residual, bool) else list(residual)
        mask = torch.tensor(flags, dtype=torch.bool)
        if self.head_stack is not None:
            mask = mask & self.head_stack.residual_mask()
        self.register_buffer("residual_mask", mask, persistent=False)

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return self.scale.normalize(key, x)

    @property
    def y_stretch(self) -> torch.Tensor:
        return self.scale.y_stretch

    @property
    def y_shift(self) -> torch.Tensor:
        return self.scale.y_shift

    def sign_step(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """The core plus the residual, before any head and after any derived
        input, normalised: what the gradient sign penalty differentiates. Every
        head's activation is monotone increasing, so the signs it checks are the
        activated output's too."""
        z = self.core(y, u, d)
        return torch.where(self.residual_mask, z + y[:, -1], z)

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        y = self.scale.normalize("Y", y)
        u = self.scale.normalize("U", u)
        d = self.scale.normalize("D", d)
        if self.pre_stack is not None:
            u = self.pre_stack(y, u)
        z = self.sign_step(y, u, d)
        if self.head_stack is not None:
            z = self.head_stack(z)
        return self.scale.physical_y(z)
