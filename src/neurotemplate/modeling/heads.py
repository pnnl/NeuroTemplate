"""
Per-channel output heads.

Each Y channel has a head that maps the model's raw output `z` to the value fed
back into the rollout and compared with the truth, and scores it. Three kinds:

  regression    the identity, MSE. The default.
  bounded       inside [lo, hi]:
                    both finite     y = lo + (hi - lo) * sigmoid(beta * z / (hi - lo))
                    lower only      y = lo + softplus(z, beta)
                    upper only      y = hi - softplus(-z, beta)
                Pass float("inf") / float("-inf") on an open side. Every form has a
                nonzero gradient everywhere; a ReLU's would stall a channel that
                drifts to the wrong side. `beta` is the head's sharpness in the
                units of the model that holds it.

A residual (y_next = z + y_last) only means something where `z` lives in the same
space as `y`, so a bounded head takes none (`Head.residual_compatible`).

A head is declared in the channel's physical units. A model that works in normalised
units (every built-in family) holds the same stack rescaled into its own,
`HeadStack.normalized(norm_stats)`. Heads are rebuilt on reload from the JSON-safe
`HeadStack.to_config()` the run logs.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurotemplate._typing import beartype


class Head(nn.Module, ABC):
    """One head per Y channel: `activate`, `loss`, and `config()` for reload."""

    kind: str = ""
    residual_compatible: bool = False

    @abstractmethod
    def activate(self, z: torch.Tensor) -> torch.Tensor:
        """The channel's value from its raw output, same shape."""

    def loss(self, y_state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_state, target)

    def config(self) -> dict:
        """JSON-safe kwargs needed to reconstruct this head. Default: none."""
        return {}


class RegressionHead(Head):
    """Identity activation, MSE loss."""

    kind = "regression"
    residual_compatible = True

    def activate(self, z: torch.Tensor) -> torch.Tensor:
        return z


class BoundedHead(Head):
    """A value inside [lo, hi], one side possibly open; see the module docstring."""

    kind = "bounded"

    def __init__(self, lo: float, hi: float, beta: float = 5.0):
        super().__init__()
        self._lo_finite, self._hi_finite = math.isfinite(lo), math.isfinite(hi)
        if not (self._lo_finite or self._hi_finite):
            raise ValueError("BoundedHead requires at least one finite bound "
                             "(use RegressionHead for an unbounded channel)")
        if self._lo_finite and self._hi_finite and hi <= lo:
            raise ValueError(f"BoundedHead requires hi > lo, got lo={lo} hi={hi}")
        if beta <= 0:
            raise ValueError(f"BoundedHead requires beta > 0, got beta={beta}")
        self.lo, self.hi, self.beta = float(lo), float(hi), float(beta)

    def activate(self, z: torch.Tensor) -> torch.Tensor:
        if self._lo_finite and self._hi_finite:
            # Scaled by the gap, so the same beta means the same transition width
            # as a fraction of the band, at any band.
            return self.lo + (self.hi - self.lo) * torch.sigmoid(
                self.beta * z / (self.hi - self.lo))
        if self._lo_finite:
            return self.lo + F.softplus(z, beta=self.beta)
        return self.hi - F.softplus(-z, beta=self.beta)

    def config(self) -> dict:
        return {"lo": _float_to_json(self.lo), "hi": _float_to_json(self.hi), "beta": self.beta}


_KIND_TO_CLASS: dict[str, type[Head]] = {cls.kind: cls for cls in (RegressionHead, BoundedHead)}


def _float_to_json(x: float):
    """Encode floats so JSON can represent infinities portably."""
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _json_to_float(x) -> float:
    if isinstance(x, str):
        if x in ("inf", "-inf"):
            return float(x)
        raise ValueError(f"Unrecognized float string in head config: {x!r}")
    return float(x)


class HeadStack(nn.Module):
    """One head per channel of a Y vector: activates each and sums their losses."""

    @beartype
    def __init__(self, heads: list[Head], channel_names: list[str]):
        super().__init__()
        if len(heads) != len(channel_names):
            raise ValueError(f"heads ({len(heads)}) and channel_names ({len(channel_names)}) "
                             "must have the same length")
        if len(set(channel_names)) != len(channel_names):
            raise ValueError(f"channel_names must be unique, got {channel_names}")
        self.heads = nn.ModuleList(heads)
        self.channel_names = list(channel_names)
        self.n_channels = len(heads)
        # The exact class: a subclass may activate.
        self._all_identity = all(type(h) is RegressionHead for h in heads)
        self.register_buffer("_residual", torch.tensor(
            [h.residual_compatible for h in heads]), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[..., n_channels] raw outputs -> [..., n_channels] values."""
        if self._all_identity:
            return z
        return torch.stack([self.activate_channel(z[..., i], i)
                            for i in range(self.n_channels)], dim=-1)

    def activate_channel(self, z: torch.Tensor, i: int) -> torch.Tensor:
        """Channel i's value from its raw output, what a block driven by it reads."""
        return self.heads[i].activate(z)

    def residual_mask(self) -> torch.Tensor:
        """Which outputs read their `z` as the value, so may take a residual."""
        return self._residual

    def loss(self, y_state: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Mean across channels of each head's loss. A mean, so swapping head types
        does not change the loss's magnitude: an all-regression stack is a blanket
        MSE over the Y tensor."""
        total = y_state.new_zeros(())
        for i, head in enumerate(self.heads):
            total = total + head.loss(y_state[..., i], y_true[..., i])
        return total / self.n_channels

    def to_config(self) -> dict[str, dict]:
        """{column: {kind, ...kwargs}}, JSON-safe, in this stack's own units."""
        return {name: {"kind": head.kind, **head.config()}
                for name, head in zip(self.channel_names, self.heads)}

    @classmethod
    def from_config(cls, data: dict[str, dict], channel_names: list[str]) -> "HeadStack":
        """Inverse of to_config: exactly one entry per channel, or it raises."""
        unknown = set(data) - set(channel_names)
        missing = [name for name in channel_names if name not in data]
        if unknown or missing:
            raise ValueError("head config does not match the Y channels: "
                             f"not in Y {sorted(unknown)}, missing {missing}")
        heads: list[Head] = []
        for name in channel_names:
            kwargs = dict(data[name])
            kind = kwargs.pop("kind")
            if kind not in _KIND_TO_CLASS:
                raise ValueError(f"Unknown head kind '{kind}' for channel '{name}'. "
                                 f"Expected one of: {sorted(_KIND_TO_CLASS)}.")
            for key in ("lo", "hi"):
                if key in kwargs:
                    kwargs[key] = _json_to_float(kwargs[key])
            heads.append(_KIND_TO_CLASS[kind](**kwargs))
        return cls(heads, channel_names)

    def normalized(self, norm_stats: dict[str, dict]) -> "HeadStack":
        """The same heads for a model that emits each channel normalised by
        `norm_stats`: every bound mapped through `(x - shift) / stretch`."""
        return self._rescaled(
            lambda name, x: (x - norm_stats[name]["shift"]) / norm_stats[name]["stretch"])

    def physical(self, norm_stats: dict[str, dict]) -> "HeadStack":
        """Inverse of `normalized`: `x * stretch + shift`."""
        return self._rescaled(
            lambda name, x: x * norm_stats[name]["stretch"] + norm_stats[name]["shift"])

    def _rescaled(self, to) -> "HeadStack":
        config = self.to_config()
        for name, entry in config.items():
            for key in ("lo", "hi"):
                if key in entry:
                    entry[key] = _float_to_json(to(name, _json_to_float(entry[key])))
        return HeadStack.from_config(config, self.channel_names)
