"""
A physical parameter fitted inside a declared box.

    value = exp(log low + (log high − log low) · σ(θ))     a log box
    value = low + (high − low) · σ(θ)                       a linear box

A box rather than a free parameter or a frozen one: the prior is held where
it is known — "this building's slow mode is 20 to 60 hours" — and training
moves the parameter only inside it. θ starts at 0, the middle of the box, so
a box centred on a data estimate starts the fit there; σ flattens toward
either edge, so small moves are cheap and large ones are not. Log-space for
scales (8 h → 16 h is the same move as 30 h → 60 h); linear for a quantity
that may sit at zero, such as a thermostat threshold. A box with `low ==
high` pins the parameter, and a scalar declaration means exactly that.

A box is a guardrail, not a cure. A parameter the data pulls against its
bound sits there, and that is the diagnostic worth printing: `fraction` is
its position in the box, 0 at the lower bound and 1 at the upper.

Code finds these with `isinstance` over `named_modules()` — the optimizer
group, the reportcard — so a module that owns one needs no registration.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

#: Within this share of the box of either bound, a parameter is reported as
#: pressed against it.
NEAR_EDGE = 0.1


def parse_box(name: str, value, log: bool = True) -> tuple[float, float]:
    """`[low, high]`, or a scalar meaning `low == high`: positive for a log
    box, non-negative for a linear one."""
    pair = value if isinstance(value, (list, tuple)) else [value, value]
    numbers = len(pair) == 2 and not any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in pair)
    if not numbers or not (0 < pair[0] if log else 0 <= pair[0]) or pair[0] > pair[1]:
        sign, op = ("positive", "<") if log else ("non-negative", "<=")
        raise ValueError(f"{name} must be a {sign} number, or [min, max] with "
                         f"0 {op} min <= max; got {value!r}.")
    return float(pair[0]), float(pair[1])


class BoundedParameter(nn.Module):
    """One value per label, each inside the same box `[low, high]`, in log
    space (positive scales) or linear space (`log=False`).
    """

    def __init__(self, name: str, low: float, high: float, labels: tuple[str, ...],
                 log: bool = True):
        super().__init__()
        self.name = name
        self.low, self.high = float(low), float(high)
        self.labels = tuple(labels)
        self.log = log
        self.theta = nn.Parameter(torch.zeros(len(labels)))
        to_space = math.log if log else float
        self.register_buffer("lower", torch.tensor(to_space(low)))
        self.register_buffer("upper", torch.tensor(to_space(high)))

    @property
    def pinned(self) -> bool:
        return self.low == self.high

    def fraction(self) -> torch.Tensor:
        """Position in the box, per label: 0 at `low`, 1 at `high`."""
        return torch.sigmoid(self.theta)

    def value(self) -> torch.Tensor:
        inside = self.lower + (self.upper - self.lower) * self.fraction()
        return torch.exp(inside) if self.log else inside

    def describe(self) -> dict:
        """The box and, per label, the value and its position in the box."""
        with torch.no_grad():
            value, fraction = self.value(), self.fraction()
        return {
            "low": self.low, "high": self.high, "pinned": self.pinned,
            "value": {label: round(float(v), 4) for label, v in zip(self.labels, value)},
            "fraction": {label: round(float(f), 3) for label, f in zip(self.labels, fraction)},
        }


def bounded_parameters(module: nn.Module) -> list[BoundedParameter]:
    """Every `BoundedParameter` under `module`."""
    return [m for m in module.modules() if isinstance(m, BoundedParameter)]


def edge_note(name: str, fraction: float) -> str | None:
    """The diagnostic for a parameter pressed against a bound, or None."""
    if fraction <= NEAR_EDGE:
        return (f"{name} at {(1 - fraction) * 100:.0f}% of the way to its lower "
                "bound — the data (or the seed) disagrees with the prior")
    if fraction >= 1 - NEAR_EDGE:
        return (f"{name} at {fraction * 100:.0f}% of the way to its upper "
                "bound — the data (or the seed) disagrees with the prior")
    return None


def physical_parameters(module: nn.Module) -> list[nn.Parameter]:
    """The raw θ of every bounded parameter under `module`.

    Training steps these at their own learning rate (`cfg.physical_lr`):
    Adam moves a raw parameter by about `lr` per update, and a box position
    has its whole box to travel where a network weight moves by thousandths.
    A pinned one is included too; its gradient is zero, so Adam leaves it.
    """
    return [b.theta for b in bounded_parameters(module)]
