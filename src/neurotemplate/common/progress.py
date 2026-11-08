"""
What a training run prints, at one of four levels (`cfg.progress`):

    silent   nothing
    summary  seconds per step and the run's estimate after its first step, then the end summary
    live     one line rewritten in place at every evaluation, then the summary
    log      the same line printed as a new line, then the summary

The end summary says where the best dev loss was, which is the one thing the loss
trajectory says that changes what to do next.
"""

from __future__ import annotations

from math import isfinite
from time import perf_counter


def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


class Progress:
    """One training call's progress; also where a lost run stops."""

    def __init__(self, level: str):
        self.level = level

    def start(self, iters: int) -> None:
        self.begin = perf_counter()
        self.iters = iters
        self.evaluations = 0
        self.best_step = None
        self.best_dev = float("inf")

    def first_step(self) -> None:
        """After the first step: what the run will cost, before it is spent."""
        if self.level == "silent":
            return
        per_step = perf_counter() - self.begin
        print(f"{per_step:.2f} s per step; {self.iters} steps: about "
              f"{_duration(per_step * self.iters)}, before the dev evaluations.", flush=True)

    def update(self, step: int, train: float, dev: float) -> None:
        self.evaluations += 1
        if not isfinite(train):
            raise RuntimeError(
                f"The training loss is {train} at step {step}: a non-finite gradient has gone "
                "through the optimizer, so every weight is lost and no later step can recover. "
                "Look for a parameter or an input at an extreme (a capacity near zero, a box far "
                "off the data) in what this run declares.")
        if dev < self.best_dev:
            self.best_dev, self.best_step = dev, step
        per_step = (perf_counter() - self.begin) / step

        if self.level in ("live", "log"):
            line = (f"step {step}/{self.iters}  train {train:.4g}  dev {dev:.4g}  best "
                    f"{self.best_dev:.4g} @ {self.best_step}  {per_step:.2f} s/step")
            if self.level == "live":
                print(f"\r{line}\033[K", end="", flush=True)
            else:
                print(line, flush=True)

    def finish(self) -> None:
        if self.best_step is None:
            raise RuntimeError(
                f"No finite dev loss in {self.evaluations} evaluations, so there is no best "
                "model: what training would return is the model as built. The rollouts on the "
                "dev split blow up or read a missing value.")
        if self.level == "silent":
            return
        if self.level == "live":
            print()
        print(f"Training finished. Best dev loss {self.best_dev:.4g} at step "
              f"{self.best_step} of {self.iters}; "
              f"{_duration(perf_counter() - self.begin)}.", flush=True)
