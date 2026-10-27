"""
What a training run prints, at one of four levels (`cfg.progress`):

    silent   nothing
    summary  seconds per epoch after the first, then the end summary
    live     one line rewritten in place every `update_every` epochs, then the summary
    log      the same line printed as a new line every `update_every` epochs, then the summary

The end summary says why training stopped and where the best dev loss was, which
is the one thing the loss trajectory says that changes what to do next.
"""

from __future__ import annotations

from time import perf_counter

from neuromancer.callbacks import Callback


def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


class Progress(Callback):
    """A Neuromancer `Callback` printing one training call's progress."""

    def __init__(self, level: str, update_every: int):
        super().__init__()
        self.level = level
        self.update_every = update_every

    def begin_train(self, trainer) -> None:
        self.start = perf_counter()
        self.epochs_run = 0
        self.best_epoch = None
        self.best_dev = float("inf")

    def end_epoch(self, trainer, output) -> None:
        self.epochs_run += 1
        dev = float(output[trainer.eval_metric])
        if dev < self.best_dev:
            self.best_dev, self.best_epoch = dev, self.epochs_run
        per_epoch = (perf_counter() - self.start) / self.epochs_run

        if self.level == "summary" and self.epochs_run == 1:
            print(f"{per_epoch:.1f} s per epoch; {trainer.epochs} epochs at most: "
                  f"about {_duration(per_epoch * trainer.epochs)}.", flush=True)
        if self.level in ("live", "log") and (
                self.epochs_run == 1 or self.epochs_run % self.update_every == 0):
            train = float(output[f"mean_{trainer.train_metric}"])
            line = (f"epoch {self.epochs_run}/{trainer.epochs}  train {train:.4g}  "
                    f"dev {dev:.4g}  best {self.best_dev:.4g} "
                    f"@ {self.best_epoch}  {per_epoch:.1f} s/epoch")
            if self.level == "live":
                print(f"\r{line}\033[K", end="", flush=True)
            else:
                print(line, flush=True)

    def end_train(self, trainer, output) -> None:
        if self.level == "silent" or self.best_epoch is None:
            return
        if self.level == "live":
            print()
        best = self.best_epoch
        if trainer.badcount > trainer.patience:
            why = f"stopped by patience at epoch {self.epochs_run}"
        else:
            # Patience never ran out, so the dev loss was still improving.
            why = (f"reached the {trainer.epochs}-epoch cap before patience "
                   f"({trainer.patience}) ran out; more epochs may help")
        print(f"Training {why}. Best dev loss {self.best_dev:.4g} at epoch {best}; "
              f"{_duration(perf_counter() - self.start)}.", flush=True)
