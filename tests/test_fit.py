"""The training loop's budget, schedule and rollback (`common.fit`)."""

import pytest
import torch

from neurotemplate.common.fit import COLLAPSE, fit, schedule
from neurotemplate.common.progress import Progress


class Scripted(torch.nn.Module):
    """One weight to train; the dev losses it reports are given in advance."""

    def __init__(self, devs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.devs = iter(devs)

    def forward(self, batch):
        if self.training:
            return {"train_loss": 3.0 * self.weight ** 2}
        return {"dev_loss": torch.tensor(next(self.devs))}


def test_fit_runs_its_budget_on_the_schedule_and_rolls_back_a_collapse():
    share = schedule(100, final_ratio=0.01)
    assert share(0) == pytest.approx(1 / 5)       # ramps over 5% of the steps
    assert share(4) == pytest.approx(1.0)         # to the peak
    assert share(99) == pytest.approx(0.01, abs=1e-3)

    # Above COLLAPSE times the best at two evaluations running rolls back, and the
    # count starts again after each.
    high = COLLAPSE * 1.1
    devs = [1.0, high, high, 1.0, high, high, high, high, 0.5, 0.5]
    problem = Scripted(devs)
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.1)
    fitted = fit(problem, optimizer, lambda: {}, [{}], iters=10, eval_every=1,
                 final_ratio=0.01, device="cpu", progress=Progress("silent"))
    assert fitted.rollbacks == [3, 6, 8]
    assert fitted.best_dev == 0.5
    # The schedule kept its place through the rollbacks.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * schedule(10, 0.01)(10))
