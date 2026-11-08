"""The training loop's budget and schedule (`common.fit`)."""

import itertools

import pytest
import torch

from neurotemplate.common.fit import EVALUATIONS, FINAL_SHARE, fit, schedule
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


def test_fit_runs_its_budget_on_the_schedule_and_keeps_the_best_dev():
    share = schedule(100)
    assert share(0) == pytest.approx(1 / 5)       # ramps over 5% of the steps
    assert share(4) == pytest.approx(1.0)         # to the peak
    assert share(99) == pytest.approx(FINAL_SHARE, abs=1e-3)

    # One evaluation per step at this budget; the best is the one kept, not the last.
    devs = [3.0, 2.0, 0.5] + [1.0] * (EVALUATIONS - 3)
    problem = Scripted(devs)
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.1)
    fitted = fit(problem, optimizer, itertools.repeat({}), [{}], iters=EVALUATIONS,
                 device="cpu", progress=Progress("silent"))
    assert fitted.best_dev == 0.5
    assert torch.equal(problem.weight.detach(), fitted.best_state["weight"])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * schedule(EVALUATIONS)(EVALUATIONS))
