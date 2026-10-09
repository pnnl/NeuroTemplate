"""
The seed ensemble: pipeline.train_and_evaluate trains cfg.ensemble_seeds
models (at cfg.seed, cfg.seed + 1, ...) and reports the spread across them.

Every number the reportcard prints is otherwise a single seed's draw. These are
the two claims that matter:

  - ensemble_seeds=1 reproduces today's single-model behaviour exactly (no
    ensemble, no extra section).
  - ensemble_seeds=3 trains two more models but does not change the primary
    (seed + 0) one: same weights, same metrics, whether or not a spread is
    also being computed — and the spread it adds is a real "Seed ensemble
    spread" section on the reportcard.

Both are checked with a real (tiny) run through train_and_evaluate rather
than a mock, since the whole point is the wiring between "trained N models"
and "the primary's artifact is untouched".
"""

import numpy as np
import pandas as pd
import torch

from neurotemplate.common.data_utils import compute_norm_stats, normalize_df
from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.pipeline import train_and_evaluate
from fake_logger import FakeLogger
from model_fixtures import trainable_cfg

NM_MAPPING = {"Y": ["y0"], "U": ["u0"], "D": ["d0"], "t": ["sec_elapsed"]}


def _synthetic_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "y0": rng.normal(20.0, 2.0, n),
        "u0": rng.uniform(0.0, 1.0, n),
        "d0": rng.normal(15.0, 5.0, n),
        "sec_elapsed": np.arange(n) * 300.0,
    })


def _make_cfg(savedir, **over) -> Config:
    return trainable_cfg(savedir, **{"seed": 123, **over})


def _run(cfg: Config) -> dict:
    df = _synthetic_df()
    norm_stats = compute_norm_stats(df, norm_type=cfg.norm_type)
    dfs = [normalize_df(df, norm_stats)]
    return train_and_evaluate(
        cfg=cfg,
        train_dfs=dfs, dev_dfs=dfs, test_dfs=dfs,
        nm_mapping=NM_MAPPING, norm_stats=norm_stats,
        grad_masks={}, synth_cfg=None, freq_s=300.0,
        logger=FakeLogger(),
    )


def test_ensemble_seeds_one_has_no_ensemble_and_no_spread_section(tmp_path):
    result = _run(_make_cfg(tmp_path, ensemble_seeds=1))
    assert result["ensemble"] is None
    reportcard = (tmp_path / "reportcard.md").read_text()
    assert "Seed ensemble spread" not in reportcard


def test_ensemble_seeds_three_reports_a_spread_without_disturbing_the_primary(tmp_path):
    """The primary model must come out identical whether or not more seeds
    train after it — the ensemble adds error bars, it does not change what
    artifact you get. This would fail if the extra seeds' training leaked
    into the primary's RNG state."""
    result_one = _run(_make_cfg(tmp_path / "one", ensemble_seeds=1))
    result_three = _run(_make_cfg(tmp_path / "three", ensemble_seeds=3))

    state_one = result_one["dyn_sys"].state_dict()
    state_three = result_three["dyn_sys"].state_dict()
    assert state_one.keys() == state_three.keys()
    for key in state_one:
        assert torch.equal(state_one[key], state_three[key]), (
            f"primary model's {key!r} differs with ensemble_seeds=3"
        )
    assert result_one["best_dev_loss"] == result_three["best_dev_loss"]

    assert result_one["ensemble"] is None
    ensemble = result_three["ensemble"]
    assert ensemble is not None
    assert ensemble.n_seeds == 3
    assert "y0" in ensemble.rmse
    spread = ensemble.rmse["y0"]
    assert spread.min <= spread.median <= spread.max

    reportcard_three = (tmp_path / "three" / "reportcard.md").read_text()
    assert "## Seed ensemble spread" in reportcard_three
    reportcard_one = (tmp_path / "one" / "reportcard.md").read_text()
    assert "Seed ensemble spread" not in reportcard_one
