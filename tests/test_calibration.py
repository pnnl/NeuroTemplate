"""
Tests for the data-driven settings of staged channels: balanced level
weights before training, calibrated decision thresholds after it.

Three levels of test, cheapest first:

  - `share_match_thresholds` as a pure function: it is a quantile, not a
    search, so its correctness is checkable in closed form.
  - The iteration loop's scaffolding (the cap, "keep lowest error, not
    last") against a fake System and stubbed rollouts — deterministic,
    no numerics required.
  - End to end on a real (untrained) System, exercising the forward hook
    that reads z, the alignment with the truth, and the in-place update
    of the System's own stack the way `nt model train` actually calls it.
"""


import pandas as pd
import pytest
import torch

from neurotemplate.modeling.calibration import (
    balance_level_weights,
    collect_rollout_z,
    share_match_thresholds,
)
from neurotemplate.common.data_utils import get_sw_dict_multi
from neurotemplate.modeling.dynamics import get_dynamics
from model_fixtures import unit_stats
from neurotemplate.modeling.heads import HeadStack, OrdinalHead, RegressionHead
from control_fixtures import (
    STAGE, make_nm_mapping, make_norm_stats, make_raw_df, make_sysid_cfg,
)
from fake_logger import FakeLogger
from neurotemplate.modeling.contract import normalized_core


# ── share_match_thresholds: a quantile, not a search ──────────────

def test_three_levels_produce_increasing_thresholds_matching_each_share():
    g = torch.Generator().manual_seed(1)
    true_idx = torch.cat([torch.zeros(800), torch.ones(150), torch.full((50,), 2)]).long()
    z = torch.randn(1000, generator=g)
    thresholds = share_match_thresholds(z, true_idx, n_levels=3)
    assert thresholds[0] < thresholds[1]
    pred_idx = (z.unsqueeze(-1) > thresholds).sum(-1)
    pred_shares = torch.bincount(pred_idx, minlength=3).float() / 1000
    assert pred_shares.tolist() == pytest.approx([0.8, 0.15, 0.05], abs=0.02)


# ── Balanced level weights from the training data ─────────────────

def test_balance_level_weights_reads_each_levelled_channel_from_the_data():
    """1/16 of samples at the rare level -> sqrt(9) = 3x the common level's
    weight, not the full 9x inverse-frequency would give (see heads.py's
    inverse_sqrt_frequency_weights for why: full inverse frequency also
    multiplies the gradient of exactly the samples an STE surrogate trusts
    least)."""
    stack = HeadStack([OrdinalHead(n_levels=2), RegressionHead()], ["stage", "temp"])
    df = pd.DataFrame({"stage": [0.0] * 90 + [1.0] * 10, "temp": range(100)})
    balance_level_weights(stack, [df])
    w = stack.heads[0].level_weights
    assert w[1] / w[0] == pytest.approx(3.0)
    assert w.mean().item() == pytest.approx(1.0)


# ── End to end on a real System ───────────────────────────────────

def _staged_system(cfg, nm, norm_stats):
    stats = norm_stats[STAGE]
    levels = torch.tensor([(v - stats["shift"]) / stats["stretch"] for v in (0.0, 1.0, 2.0)])
    heads = [OrdinalHead(n_levels=3, level_values=levels) if col == STAGE
             else RegressionHead() for col in nm["Y"]]
    stack = HeadStack(heads, nm["Y"])
    return get_dynamics(cfg, nm, unit_stats(nm), head_stack=stack), stack


def test_the_system_owns_its_own_copy_of_the_stack():
    """Calibration writes into the stack the System trained with; a caller
    reusing one stack across seeds or trials must not inherit that."""
    nm = make_nm_mapping()
    norm_stats = make_norm_stats(make_raw_df(n_rows=60))
    dyn_sys, stack = _staged_system(make_sysid_cfg(), nm, norm_stats)
    owned = normalized_core(dyn_sys).head_stack
    assert owned is not stack
    owned.heads[nm["Y"].index(STAGE)].decision_thresholds.fill_(0.0)
    assert not torch.equal(
        stack.heads[nm["Y"].index(STAGE)].decision_thresholds,
        owned.heads[nm["Y"].index(STAGE)].decision_thresholds,
    )


def test_collected_z_lines_up_with_the_scored_truth():
    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=60)
    norm_stats = make_norm_stats(raw)
    cfg = make_sysid_cfg()
    cfg.dev_pred_steps = 8
    dyn_sys, _ = _staged_system(cfg, nm, norm_stats)
    data = get_sw_dict_multi([raw], cfg, nm,
                             steps=cfg.history_steps + cfg.dev_pred_steps)
    data.pop("source_idx")
    z, y_true = collect_rollout_z(dyn_sys, data, cfg.history_steps)
    assert z.shape[:2] == y_true.shape[:2] == (data["Y"].shape[0], cfg.dev_pred_steps)
    assert z.shape[-1] == normalized_core(dyn_sys).head_stack.raw_width
    assert torch.equal(y_true, data["Y"][:, cfg.history_steps:])


# ── Through train_and_evaluate ────────────────────────────────────

def _staged_run(tmp_path, **over):
    """A tiny real run whose one Y channel is a 20%-positive two-level stage."""
    import numpy as np
    from neurotemplate.common.data_utils import compute_norm_stats
    from model_fixtures import trainable_cfg
    from neurotemplate.modeling.pipeline import train_and_evaluate

    rng = np.random.default_rng(0)
    n = 160
    fires = (rng.uniform(size=n) < 0.2).astype(float)
    df = pd.DataFrame({
        "stage": fires,
        "u0": rng.uniform(0.0, 1.0, n),
        "d0": rng.normal(15.0, 5.0, n),
        "sec_elapsed": np.arange(n) * 300.0,
    })
    nm = {"Y": ["stage"], "U": ["u0"], "D": ["d0"], "t": ["sec_elapsed"]}
    cfg = trainable_cfg(tmp_path, nm, **over)

    norm_stats = compute_norm_stats(df, norm_type=cfg.norm_type)
    dfs = [df]
    stack = HeadStack([OrdinalHead(n_levels=2, level_values=[0.0, 1.0])], ["stage"])
    balance_level_weights(stack, dfs)
    logger = FakeLogger()
    result = train_and_evaluate(
        cfg=cfg, train_dfs=dfs, dev_dfs=dfs, test_dfs=dfs,
        nm_mapping=nm, norm_stats=norm_stats, grad_masks={}, synth_cfg=None,
        freq_s=300.0, logger=logger, head_stack=stack,
    )
    return result, stack, logger


def test_a_run_persists_the_calibrated_rule_and_reports_its_search(tmp_path):
    result, stack, logger = _staged_run(tmp_path)
    trained = normalized_core(result.dyn_sys).head_stack.heads[0]
    midpoint = stack.heads[0].decision_thresholds
    # head_config.json is what `nt model load` rebuilds from, so it has to
    # carry the rule the run was evaluated with, not the constructor default.
    saved = logger.dicts["head_config.json"]["stage"]
    assert saved["decision_thresholds"] == trained.decision_thresholds.tolist()
    assert saved["level_weights"] == trained.level_weights.tolist()
    assert trained.level_weights[1] > trained.level_weights[0]
    assert torch.equal(midpoint, 0.5 * (stack.heads[0].level_values[:-1]
                                        + stack.heads[0].level_values[1:]))
    reportcard = (tmp_path / "reportcard.md").read_text()
    assert "## Level metrics (staged channels)" in reportcard
    assert "Decision thresholds" in reportcard
    assert "## Threshold calibration" in reportcard
    assert "true level shares" in reportcard
