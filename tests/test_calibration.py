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

import warnings

import pandas as pd
import pytest
import torch

import neurotemplate.modeling.calibration as calibration
from neurotemplate.modeling.calibration import (
    CalibrationStep,
    MAX_CALIBRATION_ITERATIONS,
    balance_level_weights,
    calibrate_decision_thresholds,
    collect_rollout_z,
    share_match_thresholds,
)
from neurotemplate.common.data_utils import get_sw_dict_multi, normalize_df
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.heads import HeadStack, OrdinalHead, RegressionHead
from control_fixtures import (
    STAGE, make_nm_mapping, make_norm_stats, make_raw_df, make_sysid_cfg,
)
from fake_logger import FakeLogger


def _fake_dyn_sys(head_stack):
    """The one attribute `calibrate_decision_thresholds` reads off `dyn_sys`
    before it ever rolls anything out — enough to unit-test the loop
    scaffolding without a real model or data."""
    from types import SimpleNamespace
    return SimpleNamespace(
        nodes=[SimpleNamespace(callable=SimpleNamespace(head_stack=head_stack))]
    )


# ── share_match_thresholds: a quantile, not a search ──────────────

def test_a_single_threshold_lands_at_the_true_shares_quantile():
    g = torch.Generator().manual_seed(0)
    true_idx = (torch.rand(2000, generator=g) < 0.1).long()   # 10% rare
    z = torch.randn(2000, generator=g)
    theta = share_match_thresholds(z, true_idx, n_levels=2)
    pred_idx = (z > theta).long()
    assert pred_idx.float().mean().item() == pytest.approx(0.1, abs=0.01)


def test_three_levels_produce_increasing_thresholds_matching_each_share():
    g = torch.Generator().manual_seed(1)
    true_idx = torch.cat([torch.zeros(800), torch.ones(150), torch.full((50,), 2)]).long()
    z = torch.randn(1000, generator=g)
    thresholds = share_match_thresholds(z, true_idx, n_levels=3)
    assert thresholds[0] < thresholds[1]
    pred_idx = (z.unsqueeze(-1) > thresholds).sum(-1)
    pred_shares = torch.bincount(pred_idx, minlength=3).float() / 1000
    assert pred_shares.tolist() == pytest.approx([0.8, 0.15, 0.05], abs=0.02)


def test_a_perfectly_separated_distribution_is_matched_exactly():
    true_idx = torch.tensor([0] * 70 + [1] * 30)
    z = torch.where(true_idx == 1, torch.full((100,), 5.0), torch.full((100,), -5.0))
    theta = share_match_thresholds(z, true_idx, n_levels=2)
    assert -5.0 < theta.item() < 5.0
    assert torch.equal((z > theta).long(), true_idx)


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


# ── The iteration loop's scaffolding (fake System, stubbed rollouts) ──

def test_a_head_that_never_snaps_is_not_calibrated():
    """use_ste=False feeds the smooth activation back, so decode — and its
    thresholds — never runs. Bails before ever rolling anything out."""
    head = OrdinalHead(n_levels=2, use_ste=False)
    dyn_sys = _fake_dyn_sys(HeadStack([head], ["stage"]))
    with pytest.warns(UserWarning, match="use_ste=False"):
        assert calibrate_decision_thresholds(dyn_sys, make_sysid_cfg(), [], {}) == {}


def test_a_stack_without_ordinal_heads_is_a_no_op():
    dyn_sys = _fake_dyn_sys(HeadStack([RegressionHead()], ["y"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert calibrate_decision_thresholds(dyn_sys, make_sysid_cfg(), [], {}) == {}


def test_a_level_dev_never_visits_is_skipped_with_a_warning(monkeypatch):
    head = OrdinalHead(n_levels=3)   # levels 0, 1, 2
    dyn_sys = _fake_dyn_sys(HeadStack([head], ["stage"]))
    before = head.decision_thresholds.clone()
    # y_true only ever hits levels 0 and 1 -- level 2 has no target share.
    y_true = torch.tensor([0.0, 1.0, 0.0, 1.0]).reshape(1, 4, 1)
    monkeypatch.setattr(calibration, "_rollout_z",
                        lambda *a, **k: (torch.zeros(1, 4, 1), y_true))

    with pytest.warns(UserWarning, match="never visits level"):
        result = calibrate_decision_thresholds(dyn_sys, make_sysid_cfg(), [], {})

    assert result == {}
    assert torch.equal(head.decision_thresholds, before)


def test_the_iteration_count_is_capped(monkeypatch):
    """A candidate that never settles must not be chased past the cap."""
    head = OrdinalHead(n_levels=2)
    dyn_sys = _fake_dyn_sys(HeadStack([head], ["stage"]))
    y_true = torch.tensor([0.0, 1.0, 0.0, 1.0]).reshape(1, 4, 1)
    calls = []
    monkeypatch.setattr(calibration, "_rollout_z", lambda *a, **k: (
        calls.append(1), (torch.zeros(1, 4, 1), y_true))[1])
    toggle = iter([0.1, 0.9] * 10)
    monkeypatch.setattr(calibration, "share_match_thresholds",
                        lambda z, true_idx, n_levels: torch.tensor([next(toggle)]))

    result = calibrate_decision_thresholds(dyn_sys, make_sysid_cfg(), [], {}, max_iterations=4)

    assert len(calls) == 4
    assert len(result["stage"].trace) == 4


def test_the_lowest_error_pass_is_kept_even_when_a_later_one_is_worse(monkeypatch):
    """A later pass can overshoot; the search keeps whichever rollout
    actually matched best, not whichever ran last."""
    head = OrdinalHead(n_levels=2)
    dyn_sys = _fake_dyn_sys(HeadStack([head], ["stage"]))
    y_true = torch.tensor([0.0, 1.0, 0.0, 1.0]).reshape(1, 4, 1)
    monkeypatch.setattr(calibration, "_rollout_z",
                        lambda *a, **k: (torch.zeros(1, 4, 1), y_true))
    errors = iter([0.9, 0.05, 0.6])          # bad, great, bad again
    candidates = iter([torch.tensor([0.2]), torch.tensor([0.4]), torch.tensor([0.6])])
    monkeypatch.setattr(
        calibration, "_score",
        lambda z, true_idx, thresholds, n_levels: CalibrationStep(
            thresholds.tolist(), [0.5, 0.5], next(errors), 0.5))
    monkeypatch.setattr(calibration, "share_match_thresholds",
                        lambda z, true_idx, n_levels: next(candidates))

    result = calibrate_decision_thresholds(
        dyn_sys, make_sysid_cfg(), [], {}, max_iterations=3)

    c = result["stage"]
    # trace[i] is scored with the thresholds iteration i actually ran with
    # (0.5, then 0.2, then 0.4); errors [0.9, 0.05, 0.6] make pass 1 (0.2)
    # the one to keep, even though pass 2 (0.4, error 0.6) ran later.
    assert [round(s.error, 2) for s in c.trace] == [0.9, 0.05, 0.6]
    assert c.kept == 1
    assert head.decision_thresholds.tolist() == pytest.approx([0.2])


def test_a_fixed_point_stops_the_loop_early(monkeypatch):
    """Feeding the same rollout back every call (a stand-in for the model
    having settled) must let the loop detect it has nothing left to gain,
    rather than spending every remaining iteration confirming the same
    answer."""
    head = OrdinalHead(n_levels=2)
    dyn_sys = _fake_dyn_sys(HeadStack([head], ["stage"]))
    g = torch.Generator().manual_seed(0)
    true_idx = (torch.rand(2000, generator=g) < 0.1).long()
    z = (torch.randn(2000, generator=g) + 1.5 * true_idx).reshape(1, 2000, 1)
    y_true = head.level_values[true_idx].reshape(1, 2000, 1)
    calls = []
    monkeypatch.setattr(calibration, "_rollout_z", lambda *a, **k: (
        calls.append(1), (z, y_true))[1])

    result = calibrate_decision_thresholds(dyn_sys, make_sysid_cfg(), [], {})

    assert len(calls) == 2   # one pass to find the candidate, one to confirm it
    assert len(result["stage"].trace) == 2


# ── End to end on a real System ───────────────────────────────────

def _staged_system(cfg, nm, norm_stats):
    stats = norm_stats[STAGE]
    levels = torch.tensor([(v - stats["shift"]) / stats["stretch"] for v in (0.0, 1.0, 2.0)])
    heads = [OrdinalHead(n_levels=3, level_values=levels) if col == STAGE
             else RegressionHead() for col in nm["Y"]]
    stack = HeadStack(heads, nm["Y"])
    return get_dynamics(cfg, nm, head_stack=stack), stack


def test_the_system_owns_its_own_copy_of_the_stack():
    """Calibration writes into the stack the System trained with; a caller
    reusing one stack across seeds or trials must not inherit that."""
    nm = make_nm_mapping()
    norm_stats = make_norm_stats(make_raw_df(n_rows=60))
    dyn_sys, stack = _staged_system(make_sysid_cfg(), nm, norm_stats)
    owned = dyn_sys.nodes[0].callable.head_stack
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
    data = get_sw_dict_multi([normalize_df(raw, norm_stats)], cfg, nm,
                             steps=cfg.in_steps + cfg.dev_pred_steps)
    data.pop("source_idx")
    z, y_true = collect_rollout_z(dyn_sys, data, cfg.in_steps)
    assert z.shape[:2] == y_true.shape[:2] == (data["Y"].shape[0], cfg.dev_pred_steps)
    assert z.shape[-1] == dyn_sys.nodes[0].callable.head_stack.raw_width
    assert torch.equal(y_true, data["Y"][:, cfg.in_steps:])


def test_calibration_updates_the_ordinal_head_in_place_and_reports_it():
    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=120)
    norm_stats = make_norm_stats(raw)
    cfg = make_sysid_cfg()
    cfg.dev_pred_steps = 8
    dyn_sys, _ = _staged_system(cfg, nm, norm_stats)
    head = dyn_sys.nodes[0].callable.head_stack.heads[nm["Y"].index(STAGE)]

    calibrated = calibrate_decision_thresholds(
        dyn_sys, cfg, [normalize_df(raw, norm_stats)], nm)

    assert list(calibrated) == [STAGE]
    c = calibrated[STAGE]
    assert c.thresholds == head.decision_thresholds.tolist()
    assert 1 <= len(c.trace) <= MAX_CALIBRATION_ITERATIONS
    assert 0 <= c.kept < len(c.trace)
    assert c.thresholds[0] < c.thresholds[1]
    assert sum(c.true_shares) == pytest.approx(1.0)
    # The activation thresholds are not the knob and must not move.
    assert torch.equal(head.thresholds, 0.5 * (head.level_values[:-1] + head.level_values[1:]))
    # The config that goes to disk carries the calibrated rule.
    cfg_entry = dyn_sys.nodes[0].callable.head_stack.to_config()[STAGE]
    assert cfg_entry["decision_thresholds"] == head.decision_thresholds.tolist()


def test_a_level_dev_never_visits_leaves_the_thresholds_alone_with_a_warning():
    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=120)
    raw[STAGE] = raw[STAGE].clip(upper=1.0)   # level 2 never happens on dev
    norm_stats = make_norm_stats(make_raw_df(n_rows=120))
    cfg = make_sysid_cfg()
    cfg.dev_pred_steps = 8
    dyn_sys, _ = _staged_system(cfg, nm, norm_stats)
    head = dyn_sys.nodes[0].callable.head_stack.heads[nm["Y"].index(STAGE)]
    before = head.decision_thresholds.clone()

    with pytest.warns(UserWarning, match="never visits level"):
        calibrated = calibrate_decision_thresholds(
            dyn_sys, cfg, [normalize_df(raw, norm_stats)], nm)

    assert calibrated == {}
    assert torch.equal(head.decision_thresholds, before)


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
    cfg = trainable_cfg(tmp_path, **over)

    norm_stats = compute_norm_stats(df, norm_type=cfg.norm_type)
    dfs = [normalize_df(df, norm_stats)]
    stats = norm_stats["stage"]
    levels = torch.tensor([(v - stats["shift"]) / stats["stretch"] for v in (0.0, 1.0)])
    stack = HeadStack([OrdinalHead(n_levels=2, level_values=levels)], ["stage"])
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
    trained = result["dyn_sys"].nodes[0].callable.head_stack.heads[0]
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


def test_calibration_can_be_switched_off(tmp_path):
    result, stack, _ = _staged_run(tmp_path, calibrate_thresholds=False)
    trained = result["dyn_sys"].nodes[0].callable.head_stack.heads[0]
    assert torch.equal(trained.decision_thresholds, stack.heads[0].decision_thresholds)
    reportcard = (tmp_path / "reportcard.md").read_text()
    assert "Threshold calibration" not in reportcard


def test_the_ensemble_spread_carries_balanced_accuracy(tmp_path):
    result, _, _ = _staged_run(tmp_path, ensemble_seeds=2)
    assert "stage" in result["ensemble"].balanced_acc
    assert "Test balanced accuracy (staged channels)" in (tmp_path / "reportcard.md").read_text()
