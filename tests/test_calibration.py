"""
Tests for the data-driven settings of staged channels: balanced level
weights before training, calibrated decision thresholds after it.

The threshold search is pinned on the shape of the bake-off's furnace: a
rare positive whose z ranks well but sits below the midpoint, where the
midpoint rule never fires and a lower threshold catches most of it. The
end-to-end tests run a real (untrained) System so the hook that reads z
off the rollout, the alignment with the truth, and the in-place update of
the System's own stack are all exercised the way `nt model train` does.
"""

import pandas as pd
import pytest
import torch

from neurotemplate.modeling.calibration import (
    balance_level_weights,
    calibrate_decision_thresholds,
    collect_rollout_z,
    tune_thresholds,
)
from neurotemplate.common.data_utils import get_sw_dict_multi, normalize_df
from neurotemplate.common.metrics import level_metrics
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.heads import HeadStack, OrdinalHead, RegressionHead
from control_fixtures import (
    STAGE, make_nm_mapping, make_norm_stats, make_raw_df, make_sysid_cfg,
)


def _rare_event_z(seed=0, n=2000, rate=0.03):
    """z ranks the rare level well (shifted up) but not past the midpoint —
    the bake-off's furnace: p99 of z on the common level ≈ 0.35 while the
    rare level sits around it, all below the 0.5 snap."""
    g = torch.Generator().manual_seed(seed)
    true_idx = (torch.rand(n, generator=g) < rate).long()
    z = torch.randn(n, generator=g) * 0.15 + 0.35 * true_idx
    return z, true_idx


def _balanced_acc(z, thresholds, true_idx, n_levels):
    pred = (z.unsqueeze(-1) > thresholds).sum(-1)
    return level_metrics(pred, true_idx, list(range(n_levels)), "").balanced_accuracy


# ── The search itself ─────────────────────────────────────────────

def test_the_tuned_threshold_catches_the_rare_level_the_midpoint_misses():
    z, true_idx = _rare_event_z()
    midpoint = torch.tensor([0.5])
    tuned = tune_thresholds(z, true_idx, midpoint)
    assert tuned.item() < midpoint.item()
    before = _balanced_acc(z, midpoint, true_idx, 2)
    after = _balanced_acc(z, tuned, true_idx, 2)
    assert before < 0.75 and after > 0.85


def test_a_threshold_the_data_cannot_improve_stays_where_it_is():
    """Perfectly separable at the midpoint: nothing beats it, so it stays."""
    true_idx = torch.tensor([0] * 50 + [1] * 50)
    z = torch.where(true_idx == 1, torch.full((100,), 0.9), torch.full((100,), 0.1))
    tuned = tune_thresholds(z, true_idx, torch.tensor([0.5]))
    assert tuned.item() == pytest.approx(0.5)


def test_three_levels_keep_their_order_and_improve():
    g = torch.Generator().manual_seed(1)
    true_idx = torch.cat([torch.zeros(900), torch.ones(80), torch.full((20,), 2)]).long()
    z = torch.randn(1000, generator=g) * 0.3 + 0.6 * true_idx
    init = torch.tensor([0.5, 1.5])
    tuned = tune_thresholds(z, true_idx, init)
    assert tuned[0] < tuned[1]
    assert _balanced_acc(z, tuned, true_idx, 3) >= _balanced_acc(z, init, true_idx, 3)


# ── Balanced level weights from the training data ─────────────────

def test_balance_level_weights_reads_each_levelled_channel_from_the_data():
    stack = HeadStack([OrdinalHead(n_levels=2), RegressionHead()], ["stage", "temp"])
    df = pd.DataFrame({"stage": [0.0] * 90 + [1.0] * 10, "temp": range(100)})
    balance_level_weights(stack, [df])
    w = stack.heads[0].level_weights
    assert w[1] / w[0] == pytest.approx(9.0)
    assert w.mean().item() == pytest.approx(1.0)


# ── End to end on a real System ───────────────────────────────────

def _staged_system(cfg, nm, norm_stats):
    stats = norm_stats[STAGE]
    levels = torch.tensor([(v - stats["shift"]) / stats["stretch"] for v in (0.0, 1.0, 2.0)])
    heads = [OrdinalHead(n_levels=3, level_values=levels) if col == STAGE
             else RegressionHead() for col in nm["Y"]]
    stack = HeadStack(heads, nm["Y"])
    return get_dynamics(cfg, nm, head_stack=stack), stack


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
    before = head.decision_thresholds.clone()

    calibrated = calibrate_decision_thresholds(
        dyn_sys, cfg, [normalize_df(raw, norm_stats)], nm)

    assert list(calibrated) == [STAGE]
    assert calibrated[STAGE].thresholds == head.decision_thresholds.tolist()
    assert calibrated[STAGE].dev_after >= calibrated[STAGE].dev_before
    assert head.decision_thresholds[0] < head.decision_thresholds[1]
    # The activation thresholds are not the knob and must not move.
    assert torch.equal(head.thresholds, 0.5 * (head.level_values[:-1] + head.level_values[1:]))
    # The config that goes to disk carries the calibrated rule.
    cfg_entry = dyn_sys.nodes[0].callable.head_stack.to_config()[STAGE]
    assert cfg_entry["decision_thresholds"] == head.decision_thresholds.tolist()
    del before  # only here to make the intent of the clone explicit


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


def test_a_tuned_rule_that_does_not_hold_up_once_fed_back_is_reverted(monkeypatch):
    """The rule is tuned on z produced under the old rule and then changes
    what is fed back. If the realized dev score is worse, keep the old
    rule and say so — quietly shipping a worse decision rule is the failure
    mode the bake-off warned about."""
    import neurotemplate.modeling.calibration as calibration

    nm = make_nm_mapping()
    raw = make_raw_df(n_rows=120)
    norm_stats = make_norm_stats(raw)
    cfg = make_sysid_cfg()
    cfg.dev_pred_steps = 8
    dyn_sys, _ = _staged_system(cfg, nm, norm_stats)
    head = dyn_sys.nodes[0].callable.head_stack.heads[nm["Y"].index(STAGE)]
    before = head.decision_thresholds.clone()

    # Rig the search to move the rule, and the score to prefer the old one:
    # an untrained model scores the floor either way, so the verdict has to
    # be forced to exercise the revert path deterministically.
    moved = before - 0.1
    monkeypatch.setattr(calibration, "tune_thresholds", lambda z, true_idx, init: moved)
    monkeypatch.setattr(
        calibration, "_balanced_accuracy",
        lambda z, thresholds, true_idx, n: 0.9 if torch.equal(thresholds, before) else 0.1,
    )
    with pytest.warns(UserWarning, match="were reverted"):
        calibrated = calibrate_decision_thresholds(
            dyn_sys, cfg, [normalize_df(raw, norm_stats)], nm)

    assert torch.equal(head.decision_thresholds, before)
    assert calibrated[STAGE].dev_after == calibrated[STAGE].dev_before == 0.9


# ── Through train_and_evaluate ────────────────────────────────────

class _RecordingLogger:
    """Duck-typed MLFlowLogger that keeps what was logged as a dict."""
    save_weights = False

    def __init__(self):
        self.dicts = {}

    def log_weights(self, *a, **kw):
        pass

    def log_metrics(self, *a, **kw):
        pass

    def log_dict(self, d, name):
        self.dicts[name] = d

    def log_artifacts(self, *a, **kw):
        pass


def _staged_run(tmp_path, **over):
    """A tiny real run whose one Y channel is a 3%-positive two-level stage."""
    import numpy as np
    from neurotemplate.common.data_utils import compute_norm_stats
    from neurotemplate.modeling.config import ModelingConfig as Config
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
    cfg = Config()
    cfg.model, cfg.in_steps, cfg.log2_hidden_size, cfg.num_layers = "mlp", 4, 2, 1
    cfg.epochs, cfg.warmup, cfg.patience = 2, 0, 5
    cfg.train_pred_steps = cfg.dev_pred_steps = 4
    cfg.test_window = cfg.test_stride = 10
    cfg.batch_size, cfg.norm_type = 8, "min_max"
    cfg.plot, cfg.verbose, cfg.device, cfg.seed = False, False, torch.device("cpu"), 7
    cfg.savedir = str(tmp_path)
    for key, value in over.items():
        setattr(cfg, key, value)

    norm_stats = compute_norm_stats(df, norm_type=cfg.norm_type)
    dfs = [normalize_df(df, norm_stats)]
    stats = norm_stats["stage"]
    levels = torch.tensor([(v - stats["shift"]) / stats["stretch"] for v in (0.0, 1.0)])
    stack = HeadStack([OrdinalHead(n_levels=2, level_values=levels)], ["stage"])
    balance_level_weights(stack, dfs)
    logger = _RecordingLogger()
    result = train_and_evaluate(
        cfg=cfg, train_dfs=dfs, dev_dfs=dfs, test_dfs=dfs,
        nm_mapping=nm, norm_stats=norm_stats, grad_masks={}, synth_cfg=None,
        freq_s=300.0, logger=logger, head_stack=stack,
    )
    return result, stack, logger


def test_a_run_persists_the_calibrated_rule_and_leaves_the_callers_stack_alone(tmp_path):
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


def test_the_ensemble_spread_carries_balanced_accuracy(tmp_path):
    result, _, _ = _staged_run(tmp_path, ensemble_seeds=2)
    assert "stage" in result["ensemble"].balanced_acc
    assert "Test balanced accuracy (staged channels)" in (tmp_path / "reportcard.md").read_text()

